"""
microGPT: NumPy edition with an array-level autograd engine.

Same architecture and the same autograd idea as the pure-Python version. What
changes is the unit of computation. Version 01 builds a graph of scalars, so a
matrix multiply becomes thousands of Python-level nodes. Here a whole array is
one node, that matrix multiply is a single call into BLAS, and one training
step builds a graph of a few dozen nodes.

The `Tensor` class below has the same shape as version 01's `Value`: a handful
of primitives, each recording how to push a gradient back to its inputs, and a
topological walk that applies them in reverse. RMSNorm and softmax stay
ordinary functions composed from those primitives, so the graph produces their
gradients without anyone deriving them.

Reverse-mode automatic differentiation is the algorithm from "Learning
representations by back-propagating errors" (Rumelhart, Hinton & Williams,
1986), in the modern framing surveyed by "Automatic Differentiation in Machine
Learning: a Survey" (Baydin et al., 2018, https://arxiv.org/abs/1502.05767).
The transformer follows "Attention Is All You Need" (Vaswani et al., 2017,
https://arxiv.org/abs/1706.03762) as a decoder-only variant, and normalization
uses RMSNorm (Zhang & Sennrich, 2019, https://arxiv.org/abs/1910.07467) with
the paper's learnable gain dropped, as in version 01.

Every training step prints its wall-clock time, so the "NumPy is faster than
pure Python" claim is something you measure rather than take on trust.
"""

import math
import os
import random
import time

import numpy as np

random.seed(42)
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Autograd over arrays
# ---------------------------------------------------------------------------
def unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, undoing NumPy's broadcasting.

    Scalars never broadcast, so version 01's `Value` never needed this. When
    `(n, 1)` meets `(n, d)` the forward pass silently copies a column d times,
    and the backward pass has to add those d gradients back together.
    """
    while grad.ndim > len(shape):  # drop axes that broadcasting prepended
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):  # collapse axes that were stretched from 1
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """A node in the computation graph, holding a whole array.

    Version 01 stores a *local derivative* per child, because for scalars every
    local derivative is a number to multiply by. Arrays need more than a
    multiplier: the gradient of a matmul is another matmul, of a reshape is the
    inverse reshape, of an index is a scatter-add. Each operation here records a
    small closure, which is the one structural difference from version 01.
    """

    __slots__ = ("_backward", "_children", "data", "grad")

    def __init__(self, data, children=()):
        self.data = np.asarray(data, dtype=np.float64)  # value from the forward pass
        self.grad = np.zeros_like(self.data)  # d(loss)/d(self), filled in by backward
        self._children = children  # the nodes this one was computed from
        self._backward = _noop  # pushes self.grad into those children

    # --- binary primitives ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            # Addition passes the gradient through untouched, but broadcasting
            # may have changed its shape on the way in, so undo that.
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            # The product rule, applied elementwise.
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            # For C = A @ B: dA = dC @ B^T and dB = A^T @ dC. swapaxes rather
            # than .T so this also covers the batched (nh, n, hd) arrays that
            # multi-head attention works with.
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    # --- elementwise primitives, given as version 01 gives them: the value,
    # --- and the local derivative to multiply the incoming gradient by ---

    def _elementwise(self, value, local_grad):
        out = Tensor(value, (self,))

        def _backward():
            self.grad += out.grad * local_grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        return self._elementwise(self.data**k, k * self.data ** (k - 1))

    def log(self):
        return self._elementwise(np.log(self.data), 1.0 / self.data)

    def relu(self):
        return self._elementwise(np.maximum(0.0, self.data), self.data > 0)

    def exp(self):
        value = np.exp(self.data)
        return self._elementwise(value, value)  # d(e^x) = e^x, already computed

    # --- shape primitives, which version 01 had no need for ---

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            # Every element that went into the sum receives the same
            # incoming gradient.
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes):
        out = Tensor(self.data.transpose(*axes), (self,))

        def _backward():
            self.grad += out.grad.transpose(np.argsort(axes))  # undo the permutation

        out._backward = _backward
        return out

    def __getitem__(self, index):
        out = Tensor(self.data[index], (self,))

        def _backward():
            # np.add.at, not `+=`, because a token repeated in the sequence
            # indexes the same embedding row twice and both gradients must land.
            np.add.at(self.grad, index, out.grad)

        out._backward = _backward
        return out

    # --- conveniences composed from the primitives above ---

    def mean(self, axis=None, keepdims=False):
        count = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis, keepdims) * (1.0 / count)

    @property
    def T(self):
        return self.transpose(1, 0)

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def backward(self):
        # Topological order, then walk it in reverse so that every node's
        # gradient is complete before it is pushed to its children. Identical to
        # version 01, except this graph holds dozens of nodes where that one
        # held thousands.
        topo, visited = [], set()

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


def _noop():
    """The backward pass of a leaf: parameters and constants have no children."""


# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


def param(shape, std=0.08):
    return Tensor(rng.standard_normal(shape) * std)


# Every parameter is a graph node, so its gradient arrives in `.grad` for free.
P = {
    "wte": param((vocab_size, n_embd)),
    "wpe": param((block_size, n_embd)),
    "lm_head": param((vocab_size, n_embd)),
}
for i in range(n_layer):
    P[f"l{i}.wq"] = param((n_embd, n_embd))
    P[f"l{i}.wk"] = param((n_embd, n_embd))
    P[f"l{i}.wv"] = param((n_embd, n_embd))
    P[f"l{i}.wo"] = param((n_embd, n_embd))
    P[f"l{i}.fc1"] = param((n_embd, 4 * n_embd))
    P[f"l{i}.fc2"] = param((4 * n_embd, n_embd))

print(f"num params: {sum(p.data.size for p in P.values())}")


# ---------------------------------------------------------------------------
# Model
#
# Only the forward pass is written. Every function below is the formula as you
# would write it on paper, and the engine supplies the gradients.
# ---------------------------------------------------------------------------
def rmsnorm(x):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x * (ms + 1e-5) ** -0.5


def softmax(x):
    # Subtracting the row max is for numerical stability only and cancels out of
    # the result, so it is taken from `.data` and kept off the graph.
    e = (x - x.data.max(axis=-1, keepdims=True)).exp()
    return e / e.sum(axis=-1, keepdims=True)


def forward(token_ids):
    """token_ids: (n,) int array -> logits (n, vocab_size)."""
    n = len(token_ids)
    x = P["wte"][token_ids] + P["wpe"][:n]  # token and position embeddings
    x = rmsnorm(x)  # not redundant: the residual path carries gradient back here

    for li in range(n_layer):
        # 1) Multi-head attention block
        x_residual = x
        xn = rmsnorm(x)
        # (n, D) -> (n, nh, hd) -> (nh, n, hd), so each head attends on its own
        Q = (xn @ P[f"l{li}.wq"]).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        K = (xn @ P[f"l{li}.wk"]).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        V = (xn @ P[f"l{li}.wv"]).reshape(n, n_head, head_dim).transpose(1, 0, 2)

        att = (Q @ K.transpose(0, 2, 1)) * (1.0 / math.sqrt(head_dim))
        # An additive -1e9 above the diagonal drives those weights to zero
        # through softmax. Adding a constant needs no gradient rule of its own.
        att = softmax(att + np.triu(np.full((n, n), -1e9), k=1))

        heads = (att @ V).transpose(1, 0, 2).reshape(n, n_embd)  # concatenate heads
        x = (heads @ P[f"l{li}.wo"]) + x_residual

        # 2) MLP block
        x_residual = x
        hidden = (rmsnorm(x) @ P[f"l{li}.fc1"]).relu()
        x = (hidden @ P[f"l{li}.fc2"]) + x_residual

    return x @ P["lm_head"].T


def loss_fn(tokens):
    """Cross-entropy of predicting each token from the ones before it."""
    n = min(block_size, len(tokens) - 1)
    probs = softmax(forward(tokens[:n]))
    hit = probs[np.arange(n), tokens[1 : n + 1]]  # probability given to the right token
    return -(hit + 1e-10).log().mean()


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8
M = {k: np.zeros_like(p.data) for k, p in P.items()}  # first moment
V = {k: np.zeros_like(p.data) for k, p in P.items()}  # second moment

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
num_steps = 1000
step_ms = []  # wall-clock time per step, so speed is measured instead of asserted
for step in range(num_steps):
    t_step = time.perf_counter()
    doc = docs[step % len(docs)]
    tokens = np.array([BOS] + [uchars.index(ch) for ch in doc] + [BOS])

    for p in P.values():
        p.grad.fill(0.0)

    loss = loss_fn(tokens)
    loss.backward()  # every parameter's .grad is filled in by this call

    lr_t = learning_rate * (1 - step / num_steps)  # linear learning rate decay
    for k, p in P.items():
        M[k] = beta1 * M[k] + (1 - beta1) * p.grad
        V[k] = beta2 * V[k] + (1 - beta2) * p.grad**2
        m_hat = M[k] / (1 - beta1 ** (step + 1))
        v_hat = V[k] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)

    step_ms.append((time.perf_counter() - t_step) * 1000)

    if (step + 1) % 10 == 0 or step == 0:
        # Average over the steps since the last print: single steps are noisy.
        recent = step_ms[-10:]
        avg_ms = sum(recent) / len(recent)
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {float(loss.data):.4f} | {avg_ms:6.2f} ms/step")

print(f"\nmean {sum(step_ms) / len(step_ms):.2f} ms/step over {num_steps} steps (forward + autograd backward + Adam)")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")

for sample_idx in range(20):
    generated, sample = [BOS], []
    for _ in range(block_size):
        logits = forward(np.array(generated)).data[-1]  # only the last position matters
        probs = softmax(Tensor(logits / temperature)).data
        token_id = rng.choice(vocab_size, p=probs)
        if token_id == BOS:
            break
        generated.append(token_id)
        sample.append(uchars[token_id])
    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
