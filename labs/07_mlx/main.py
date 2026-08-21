"""
microGPT — MLX edition.

A framework port of the same decoder-only GPT architecture from
"Attention Is All You Need" (Vaswani et al., 2017),
https://arxiv.org/abs/1706.03762, running on Apple Silicon GPU via MLX.
MLX is described in "MLX: Efficient and flexible machine learning on Apple
silicon" (Hannun et al., 2023), https://github.com/ml-explore/mlx.

MLX has a NumPy-like API with automatic differentiation and lazy evaluation.
Arrays live in unified memory, so Apple Silicon's unified memory architecture
eliminates CPU-GPU transfer overhead entirely.

The training step is wrapped in mx.compile, MLX's analogue of jax.jit, and the
loop prints ms/step. Set use_compile = False below to see what the compiler is
worth on your machine instead of reading a claim about it.
"""

import math
import os
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

random.seed(42)
mx.random.seed(42)

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
# Model
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


class RMSNorm(nn.Module):
    def __init__(self, _dim):
        super().__init__()
        self.eps = 1e-5

    def __call__(self, x):
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(ms + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def __call__(self, x):
        n = x.shape[0]
        q = self.wq(x).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        k = self.wk(x).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        v = self.wv(x).reshape(n, n_head, head_dim).transpose(1, 0, 2)

        att = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        mask = mx.triu(mx.ones((n, n)), k=1).astype(mx.bool_)
        att = mx.where(mask, -1e9, att)
        att = mx.softmax(att, axis=-1)

        out = (att @ v).transpose(1, 0, 2).reshape(n, n_embd)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        return self.fc2(mx.maximum(self.fc1(x), 0))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = [Block() for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, idx):
        n = idx.shape[0]
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(mx.arange(n))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model = MicroGPT()
# Match original init: N(0, 0.08) for all weights.
# nn.Module.apply maps a function over every parameter array in place, which is
# the idiomatic way to re-initialise (or cast, or quantise) a whole model.
model.apply(lambda w: mx.random.normal(w.shape) * 0.08)
num_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
print(f"num params: {num_params}")

learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8


def loss_fn(model, input_ids, targets):
    logits = model(input_ids)  # (n, V)
    # Cross-entropy
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    n = input_ids.shape[0]
    loss = -mx.mean(log_probs[mx.arange(n), targets])
    return loss


loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=learning_rate, betas=[beta1, beta2], eps=eps_adam)

# ---------------------------------------------------------------------------
# The training step, optionally compiled
# ---------------------------------------------------------------------------
# mx.compile is MLX's counterpart to jax.jit: it traces the function once and
# hands XLA-style fused kernels to the device instead of dispatching operation by
# operation. Like jit, it specialises on input shapes, so a new sequence length
# means another trace (see the two averages printed after training).
#
# `inputs`/`outputs` tell the compiler that the model parameters and the
# optimizer state are read and written by the step, since they are not passed as
# arguments. Flip use_compile to False and compare the printed ms/step.
use_compile = True
state = [model.state, optimizer.state]


def train_step(input_ids, targets):
    loss_val, grads = loss_and_grad(model, input_ids, targets)
    optimizer.update(model, grads)
    return loss_val


if use_compile:
    train_step = mx.compile(train_step, inputs=state, outputs=state)

num_steps = 1000
seen_lengths = set()  # each new sequence length costs one more trace
first_shape_ms, cached_shape_ms = [], []

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = mx.array(tokens[:n])
    targets = mx.array(tokens[1 : n + 1])

    is_new_shape = n not in seen_lengths
    seen_lengths.add(n)

    # Linear LR decay. The learning rate lives in the optimizer state, which the
    # compiled step captures, so changing it here is picked up without recompiling.
    optimizer.learning_rate = mx.array(learning_rate * (1 - step / num_steps))

    t0 = time.perf_counter()
    loss_val = train_step(input_ids, targets)
    # Why this line exists: MLX is lazy, so the update above is only a graph.
    # loss_val.item() further down would force the loss, but not the parameters
    # or the optimizer moments. Without an explicit eval, the pending graph would
    # keep growing step after step until scheduling and memory dominate. This is
    # a bound on the graph, not a way to "get the number".
    mx.eval(state)
    dt = (time.perf_counter() - t0) * 1000
    (first_shape_ms if is_new_shape else cached_shape_ms).append(dt)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss_val.item():.4f} | {dt:7.2f} ms")

print(f"\ndistinct sequence lengths seen: {len(seen_lengths)} (compile = {use_compile})")
print(f"  first-time-shape steps: {sum(first_shape_ms) / len(first_shape_ms):8.2f} ms mean")
print(f"  repeated-shape steps:   {sum(cached_shape_ms) / len(cached_shape_ms):8.2f} ms mean")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    tokens = [BOS]
    for _ in range(block_size):
        input_ids = mx.array(tokens)
        logits = model(input_ids)
        logits = logits[-1] / temperature
        token_id = mx.random.categorical(logits).item()
        if token_id == BOS:
            break
        tokens.append(token_id)
    name = "".join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx + 1:2d}: {name}")
