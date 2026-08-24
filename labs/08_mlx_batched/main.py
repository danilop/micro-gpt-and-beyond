"""
microGPT: MLX batched edition.

Same architecture as 07_mlx, but with mini-batch training:
  - Batches of 32 sequences per step
  - Padding + masking at the data level
  - Scaled-up model (2 layers, 64-dim embeddings, context 16)
  - 1000 training steps

MLX has mx.vmap, and this lab uses it, so the model code below has no batch
dimension anywhere: the forward pass is written for one sequence and mx.vmap
lifts it over the batch, exactly as jax.vmap does in 06_jax_batched. That makes
labs 04, 06 and 08 a genuine three-way comparison of the same batching problem:
PyTorch's manual (B, T, ...) reshape, jax.vmap, and mx.vmap.

The step is also wrapped in mx.compile (MLX's jax.jit) and the loop prints
ms/step, so the framework claims here are measured rather than asserted. In the
same spirit, after training the lab times mx.vmap against a hand-written
(B, T, ...) forward pass on the same weights, and checks that the two agree.

Reference: "Attention Is All You Need" (Vaswani et al., 2017),
https://arxiv.org/abs/1706.03762
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
PAD = vocab_size
vocab_size_with_pad = vocab_size + 1
print(f"vocab size: {vocab_size} (+1 pad = {vocab_size_with_pad})")

# ---------------------------------------------------------------------------
# Hyperparameters (scaled up)
# ---------------------------------------------------------------------------
n_embd = 64  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 2  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head
batch_size = 32
num_steps = 1000


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
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

    def __call__(self, x, pad_mask):
        # x: (T, C), one sequence with no batch dimension. mx.vmap adds it later.
        T, C = x.shape
        q = self.wq(x).reshape(T, n_head, head_dim).transpose(1, 0, 2)
        k = self.wk(x).reshape(T, n_head, head_dim).transpose(1, 0, 2)
        v = self.wv(x).reshape(T, n_head, head_dim).transpose(1, 0, 2)

        att = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        # Causal mask
        causal = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
        att = mx.where(causal, -1e9, att)
        # Padding mask: block attention to PAD keys.
        #
        # This one is inert in this configuration, and it is worth knowing why.
        # Padding is appended at the end of a sequence, never interleaved, and
        # the causal mask already stops position t from looking past t. So every
        # key a real query can reach is a real token, and this line only changes
        # logits at pad query positions, which target_mask multiplies by zero.
        # It is kept because it becomes load-bearing the moment you left-pad,
        # use bidirectional attention, or pack several documents into one row.
        att = mx.where(pad_mask[None, None, :], -1e9, att)
        # No NaN guard is needed: masking with -1e9 rather than -inf means even a
        # fully masked row would softmax to a uniform distribution instead of
        # NaN. No row can be fully masked anyway, since row 0 always keeps position 0,
        # which is BOS, never PAD.
        att = mx.softmax(att, axis=-1)

        out = (att @ v).transpose(1, 0, 2).reshape(T, C)
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

    def __call__(self, x, pad_mask):
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size_with_pad, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = [Block() for _ in range(n_layer)]
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward_single(self, idx, pad_mask):
        """
        Forward pass for ONE sequence.
        idx: (T,) int array, pad_mask: (T,) bool array, True where padded.
        Returns: logits (T, vocab_size). Note the absence of a batch dimension.
        """
        T = idx.shape[0]
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(mx.arange(T))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.lm_head(x)

    def forward_batched(self, idx, pad_mask):
        """The same forward pass with the batch dimension written out by hand.

        Nothing trains through this: it exists so the lab can measure mx.vmap
        against the (B, T, ...) style 04_pytorch_batched writes by hand, on the
        same parameters and the same batch. Only attention has to be rewritten, since
        RMSNorm and the MLP are shape-agnostic, so those modules are reused
        as they are, and any difference in output would be a bug in one of them.
        """
        B, T = idx.shape
        x = self.norm_in(self.wte(idx) + self.wpe(mx.arange(T)))
        causal = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
        for layer in self.layers:
            h, attn = layer.norm1(x), layer.attn
            q = attn.wq(h).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
            k = attn.wk(h).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
            v = attn.wv(h).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
            att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
            att = mx.where(causal, -1e9, att)
            att = mx.where(pad_mask[:, None, None, :], -1e9, att)
            att = mx.softmax(att, axis=-1)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, n_embd)
            x = x + attn.wo(out)
            x = x + layer.mlp(layer.norm2(x))
        return self.lm_head(x)

    def __call__(self, idx, pad_mask=None):
        """
        Forward pass for a BATCH: idx (B, T) -> logits (B, T, vocab_size).

        mx.vmap lifts forward_single over axis 0 of both arguments. The model
        parameters are captured from self and shared across the batch, which is
        what in_axes=None would say explicitly in jax.vmap. Nothing above this
        line knows a batch exists.
        """
        if pad_mask is None:  # inference: one sequence, nothing padded
            pad_mask = mx.zeros(idx.shape, dtype=mx.bool_)
        return mx.vmap(self.forward_single, in_axes=(0, 0))(idx, pad_mask)


# ---------------------------------------------------------------------------
# Batching utility
# ---------------------------------------------------------------------------
def make_batch(docs, step, batch_size):
    """Create a padded batch of token sequences."""
    batch_docs = [docs[(step * batch_size + i) % len(docs)] for i in range(batch_size)]
    sequences = []
    for doc in batch_docs:
        toks = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        toks = toks[: block_size + 1]
        sequences.append(toks)

    # Padded to the longest sequence in this batch, the same way 04_pytorch_batched
    # does it. That means the batch shape changes from step to step, and mx.compile
    # (like jax.jit) traces once per shape, and the printed shape count below shows how
    # many times. 06_jax_batched shows the fix: pad to a fixed length instead.
    max_len = max(len(s) for s in sequences)
    input_ids, target_ids, pad_masks, target_masks = [], [], [], []
    for s in sequences:
        n = len(s) - 1  # real tokens; padding is appended after them, never interleaved
        inp = s[:n] + [PAD] * (max_len - 1 - n)
        # The dummy target 0 is a real character ('a'), which is exactly why
        # target_mask below is not optional: without it the model would be
        # trained to predict 'a' at every padded position.
        tgt = s[1 : n + 1] + [0] * (max_len - 1 - n)
        pmask = [False] * n + [True] * (max_len - 1 - n)
        tmask = [1.0] * n + [0.0] * (max_len - 1 - n)
        input_ids.append(inp)
        target_ids.append(tgt)
        pad_masks.append(pmask)
        target_masks.append(tmask)

    return (
        mx.array(input_ids),
        mx.array(target_ids),
        mx.array(pad_masks),
        mx.array(target_masks),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model = MicroGPT()
# Match original init: N(0, 0.08) for all weights. nn.Module.apply maps over
# every parameter array in place.
model.apply(lambda w: mx.random.normal(w.shape) * 0.08)
# Re-zero the padding embedding after init
model.wte.weight[PAD] = mx.zeros((n_embd,))
mx.eval(model.parameters())
num_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
print(f"num params: {num_params}")

learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8


def loss_fn(model, input_ids, targets, pad_mask, target_mask):
    logits = model(input_ids, pad_mask)  # (B, T, V)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    B, T, V = log_probs.shape
    # Gather log-probs for target tokens
    target_log_probs = log_probs[mx.arange(B)[:, None], mx.arange(T)[None, :], targets]
    target_log_probs = target_log_probs * target_mask
    loss = -mx.sum(target_log_probs) / mx.sum(target_mask)
    return loss


loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=learning_rate, betas=[beta1, beta2], eps=eps_adam)

# mx.compile is MLX's counterpart to jax.jit. `inputs`/`outputs` declare the
# state the step reads and writes without taking as arguments. Set use_compile
# to False and compare the printed ms/step.
use_compile = True
state = [model.state, optimizer.state]


def train_step(input_ids, targets, pad_mask, target_mask):
    loss_val, grads = loss_and_grad(model, input_ids, targets, pad_mask, target_mask)
    optimizer.update(model, grads)
    return loss_val


if use_compile:
    train_step = mx.compile(train_step, inputs=state, outputs=state)

seen_shapes = {}  # batch shape -> step it first appeared on; one trace per shape
first_shape_ms, cached_shape_ms = [], []

for step in range(num_steps):
    input_ids, targets, pad_mask, target_mask = make_batch(docs, step, batch_size)

    is_new_shape = input_ids.shape not in seen_shapes
    seen_shapes.setdefault(input_ids.shape, step)

    optimizer.learning_rate = mx.array(learning_rate * (1 - step / num_steps))

    t0 = time.perf_counter()
    loss_val = train_step(input_ids, targets, pad_mask, target_mask)
    # MLX is lazy: the update above is only a graph until something evaluates it.
    # loss_val.item() would force the loss but not the parameters or the optimizer
    # moments, so without this the pending graph would grow for the whole run.
    mx.eval(state)
    dt = (time.perf_counter() - t0) * 1000
    (first_shape_ms if is_new_shape else cached_shape_ms).append(dt)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss_val.item():.4f} | {dt:7.2f} ms")

print(f"\ndistinct batch shapes seen: {len(seen_shapes)} (compile = {use_compile})")
print(f"  first-time-shape steps: {sum(first_shape_ms) / len(first_shape_ms):8.2f} ms mean")
print(f"  repeated-shape steps:   {sum(cached_shape_ms) / len(cached_shape_ms):8.2f} ms mean")
# Do not read that gap as the price of tracing. Run with use_compile = False and it
# is still there, because new shapes turn up mostly in the first few dozen steps,
# so "first-time shape" is largely a synonym for "early step", when nothing is warm.
print(f"  each shape first seen at step: {sorted(seen_shapes.values())}")

# ---------------------------------------------------------------------------
# mx.vmap against a hand-written batch dimension, measured rather than asserted
# ---------------------------------------------------------------------------
# The model above is written for one sequence because that reads better. The fair
# question is what the transform costs, and `forward_batched` is the same
# computation with the batch axis written out by hand, so the two can be compared
# on the same weights: first for agreement, then for speed. Forward pass only,
# that is where the transform lives, and it keeps the comparison free of the
# optimizer state a training step would mutate.
bench_ids, _, bench_pmask, _ = make_batch(docs, 0, batch_size)
diff = mx.max(mx.abs(model(bench_ids, bench_pmask) - model.forward_batched(bench_ids, bench_pmask)))
mx.eval(diff)


def best_forward_ms(forward, reps=20):
    """Fastest of `reps` forward passes over the benchmark batch, in ms.

    The minimum, not the mean: anything else sharing the machine can only make a
    pass slower. mx.eval inside the loop is essential: MLX is lazy, so without
    it this would time graph construction and nothing else.
    """
    mx.eval(forward(bench_ids, bench_pmask))  # warm up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        mx.eval(forward(bench_ids, bench_pmask))
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


ms_vmap = best_forward_ms(model.__call__)
ms_manual = best_forward_ms(model.forward_batched)
print(f"\n--- mx.vmap vs a hand-written (B, T, ...) forward, batch {bench_ids.shape} ---")
print(f"  max |logit difference|:     {diff.item():.2e}  (same parameters, same batch)")
print(f"  mx.vmap:                    {ms_vmap:7.2f} ms  (best of 20)")
print(f"  hand-written batch axis:     {ms_manual:6.2f} ms  (best of 20)")
gap = abs(ms_vmap - ms_manual) / min(ms_vmap, ms_manual)
if gap < 0.1:
    print(f"  Same speed to within {gap * 100:.1f}% on this machine, so writing the model for one")
    print("  sequence and lifting it is a readability win rather than a trade.")
else:
    slower = "mx.vmap" if ms_vmap > ms_manual else "the hand-written version"
    print(f"  {slower} is {gap * 100:.1f}% slower on this machine, a real trade at this size,")
    print("  not the wash it usually is. Worth re-measuring on the hardware you deploy on.")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    tokens = [BOS]
    for _ in range(block_size):
        input_ids = mx.array([tokens[-block_size:]])  # (1, T)
        logits = model(input_ids)
        logits = logits[0, -1] / temperature
        token_id = mx.random.categorical(logits).item()
        if token_id == BOS:
            break
        tokens.append(token_id)
    name = "".join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx + 1:2d}: {name}")
