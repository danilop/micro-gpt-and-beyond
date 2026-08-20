"""
microGPT — JAX batched edition.

Same architecture as 05_jax, but with mini-batch training via jax.vmap:
  - Write the forward pass for a single example
  - vmap automatically vectorizes it across a batch
  - Padding at the data level, but the model code stays single-example
  - Scaled-up model (2 layers, 64-dim embeddings, context 16)
  - Batches of 32, 1000 training steps

This is JAX's signature trick: "write for one, run for many."

It also fixes the problem 05_jax leaves you with. There, the sequence length
changed from step to step, so the jitted step recompiled a dozen times. Here
every batch is padded to the *static* block_size, so the shape never changes
and XLA compiles exactly once. The training loop counts both, so you can see
the recompilations disappear rather than take it on faith.

Reference: "Attention Is All You Need" (Vaswani et al., 2017),
https://arxiv.org/abs/1706.03762
"""

import math
import os
import random
import time

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap

random.seed(42)

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
# Parameters
# ---------------------------------------------------------------------------
key = jax.random.key(42)  # typed key, the modern jax.random API


def init_param(key, shape, std=0.08):
    return jax.random.normal(key, shape) * std


# One key per parameter tensor: wte, wpe, lm_head, plus six per layer.
num_param_tensors = 3 + 6 * n_layer
keys = jax.random.split(key, num_param_tensors)
ki = iter(keys)

params = {
    "wte": init_param(next(ki), (vocab_size_with_pad, n_embd)),
}
# Zero the PAD embedding row so padding tokens contribute nothing
params["wte"] = params["wte"].at[PAD].set(jnp.zeros(n_embd))
params.update({
    "wpe": init_param(next(ki), (block_size, n_embd)),
    "lm_head": init_param(next(ki), (vocab_size, n_embd)),
})
for i in range(n_layer):
    params[f"l{i}.wq"] = init_param(next(ki), (n_embd, n_embd))
    params[f"l{i}.wk"] = init_param(next(ki), (n_embd, n_embd))
    params[f"l{i}.wv"] = init_param(next(ki), (n_embd, n_embd))
    params[f"l{i}.wo"] = init_param(next(ki), (n_embd, n_embd))
    params[f"l{i}.fc1"] = init_param(next(ki), (n_embd, 4 * n_embd))
    params[f"l{i}.fc2"] = init_param(next(ki), (4 * n_embd, n_embd))

num_params = sum(p.size for p in jax.tree.leaves(params))
print(f"num params: {num_params}")


# ---------------------------------------------------------------------------
# Forward pass (single example — vmap will handle the batch)
# ---------------------------------------------------------------------------
def rmsnorm(x):
    ms = jnp.mean(x**2, axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + 1e-5)


def forward_single(params, input_ids, pad_mask):
    """
    Forward pass for ONE sequence.
    input_ids: (T,) int array
    pad_mask: (T,) bool array, True where padded
    Returns: logits (T, vocab_size)
    """
    T = input_ids.shape[0]
    tok_emb = params["wte"][input_ids]  # (T, D)
    pos_emb = params["wpe"][jnp.arange(T)]  # (T, D)
    x = rmsnorm(tok_emb + pos_emb)

    for li in range(n_layer):
        x_res = x
        x_n = rmsnorm(x)

        Q = x_n @ params[f"l{li}.wq"]
        K = x_n @ params[f"l{li}.wk"]
        V = x_n @ params[f"l{li}.wv"]

        Q_h = Q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        K_h = K.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        V_h = V.reshape(T, n_head, head_dim).transpose(1, 0, 2)

        att = Q_h @ K_h.transpose(0, 2, 1) / math.sqrt(head_dim)
        # Causal mask
        causal = jnp.triu(jnp.ones((T, T)), k=1).astype(bool)
        att = jnp.where(causal, -1e9, att)
        # Padding mask: block attention to PAD keys.
        #
        # Padding is always appended at the end of a sequence, never interleaved
        # with real tokens, and the causal mask already stops position t from
        # looking past t. So every key a real query can reach is a real token,
        # and this line only changes logits at pad *query* positions, which
        # target_mask then multiplies by zero. It is redundant here, and kept
        # because it stops being redundant with left-padding, bidirectional
        # attention, or several documents packed into one row.
        att = jnp.where(pad_mask[None, None, :], -1e9, att)
        # No nan_to_num guard needed: masking with -1e9 rather than -inf means
        # even a hypothetical fully masked row softmaxes to a uniform
        # distribution, not NaN. And no row can be fully masked anyway — row 0
        # always keeps position 0, which is BOS.
        att = jax.nn.softmax(att, axis=-1)

        attn_out = att @ V_h
        attn_cat = attn_out.transpose(1, 0, 2).reshape(T, n_embd)
        x = attn_cat @ params[f"l{li}.wo"] + x_res

        # MLP
        x_res2 = x
        x_n2 = rmsnorm(x)
        h = x_n2 @ params[f"l{li}.fc1"]
        h = jax.nn.relu(h)
        x = h @ params[f"l{li}.fc2"] + x_res2

    logits = x @ params["lm_head"].T  # (T, V)
    return logits


# ---------------------------------------------------------------------------
# vmap: the JAX way to batch
# ---------------------------------------------------------------------------
# forward_single works on one sequence. vmap lifts it to work on a batch.
# in_axes=(None, 0, 0) means: shared params, batched input_ids, batched pad_mask.
forward_batch = vmap(forward_single, in_axes=(None, 0, 0))


def loss_fn(params, input_ids, targets, pad_mask, target_mask):
    """
    Batched cross-entropy loss.
    input_ids: (B, T), targets: (B, T), pad_mask: (B, T), target_mask: (B, T)
    """
    logits = forward_batch(params, input_ids, pad_mask)  # (B, T, V)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B, T, V = log_probs.shape
    # Gather log-probs for target tokens
    target_log_probs = log_probs[jnp.arange(B)[:, None], jnp.arange(T)[None, :], targets]  # (B, T)
    # Mask out padding positions
    target_log_probs = target_log_probs * target_mask
    loss = -jnp.sum(target_log_probs) / jnp.sum(target_mask)
    return loss


# ---------------------------------------------------------------------------
# Adam optimizer (functional)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8
m_state = jax.tree.map(jnp.zeros_like, params)
v_state = jax.tree.map(jnp.zeros_like, params)


# One jitted function for the whole step: loss, gradients and Adam together.
# value_and_grad gives the loss and the gradients from a single backward pass,
# and jax.tree.map applies the Adam arithmetic to every leaf of the parameter
# pytree without a Python loop.
@jit
def train_step(params, m_state, v_state, batch, step, lr):
    input_ids, targets, pad_mask, target_mask = batch
    loss, grads = value_and_grad(loss_fn)(params, input_ids, targets, pad_mask, target_mask)
    new_m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, m_state, grads)
    new_v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g**2, v_state, grads)
    bias1 = 1 - beta1 ** (step + 1)
    bias2 = 1 - beta2 ** (step + 1)
    new_params = jax.tree.map(
        lambda p, m, v: p - lr * (m / bias1) / (jnp.sqrt(v / bias2) + eps_adam),
        params,
        new_m,
        new_v,
    )
    return loss, new_params, new_m, new_v


# ---------------------------------------------------------------------------
# Batching utility
# ---------------------------------------------------------------------------
def make_batch(docs, step, batch_size):
    """
    Create a batch of token sequences, padded to the static block_size.

    Padding to block_size instead of to the longest sequence in the batch is the
    whole point. The obvious version pads to `max(len(s) for s in sequences)`,
    which changes from batch to batch, and every new value is a new input shape,
    and every new input shape makes jit trace and compile the step again. Padding
    to a constant means one compilation for the entire run.

    The price is arithmetic on padding: a batch whose longest name is 5
    characters still runs the full 16 columns. Static shapes are usually worth
    that, which is why real training pipelines bucket or pad to fixed lengths.

    Returns the batch plus the max_len that dynamic padding *would* have used,
    so the training loop can count the compilations this avoids.
    """
    batch_docs = [docs[(step * batch_size + i) % len(docs)] for i in range(batch_size)]
    sequences = []
    for doc in batch_docs:
        toks = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        # Safety bound, never hit by this corpus: the longest name is 15 chars,
        # so BOS + name + EOS is at most block_size + 1 tokens.
        toks = toks[: block_size + 1]
        sequences.append(toks)

    dynamic_len = max(len(s) for s in sequences) - 1  # what a max_len-padded batch would be
    T = block_size  # static, identical for every batch
    input_ids, target_ids, pad_masks, target_masks = [], [], [], []
    for s in sequences:
        n = len(s) - 1  # real input tokens; padding is appended after these
        inp = s[:n] + [PAD] * (T - n)
        tgt = s[1 : n + 1] + [0] * (T - n)  # 0 as dummy target for masked positions
        pmask = [False] * n + [True] * (T - n)
        tmask = [1.0] * n + [0.0] * (T - n)
        input_ids.append(inp)
        target_ids.append(tgt)
        pad_masks.append(pmask)
        target_masks.append(tmask)

    batch = (
        jnp.array(input_ids),
        jnp.array(target_ids),
        jnp.array(pad_masks),
        jnp.array(target_masks),
    )
    return batch, dynamic_len


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
static_shapes = set()  # input shapes actually fed to train_step
dynamic_shapes = set()  # shapes a max_len-padded loop would have produced
step_ms = []

for step in range(num_steps):
    batch, dynamic_len = make_batch(docs, step, batch_size)
    dynamic_shapes.add((batch_size, dynamic_len))

    is_new_shape = batch[0].shape not in static_shapes
    static_shapes.add(batch[0].shape)

    t0 = time.perf_counter()
    loss_val, params, m_state, v_state = train_step(
        params, m_state, v_state, batch, step, learning_rate * (1 - step / num_steps)
    )
    # JAX dispatch is asynchronous, so block before reading the clock.
    jax.block_until_ready((loss_val, params))
    step_ms.append((time.perf_counter() - t0) * 1000)

    if is_new_shape:
        print(f"step {step + 1:4d} | new input shape {batch[0].shape} -> XLA trace #{len(static_shapes)}")

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss_val:.4f} | {step_ms[-1]:7.2f} ms")

# The measurement that justifies padding to block_size. Every distinct shape
# costs one trace + compile of train_step (seconds); every repeat is free.
print(f"\ncompilations of train_step: {len(static_shapes)} (padding to the static block_size)")
print(f"  in-batch max_len values seen: {len(dynamic_shapes)} -> that many compilations if we padded dynamically")
print(f"  first step: {step_ms[0]:.1f} ms (trace + compile)")
print(f"  steps 2..{num_steps}: {sum(step_ms[1:]) / (num_steps - 1):.2f} ms mean, max {max(step_ms[1:]):.2f} ms")

# ---------------------------------------------------------------------------
# Inference (single-example, no vmap needed)
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
rng_key = jax.random.key(0)
for sample_idx in range(20):
    tokens = [BOS]
    for _ in range(block_size):
        input_ids = jnp.array(tokens)
        pad_mask = jnp.zeros(len(tokens), dtype=bool)
        logits = forward_single(params, input_ids, pad_mask)
        logits = logits[-1] / temperature
        rng_key, subkey = jax.random.split(rng_key)
        token_id = jax.random.categorical(subkey, logits).item()
        if token_id == BOS:
            break
        tokens.append(token_id)
    name = "".join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx + 1:2d}: {name}")
