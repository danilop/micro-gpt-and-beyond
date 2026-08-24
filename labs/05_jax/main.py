"""
microGPT: JAX edition.

Same decoder-only GPT architecture as the PyTorch version, based on
"Attention Is All You Need" (Vaswani et al., 2017),
https://arxiv.org/abs/1706.03762, but expressed in JAX's purely functional
paradigm, with no classes, no mutation and explicit PRNG keys. The functional
approach reflects the style described in "Compiling machine learning programs
via high-level tracing" (Frostig et al., 2018),
https://mlsys.org/Conferences/doc/2018/146.pdf.

Key differences from the PyTorch version:
  - All parameters are explicit pytrees (no hidden state)
  - Forward pass is a pure function (no side effects)
  - Gradients via jax.value_and_grad (automatic, like PyTorch, but functional)
  - The whole training step (loss, gradients, Adam) is one jitted function
  - Explicit PRNG key threading, with modern typed keys (jax.random.key)

The training loop also reports where XLA compiles: jit specializes on input
shapes, and one name per step means a new sequence length every so often. The
corpus allows 14 lengths and a 1000-step run hits 12 of them, so the same
function is traced a dozen times, scattered through training. That is the JAX
gotcha this lab makes visible instead of hiding.
"""

import math
import os
import random
import time

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad

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
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Parameters as a flat dict (pytree)
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head

# jax.random.key is the modern typed-key constructor; jax.random.PRNGKey is the
# legacy uint32-pair form. Same algorithm, but a typed key carries its
# implementation in its dtype, which is what current JAX expects everywhere.
key = jax.random.key(42)


def init_param(key, shape, std=0.08):
    return jax.random.normal(key, shape) * std


# One key per parameter tensor: wte, wpe, lm_head, plus six per layer.
num_param_tensors = 3 + 6 * n_layer
keys = jax.random.split(key, num_param_tensors)
ki = iter(keys)

params = {
    "wte": init_param(next(ki), (vocab_size, n_embd)),
    "wpe": init_param(next(ki), (block_size, n_embd)),
    "lm_head": init_param(next(ki), (vocab_size, n_embd)),
}
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
# Pure-function forward pass
# ---------------------------------------------------------------------------
def rmsnorm(x):
    ms = jnp.mean(x**2, axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + 1e-5)


def forward(params, input_ids):
    """
    params: dict of arrays
    input_ids: (n,) int array
    Returns: logits (n, vocab_size)
    """
    n = input_ids.shape[0]
    tok_emb = params["wte"][input_ids]  # (n, D)
    pos_emb = params["wpe"][jnp.arange(n)]  # (n, D)
    x = rmsnorm(tok_emb + pos_emb)

    for li in range(n_layer):
        x_res = x
        x_n = rmsnorm(x)

        Q = x_n @ params[f"l{li}.wq"]  # (n, D)
        K = x_n @ params[f"l{li}.wk"]
        V = x_n @ params[f"l{li}.wv"]

        # Multi-head reshape: (n, nh, hd) -> (nh, n, hd)
        Q_h = Q.reshape(n, n_head, head_dim).transpose(1, 0, 2)
        K_h = K.reshape(n, n_head, head_dim).transpose(1, 0, 2)
        V_h = V.reshape(n, n_head, head_dim).transpose(1, 0, 2)

        att = Q_h @ K_h.transpose(0, 2, 1) / math.sqrt(head_dim)
        causal_mask = jnp.triu(jnp.ones((n, n)), k=1).astype(bool)
        att = jnp.where(causal_mask, -1e9, att)
        att = jax.nn.softmax(att, axis=-1)

        attn_out = att @ V_h  # (nh, n, hd)
        attn_cat = attn_out.transpose(1, 0, 2).reshape(n, n_embd)
        x = attn_cat @ params[f"l{li}.wo"] + x_res

        # MLP
        x_res2 = x
        x_n2 = rmsnorm(x)
        h = x_n2 @ params[f"l{li}.fc1"]
        h = jax.nn.relu(h)
        x = h @ params[f"l{li}.fc2"] + x_res2

    logits = x @ params["lm_head"].T  # (n, V)
    return logits


def loss_fn(params, input_ids, targets):
    """Cross-entropy loss, pure function suitable for jax.value_and_grad."""
    logits = forward(params, input_ids)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    n = input_ids.shape[0]
    loss = -jnp.mean(log_probs[jnp.arange(n), targets])
    return loss


# ---------------------------------------------------------------------------
# Adam optimizer (functional, with no hidden state mutation)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8
m_state = jax.tree.map(jnp.zeros_like, params)
v_state = jax.tree.map(jnp.zeros_like, params)


# ---------------------------------------------------------------------------
# One jitted training step: loss + gradients + Adam
# ---------------------------------------------------------------------------
# Two things worth noticing here.
#
# 1. Use value_and_grad rather than grad. jax.grad alone returns only the gradients,
#    so printing the loss as well would mean running the forward pass twice. The
#    backward pass already computes the loss on its way through, so
#    value_and_grad hands it back for free.
#
# 2. The optimizer is expressed with jax.tree.map instead of a Python loop over
#    dict keys. Same arithmetic, but now the whole step (forward, backward and
#    update) is a single pure function that jit can compile as one XLA program,
#    instead of a compiled gradient call surrounded by interpreted Python.
@jit
def train_step(params, m_state, v_state, input_ids, targets, step, lr):
    loss, grads = value_and_grad(loss_fn)(params, input_ids, targets)
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
# Training loop
# ---------------------------------------------------------------------------
num_steps = 1000
seen_lengths = set()  # every new sequence length costs one XLA compilation
compile_ms, cached_ms = [], []

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = jnp.array(tokens[:n])
    targets = jnp.array(tokens[1 : n + 1])

    is_new_shape = n not in seen_lengths
    seen_lengths.add(n)

    t0 = time.perf_counter()
    loss_val, params, m_state, v_state = train_step(
        params, m_state, v_state, input_ids, targets, step, learning_rate * (1 - step / num_steps)
    )
    # JAX dispatch is asynchronous: without blocking, we would be timing the
    # enqueue and not the work.
    jax.block_until_ready((loss_val, params))
    dt = (time.perf_counter() - t0) * 1000

    if is_new_shape:
        compile_ms.append(dt)
        print(f"step {step + 1:4d} | new sequence length n={n:2d} -> XLA trace #{len(seen_lengths)} | {dt:7.1f} ms")
    else:
        cached_ms.append(dt)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss_val:.4f} | {dt:6.2f} ms")

# The point of the two numbers below: jit is not "slow once, then fast". It is
# "slow once per input shape", and the shapes here keep arriving throughout
# training because names have 14 different lengths.
print(f"\ndistinct sequence lengths: {len(seen_lengths)} -> {len(compile_ms)} traces of the same train_step")
print(f"  first-time-shape steps: {sum(compile_ms) / len(compile_ms):8.1f} ms mean (compilation included)")
print(f"  cached-shape steps:     {sum(cached_ms) / len(cached_ms):8.2f} ms mean")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
rng_key = jax.random.key(0)
for sample_idx in range(20):
    tokens = [BOS]
    for _ in range(block_size):
        input_ids = jnp.array(tokens)
        logits = forward(params, input_ids)
        logits = logits[-1] / temperature
        rng_key, subkey = jax.random.split(rng_key)
        token_id = jax.random.categorical(subkey, logits).item()
        if token_id == BOS:
            break
        tokens.append(token_id)
    name = "".join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx + 1:2d}: {name}")
