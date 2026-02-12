"""
microGPT — JAX batched edition.

Same architecture as 05_jax, but with mini-batch training via jax.vmap:
  - Write the forward pass for a single example
  - vmap automatically vectorizes it across a batch
  - No manual padding logic — vmap handles the batch dimension
  - Scaled-up model (2 layers, 64-dim embeddings, context 16)
  - Batches of 32, 1000 training steps

This is JAX's signature trick: "write for one, run for many."
"""

import os
import math
import random
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

random.seed(42)

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'input.txt')
if not os.path.exists(input_path):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
PAD = vocab_size
vocab_size_with_pad = vocab_size + 1
print(f"vocab size: {vocab_size} (+1 pad = {vocab_size_with_pad})")

# ---------------------------------------------------------------------------
# Hyperparameters (scaled up)
# ---------------------------------------------------------------------------
n_embd = 64
n_head = 4
n_layer = 2
block_size = 16
head_dim = n_embd // n_head
batch_size = 32
num_steps = 1000

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
key = jax.random.PRNGKey(42)

def init_param(key, shape, std=0.02):
    return jax.random.normal(key, shape) * std

def split_keys(key, n):
    return jax.random.split(key, n)

keys = split_keys(key, 30)
ki = iter(keys)

params = {
    'wte': init_param(next(ki), (vocab_size_with_pad, n_embd)),
    'wpe': init_param(next(ki), (block_size, n_embd)),
    'lm_head': init_param(next(ki), (vocab_size, n_embd)),
}
for i in range(n_layer):
    params[f'l{i}.wq'] = init_param(next(ki), (n_embd, n_embd))
    params[f'l{i}.wk'] = init_param(next(ki), (n_embd, n_embd))
    params[f'l{i}.wv'] = init_param(next(ki), (n_embd, n_embd))
    params[f'l{i}.wo'] = jnp.zeros((n_embd, n_embd))
    params[f'l{i}.fc1'] = init_param(next(ki), (n_embd, 4 * n_embd))
    params[f'l{i}.fc2'] = jnp.zeros((4 * n_embd, n_embd))

num_params = sum(p.size for p in jax.tree.leaves(params))
print(f"num params: {num_params}")

# ---------------------------------------------------------------------------
# Forward pass (single example — vmap will handle the batch)
# ---------------------------------------------------------------------------
def rmsnorm(x):
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + 1e-5)

def forward_single(params, input_ids, pad_mask):
    """
    Forward pass for ONE sequence.
    input_ids: (T,) int array
    pad_mask: (T,) bool array, True where padded
    Returns: logits (T, vocab_size)
    """
    T = input_ids.shape[0]
    tok_emb = params['wte'][input_ids]          # (T, D)
    pos_emb = params['wpe'][jnp.arange(T)]      # (T, D)
    x = rmsnorm(tok_emb + pos_emb)

    for li in range(n_layer):
        x_res = x
        x_n = rmsnorm(x)

        Q = x_n @ params[f'l{li}.wq']
        K = x_n @ params[f'l{li}.wk']
        V = x_n @ params[f'l{li}.wv']

        Q_h = Q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        K_h = K.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        V_h = V.reshape(T, n_head, head_dim).transpose(1, 0, 2)

        att = Q_h @ K_h.transpose(0, 2, 1) / math.sqrt(head_dim)
        # Causal mask
        causal = jnp.triu(jnp.ones((T, T)), k=1).astype(bool)
        att = jnp.where(causal, -1e9, att)
        # Padding mask: don't attend to PAD positions
        att = jnp.where(pad_mask[None, None, :], -1e9, att)
        att = jax.nn.softmax(att, axis=-1)
        att = jnp.nan_to_num(att)

        attn_out = att @ V_h
        attn_cat = attn_out.transpose(1, 0, 2).reshape(T, n_embd)
        x = attn_cat @ params[f'l{li}.wo'] + x_res

        # MLP
        x_res2 = x
        x_n2 = rmsnorm(x)
        h = x_n2 @ params[f'l{li}.fc1']
        h = jax.nn.relu(h) ** 2
        x = h @ params[f'l{li}.fc2'] + x_res2

    logits = x @ params['lm_head'].T  # (T, V)
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
    target_log_probs = log_probs[
        jnp.arange(B)[:, None], jnp.arange(T)[None, :], targets
    ]  # (B, T)
    # Mask out padding positions
    target_log_probs = target_log_probs * target_mask
    loss = -jnp.sum(target_log_probs) / jnp.sum(target_mask)
    return loss

grad_fn = jit(grad(loss_fn))
loss_fn_jit = jit(loss_fn)

# ---------------------------------------------------------------------------
# Adam optimizer (functional)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m_state = jax.tree.map(jnp.zeros_like, params)
v_state = jax.tree.map(jnp.zeros_like, params)

def adam_update(params, grads, m_state, v_state, step, lr):
    new_params, new_m, new_v = {}, {}, {}
    for k in params:
        new_m[k] = beta1 * m_state[k] + (1 - beta1) * grads[k]
        new_v[k] = beta2 * v_state[k] + (1 - beta2) * grads[k] ** 2
        m_hat = new_m[k] / (1 - beta1 ** (step + 1))
        v_hat = new_v[k] / (1 - beta2 ** (step + 1))
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    return new_params, new_m, new_v

# ---------------------------------------------------------------------------
# Batching utility
# ---------------------------------------------------------------------------
def make_batch(docs, step, batch_size):
    """Create a padded batch of token sequences."""
    batch_docs = [docs[(step * batch_size + i) % len(docs)] for i in range(batch_size)]
    sequences = []
    for doc in batch_docs:
        toks = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        toks = toks[:block_size + 1]
        sequences.append(toks)

    max_len = max(len(s) for s in sequences)
    input_ids, target_ids, pad_masks, target_masks = [], [], [], []
    for s in sequences:
        n = len(s) - 1
        inp = s[:n] + [PAD] * (max_len - 1 - n)
        tgt = s[1:n+1] + [0] * (max_len - 1 - n)  # 0 as dummy target for masked positions
        pmask = [False] * n + [True] * (max_len - 1 - n)
        tmask = [1.0] * n + [0.0] * (max_len - 1 - n)
        input_ids.append(inp)
        target_ids.append(tgt)
        pad_masks.append(pmask)
        target_masks.append(tmask)

    return (
        jnp.array(input_ids),
        jnp.array(target_ids),
        jnp.array(pad_masks),
        jnp.array(target_masks),
    )

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for step in range(num_steps):
    input_ids, targets, pad_mask, target_mask = make_batch(docs, step, batch_size)

    loss_val = loss_fn_jit(params, input_ids, targets, pad_mask, target_mask)
    grads = grad_fn(params, input_ids, targets, pad_mask, target_mask)

    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    params, m_state, v_state = adam_update(params, grads, m_state, v_state, step, lr_t)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss_val:.4f}")

# ---------------------------------------------------------------------------
# Inference (single-example, no vmap needed)
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference ---")
rng_key = jax.random.PRNGKey(0)
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
    name = ''.join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx+1:2d}: {name}")
