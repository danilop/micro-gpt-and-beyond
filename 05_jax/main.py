"""
microGPT — JAX edition.

Same architecture as the PyTorch version, but in JAX's functional style:
  - All parameters are explicit pytrees (no hidden state)
  - Forward pass is a pure function (no side effects)
  - Gradients via jax.grad (automatic, like PyTorch, but functional)
  - JIT compilation for speed
  - Explicit PRNG key threading

This shows how the same transformer looks in a purely functional framework.
"""

import os
import math
import random
import jax
import jax.numpy as jnp
from jax import grad, jit

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
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Parameters as a flat dict (pytree)
# ---------------------------------------------------------------------------
n_embd = 16
n_head = 4
n_layer = 1
block_size = 8
head_dim = n_embd // n_head

key = jax.random.PRNGKey(42)

def init_param(key, shape, std=0.02):
    return jax.random.normal(key, shape) * std

def split_keys(key, n):
    return jax.random.split(key, n)

keys = split_keys(key, 20)
ki = iter(keys)

params = {
    'wte': init_param(next(ki), (vocab_size, n_embd)),
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
# Pure-function forward pass
# ---------------------------------------------------------------------------
def rmsnorm(x):
    ms = jnp.mean(x ** 2, axis=-1, keepdims=True)
    return x / jnp.sqrt(ms + 1e-5)

def forward(params, input_ids):
    """
    params: dict of arrays
    input_ids: (n,) int array
    Returns: logits (n, vocab_size)
    """
    n = input_ids.shape[0]
    tok_emb = params['wte'][input_ids]          # (n, D)
    pos_emb = params['wpe'][jnp.arange(n)]      # (n, D)
    x = rmsnorm(tok_emb + pos_emb)

    for li in range(n_layer):
        x_res = x
        x_n = rmsnorm(x)

        Q = x_n @ params[f'l{li}.wq']  # (n, D)
        K = x_n @ params[f'l{li}.wk']
        V = x_n @ params[f'l{li}.wv']

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
        x = attn_cat @ params[f'l{li}.wo'] + x_res

        # MLP
        x_res2 = x
        x_n2 = rmsnorm(x)
        h = x_n2 @ params[f'l{li}.fc1']
        h = jax.nn.relu(h) ** 2
        x = h @ params[f'l{li}.fc2'] + x_res2

    logits = x @ params['lm_head'].T  # (n, V)
    return logits


def loss_fn(params, input_ids, targets):
    """Cross-entropy loss, pure function suitable for jax.grad."""
    logits = forward(params, input_ids)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    n = input_ids.shape[0]
    loss = -jnp.mean(log_probs[jnp.arange(n), targets])
    return loss

# JIT-compile the gradient function
grad_fn = jit(grad(loss_fn))
loss_fn_jit = jit(loss_fn)

# ---------------------------------------------------------------------------
# Adam optimizer (functional — no hidden state mutation)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8
m_state = jax.tree.map(jnp.zeros_like, params)
v_state = jax.tree.map(jnp.zeros_like, params)

def adam_update(params, grads, m_state, v_state, step, lr):
    new_params = {}
    new_m = {}
    new_v = {}
    for k in params:
        new_m[k] = beta1 * m_state[k] + (1 - beta1) * grads[k]
        new_v[k] = beta2 * v_state[k] + (1 - beta2) * grads[k] ** 2
        m_hat = new_m[k] / (1 - beta1 ** (step + 1))
        v_hat = new_v[k] / (1 - beta2 ** (step + 1))
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    return new_params, new_m, new_v

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
num_steps = 500
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = jnp.array(tokens[:n])
    targets = jnp.array(tokens[1:n+1])

    loss_val = loss_fn_jit(params, input_ids, targets)
    grads = grad_fn(params, input_ids, targets)

    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    params, m_state, v_state = adam_update(params, grads, m_state, v_state, step, lr_t)

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss_val:.4f}")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference ---")
rng_key = jax.random.PRNGKey(0)
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
    name = ''.join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx+1:2d}: {name}")
