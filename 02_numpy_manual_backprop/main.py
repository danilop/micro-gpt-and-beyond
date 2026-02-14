"""
microGPT — NumPy edition with manual backpropagation.

Same architecture as the pure-Python version, but using NumPy arrays for
vectorized math. Every gradient is derived and coded by hand — no autograd.
This is what PyTorch computes for you behind the scenes.
"""

import os
import math
import random
import numpy as np

random.seed(42)
np.random.seed(42)

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
# Parameters — plain NumPy arrays + matching gradient arrays
# ---------------------------------------------------------------------------
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

def param(shape, std=0.08):
    w = np.random.randn(*shape).astype(np.float64) * std
    return w

# All parameters stored as a dict of arrays
P = {
    'wte': param((vocab_size, n_embd)),
    'wpe': param((block_size, n_embd)),
    'lm_head': param((vocab_size, n_embd)),
}
for i in range(n_layer):
    P[f'l{i}.wq'] = param((n_embd, n_embd))
    P[f'l{i}.wk'] = param((n_embd, n_embd))
    P[f'l{i}.wv'] = param((n_embd, n_embd))
    P[f'l{i}.wo'] = param((n_embd, n_embd))
    P[f'l{i}.fc1'] = param((n_embd, 4 * n_embd))
    P[f'l{i}.fc2'] = param((4 * n_embd, n_embd))

# Gradient buffers (same shapes, zeroed)
G = {k: np.zeros_like(v) for k, v in P.items()}

num_params = sum(v.size for v in P.values())
print(f"num params: {num_params}")

# ---------------------------------------------------------------------------
# Forward + Backward (manual)
# ---------------------------------------------------------------------------
def rmsnorm_fwd(x):
    """x: (T, D) -> out: (T, D), cache"""
    ms = np.mean(x ** 2, axis=-1, keepdims=True)  # (T, 1)
    scale = 1.0 / np.sqrt(ms + 1e-5)              # (T, 1)
    out = x * scale
    return out, (x, scale, ms)

def rmsnorm_bwd(dout, cache):
    x, scale, ms = cache
    D = x.shape[-1]
    dx = dout * scale - x * scale ** 3 * np.sum(dout * x, axis=-1, keepdims=True) / D
    return dx

def softmax_fwd(logits):
    """logits: (T, V) or (nh, T, T2) -> probs same shape"""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)

def softmax_bwd(probs, dprobs):
    """Element-wise: dp - p * sum(dp * p)"""
    s = np.sum(dprobs * probs, axis=-1, keepdims=True)
    return probs * (dprobs - s)

def relu_fwd(x):
    mask = (x > 0).astype(x.dtype)
    relu_x = x * mask
    return relu_x, mask

def relu_bwd(dout, cache):
    mask = cache
    return dout * mask


def forward(tokens_np):
    """
    tokens_np: (T,) int array of token ids
    Returns: loss (scalar), cache dict for backward
    """
    T = len(tokens_np)
    n = min(block_size, T - 1)
    input_ids = tokens_np[:n]   # (n,)
    targets = tokens_np[1:n+1]  # (n,)

    cache = {}

    # Embeddings
    tok_emb = P['wte'][input_ids]   # (n, D)
    pos_emb = P['wpe'][np.arange(n)] # (n, D)
    x = tok_emb + pos_emb

    # Initial RMSNorm
    x, cache['norm_in'] = rmsnorm_fwd(x)

    for li in range(n_layer):
        x_res = x.copy()

        # Pre-attention RMSNorm
        x_normed, cache[f'l{li}.norm1'] = rmsnorm_fwd(x)

        # QKV projections: (n, D) @ (D, D) -> (n, D)
        Q = x_normed @ P[f'l{li}.wq']  # (n, D)
        K = x_normed @ P[f'l{li}.wk']
        V = x_normed @ P[f'l{li}.wv']

        # Reshape for multi-head: (n, nh, hd) -> (nh, n, hd)
        Q_h = Q.reshape(n, n_head, head_dim).transpose(1, 0, 2)  # (nh, n, hd)
        K_h = K.reshape(n, n_head, head_dim).transpose(1, 0, 2)
        V_h = V.reshape(n, n_head, head_dim).transpose(1, 0, 2)

        # Attention scores
        att = Q_h @ K_h.transpose(0, 2, 1) / math.sqrt(head_dim)  # (nh, n, n)
        # Causal mask
        causal_mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        att = np.where(causal_mask, -1e9, att)
        att_probs = softmax_fwd(att)  # (nh, n, n)

        # Weighted sum of values
        attn_out = att_probs @ V_h  # (nh, n, hd)
        # Reshape back: (n, D)
        attn_out_cat = attn_out.transpose(1, 0, 2).reshape(n, n_embd)

        # Output projection
        attn_proj = attn_out_cat @ P[f'l{li}.wo']  # (n, D)

        # Residual
        x = attn_proj + x_res

        cache[f'l{li}.attn'] = (x_res, x_normed, Q, K, V, Q_h, K_h, V_h, att_probs, attn_out, attn_out_cat)

        # Pre-MLP RMSNorm
        x_res2 = x.copy()
        x_normed2, cache[f'l{li}.norm2'] = rmsnorm_fwd(x)

        # MLP
        hidden = x_normed2 @ P[f'l{li}.fc1']  # (n, 4D)
        act, cache[f'l{li}.act'] = relu_fwd(hidden)
        mlp_out = act @ P[f'l{li}.fc2']  # (n, D)

        x = mlp_out + x_res2
        cache[f'l{li}.mlp'] = (x_res2, x_normed2, hidden, act)

    # LM head
    logits = x @ P['lm_head'].T  # (n, vocab_size)

    # Loss: cross-entropy
    probs = softmax_fwd(logits)  # (n, vocab_size)
    log_probs = -np.log(probs[np.arange(n), targets] + 1e-10)
    loss = log_probs.mean()

    cache['final'] = (x, logits, probs, targets, input_ids, n)
    return loss, cache


def backward(cache):
    """Compute gradients for all parameters, accumulate into G."""
    x, logits, probs, targets, input_ids, n = cache['final']

    # d(loss)/d(logits) via softmax + cross-entropy shortcut
    dlogits = probs.copy()  # (n, V)
    dlogits[np.arange(n), targets] -= 1.0
    dlogits /= n

    # LM head: logits = x @ lm_head.T
    G['lm_head'] += dlogits.T @ x       # (V, D)
    dx = dlogits @ P['lm_head']          # (n, D)

    for li in reversed(range(n_layer)):
        # --- MLP backward ---
        x_res2, x_normed2, hidden, act = cache[f'l{li}.mlp']

        dx_res2 = dx.copy()
        # mlp_out = act @ fc2
        dact = dx @ P[f'l{li}.fc2'].T           # (n, 4D)
        G[f'l{li}.fc2'] += act.T @ dx            # (4D, D)

        dhidden = relu_bwd(dact, cache[f'l{li}.act'])

        # hidden = x_normed2 @ fc1
        dx_normed2 = dhidden @ P[f'l{li}.fc1'].T  # (n, D)
        G[f'l{li}.fc1'] += x_normed2.T @ dhidden   # (D, 4D)

        # RMSNorm backward
        dx_pre_mlp = rmsnorm_bwd(dx_normed2, cache[f'l{li}.norm2'])
        dx = dx_pre_mlp + dx_res2  # residual

        # --- Attention backward ---
        x_res, x_normed, Q, K, V, Q_h, K_h, V_h, att_probs, attn_out, attn_out_cat = cache[f'l{li}.attn']

        dx_res = dx.copy()

        # attn_proj = attn_out_cat @ wo
        dattn_out_cat = dx @ P[f'l{li}.wo'].T   # (n, D)
        G[f'l{li}.wo'] += attn_out_cat.T @ dx    # (D, D)

        # Reshape back to multi-head: (n, D) -> (nh, n, hd)
        dattn_out = dattn_out_cat.reshape(n, n_head, head_dim).transpose(1, 0, 2)

        # attn_out = att_probs @ V_h
        datt_probs = dattn_out @ V_h.transpose(0, 2, 1)  # (nh, n, n)
        dV_h = att_probs.transpose(0, 2, 1) @ dattn_out   # (nh, n, hd)

        # Softmax backward on attention
        datt = softmax_bwd(att_probs, datt_probs)  # (nh, n, n)
        # Causal mask: zero out upper triangle grads (they were -inf)
        causal_mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        datt = np.where(causal_mask, 0.0, datt)
        datt /= math.sqrt(head_dim)

        # att = Q_h @ K_h^T
        dQ_h = datt @ K_h           # (nh, n, hd)
        dK_h = datt.transpose(0, 2, 1) @ Q_h  # (nh, n, hd)

        # Reshape heads back to (n, D)
        dQ = dQ_h.transpose(1, 0, 2).reshape(n, n_embd)
        dK = dK_h.transpose(1, 0, 2).reshape(n, n_embd)
        dV = dV_h.transpose(1, 0, 2).reshape(n, n_embd)

        # QKV projections: Q = x_normed @ wq, etc.
        dx_normed = dQ @ P[f'l{li}.wq'].T + dK @ P[f'l{li}.wk'].T + dV @ P[f'l{li}.wv'].T
        G[f'l{li}.wq'] += x_normed.T @ dQ
        G[f'l{li}.wk'] += x_normed.T @ dK
        G[f'l{li}.wv'] += x_normed.T @ dV

        # RMSNorm backward
        dx_pre_attn = rmsnorm_bwd(dx_normed, cache[f'l{li}.norm1'])
        dx = dx_pre_attn + dx_res  # residual

    # Initial RMSNorm backward
    dx = rmsnorm_bwd(dx, cache['norm_in'])

    # Embedding gradients
    for i, tid in enumerate(input_ids):
        G['wte'][tid] += dx[i]
    for i in range(n):
        G['wpe'][i] += dx[i]


# ---------------------------------------------------------------------------
# Adam optimizer (manual, operating on the P/G dicts)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8
M = {k: np.zeros_like(v) for k, v in P.items()}  # first moment
Vb = {k: np.zeros_like(v) for k, v in P.items()}  # second moment

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    tokens_np = np.array(tokens)

    # Zero gradients
    for k in G:
        G[k].fill(0.0)

    loss, cache = forward(tokens_np)
    backward(cache)

    # Adam update with linear LR decay
    lr_t = learning_rate * (1 - step / num_steps)
    for k in P:
        M[k] = beta1 * M[k] + (1 - beta1) * G[k]
        Vb[k] = beta2 * Vb[k] + (1 - beta2) * G[k] ** 2
        m_hat = M[k] / (1 - beta1 ** (step + 1))
        v_hat = Vb[k] / (1 - beta2 ** (step + 1))
        P[k] -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss:.4f}")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")

def inference_forward(token_ids):
    """Simplified forward pass for inference (no loss, no cache needed)."""
    n = len(token_ids)
    tok_emb = P['wte'][token_ids]
    pos_emb = P['wpe'][np.arange(n)]
    x = tok_emb + pos_emb
    x, _ = rmsnorm_fwd(x)
    for li in range(n_layer):
        x_res = x.copy()
        x_n, _ = rmsnorm_fwd(x)
        Q = x_n @ P[f'l{li}.wq']
        K = x_n @ P[f'l{li}.wk']
        V_val = x_n @ P[f'l{li}.wv']
        Q_h = Q.reshape(n, n_head, head_dim).transpose(1, 0, 2)
        K_h = K.reshape(n, n_head, head_dim).transpose(1, 0, 2)
        V_h = V_val.reshape(n, n_head, head_dim).transpose(1, 0, 2)
        att = Q_h @ K_h.transpose(0, 2, 1) / math.sqrt(head_dim)
        causal_mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        att = np.where(causal_mask, -1e9, att)
        att_probs = softmax_fwd(att)
        attn_out = att_probs @ V_h
        attn_cat = attn_out.transpose(1, 0, 2).reshape(n, n_embd)
        x = attn_cat @ P[f'l{li}.wo'] + x_res
        x_res2 = x.copy()
        x_n2, _ = rmsnorm_fwd(x)
        h = x_n2 @ P[f'l{li}.fc1']
        mask = (h > 0).astype(h.dtype)
        h = h * mask
        x = h @ P[f'l{li}.fc2'] + x_res2
    return x @ P['lm_head'].T  # (n, V)

for sample_idx in range(20):
    generated = [BOS]
    sample = []
    for _ in range(block_size):
        logits = inference_forward(np.array(generated))
        logits = logits[-1] / temperature
        probs = softmax_fwd(logits.reshape(1, -1))[0]
        token_id = np.random.choice(vocab_size, p=probs)
        if token_id == BOS:
            break
        generated.append(token_id)
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
