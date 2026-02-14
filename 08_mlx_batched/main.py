"""
microGPT — MLX batched edition.

Same architecture as 07_mlx, but with mini-batch training:
  - Batches of 32 sequences per step
  - Padding + attention masking (same approach as PyTorch batched)
  - Scaled-up model (2 layers, 64-dim embeddings, context 16)
  - 1000 training steps

Unlike JAX's vmap, MLX batching works the PyTorch way: reshape your tensors
to include a batch dimension and write the forward pass to handle (B, T, ...).
Same idea, same manual padding — unified memory is the difference.
"""

import os
import math
import random
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

random.seed(42)
mx.random.seed(42)

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
# Model
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim):
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

    def __call__(self, x, pad_mask=None):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
        # Causal mask
        causal = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
        att = mx.where(causal, mx.array(-1e9), att)
        # Padding mask: don't attend to PAD positions
        if pad_mask is not None:
            att = mx.where(pad_mask[:, None, None, :], mx.array(-1e9), att)
        att = mx.softmax(att, axis=-1)
        att = mx.where(mx.isnan(att), mx.array(0.0), att)

        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def __call__(self, x):
        h = self.fc1(x)
        h = mx.maximum(h, 0)
        return self.fc2(h)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def __call__(self, x, pad_mask=None):
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

    def __call__(self, idx, pad_mask=None):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(mx.arange(T))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.lm_head(x)


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
        tgt = s[1:n+1] + [0] * (max_len - 1 - n)
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
# Match original init: N(0, 0.08) for all weights
model.load_weights([(k, mx.random.normal(v.shape) * 0.08)
                    for k, v in mlx.utils.tree_flatten(model.parameters())])
num_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
print(f"num params: {num_params}")

learning_rate, beta1, beta2, eps_adam = 1e-2, 0.85, 0.99, 1e-8

def loss_fn(model, input_ids, targets, pad_mask, target_mask):
    logits = model(input_ids, pad_mask)  # (B, T, V)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    B, T, V = log_probs.shape
    # Gather log-probs for target tokens
    target_log_probs = log_probs[
        mx.arange(B)[:, None], mx.arange(T)[None, :], targets
    ]
    target_log_probs = target_log_probs * target_mask
    loss = -mx.sum(target_log_probs) / mx.sum(target_mask)
    return loss

loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=learning_rate, betas=[beta1, beta2], eps=eps_adam)

for step in range(num_steps):
    input_ids, targets, pad_mask, target_mask = make_batch(docs, step, batch_size)

    loss_val, grads = loss_and_grad(model, input_ids, targets, pad_mask, target_mask)

    lr_t = learning_rate * (1 - step / num_steps)
    optimizer.learning_rate = mx.array(lr_t)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss_val.item():.4f}")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
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
    name = ''.join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx+1:2d}: {name}")
