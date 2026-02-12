"""
microGPT — MLX edition.

Same architecture, running on Apple Silicon GPU via MLX.
MLX has a NumPy-like API with automatic differentiation and lazy evaluation.
Arrays live in unified memory — no CPU/GPU transfers needed.
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
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
n_embd = 16
n_head = 4
n_layer = 1
block_size = 8
head_dim = n_embd // n_head


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
        self.wo.weight = mx.zeros((n_embd, n_embd))

    def __call__(self, x):
        n = x.shape[0]
        q = self.wq(x).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        k = self.wk(x).reshape(n, n_head, head_dim).transpose(1, 0, 2)
        v = self.wv(x).reshape(n, n_head, head_dim).transpose(1, 0, 2)

        att = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
        mask = mx.triu(mx.ones((n, n)), k=1).astype(mx.bool_)
        att = mx.where(mask, mx.array(-1e9), att)
        att = mx.softmax(att, axis=-1)

        out = (att @ v).transpose(1, 0, 2).reshape(n, n_embd)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.fc2.weight = mx.zeros((n_embd, 4 * n_embd))

    def __call__(self, x):
        h = self.fc1(x)
        h = mx.maximum(h, 0) ** 2  # squared ReLU
        return self.fc2(h)


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
num_params = sum(p.size for _, p in mlx.utils.tree_flatten(model.parameters()))
print(f"num params: {num_params}")

learning_rate, beta1, beta2, eps_adam = 1e-2, 0.9, 0.95, 1e-8

def loss_fn(model, input_ids, targets):
    logits = model(input_ids)  # (n, V)
    # Cross-entropy
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    n = input_ids.shape[0]
    loss = -mx.mean(log_probs[mx.arange(n), targets])
    return loss

loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.Adam(learning_rate=learning_rate, betas=[beta1, beta2], eps=eps_adam)

num_steps = 500
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = mx.array(tokens[:n])
    targets = mx.array(tokens[1:n+1])

    loss_val, grads = loss_and_grad(model, input_ids, targets)

    # Cosine LR decay
    lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    optimizer.learning_rate = mx.array(lr_t)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss_val.item():.4f}")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference ---")
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
    name = ''.join(uchars[t] for t in tokens[1:])
    print(f"sample {sample_idx+1:2d}: {name}")
