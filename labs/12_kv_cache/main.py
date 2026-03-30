"""
microGPT — KV Cache edition.

Same model as lab 03, but with a KV-cache-aware inference path that avoids
recomputing Key and Value tensors for already-processed positions. This is
THE fundamental optimization behind every fast LLM serving system.
Without KV cache: each new token reprocesses the full sequence -> O(n^3) total.
With KV cache: each new token only computes attention for ONE position -> O(n^2) total.

KV caching is a standard inference optimization for autoregressive transformers,
implicit in the original "Attention Is All You Need" (Vaswani et al., 2017),
https://arxiv.org/abs/1706.03762 decoder design. The technique became essential
at scale as documented in "Efficient Transformers: A Survey" (Tay et al., 2022),
https://arxiv.org/abs/2009.06732. This lab implements the basic form: cache K and
V tensors from previous positions, compute only the new position's Q/K/V, and
concatenate with the cache. Every production serving system (vLLM, TensorRT-LLM,
etc.) uses this as its foundation.
"""

import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

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
# Model config
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head

# ===========================================================================
# Unified model — supports both standard and KV-cached inference
# ===========================================================================
# One model, two inference paths. When kv_cache=None, attention behaves like
# standard causal attention (same as lab 03). When kv_cache is provided,
# only the new positions compute Q/K/V and cached K/V are prepended.


class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    """Attention with optional KV cache for efficient autoregressive decoding."""

    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, kv_cache=None):
        B, T_new, C = x.shape
        q = self.wq(x).view(B, T_new, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T_new, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T_new, n_head, head_dim).transpose(1, 2)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)
        T_total = k.shape[2]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # Causal mask: row i (absolute pos T_total-T_new+i) attends to cols 0..abs_pos
        mask = torch.triu(torch.ones(T_new, T_total, device=x.device, dtype=torch.bool), diagonal=T_total - T_new + 1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).reshape(B, T_new, C)
        return self.wo(out), new_cache


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx, past_caches=None, start_pos=0):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(start_pos, start_pos + T, device=idx.device))
        x = self.norm_in(tok_emb + pos_emb)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = past_caches[i] if past_caches is not None else None
            x, new_cache = layer(x, kv_cache=cache_i)
            new_caches.append(new_cache)
        return self.lm_head(x), new_caches


# ===========================================================================
# Training (identical to lab 03 — ignores caches during training)
# ===========================================================================
device = "cpu"
model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = torch.tensor([tokens[:n]], device=device)
    targets = torch.tensor([tokens[1 : n + 1]], device=device)

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    lr_t = 1e-2 * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t
    optimizer.step()

    if (step + 1) % 200 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# ===========================================================================
# Inference comparison — naive (no cache) vs KV cache
# ===========================================================================
# Same model, same weights — two generation strategies.
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
num_samples = 20
model.eval()


def sample_token(logits):
    probs = F.softmax(logits[0, -1] / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_naive(num_samples=20):
    """Standard generation: re-run full sequence at every step. O(n^3) total attention."""
    total_attn_ops = 0
    names = []
    for sample_idx in range(num_samples):
        torch.manual_seed(1000 + sample_idx)
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], device=device)
            T = idx.shape[1]
            total_attn_ops += T * T * n_head
            logits, _ = model(idx)  # no cache — recomputes everything
            token_id = sample_token(logits)
            if token_id == BOS:
                break
            tokens.append(token_id)
        names.append("".join(uchars[t] for t in tokens[1:]))
    return names, total_attn_ops


def generate_with_cache(num_samples=20):
    """KV-cache generation: only process new token each step. O(n^2) total attention."""
    total_attn_ops = 0
    names = []
    for sample_idx in range(num_samples):
        torch.manual_seed(1000 + sample_idx)
        tokens = [BOS]
        past_caches = None
        for _ in range(block_size):
            if past_caches is None:
                idx = torch.tensor([tokens], device=device)
                T_new, start_pos = len(tokens), 0
            else:
                idx = torch.tensor([[tokens[-1]]], device=device)
                T_new, start_pos = 1, len(tokens) - 1
            total_attn_ops += T_new * len(tokens) * n_head
            logits, past_caches = model(idx, past_caches=past_caches, start_pos=start_pos)
            token_id = sample_token(logits)
            if token_id == BOS:
                break
            tokens.append(token_id)
        names.append("".join(uchars[t] for t in tokens[1:]))
    return names, total_attn_ops


print("\n" + "=" * 60)
print("INFERENCE COMPARISON: Naive vs KV Cache")
print("=" * 60)

with torch.no_grad():
    t0 = time.perf_counter()
    names_naive, ops_naive = generate_naive(num_samples)
    t_naive = time.perf_counter() - t0

    t0 = time.perf_counter()
    names_cached, ops_cached = generate_with_cache(num_samples)
    t_cached = time.perf_counter() - t0

# Verify identical outputs
print("\n--- Generated names (both methods produce identical output) ---")
all_match = all(n1 == n2 for n1, n2 in zip(names_naive, names_cached))
for i, (n1, n2) in enumerate(zip(names_naive, names_cached)):
    match = "OK" if n1 == n2 else "MISMATCH!"
    print(f"  {i + 1:2d}: {n1:12s}  |  {n2:12s}  [{match}]")
print(f"\nAll outputs identical: {all_match}")

# Operation counts
print("\n--- Attention operation counts (Q*K multiply-adds) ---")
print(f"  Naive (full recompute):  {ops_naive:,} ops")
print(f"  KV cache (incremental):  {ops_cached:,} ops")
print(f"  Reduction:               {ops_naive / ops_cached:.1f}x fewer operations")

# Timing
print("\n--- Wall-clock time ---")
print(f"  Naive:    {t_naive * 1000:.1f} ms")
print(f"  KV cache: {t_cached * 1000:.1f} ms")
if t_cached > 0:
    print(f"  Speedup:  {t_naive / t_cached:.2f}x")

# Theoretical analysis
T = block_size
print("\n--- Theoretical analysis ---")
print(f"  For a sequence of length T = {T}:")
print(f"  Naive total attention:   sum(t^2 for t=1..T) = T(T+1)(2T+1)/6 = {T * (T + 1) * (2 * T + 1) // 6} (per head)")
print(f"  Cached total attention:  sum(t   for t=1..T) = T(T+1)/2       = {T * (T + 1) // 2} (per head)")
print(f"  Ratio:                   (2T+1)/3 = {(2 * T + 1) / 3:.1f}x")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("WHY KV CACHE MATTERS")
print("=" * 60)
print("""
  Cache stores K,V from previous positions. Each new token computes only its
  own Q,K,V and attends to all cached K,V — no recomputation.

                  Without cache         With cache
  Step 1:         1 x 1  attention      1 x 1  attention
  Step 2:         2 x 2  attention      1 x 2  attention
  Step 3:         3 x 3  attention      1 x 3  attention
  ...
  Step T:         T x T  attention      1 x T  attention
  ─────────────────────────────────────────────────────────
  Total:          ~T^3/3 operations     ~T^2/2 operations

  At scale (7B model, 2048 ctx): ~680x speedup, ~1 GB cache per request.
  Prerequisite for speculative decoding, PagedAttention, disaggregated serving.
""")
