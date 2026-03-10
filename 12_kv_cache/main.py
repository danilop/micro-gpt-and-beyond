"""
microGPT — KV Cache edition.

Same model as lab 03, but with a KV-cache-aware inference path that avoids
recomputing Key and Value tensors for already-processed positions. This is
THE fundamental optimization behind every fast LLM serving system.

Without KV cache: each new token reprocesses the full sequence → O(n³) total.
With KV cache: each new token only computes attention for ONE position → O(n²) total.
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
# Dataset
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ---------------------------------------------------------------------------
# Tokenizer (character-level, identical to the original)
# ---------------------------------------------------------------------------
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

# ===========================================================================
# Part 1: Standard model (same as lab 03)
# ===========================================================================


class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ===========================================================================
# Part 2: KV-Cache-aware model (same weights, different inference path)
# ===========================================================================


class CausalSelfAttentionKV(nn.Module):
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

        # Concatenate with cached keys/values if available
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # (B, nh, T_total, hd)
            v = torch.cat([cached_v, v], dim=2)

        new_cache = (k, v)

        T_total = k.shape[2]

        # Q has T_new rows, K has T_total columns
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Causal mask: each new query position can only attend to positions <= itself
        # Row i of Q corresponds to absolute position (T_total - T_new + i)
        # It can attend to columns 0 .. (T_total - T_new + i)
        mask = torch.zeros(T_new, T_total, device=x.device, dtype=torch.bool)
        for i in range(T_new):
            abs_pos = T_total - T_new + i
            mask[i, abs_pos + 1 :] = True
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T_new, C)
        return self.wo(out), new_cache


class BlockKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttentionKV()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class MicroGPTKV(nn.Module):
    """Same model as MicroGPT, but threads KV cache through for fast inference."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([BlockKV() for _ in range(n_layer)])
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
# Part 3: Training (standard model, identical to lab 03)
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

    logits = model(input_ids)
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
# Part 4: Inference comparison — naive vs KV cache
# ===========================================================================

# Copy trained weights into the KV-cache model
model_kv = MicroGPTKV().to(device)
model_kv.wte.weight.data.copy_(model.wte.weight.data)
model_kv.wpe.weight.data.copy_(model.wpe.weight.data)
for layer_kv, layer_std in zip(model_kv.layers, model.layers):
    layer_kv.attn.wq.weight.data.copy_(layer_std.attn.wq.weight.data)
    layer_kv.attn.wk.weight.data.copy_(layer_std.attn.wk.weight.data)
    layer_kv.attn.wv.weight.data.copy_(layer_std.attn.wv.weight.data)
    layer_kv.attn.wo.weight.data.copy_(layer_std.attn.wo.weight.data)
    layer_kv.mlp.fc1.weight.data.copy_(layer_std.mlp.fc1.weight.data)
    layer_kv.mlp.fc2.weight.data.copy_(layer_std.mlp.fc2.weight.data)
model_kv.lm_head.weight.data.copy_(model.lm_head.weight.data)

temperature = 0.5
num_samples = 20

model.eval()
model_kv.eval()


def generate_naive(model, num_samples=20):
    """Standard generation: re-run full sequence at every step. O(n^3) total attention."""
    total_attn_ops = 0
    names = []
    for sample_idx in range(num_samples):
        torch.manual_seed(1000 + sample_idx)
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], device=device)
            T = idx.shape[1]
            total_attn_ops += T * T * n_head  # T queries x T keys x n_head heads
            logits = model(idx)
            logits = logits[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
        name = "".join(uchars[t] for t in tokens[1:])
        names.append(name)
    return names, total_attn_ops


def generate_with_cache(model_kv, num_samples=20):
    """KV-cache generation: only process new token each step. O(n^2) total attention."""
    total_attn_ops = 0
    names = []
    for sample_idx in range(num_samples):
        torch.manual_seed(1000 + sample_idx)
        tokens = [BOS]
        past_caches = None
        for step_i in range(block_size):
            if past_caches is None:
                # Prefill: process all tokens so far (just BOS on first call)
                idx = torch.tensor([tokens], device=device)
                T_new = len(tokens)
                start_pos = 0
            else:
                # Decode: process only the new token
                idx = torch.tensor([[tokens[-1]]], device=device)
                T_new = 1
                start_pos = len(tokens) - 1

            T_total = len(tokens)  # total sequence length after this step
            total_attn_ops += T_new * T_total * n_head

            logits, past_caches = model_kv(idx, past_caches=past_caches, start_pos=start_pos)
            logits = logits[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
        name = "".join(uchars[t] for t in tokens[1:])
        names.append(name)
    return names, total_attn_ops


print("\n" + "=" * 60)
print("INFERENCE COMPARISON: Naive vs KV Cache")
print("=" * 60)

with torch.no_grad():
    # Naive generation
    t0 = time.perf_counter()
    names_naive, ops_naive = generate_naive(model, num_samples)
    t_naive = time.perf_counter() - t0

    # KV-cache generation
    t0 = time.perf_counter()
    names_cached, ops_cached = generate_with_cache(model_kv, num_samples)
    t_cached = time.perf_counter() - t0

# Verify identical outputs
print("\n--- Generated names (both methods produce identical output) ---")
all_match = True
for i, (n1, n2) in enumerate(zip(names_naive, names_cached)):
    match = "OK" if n1 == n2 else "MISMATCH!"
    if n1 != n2:
        all_match = False
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
print("\n--- Theoretical analysis ---")
print(f"  For a sequence of length T = {block_size}:")
print(
    f"  Naive total attention:   sum(t^2 for t=1..T) = T(T+1)(2T+1)/6 = {block_size * (block_size + 1) * (2 * block_size + 1) // 6} (per head)"
)
print(
    f"  Cached total attention:  sum(t   for t=1..T) = T(T+1)/2       = {block_size * (block_size + 1) // 2} (per head)"
)
print(f"  Ratio:                   (2T+1)/3 = {(2 * block_size + 1) / 3:.1f}x")

# ===========================================================================
# Part 5: Summary
# ===========================================================================
print("\n" + "=" * 60)
print("WHY KV CACHE MATTERS")
print("=" * 60)
print("""
During autoregressive generation, a transformer produces one token at a time.
Without caching, every new token requires recomputing Keys and Values for ALL
previous positions — even though those K,V values haven't changed.

KV cache stores the K and V tensors from previous positions. When generating
the next token, only the NEW position's Q, K, V are computed. The new K,V are
appended to the cache, and attention is computed between the one new Q and ALL
cached K,V vectors.

                  Without cache         With cache
  Step 1:         1 x 1  attention      1 x 1  attention
  Step 2:         2 x 2  attention      1 x 2  attention
  Step 3:         3 x 3  attention      1 x 3  attention
  ...
  Step T:         T x T  attention      1 x T  attention
  ─────────────────────────────────────────────────────────
  Total:          ~T^3/3 operations     ~T^2/2 operations

At scale this is enormous. For a 7B-parameter model with 2048 context:
  - KV cache size: ~1 GB per request (stored in GPU memory)
  - Without cache: generation would be ~680x slower (2*2048+1)/3
  - Every production LLM serving system uses KV caching

KV caching is also the prerequisite for:
  - Speculative decoding (lab 17): draft model fills cache, verified in batch
  - PagedAttention (lab 19): virtual memory for KV cache blocks
  - Continuous batching: share GPU across requests with different cache sizes
""")
