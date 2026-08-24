"""
microGPT: KV Cache edition.

Same model as lab 03, but with a KV-cache-aware inference path that avoids
recomputing Key and Value tensors for already-processed positions. This is
THE fundamental optimization behind every fast LLM serving system.

Be precise about which cost is being counted, because two different quantities
get called "the complexity of decoding" and they have different exponents. Over
a T-token generation:

  attention scores    naive O(T^3)   cached O(T^2)
  projections + MLP   naive O(T^2)   cached O(T)

Every "cubic to quadratic" claim in this lab, including the operation counter it
prints, is about the attention term, the one that dominates at long context and
the only one the cache changes asymptotically.

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
from itertools import pairwise

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
# Unified model, supporting both standard and KV-cached inference
# ===========================================================================
# One model with two inference paths. When kv_cache=None, attention behaves like
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
# Training (identical to lab 03, ignoring caches during training)
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
# Inference comparison: naive (no cache) vs KV cache
# ===========================================================================
# Same model, same weights, two generation strategies.
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
num_samples = 20
model.eval()


def sample_token(logits):
    probs = F.softmax(logits[0, -1] / temperature, dim=-1)
    return torch.multinomial(probs, 1).item()


def generate_naive(num_samples=20):
    """Standard generation: re-run full sequence at every step. O(T^3) attention work."""
    total_attn_ops = 0
    total_steps = 0
    names = []
    for sample_idx in range(num_samples):
        torch.manual_seed(1000 + sample_idx)
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], device=device)
            T = idx.shape[1]
            total_attn_ops += T * T * n_head
            total_steps += 1
            logits, _ = model(idx)  # no cache, so this recomputes everything
            token_id = sample_token(logits)
            if token_id == BOS:
                break
            tokens.append(token_id)
        names.append("".join(uchars[t] for t in tokens[1:]))
    return names, total_attn_ops, total_steps


def generate_with_cache(num_samples=20):
    """KV-cache generation: only process new token each step. O(T^2) attention work."""
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
    names_naive, ops_naive, steps_naive = generate_naive(num_samples)
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

# ---------------------------------------------------------------------------
# The headline number: attention operations counted, not timed
# ---------------------------------------------------------------------------
# This is a count of work actually performed, so it is exact and it is what the
# optimization is about. Wall clock comes later, with caveats.
print("\n--- Attention operation counts (Q*K multiply-adds, counted exactly) ---")
print(f"  Naive (full recompute):  {ops_naive:,} ops")
print(f"  KV cache (incremental):  {ops_cached:,} ops")
print(f"  Reduction:               {ops_naive / ops_cached:.1f}x fewer attention operations")

# Reconcile that measured reduction with the closed form, because they differ
# and the reason is worth one line: the formula is evaluated at T = block_size,
# but generation stops at BOS long before position 16.
T = block_size
avg_steps = steps_naive / num_samples
print("\n--- Reconciling with the closed form ---")
print(f"  For a full sequence of length T = {T}:")
print(f"  Naive total attention:   sum(t^2 for t=1..T) = T(T+1)(2T+1)/6 = {T * (T + 1) * (2 * T + 1) // 6} (per head)")
print(f"  Cached total attention:  sum(t   for t=1..T) = T(T+1)/2       = {T * (T + 1) // 2} (per head)")
print(f"  Ratio:                   (2T+1)/3 = {(2 * T + 1) / 3:.1f}x")
print(f"\n  The measured reduction is {ops_naive / ops_cached:.1f}x, not {(2 * T + 1) / 3:.1f}x, and nothing is wrong.")
print(f"  These names average {avg_steps:.1f} decode steps, not {T}: generation stops on BOS.")
print(
    f"  Evaluate the same formula at T = {avg_steps:.1f} and you get {(2 * avg_steps + 1) / 3:.1f}x. The saving grows"
)
print("  with sequence length, so a toy corpus of six-letter names sees the small end of it.")

# ---------------------------------------------------------------------------
# Wall clock, end to end, and why it says the opposite
# ---------------------------------------------------------------------------
print("\n--- Wall-clock time, whole generation loop ---")
print(f"  Naive:    {t_naive * 1000:.1f} ms")
print(f"  KV cache: {t_cached * 1000:.1f} ms")
print(f"  Ratio:    {t_naive / t_cached:.2f}x  <- naive time / cached time, not a speedup figure")
print("\n  Do not read that ratio as the value of the KV cache. It is a measurement of")
print("  Python. One layer, 16 dimensions, six-token names: each forward pass is a few")
print("  dozen microseconds of tensor arithmetic wrapped in a few hundred microseconds of")
print("  interpreter and dispatch overhead. Both paths make the same NUMBER of forward")
print("  calls, so both pay that overhead the same number of times, and the arithmetic the")
print("  cache eliminates is too small to rise above the noise. Re-run this file and the")
print("  ratio moves; it has been seen below 1.00x on a loaded machine. That is a statement")
print("  about the benchmark, not about the optimization. The section below fixes the")
print("  benchmark instead of explaining away the number.")

# ---------------------------------------------------------------------------
# Where the speedup actually lives: one decode step, at real context lengths
# ---------------------------------------------------------------------------
# To see the effect, measure the thing the cache changes and nothing else: a
# single decode step, at a context length a serving system would actually reach.
# A fresh attention module on random activations is enough, since timing depends on
# tensor shapes, not on trained weights.
#
#   naive  = recompute Q,K,V for the whole prefix, run a T x T attention
#   cached = compute Q,K,V for one position, run a 1 x T attention
#
# Theory says the attention term shrinks by a factor of T. Reality falls short
# of that, and the gap is instructive.
#
# A microbenchmark is only worth printing if its numbers are stable, so this one
# is defended twice. Inside a timing run, take the minimum of many repetitions:
# anything else sharing the CPU can only ever make a call slower, so the mean
# measures the machine's mood while the minimum measures the code. Across runs,
# take the median, so one throttled run cannot set the reported number.
#
# The repetition count is not a fixed number, because no fixed number serves both
# ends of this table: a cached call is microseconds, where hundreds of reps cost
# nothing and five reps measure mostly noise, while a naive call at T = 2048 can
# take a large fraction of a second all by itself. So each run repeats until a
# time budget is spent, with a floor and a ceiling, and prints how many reps it
# actually got. A cell that lands on the floor did so because one call already
# costs more than the whole budget: fine when that call is tens of milliseconds,
# a reason to distrust the number when it is microseconds.
BENCH_ROUNDS = 5  # independent timing runs per cell; the median of these is reported
BENCH_BUDGET = 0.1  # seconds of repetitions inside one run
BENCH_MIN_REPS = 5  # never fewer, however slow the call
BENCH_MAX_REPS = 500  # never more, however fast the call


def best_of(module, x, kv_cache):
    """Fastest call to module(x, kv_cache=kv_cache), in seconds, and the rep count."""
    best, reps = float("inf"), 0
    deadline = time.perf_counter() + BENCH_BUDGET
    while reps < BENCH_MIN_REPS or (reps < BENCH_MAX_REPS and time.perf_counter() < deadline):
        t0 = time.perf_counter()
        module(x, kv_cache=kv_cache)
        best = min(best, time.perf_counter() - t0)
        reps += 1
    return best, reps


def bench_step(module, x, kv_cache):
    """Median over BENCH_ROUNDS runs of best_of, plus the total reps behind it."""
    module(x, kv_cache=kv_cache)  # warm up: first call pays allocator and cache misses
    runs = sorted(best_of(module, x, kv_cache) for _ in range(BENCH_ROUNDS))
    return runs[len(runs) // 2][0], sum(reps for _, reps in runs)


print("\n--- One decode step, attention module only, synthetic input ---")
print(f"  median of {BENCH_ROUNDS} runs, each the fastest of as many reps as fit in {BENCH_BUDGET:.2f}s")
print(f"  (floor {BENCH_MIN_REPS}, ceiling {BENCH_MAX_REPS} per run), after one warm-up call")
print(f"  {'T':>6} {'naive':>11} {'cached':>11} {'measured':>10} {'theory':>8} {'reps n/c':>12}")
bench_attn = CausalSelfAttention().eval()
rows = []
with torch.no_grad():
    for T_bench in (64, 256, 1024, 2048):
        x_full = torch.randn(1, T_bench, n_embd)
        x_one = torch.randn(1, 1, n_embd)
        bench_cache = (
            torch.randn(1, n_head, T_bench - 1, head_dim),
            torch.randn(1, n_head, T_bench - 1, head_dim),
        )
        t_step_naive, reps_naive = bench_step(bench_attn, x_full, None)
        t_step_cached, reps_cached = bench_step(bench_attn, x_one, bench_cache)
        rows.append((T_bench, t_step_naive, t_step_cached))
        print(
            f"  {T_bench:>6} {t_step_naive * 1e6:>9.1f}us {t_step_cached * 1e6:>9.1f}us "
            f"{t_step_naive / t_step_cached:>9.1f}x {T_bench:>7}x "
            f"{f'{reps_naive}/{reps_cached}':>11}"
        )
print("""
  The speedup grows with T, because naive attention is T x T against the cache's
  1 x T. It stays well short of the theoretical T because the projections and the
  fixed per-call overhead do not shrink at all, and at T = 64 that overhead still
  swamps everything, which is precisely what the 20-name wall clock above was
  measuring. This is the whole reason the operation counter, not the stopwatch, is
  this lab's headline.""")

# Both columns time work that grows with T, so both must grow with T. Check that
# rather than assert it: if this machine is too busy for the trend to survive,
# say so here instead of leaving the reader to trust an impossible table.
noisy = [name for name, col in (("naive", 1), ("cached", 2)) if not all(b[col] > a[col] for a, b in pairwise(rows))]
if noisy:
    print(f"""
  NOTE: on this run the {" and ".join(noisy)} column did not increase monotonically
  with T, which cannot be true of work that grows with T. Read that as contention
  on this machine, not as a property of the code, since something else had the CPU
  during the fast rows. Re-run on an idle machine, or raise BENCH_ROUNDS and
  BENCH_BUDGET above, before drawing any conclusion from the table.""")
else:
    print("  (Both timing columns increase monotonically with T on this run, as they must.)")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("WHY KV CACHE MATTERS")
print("=" * 60)
print("""
  Cache stores K,V from previous positions. Each new token computes only its
  own Q,K,V and attends to all cached K,V, with no recomputation.

                  Without cache         With cache
  Step 1:         1 x 1  attention      1 x 1  attention
  Step 2:         2 x 2  attention      1 x 2  attention
  Step 3:         3 x 3  attention      1 x 3  attention
  ...
  Step T:         T x T  attention      1 x T  attention
  ─────────────────────────────────────────────────────────
  Total:          ~T^3/3 operations     ~T^2/2 operations

  At 2048 context the ratio (2T+1)/3 is 1366x FEWER ATTENTION OPERATIONS. That
  is not a 1366x speedup and the distinction matters: the projections, the MLP,
  and the memory traffic reading the cache back do not shrink by that factor,
  and past a few thousand tokens decode is memory-bound rather than compute-
  bound. The end-to-end win is large, large enough that no serving system ships
  without it, but it is smaller than the operation count, and it is bounded by
  whatever does not get cheaper.

  A 7B model (32 layers, 32 heads, head_dim 128, fp16) needs 512 KB of cache per
  token, so ~1 GB for a 2048-token request. Prerequisite for speculative
  decoding, PagedAttention, and disaggregated serving.
""")
