"""
microGPT — Tiled attention edition.

Same architecture as the PyTorch version (03), but demonstrating three
attention implementations for inference: standard, online-softmax, and
tiled (FlashAttention algorithm). All three produce identical outputs;
the difference is how many "slow memory" (HBM) operations they need.

Training uses PyTorch; the attention implementations are in explicit NumPy
to make every memory access visible and countable.

The tiled algorithm implements the core idea from "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022),
https://arxiv.org/abs/2205.14135. The online softmax technique used within
tiles is from "Online normalizer calculation for softmax" (Milakov &
Gimelshein, 2018), https://arxiv.org/abs/1805.02867. Also see
"FlashAttention-2: Faster Attention with Better Parallelism and Work
Partitioning" (Dao, 2023), https://arxiv.org/abs/2307.08691. Note that this
is a pedagogical NumPy implementation of the algorithmic idea -- the real
FlashAttention is a fused CUDA kernel that exploits GPU SRAM directly. The
math is identical; the performance difference comes from hardware-level
memory management.
"""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
rng = np.random.default_rng(42)
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)  # Intentional for this standalone lab (float64 needed for numerical accuracy demos)

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt", input_path
    )

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS, vocab_size = len(uchars), len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model (PyTorch — identical architecture to Lab 03)
# ---------------------------------------------------------------------------
n_embd, n_head, n_layer, block_size = 16, 4, 1, 16
head_dim = n_embd // n_head


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
        out = (F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.relu(self.fc1(self.norm2(x))))
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
        x = self.norm_in(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
model = MicroGPT()
print(f"num params: {sum(p.numel() for p in model.parameters())}")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    input_ids = torch.tensor([tokens[:n]])
    targets = torch.tensor([tokens[1 : n + 1]])
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

# ---------------------------------------------------------------------------
# Extract trained weights to NumPy for attention experiments
# ---------------------------------------------------------------------------
# Training used PyTorch for convenience. The attention implementations below
# are in explicit NumPy so every memory access is visible and countable.
sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
P = {"wte": sd["wte.weight"], "wpe": sd["wpe.weight"], "lm_head": sd["lm_head.weight"]}
for i in range(n_layer):
    for w in ("wq", "wk", "wv", "wo"):
        P[f"l{i}.{w}"] = sd[f"layers.{i}.attn.{w}.weight"].T
    for w in ("fc1", "fc2"):
        P[f"l{i}.{w}"] = sd[f"layers.{i}.{w}.weight"].T


def softmax_np(logits):
    exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)


def rmsnorm_np(x):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-5)


# ---------------------------------------------------------------------------
# Three attention implementations — the core of this lab
# ---------------------------------------------------------------------------
print("\n--- comparing attention implementations ---\n")


def inference_forward(token_ids, attn_fn):
    """Forward pass using a pluggable attention function (all NumPy)."""
    n = len(token_ids)
    last_stats = {}
    x = rmsnorm_np(P["wte"][token_ids] + P["wpe"][np.arange(n)])
    for li in range(n_layer):
        x_res = x.copy()
        x_n = rmsnorm_np(x)
        Q_h, K_h, V_h = (
            (x_n @ P[f"l{li}.{w}"]).reshape(n, n_head, head_dim).transpose(1, 0, 2) for w in ("wq", "wk", "wv")
        )
        attn_out, last_stats = attn_fn(Q_h, K_h, V_h, n)
        x = attn_out.transpose(1, 0, 2).reshape(n, n_embd) @ P[f"l{li}.wo"] + x_res
        x_res = x.copy()
        x_n = rmsnorm_np(x)
        h = x_n @ P[f"l{li}.fc1"]
        x = np.maximum(h, 0) @ P[f"l{li}.fc2"] + x_res
    return x @ P["lm_head"].T, last_stats


# --- 1) Standard attention: materialize the full NxN matrix ---


def standard_attention(Q, K, V, seq_len):
    """
    Classic attention: compute the full attention matrix, write it to memory.
    On a real GPU this writes the NxN matrix to HBM (slow main memory).

    Shapes are read off Q (n_head, seq_len, head_dim) rather than the module
    globals, so the same function can be called at the model's tiny dimensions
    and at production dimensions in the scaling sweep further down.
    """
    nh, _, d = Q.shape
    stats = {"hbm_reads": 0, "hbm_writes": 0}
    stats["hbm_reads"] += 2 * nh * seq_len * d
    att = Q @ K.transpose(0, 2, 1) / math.sqrt(d)
    stats["hbm_writes"] += nh * seq_len * seq_len
    causal = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    att = np.where(causal, -1e9, att)
    stats["hbm_reads"] += nh * seq_len * seq_len
    att_probs = softmax_np(att)
    stats["hbm_writes"] += nh * seq_len * seq_len
    stats["hbm_reads"] += nh * seq_len * seq_len + nh * seq_len * d
    out = att_probs @ V
    stats["hbm_writes"] += nh * seq_len * d
    # Largest single intermediate that has to be allocated: the NxN score
    # matrix, for every head. This is the array FlashAttention refuses to build.
    stats["max_intermediate"] = att.nbytes
    return out, stats


# --- 2) Online softmax attention: never store the full NxN matrix ---


def online_softmax_attention(Q, K, V, seq_len):
    """
    Online softmax (Milakov & Gimelshein, 2018): compute attention row by row
    without ever materializing the full NxN attention matrix.

    This is the algorithmic *stepping stone* to FlashAttention, not an optimum.
    It fixes the memory footprint (no NxN array anywhere), but it pays for that
    by re-reading all of K and V once per query row: O(N^2 * d) element reads
    instead of standard attention's O(N^2 + N*d). At the tiny head_dim of this
    model that trade looks free; at head_dim=128 it is dramatically worse than
    standard attention. The scaling sweep below measures exactly that.

    Tiling (next function) is what turns this idea into a win: same running
    softmax, but K/V are read once per *block* of queries instead of once per
    query row.
    """
    nh, _, d = Q.shape
    stats = {"hbm_reads": 0, "hbm_writes": 0}
    out = np.zeros_like(Q)

    for h in range(nh):
        for i in range(seq_len):
            q_i = Q[h, i]
            stats["hbm_reads"] += d

            running_max = -np.inf
            running_sum = 0.0
            running_out = np.zeros(d)

            for j in range(i + 1):  # causal: only attend to positions <= i
                k_j = K[h, j]
                v_j = V[h, j]
                stats["hbm_reads"] += 2 * d

                score = np.dot(q_i, k_j) / math.sqrt(d)
                new_max = max(running_max, score)
                # On the first key running_max is -inf, so this is exp(-inf) = 0
                # and the accumulators start from scratch. No special case needed.
                correction = math.exp(running_max - new_max)
                running_sum = running_sum * correction + math.exp(score - new_max)
                running_out = running_out * correction + math.exp(score - new_max) * v_j
                running_max = new_max

            out[h, i] = running_out / running_sum
            stats["hbm_writes"] += d

    # Largest intermediate: a single head_dim accumulator. Nothing NxN, and
    # nothing that grows with N at all — that is the one thing this buys us.
    stats["max_intermediate"] = d * Q.itemsize
    return out, stats


# --- 3) Tiled attention (FlashAttention algorithm) ---


def tiled_attention(Q, K, V, seq_len, block_size_tile=4):
    """
    FlashAttention: tile the computation into blocks.

    Instead of processing one key at a time (online softmax) or the full
    matrix (standard), process BLOCKS of keys at once. Each block fits in
    fast on-chip memory (SRAM).
    """
    nh, _, d = Q.shape
    stats = {"hbm_reads": 0, "hbm_writes": 0, "max_intermediate": 0}
    Bc = block_size_tile
    Br = block_size_tile
    out = np.zeros_like(Q)

    for h in range(nh):
        for i_start in range(0, seq_len, Br):
            i_end = min(i_start + Br, seq_len)
            Qi = Q[h, i_start:i_end]
            block_rows = i_end - i_start
            stats["hbm_reads"] += block_rows * d

            row_max = np.full(block_rows, -np.inf)
            row_sum = np.zeros(block_rows)
            row_out = np.zeros((block_rows, d))

            max_j = i_end  # causal: only attend up to current position
            for j_start in range(0, max_j, Bc):
                j_end = min(j_start + Bc, max_j)
                Kj = K[h, j_start:j_end]
                Vj = V[h, j_start:j_end]
                block_cols = j_end - j_start
                stats["hbm_reads"] += 2 * block_cols * d

                scores = Qi @ Kj.T / math.sqrt(d)
                # Biggest thing this algorithm ever allocates: one Br x Bc score
                # tile (plus the Br x d output accumulator). Both are chosen to
                # fit in SRAM, and neither depends on N.
                stats["max_intermediate"] = max(stats["max_intermediate"], scores.nbytes + row_out.nbytes)

                # Apply causal mask within this block
                for bi in range(block_rows):
                    for bj in range(block_cols):
                        if (j_start + bj) > (i_start + bi):
                            scores[bi, bj] = -1e9

                # Online softmax update across blocks
                block_max = scores.max(axis=-1)
                new_max = np.maximum(row_max, block_max)
                old_correction = np.exp(row_max - new_max)
                exp_scores = np.exp(scores - new_max[:, None])

                row_sum = row_sum * old_correction + exp_scores.sum(axis=-1)
                row_out = row_out * old_correction[:, None] + exp_scores @ Vj
                row_max = new_max

            out[h, i_start:i_end] = row_out / row_sum[:, None]
            stats["hbm_writes"] += block_rows * d

    return out, stats


# ---------------------------------------------------------------------------
# Compare all three on the same input
# ---------------------------------------------------------------------------
sample_tokens = np.array([BOS] + [uchars.index(ch) for ch in docs[0]])[:block_size]
seq_len = len(sample_tokens)

logits_std, stats_std = inference_forward(sample_tokens, standard_attention)
logits_online, stats_online = inference_forward(sample_tokens, online_softmax_attention)
logits_tiled, stats_tiled = inference_forward(sample_tokens, tiled_attention)

diffs = [
    ("online softmax", np.max(np.abs(logits_std - logits_online)), 1e-10),
    ("tiled", np.max(np.abs(logits_std - logits_tiled)), 1e-6),
]
print(f"sequence length: {seq_len}")
for label, diff, tol in diffs:
    print(f"max diff (standard vs {label + '):':<16s} {diff:.2e}")
    assert diff < tol, f"{label} diverged: {diff}"
print("All three produce identical output (within floating-point tolerance).\n")

# ---------------------------------------------------------------------------
# Memory operation comparison
# ---------------------------------------------------------------------------
print("--- memory operation counts (HBM reads + writes) ---\n")

for name, stats, note in [
    ("Standard attention", stats_std, "writes N*N attention matrix to HBM, reads it back for softmax"),
    ("Online softmax", stats_online, "never stores N*N matrix, but re-reads all of K/V for every query row"),
    (
        "Tiled (FlashAttention, block_size=4)",
        stats_tiled,
        "reads K/V once per block of queries, so the K/V traffic is amortized by Bc",
    ),
]:
    r, w = stats["hbm_reads"], stats["hbm_writes"]
    print(f"{name}:\n  HBM reads: {r:>6d}  writes: {w:>6d}  total: {r + w:>6d}")
    print(f"  largest intermediate array: {stats['max_intermediate']:>6d} bytes")
    print(f"  Key insight: {note}\n")

print(
    f"Careful: this ranking is an artefact of head_dim={head_dim}. Online softmax looks\n"
    f"cheap here only because head_dim is tiny, so re-reading K/V costs almost nothing.\n"
    f"The sweep below shows what happens at a realistic head_dim.\n"
)

# ---------------------------------------------------------------------------
# Scaling sweep — the quadratic-vs-linear story needs more than 7 tokens
# ---------------------------------------------------------------------------
# The comparison above runs on one short name (seq_len=7) at head_dim=4. That is
# far too small to show anything about scaling: at head_dim=4 the N*N score
# matrix is barely larger than the N*d inputs. Sweep N on synthetic Q/K/V at a
# realistic head_dim so the crossover is actually visible.
sweep_d = 64  # a realistic per-head dimension (the model above uses 4)
sweep_tile = 64  # a realistic FlashAttention block size (the demo above uses 4)
print(f"--- scaling sweep: synthetic Q/K/V, n_head=1, head_dim={sweep_d}, tile={sweep_tile} ---\n")


def hbm_total(stats):
    return stats["hbm_reads"] + stats["hbm_writes"]


hdr = f"  {'N':>4}  {'standard':>10}  {'online':>10}  {'tiled':>10}  {'tiled/std':>9}  {'online/std':>10}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for N_sweep in (64, 128, 256, 512):
    Qs, Ks, Vs = (rng.standard_normal((1, N_sweep, sweep_d)) for _ in range(3))
    o_std, s_std = standard_attention(Qs, Ks, Vs, N_sweep)
    o_on, s_on = online_softmax_attention(Qs, Ks, Vs, N_sweep)
    o_ti, s_ti = tiled_attention(Qs, Ks, Vs, N_sweep, block_size_tile=sweep_tile)
    # Same algebra at a real head_dim: all three still agree to float64 precision.
    assert np.max(np.abs(o_std - o_on)) < 1e-12, "online softmax diverged in sweep"
    assert np.max(np.abs(o_std - o_ti)) < 1e-12, "tiled diverged in sweep"
    t_std, t_on, t_ti = hbm_total(s_std), hbm_total(s_on), hbm_total(s_ti)
    print(
        f"  {N_sweep:>4}  {t_std:>10d}  {t_on:>10d}  {t_ti:>10d}"
        f"  {t_ti / t_std:>8.2f}x  {t_on / t_std:>9.2f}x"
    )

# The counters above are proxies for traffic. This is the thing you can actually
# run out of: the largest single array each algorithm has to hold at once.
print(f"\n  largest intermediate array actually allocated, N=512, head_dim={sweep_d}:")
Qs, Ks, Vs = (rng.standard_normal((1, 512, sweep_d)) for _ in range(3))
for label, stats in [
    ("standard: the N x N score matrix", standard_attention(Qs, Ks, Vs, 512)[1]),
    ("online:   one head_dim accumulator", online_softmax_attention(Qs, Ks, Vs, 512)[1]),
    (f"tiled:    one {sweep_tile}x{sweep_tile} tile + output block", tiled_attention(Qs, Ks, Vs, 512, sweep_tile)[1]),
]:
    print(f"    {label:<40s} {stats['max_intermediate']:>9,d} bytes")
print(
    "  Standard grows as N^2. The other two are set by head_dim and tile size and do\n"
    "  not grow with N at all — that is why FlashAttention can run context lengths\n"
    "  that standard attention cannot fit in memory, whatever the traffic counts say.\n"
)

# ---------------------------------------------------------------------------
# Projection to a production shape (same counting formulas, closed form)
# ---------------------------------------------------------------------------
# These are the closed forms of the counters above, not new measurements, so the
# numbers are directly comparable with the sweep.
N, d = 2048, 128
std_ops = 4 * N * N + 4 * N * d
online_ops = 2 * N * d + d * N * (N + 1)


def tiled_ops(n, dim, tile):
    n_blocks = n // tile
    return n_blocks * 2 * tile * dim + dim * tile * n_blocks * (n_blocks + 1)


print(f"--- same counters projected to N={N}, d={d}, one head ---\n")
print(f"  standard          {std_ops:>12,d}")
print(f"  online softmax    {online_ops:>12,d}   {online_ops / std_ops:>6.1f}x MORE than standard")
for tile in (64, 256, 512):
    ops = tiled_ops(N, d, tile)
    print(f"  tiled, Bc={tile:<4d}    {ops:>12,d}   {std_ops / ops:>6.2f}x fewer than standard")
print(
    "\n  Two things to read off this table. First, online softmax is a memory-footprint\n"
    "  fix, not a memory-traffic fix: it is 30x WORSE than standard on total HBM ops at\n"
    "  d=128, because it streams all of K and V once per query row. Second, tiling's\n"
    "  advantage is tile-size dependent — there is no single 'Nx fewer' number.\n"
)

# ---------------------------------------------------------------------------
# GPU memory hierarchy context
# ---------------------------------------------------------------------------
print("""
--- GPU memory hierarchy (why this matters) ---

  Level         Size        Bandwidth       What lives here
  ──────────    ────────    ────────────    ───────────────────────────
  Registers     ~few KB     fastest         Current computation
  SRAM (L1)     ~20 MB      ~19 TB/s       FlashAttention keeps Q,K,V blocks here
  HBM (VRAM)    ~80 GB      ~2 TB/s        Standard attention writes N*N matrix here
  CPU DRAM      ~TBs        ~50 GB/s        Model doesn't fit in GPU? Swap here

Standard attention writes the full N*N attention matrix to HBM — 10x slower
memory. FlashAttention tiles the computation so it stays in SRAM, never
materializing the full matrix. Same math, same forward-pass FLOPs, ~2-4x faster
on real hardware. (The often-quoted "FlashAttention does extra FLOPs" applies to
the BACKWARD pass, which recomputes the score tiles instead of storing them.
The forward pass implemented above does not do any extra arithmetic.)""")

# ---------------------------------------------------------------------------
# Inference — generate names using tiled attention
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (tiled attention) ---")
for sample_idx in range(20):
    generated = [BOS]
    sample = []
    for _ in range(block_size):
        logits, _ = inference_forward(np.array(generated), tiled_attention)
        logits = logits[-1] / temperature
        probs = softmax_np(logits.reshape(1, -1))[0]
        token_id = rng.choice(vocab_size, p=probs)
        if token_id == BOS:
            break
        generated.append(token_id)
        sample.append(uchars[token_id])
    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
