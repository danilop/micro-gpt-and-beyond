# Understanding LLMs by Building One: Tiled Attention (FlashAttention)

Same architecture as the PyTorch version (03), but with three attention implementations for inference: standard, online softmax, and tiled (the FlashAttention algorithm). All three produce identical outputs, and the difference is how many trips to slow memory they need. This shows why FlashAttention is the single most impactful optimization in modern LLM inference.

## Why this version exists

Attention is the bottleneck. Not because the math is hard (it's just matrix multiplications), but because the standard implementation writes a huge intermediate matrix to slow GPU memory. FlashAttention fixes this by tiling the computation so it stays in fast on-chip memory, using an elegant incremental softmax algorithm that this lab implements from scratch.

## What makes it interesting

### The memory wall

Modern GPUs can do trillions of math operations per second, but they can only move data at ~2 TB/s from their main memory (HBM). Attention is **memory-bound**: it spends more time moving data than computing. The key metric isn't FLOPs but bytes transferred.

Standard attention computes the full N×N attention matrix and writes it to HBM:

```
Q, K → S = Q @ K^T     (write N×N to HBM)
S → P = softmax(S)     (read N×N, write N×N to HBM)
P, V → O = P @ V       (read N×N from HBM)
```

For a 2048-token sequence with 32 heads, that's 32 × 2048² = 134M elements written to slow memory, all for the intermediate attention matrix that gets used once and discarded.

### Online softmax (Milakov & Gimelshein, 2018)

Standard softmax needs two passes over the data: one to find the max (for numerical stability), one to compute exp and normalize. Online softmax does it in **one pass** by maintaining a running max and correcting previous values:

```python
running_max = -inf
running_sum = 0
running_out = zeros(d)

for j in range(seq_len):
    score = dot(q, k[j]) / sqrt(d)
    new_max = max(running_max, score)
    # Rescale everything accumulated so far
    correction = exp(running_max - new_max)
    running_sum = running_sum * correction + exp(score - new_max)
    running_out = running_out * correction + exp(score - new_max) * v[j]
    running_max = new_max

output = running_out / running_sum
```

The correction factor `exp(old_max - new_max)` rescales all previously accumulated values when a new maximum is discovered. This is mathematically identical to standard softmax but never needs the full score vector in memory.

### Tiled attention (FlashAttention)

Online softmax processes one key at a time. FlashAttention processes **blocks** of keys at once, getting the memory benefits of online softmax with the compute efficiency of matrix multiplication:

```
For each block of Q rows (Br rows):
    For each block of K/V columns (Bc columns):
        Load Qi, Kj, Vj blocks from HBM → SRAM
        Compute block scores: Sij = Qi @ Kj^T
        Update running softmax statistics
        Accumulate output: Oi += softmax(Sij) @ Vj
    Write final Oi block from SRAM → HBM
```

The full N×N attention matrix never exists in memory. Each block fits in fast SRAM (~20 MB, ~19 TB/s), and the algorithm only reads/writes the input and output matrices from HBM.

### Memory operation counts

The lab counts "HBM operations" for each implementation and reports them. For a sequence of length N with head dimension d and h heads:

| Implementation | HBM reads + writes | Key insight |
|---|---|---|
| Standard | O(hN² + hNd) | Writes the N×N matrix to slow memory |
| Online softmax | O(hN²d) | No N×N matrix, but reads K/V per query position |
| Tiled | O(hNd × N/Bc) | Block reads amortize HBM access by factor Bc |

At scale (N=2048, d=128, Bc=256), tiled attention does ~8× fewer HBM operations than standard, and that directly translates to ~2-4× wall-clock speedup on real GPUs.

## The GPU memory hierarchy

```
Level         Size        Bandwidth       Role
──────────    ────────    ────────────    ──────────────────────────────
Registers     ~few KB     fastest         Current arithmetic operation
SRAM (L1)     ~20 MB      ~19 TB/s       FlashAttention block workspace
HBM (VRAM)    ~80 GB      ~2 TB/s        Where tensors live (10x slower)
CPU DRAM      ~TBs        ~50 GB/s        Overflow / CPU offloading
```

The insight: SRAM is ~10× faster than HBM, but ~4000× smaller. Algorithms that keep their working set in SRAM, even at the cost of extra computation, win. FlashAttention does ~2× more FLOPs than standard attention (recomputing rather than storing intermediate results), but runs 2-4× faster because it avoids slow memory.

## What you learn here

- The memory wall: why inference speed is limited by memory bandwidth, not compute
- How tiling trades extra computation for fewer memory accesses (IO-awareness)
- The online softmax algorithm: numerically stable incremental computation
- Why FlashAttention is the most impactful optimization in modern LLM inference
- The GPU memory hierarchy (HBM vs. SRAM) and why algorithm design must account for it

## What's not covered (but exists in practice)

- **FlashAttention-2** (Dao, 2023): Better work partitioning across GPU thread blocks, reducing shared memory reads/writes within each block. ~2× faster than FlashAttention-1.
- **FlashAttention-3** (Dao et al., 2024): Exploits asynchronous execution on Hopper GPUs (H100), overlapping computation with data movement. Approaches theoretical peak throughput.
- **The roofline model**: A framework for analyzing whether an operation is memory-bound or compute-bound. Attention is memory-bound during decoding (low arithmetic intensity), compute-bound during prefill (high arithmetic intensity).
- **Kernel fusion**: Combining multiple operations (attention + softmax + masking) into a single GPU kernel to avoid intermediate HBM writes. FlashAttention is a fused kernel.
- **Triton**: A Python-based GPU programming language that makes writing custom kernels accessible. FlashAttention-2's reference implementation is in Triton.
- **Hardware evolution**: HBM4 (~8 TB/s), Groq LPU (SRAM-only, no HBM bottleneck), Apple Silicon (unified memory, where CPU and GPU share the same physical RAM).
- **Key papers**: Dao et al. "FlashAttention" (NeurIPS 2022), Dao "FlashAttention-2" (2023), Milakov & Gimelshein "Online normalizer calculation for softmax" (2018).

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, then runs all three attention implementations on the same input, verifies they produce identical outputs, reports memory operation counts, and generates 20 sample names using tiled attention.

## Why the memory wall matters

Every major inference optimization, including FlashAttention, speculative decoding, KV cache paging, and continuous batching, exists because of the memory wall. The GPU can compute far faster than it can move data. Understanding this one constraint explains why these algorithms exist, why they work, and why they're essential for serving LLMs at scale.
