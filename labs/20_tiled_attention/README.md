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

Online softmax is a **stepping stone**, not an optimum, and the lab is explicit about this. It fixes the memory *footprint*: the largest array it allocates is a single `head_dim` accumulator, which does not grow with N at all. It does not fix memory *traffic*, because it streams all of K and V once per query row. The lab's own counters, projected to N=2048 and d=128, put online softmax at 537,657,344 HBM operations against standard attention's 17,825,792, so it is about 30x worse on traffic. Tiling is what turns the running-softmax idea into a win: same algebra, but K and V are read once per block of queries instead of once per query row.

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
| Tiled | O(hN²d / Bc + hNd) | Block reads amortize the K/V traffic by a factor of Bc |

Two caveats the lab now measures rather than asserts.

**The ranking at the model's own scale is an artefact of `head_dim=4`.** On the 7-token demo sequence the counts come out standard 1232, online 1120, tiled 576, which makes online softmax look cheap. It only looks cheap because re-reading K and V costs almost nothing when each vector is 4 numbers wide. At `head_dim=64` online softmax is already 8-14x worse than standard across the swept sequence lengths.

**The tiled advantage depends on the tile size, so there is no single "Nx fewer" number.** Projecting the same counters to N=2048, d=128, one head:

| Implementation | HBM operations | vs standard |
|---|---|---|
| Standard | 17,825,792 | 1.00x |
| Online softmax | 537,657,344 | 30.2x **more** |
| Tiled, Bc=64 | 9,175,040 | 1.94x fewer |
| Tiled, Bc=256 | 2,883,584 | 6.18x fewer |
| Tiled, Bc=512 | 1,835,008 | 9.71x fewer |

The K/V traffic term is roughly `d·N²/Bc`, so tiling only reduces traffic when the tile is large relative to the head dimension. At Bc=256 and d=128 the reduction is 6.18x, not the 8x that the asymptotic term alone suggests, because the per-block Q reads and output writes are also counted.

The lab also prints the largest array each algorithm actually allocates. At N=512 and head_dim=64: standard 2,097,152 bytes (the N×N score matrix), tiled 65,536 bytes (one 64×64 score tile plus the output block), online softmax 512 bytes (one accumulator). Standard grows as N²; the other two are fixed by head dimension and tile size. That, more than the traffic counts, is why FlashAttention can run context lengths standard attention simply cannot fit.

## The GPU memory hierarchy

```
Level         Size        Bandwidth       Role
──────────    ────────    ────────────    ──────────────────────────────
Registers     ~few KB     fastest         Current arithmetic operation
SRAM (L1)     ~20 MB      ~19 TB/s       FlashAttention block workspace
HBM (VRAM)    ~80 GB      ~2 TB/s        Where tensors live (10x slower)
CPU DRAM      ~TBs        ~50 GB/s        Overflow / CPU offloading
```

The insight: SRAM is ~10× faster than HBM, but ~4000× smaller. Algorithms that keep their working set in SRAM win, sometimes even at the cost of extra computation.

One clarification, since it is widely misquoted: the extra-FLOPs tradeoff is a property of the **backward** pass, which recomputes score tiles instead of storing them for the gradient. The forward pass this lab implements does exactly the same arithmetic as standard attention. It is the same number of multiply-adds, just arranged so the intermediates never leave fast memory.

## What you learn here

- The memory wall: why inference speed is limited by memory bandwidth, not compute
- How tiling restructures the same arithmetic to cut memory accesses (IO-awareness)
- The online softmax algorithm: numerically stable incremental computation
- Why online softmax alone is not enough, and what tiling adds on top of it
- That "fewer HBM operations" is a function of tile size and head dimension, not a constant
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

Trains for 1000 steps, then runs all three attention implementations on the same input and verifies they produce identical outputs (max difference ~9e-16 against the standard reference). It then sweeps sequence length over 64/128/256/512 on synthetic Q/K/V at `head_dim=64` with a 64-wide tile, re-verifying the algebra at each size, and reports HBM operation counts, the ratios against standard attention, and the largest intermediate array each algorithm allocates. Finally it projects the same counters to N=2048, d=128 for three tile sizes, and generates 20 sample names using tiled attention.

## Why the memory wall matters

Every major inference optimization, including FlashAttention, speculative decoding, KV cache paging, and continuous batching, exists because of the memory wall. The GPU can compute far faster than it can move data. Understanding this one constraint explains why these algorithms exist, why they work, and why they're essential for serving LLMs at scale.
