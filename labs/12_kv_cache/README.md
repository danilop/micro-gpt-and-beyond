# Understanding LLMs by Building One: KV Cache

Same model as lab 03, but with the single most important inference optimization in all of LLM serving: caching Key and Value tensors so each new token only computes attention for ONE position instead of reprocessing the entire sequence.

## Why KV cache exists

During autoregressive generation, a transformer produces tokens one at a time. At step `t`, the standard approach feeds all `t` tokens through the model to get the next one. But the Keys and Values for positions `1..t-1` are identical to what was computed at step `t-1`, so we are recomputing them for nothing.

KV cache eliminates this redundancy: store the K,V tensors, and at each new step only compute the new position's Q, K, V. Attention is then computed between the single new query and ALL cached keys/values.

## Prefill vs decode

Generation has two phases:

1. **Prefill**: process the initial prompt (or just `[BOS]`, Beginning of Sequence). Compute Q, K, V for all positions. Store K, V in the cache. This is compute-bound, essentially a large matrix multiply.

2. **Decode**: generate tokens one at a time. Each step computes Q, K, V for ONE new position, appends K, V to the cache, and computes attention of the new Q against the full cached K, V. This is memory-bound, dominated by reading the cache from memory.

## Which cost, exactly

Two different quantities get called "the complexity of decoding" and they have different exponents. Over a T-token generation:

| | naive | cached |
|---|---|---|
| attention scores | O(T³) | O(T²) |
| projections + MLP | O(T²) | O(T) |

Every "cubic to quadratic" claim in this lab, including the operation counter `main.py` prints, is about the **attention** term. That is the one that dominates at long context and the only one the cache improves asymptotically. When you see O(T³) → O(T²) here, it means attention scores.

## The computation reduction

Without cache, total attention operations across T generation steps:

```
Step 1: 1 x 1  = 1
Step 2: 2 x 2  = 4
Step 3: 3 x 3  = 9
...
Step T: T x T  = T^2
Total: sum(t^2 for t=1..T) = T(T+1)(2T+1)/6 ≈ T^3/3
```

With cache:

```
Step 1: 1 x 1  = 1
Step 2: 1 x 2  = 2
Step 3: 1 x 3  = 3
...
Step T: 1 x T  = T
Total: sum(t for t=1..T) = T(T+1)/2 ≈ T^2/2
```

The reduction factor is `(2T+1)/3`. For T=2048 that is roughly **1366x fewer attention operations** — which is not a 1366x speedup, and the distinction is the one this lab most needs to keep straight. The projections, the MLP, and the memory traffic reading the cache back do not shrink by that factor, and past a few thousand tokens decode is memory-bound rather than compute-bound anyway. The end-to-end win is large enough that no serving system ships without it, and it is bounded by whatever does not get cheaper.

### What the lab measures, and what it cannot

The **operation counter is exact** — it counts work actually performed — and it is the headline:

```
  Naive (full recompute):  6,716 ops
  KV cache (incremental):  1,572 ops
  Reduction:               4.3x fewer attention operations
```

That 4.3x does not match the 11.0x the closed form gives for T=16, and the reason is worth stating rather than leaving as a loose end: these names average 5.8 decode steps, because generation stops on BOS long before position 16. Evaluate `(2T+1)/3` at T=5.8 and you get 4.2x. The saving grows with sequence length, and a corpus of six-letter names sits at the small end of it.

**End-to-end wall clock, by contrast, measures Python.** One layer, 16 dimensions, six-token names: each forward pass is a few dozen microseconds of tensor arithmetic wrapped in a few hundred microseconds of interpreter and dispatch overhead. Both paths make the same *number* of forward calls, so both pay that overhead the same number of times, and the arithmetic the cache eliminates never rises above the noise. The printed ratio moves between runs and has been observed below 1.00x on a loaded machine. `main.py` prints it labelled as a ratio rather than a speedup, and says outright that it is a statement about the benchmark.

**So the lab fixes the benchmark instead of explaining away the number.** It times one decode step of the attention module alone, on synthetic activations, at context lengths a serving system would actually reach. Getting a number worth printing takes some care, and `main.py` spells the recipe out: warm up once, then inside a timing run take the *minimum* over repetitions (anything else sharing the CPU can only make a call slower, so the minimum measures the code and the mean measures the machine), and across five such runs take the *median*, so one throttled run cannot set the reported figure. The repetition count is not fixed — each run repeats until a 0.1s budget is spent, floor 5 and ceiling 500 — because one count cannot serve both a cached call of a few microseconds and a naive call at T=2048 that takes a large fraction of a second. The rep counts are printed alongside the times, so a row measured too few times is visible rather than implied:

```
       T       naive      cached   measured   theory     reps n/c
      64     167.9us      96.9us       1.7x      64x   2301/2500
     256     631.5us     107.6us       5.9x     256x    666/2500
    1024   19729.8us     141.6us     139.4x    1024x     26/2258
    2048   95327.6us     224.0us     425.5x    2048x     25/1664
```

(One run on a 2-core Linux box. Absolute microseconds depend on the machine; the shape of the last two columns does not.)

Both timed columns measure work that grows with T, so both must grow with T — and `main.py` checks that rather than asserting it, printing a warning if the machine was too busy for the trend to survive. An earlier version of this benchmark used too few repetitions and reported a *cached* time that fell as T rose, which is impossible; the fix was to make the sample count adequate and visible, not to reword the paragraph underneath it.

The speedup grows with T because naive attention is T x T against the cache's 1 x T. It stays well short of the theoretical T because the projections and the fixed per-call overhead do not shrink at all — and at T=64 that overhead still swamps everything, which is exactly what the 20-name wall clock was measuring.

## The core change

The entire optimization lives in the attention module, about 20 lines of code:

```python
class CausalSelfAttention(nn.Module):
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
        # Q has T_new rows, K has T_total columns
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # ... mask and softmax ...
        return self.wo(out), new_cache
```

During prefill, `kv_cache=None` and it behaves like standard attention. During decode, `T_new=1` and the cached K,V are prepended, so the single new query attends to the full history.

## At scale

For a 7B-parameter model (32 layers, 32 heads, head_dim=128, float16):

```
KV cache per token = 2 * 32 layers * 32 heads * 128 dim * 2 bytes = 524 KB
For 2048 context:   2048 * 524 KB ≈ 1 GB per request
```

This is why GPU memory, not compute, is the bottleneck for LLM serving. Systems like PagedAttention (lab 21) exist specifically to manage this memory efficiently.

## What builds on KV cache

- **Speculative decoding** (lab 19): a small draft model fills its own KV cache cheaply, then the large model verifies multiple tokens in one forward pass using its cache
- **PagedAttention** (lab 21): virtual memory management for KV cache blocks, enabling efficient batching of requests with different sequence lengths
- **Disaggregated serving** (lab 22): separate prefill and decode onto different workers, since they have different hardware profiles
- **Continuous batching**: new requests join a running batch by allocating fresh cache space
- **Quantized KV cache**: store cache in int8/int4 to fit more requests in memory

## What you learn here

- Why autoregressive generation without caching wastes most of its compute
- How to thread a KV cache through the attention and block modules
- The difference between prefill (compute-bound) and decode (memory-bound)
- How to verify correctness: cached and uncached paths produce identical outputs
- Quantifying the *attention* operation count reduction from O(T³) to O(T²), and why that is not the same as a speedup of the same size
- Why a benchmark too small to show an effect produces a number that looks like a refutation, and how to build one that does show it

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, then generates 20 names both with and without KV cache, verifying the outputs are identical, comparing exact operation counts, and benchmarking a single decode step at context lengths from 64 to 2048.
