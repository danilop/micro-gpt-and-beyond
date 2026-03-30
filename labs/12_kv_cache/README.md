# Understanding LLMs by Building One: KV Cache

Same model as lab 03, but with the single most important inference optimization in all of LLM serving: caching Key and Value tensors so each new token only computes attention for ONE position instead of reprocessing the entire sequence.

## Why KV cache exists

During autoregressive generation, a transformer produces tokens one at a time. At step `t`, the standard approach feeds all `t` tokens through the model to get the next one. But the Keys and Values for positions `1..t-1` are identical to what was computed at step `t-1`, so we are recomputing them for nothing.

KV cache eliminates this redundancy: store the K,V tensors, and at each new step only compute the new position's Q, K, V. Attention is then computed between the single new query and ALL cached keys/values.

## Prefill vs decode

Generation has two phases:

1. **Prefill**: process the initial prompt (or just `[BOS]`, Beginning of Sequence). Compute Q, K, V for all positions. Store K, V in the cache. This is compute-bound, essentially a large matrix multiply.

2. **Decode**: generate tokens one at a time. Each step computes Q, K, V for ONE new position, appends K, V to the cache, and computes attention of the new Q against the full cached K, V. This is memory-bound, dominated by reading the cache from memory.

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

The reduction factor is `(2T+1)/3`. For T=2048, that is roughly **1365x fewer attention operations**.

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
- Quantifying the operation count reduction from O(T^3) to O(T^2)

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, then generates 20 names both with and without KV cache, verifying identical outputs and comparing operation counts.
