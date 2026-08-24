# Understanding LLMs by Building One: Grouped-Query Attention (GQA/MQA)

This lab shows the progression from Multi-Head Attention (MHA) to Multi-Query Attention (MQA) to Grouped-Query Attention (GQA), and measures what sharing KV heads costs in parameters and saves in KV cache memory.

One thing up front, because it is the easiest claim to overstate: this lab does not demonstrate that GQA preserves quality. It cannot. Three 1-layer, 16-dimension models trained for 1000 single-name steps produce losses separated by less than the run-to-run noise, and `main.py` says so where it prints them. The quality result is real and it comes from Ainslie et al. (2023), who uptrained 64-head checkpoints on real corpora. What you can see here is the mechanism and the memory arithmetic.

## Why GQA exists

During inference, the dominant memory cost is not the model weights or the compute but rather the **KV cache**. Every generated token requires storing key and value vectors for all previous positions, across all layers and all heads. For long sequences and large batch sizes this cache dwarfs everything else.

The insight behind GQA is simple: query heads need to be expressive (they encode "what am I looking for?"), but key/value heads mostly encode "what information is available here?", and that information can be shared across multiple query heads without losing much.

## The three variants

### MHA, Multi-Head Attention (standard)

N query heads, N KV heads. Every query head gets its own dedicated key and value projections. This is the original Transformer design.

- Full expressiveness, but full KV cache cost.

### MQA, Multi-Query Attention

N query heads, **1** KV head. All query heads share a single key and a single value projection. Proposed by Noam Shazeer (2019).

- Aggressive sharing: KV cache shrinks by N times.
- Reported to hurt quality on complex tasks because all queries see identical keys/values. Not something this lab is large enough to reproduce.

### GQA, Grouped-Query Attention

N query heads, **G** KV head groups (1 < G < N). Each group of N/G query heads shares one KV head. Proposed by Ainslie et al. (2023).

- Sweet spot: significantly smaller KV cache than MHA, more capacity than MQA.
- LLaMA 2 70B uses GQA with 8 KV heads for 64 query heads, an 8x reduction in KV cache memory.

## What the lab measures

Parameter counts and cache sizes, both exact:

```
      kv_heads  KV proj   total  cache B  vs MHA  avg loss
  MHA        4      512    4192     1024   1.00x    2.2769
  GQA        2      256    3936      512   0.50x    2.2875
  MQA        1      128    3808      256   0.25x    2.2843
```

The **ratio** column is the part that transfers. Raw byte counts from a 16-dimension model are meaningless on their own, but 1.00x / 0.50x / 0.25x holds at any scale, because the cache is linear in `n_kv_head`. LLaMA 2 70B's 8-of-64 configuration is 0.125x by the same arithmetic.

Two caveats on that cache column. It is an analytic figure, fp16 with batch 1 and a full `block_size` context, rather than an observed one, because nothing in this lab actually caches anything. Building a real KV cache is lab 12. And the `avg loss` column is a trailing average over the last 100 steps rather than a final-step loss, since at batch size 1 a single step's loss depends mostly on which name it landed on. Even averaged, the three numbers are a tie; read them as "all three still learn to spell names".

## The key implementation trick

A single `FlexAttention` module handles all three variants. The only difference is the KV projection size and a `repeat_interleave` call to expand KV heads to match query heads:

```python
class FlexAttention(nn.Module):
    def __init__(self, n_kv_head):
        super().__init__()
        self.n_kv_head = n_kv_head
        self.repeats = n_head // n_kv_head

        self.wq = nn.Linear(n_embd, n_embd, bias=False)                    # full Q heads
        self.wk = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)      # fewer KV heads
        self.wv = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)
            v = v.repeat_interleave(self.repeats, dim=1)

        # ... standard scaled dot-product attention from here
```

When `n_kv_head == n_head`, `repeats == 1` and nothing is repeated, giving you standard MHA. When `n_kv_head == 1`, every KV vector is broadcast to all query heads, which is MQA. Anything in between is GQA. `repeat_interleave` maps KV head `j` onto query heads `j*r` through `(j+1)*r-1`, so the groups are contiguous.

That `repeat_interleave` is the clearest way to write the expansion and the worst way to run it: it materialises `repeats` identical copies of K and V, giving back the memory saving for the duration of the forward pass. Production kernels avoid it: PyTorch's `scaled_dot_product_attention` with `enable_gqa=True`, and FlashAttention's GQA support, index the shared KV head directly from each query head, with the same maths and no copies.

## What you learn here

- Why the KV cache is the memory bottleneck in LLM serving, not model weights or compute
- How `repeat_interleave` enables KV head sharing with zero changes to the attention math
- The parameter count and memory tradeoff between MHA, GQA, and MQA, as a ratio rather than raw bytes
- Why GQA has become the default for large models (LLaMA 2 70B, Mistral, Gemma)
- Where a toy-scale lab can prove something (the mechanism, the arithmetic) and where it can only cite a paper (the quality result)

## Run

```bash
uv run python main.py
```

Trains all three variants (MHA, GQA, MQA) for 1000 steps each on the same data and from the same seed. Compares parameter counts, trailing-average loss, and KV cache size both in bytes and as a ratio to MHA, then generates 10 names per variant.
