# Understanding LLMs by Building One: Grouped-Query Attention (GQA/MQA)

This lab shows the progression from Multi-Head Attention (MHA) to Multi-Query Attention (MQA) to Grouped-Query Attention (GQA), demonstrating how sharing KV heads reduces memory during inference while preserving quality.

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
- Can hurt quality on complex tasks because all queries see identical keys/values.

### GQA, Grouped-Query Attention

N query heads, **G** KV head groups (1 < G < N). Each group of N/G query heads shares one KV head. Proposed by Ainslie et al. (2023).

- Sweet spot: significantly smaller KV cache than MHA, more capacity than MQA.
- LLaMA 2 70B uses GQA with 8 KV heads for 64 query heads, an 8x reduction in KV cache memory.

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

When `n_kv_head == n_head`, `repeats == 1` and nothing is repeated, giving you standard MHA. When `n_kv_head == 1`, every KV vector is broadcast to all query heads, which is MQA. Anything in between is GQA.

## What you learn here

- Why the KV cache is the memory bottleneck in LLM serving, not model weights or compute
- How `repeat_interleave` enables KV head sharing with zero changes to the attention math
- The parameter count and memory tradeoff between MHA, GQA, and MQA
- Why GQA has become the default for large models (LLaMA 2 70B, Mistral, Gemma)

## Run

```bash
uv run python main.py
```

Trains all three variants (MHA, GQA, MQA) for 1000 steps each on the same data. Compares parameter counts, final loss, KV cache memory, and generates 10 names per variant.
