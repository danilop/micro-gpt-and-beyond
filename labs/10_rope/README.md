# Understanding LLMs by Building One: Rotary Position Embeddings (RoPE)

Same architecture as version 03 (PyTorch), but learned positional embeddings are replaced with Rotary Position Embeddings. RoPE encodes position by rotating query and key vectors in complex space, so the attention dot product naturally captures *relative* position without any learned parameters.

## Why this version exists

Learned positional embeddings (like `wpe` in version 03) have two problems:

1. **They don't generalize.** A model trained with `block_size=16` has no embedding for position 17. At inference time, you cannot process longer sequences than you trained on.
2. **They lose relative information.** The model has to *learn* that position 5 and position 7 are two apart. This relationship is not built into the representation, so the model must discover it from data.

RoPE solves both problems by encoding position through rotation rather than addition. Every modern large language model (LLaMA, Mistral, GPT-NeoX, Gemma) uses RoPE.

## What makes it interesting

### Rotation in complex space

The core idea: treat each pair of dimensions in a query or key vector as a complex number, then rotate it by an angle proportional to its position. Position `m` rotates by angle `m * theta`, where `theta` varies across dimension pairs (low-frequency for early pairs, high-frequency for later ones).

The rotation frequencies follow a geometric schedule:

```python
def precompute_freqs(dim, max_len):
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    theta = 1.0 / (10000.0 ** (i / dim))       # different freq per pair
    positions = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(positions, theta)       # (max_len, dim//2)
    return torch.cos(angles), torch.sin(angles)
```

### Applying the rotation

Each pair `[x1, x2]` is rotated by angle `theta`:

```python
def apply_rope(x, cos_freqs, sin_freqs):
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t
    return torch.stack((out1, out2), dim=-1).flatten(-2)
```

This is just the 2D rotation matrix applied to each pair independently. RoPE is applied to Q and K only, not V, because values carry content, not position.

### Why relative position emerges

The key insight is mathematical. When you compute `q_m . k_n` (the attention score between positions `m` and `n`), the rotation angles combine as:

```
q_m . k_n = Re[(q * e^{i*m*theta}) . conj(k * e^{i*n*theta})]
           = Re[(q . conj(k)) * e^{i*(m-n)*theta}]
```

The dot product depends only on `(m-n)`, the relative distance, not on the absolute positions `m` and `n` separately. This is exactly what we want: the attention between "the" and "cat" should be the same whether they appear at positions (2,4) or (100,102).

### No learned parameters

Unlike `wpe`, RoPE adds zero trainable parameters. The rotation frequencies are fixed by the formula. This means:

- Fewer parameters to train
- No risk of overfitting positional patterns
- Position information is injected via geometry, not learning

## What you learn here

- Why every modern LLM (LLaMA, Mistral, GPT-NeoX, Gemma) uses RoPE instead of learned positional embeddings
- How rotation in 2D encodes position without any learned parameters
- Why the dot product of rotated vectors naturally captures relative position
- The practical implementation: ~15 lines of code replace an entire embedding table

## Run

```bash
uv run python main.py
```

Trains both the baseline (learned positions) and the RoPE variant for 1000 steps each, then compares final losses and generated names.
