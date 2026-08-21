# Understanding LLMs by Building One: Rotary Position Embeddings (RoPE)

Same architecture as version 03 (PyTorch), but learned positional embeddings are replaced with Rotary Position Embeddings. RoPE encodes position by rotating query and key vectors in complex space, so the attention dot product naturally captures *relative* position without any learned parameters.

## Why this version exists

Learned positional embeddings (like `wpe` in version 03) have two problems:

1. **They have a hard ceiling.** A model trained with `block_size=16` has 16 rows in its position table, so position 16 does not exist. Feed it a longer sequence and it raises `IndexError` — the lab does exactly that and prints the exception.
2. **They lose relative information.** The model has to *learn* that position 5 and position 7 are two apart. This relationship is not built into the representation, so the model must discover it from data.

RoPE removes the ceiling outright and builds relative position into the representation. Every modern large language model (LLaMA, Mistral, GPT-NeoX, Gemma) uses it.

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

Note that `max_len` here is `4 * block_size`, not `block_size`. There is no table to size, only a formula, so precomputing positions the model never trains on costs nothing — and it is what makes the length test below possible.

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

Unlike `wpe`, RoPE adds zero trainable parameters. The rotation frequencies are fixed by the formula. Measured: 4192 parameters for the baseline against 3936 for RoPE, a difference of exactly the 16x16 `wpe` table.

### Length generalization, split into the two claims it actually contains

"RoPE generalizes to longer sequences" is usually stated as one fact. It is two, and they are not equally true. The lab evaluates 64 BOS-separated concatenations of names, 48 tokens each, three times the trained `block_size`.

**Claim one: it runs.** True, and demonstrated rather than asserted. The baseline raises `IndexError` on the 48-token input because `wpe` has 16 rows. The RoPE model, same weights it trained with, processes all 48 positions and reports a loss.

**Claim two: it stays accurate.** This lab cannot settle it, and says so. Comparing chunks at their true positions against the same chunks re-based to positions 0–15 controls for content, and the average gap comes out at -0.0004 nats. That is not evidence that RoPE extrapolates, for two reasons the lab prints rather than glosses.

First, the re-based pass loses more than position: it also throws away everything before the chunk, so the gap mixes the position effect with a context effect. The lab measures the context term separately, with position held fixed — predict the last 16 targets twice, once from the real history and once with everything older than 16 tokens swapped for a neighbouring sequence's, same absolute positions and same targets either way. That prices all far context at +0.0213 nats. The per-chunk gaps are ±0.03, the same order — and the confound has a sign: losing context can only make the re-based column worse, which flatters the true-position column. So the gaps understate the position cost by roughly their own size. Correct for it by hand and the average moves from -0.0004 to around +0.02 nats, which is under 1% of a 2.75-nat loss.

Second, +0.0213 nats against a loss of 2.7496 is the real point: a 1-layer, 16-dimension model on a corpus of independent names has almost no long-range structure to lose, so a test at this scale has nothing at stake and cannot fail. The lab's verdict threshold is that measured confound rather than a hand-picked epsilon — an effect smaller than the noise it is measured against has not been observed.

(The tempting shortcut is to justify "nothing at stake" with the 0.0002-nat gap the lab prints between the full pass and the windowed pass. That gap is not a measure of context: it changes context and position together, which is exactly the confusion being untangled. Hence the separate probe.)

At real scale fixed-base RoPE does degrade past its trained span, which is why length extension is its own research area: YaRN rescales the base frequency, and NTK-aware interpolation or fine-tuning at the target length are the other standard answers.

The per-chunk table is also a small lesson in controls. Raw loss rises monotonically down the chunks (2.6428, 2.7545, 2.8516), which looks like extrapolation damage and is not — chunk 0 starts on a name boundary and the others start mid-name. Only the difference against the re-based column means anything, and chunk 0 must read exactly +0.0000 for the method to be trustworthy.

### The comparison is set up so the two models really do start equal

Re-seeding before each constructor looks sufficient and is not. `apply(_init_weights)` walks modules in registration order, and the baseline registers `wpe` second, so it draws 256 random numbers the RoPE model never draws; from there the two streams are offset and every remaining weight differs. The lab therefore builds both models and then copies across all 8 shared weight tensors, leaving exactly one difference between them: how position enters.

Even so, the training-loss gap (2.2769 vs 2.2981) is a tie at this scale, and the lab says so rather than reading a winner into it. Both figures are trailing averages over the last 100 steps, because a single last-step loss at batch size 1 is noise.

## What you learn here

- Why every modern LLM (LLaMA, Mistral, GPT-NeoX, Gemma) uses RoPE instead of learned positional embeddings
- How rotation in 2D encodes position without any learned parameters
- Why the dot product of rotated vectors naturally captures relative position
- What "generalizes to longer sequences" does and does not buy you, measured both ways
- Why a fair A/B between two model variants needs shared weights and a trailing average, not a re-seed and a final loss
- The practical implementation: ~15 lines of code replace an entire embedding table

## Run

```bash
uv run python main.py
```

Trains both the baseline (learned positions) and the RoPE variant for 1000 steps each from identical shared weights, compares trailing-average losses, evaluates both on sequences 3x longer than the trained context, and generates names from each.
