# Understanding LLMs by Building One: Text Diffusion

> *Instead of generating names left-to-right, names emerge from pure noise, all [MASK] tokens, through iterative unmasking.*

A **masked diffusion language model** in PyTorch. Same transformer building blocks as the PyTorch labs (03, 04), but a fundamentally different generative paradigm.

Where the original GPT predicts the next token given all previous tokens (autoregressive, left-to-right), this model predicts *all masked tokens simultaneously* given the unmasked context (diffusion, all-at-once). The name materializes from noise like a photograph developing in a darkroom.

---


## What Does It Do?

It trains a tiny bidirectional transformer on the same ~32,000 names dataset as lab 01, then generates new names by starting from pure noise and iteratively unmasking:

```
sample  1: ayay
sample  2: tai
sample  3: camiya
sample  4: lilan
sample  5: jalya
...
```

The names aren't memorized. They emerge from a diffusion process that has learned the statistical patterns of how characters combine. Unlike the autoregressive GPT, which writes names left-to-right, this model fills in all positions at once, refining its guesses over a handful of denoising steps. The lab runs the same trained model at 16, 8 and 4 steps so you can see what the step count buys you.

## The Algorithm, Step by Step

Let's walk through `main.py`. The model architecture (RMSNorm, multi-head attention, MLP, residual connections) is the same as labs 03/04. Here we focus on what's different.

### 1. The Tokenizer

```python
uchars = sorted(set("".join(docs)))
MASK = len(uchars)  # the "noise" state
PAD = len(uchars) + 1  # fills unused positions
vocab_size = len(uchars) + 2
```

Lab 03 uses 26 characters + BOS = 27 tokens. Here, BOS is replaced by two special tokens: `MASK` (the noise that the model learns to denoise) and `PAD` (fills positions beyond the name's length). The vocabulary is 28 tokens.

### 2. The Model: Bidirectional, Not Causal

```python
class BidirectionalSelfAttention(nn.Module):
    def forward(self, x):
        # ... Q, K, V projections ...
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = F.softmax(att, dim=-1)  # no causal mask!
```

Lab 03's `CausalSelfAttention` masks future positions so each token only sees the past. `BidirectionalSelfAttention` has no mask, so every position attends to every other. This lets the model use context from both sides to fill in the blanks.

The architecture is otherwise identical: same embedding, same RMSNorm, same multi-head attention, same MLP with ReLU. Key hyperparameter differences:

```python
n_layer = 2  # diffusion needs depth to gather scattered clues
num_steps = 3000
batch_size = 32  # critical for diffusion
```

### 3. Training: Noise Instead of Next-Token Prediction

Training is where diffusion diverges most from the autoregressive labs. Lab 03 wraps each name with BOS and predicts each next character. Here, we corrupt the name with random masks and predict what's underneath:

```python
t = math.exp(random.uniform(math.log(0.2), 0))  # log-uniform noise level
noisy = [MASK if random.random() < t else c for c in clean]
```

The noise level `t` is sampled log-uniformly rather than uniformly. The log-uniform draw is importance sampling that cancels the `1/t` weight in the ELBO, which removes the gradient spikes.

The loss is computed only on masked positions, and the MASK logit is suppressed so no probability mass is wasted on an impossible prediction:

```python
logits[:, :, MASK] = logits[:, :, MASK] - 1e6  # never predict MASK
loss = F.cross_entropy(logits[mask], targets[mask])
```

The MASK zeroing follows [MDLM](https://github.com/kuleshov-group/mdlm)'s `_subs_parameterization`.

### 4. Why Batching Is Critical

Unlike the autoregressive labs (03, 04) which work fine with batch_size=1, diffusion **needs batching**. With a single sample, each gradient is based on one random masking pattern, far too noisy for the model to learn. Batching averages over 32 different mask patterns per step, giving stable gradients. This was the single biggest factor in getting the model to produce plausible names.

### 5. Inference: Names Emerge From Noise

Lab 03 generates left-to-right: start with BOS, sample next token, feed it back, repeat until BOS again. Here, generation is a denoising process:

```python
seq = [MASK] * block_size  # start from pure noise

for step_i in range(num_denoise_steps, 0, -1):
    t = math.cos(math.pi / 2 * (1 - step_i / num_denoise_steps))  # cosine schedule
    s = math.cos(math.pi / 2 * (1 - (step_i - 1) / num_denoise_steps))
    temperature = 0.3 + 0.5 * t  # anneal: explore early, commit late

    logits = model(input_ids)[0]
    # ... predict tokens, track confidence ...

    # Keep the number of positions the schedule wants masked; commit the rest
    if confidences:
        n_to_remask = min(int(block_size * s), len(confidences) - 1)
        confidences.sort()  # lowest confidence first
        for _, i in confidences[:n_to_remask]:
            predicted[i] = MASK
```

Three techniques improve generation quality:

- **Confidence-based remasking**: instead of randomly re-corrupting predictions, the model keeps the tokens it's most sure about and reconsiders the rest. This is the same `low_confidence` strategy used in [LLaDA's inference code](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) and originally introduced by [MaskGIT](https://arxiv.org/abs/2202.04200).
- **Temperature annealing**: high temperature early on encourages exploration when everything is uncertain, low temperature at the end sharpens the final choices. The formula is `0.3 + 0.5 * t` with `t` the cosine schedule's mask fraction, so it starts at exactly 0.8 and the `0.3` is an asymptote it never reaches: the last step's temperature is 0.349 at 16 steps, 0.398 at 8 and 0.491 at 4. The fewer steps you take, the less the anneal has time to cool.
- **Cosine schedule**: instead of linearly decreasing the noise level, a cosine curve spends more of its steps in the middle range where the name's structure is being decided. Note that the target is `block_size * s`, a fraction of the *whole sequence* rather than of whatever is masked right now. Anchoring it on the sequence is what lets the schedule set the pace; anchor it on the current mask count and it collapses to one token per step regardless of the curve.

  There is a second constraint next to the schedule: `len(confidences) - 1`, a floor that forces every step to commit at least one position so the loop cannot stall. Which of the two binds is a property of the step count, and at the top of the sweep it is the floor. With 16 steps for 16 positions there is nothing else it could be, since 16 positions over 16 steps is one per step by arithmetic, so the 16-step arm is *not* a demonstration of the schedule driving anything, and the parallel-decoding saving there is zero. The lab prints the per-step commit counts for each arm, and says so explicitly when every step committed exactly one:

  ```
  --- inference: 16 denoising steps ---
    16.0 forward passes per name (left-to-right needs up to 16)
    positions committed per step: 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
    Every step committed exactly one position: ...
  --- inference: 8 denoising steps ---
    positions committed per step: 1,1,1,2,3,2,3,3
  --- inference: 4 denoising steps ---
    positions committed per step: 2,3,5,6
  ```

  The 8- and 4-step arms are where the schedule actually sets the pace.

### Watching the names denoise

Each arm reports where denoising ends up. It also prints how one name got there:
the whole 16-position sequence after every step. The name shown is the longest of
the ten, since a four-letter name spends most of its rows placing `[PAD]` and has
little else to show.

```
legend for the tables below: `_` = still masked (noise), `.` = [PAD], letters = committed

--- inference: 16 denoising steps ---
  positions committed per step: 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

  sample 2, step by step:
  step  masked  commit  temp   sequence
     0      16       -     -   |________________|  pure noise
     1      15       1  0.80   |_______________.|
     2      14       1  0.80   |______________..|
     3      13       1  0.79   |_____________...|
     4      12       1  0.78   |____________....|
     5      11       1  0.76   |___________.....|
     6      10       1  0.74   |__________......|
     7       9       1  0.72   |_________.......|
     8       8       1  0.69   |________........|
     9       7       1  0.65   |_______.........|
    10       6       1  0.62   |______..........|
    11       5       1  0.58   |_____...........|
    12       4       1  0.54   |____a...........|
    13       3       1  0.49   |_a__a...........|
    14       2       1  0.45   |_a_ya...........|
    15       1       1  0.40   |_arya...........|
    16       0       1  0.35   |sarya...........|
  result: 'sarya'

--- inference: 8 denoising steps ---
  positions committed per step: 1,1,1,2,3,2,3,3

  sample 4, step by step:
  step  masked  commit  temp   sequence
     0      16       -     -   |________________|  pure noise
     1      15       1  0.80   |_______________.|
     2      14       1  0.79   |______________..|
     3      13       1  0.76   |_____________...|
     4      11       2  0.72   |___________.....|
     5       8       3  0.65   |________........|
     6       6       2  0.58   |______n.........|
     7       3       3  0.49   |_e_ll_n.........|
     8       0       3  0.40   |jeallyn.........|
  result: 'jeallyn'

--- inference: 4 denoising steps ---
  positions committed per step: 2,3,5,6

  sample 7, step by step:
  step  masked  commit  temp   sequence
     0      16       -     -   |________________|  pure noise
     1      14       2  0.80   |______________..|
     2      11       3  0.76   |___________.....|
     3       6       5  0.65   |______e.........|
     4       0       6  0.49   |hlraine.........|
  result: 'hlraine'
```

A position mid-denoise holds no character, so it needs a glyph exactly one column
wide or the rows stop lining up. `_` is a slot still waiting to be filled, and it
sits low enough that the letters appearing around it are what your eye lands on;
an asterisk would be the wrong way round, louder than the name it is revealing.
`[PAD]` gets a dot. The finished name simply ends there, so a blank would be the
honest glyph for the result, but mid-run a committed `[PAD]` is a decision, and the
model's most confident one. As whitespace it reads as nothing having happened, and
a row of blanks cannot be counted. Because every row is the same width, **a column
that changes is a position committing**, and no column ever changes twice: once a
position is committed it is final.

Four things are worth reading off those tables.

**The model decides the length before it decides the letters.** At 8 steps, the
five positions committed over steps 1 to 4 are all `[PAD]`, and step 5 places three
more, so eight of `jeallyn`'s nine `[PAD]`s are down before a single letter is.
Nothing in the training objective asked for that ordering. It falls out of
confidence-based remasking: with everything else still masked there is very little
evidence about which letter goes where, but plenty
about where the name stops. Positions 11 through 15 are `[PAD]` in essentially
every name in the training set, which the model can read off the position
embedding alone, so `[PAD]` is what it is surest of and `[PAD]` is what gets
committed first.

**Commits accelerate: 1, 1, 1, 2, 3, 2, 3, 3.** That shape is the cosine schedule,
and it is the same for every name at 8 steps. `main.py` asserts this, checking each
of the ten samples against the first. Early on, each position is being guessed from
almost nothing, so the schedule commits one at a time. Once enough letters are
fixed, the rest are nearly determined by them, and three can safely go in one pass.
This is the whole economic argument for diffusion decoding.

**The letters do not arrive left to right.** At 16 steps, `sarya` places its `a` at
position 4 on step 12, the `a` at position 1 on step 13, then fills positions 3 and
2, and only on the last step does the leading `s` appear. A causal model cannot do
this; the bidirectional attention from section 2 is what lets a position be decided
from context on both sides of it.

**The three tables show where the step count spends its budget.** At 16 steps every
letter is placed with all its committed neighbours already visible. At 4 steps, step
4 commits six positions at once with a single `e` to condition on, and the result is
`hlraine`, which opens on a cluster that starts no name in the training set. The
degradation in the sample lists is this row.

## Why These Choices

### Two layers instead of one

Lab 03's GPT works with a single layer because the causal mask provides a strong structural prior: position 5 always sees exactly positions 0–4. Diffusion has no such luxury, as each position sees a random, varying subset of unmasked neighbors. Two layers let the model reason about what its neighbors learned about *their* neighbors.

### Batch size 32

Batch size is the biggest difference from the autoregressive labs. In autoregressive training, every position contributes to the loss, giving a stable gradient even from one sample. In diffusion, only masked positions contribute, and the masking is random, so each single-sample gradient points in a different noisy direction. Batching averages 32 gradients per step, smoothing out the noise. Without it, the model produces gibberish.

### 3000 steps

In autoregressive training, every position contributes to the loss, so the model learns from all 16 characters per step. In diffusion, only the masked positions contribute. With log-uniform `t` over [0.2, 1.0], the average step masks about 8 out of 16 tokens, roughly half the signal per step. More steps compensate.

### Log-uniform noise schedule

The ELBO loss has a `1/t` weight that makes gradients spike when `t` is small. Sampling `t` log-uniformly (density `∝ 1/t`) exactly cancels this weight via importance sampling, so the loss simplifies to an unweighted average cross-entropy.

### Weight tying

The input embeddings (`wte`) and output projection (`lm_head`) share the same matrix. This is standard in GPT-2, BERT, T5, and most modern language models. It saves parameters and acts as a regularizer.

## What's Different From the GPT (Lab 03)

| | Lab 03, microGPT | Lab 16, microDiffusion |
|---|---|---|
| Generation | Left-to-right, one token at a time | All-at-once, iteratively refined from noise |
| Attention | Causal (each token sees only past) | Bidirectional (each token sees all others) |
| Special tokens | BOS (start/end marker) | MASK (noise) + PAD (fixed-length padding) |
| Sequence length | Variable (generate until BOS) | Fixed (model learns to predict PAD) |
| Batch size | 1 (works fine) | 32 (critical, single-sample is too noisy) |
| Layers | 1 | 2 (diffusion needs depth for message-passing) |
| Loss masking | All positions | Only masked positions (MASK logit suppressed) |
| Noise schedule | — | Log-uniform (importance sampling) |
| Inference passes | ~7 (mean name length is 6.1 characters, plus the terminal BOS) | 16 / 8 / 4 (a dial, though 16 steps over 16 positions saves nothing) |
| Inference strategy | Sampling with temperature | Confidence remasking + temperature annealing |
| Weight tying | No | Yes (wte = lm_head) |

## Running It

```bash
# From the project root:
python run_lab.py 16

# Or directly:
cd 16_text_diffusion && uv run python main.py
```

Trains for 3000 steps with batch size 32, then generates 10 names at each of three step counts (16, 8, 4), reporting forward passes, the per-step commit counts, and the full step-by-step denoising of one name for each. The whole run took about 90 seconds on the 2-core CPU container it was last timed on.

## Why This Matters

Autoregressive generation is the dominant paradigm for language models. GPT, LLaMA, and Claude all generate left-to-right, one token at a time. Masked diffusion is a fundamentally different approach: generate everything at once, then refine. It's the same shift that happened in image generation, where diffusion models (DALL-E 2, Stable Diffusion) overtook autoregressive ones (DALL-E 1).

For text, diffusion is still catching up. But it has structural advantages: parallel generation (several positions per forward pass, not one), bidirectional context (no "reversal curse" because the model sees the whole sequence), and natural support for editing (re-mask and re-generate any part).

The step-count sweep at the end of the run is where that advantage becomes concrete, and where its price shows up too. At 16 steps the names are clean (`kina`, `sarya`, `kalan`, `jalya`, `jamie`) but the run costs one forward pass per position, exactly like left-to-right decoding, so that arm is the quality reference, not the speed result. The saving starts at 8 steps, where they begin to fray, and at 4 they are mostly mush. Fewer steps means more positions committed per pass with less information about their neighbours, and a 6.8K-parameter model has very little slack. How few steps you can get away with is exactly the question production diffusion LMs are trying to answer.

### Why the generated names aren't as good as lab 03's

The AR model has a massive inductive bias advantage at small scale. The causal mask gives it a free, perfect decomposition: each position only needs to learn "given these exact characters to my left, what comes next?" Diffusion has to learn a much harder function: each position must predict its token from a *random, varying* subset of visible neighbors.

LLaDA's scaling curves show this directly: diffusion consistently underperforms AR at smaller sizes, and the gap narrows as you scale up. At our 6,848 parameters, AR wins comfortably. The purpose of this lab is to show *how* diffusion works, not to beat AR at a scale where it can't.

## References

1. **MDLM**: Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524), [GitHub](https://github.com/kuleshov-group/mdlm)

2. **LLaDA**: Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). *Large Language Diffusion Models.* ICML 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992), [GitHub](https://github.com/ML-GSAI/LLaDA)

3. **MaskGIT**: Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)

4. **RADD**: Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., & Li, C. (2025). *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.* [arXiv:2406.03736](https://arxiv.org/abs/2406.03736)
