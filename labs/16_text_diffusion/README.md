# Understanding LLMs by Building One: Text Diffusion

> *Instead of generating names left-to-right, names emerge from pure noise, all [MASK] tokens, through iterative unmasking.*

This is a **masked diffusion language model** in PyTorch. Same transformer building blocks as the PyTorch labs (03, 04), but a fundamentally different generative paradigm.

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

The names aren't memorized. They emerge from a diffusion process that has learned the statistical patterns of how characters combine. Unlike the autoregressive GPT, which writes names left-to-right, this model fills in all positions at once, refining its guesses over 64 denoising steps.

## The Algorithm, Step by Step

Let's walk through `main.py`. The model architecture (RMSNorm, multi-head attention, MLP, residual connections) is the same as labs 03/04. Here we focus on what's different.

### 1. The Tokenizer

```python
uchars = sorted(set(''.join(docs)))
MASK = len(uchars)     # the "noise" state
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

Lab 03's `CausalSelfAttention` masks future positions so each token only sees the past. `BidirectionalSelfAttention` has no mask — every position attends to every other. This lets the model use context from both sides to fill in the blanks.

The architecture is otherwise identical: same embedding, same RMSNorm, same multi-head attention, same MLP with ReLU. Key hyperparameter differences:

```python
n_layer = 2       # diffusion needs depth to gather scattered clues
num_steps = 3000
batch_size = 32   # critical for diffusion
```

### 3. Training: Noise Instead of Next-Token Prediction

This is where diffusion diverges most from autoregressive training. Lab 03 wraps each name with BOS and predicts each next character. Here, we corrupt the name with random masks and predict what's underneath:

```python
t = math.exp(random.uniform(math.log(0.2), 0))  # log-uniform noise level
noisy = [MASK if random.random() < t else c for c in clean]
```

The noise level `t` is sampled log-uniformly rather than uniformly. This is importance sampling that cancels the `1/t` weight in the ELBO, eliminating gradient spikes.

The loss is computed only on masked positions, and the MASK logit is suppressed so no probability mass is wasted on an impossible prediction:

```python
logits[:, :, MASK] = logits[:, :, MASK] - 1e6  # never predict MASK
loss = F.cross_entropy(logits[mask], targets[mask])
```

The MASK zeroing follows [MDLM](https://github.com/kuleshov-group/mdlm)'s `_subs_parameterization`.

### 4. Why Batching Is Critical

Unlike the autoregressive labs (03, 04) which work fine with batch_size=1, diffusion **needs batching**. With a single sample, each gradient is based on one random masking pattern — far too noisy for the model to learn. Batching averages over 32 different mask patterns per step, giving stable gradients. This was the single biggest factor in getting the model to produce plausible names.

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

    # Remask the least confident predictions
    if s > 0 and confidences:
        n_to_remask = int(len(confidences) * s / t)
        confidences.sort()  # lowest confidence first
        for _, i in confidences[:n_to_remask]:
            predicted[i] = MASK
```

Three techniques improve generation quality:

- **Confidence-based remasking**: instead of randomly re-corrupting predictions, the model keeps the tokens it's most sure about and reconsiders the rest. This is the same `low_confidence` strategy used in [LLaDA's inference code](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) and originally introduced by [MaskGIT](https://arxiv.org/abs/2202.04200).
- **Temperature annealing** (0.8 to 0.3): high temperature early on encourages exploration when everything is uncertain; low temperature at the end sharpens the final choices.
- **Cosine schedule**: instead of linearly decreasing the noise level, a cosine curve spends more time in the critical middle range where the name's structure is being decided.

```
Step 64 (t=1.00, temp=0.80): [M]  [M]  [M]  [M]  [M]  [M]  [M]  [M]  ...
Step 48 (t=0.71, temp=0.65): [M]   a   [M]  [M]   a   [M]  [M]  [PAD] ...
Step 32 (t=0.38, temp=0.49):  m    a   [M]   i    a   [M]  [PAD] [PAD] ...
Step 16 (t=0.10, temp=0.35):  m    a    r    i    a   [M]  [PAD] [PAD] ...
Step  1 (t=0.00, temp=0.30):  m    a    r    i    a   [PAD] [PAD] [PAD] ...
Result:                        maria
```

## Why These Choices

### Two layers instead of one

Lab 03's GPT works with a single layer because the causal mask provides a strong structural prior: position 5 always sees exactly positions 0–4. Diffusion has no such luxury, as each position sees a random, varying subset of unmasked neighbors. Two layers let the model reason about what its neighbors learned about *their* neighbors.

### Batch size 32

This is the single most important difference from the autoregressive labs. In autoregressive training, every position contributes to the loss, giving a stable gradient even from one sample. In diffusion, only masked positions contribute, and the masking is random — so each single-sample gradient points in a different noisy direction. Batching averages 32 gradients per step, smoothing out the noise. Without it, the model produces gibberish.

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
| Batch size | 1 (works fine) | 32 (critical — single-sample is too noisy) |
| Layers | 1 | 2 (diffusion needs depth for message-passing) |
| Loss masking | All positions | Only masked positions (MASK logit suppressed) |
| Noise schedule | — | Log-uniform (importance sampling) |
| Inference passes | ~8 (average name length) | 64 (denoising steps, cosine schedule) |
| Inference strategy | Sampling with temperature | Confidence remasking + temperature annealing |
| Weight tying | No | Yes (wte = lm_head) |

## Running It

```bash
# From the project root:
python run_lab.py 15

# Or directly:
cd 16_text_diffusion && uv run python main.py
```

Trains for 3000 steps with batch size 32 and generates 20 names. Takes a few seconds on a laptop thanks to PyTorch.

## Why This Matters

Autoregressive generation is the dominant paradigm for language models. GPT, LLaMA, and Claude all generate left-to-right, one token at a time. Masked diffusion is a fundamentally different approach: generate everything at once, then refine. It's the same shift that happened in image generation, where diffusion models (DALL-E 2, Stable Diffusion) overtook autoregressive ones (DALL-E 1).

For text, diffusion is still catching up. But it has structural advantages: parallel generation (fill in all blanks at once, not one by one), bidirectional context (no "reversal curse" because the model sees the whole sequence), and natural support for editing (re-mask and re-generate any part).

### Why the generated names aren't as good as lab 03's

The AR model has a massive inductive bias advantage at small scale. The causal mask gives it a free, perfect decomposition: each position only needs to learn "given these exact characters to my left, what comes next?" Diffusion has to learn a much harder function: each position must predict its token from a *random, varying* subset of visible neighbors.

LLaDA's scaling curves show this directly: diffusion consistently underperforms AR at smaller sizes, and the gap narrows as you scale up. At our 6,848 parameters, AR wins comfortably. The purpose of this lab is to show *how* diffusion works, not to beat AR at a scale where it can't.

## References

1. **MDLM**: Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524), [GitHub](https://github.com/kuleshov-group/mdlm)

2. **LLaDA**: Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). *Large Language Diffusion Models.* ICML 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992), [GitHub](https://github.com/ML-GSAI/LLaDA)

3. **MaskGIT**: Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)

4. **RADD**: Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., & Li, C. (2025). *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.* [arXiv:2406.03736](https://arxiv.org/abs/2406.03736)
