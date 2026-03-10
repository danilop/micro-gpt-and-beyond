# microGPT and Beyond — Text Diffusion

> *Instead of generating names left-to-right, names emerge from pure noise — all [MASK] tokens — through iterative unmasking.*

This is a **masked diffusion language model** implemented from scratch in pure Python, with zero dependencies. Same autograd engine as [Karpathy's microGPT](https://karpathy.ai/microgpt.html) in `01_pure_python`, same transformer building blocks, but a fundamentally different generative paradigm.

Where the original GPT predicts the next token given all previous tokens (autoregressive, left-to-right), this model predicts *all masked tokens simultaneously* given the unmasked context (diffusion, all-at-once). The name materializes from noise like a photograph developing in a darkroom.

---

## What Does It Do?

It trains a tiny bidirectional transformer on the same ~32,000 names dataset as lab 01, then generates new names by starting from pure noise and iteratively unmasking:

```
sample  1: jenen
sample  2: mala
sample  3: kanan
sample  4: parsan
sample  5: daiha
...
```

The names aren't memorized — they emerge from a diffusion process that has learned the statistical patterns of how characters combine. Unlike the autoregressive GPT, which writes names left-to-right, this model fills in all positions at once, refining its guesses over 40 denoising steps.

---

## The Algorithm, Step by Step

Let's walk through `microdiffusion.py`. Everything that's shared with lab 01 (the `Value` autograd engine, `linear`, `softmax`, `rmsnorm`, Adam optimizer) works identically — see the [lab 01 README](../01_pure_python/README.md) for those explanations. Here we focus on what's different.

### 1. The Tokenizer

```python
uchars = sorted(set(''.join(docs)))
MASK = len(uchars)     # the "noise" state
PAD = len(uchars) + 1  # fills unused positions
vocab_size = len(uchars) + 2
```

Lab 01 uses 26 characters + BOS = 27 tokens. Here, BOS is replaced by two special tokens: `MASK` (the noise that the model learns to denoise) and `PAD` (fills positions beyond the name's length). The vocabulary is 28 tokens.

### 2. The Model — Bidirectional, Not Causal

```python
def mask_predictor(token_ids):
    n = len(token_ids)
    xs = [rmsnorm([t + p for t, p in zip(state_dict['wte'][tid], state_dict['wpe'][pos])])
          for pos, tid in enumerate(token_ids)]
    for li in range(n_layer):
        # ... attention over ALL positions (no causal mask) ...
        attn_logits = [sum(q_h[j] * ks[t][hs+j] ...) for t in range(n)]
```

Lab 01's `gpt()` function processes one token at a time, using a KV cache and only attending to past positions (causal). The `mask_predictor()` processes all positions at once, and every position attends to every other — including future ones. This bidirectional attention is what lets the model use context from both sides to fill in the blanks.

The architecture is otherwise identical: same embedding, same RMSNorm, same multi-head attention, same MLP with ReLU. Two differences in the hyperparameters:

```python
n_layer = 2     # diffusion needs depth to gather scattered clues
num_steps = 2000
```

Lab 01 uses 1 layer and 1000 steps. We use 2 layers because diffusion has a harder job — see [Why These Choices](#why-these-choices) below.

### 3. Training — Noise Instead of Next-Token Prediction

This is where diffusion diverges most from autoregressive training. Lab 01 wraps each name with BOS and predicts each next character. Here, we corrupt the name with random masks and predict what's underneath:

```python
# Pad to fixed length, then corrupt
clean = [uchars.index(ch) for ch in doc] + [PAD] * (block_size - len(doc))
t = math.exp(random.uniform(math.log(0.2), 0))  # log-uniform noise level
noisy = [MASK if random.random() < t else c for c in clean]
```

The noise level `t` is sampled log-uniformly rather than uniformly — this is importance sampling that cancels the `1/t` weight in the ELBO, eliminating gradient spikes.

The loss is computed only on masked positions, and the MASK logit is zeroed out so no probability mass is wasted on an impossible prediction:

```python
for i in range(block_size):
    if noisy[i] == MASK:
        logits_i = all_logits[i][:]
        logits_i[MASK] = logits_i[MASK] - 1e6  # never predict MASK
        probs = softmax(logits_i)
        losses.append(-probs[clean[i]].log())
loss = (1.0 / n_masked) * sum(losses)
```

The MASK zeroing follows [MDLM](https://github.com/kuleshov-group/mdlm)'s `_subs_parameterization`, which adds `-1e6` to the MASK logit. Note: unlike MDLM, we do *not* exclude PAD from the loss. MDLM can do this because it also masks PAD in attention via `attention_mask`. In our implementation, PAD tokens participate in attention normally, so the model needs the PAD loss signal to learn where names end — without it, every position becomes a character and the model loses all length control.

### 4. Inference — Names Emerge From Noise

Lab 01 generates left-to-right: start with BOS, sample next token, feed it back, repeat until BOS again. Here, generation is a denoising process:

```python
seq = [MASK] * block_size  # start from pure noise

for step_i in range(num_denoise_steps, 0, -1):
    t = math.cos(math.pi / 2 * (1 - step_i / num_denoise_steps))  # cosine schedule
    s = math.cos(math.pi / 2 * (1 - (step_i - 1) / num_denoise_steps))
    temperature = 0.3 + 0.5 * t  # anneal: explore early, commit late

    all_logits = mask_predictor(seq)
    # ... predict tokens, track confidence ...

    # Remask the least confident predictions
    if s > 0 and confidences:
        n_to_remask = int(len(confidences) * s / t)
        confidences.sort()  # lowest confidence first
        for _, i in confidences[:n_to_remask]:
            predicted[i] = MASK
```

Two techniques improve generation quality:

- **Confidence-based remasking**: instead of randomly re-corrupting predictions, the model keeps the tokens it's most sure about and reconsiders the rest. This is the same `low_confidence` strategy used in [LLaDA's inference code](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) and originally introduced by [MaskGIT](https://arxiv.org/abs/2202.04200).
- **Temperature annealing** (0.8 → 0.3): high temperature early on encourages exploration when everything is uncertain; low temperature at the end sharpens the final choices.
- **Cosine schedule**: instead of linearly decreasing the noise level, a cosine curve spends more time in the critical middle range (t ≈ 0.3–0.7) where the name's structure is being decided, and rushes through the trivial endpoints.

```
Step 64 (t=1.00, temp=0.80): [M]  [M]  [M]  [M]  [M]  [M]  [M]  [M]  ...
Step 48 (t=0.71, temp=0.65): [M]   a   [M]  [M]   a   [M]  [M]  [PAD] ...
Step 32 (t=0.38, temp=0.49):  m    a   [M]   i    a   [M]  [PAD] [PAD] ...
Step 16 (t=0.10, temp=0.35):  m    a    r    i    a   [M]  [PAD] [PAD] ...
Step  1 (t=0.00, temp=0.30):  m    a    r    i    a   [PAD] [PAD] [PAD] ...
Result:                        maria
```

---

## Why These Choices

Several decisions go beyond the minimal textbook recipe. Each addresses a challenge that emerges at tiny scale, informed by the [MDLM](https://github.com/kuleshov-group/mdlm) and [LLaDA](https://github.com/ML-GSAI/LLaDA) reference implementations.

### Two layers instead of one

Lab 01's GPT works with a single layer because the causal mask provides a strong structural prior: position 5 always sees exactly positions 0–4. Diffusion has no such luxury — each position sees a random, varying subset of unmasked neighbors. One layer of attention gets a single round of "look at what's visible." Two layers let the model reason about what its neighbors learned about *their* neighbors. At large scale this difference vanishes (LLaDA matches autoregressive quality at 8B parameters with the same architecture), but at 7K parameters the extra layer is the single biggest improvement.

### 2000 steps instead of 1000

In autoregressive training, every position contributes to the loss — the model learns from all 16 characters per step. In diffusion, only the masked positions contribute. With log-uniform `t` over [0.2, 1.0], the average step masks about 8 out of 16 tokens — roughly half the signal per step. Doubling the steps from 1000 to 2000 compensates.

The minimum noise level `t_min=0.2` also matters for efficiency. With `t_min=0.1`, about 22% of training steps mask only 0–2 tokens — the model sees nearly the entire clean sequence and learns almost nothing. Raising the floor to 0.2 cuts that waste to ~6%, making more of the 2000 steps count.

### Log-uniform noise schedule

The ELBO loss has a `1/t` weight that makes gradients spike when `t` is small. Sampling `t` log-uniformly (density `∝ 1/t`) exactly cancels this weight via importance sampling, so the loss simplifies to an unweighted average cross-entropy. The reference implementations handle this differently: MDLM offers an `importance_sampling` flag; LLaDA uses uniform `t` with `1/t` weight but normalizes by total sequence length instead of masked count. Both yield the same expected gradient — they differ only in variance.

### MASK zeroing

Zeroing the MASK logit (MDLM's `_subs_parameterization`) prevents wasting probability mass on an impossible prediction — `[MASK]` is noise, not a real token. PAD, on the other hand, stays in the loss: the model needs to learn that positions beyond the name should be PAD. MDLM can exclude PAD because it also masks PAD in attention; we don't, so the loss signal is the only way the model learns name length.

### Weight tying

The input embeddings (`wte`) and output projection (`lm_head`) share the same matrix. The logit for token "a" is the dot product of the hidden state with "a"'s embedding — the same vector used to represent "a" as input. This is standard in GPT-2, BERT, T5, and most modern language models. It saves 448 parameters (~6% of our 7K total) and acts as a regularizer: the model is forced to learn a single, consistent embedding space where similar characters cluster together, rather than learning separate input and output representations that might disagree.

### Confidence remasking and temperature annealing

Random remasking might throw away a confident prediction while keeping an uncertain one. LLaDA's own [inference code](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py) defaults to `low_confidence` remasking — keeping the most certain tokens and reconsidering the rest. Temperature annealing complements this: explore when uncertain, commit when the context is rich.

---

## What's Different From the GPT (Lab 01)

| | Lab 01 — microGPT | Lab 09 — microDiffusion |
|---|---|---|
| Generation | Left-to-right, one token at a time | All-at-once, iteratively refined from noise |
| Attention | Causal (each token sees only past) | Bidirectional (each token sees all others) |
| Special tokens | BOS (start/end marker) | MASK (noise) + PAD (fixed-length padding) |
| Sequence length | Variable (generate until BOS) | Fixed (model learns to predict PAD) |
| KV cache | Yes (reuse past computations) | No (recompute everything each step) |
| Layers | 1 | 2 (diffusion needs depth for message-passing) |
| Loss masking | All positions | All masked positions (MASK logit zeroed) |
| Noise schedule | — | Log-uniform (importance sampling) |
| Inference passes | ~8 (average name length) | 64 (denoising steps, cosine schedule) |
| Inference strategy | Sampling with temperature | Confidence remasking + temperature annealing |
| Weight tying | no | yes (wte = lm_head) |

The building blocks are identical: same `Value` autograd engine, same `linear`, `softmax`, `rmsnorm`, same embedding dimension (16), same number of heads (4). The difference is how the computation is organized — all positions at once with bidirectional attention, instead of one at a time with causal masking.

---

## Running It

```bash
python microdiffusion.py
```

No `pip install` needed. Trains for 2000 steps and generates 20 names. Takes several minutes on a laptop (longer than lab 01 due to the second layer and bidirectional attention).

---

## Why This Matters

Autoregressive generation is the dominant paradigm for language models — GPT, LLaMA, Claude all generate left-to-right, one token at a time. Masked diffusion is a fundamentally different approach: generate everything at once, then refine. It's the same shift that happened in image generation, where diffusion models (DALL-E 2, Stable Diffusion) overtook autoregressive ones (DALL-E 1).

For text, diffusion is still catching up. But it has structural advantages that don't show at tiny scale: parallel generation (fill in all blanks at once, not one by one), bidirectional context (no "reversal curse" — the model sees the whole sequence), and natural support for editing (re-mask and re-generate any part).

### Why the generated names aren't as good as lab 01's

If you run both labs, you'll notice the autoregressive GPT produces cleaner names. This isn't a bug — it's a fundamental scaling property of diffusion models.

The AR model has a massive inductive bias advantage at small scale. The causal mask gives it a free, perfect decomposition: each position only needs to learn "given these exact characters to my left, what comes next?" A 1-layer, 16-dim model can learn "after 'an', 'n' is likely" because it always sees exactly those two characters.

Diffusion has to learn a much harder function. Each position must predict its token from a *random, varying* subset of visible neighbors — position 3 might see {0, 5, 7} one step and {1, 2, 4, 6} the next. Representing all possible conditioning sets requires capacity that grows with depth and width. A tiny model can't do it well.

LLaDA's scaling curves (Section 2.2, Figure 3) show this directly: diffusion consistently underperforms AR at smaller sizes, and the gap narrows as you scale up. The crossover happens in the billions of parameters — at 8B, they match. At our 7K parameters, AR wins comfortably. The purpose of this lab is to show *how* diffusion works, not to beat AR at a scale where it can't.

This implementation shows the core algorithm in ~230 lines of pure Python. If you understand how lab 01's GPT works, comparing it with this file shows you exactly what changes when you swap autoregressive for diffusion — and what stays the same.

---

## References

1. **MDLM** — Sahoo, S. S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) — [GitHub](https://github.com/kuleshov-group/mdlm)

2. **LLaDA** — Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., & Li, C. (2025). *Large Language Diffusion Models.* ICML 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992) — [GitHub](https://github.com/ML-GSAI/LLaDA)

3. **MaskGIT** — Chang, H., Zhang, H., Jiang, L., Liu, C., & Freeman, W. T. (2022). *MaskGIT: Masked Generative Image Transformer.* CVPR 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)

4. **RADD** — Ou, J., Nie, S., Xue, K., Zhu, F., Sun, J., & Li, C. (2025). *Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data.* [arXiv:2406.03736](https://arxiv.org/abs/2406.03736)

The autograd engine and transformer architecture are from [microGPT](https://karpathy.ai/microgpt.html) by [Andrej Karpathy](https://github.com/karpathy), included in this project as `01_pure_python`.

Key theoretical results used: the training loss is a valid ELBO (Sahoo et al., 2024; Ou et al., 2025); the model doesn't need time `t` as input — the conditional distribution is time-invariant (Ou et al., 2025); log-uniform sampling is importance sampling that eliminates the `1/t` variance (standard technique; see Kingma et al., 2021 for the continuous-diffusion analogue).
