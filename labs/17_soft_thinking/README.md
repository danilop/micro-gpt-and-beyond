# Understanding LLMs by Building One: Soft Thinking

Same architecture as the PyTorch version (03), but with soft decoding at inference time. Instead of collapsing to a single sampled token at each step, soft thinking passes a "concept token," a probability-weighted blend of all token embeddings, to the next step. The full probability distribution flows forward, preserving information that hard decoding discards.

## Why this version exists

Standard autoregressive decoding forces the model's rich internal state through an **information bottleneck** at every step: the entire output distribution (a vector over the full vocabulary) collapses into a single integer, the chosen token ID. Every subsequent step sees only that one embedding, with no trace of what the model "almost said." Soft thinking removes this bottleneck by passing the full distribution forward as a continuous embedding.

## What makes it interesting

### The information bottleneck

At each decoding step, the model outputs logits over the full vocabulary, a rich signal encoding probabilities for every possible next token. Standard decoding discards almost all of this:

```
Standard (hard):   logits -> sample -> token_id -> embed(token_id)   -> next input
                   [27 values]        [1 integer]  [16-dim vector]

Soft thinking:     logits -> softmax(logits/T) @ embed_table         -> next input
                   [27 values]  [27 probabilities]  [16-dim vector]
```

The hard path compresses 27 logits into 1 integer. The soft path preserves the full distribution by computing a weighted average of all token embeddings, a "concept token" that encodes the model's uncertainty.

Note that "hard" here means **sampling**, not argmax. The code calls `torch.multinomial` on `softmax(logits / 0.5)`, so hard decoding is still stochastic; what makes it hard is that only the sampled token's embedding survives into the next step. Argmax would be a further restriction on top.

### Concept tokens live in embedding space

The concept token is computed as:

```python
soft_probs = softmax(logits / T)  # (vocab_size,)
concept_token = soft_probs @ embed_table  # (n_embd,)
```

This is a point in the same n_embd-dimensional space as regular token embeddings, but instead of representing a single discrete token, it represents a blend. If the model is 80% confident about 'a' and 20% about 'e', the concept token sits somewhere between the embeddings for 'a' and 'e', carrying both possibilities forward.

### Temperature controls the blend

The soft temperature T determines how much information flows through:

| Temperature | Behavior | Concept token |
|---|---|---|
| T -> 0 | softmax becomes one-hot | Identical to hard decoding |
| T = 0.5 | Peaked but not collapsed | Dominated by top tokens |
| T = 1.0 | Standard softmax | Moderate blend of candidates |
| T = 2.0 | Flattened distribution | Many tokens contribute |
| T -> inf | Uniform distribution | Mean of all embeddings (noise) |

The lab generates names at each temperature, reporting the Shannon entropy of **the distribution that builds the next input**, `softmax(logits / soft_temp)`, the one soft temperature actually controls. Measuring the sampling distribution instead would be measuring a quantity that is identical in every row of the table, and would read flat no matter what soft temperature does.

For hard decoding that input distribution is a one-hot delta, so its entropy is exactly 0. That is the information bottleneck expressed as a number, and it is the reason hard decoding is the baseline rather than a competitor in the entropy column.

### Benefit and cost, measured together

Entropy only measures the benefit, how much of the distribution survives into the next step. On its own it would make the highest temperature look best. So the lab prints two cost columns next to it. Measured over 50 samples per row:

| Mode | Concept H (max 3.30) | Sample NLL | Adjacent-dup rate |
|---|---|---|---|
| Hard (standard decoding) | 0.000 | 1.8667 | 1.9% |
| Soft T=0.5 (mild blend) | 1.706 | 2.0875 | 11.0% |
| Soft T=1.0 (moderate blend) | 2.509 | 2.1113 | 12.6% |
| Soft T=2.0 (diffuse blend) | 2.995 | 2.2805 | 20.8% |
| Real held-out names (500) | — | 2.4001 | 4.9% |

"Sample NLL" is the per-token negative log-likelihood of each row's generated names, scored under the same model on the ordinary hard path. It answers "would the model itself have written this?". Read it as a trend down the column, where higher means the output has drifted away from what the model was trained on, rather than as pass/fail against the real-names row. Sampling runs at temperature 0.5, which sharpens the output, so every generated row scores below real names whether it has drifted or not; the reference row is there to give the numbers a scale, not a threshold.

"Adjacent-dup rate" is the fraction of neighbouring character pairs that repeat the same character, of any kind, which is the specific way this technique degrades: at T=2.0 the output is full of stutters (`aayay`, `maaea`, `aall`, where the last one repeats a consonant, so this is not a vowel-only effect).

Read the table left to right and the tradeoff is not rhetorical any more. Entropy rises monotonically with soft temperature, exactly as the theory says. So does NLL, and so does the stutter rate, from 1.9% at hard decoding to 20.8% at T=2.0, which is 4.3x the 4.9% you see in real names. Richer information in, more drift out. Why the drift takes the form of repeated characters is a plausible story (a flat blend carries little information about which token was just emitted) rather than something measured here.

### The out-of-distribution challenge

There's a fundamental tension: the model was **trained** on discrete token embeddings (points on the embedding manifold), but concept tokens are weighted averages that may lie **between** those points, in regions the model has never seen. This is the train-test mismatch that Lab 18 addresses.

## What you learn here

- The information bottleneck in autoregressive decoding (collapsing distributions to integers)
- How concept tokens preserve uncertainty by blending all token embeddings
- The role of temperature in controlling the hard-to-soft spectrum
- Shannon entropy as a measure of distribution "spread", and the discipline of measuring the distribution you actually care about rather than the nearest one to hand
- Why this is training-free, since only the decoding loop changes, not the model
- The out-of-distribution challenge when feeding soft inputs to a hard-trained model, quantified: sample NLL and adjacent-duplicate rate both climb with soft temperature

## What's not covered (but exists in practice)

- **Gumbel-Softmax** (Jang et al., 2017): Adds Gumbel noise before softmax for differentiable discrete sampling. Used in the Soft Thinking paper to inject exploration into concept tokens.
- **Cold Stop mechanism**: Monitor entropy during soft decoding; when it drops below a threshold (model is very confident), switch back to hard decoding. Prevents out-of-distribution (OOD) drift.
- **Coconut, Chain of Continuous Thought** (Hao et al., Meta, 2024): Feeds hidden states directly back as input (bypassing both the output head and embedding lookup), but requires multi-stage curriculum training.
- **Quiet-STaR** (Zelikman et al., 2024): LLMs learn to generate hidden "thought" rationales at every token position using learned start/end-of-thought tokens.
- **SoftCoT** (Xu et al., 2025): Uses a small assistant model to generate soft thought tokens, projected into the main model's space.
- **Key papers**: Zhang et al. "Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space" (NeurIPS 2025), Hao et al. "Training Large Language Models to Reason in a Continuous Latent Space" (2024).

## Run

```bash
uv run python main.py
```

Trains for 1000 steps (identical to Lab 03), then generates 50 names using hard decoding and soft decoding at three temperatures (T=0.5, 1.0, 2.0). For each it reports the Shannon entropy of the concept-token distribution, the per-token NLL of the generated names under the model, and the adjacent-duplicate-character rate, with a row of 500 real unseen names as the reference.

This lab is self-contained. Lab 18 duplicates the model and the generation function rather than importing them, because each lab is meant to be readable end to end from a single file.

## Why soft thinking matters

Large language models spend enormous compute producing rich output distributions at every step, only to throw almost all of it away by picking a single token. Soft thinking is the insight that this information doesn't have to be wasted because the full distribution can flow forward as a continuous signal. At scale it appears to pay off: Zhang et al. (2025) report +2.5 pass@1 while using 22% fewer tokens on large reasoning models. That is their result, not this lab's, since nothing at 4,192 parameters and a names dataset could test it. What this lab shows is the core mechanism: one line of code (`softmax @ embedding_table`) is the entire difference.
