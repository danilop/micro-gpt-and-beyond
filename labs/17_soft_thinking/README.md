# Understanding LLMs by Building One: Soft Thinking

Same architecture as the PyTorch version (03), but with soft decoding at inference time. Instead of collapsing to a single token (argmax) at each step, soft thinking passes a "concept token," a probability-weighted blend of all token embeddings, to the next step. The full probability distribution flows forward, preserving information that hard decoding discards.

## Why this version exists

Standard autoregressive decoding forces the model's rich internal state through an **information bottleneck** at every step: the entire output distribution (a vector over the full vocabulary) collapses into a single integer, the chosen token ID. Every subsequent step sees only that one embedding, with no trace of what the model "almost said." Soft thinking removes this bottleneck by passing the full distribution forward as a continuous embedding.

## What makes it interesting

### The information bottleneck

At each decoding step, the model outputs logits over the full vocabulary, a rich signal encoding probabilities for every possible next token. Standard decoding discards almost all of this:

```
Standard (hard):   logits -> argmax -> token_id -> embed(token_id)   -> next input
                   [27 values]        [1 integer]  [16-dim vector]

Soft thinking:     logits -> softmax(logits/T) @ embed_table         -> next input
                   [27 values]  [27 probabilities]  [16-dim vector]
```

The hard path compresses 27 logits into 1 integer. The soft path preserves the full distribution by computing a weighted average of all token embeddings, a "concept token" that encodes the model's uncertainty.

### Concept tokens live in embedding space

The concept token is computed as:

```python
soft_probs = softmax(logits / T)             # (vocab_size,)
concept_token = soft_probs @ embed_table      # (n_embd,)
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

The lab generates names at each temperature, reporting the Shannon entropy of the distribution at each step. Higher entropy means more tokens contribute to the concept token.

### The out-of-distribution challenge

There's a fundamental tension: the model was **trained** on discrete token embeddings (points on the embedding manifold), but concept tokens are weighted averages that may lie **between** those points, in regions the model has never seen. This is the train-test mismatch that Lab 18 addresses.

## What you learn here

- The information bottleneck in autoregressive decoding (collapsing distributions to integers)
- How concept tokens preserve uncertainty by blending all token embeddings
- The role of temperature in controlling the hard-to-soft spectrum
- Shannon entropy as a measure of distribution "spread"
- Why this is training-free, since only the decoding loop changes, not the model
- The out-of-distribution challenge when feeding soft inputs to a hard-trained model

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

Trains for 1000 steps (identical to Lab 03), then generates 20 names using hard decoding and soft decoding at three temperatures (T=0.5, 1.0, 2.0). Reports the Shannon entropy of each step's distribution.

## Why soft thinking matters

Large language models spend enormous compute producing rich output distributions at every step, only to throw almost all of it away by picking a single token. Soft thinking is the insight that this information doesn't have to be wasted because the full distribution can flow forward as a continuous signal. At scale, this improves reasoning accuracy by +2.5% while using 22% fewer tokens. This lab shows the core mechanism at microGPT scale: one line of code (`softmax @ embedding_table`) is the entire difference.
