# Understanding LLMs by Building One: Soft Training

Builds on Lab 17 (soft thinking): instead of only using concept tokens at inference, this version also uses them during training. A curriculum gradually replaces ground-truth token embeddings with the model's own soft predictions, narrowing the train-test gap that limits inference-only soft thinking, and the lab measures that gap instead of asserting it.

## Why this version exists

Lab 17 showed that soft decoding preserves information by passing concept tokens instead of discrete embeddings. But there's a mismatch: the model was trained on discrete token embeddings (teacher forcing), yet at inference it receives blended concept tokens, inputs from a region of embedding space it has never seen. This version trains the model to handle soft inputs, which narrows that gap measurably, by about half, at a cost on hard inputs.

## What makes it interesting

### The train-test mismatch

In standard teacher forcing (feeding ground-truth tokens as input during training, rather than the model's own predictions), the model always sees perfect ground-truth embeddings:

```
Training:   embed("a"), embed("l"), embed("i"), embed("c"), embed("e")
Inference:  embed(BOS), concept_1,  concept_2,  concept_3,  concept_4
```

Concept tokens are weighted averages of many embeddings, and they don't look like any single token the model trained on. The further the concept token drifts from the discrete embedding manifold, the more the model's behavior degrades.

### Scheduled soft tokens (the curriculum)

The fix is to gradually introduce concept tokens during training:

```python
mix = step / num_steps    # 0 -> 1 over training

# Model's own soft predictions (detached)
soft_embeds = softmax(logits / T) @ embed_table

# Mix: BOS stays ground truth, rest blended
input = (1 - mix) * embed(ground_truth) + mix * concept_token
```

Early in training (mix near 0), inputs are almost pure ground truth, so the model learns the language normally. Late in training (mix near 1), inputs are almost pure concept tokens, so the model learns to work with soft inputs. This is scheduled sampling with continuous tokens instead of discrete samples.

### Two forward passes per step

Each training step requires:

1. **Standard forward** (detached, no gradient): get logits at each position, compute concept tokens via `softmax(logits/T) @ embedding_table`
2. **Mixed forward** (with gradient): feed the blended inputs, compute cross-entropy loss, backpropagate

The concept token at position i comes from the model's prediction at position i-1 (what it thinks the next token should be), replacing the ground-truth embedding that teacher forcing would normally provide.

### Fair comparison

The lab trains two models from identical initial weights:
- **Standard-trained**: normal teacher forcing (same as Lab 03 and Lab 17)
- **Soft-trained**: soft input curriculum

Both are then evaluated with hard decoding and soft decoding, creating a 2×2 comparison:

| | Hard decoding | Soft decoding |
|---|---|---|
| **Standard-trained** | Baseline (Lab 03) | Lab 17's approach (mismatch) |
| **Soft-trained** | Does soft training help hard decoding? | Full approach (no mismatch) |

### Measuring the gap

The 2×2 above is qualitative: four columns of plausible-looking names. It does not measure the thing the lab claims, so the lab also computes per-token negative log-likelihood on 2,000 names it never trained on, with hard inputs (what teacher forcing trains on) against fully-soft inputs (the `mix = 1.0` end of the curriculum):

| Model | Hard inputs | Soft inputs | Gap |
|---|---|---|---|
| Standard-trained | 2.3892 | 2.6501 | +0.2609 |
| Soft-trained | 2.4699 | 2.6082 | +0.1384 |

Read the two gaps, not the two best numbers. The curriculum does what it advertises: the penalty for feeding soft inputs drops from +0.2609 to +0.1384 nats, a 47% reduction.

Both columns are teacher-forced: the concept tokens are computed from the real name, not from a prefix the model generated itself. That is deliberate, since it keeps the two columns scoring the same targets so their difference isolates the input representation, but it also means the soft column is the friendlier of the two soft conditions. Free-running soft decoding compounds the drift, because each concept token is built from a prefix that is already soft. Read the soft numbers as a lower bound on the gap at inference.

It is not free. On hard inputs the soft-trained model is *worse*, 2.4699 against 2.3892. Both models start from identical weights and walk the same names in the same order, so the curriculum is what caused that. Why it costs this much is a separate question the two numbers do not settle: capacity spent on reading blended embeddings, and simply taking fewer gradient steps on clean inputs as `mix` climbs to 1.0, both predict the same sign. The single best cell in the table is still standard-trained on hard inputs.

That tradeoff is the lesson. Soft training buys robustness to soft inputs and pays for it on hard inputs, and whether that is a good deal depends entirely on which one you deploy with. A version of this lab that only printed the soft-input column would look like a clean win and would be misleading.

### What the entropy column does and does not say

The 2×2 also reports the Shannon entropy of the distribution that builds each next input, `softmax(logits / soft_temp)`, the quantity soft decoding actually changes. Both hard rows come out at exactly 0, because hard decoding feeds one embedding and there is no distribution to measure.

Under soft decoding, the soft-trained model's concept tokens are slightly *more* spread than the standard-trained model's: 2.68 against 2.52 nats. If you expected soft training to produce the most confident, lowest-entropy predictions, that expectation is not what happens here. Entropy measures how much of the distribution flows forward, not how well the model uses it. The held-out gap above is the metric that answers the actual question.

## What you learn here

- Why teacher forcing creates a train-test mismatch for soft decoding
- How scheduled sampling with continuous tokens narrows the gap, and by how much (47% here)
- What it costs: the soft-trained model is measurably worse on hard inputs
- The curriculum approach: gradually shifting from discrete to soft inputs
- How to use detached forward passes for computing training signals
- The 2×2 experimental design for isolating the effect of soft training, and why a 2×2 of sample text is not yet a measurement

## What's not covered (but exists in practice)

- **Coconut multi-stage curriculum** (Hao et al., 2024): A more principled approach that progressively replaces reasoning tokens with continuous thoughts over multiple training stages, using special `<bot>`/`<eot>` markers.
- **Exposure bias** (Ranzato et al., 2016): The broader problem of train-test mismatch in sequence models. Scheduled sampling (Bengio et al., 2015) was the first fix; soft training is the continuous-token variant.
- **Self-distillation / Born-Again Networks** (Furlanello et al., 2018): A related idea on the output side, training against the model's own soft predictions instead of hard labels.
- **SofT-GRPO** (2025): Applies Group Relative Policy Optimization (GRPO) reinforcement learning to soft-thinking models using Gumbel-Softmax reparameterization for differentiable soft token sampling.
- **Consistency training**: Training the model so that soft-decoded outputs match hard-decoded outputs, ensuring concept tokens don't cause distribution drift.

## Run

```bash
uv run python main.py
```

Trains two models (standard and soft-trained) from identical initial weights, 1000 steps each. Then generates 20 names from each model with both hard and soft decoding, reporting concept-token entropy, and finishes with held-out per-token NLL on 2,000 unseen names for hard versus soft inputs.

The model definition and the generation loop are duplicated from Lab 17 rather than imported. Every lab in this series is meant to be readable end to end from one file, so there are no cross-lab imports to chase; the price is that the two files share a few identical blocks.

## Why soft training matters

Soft thinking (Lab 17) is training-free but limited by the gap between what the model trained on (discrete tokens) and what it sees at inference (concept tokens). Soft training narrows that gap by gradually teaching the model to work with continuous inputs. This is the same insight behind scheduled sampling, applied to the continuous embedding space rather than discrete token sampling.

At this scale the gap halves rather than disappears, and the model pays for it on hard inputs. That is a more useful thing to learn than a clean win would be: exposure-bias fixes are trades, and the only way to know whether the trade is worth taking is to measure both sides of it.
