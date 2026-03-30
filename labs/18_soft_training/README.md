# Understanding LLMs by Building One: Soft Training

Builds on Lab 17 (soft thinking): instead of only using concept tokens at inference, this version also uses them during training. A curriculum gradually replaces ground-truth token embeddings with the model's own soft predictions, closing the train-test gap that limits inference-only soft thinking.

## Why this version exists

Lab 17 showed that soft decoding preserves information by passing concept tokens instead of discrete embeddings. But there's a mismatch: the model was trained on discrete token embeddings (teacher forcing), yet at inference it receives blended concept tokens, inputs from a region of embedding space it has never seen. This version trains the model to handle soft inputs, closing that gap.

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
- **Standard-trained**: normal teacher forcing (same as Lab 03/20)
- **Soft-trained**: soft input curriculum

Both are then evaluated with hard decoding and soft decoding, creating a 2×2 comparison:

| | Hard decoding | Soft decoding |
|---|---|---|
| **Standard-trained** | Baseline (Lab 03) | Lab 17's approach (mismatch) |
| **Soft-trained** | Does soft training help hard decoding? | Full approach (no mismatch) |

## What you learn here

- Why teacher forcing creates a train-test mismatch for soft decoding
- How scheduled sampling with continuous tokens closes the gap
- The curriculum approach: gradually shifting from discrete to soft inputs
- How to use detached forward passes for computing training signals
- The 2×2 experimental design for isolating the effect of soft training

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

Trains two models (standard and soft-trained) from identical initial weights, 1000 steps each. Then generates 20 names from each model with both hard and soft decoding, reporting entropy statistics. The code reuses the model and generation logic from Lab 17, and only the training loop is new.

## Why soft training matters

Soft thinking (Lab 17) is training-free but limited by the gap between what the model trained on (discrete tokens) and what it sees at inference (concept tokens). Soft training closes this gap by gradually teaching the model to work with continuous inputs. This is the same insight behind scheduled sampling, but applied to the continuous embedding space rather than discrete token sampling. The result: a model that's designed for soft inference from the ground up, not just adapted to it after the fact.
