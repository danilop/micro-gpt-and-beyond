# Understanding LLMs by Building One: Self-Improving Model

This lab shows a filtered self-training loop: the model generates candidate
names, scores them with a verifier derived from real data, keeps only the best
outputs, and retrains on a mix of real plus self-generated names.

## Why this version exists

Naive self-training is unstable. If a model retrains on all of its own outputs,
it can quickly amplify mistakes and collapse toward repetitive garbage. This lab
adds the missing control loop: verify first, then learn only from the outputs
that clear the quality bar.

That makes it a compact, fully local analogue of:
- STaR, where only correct self-generated rationales are kept
- Self-rewarding models, where a judge filters outputs before reuse
- Karpathy's autoresearch loop, where only changes that improve the metric survive

## What makes it interesting

### The verifier is explicit

Instead of using another model as a judge, this lab builds a simple bigram
language model from the real names corpus and uses its log-probability as the
quality signal. That keeps the improvement loop inspectable: you can see
exactly why one generated name survives and another is discarded.

### Retraining is mixed, not replaced

The model does not jump to 100% self-generated data. Each round keeps most of
the training stream grounded in real names and mixes in only a controlled slice
of filtered self-generated examples. That is the stabilizing idea the lab is
trying to teach.

### It demonstrates both progress and failure modes

The script prints baseline samples, per-round quality, kept fraction, and final
outputs. It also explains why filtering helps but still does not fully solve
mode collapse: the verifier can still reward a narrow family of high-scoring
names unless you add diversity controls or rollback logic.

## What you learn here

- How to build a simple verifier from real data
- Why filtered self-training is safer than naive self-training
- How mix ratio, keep fraction, and verifier choice shape stability
- Why "self-improvement" can still collapse if the metric is too narrow

## How to run it

```bash
uv run python main.py
```

The output is organized into:
1. Initial supervised training on real names
2. Multiple self-improvement rounds
3. A before/after quality comparison
4. An explanation of the generate-score-filter-retrain loop

## Suggested experiments

- Remove the filter and retrain on every generated name
- Raise `MIX_RATIO` to make the loop more self-referential
- Tighten or loosen the keep fraction
- Replace the bigram verifier with a different heuristic and compare outcomes
