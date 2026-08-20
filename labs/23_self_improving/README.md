# Understanding LLMs by Building One: Self-Improving Model

This lab shows a filtered self-training loop: the model generates candidate
names, scores them with a verifier built from held-out real data, keeps only
the best outputs, and retrains on a mix of real plus self-generated names.

It also shows the loop failing, which is the more useful half. The quality
metric goes up and the model's diversity goes down at the same time, and both
numbers are printed side by side every round.

## Why this version exists

Naive self-training is unstable. If a model retrains on all of its own outputs,
it can amplify mistakes and collapse toward repetitive garbage. This lab adds
the missing control loop: verify first, then learn only from the outputs that
clear the quality bar.

That makes it a compact, fully local analogue of:
- STaR, where only correct self-generated rationales are kept
- Self-rewarding models, where a judge filters outputs before reuse
- Karpathy's autoresearch loop, where only changes that improve the metric survive

## What makes it interesting

### The verifier is explicit, and it does not read the training data

Instead of using another model as a judge, this lab builds a bigram language
model from 3,000 held-out names and uses its log-probability as the quality
signal. The model trains on the other 29,033. Two consequences:

- the quality score is not measuring memorization, because the verifier's names
  are ones the model never saw
- you can inspect exactly why one generated name survives and another is cut

The script also reports novelty: the fraction of generated names that do not
appear verbatim in the training corpus.

### Three arms, one experiment

The loop runs three times from identical starting weights with identical seeds:

| Arm | Keep fraction | Mix ratio | Question it answers |
|---|---|---|---|
| filtered | top 20% | 15% | does filtered self-training improve the metric? |
| unfiltered | 100% | 15% | does the naive loop amplify its own errors? |
| control | top 20% | 0% | would the same step budget on real data do as well? |

The control is the arm most self-improvement demos leave out. Without it, any
improvement could just be 1,800 extra training steps.

### The metric improves while the model collapses

Every round prints quality, unique fraction of 200 samples, the most-repeated
name with its count, and novelty. In a representative run the filtered arm
moved quality by **+0.3955** while the unique fraction fell from **96.0% to
46.5%**, with one name (`kanan`) accounting for **97 of 200 samples**. The
control reached only **+0.1572**, so the self-generated data really is doing
work on the metric. The metric is simply not measuring the thing that broke.

That is Goodhart's law in a file you can read in one sitting. The verifier
scores names one at a time, so it has no opinion about the distribution, and a
model that finds one high-scoring string and repeats it wins.

Note which arm collapsed. The folklore says the unfiltered loop is the
dangerous one, and at large mix ratios it is. But filtering is itself a
distribution-narrowing operation: taking the top 20% by a per-name score
deliberately throws away the tails. With only 15% of the retraining stream
coming from the model, the filtered arm is the one that concentrates. The
numbers your run prints may differ; the habit of printing all three columns is
the point.

### Retraining is mixed, not replaced

The model does not jump to 100% self-generated data. Each round keeps most of
the training stream grounded in real names and mixes in a controlled slice of
filtered self-generated examples. `MIX_RATIO` decides how many of the 300
retraining steps see a self-generated name (45 of 300 by default), which is
fewer than the 60 names the filter kept, so those 45 slots are sampled
uniformly from the whole kept set rather than taken in score order.

The per-round output prints how many of those slots contain *distinct* names.
In the run above that count goes **39, 28, 15, 5, 2, 4** while the slot count
stays at 45: the kept set itself has filled up with copies of the same winner.
That is the clearest single signal of collapse in the whole lab, and it costs
one call to `len(set(...))`.

## What you learn here

- How to build a verifier from data the model does not train on
- Why a rising metric is not evidence of a better model
- Why every self-improvement loop needs a same-budget control arm
- That filtering trades diversity for score, and you have to measure both
- How mix ratio, keep fraction, and verifier choice shape stability

## How to run it

```bash
uv run python main.py
```

Takes about 70 seconds. The output is organized into:
1. Initial supervised training on real names
2. Three arms of the self-improvement loop, with per-round diversity
3. A side-by-side comparison of quality and diversity across arms
4. Baseline and final sample names, with repeat counts
5. An explanation of what the numbers actually say

## Suggested experiments

- Raise `MIX_RATIO` toward 1.0 and watch the unfiltered arm start to collapse too
- Tighten `KEEP_TOP_FRACTION` to 0.05 and see the filtered arm collapse faster
- Deduplicate `kept` before retraining and compare the diversity curves
- Add a penalty for repeats to the verifier, so the metric can see collapse
- Add revert-on-regression: discard a round whose diversity fell
- Shrink `HELDOUT_SIZE` and see how a noisier verifier changes the loop
