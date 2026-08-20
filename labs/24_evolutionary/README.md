# Understanding LLMs by Building One: Evolutionary Self-Improvement

This lab improves the model itself rather than its outputs. A population of tiny
GPT variants trains briefly, competes on validation loss, and reproduces by
mutating the best configurations.

It also does the bookkeeping that makes the result believable: a best-ever
archive, a compute ledger, and a diversity count per generation.

## Why this version exists

Lab 23 improves the training data. This lab improves the training recipe and the
model shape. The two labs together show the two main axes of self-improvement:
- better examples to learn from
- better configurations to learn with

The loop is inspired by Population Based Training (PBT): evaluate many model
variants, keep the strongest, perturb them, and repeat. The implementation here
is intentionally small and explicit, so you can follow the selection pressure
without hidden orchestration.

## What makes it interesting

### The search space is concrete

The lab evolves a handful of hyperparameters that matter immediately in a tiny
GPT: embedding size, number of heads, number of layers, learning rate, and
Adam's beta1. That keeps the tradeoffs visible instead of burying them in a
large tuning framework.

### Exploit and explore happen in code you can read

Survivors are deep-copied into the next generation. Children mutate one or two
hyperparameters. If the architecture is unchanged, the child inherits the
parent's weights; otherwise it starts fresh. That is the core exploit/explore
split in a form that is easy to inspect.

### The best model of the run is archived, not assumed

Per-generation best in a representative run: 2.5289, 2.5000, 2.5118, **2.4962**,
2.5326, 2.5367. The search peaks at generation 4 and then regresses, so the best
member of the final population (2.5321) is *not* the best model found. A
best-ever archive keeps the peak, and the lab reports from the archive.

### The compute ledger

Evolution spends `POP_SIZE × NUM_GENERATIONS × STEPS_PER_GEN` plus the final
children's training: **10,600 steps** against the single baseline's **500**.
That is 21x the compute, so "evolved beats baseline by +0.2259" is mostly a
statement about budget. The lab therefore prints three numbers:

| Comparison | Measured | What it means |
|---|---|---|
| evolved vs 500-step baseline | +0.2259 | not a fair comparison, 21x the compute |
| config advantage at matched budget | +0.0817 | both configs from scratch, val loss every 200 steps to 1,200 |
| evolved vs equal-budget random search | +0.0328 | what selection and mutation bought over 48 random configs |

The configuration evolution found is genuinely better. It is better by about
0.08, not 0.23.

### Cumulative steps are printed next to val loss

Every generation lists each member's val loss *and* the total training its
weights have received. Survivors accumulate steps across generations while
children with a new architecture restart at zero, so a generation ranking is
not a clean configuration comparison. Each generation grants every member the
same number of *new* steps, which is not the same as the same total training.
The `steps` column is there so you can see the confound instead of assuming it
away.

### Diversity dies, and the lab says so

Distinct architectures per generation in the same run: 7, 6, 4, 3, 3, 4, and
every member of the final population has the same parameter count (7,264). Once
the population is one size, mutation is only shuffling learning rate and head
count. The population average also degrades (2.5702 to 2.5717 across six
generations), because every fresh architecture pays a restart cost. Neither of
those facts is fatal to the lesson, but a lab that printed only the best-of-run
number would hide both.

### It is deliberately simpler than full PBT

The docstring is honest about the simplifications: no optimizer-state carryover,
no continuous schedule perturbations, and no true parallel training. That makes
the algorithm smaller, but it still teaches the selection-and-mutation pattern
used in larger search systems.

## What you learn here

- How to encode a hyperparameter search space
- Why validation loss, not training loss, must drive selection
- Why a search needs a best-ever archive, not just a final population
- How to compare a search result honestly against equal compute
- How weight inheritance changes the economics of exploration, and how it
  confounds within-generation fitness
- Why diversity has to be measured, not hoped for
- Where this toy loop differs from canonical PBT

## How to run it

```bash
uv run python main.py
```

Takes a couple of minutes; the equal-budget random-search control is the
expensive part (48 configs, many of them larger than the ones evolution
converges to). The script prints:
1. A single random baseline run at 500 steps
2. Six generations of evolutionary search, with per-member cumulative steps
3. Best/average/worst validation loss and diversity counts by generation
4. The compute ledger: matched-budget curves and a random-search control
5. A final comparison with all three deltas

## Suggested experiments

- Increase `POP_SIZE` and see whether diversity survives longer
- Add a mutation floor: reject a child whose genome duplicates an existing member
- Replace one member per generation with a random immigrant and re-measure
- Restrict the search space to show how quickly evolution saturates
- Add partial weight inheritance when `n_embd` changes
- Give survivors no free ride: reset weights every generation and see what
  happens to the ranking
- Track how often each hyperparameter value survives into later generations
