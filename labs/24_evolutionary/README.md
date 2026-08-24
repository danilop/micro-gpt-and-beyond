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

Per-generation best in a representative run rises and falls rather than
descending: it bottoms out at generation 2 and drifts back up afterwards, so the
best member of the final population is *not* the best model found: the winner is a generation-2 model with only 400 steps on it. A
best-ever archive keeps the peak, and the lab reports from the archive.

### The reported number is selected on the number reported

There is a bias in the headline that the ledger below does not fix. Fitness is
validation loss on the first 200 names of the validation split. Selection,
mutation, the best-ever archive and the random-search control all rank models by
that one number, and then the lab reports that same number as the quality of
the winner. The reported loss is a minimum over dozens of noisy evaluations of
the same statistic, so it is optimistically biased: part of the gap to the
baseline is genuine configuration quality, and part of it is the luckiest draw
of evaluation noise, which is exactly what taking a minimum selects for. The
more models the search evaluates, the better that number looks even if no
configuration is any better than the others. The random-search arm is a minimum
over its own 48 evaluations too, so *that* comparison is at least symmetric; the
comparison against the single baseline run is not.

The clean version costs almost nothing here. Split three ways: train, a search
split that fitness scores against, and a final split nothing in the loop ever
sees, and report the winner on the third. This lab is one line away from it
already: `val_size` is 1,000 names but fitness only ever reads the first
`VAL_SAMPLES` = 200, so the remaining 800 are untouched by the search and would
serve as an honest held-out set for the winner. Until that number is printed,
read every validation loss in this lab as a ranking signal, not as a measurement
of how good the model is.

### The compute ledger

Evolution spends `POP_SIZE × NUM_GENERATIONS × STEPS_PER_GEN` plus the final
children's training: **10,600 steps** against the single baseline's **500**.
That is 21x the compute, so the headline "evolved beats baseline" is mostly a
statement about budget. The lab therefore prints three numbers:

| Comparison | Roughly | What it means |
|---|---|---|
| evolved vs 500-step baseline | ~0.27 better | not a fair comparison, 21x the compute |
| config advantage at matched budget | ~0.11 better | both configs from scratch, val loss every 200 steps to 1,200 |
| evolved vs equal-budget random search | ~0.08 better | what selection and mutation bought over 48 random configs |

The configuration evolution found is genuinely better. It is better by about
0.1, not 0.3.

The matched-budget delta is a best-of-curve difference, and the curves it comes
from do not descend cleanly:

```
          config      200      400      600      800     1000     1200
        baseline     ~2.64    ~2.58    ~2.68    ~2.58    ~2.59    ~2.77
         evolved     ~2.54    ~2.48    ~2.52    ~2.47    ~2.48    ~2.48
```

Both wobble, and the baseline ends at its worst checkpoint. That is not
overfitting: each 200-step chunk reads a *fresh* slice of `train_docs`, so 1,200
steps is 1,200 distinct names out of 31,033 and the model has seen 4% of the
training set once. It is measurement noise from a training loop that takes one
gradient step per example, plus a fresh Adam state at the start of every chunk,
which throws away the moment estimates and briefly destabilises the model each
time. Taking endpoints instead of minima would turn that noise into a result, so
the lab compares the best point on each curve.

### Cumulative steps are printed next to val loss

Every generation lists each member's val loss *and* the total training its
weights have received. Survivors accumulate steps across generations while
children with a new architecture restart at zero, so a generation ranking is
not a clean configuration comparison. Each generation grants every member the
same number of *new* steps, which is not the same as the same total training.
The `steps` column is there so you can see the confound instead of assuming it
away.

### Diversity dies, and the lab says so

Distinct architectures per generation in the same run: 7, 6, 4, 3, 4, 2, and
every member of the final population has the same parameter count (14,528). Once
the population is one size, mutation is only shuffling learning rate and head
count. The population average also degrades slightly across the six generations,
because every fresh architecture pays a restart cost. Neither of
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
- Why the winner's fitness score is not the winner's quality: a minimum over many noisy evaluations of the same statistic is biased low, and the fix is a split the search never scored against
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
