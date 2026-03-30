# Understanding LLMs by Building One: Evolutionary Self-Improvement

This lab improves the model itself rather than its outputs. A population of tiny
GPT variants trains briefly, competes on validation loss, and reproduces by
mutating the best configurations.

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

### It is deliberately simpler than full PBT

The docstring is honest about the simplifications: no optimizer-state carryover,
no continuous schedule perturbations, and no true parallel training. That makes
the algorithm smaller, but it still teaches the selection-and-mutation pattern
used in larger search systems.

## What you learn here

- How to encode a hyperparameter search space
- Why validation loss, not training loss, must drive selection
- How weight inheritance changes the economics of exploration
- Where this toy loop differs from canonical PBT

## How to run it

```bash
uv run python main.py
```

The script prints:
1. A single random baseline run
2. Six generations of evolutionary search
3. Best/average/worst validation loss by generation
4. A final comparison between the baseline and the evolved winner

## Suggested experiments

- Increase `POP_SIZE` and see how much diversity you gain per generation
- Restrict the search space to show how quickly evolution saturates
- Add partial weight inheritance when `n_embd` changes
- Track how often each hyperparameter value survives into later generations
