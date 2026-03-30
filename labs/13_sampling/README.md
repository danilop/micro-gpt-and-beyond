# Understanding LLMs by Building One: Sampling Strategies

Same microGPT architecture as the PyTorch edition (03), trained identically, but focused entirely on what happens after training: how you turn logits into tokens. Five sampling strategies (greedy, temperature, top-k, top-p/nucleus, min-p), each producing completely different output from the same model.

## Why sampling matters

A language model outputs a probability distribution over the vocabulary at each step. The sampling strategy decides how to pick a token from that distribution. This single choice controls whether your model produces boring-but-safe output or creative-but-risky output. The model weights are fixed; the sampler is the knob.

## The five strategies

### Greedy

Always pick the highest-probability token. Deterministic: run it twice, get the same output. Tends to produce repetitive, generic results because it never explores lower-probability paths.

```python
return logits.argmax().item()
```

### Temperature

Scale logits by `1/T` before softmax. This reshapes the distribution without changing which tokens are available:

- **T < 1** (e.g., 0.3): sharpens the distribution, concentrating mass on top tokens. Approaches greedy as T approaches 0.
- **T = 1**: the model's learned distribution, unchanged.
- **T > 1** (e.g., 2.0): flattens the distribution, giving low-probability tokens more chance. Approaches uniform random as T approaches infinity.

```python
probs = F.softmax(logits / T, dim=-1)
return torch.multinomial(probs, 1).item()
```

Temperature is almost always combined with one of the filtering strategies below.

### Top-k

Keep only the k highest-probability tokens, set everything else to zero, renormalize, sample. A hard cutoff that prevents sampling from the long tail of unlikely tokens.

The problem: k is fixed regardless of how the distribution looks. If the model is very confident (one token at 95%), k=10 still keeps 10 tokens. If the model is uncertain (flat distribution), k=10 might cut off reasonable options.

```python
topk_vals, _ = torch.topk(scaled, k)
scaled[scaled < topk_vals[-1]] = float("-inf")
probs = F.softmax(scaled, dim=-1)
return torch.multinomial(probs, 1).item()
```

### Top-p (nucleus sampling)

Sort tokens by probability, keep the smallest set whose cumulative probability is at least p, zero out the rest. Unlike top-k, this adapts to the shape of the distribution:

- **Peaked distribution** (model is confident): few tokens needed to reach p, so few candidates.
- **Flat distribution** (model is uncertain): many tokens needed, so many candidates.

This is the key insight from Holtzman et al. (2020): the number of reasonable next tokens varies wildly depending on context.

```python
sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cumsum = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum - sorted_probs > p
sorted_probs[mask] = 0.0
```

### Min-p

Keep tokens whose probability is at least `min_p * max_probability`. The simplest adaptive method: a single relative threshold.

- **Peaked distribution**: max probability is high, threshold is high, few tokens pass.
- **Flat distribution**: max probability is low, threshold is low, many tokens pass.

Achieves similar adaptivity to top-p with a more intuitive parameter. Newer and gaining adoption in llama.cpp and related projects.

```python
probs = F.softmax(logits / T, dim=-1)
threshold = min_p * probs.max()
probs[probs < threshold] = 0.0
```

## Practical guidance

| Situation | Strategy | Why |
|---|---|---|
| Code generation | Greedy or T=0.1 | Correctness matters more than creativity |
| Chat / general use | Top-p=0.9, T=0.7 | Good balance of quality and variety |
| Creative writing | Top-p=0.95, T=1.0 | Allow surprising word choices |
| Brainstorming | Top-k=50, T=1.2 | Wide exploration, accept some noise |
| Simple and adaptive | Min-p=0.05, T=0.8 | One intuitive parameter, adapts automatically |

Most production systems combine temperature with either top-p or min-p. Temperature controls the overall sharpness; the filter controls the tail.

## What you learn here

- How the same model produces completely different outputs depending on the sampling strategy
- Temperature reshapes distributions without filtering; top-k/top-p/min-p filter without reshaping
- Top-k uses a fixed cutoff; top-p and min-p adapt to the distribution shape
- The tradeoff between quality (low temperature, tight filtering) and diversity (high temperature, loose filtering)
- Why practical systems combine temperature with a filtering strategy

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, then generates 20 names with each of 10 sampling configurations and visualizes how each filter reshapes the probability distribution for an example prefix.
