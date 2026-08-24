# Understanding LLMs by Building One: Speculative Decoding

Same architecture as the PyTorch version (03/16), but with two model sizes, a small "draft" model and a larger "target" model, demonstrating speculative decoding. The draft model guesses multiple tokens ahead, the target model verifies them all in a single forward pass. The output distribution is mathematically identical to target-only generation, and the lab tests that rather than asserting it.

It cuts target forward passes per name from 5.0 to 2.1. It does **not** run faster on this CPU, 0.74x on an unloaded machine or about 26% slower, and the lab measures why: at this model size the target forward costs only about 2.5x the draft forward, so there is nothing worth amortizing. Forward-pass count is the metric that transfers to a GPU; wall clock at 100K parameters is not.

## Why this version exists

Autoregressive decoding is slow. Each token requires a full forward pass through the model, reading every weight from memory. But the GPU's compute units are barely used, waiting for data to arrive from slow memory. This is the fundamental bottleneck: decoding is **memory-bound**, not compute-bound.

Speculative decoding exploits this asymmetry: if the bottleneck is reading weights (which takes the same time whether you process 1 token or K tokens), have a cheap model draft K tokens and then verify them all in one target forward pass.

## What makes it interesting

### Why decoding is memory-bound

Consider a 7B parameter model generating one token:
- **Weights to read**: 7 billion × 2 bytes (FP16) = 14 GB
- **Compute per token**: 7 billion × 2 FLOPs = 14 GFLOPs
- **GPU memory bandwidth**: ~2 TB/s (A100)
- **GPU compute**: ~312 TFLOPS (A100)

Time to read weights: 14 GB / 2 TB/s = **7 ms**
Time to compute: 14 GFLOPs / 312 TFLOPS = **0.04 ms**

The GPU computes 175× faster than it can read data. It sits idle 99.4% of the time during decoding. This is why generating tokens one at a time is fundamentally wasteful.

### The speculative decoding algorithm

1. **Draft phase**: A small, fast model generates K candidate tokens autoregressively
2. **Verify phase**: The target model processes ALL tokens (original + K drafted) in one forward pass
3. **Accept/reject**: For each drafted token, compare draft probability q(x) with target probability p(x):
   - Accept with probability `min(1, p(x) / q(x))`
   - On rejection: sample from adjusted distribution `max(0, p(x) - q(x))` (normalized)
4. **If all K tokens accepted**: sample one bonus token from the target model's distribution

This acceptance/rejection scheme guarantees the output distribution is **exactly** the same as sampling from the target model alone. It's not an approximation. It's mathematically lossless.

The guarantee only holds if step 3 compares each drafted token against the *right* row of the target's output. That is easy to get wrong here: the models take at most `block_size = 16` tokens, so once the prefix plus the K drafted tokens is longer than the window, the front of the sequence falls off and row 0 of the target's output is no longer the first drafted token. The lab converts absolute token positions into row indices by subtracting what fell off (`row_offset`). Without that subtraction some drafted tokens near the length cap get accepted or rejected against another position's distribution, which breaks losslessness silently: the generated names still look fine, and only the acceptance statistics move.

### The acceptance rate tradeoff

The key metric is the **per-token acceptance rate**, usually written α: given that the target model looked at a drafted token, how often did it accept? Rules of thumb:
- α above ~80%: the draft closely tracks the target, so long accepted runs and a large speedup
- α below ~30%: the draft is too different, so almost every round rejects on the first token and the draft work is wasted
- The draft model should be 5-20× smaller than the target for the arithmetic to work out

In this lab the draft model (1 layer, 32-dim, 14,528 parameters) is **7.1× smaller** than the target (2 layers, 64-dim, 102,784 parameters), so it sits inside that 5-20× window.

**α is not "accepted ÷ drafted".** This distinction is easy to get wrong, and the lab prints all three numbers so the difference is visible:

```
  per-token acceptance rate alpha: 85.5%  (accepted / evaluated)
  mean accepted run per round:     2.26 tokens of K=4
  draft window utilisation:        56.5%  (accepted / all K proposals)
```

The draft proposes K=4 tokens per round, but the target stops examining the window at the first rejection, so tokens after that point are discarded without ever being evaluated. Dividing accepted tokens by *all K proposals* therefore measures how much of the draft window survived, which shrinks as you raise K no matter how good the draft model is. Dividing by tokens actually *evaluated* gives α, which is a property of the two models and does not move with K. Here that is the difference between 56.5% and 85.5%, large enough to change which rule of thumb you think you are in.

The third number, mean accepted run per round, is the one that predicts the speedup: 2.26 accepted tokens per target forward pass.

### High acceptance, and still no speedup

At α = 85.5% this lab is comfortably in the ">80%, large speedup" band, and it still runs **slower** than plain autoregressive decoding, at 0.74x on wall clock. That is not a contradiction; it is the second half of the tradeoff, which the acceptance rate alone does not capture. (The same run on a contended machine printed 1.50x, i.e. speculative decoding came out *faster*. A ratio that flips sign with background load is not measuring the algorithm.)

The standard model for the expected speedup is

```
speedup = (1 - α^(K+1)) / ((1 - α) · (K·c + 1))
```

where `c` is the cost of one draft forward pass as a fraction of one target forward pass. The lab measures `c` rather than assuming it:

```
measured cost per forward pass: draft 0.456 ms, target 1.136 ms
  -> target is 2.49x the draft's cost, against a 7.1x parameter ratio
```

Those absolute times move a lot with machine load: on a busy host the same measurement collapsed to 1.07x, because contention inflates the fixed per-call overhead both models pay. Take the unloaded number: 2.49x, so `c ≈ 0.40`. Plug that in with α = 0.855 and K = 4 and the model predicts about 1.44x, some headroom but nothing like the production figures, and all of it before any implementation overhead. Set `c = 0.05`, which is the regime a 1B draft against a 70B target lives in, and the same formula gives about 3.1x. That is where the production numbers come from.

Two things follow. First, the parameter ratio is not the cost ratio: 7.1x fewer parameters bought only about 2.5x less time here, because at this size both models are dominated by per-call Python and dispatch overhead rather than by arithmetic. Second, the measured 0.74x is below even the 1.44x the model predicts, because the model does not charge for the Python accept/reject loop, the tensor construction per round, or the repeated re-encoding of the prefix.

The transferable metric is therefore the **target forward pass count**: 2.1 per name speculatively against 5.0 autoregressively, a 2.36x reduction. On a GPU running a real model, the target forward is essentially the whole cost, so that reduction is the speedup. On this CPU it is not.

### Proving it is lossless

The lab claims the output distribution is *exactly* the target model's. A speedup measurement says nothing about that, so the lab tests it directly: 500 names sampled autoregressively from the target, 500 sampled speculatively, and a comparison of mean length and character-unigram frequency (total variation distance).

The part that makes the test meaningful is the noise floor. Two independent autoregressive runs also differ from each other, because each is a finite sample. Without that reference, "the distributions match" cannot fail. So the lab runs autoregressive sampling twice:

```
  run                        n  mean length
  autoregressive A         500        5.164
  autoregressive B         500        5.208
  speculative (K=4)        500        5.200

  character-unigram TV distance, AR vs AR:          0.0335   <- noise floor
  character-unigram TV distance, AR vs speculative:  0.0330
  ratio to noise floor: 0.99x
```

The speculative sample lands on the noise floor, a hair *closer* to autoregressive run A than run B is. That is what lossless looks like as a measurement: not a distance of zero, but a distance no larger than sampling error.

Two honest caveats. This compares two summary statistics, not the full joint distribution over strings, and it does so at n=500, so it can fail to detect a real difference, and it cannot prove there is none. And it is a check on *this implementation*; the proof that the accept/reject rule preserves the target distribution is in Leviathan et al. (2023).

### Verification is cheap

The magic of speculative decoding is that verification costs almost nothing extra. The target model already reads all its weights from memory for one token, so processing K+1 tokens instead of 1 barely changes the wall-clock time on a GPU, because the bottleneck is memory bandwidth, not compute.

At our tiny scale (CPU, Python loops) this overlap isn't visible, and the lab measures why: the target forward costs only about 2.5x the draft forward, so there is nothing much to amortize. On real hardware with billion-parameter models that ratio is 10-100x, and the overlap is where the speedup comes from.

## What you learn here

- Why autoregressive decoding is memory-bound (the key insight behind ALL inference optimization)
- How speculation + verification preserves output quality, and how to test that claim with a noise floor instead of asserting it
- What the acceptance rate α actually is, why "accepted ÷ drafted" is a different and K-dependent quantity, and why the difference matters
- That a high acceptance rate is necessary but not sufficient: the draft/target *cost* ratio is the other half, and it is not the parameter ratio
- Why the forward-pass count is the metric that transfers across hardware and the wall clock is not
- The acceptance/rejection sampling algorithm and why it's mathematically correct
- Why this is the #1 technique used in production inference systems

## What's not covered (but exists in practice)

- **EAGLE / EAGLE-2 / EAGLE-3** (Li et al., 2024-2025): Instead of a separate draft model, EAGLE uses the target model's own hidden states to predict future tokens. Achieves higher acceptance rates than separate draft models.
- **Medusa** (Cai et al., 2024): Adds multiple "heads" to the target model that predict tokens at different future positions simultaneously. No separate draft model needed.
- **Lookahead Decoding** (Fu et al., 2024): Uses Jacobi iteration to generate multiple token positions in parallel without any draft model.
- **Cascade inference**: Route "easy" tokens (high confidence) to a small model, "hard" tokens (low confidence) to a large model. Different from speculative decoding because it changes the output distribution.
- **Self-speculative decoding**: Use early layers of the target model as the draft model, skipping later layers for the draft phase.
- **Tree-based speculation**: Draft multiple candidate continuations (a tree, not a chain), verify entire branches at once. Used in SpecInfer and Sequoia.
- **vLLM / SGLang / TensorRT-LLM**: All implement speculative decoding as a first-class feature for production serving.
- **Key papers**: Leviathan et al. "Fast Inference from Transformers via Speculative Decoding" (ICML 2023), Chen et al. "Accelerating Large Language Model Decoding with Speculative Sampling" (2023).

## Run

```bash
uv run python main.py
```

Trains both models (draft: 14,528 params, target: 102,784 params), then generates samples using autoregressive decoding, speculative decoding, and draft-only decoding. It reports:

- Target forward passes per name: 2.1 speculative against 5.0 autoregressive, a 2.36x reduction
- Acceptance three ways: α = 85.5%, mean accepted run 2.26 of K=4, draft window utilisation 56.5%
- Measured per-forward-pass cost for each model, giving a cost ratio of 2.49x on an idle machine (and as little as 1.07x on a busy one)
- Wall clock: 0.74x, i.e. speculative decoding is slower here, with the reason spelled out
- A distributional check against an autoregressive-vs-autoregressive noise floor

## Why speculative decoding matters

Every major inference provider uses speculative decoding. It's the only technique that accelerates generation with **zero quality loss** because the output is statistically identical to the target model alone. Combined with other optimizations (FlashAttention, KV cache paging, continuous batching), it's how systems serve billions of tokens per day at acceptable latency.
