# Understanding LLMs by Building One: Speculative Decoding

Same architecture as the PyTorch version (03/16), but with two model sizes, a small "draft" model and a larger "target" model, demonstrating speculative decoding. The draft model guesses multiple tokens ahead, the target model verifies them all in a single forward pass. The output distribution is mathematically identical to target-only generation, but faster.

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

### The acceptance rate tradeoff

The key metric is **acceptance rate**, the fraction of drafted tokens the target model accepts:
- High acceptance rate (>80%): draft model closely matches target, yielding large speedup
- Low acceptance rate (<30%): draft model is too different, adding overhead without benefit
- The draft model should be 5-20× smaller than the target for practical speedup

In this lab, the draft model (1 layer, 32-dim) is ~8× smaller than the target (2 layers, 64-dim).

### Verification is cheap

The magic of speculative decoding is that verification costs almost nothing extra. The target model already reads all its weights from memory for one token, so processing K+1 tokens instead of 1 barely changes the wall-clock time on a GPU, because the bottleneck is memory bandwidth, not compute.

At our tiny scale (CPU, Python loops), this overlap isn't visible. On real hardware with billion-parameter models, it's transformative.

## What you learn here

- Why autoregressive decoding is memory-bound (the key insight behind ALL inference optimization)
- How speculation + verification preserves output quality (lossless acceleration)
- The acceptance rate tradeoff (better draft leads to more accepted tokens and more speedup)
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

Trains both models (draft: ~3K params, target: ~30K params), then generates samples using autoregressive decoding, speculative decoding, and draft-only decoding. Reports acceptance rate, forward pass counts, and speedup.

## Why speculative decoding matters

Every major inference provider uses speculative decoding. It's the only technique that accelerates generation with **zero quality loss** because the output is statistically identical to the target model alone. Combined with other optimizations (FlashAttention, KV cache paging, continuous batching), it's how systems serve billions of tokens per day at acceptable latency.
