# Understanding LLMs by Building One: PyTorch Quantized

Inference-only quantization: compress a trained model from 32-bit floats (FP32) to 8-bit integers (INT8), the technique used to deploy models on phones, edge devices, and production servers.

The dimensions come from version 04 (64-dim embeddings, 4 heads, 2 layers, 102,784 parameters), but the training loop here is the unbatched one from version 03: one name per step, no padding and no attention mask over padded positions. If you are looking for the batched trainer, that is 04/06/08, not this lab.

## Why this version exists

After training a model in PyTorch (version 03), you often want to deploy it somewhere with limited memory or compute. Quantization shrinks the weights by storing them as INT8 plus a scale factor instead of FP32.

Be clear about what this lab does and does not show. It measures a real size reduction: roughly 0.4 MB down to roughly 0.11 MB, about 3.5x smaller. It does **not** show a speedup, and it is not supposed to: the forward pass here dequantizes INT8 back to FP32 and then runs an ordinary FP32 matmul, so INT8 comes out slower than FP32. How much slower is a property of the machine and the moment, not a constant: on the 2-core CPU container this was last measured on, INT8 came out roughly a quarter to a third slower than FP32 across a dozen quiet runs. Read the numbers your own run prints, not these. Speed requires INT8 GEMM kernels that do the arithmetic in integers, which this lab deliberately does not implement. Size savings are the honest result; the speed claim belongs to TensorRT and friends.

## What makes it interesting

### Training stays FP32

The model trains identically to version 03: full 32-bit precision, Adam optimizer, 1000 steps. Quantization happens *after* training, so you don't sacrifice accuracy during learning.

### Dynamic quantization

Instead of using PyTorch's built-in quantization API (which has platform-specific requirements), this version implements a simple manual quantization scheme that works everywhere:

```python
class QuantizedLinear(nn.Module):
    def __init__(self, fp32_linear):
        super().__init__()
        weight = fp32_linear.weight.data
        scale = weight.abs().max() / 127.0
        weight_int8 = torch.round(weight / scale).to(torch.int8)
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        weight_fp32 = self.weight_int8.to(x.dtype) * self.scale
        return F.linear(x, weight_fp32, self.bias)
```

This is symmetric quantization:
- Find the maximum absolute weight value
- Map the range `[-max, max]` to `[-127, 127]` (INT8 range)
- Store weights as INT8 + a single scale factor per layer
- Dequantize on the fly during forward pass

**Tradeoff**: Dequantizing on every forward pass adds overhead, but keeps memory usage low. This is measured, not hypothetical: the lab prints the ratio it observed, and on the machine used to write this that landed between +13% and +38% depending on the run. Production systems use specialized INT8 matrix multiplication kernels that operate directly on INT8 without dequantization, achieving both memory savings and speed improvements.

### Model size comparison

The script measures model size and inference speed with a unified `benchmark()` function that saves the state dict, measures file size, warms the model up, and times sample generation. Both timing runs reseed the sampler immediately before generating, so FP32 and INT8 draw the same sampling decisions and therefore run the same number of forward passes on the same-length sequences. Without that, the "comparison" would be comparing two different workloads. The warmup matters for the same reason: FP32 is timed first, so without it FP32 pays for one-time setup (thread pool, kernel selection, cold caches) that INT8 does not.

For this model (102,784 parameters), the measured results:
- FP32: about 0.4 MB
- INT8: about 0.11 MB, a bit under 30% of the original, so roughly 3.5x smaller

The reduction approaches 25% (4× compression) as the proportion of Linear layer parameters increases. Embeddings aren't quantized in this implementation, which is why it's not exactly 25%.

For a 7B parameter model, that's 28 GB to 7 GB, the difference between fitting in VRAM or not.

### Quantization error

Rounding weights onto a 256-value grid is lossy, so the lab measures how lossy. For each quantized layer it prints `max|W - dequant(W)|` next to the layer's scale:

```
  layer                     max|W|      scale max abs err  as % of max|W|
  layers.0.attn.wq            ~0.65   ~5.1e-03    ~2.6e-03           ~0.39%
  ...
  lm_head                     ~0.81   ~6.4e-03    ~3.2e-03           ~0.39%
```

Every layer lands at the same percentage of its own largest weight, and that is not a coincidence. Symmetric per-tensor quantization maps `[-max|W|, +max|W|]` onto `[-127, +127]`, so the grid step is `scale = max|W|/127` and rounding can be off by at most `scale/2 = max|W|/254`, which is about 0.39%. The measurement confirms the bound rather than discovering something new, which is exactly what you want from an error check.

The reason to state it as a percentage of `max|W|` is that this is per-tensor quantization's weakness: one outlier weight stretches the grid for every other weight in the same tensor. Per-channel quantization exists because of this, and LLM.int8() exists because at transformer scale the outliers are much worse than they are here.

Weight error only matters if it reaches the output, so the lab also measures per-token cross-entropy on 2,000 names the model never trained on:

```
    FP32: ~2.406
    INT8: ~2.407  (about +0.001, roughly +0.04%)
```

A loss increase that small, for roughly 3.5x less memory, is the whole argument for quantization in one line.

### Inference speed comparison

**Important**: this implementation does not just fail to speed up, it measurably slows down. Dequantizing on every forward pass costs more than the FP32 baseline. On the 2-core CPU container used to write this, a dozen quiet runs put INT8 roughly a quarter to a third slower than FP32.

Treat that spread as the result, and treat it as one machine's, not a law. This is a 100K-parameter model on CPU timed for a few seconds, so the ratio is noisy: a loaded machine roughly doubled the gap, and one cold first-ever run in a fresh container reported INT8 as *faster*, because the FP32 arm is timed first and absorbed the process's warm-up cost (which is why `benchmark()` now warms each model up before timing). What is structural is the direction, since an extra dequantize per matmul cannot be free. The benefit here is entirely **memory**: roughly 3.5x smaller.

Production quantization systems (TensorRT, ONNX Runtime, PyTorch native) use specialized INT8 kernels that operate directly on INT8 data, achieving both memory savings and 2-4× speed improvements on large models.

### Output quality

The script generates 10 names from each model off the same seed, listed side by side, and it does this for **two** seeds. That second seed is the point of the section.

```
--- samples, seed 42: 0 of 10 differ ---
   #  FP32          INT8
   1  alilen        alilen
   2  nanea         nanea
   ...

--- samples, seed 1: 3 of 10 differ ---
   #  FP32          INT8
   1  an            an
   2  alena         alaha         <- differs
   3  narera        nare          <- differs
   4  alys          alaryn        <- differs
   ...
```

Seed 42 agrees on all ten, and it would be easy to stop there and conclude that INT8 is free. Seed 1 shows what that conclusion is worth. Nothing differs between the two runs except weights that moved by about 0.39% of their layer's maximum; that is enough to push three of ten `torch.multinomial` draws across a boundary and send the rest of the name somewhere else.

So the lab sweeps seeds and reports the rate, which is the number worth quoting:

```
  across 40 seeds x 25 names: ~165/1000 differ (~16.5%)
```

Roughly one name in six. Matching output at a single seed is a small-sample accident, not a property of INT8, and the script asserts the sweep stays in a wide band so the claim cannot quietly rot. The held-out loss delta above is the measure that does not depend on which seed you happened to print.

Narrowing that gap is what the schemes this lab skips are for:
- **Per-channel quantization**: Different scale factors per output channel (better accuracy)
- **Asymmetric quantization**: Map `[min, max]` to `[0, 255]` instead of symmetric `[-127, 127]`
- **Quantization-aware training (QAT)**: Fine-tune with quantization in the loop

## What you learn here

- The difference between training precision (FP32) and inference precision (INT8)
- How symmetric quantization works: mapping FP32 weights to INT8 with a scale factor
- Why model size matters for deployment (memory, bandwidth, storage)
- How to quantify quantization error, both on the weights (`max|W - dequant(W)|`, and why it equals `max|W|/254`) and on the model's output (held-out loss)
- Why dequantize-to-FP32 quantization saves memory but costs speed, and what a real INT8 kernel would change
- How to set up a controlled timing comparison (same seed, same workload for both models)
- Why one seed is not a measurement: identical samples at seed 42 and 16% divergence across 40 seeds are the same model
- How to implement quantization manually without framework-specific APIs

## What's not covered (but exists in practice)

- **PyTorch's quantization API**: `torch.quantization.quantize_dynamic` (platform-specific)
- **Static quantization**: Quantizes activations too (requires calibration data)
- **Quantization-aware training (QAT)**: Simulates quantization during training
- **Per-channel quantization**: Different scale factors per output channel
- **INT8 kernels**: Specialized implementations that operate directly on INT8
- **4-bit quantization**: GPTQ, GGUF, and other extreme compression schemes

This version focuses on FP32 to INT8 weight compression for memory savings.

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, quantizes the model, then reports:
- Model size reduction: roughly 0.4 MB to roughly 0.11 MB, about 3.5x
- Inference time, seeded identically and warmed up for both models: INT8 slower (+13% to +38% across runs on the machine this was written on, but read your own run)
- Per-layer quantization error, `max|W - dequant(W)|`, about 0.39% of `max|W|` everywhere
- Held-out per-token loss on 2,000 unseen names: FP32 and INT8 within about +0.001 of each other
- 10 names from each version at two seeds, side by side: one seed where they agree, one where they do not
- Sample divergence swept over 40 seeds: roughly 16% of names differ between FP32 and INT8

## Why quantization matters

Modern LLMs have billions of parameters. A 7B model in FP32 is 28 GB, too large for most GPUs. In INT8, it's 7 GB, which fits in consumer hardware. In 4-bit (not covered here), it's 3.5 GB, small enough to run on a phone.

Quantization is the reason you can run Llama 3 on a laptop. This version shows the core idea at microGPT scale.
