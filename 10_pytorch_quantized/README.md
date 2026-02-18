# microGPT and Beyond — PyTorch Quantized

Same architecture as the batched versions (04/06/08), but with inference-only quantization. This version shows how to compress a trained model from 32-bit floats (FP32) to 8-bit integers (INT8) — the technique used to deploy models on phones, edge devices, and production servers.

## Why this version exists

After training a model in PyTorch (version 03), you often want to deploy it somewhere with limited memory or compute. Quantization reduces model size by ~4× and speeds up inference by converting weights from FP32 to INT8. This is how real-world models run efficiently.

## What makes it interesting

### Training stays FP32

The model trains identically to version 03 — full 32-bit precision, Adam optimizer, 1000 steps. Quantization happens *after* training, so you don't sacrifice accuracy during learning.

### Dynamic quantization

Instead of using PyTorch's built-in quantization API (which has platform-specific requirements), this version implements a simple manual quantization scheme that works everywhere:

```python
class QuantizedLinear(nn.Module):
    def __init__(self, fp32_linear):
        super().__init__()
        weight = fp32_linear.weight.data
        scale = weight.abs().max() / 127.0
        weight_int8 = torch.round(weight / scale).to(torch.int8)
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('scale', torch.tensor(scale))
        
    def forward(self, x):
        weight_fp32 = self.weight_int8.to(x.dtype) * self.scale
        return F.linear(x, weight_fp32, self.bias)
```

This is symmetric quantization:
- Find the maximum absolute weight value
- Map the range `[-max, max]` to `[-127, 127]` (INT8 range)
- Store weights as INT8 + a single scale factor per layer
- Dequantize on the fly during forward pass

**Tradeoff**: Dequantizing on every forward pass adds overhead, but keeps memory usage low. Production systems use specialized INT8 matrix multiplication kernels that operate directly on INT8 without dequantization, achieving both memory savings and speed improvements.

### Model size comparison

The script measures model size and inference speed with a unified `benchmark()` function that saves the state dict, measures file size, and times sample generation.

For this model (~30,000 parameters), the results show:
- FP32: ~0.4 MB
- INT8: ~0.12 MB (~29% of original)

The reduction approaches 25% (4× compression) as the proportion of Linear layer parameters increases. Embeddings aren't quantized in this implementation, which is why it's not exactly 25%.

For a 7B parameter model, that's 28 GB → 7 GB — the difference between fitting in VRAM or not.

### Inference speed comparison

**Important**: This simple implementation may not show speed improvements because it dequantizes weights on every forward pass. The main benefit here is **memory savings** (~4× smaller).

Production quantization systems (TensorRT, ONNX Runtime, PyTorch native) use specialized INT8 kernels that operate directly on INT8 data, achieving both memory savings and 2-4× speed improvements on large models.

### Output quality

The script generates 10 samples from both FP32 and INT8 models. The outputs should be nearly identical — quantization introduces small numerical differences, but the model's behavior is preserved.

If outputs diverge significantly, it means the model is sensitive to precision. For production, you'd use more sophisticated quantization schemes:
- **Per-channel quantization**: Different scale factors per output channel (better accuracy)
- **Asymmetric quantization**: Map `[min, max]` to `[0, 255]` instead of symmetric `[-127, 127]`
- **Quantization-aware training (QAT)**: Fine-tune with quantization in the loop

## What you learn here

- The difference between training precision (FP32) and inference precision (INT8)
- How symmetric quantization works: mapping FP32 weights to INT8 with a scale factor
- Why model size matters for deployment (memory, bandwidth, storage)
- The tradeoff between compression and accuracy
- How to measure model size and inference speed in PyTorch
- How to implement quantization manually without framework-specific APIs

## What's not covered (but exists in practice)

- **PyTorch's quantization API**: `torch.quantization.quantize_dynamic` (platform-specific)
- **Static quantization**: Quantizes activations too (requires calibration data)
- **Quantization-aware training (QAT)**: Simulates quantization during training
- **Per-channel quantization**: Different scale factors per output channel
- **INT8 kernels**: Specialized implementations that operate directly on INT8
- **4-bit quantization**: GPTQ, GGUF, and other extreme compression schemes

This version focuses on the core idea: FP32 → INT8 weight compression for memory savings.

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, quantizes the model, compares size and speed, and generates 10 samples from each version. The output shows:
- Model size reduction (FP32 → INT8)
- Inference time comparison
- Sample quality comparison

## Why quantization matters

Modern LLMs have billions of parameters. A 7B model in FP32 is 28 GB — too large for most GPUs. In INT8, it's 7 GB — fits in consumer hardware. In 4-bit (not covered here), it's 3.5 GB — runs on a phone.

Quantization is the reason you can run Llama 3 on a laptop. This version shows the core idea at microGPT scale.
