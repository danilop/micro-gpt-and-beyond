# Understanding LLMs by Building One: LoRA (Low-Rank Adaptation)

Parameter-efficient fine-tuning: freeze a trained model's weights, inject small low-rank matrices, and train only those. The base model never changes. You learn a tiny adapter that shifts its behavior.

## Why LoRA exists

Full fine-tuning of a 7B-parameter model means storing a complete copy of the model plus optimizer states, roughly 28 GB of memory just for Adam's momentum and variance. If you want to fine-tune for five different tasks, that's five full copies.

LoRA replaces this with adapters that are typically 0.1% the size of the base model. You can fine-tune on a single GPU and keep multiple task-specific adapters that swap in at serving time.

## The math

During fine-tuning, we don't update the original weight matrix W directly. Instead, we learn a low-rank decomposition of the update:

```
W' = W + delta_W = W + B @ A
```

Where W is `(d_out, d_in)`, A is `(r, d_in)`, and B is `(d_out, r)`. The rank `r` is much smaller than both `d_in` and `d_out`, typically 4, 8, or 16 even for models with dimensions in the thousands.

The forward pass becomes:

```python
def forward(self, x):
    return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T
```

The base weight is frozen. Only A and B are trainable.

## Why B is zero-initialized

B starts as all zeros, so the adapter's initial output is `B @ A @ x = 0`. The model begins fine-tuning from exactly its pre-trained behavior, with no random perturbation and stable training from step one. The adaptation builds up gradually as B learns non-zero values.

This is a deliberate design choice. If both A and B were randomly initialized, the adapter would immediately corrupt the pre-trained model's outputs, and training would need to first recover from that damage.

## Why it works

The LoRA paper (Hu et al., 2021) showed empirically that weight updates during fine-tuning have low intrinsic rank. When you fine-tune GPT-3 on a downstream task, the difference `W_finetuned - W_pretrained` can be well-approximated by a rank-4 matrix. The model doesn't need to change in all directions. It needs small adjustments along a few important dimensions.

This means we're not losing expressiveness by constraining the update to low rank. We're matching the natural structure of what fine-tuning actually does.

## What this lab demonstrates

The code runs in six phases:

1. **Pre-train** a standard microGPT on all names (1000 steps)
2. **Filter** the dataset to names starting with "m"
3. **Inject LoRA** adapters into all Linear layers in attention and MLP, then print total vs trainable parameter counts
4. **Fine-tune** with LoRA for 500 steps on the filtered subset, training only the adapter parameters
5. **Compare** generation from the base model vs the LoRA-adapted model, where the distribution shift is visible
6. **Merge** LoRA weights back into the base model and verify identical output

### LoRA injection

Every `nn.Linear` in the transformer blocks gets wrapped:

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=4):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
```

The optimizer only sees LoRA parameters:

```python
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(lora_params, ...)
```

### Merging at deployment

After fine-tuning, the adapter can be folded back into the base weight:

```python
merged.weight = nn.Parameter(child.base.weight + child.lora_B @ child.lora_A)
```

The merged model is structurally identical to the original: same size, same architecture, no extra layers. There is zero runtime overhead. This is one of LoRA's key advantages over other adapter methods.

## Multiple adapters

Because LoRA adapters are small and the base model stays frozen, you can train separate adapters for different tasks and swap them at serving time. One base model, many behaviors, where each adapter is just a pair of small matrices per layer.

## What you learn here

- How to freeze a model and inject trainable adapters without changing the architecture
- Why low-rank updates are sufficient for fine-tuning (empirical finding, not just a trick)
- The role of zero initialization in keeping training stable
- How to merge adapters back for deployment with no overhead
- That you can shift a model's behavior by training a tiny fraction of its parameters

## Run

```bash
uv run python main.py
```

Pre-trains for 1000 steps, fine-tunes with LoRA for 500 steps, then compares outputs.
