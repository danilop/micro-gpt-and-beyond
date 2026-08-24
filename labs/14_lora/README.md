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
    return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
```

The base weight is frozen. Only A and B are trainable. `scaling` is `alpha / rank`, the magnitude knob from Hu et al. Section 4.1; it is 1.0 unless you pass an explicit `alpha`.

## Why B is zero-initialized

B starts as all zeros, so the adapter's initial output is `B @ A @ x = 0`. The model begins fine-tuning from exactly its pre-trained behavior, with no random perturbation and stable training from step one. The adaptation builds up gradually as B learns non-zero values.

This is a deliberate design choice. If both A and B were randomly initialized, the adapter would immediately corrupt the pre-trained model's outputs, and training would need to first recover from that damage.

## Why it works

The LoRA paper (Hu et al., 2021) showed empirically that weight updates during fine-tuning have low intrinsic rank. When you fine-tune GPT-3 on a downstream task, the difference `W_finetuned - W_pretrained` can be well-approximated by a rank-4 matrix. The model doesn't need to change in all directions. It needs small adjustments along a few important dimensions.

This means we're not losing expressiveness by constraining the update to low rank. We're matching the natural structure of what fine-tuning actually does.

## What this lab demonstrates

The centrepiece is a **rank ablation**: how small can the adapter get before it stops working?

The lab pre-trains a 4,192-parameter microGPT on all 32,033 names, then fine-tunes it towards a phonetic style, names whose consonants all come from `{m, n, r}`, so `emmerie`, `amena`, `normani`, `manami`. That filter leaves 1,092 of the 32,033 names. The style is easy to check by eye and easy to score automatically, which is what makes an ablation possible at all.

It then fine-tunes four different adapter configurations for 500 steps each, plus a full fine-tuning control, and scores each one on the fraction of 200 generated names that match the target style. Measured:

| Config | Trainable | % of model | scaling | Soft names (n=200) | Merge |
|---|---|---|---|---|---|
| base (no fine-tuning) | 0 | 0.0% | — | ~17% | — |
| **rank 1** | **64** | **1.5%** | 1.00 | **~85%** | ok |
| rank 2 | 128 | 3.0% | 1.00 | ~88% | ok |
| rank 4 | 256 | 5.8% | 1.00 | ~90% | ok |
| rank 4, alpha 16 | 256 | 5.8% | 4.00 | ~92% | ok |
| full fine-tuning | 512 | 12.2% | — | ~80% | n/a |

**Rank 1 already gets you there.** Sixty-four trainable parameters, one vector pair per adapted matrix at 1.5% of the model, take the style rate from roughly 17% to roughly 85%. Rank 2 and rank 4 add a few points for two and four times the adapter. Most of the distance is covered at r=1.

That is the LoRA paper's central empirical claim, reproduced small enough to read in one screen: the useful part of a fine-tuning update has low intrinsic rank, so a very small r goes a long way.

The last row is the control that makes the rest of the table mean something. Full fine-tuning of the same two matrices, same 500 steps, same learning rate, 512 trainable parameters and no low-rank constraint, reaches 80%. At 200 samples an 85% rate carries roughly ±2.5 points of sampling error, so the honest reading is "rank 1 is at least as good as full fine-tuning here", not "LoRA wins". Either way, "LoRA matches full fine-tuning" stops being a slogan and becomes something the lab checks.

### Where the adapters go

Not everywhere. The lab targets the query and value projections only:

```python
def inject_lora(module, rank, alpha=None, targets=("wq", "wv")):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank=rank, alpha=alpha, targets=targets)
```

`wk`, `wo`, and both MLP matrices keep their plain `nn.Linear` and stay frozen. This follows Hu et al., who found `wq` + `wv` sufficient for most tasks, and it is why the rank-1 adapter is 64 parameters rather than the 288 it would take to wrap every `nn.Linear` in the block.

The wrapper itself:

```python
class LoRALinear(nn.Module):
    def __init__(self, base_linear, rank=4, alpha=None):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scaling = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
```

`scaling` is `alpha / rank`. When `alpha` is left at `None` it defaults to `rank`, so scaling is exactly 1.0 and the term does nothing, which is the case for three of the four adapter rows. The `rank 4, alpha 16` row exists so the factor is actually exercised: scaling becomes 4.0, the adapter's contribution is multiplied by four, and the result moves. Without that row the docstring would be advertising a mechanism the lab never runs.

The freeze is not taken on trust either. Every configuration asserts it:

```python
assert frozen == base_total, f"freeze error: {frozen} != {base_total}"
```

If a single base weight were still trainable, the count would not match and the run would stop.

### The optimizer only sees the adapters

```python
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(lora_params, ...)
```

Because B is zero-initialized, step 0 of fine-tuning produces byte-identical output to the frozen base model. The adaptation grows from an exact no-op.

### Merging at deployment

After fine-tuning, the adapter folds back into the base weight:

```python
with torch.no_grad():
    merged.weight.copy_(child.base.weight + child.scaling * child.lora_B @ child.lora_A)
```

The merged model is structurally identical to the original: same size, same architecture, no extra layers, zero runtime overhead. The lab verifies this rather than asserting it: it regenerates from the merged model with the same seed and checks the names come out character-identical. Every row of the table above reports `ok`.

## Multiple adapters

Because LoRA adapters are small and the base model stays frozen, you can train separate adapters for different tasks and swap them at serving time. One base model, many behaviors, where each adapter is just a pair of small matrices per layer. At rank 1 in this lab an adapter is 64 numbers, so you could keep hundreds of them next to one 4,192-parameter base model.

## What you learn here

- How to freeze a model and inject trainable adapters without changing the architecture, and how to assert the freeze actually held
- Why low-rank updates are sufficient for fine-tuning: rank 1 gets 85% of the way where full fine-tuning gets 80%, at one eighth the trainable parameters
- How to run a rank ablation, and how to read one without over-reading sampling noise
- What the alpha/rank scaling factor does, by seeing a row where it is not 1.0
- The role of zero initialization in keeping training stable
- How to merge adapters back for deployment with no overhead, and how to verify the merge is exact
- That you can shift a model's behavior by training a tiny fraction of its parameters

## Run

```bash
uv run python main.py
```

Pre-trains for 1000 steps on all 32,033 names, then runs five 500-step fine-tunes on the 1,092-name filtered subset: LoRA at rank 1, 2 and 4, LoRA at rank 4 with alpha 16, and a full fine-tuning control on the same two matrices. Each one prints its adapter dimensions and parameter counts, 20 sample names, the style-match rate over 200 samples, and a merge check. The run ends with the ablation summary table.
