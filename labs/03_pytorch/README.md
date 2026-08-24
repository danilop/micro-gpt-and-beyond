# Understanding LLMs by Building One: PyTorch

Same architecture as versions 01 and 02, but now the autograd engine comes from a library rather than from the file you are reading. This is where you see what a framework adds once the differentiation itself is familiar.

## Why this version exists

Versions 01 and 02 build their own autograd, so `loss.backward()` here should already look familiar. What PyTorch adds is everything around it: layers that carry their own parameters, fused kernels, optimizers, device placement, and a backward pass that is someone else's job to keep correct.

## What makes it interesting

### nn.Module structure

The model is decomposed into clean, composable modules following the standard PyTorch pattern:

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

Version 02 writes the same architecture (RMSNorm, multi-head attention with causal mask, ReLU MLP, residual connections) as bare functions over arrays. Here each piece is an `nn.Module` that owns its weights.

### The same autograd, written by someone else

Version 02's engine is about 120 lines: a `Tensor` class with eleven primitives and a topological walk. PyTorch's is tens of thousands, spread over C++ kernels and dispatch layers, and the call site is unchanged:

```python
loss.backward()
```

PyTorch records every operation during the forward pass and applies the chain rule in reverse, exactly as version 02 does. The difference is what it records into: fused kernels, multiple dtypes, and any device you have.

### Weight initialization

The model matches the original's initialization, using `N(0, 0.08)` for all weights:

```python
@staticmethod
def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.08)


self.apply(self._init_weights)
```

A larger initial standard deviation (0.08 vs the GPT-2 default of 0.02) works well for this tiny model, giving parameters enough initial variance to learn distinct features quickly.

### Inference with torch.no_grad

Generation switches to `model.eval()` and `torch.no_grad()`, disabling dropout (if any) and skipping gradient tracking for efficiency:

```python
model.eval()
with torch.no_grad():
    for sample_idx in range(20):
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], device=device)
            logits = model(idx)
            logits = logits[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
```

`torch.multinomial` replaces `random.choices` for the same sampling logic, but on tensors.

## What you learn here

- How `nn.Module` organizes a transformer into composable pieces
- What a framework adds on top of an autograd engine you have already built yourself (02)
- PyTorch idioms: `F.cross_entropy`, `torch.multinomial`, parameter groups, LR scheduling
- Why the forward pass alone fully defines the model, since the backward pass is derived automatically

## Run

```bash
uv run python main.py
```

Trains for 1000 steps and generates 20 names. Same hyperparameters as versions 01 and 02.
