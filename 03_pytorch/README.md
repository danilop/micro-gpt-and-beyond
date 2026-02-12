# 03 — PyTorch

Same architecture as versions 01 and 02, but now PyTorch handles the tensors *and* the gradients. This is where you see how much boilerplate disappears when you let a framework do the differentiation.

## Why this version exists

After writing every gradient by hand in version 02, this version shows what you get in return for adopting PyTorch: the same model, the same training loop, the same results — but the entire backward pass is replaced by a single call to `loss.backward()`.

## What makes it interesting

### nn.Module structure

The model is decomposed into clean, composable modules — the standard PyTorch pattern:

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

Compare this to the 150-line forward+backward in version 02. The architecture is identical — RMSNorm, multi-head attention with causal mask, squared ReLU MLP, residual connections — but expressed declaratively.

### Autograd replaces hand-written gradients

The entire backward pass from version 02 (RMSNorm backward, softmax backward, attention backward, MLP backward, embedding gradients) collapses to:

```python
loss.backward()
```

PyTorch records every operation during the forward pass and automatically applies the chain rule in reverse. The gradients it computes are mathematically identical to the ones we wrote by hand.

### Weight initialization

The model matches the original's initialization exactly — `N(0, 0.02)` for all weights, with output projections (`wo`, `fc2`) explicitly zeroed:

```python
self.apply(self._init_weights)
for layer in self.layers:
    nn.init.zeros_(layer.attn.wo.weight)
    nn.init.zeros_(layer.mlp.fc2.weight)
```

Zero-initializing the output projections means each layer starts as an identity function (the residual connection passes through unchanged). This is a common trick for stable training of deep transformers.

### Inference with torch.no_grad

Generation switches to `model.eval()` and `torch.no_grad()` — disabling dropout (if any) and skipping gradient tracking for efficiency:

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

`torch.multinomial` replaces `random.choices` — same sampling, but on tensors.

## What you learn here

- How `nn.Module` organizes a transformer into composable pieces
- The relationship between manual gradients (02) and autograd (this version)
- PyTorch idioms: `F.cross_entropy`, `torch.multinomial`, parameter groups, LR scheduling
- Why the forward pass alone fully defines the model — the backward pass is derived automatically

## Run

```bash
uv run python main.py
```

Trains for 500 steps and generates 20 names. Same hyperparameters as versions 01 and 02.
