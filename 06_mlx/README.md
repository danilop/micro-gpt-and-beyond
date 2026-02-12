# microGPT and Beyond — MLX (Apple Silicon)

Same architecture, running on Apple Silicon GPU via [MLX](https://ml-explore.github.io/mlx/). MLX is Apple's array framework for machine learning — it has a NumPy-like API, automatic differentiation, and runs natively on the M-series GPU.

## Why this version exists

If you have a Mac with Apple Silicon, this version trains on the GPU with zero configuration. No CUDA, no driver installs, no `device='cuda'` — arrays live in unified memory that both CPU and GPU can access directly.

## What makes it interesting

### Unified memory — no transfers

In PyTorch, you move tensors between CPU and GPU with `.to(device)`. In MLX, there's no transfer because CPU and GPU share the same memory:

```python
input_ids = mx.array(tokens[:n])   # lives in unified memory
logits = model(input_ids)           # computed on GPU
loss_val = loss_val.item()          # read on CPU — no copy needed
```

This is a fundamental hardware difference on Apple Silicon, and MLX is designed around it.

### Lazy evaluation

MLX doesn't compute anything until you ask for a result. Operations build a computation graph, and `mx.eval()` triggers actual execution:

```python
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)  # now it actually runs
```

This lets MLX fuse operations and optimize the computation graph before hitting the hardware. The `mx.eval` call at the end of each training step forces evaluation so the loss value is ready to print.

### nn.value_and_grad

MLX's differentiation API is clean — `nn.value_and_grad` returns both the loss and the gradients in one call:

```python
loss_and_grad = nn.value_and_grad(model, loss_fn)

# In the training loop:
loss_val, grads = loss_and_grad(model, input_ids, targets)
```

This is more efficient than computing the loss and gradients separately, and it's the idiomatic MLX pattern.

### Module structure

MLX modules use `__call__` instead of `forward`, and layers are stored as plain Python lists (not `nn.ModuleList`):

```python
class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = [Block() for _ in range(n_layer)]  # plain list
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
```

MLX's module system is lighter than PyTorch's — it inspects the object's attributes to find parameters, so you don't need special container types.

### Squared ReLU without F.relu

MLX doesn't have a standalone `relu` function in the same way. Instead, you use `mx.maximum`:

```python
def __call__(self, x):
    h = self.fc1(x)
    h = mx.maximum(h, 0) ** 2  # squared ReLU
    return self.fc2(h)
```

Small difference, same math.

## What you learn here

- How unified memory changes the programming model (no CPU↔GPU transfers)
- Lazy evaluation and explicit `mx.eval()` for controlling when computation happens
- MLX's module and optimizer patterns — similar to PyTorch but with key differences
- What a "native Apple Silicon" ML framework looks like in practice

## Run

Requires a Mac with Apple Silicon (M1/M2/M3/M4).

```bash
uv run python main.py
```

Trains for 500 steps and generates 20 names. Runs on the GPU automatically.
