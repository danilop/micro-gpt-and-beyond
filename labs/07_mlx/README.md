# Understanding LLMs by Building One: MLX (Apple Silicon)

Same architecture, running on Apple Silicon GPU via [MLX](https://ml-explore.github.io/mlx/). MLX is Apple's array framework for machine learning. It has a NumPy-like API, automatic differentiation, and runs natively on the M-series GPU.

## Why this version exists

If you have a Mac with Apple Silicon, this version trains on the GPU with zero configuration. No CUDA, no driver installs, no `device='cuda'`. Arrays live in unified memory that both CPU and GPU can access directly.

## What makes it interesting

### Unified memory without transfers

In PyTorch, you move tensors between CPU and GPU with `.to(device)`. In MLX, there's no transfer because CPU and GPU share the same memory:

```python
input_ids = mx.array(tokens[:n])  # lives in unified memory
logits = model(input_ids)  # computed on GPU
loss_val = loss_val.item()  # read on CPU, no copy needed
```

Unified memory is a fundamental hardware difference on Apple Silicon, and MLX is designed around it.

### Lazy evaluation, and what `mx.eval` is really for

MLX doesn't compute anything until you ask for a result. Operations build a computation graph, and `mx.eval()` triggers actual execution:

```python
optimizer.update(model, grads)
mx.eval(state)  # state = [model.state, optimizer.state]
```

It is easy to get the reason for that line wrong. It is *not* needed to print the loss: `loss_val.item()` already forces the loss to be computed, because you cannot read a Python float out of a graph. What `.item()` does not force is the parameter update or the optimizer moments, which nothing downstream reads.

So without an explicit `mx.eval`, each step would append another update to a graph that nobody ever evaluates, and it would grow for as long as training runs, until scheduling and memory dominate. The call bounds the graph at one step. That is the whole reason, and it is the same reason `08_mlx_batched` calls it.

### mx.compile, MLX's answer to jax.jit

Lazy evaluation on its own only defers work. `mx.compile` is what turns the deferred graph into fused kernels, exactly as `jax.jit` does:

```python
state = [model.state, optimizer.state]


def train_step(input_ids, targets):
    loss_val, grads = loss_and_grad(model, input_ids, targets)
    optimizer.update(model, grads)
    return loss_val


train_step = mx.compile(train_step, inputs=state, outputs=state)
```

`inputs` and `outputs` declare the state the step reads and writes but does not take as arguments, the model parameters and the optimizer moments. Without them, the compiled function would capture stale values.

And like `jit`, `mx.compile` specialises on input shapes. Here `n = min(block_size, len(tokens) - 1)` changes with the name, so the loop separates the two kinds of step and prints both averages:

```
distinct sequence lengths seen: 12 (compile = True)
  first-time-shape steps:     ~4.9 ms mean
  repeated-shape steps:       ~1.2 ms mean
```

Set `use_compile = False` at the top of the training section and run it again. On the machine this was written on the two numbers converged, at roughly 4 ms and 3 ms: without compilation there is nothing special about a new shape, and every step costs around three times as much.

So compilation is worth it here, and the tracing bill is a few milliseconds per distinct shape rather than the seconds `05_jax` pays for XLA. Still, the shape sensitivity is the same lesson, and the same fix applies: pad to a fixed length and there is only one program to compile, as `06_jax_batched` does.

### nn.value_and_grad

MLX's differentiation API is clean. `nn.value_and_grad` returns both the loss and the gradients in one call:

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

MLX's module system is lighter than PyTorch's. It inspects the object's attributes to find parameters, so you don't need special container types.

That lightness is why re-initialising every weight is one line:

```python
model.apply(lambda w: mx.random.normal(w.shape) * 0.08)
```

`nn.Module.apply` maps a function over every parameter array and updates the module in place. The same hook is how you would cast a model to `float16` or quantise it.

### ReLU via mx.maximum

This lab writes ReLU as `mx.maximum(h, 0)`:

```python
def __call__(self, x):
    h = self.fc1(x)
    h = mx.maximum(h, 0)  # ReLU
    return self.fc2(h)
```

MLX does ship `mlx.nn.relu` and `mlx.nn.ReLU`, so this is a style choice, not a workaround: at this size the explicit `maximum` makes it obvious there is no hidden behaviour, which is the point of these labs. Use `nn.relu` in real code if you prefer.

## What you learn here

- How unified memory changes the programming model (no CPU-to-GPU transfers)
- Lazy evaluation, and the real job of `mx.eval()`: bounding the pending graph, not fetching values
- `mx.compile` as MLX's `jit`, including its per-shape specialisation, measured in ms/step
- MLX's module and optimizer patterns, which are similar to PyTorch but with key differences
- What a "native Apple Silicon" ML framework looks like in practice
- When to choose MLX: if you're deploying on Apple devices, MLX's unified memory and lazy evaluation can simplify your pipeline compared to PyTorch with MPS backend

## Run

Requires a Mac with Apple Silicon (M1/M2/M3/M4).

```bash
uv run python main.py
```

Trains for 1000 steps and generates 20 names. Runs on the GPU automatically.

Every tenth step prints its wall-clock time, and the run ends with the compiled-step averages described above. Timings quoted in this README came from a CPU-only MLX build in a Linux container, so treat them as ratios rather than as numbers to expect on your Mac.
