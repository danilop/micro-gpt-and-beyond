# Understanding LLMs by Building One: NumPy with Array Autograd

Same GPT architecture as the pure-Python version, and the same autograd idea, with NumPy arrays in place of Python scalars. One graph node now holds a whole array, so a matrix multiply is one call into BLAS rather than thousands of interpreted operations.

## Why this version exists

Version 01 builds a scalar autograd engine. Every number in the model is its own graph node, which makes the chain rule easy to watch but means a single 16x16 matrix multiply costs 256 Python objects and 256 interpreter steps.

This version changes one thing: the unit of computation. The engine keeps the same structure, and the model keeps the same architecture, weights, and training schedule. What changes is that `Tensor` wraps `np.ndarray` where `Value` wrapped `float`. Running both labs back to back shows what vectorization is worth, with the programming model held constant.

## What makes it interesting

### The engine has the same shape as version 01

Version 01's `Value` records a *local derivative* for each child, then backpropagates with `child.grad += local_grad * v.grad`. That works because for scalars every local derivative is a number you multiply by.

Arrays need more than a multiplier. The gradient of a matrix multiply is another matrix multiply, the gradient of a reshape is the inverse reshape, and the gradient of an index is a scatter-add. So each operation records a closure:

```python
def __matmul__(self, other):
    out = Tensor(self.data @ other.data, (self, other))

    def _backward():
        self.grad += out.grad @ other.data.swapaxes(-1, -2)
        other.grad += self.data.swapaxes(-1, -2) @ out.grad

    out._backward = _backward
    return out
```

Everything else carries over from version 01: the same topological sort, the same reverse walk, the same accumulation into `.grad`.

### RMSNorm and softmax are not special

Neither version treats them as primitives. They are ordinary functions built from operations the engine already knows, so nobody derives their gradients:

```python
def rmsnorm(x):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x * (ms + 1e-5) ** -0.5
```

That is the formula as you would write it on paper. Compare it with version 01's scalar `rmsnorm`, which is the same three lines over lists.

### Broadcasting is the one new idea

Scalars never broadcast, so version 01 never had to think about it. Arrays do. When `(n, 1)` meets `(n, d)`, the forward pass silently copies a column `d` times, and the backward pass has to add those `d` gradients back together:

```python
def unbroadcast(grad, shape):
    while grad.ndim > len(shape):  # drop axes that broadcasting prepended
        grad = grad.sum(axis=0)
    for i, size in enumerate(shape):  # collapse axes that were stretched from 1
        if size == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
```

Every array autograd system needs this, PyTorch included. It is worth reading closely, because it is the one place where the array engine genuinely departs from the scalar one.

### The causal mask costs nothing

Masking is an addition, so it needs no gradient rule of its own:

```python
att = softmax(att + np.triu(np.full((n, n), -1e9), k=1))
```

The `-1e9` entries drive those weights to zero through softmax, and softmax's own gradient then sends zero back through them.

## What you learn here

- How a scalar autograd engine generalizes to arrays, and which parts survive unchanged
- Why array gradients need closures where scalar gradients need only a multiplier
- What broadcasting costs on the backward pass, and how `unbroadcast` pays it
- What vectorization is worth when the programming model is held fixed

## Run

```bash
uv run python main.py
```

Trains for 1000 steps and generates 20 names.

### About speed

The training loop prints wall-clock `ms/step` and a mean at the end, because the interesting claim here, that "vectorized NumPy beats a Python loop over scalars", is worth measuring rather than asserting. Both labs do the same arithmetic on the same data with the same seeds, so their loss curves match step for step and only the time differs.

The size of the gap depends on your CPU and on which BLAS your NumPy is linked against, so no fixed multiplier is quoted here. Time version 01 the same way on the same machine if you want the ratio for your hardware. For reference, the machine this was written on prints about 0.3 ms/step for this version against about 38 ms/step for version 01.

Some of that gap is spent on the graph itself. Building and walking a few dozen `Tensor` nodes per step is real work, and a version that skipped autograd and hard-coded the backward pass would run about twice as fast again. That is the trade every framework makes, and it is why PyTorch in version 03 feels familiar rather than exotic.
