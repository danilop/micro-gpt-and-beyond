# microGPT and Beyond — NumPy with Manual Backpropagation

Same GPT architecture as the pure-Python version, but using NumPy arrays for vectorized matrix operations. The key twist: every gradient is derived by hand and coded explicitly. There is no autograd — you *are* the autograd.

## Why this version exists

Version 01 builds a scalar autograd engine that computes gradients automatically. This version throws that away and asks: what if you had to derive every backward pass yourself, at the matrix level?

The result is the most educational version in the series for anyone who wants to deeply understand backpropagation. Writing `rmsnorm_bwd` and `softmax_bwd` by hand forces you to work through the calculus that frameworks hide behind a single `.backward()` call.

## What makes it interesting

### Hand-written gradient functions

Every forward operation has a matching backward function. Here's RMSNorm — the forward pass normalizes by root-mean-square, and the backward pass applies the chain rule through that normalization:

```python
def rmsnorm_fwd(x):
    ms = np.mean(x ** 2, axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(ms + 1e-5)
    out = x * scale
    return out, (x, scale, ms)

def rmsnorm_bwd(dout, cache):
    x, scale, ms = cache
    D = x.shape[-1]
    dx = dout * scale - x * scale ** 3 * np.sum(dout * x, axis=-1, keepdims=True) / D
    return dx
```

The forward pass returns a `cache` tuple — the intermediate values the backward pass needs. This is exactly the pattern PyTorch uses internally for custom autograd functions.

### Softmax backward

Softmax is one of those operations that looks simple forward but has a surprisingly elegant backward pass:

```python
def softmax_bwd(probs, dprobs):
    s = np.sum(dprobs * probs, axis=-1, keepdims=True)
    return probs * (dprobs - s)
```

If you've ever wondered what the Jacobian of softmax looks like in practice — this is it, collapsed into two lines using the identity `∂softmax_i/∂z_j = p_i(δ_ij - p_j)`.

### The full backward pass

The `backward()` function walks the computation graph in reverse, layer by layer, accumulating gradients into the `G` dict. The attention backward is the most involved — it threads gradients back through the multi-head reshape, the softmax, the QKV projections, and the causal mask:

```python
# Attention scores backward: att = Q_h @ K_h^T / sqrt(d)
dQ_h = datt @ K_h           # (nh, n, hd)
dK_h = datt.transpose(0, 2, 1) @ Q_h  # (nh, n, hd)
```

### Caching strategy

The forward pass stores everything the backward pass needs in a `cache` dict — activations, pre-softmax scores, intermediate norms. This is the same memory-vs-compute tradeoff that every deep learning framework makes: store activations during the forward pass so you don't have to recompute them during backward.

## What you learn here

- How to derive matrix-level gradients for every transformer operation
- The cache/checkpoint pattern that all autograd systems use internally
- Why autograd exists — after writing 150 lines of backward pass, you'll appreciate `loss.backward()`
- The exact relationship between the scalar chain rule (01) and the matrix chain rule (this version)

## Run

```bash
uv run python main.py
```

Trains for 500 steps and generates 20 names. Runs ~100x faster than the pure-Python version thanks to NumPy's vectorized operations, despite doing the same math.
