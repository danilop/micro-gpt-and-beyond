# microGPT and Beyond — MLX Batched

Same architecture as `07_mlx`, scaled up with mini-batch training on Apple Silicon. This version follows the same batching approach as `04_pytorch_batched` — padding, masking, and explicit `(B, T, ...)` tensor shapes — because MLX's API is intentionally PyTorch-like.

## Why this version exists

After seeing JAX's `vmap` approach in `06_jax_batched`, this version shows the contrast: MLX doesn't have a `vmap` equivalent. Batching in MLX means the same manual work as PyTorch — reshape your model to handle a batch dimension, pad sequences, thread masks through attention. The difference is where it runs: unified memory on Apple Silicon, with lazy evaluation.

## What makes it interesting

### Same pattern as PyTorch, different runtime

The forward pass looks almost identical to `04_pytorch_batched`. Compare the attention:

```python
def __call__(self, x, pad_mask=None):
    B, T, C = x.shape
    q = self.wq(x).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    k = self.wk(x).reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
    ...
    if pad_mask is not None:
        att = mx.where(pad_mask[:, None, None, :], mx.array(-1e9), att)
```

The shapes, the masking, the reshapes — all the same. What's different is that this runs on the Apple GPU through unified memory, with no explicit device transfers.

### Lazy evaluation with batches

With batches, lazy evaluation matters more. MLX builds up a larger computation graph per step (32 sequences worth), then executes it all at once:

```python
loss_val, grads = loss_and_grad(model, input_ids, targets, pad_mask, target_mask)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)  # one big eval for the whole batch
```

The `mx.eval` call triggers the entire forward + backward + optimizer update in one fused execution on the GPU.

### Scaled up

| | 07 MLX | 08 MLX Batched |
|---|---|---|
| Embedding dim | 16 | 64 |
| Layers | 1 | 2 |
| Context length | 16 | 16 |
| Batch size | 1 | 32 |
| Training steps | 1000 | 1000 |

### Padding and masking — the same work as PyTorch

The `make_batch` function pads sequences and builds masks, just like `04_pytorch_batched`. The only difference is `mx.array` instead of `torch.tensor`:

```python
def make_batch(docs, step, batch_size):
    for s in sequences:
        n = len(s) - 1
        inp = s[:n] + [PAD] * (max_len - 1 - n)
        tgt = s[1:n+1] + [0] * (max_len - 1 - n)
        pmask = [False] * n + [True] * (max_len - 1 - n)
        tmask = [1.0] * n + [0.0] * (max_len - 1 - n)

    return mx.array(input_ids), mx.array(target_ids), mx.array(pad_masks), mx.array(target_masks)
```

This is the manual work that JAX's `vmap` avoids — and the reason the 06 vs 08 comparison is instructive.

## What you learn here

- MLX batching follows the PyTorch pattern — no `vmap` shortcut
- How padding and masking work with MLX's array API
- Lazy evaluation becomes more impactful with larger computation graphs
- The practical tradeoff: MLX's API familiarity vs JAX's functional transformations

## Run

Requires a Mac with Apple Silicon (M1/M2/M3/M4).

```bash
uv run python main.py
```

Trains for 1000 steps (prints every 10) and generates 20 names. Runs on the Apple GPU automatically.
