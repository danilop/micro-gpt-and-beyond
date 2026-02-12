# microGPT and Beyond — JAX Batched

Same architecture as `05_jax`, but with mini-batch training — and this is where JAX's design really shines. Instead of rewriting the forward pass to handle a batch dimension, you use `jax.vmap` to automatically vectorize the single-example code across a batch.

## Why this version exists

In PyTorch (04), batching means rewriting your model to handle `(B, T, ...)` tensors, adding padding logic, and threading mask arguments through every layer. In JAX, you write the forward pass for one example and let `vmap` do the rest. This version shows that difference.

## What makes it interesting

### vmap: "write for one, run for many"

The core trick is a single line. `forward_single` takes one sequence. `vmap` lifts it to take a batch:

```python
forward_batch = vmap(forward_single, in_axes=(None, 0, 0))
```

`in_axes=(None, 0, 0)` means: params are shared (not batched), input_ids are batched along axis 0, pad_mask is batched along axis 0. JAX compiles this into efficient batched operations — you never write a `(B, T, ...)` reshape yourself.

### The forward pass stays clean

Compare this to the PyTorch batched version. The JAX forward pass is still written for a single sequence:

```python
def forward_single(params, input_ids, pad_mask):
    T = input_ids.shape[0]
    tok_emb = params['wte'][input_ids]       # (T, D) — not (B, T, D)
    pos_emb = params['wpe'][jnp.arange(T)]
    x = rmsnorm(tok_emb + pos_emb)
    ...
```

No batch dimension anywhere. `vmap` adds it automatically at call time.

### Padding still happens at the data level

You still need to pad sequences to the same length within a batch — `vmap` requires all inputs to have the same shape. But the model code doesn't know about padding. The mask is just another input that `vmap` broadcasts:

```python
input_ids, targets, pad_mask, target_mask = make_batch(docs, step, batch_size)
grads = grad_fn(params, input_ids, targets, pad_mask, target_mask)
```

### Scaled up

Like the PyTorch batched version, this uses a bigger model to make batching worthwhile:

| | 05 JAX | 06 JAX Batched |
|---|---|---|
| Embedding dim | 16 | 64 |
| Layers | 1 | 2 |
| Context length | 8 | 16 |
| Batch size | 1 | 32 |
| Training steps | 500 | 1000 |

## What you learn here

- `jax.vmap` — automatic vectorization without rewriting model code
- How `in_axes` controls which arguments get batched and which are shared
- The JAX philosophy: transformations (`grad`, `jit`, `vmap`) compose over pure functions
- Why functional purity pays off — `vmap` only works on side-effect-free functions

## Run

```bash
uv run python main.py
```
