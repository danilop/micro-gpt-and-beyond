# Understanding LLMs by Building One: MLX Batched

Same architecture as `07_mlx`, scaled up with mini-batch training on Apple Silicon. Batching is done with `mx.vmap`, so the model code stays single-example, the way `06_jax_batched` does it with `jax.vmap`.

## Why this version exists

Labs 04, 06 and 08 solve the same problem — turn a one-sequence model into a batched one — with the three tools the three frameworks give you:

| | how the batch dimension appears |
|---|---|
| `04_pytorch_batched` | by hand: the model is rewritten for `(B, T, ...)` tensors |
| `06_jax_batched` | `jax.vmap` over a single-example function |
| `08_mlx_batched` | `mx.vmap` over a single-example function |

MLX has `mx.vmap`, and it composes with `nn.value_and_grad` and `mx.compile`, so there is no reason to hand-write the batch dimension here. What is genuinely different about MLX is the runtime underneath: unified memory and lazy evaluation.

## What makes it interesting

### mx.vmap: no batch dimension in the model

`forward_single` handles one sequence. `__call__` lifts it over the batch:

```python
def forward_single(self, idx, pad_mask):
    # idx: (T,), pad_mask: (T,) — no batch dimension anywhere below this point
    T = idx.shape[0]
    tok_emb = self.wte(idx)
    ...

def __call__(self, idx, pad_mask=None):
    # idx: (B, T) -> logits (B, T, vocab_size)
    return mx.vmap(self.forward_single, in_axes=(0, 0))(idx, pad_mask)
```

`in_axes=(0, 0)` maps over axis 0 of `idx` and of `pad_mask`. The parameters come from `self`, are captured by the closure, and are shared across the batch — that is what `in_axes=None` states explicitly in the JAX version.

Attention therefore keeps the `(nh, T, T)` shapes from `07_mlx` instead of growing to `(B, nh, T, T)`:

```python
att = (q @ k.transpose(0, 2, 1)) / math.sqrt(head_dim)
causal = mx.triu(mx.ones((T, T)), k=1).astype(mx.bool_)
att = mx.where(causal, -1e9, att)
att = mx.where(pad_mask[None, None, :], -1e9, att)
```

Is the readable version also the slower one? The lab measures that rather than asserting it. `forward_batched` in `main.py` is the same computation with the batch axis written out by hand, sharing the same parameters, and the run prints two things about it: the largest logit difference between the two paths (zero on the CPU-only build used here — same operations, same order), and the best of 20 timed forward passes each way. On this build the two came out within a couple of percent of each other, and the verdict line the lab prints is computed from the gap it just measured, not remembered from another machine. Run it on your Mac and it will tell you what the transform costs there.

### Lazy evaluation and mx.compile

MLX doesn't compute anything until asked. With batches the pending graph per step is larger (32 sequences worth of forward, backward and Adam), and `mx.compile` is what fuses it, the same role `jax.jit` plays:

```python
state = [model.state, optimizer.state]

def train_step(input_ids, targets, pad_mask, target_mask):
    loss_val, grads = loss_and_grad(model, input_ids, targets, pad_mask, target_mask)
    optimizer.update(model, grads)
    return loss_val

train_step = mx.compile(train_step, inputs=state, outputs=state)
```

Then, once per step:

```python
mx.eval(state)
```

That call is not there to fetch the loss — `loss_val.item()` already forces the loss, since you cannot read a Python float out of an unevaluated graph. It is there because nothing downstream reads the *parameters* or the optimizer moments, so without it the graph of pending updates would keep growing for the whole run. `mx.eval` bounds it at one step.

Because `mx.compile` specialises on input shapes, and `make_batch` pads to the longest sequence in each batch, the shape changes from step to step. The loop counts it:

```
distinct batch shapes seen: 9 (compile = True)
  first-time-shape steps:   100.83 ms mean
  repeated-shape steps:      80.04 ms mean
  each shape first seen at step: [0, 1, 4, 10, 11, 15, 79, 304, 532]
```

Nine shapes, nine traces. Do not read the first-vs-repeat gap as the price of tracing, though — run the same file with `use_compile = False` and the gap is still there (93.87 against 84.02 ms), with nothing being traced at all. That last printed line is why: six of the nine shapes turn up in the first sixteen steps, so "first-time shape" is largely a synonym for "early step", when nothing is warm yet. `06_jax_batched` shows the fix for the shape churn anyway — pad to a fixed length and there is only one shape to compile.

Set `use_compile = False` and compare. On the CPU-only Linux build these numbers came from, uncompiled repeated-shape steps averaged 84.02 ms against 80.04 ms compiled: close to a wash at this batch size, because the per-operation dispatch overhead that compilation removes is small next to 32 sequences worth of matmuls. `07_mlx`, whose steps are around a millisecond, gets 2.5x from the same call on the same build (2.77 ms uncompiled against 1.12 ms compiled). Measure it on your own hardware rather than believing either number.

### Scaled up

| | 07 MLX | 08 MLX Batched |
|---|---|---|
| Embedding dim | 16 | 64 |
| Layers | 1 | 2 |
| Context length | 16 | 16 |
| Batch size | 1 | 32 |
| Training steps | 1000 | 1000 |

### Padding still happens at the data level

`vmap` removes the batch dimension from the model, not from the data. Sequences still have to be padded to a common length before they can be stacked:

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

Padding is always appended, never interleaved with real tokens. Two masks come out of this, and they are not equally important.

**`target_mask` is essential.** The dummy target at padded positions is `0`, and `0` is a real character (`'a'`). Multiply the per-position log-probabilities by `target_mask` and those positions contribute nothing; forget to, and the model is explicitly trained to predict `'a'` after the end of every short name. There is no `ignore_index` here to save you, so the mask is the whole mechanism.

**`pad_mask` is inert.** The key-side mask inside attention changes nothing measurable in this configuration: padding is a suffix, and the causal mask already prevents a query at position `t` from reading anything after `t`, so no real query can reach a pad key. The only logits it touches belong to pad queries, and `target_mask` has already zeroed those. It is kept because it becomes load-bearing the moment you left-pad, use bidirectional attention, or pack several documents into one row.

That contrast is the useful lesson: two lines that look like the same defensive measure, one carrying all the weight and one carrying none.

There is also no `nan_to_num`-style guard after the softmax, and none is needed. The mask value is `-1e9`, not `-inf`, so even a fully masked row would come out uniform rather than NaN — and no row can be fully masked anyway, since row 0 always keeps position 0, which is `BOS`.

## What you learn here

- `mx.vmap`, and that MLX has the same "write for one, run for many" transform JAX does — including how to check it against a hand-written batch dimension for both output and speed
- How padding and masking work with MLX's array API, and which of the two masks actually does anything
- `mx.eval` as a bound on the pending graph, and `mx.compile` as MLX's `jit`, per-shape specialisation included
- The practical tradeoff: MLX's API familiarity plus functional transforms, on Apple hardware
- When to choose MLX: if you're deploying on Apple devices, MLX's unified memory and lazy evaluation can simplify your pipeline compared to PyTorch with MPS backend

## Run

Requires a Mac with Apple Silicon (M1/M2/M3/M4).

```bash
uv run python main.py
```

Trains for 1000 steps (prints every 10) and generates 20 names. Runs on the Apple GPU automatically.

Every tenth step prints its wall-clock time, and the run ends with the per-shape averages above and the `mx.vmap`-against-hand-written comparison. Those numbers came from a CPU-only MLX build in a Linux container, so read them as ratios, not as what to expect on your Mac.
