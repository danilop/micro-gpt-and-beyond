# Understanding LLMs by Building One: JAX Batched

Same architecture as `05_jax`, but with mini-batch training, and this is where JAX's design really shines. Instead of rewriting the forward pass to handle a batch dimension, you use `jax.vmap` to automatically vectorize the single-example code across a batch.

## Why this version exists

In PyTorch (04), batching means rewriting your model to handle `(B, T, ...)` tensors, adding padding logic, and threading mask arguments through every layer. In JAX, you write the forward pass for one example and let `vmap` do the rest. This version shows that difference — and, because everything is jitted, it is also the right place to learn that `jit` compiles per input shape.

## What makes it interesting

### vmap: "write for one, run for many"

The core trick is a single line. `forward_single` takes one sequence. `vmap` lifts it to take a batch:

```python
forward_batch = vmap(forward_single, in_axes=(None, 0, 0))
```

`in_axes=(None, 0, 0)` means: params are shared (not batched), input_ids are batched along axis 0, pad_mask is batched along axis 0. JAX compiles this into efficient batched operations, so you never write a `(B, T, ...)` reshape yourself.

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

You still need to pad sequences to a common length because `vmap` requires all inputs to have the same shape. But the model code doesn't know about padding. The mask is just another input that `vmap` broadcasts:

```python
batch, dynamic_len = make_batch(docs, step, batch_size)
loss_val, params, m_state, v_state = train_step(params, m_state, v_state, batch, step, lr_t)
```

Padding is always *appended*: real tokens first, `PAD` afterwards, never interleaved. Everything below follows from that.

### One jitted step

As in `05_jax`, the loss, the gradients and the Adam update live in a single `@jit` function, so XLA gets the whole step as one program:

```python
@jit
def train_step(params, m_state, v_state, batch, step, lr):
    input_ids, targets, pad_mask, target_mask = batch
    loss, grads = value_and_grad(loss_fn)(params, input_ids, targets, pad_mask, target_mask)
    ...
```

`value_and_grad` rather than `grad` matters here: `grad` alone would mean a second forward pass just to get a number to print, and `vmap`-ing a forward pass over 32 sequences is not something to do twice for no reason.

### Pad to block_size, not to the longest name in the batch

This is the change that matters most in this lab. The natural way to pad a batch is to the longest sequence in it:

```python
max_len = max(len(s) for s in sequences)     # 6 in one batch, 11 in the next...
```

Under `jit`, that is a trap. XLA compiles per input shape, so a batch of `(32, 6)` and a batch of `(32, 11)` are different programs, and each new longest-name-in-batch triggers another trace and compile. `05_jax` shows the single-example version of the same problem: one name per step, twelve compilations.

So `make_batch` here pads to the static `block_size` instead, and the loop counts both what happened and what would have happened:

```
step    1 | new input shape (32, 16) -> XLA trace #1
step    1 / 1000 | loss 3.6608 | 1731.90 ms
step   10 / 1000 | loss 2.4799 |    9.10 ms
...
compilations of train_step: 1 (padding to the static block_size)
  in-batch max_len values seen: 9 -> that many compilations if we padded dynamically
  first step: 1731.9 ms (trace + compile)
  steps 2..1000: 9.42 ms mean, max 48.34 ms
```

One compile instead of nine. After the first step there are no compilation spikes at all — a 12-letter name arriving at step 300 changes nothing, because the shape it lands in was already compiled. (The `max` above is ordinary scheduling noise, not a trace.)

The cost is real but small: a batch whose longest name is five characters still does arithmetic over all 16 columns. Static shapes are usually worth that, which is why production pipelines bucket sequences or pad to fixed lengths rather than to whatever arrived.

### The two masks are not equally important

`forward_single` gets a `pad_mask` and blocks attention to `PAD` keys:

```python
att = jnp.where(causal, -1e9, att)
att = jnp.where(pad_mask[None, None, :], -1e9, att)
```

With padding appended at the end and a causal mask already in place, that second line changes nothing you can measure. A query at position `t` can only see keys up to `t`, and every one of those is a real token, so no real query ever reaches a pad key. The only logits it touches belong to pad queries, and `target_mask` multiplies those out of the loss.

It is kept because the redundancy is a property of this configuration, not of padding masks. Left-pad the batch, switch to bidirectional attention, or pack several documents into one row, and the key-side mask becomes load-bearing immediately.

What is *not* redundant is `target_mask`, below.

Note also what is missing: there is no `jnp.nan_to_num` after the softmax. A row masked entirely to `-inf` would softmax to NaN, but this code masks with `-1e9`, so such a row would come out uniform, not NaN. And no row can be fully masked in the first place, since row 0 always keeps position 0, which is `BOS`.

### Scaled up

Like the PyTorch batched version, this uses a bigger model to make batching worthwhile:

| | 05 JAX | 06 JAX Batched |
|---|---|---|
| Embedding dim | 16 | 64 |
| Layers | 1 | 2 |
| Context length | 16 | 16 |
| Batch size | 1 | 32 |
| Training steps | 1000 | 1000 |

### Loss masking — the mask that actually does the work

The loss function uses a `target_mask` to ignore padded positions, the same idea as PyTorch's `ignore_index=-100`, but done explicitly with a mask since JAX doesn't have a built-in convention. This one is essential: the dummy target at padded positions is `0`, which is a real character, so without the mask the model would be trained to predict `'a'` after the end of every short name.

```python
def loss_fn(params, input_ids, targets, pad_mask, target_mask):
    logits = forward_batch(params, input_ids, pad_mask)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B, T, V = log_probs.shape
    target_log_probs = log_probs[jnp.arange(B)[:, None], jnp.arange(T)[None, :], targets]
    target_log_probs = target_log_probs * target_mask
    loss = -jnp.sum(target_log_probs) / jnp.sum(target_mask)
    return loss
```

## What you learn here

- `jax.vmap` for automatic vectorization without rewriting model code
- How `in_axes` controls which arguments get batched and which are shared
- Why padding to a static length beats padding to the batch maximum under `jit`, measured rather than asserted
- Which mask is load-bearing (`target_mask`) and which is inherited boilerplate (the key-side `pad_mask`)
- The JAX philosophy: transformations (`value_and_grad`, `jit`, `vmap`) compose over pure functions

## Run

```bash
uv run python main.py
```

Trains for 1000 steps (prints every 10) and generates 20 names. Step 1 pays for the trace and compile (a second or two); every step after it is fast and, because the shape is static, stays fast. Numbers quoted above came from the machine this was written on.
