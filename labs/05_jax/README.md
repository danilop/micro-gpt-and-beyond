# Understanding LLMs by Building One: JAX

Same architecture, but written in JAX's purely functional style. No classes, no hidden state, no mutation. Every function takes its inputs and returns its outputs, nothing else.

## Why this version exists

PyTorch is object-oriented: models are classes, parameters live inside `self`, optimizers maintain internal state. JAX takes the opposite approach where everything is a pure function, and all state is passed explicitly. This version shows how the same transformer looks when you commit fully to functional programming.

## What makes it interesting

### Parameters are just a dict

There are no `nn.Module` classes. Parameters are a plain Python dict of JAX arrays:

```python
params = {
    'wte': init_param(next(ki), (vocab_size, n_embd)),
    'wpe': init_param(next(ki), (block_size, n_embd)),
    'lm_head': init_param(next(ki), (vocab_size, n_embd)),
}
for i in range(n_layer):
    params[f'l{i}.wq'] = init_param(next(ki), (n_embd, n_embd))
    # ...
```

This dict *is* the model. There's no wrapper object, no registration, no `state_dict()`. Just data.

### The forward pass is a pure function

`forward(params, input_ids)` takes parameters and tokens, returns logits. No side effects, no hidden state:

```python
def forward(params, input_ids):
    n = input_ids.shape[0]
    tok_emb = params['wte'][input_ids]
    pos_emb = params['wpe'][jnp.arange(n)]
    x = rmsnorm(tok_emb + pos_emb)

    for li in range(n_layer):
        x_res = x
        x_n = rmsnorm(x)
        Q = x_n @ params[f'l{li}.wq']
        K = x_n @ params[f'l{li}.wk']
        V = x_n @ params[f'l{li}.wv']
        # ... attention, MLP, residuals ...

    return x @ params['lm_head'].T
```

Because it's pure, JAX can transform it: differentiate it with `grad`, compile it with `jit`, vectorize it with `vmap`, all automatically.

### Gradients via function transformation

Instead of recording a tape and calling `.backward()`, JAX transforms the loss function into a gradient function:

```python
loss, grads = value_and_grad(loss_fn)(params, input_ids, targets)
```

`grad(loss_fn)` would return `∂loss/∂params` alone, which means running the forward pass a second time whenever you also want the loss to print. `value_and_grad` returns both, because the backward pass computes the loss on its way through anyway.

### The whole step is one jitted function

`jit` pays off in proportion to how much you put inside it. Here the loss, the gradients and the Adam update all live in one function, so XLA compiles them as a single program:

```python
@jit
def train_step(params, m_state, v_state, input_ids, targets, step, lr):
    loss, grads = value_and_grad(loss_fn)(params, input_ids, targets)
    new_m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, m_state, grads)
    new_v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g**2, v_state, grads)
    ...
    return loss, new_params, new_m, new_v
```

Note `jax.tree.map` rather than a Python `for k in params` loop. The optimizer arithmetic is identical, but a tree map is one traced operation per leaf inside the compiled program instead of interpreted Python between compiled calls. It also stops caring how many parameter tensors there are.

### Explicit PRNG threading

JAX doesn't have a global random state. Every random operation requires an explicit key, and you split keys to get new ones:

```python
key = jax.random.key(42)            # typed key, the modern API
keys = jax.random.split(key, num_param_tensors)
```

```python
rng_key = jax.random.key(0)
for sample_idx in range(20):
    # ...
    rng_key, subkey = jax.random.split(rng_key)
    token_id = jax.random.categorical(subkey, logits).item()
```

`jax.random.key` is the current constructor; `jax.random.PRNGKey` is the older form that returns a raw pair of `uint32`. Same numbers, but a typed key carries its generator implementation in its dtype, which is what the rest of modern JAX expects. Since this lab is *about* explicit PRNG handling, it may as well use the API JAX actually wants.

The key count is derived from the model (`3 + 6 * n_layer`) rather than hard-coded, so adding a layer doesn't silently exhaust the iterator.

This makes randomness reproducible and parallelizable, two things that are hard with global state.

### Functional Adam optimizer

The optimizer is a pure function too. No internal state mutation. It takes the old state and returns the new state:

```python
new_m = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, m_state, grads)
new_v = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g**2, v_state, grads)
new_params = jax.tree.map(
    lambda p, m, v: p - lr * (m / bias1) / (jnp.sqrt(v / bias2) + eps_adam),
    params, new_m, new_v,
)
```

Compare this to PyTorch's `optimizer.step()` which mutates parameters in-place. Same math, different philosophy.

### jit compiles per shape, and this lab shows you where

Everyone repeats that "the first JAX call is slow, then it's fast". That is only half true, and this lab is a good place to see the other half. `jit` specializes on the *shape* of its inputs, and here the input length is `n = min(block_size, len(tokens) - 1)`, which depends on the name. Names in the corpus run from 2 to 15 characters, so `n` ranges over 14 values, and each new one triggers a fresh trace and compile, at step 1 but also at step 156, and step 184, and later.

So the training loop times each step and announces new shapes:

```
step    1 | new sequence length n= 7 -> XLA trace #1 |  1400.8 ms
step   31 | new sequence length n=11 -> XLA trace #8 |  1518.0 ms
step  184 | new sequence length n= 3 -> XLA trace #10 |  1472.9 ms
...
distinct sequence lengths: 12 -> 12 traces of the same train_step
  first-time-shape steps:   1375.5 ms mean (compilation included)
  cached-shape steps:         0.64 ms mean
```

Three orders of magnitude between a compiling step and a cached one, and 12 of them in a 1000-step run. That is the single most common performance surprise in JAX. The fix is to make shapes static by padding everything to one fixed length, which is exactly what `06_jax_batched` does.

## What you learn here

- How to express a neural network as pure functions with no hidden state
- JAX's `value_and_grad` + `jit` composition, where differentiation and compilation are function transforms
- Why the whole training step belongs inside `jit`, and why `jax.tree.map` beats a Python loop over parameters
- Explicit PRNG key management with typed keys, and why it matters for reproducibility
- That `jit` recompiles per input shape, with the cost measured rather than assumed

## Run

```bash
uv run python main.py
```

Trains for 1000 steps and generates 20 names.

Each step prints its wall-clock time, and steps that introduce a new sequence length are flagged as XLA traces. Expect around a second on those and well under a millisecond on the rest. Numbers above came from the machine this was written on; yours will differ, but the ratio won't.
