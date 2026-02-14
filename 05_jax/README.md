# microGPT and Beyond — JAX

Same architecture, but written in JAX's purely functional style. No classes, no hidden state, no mutation. Every function takes its inputs and returns its outputs — nothing else.

## Why this version exists

PyTorch is object-oriented: models are classes, parameters live inside `self`, optimizers maintain internal state. JAX takes the opposite approach — everything is a pure function, and all state is passed explicitly. This version shows how the same transformer looks when you commit fully to functional programming.

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

This dict *is* the model. There's no wrapper object, no registration, no `state_dict()` — just data.

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

Because it's pure, JAX can transform it — differentiate it with `grad`, compile it with `jit`, vectorize it with `vmap` — all automatically.

### Gradients via function transformation

Instead of recording a tape and calling `.backward()`, JAX transforms the loss function into a gradient function:

```python
grad_fn = jit(grad(loss_fn))
```

`grad(loss_fn)` returns a new function that computes `∂loss/∂params`. `jit` compiles it to XLA for speed. The result is a single function call that returns all gradients:

```python
grads = grad_fn(params, input_ids, targets)
```

### Explicit PRNG threading

JAX doesn't have a global random state. Every random operation requires an explicit key, and you split keys to get new ones:

```python
rng_key = jax.random.PRNGKey(0)
for sample_idx in range(20):
    # ...
    rng_key, subkey = jax.random.split(rng_key)
    token_id = jax.random.categorical(subkey, logits).item()
```

This makes randomness reproducible and parallelizable — two things that are hard with global state.

### Functional Adam optimizer

The optimizer is a pure function too. No internal state mutation — it takes the old state and returns the new state:

```python
def adam_update(params, grads, m_state, v_state, step, lr):
    new_params, new_m, new_v = {}, {}, {}
    for k in params:
        new_m[k] = beta1 * m_state[k] + (1 - beta1) * grads[k]
        new_v[k] = beta2 * v_state[k] + (1 - beta2) * grads[k] ** 2
        m_hat = new_m[k] / (1 - beta1 ** (step + 1))
        v_hat = new_v[k] / (1 - beta2 ** (step + 1))
        new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)
    return new_params, new_m, new_v
```

Compare this to PyTorch's `optimizer.step()` which mutates parameters in-place. Same math, different philosophy.

## What you learn here

- How to express a neural network as pure functions with no hidden state
- JAX's `grad` + `jit` composition — differentiation and compilation as function transforms
- Explicit PRNG key management and why it matters for reproducibility
- The functional programming paradigm applied to deep learning

## Run

```bash
uv run python main.py
```

Trains for 1000 steps and generates 20 names. First run may be slow as JAX JIT-compiles the functions; subsequent calls are fast.
