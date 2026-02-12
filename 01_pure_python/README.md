# microGPT

> *"This file is the complete algorithm. Everything else is just efficiency."*
> — Andrej Karpathy

A GPT language model — trained and run — in a single Python file, with zero dependencies beyond the standard library. No PyTorch. No TensorFlow. No NumPy. Just `math`, `random`, and `os`.

This project strips a Generative Pre-trained Transformer down to its absolute essence: ~200 lines of pure Python that build an autograd engine, define a transformer architecture, train it with Adam, and generate text. If you've ever wanted to understand what *actually* happens inside a GPT, this is the place to start.

---

## What Does It Do?

It trains a tiny GPT on a dataset of ~32,000 human names and learns to generate new, plausible-sounding ones. After 500 training steps, it babbles back invented names like a creative baby naming consultant who's read too many birth certificates.

```
sample  1: mara
sample  2: kaia
sample  3: jori
sample  4: elina
...
```

The names aren't memorized — they're *generated* by a model that has learned the statistical patterns of how characters combine in English names.

---

## The Architecture, Step by Step

Let's walk through the entire file, section by section. Every piece of a modern GPT is here, just small enough to hold in your head.

### 1. The Dataset

```python
docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()]
```

The training data is a list of names (one per line), loaded from `../data/input.txt`. Each name is treated as a "document" — a short sequence of characters. If the file doesn't exist, the script downloads Karpathy's `names.txt` dataset automatically.

The names are shuffled with a fixed seed (`random.seed(42)`) so training order is randomized but reproducible.

### 2. The Tokenizer

```python
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
```

Tokenization here is character-level. Every unique character in the dataset gets an integer ID. One special token is added: `BOS` (Beginning of Sequence), which acts as both the start and end marker for each name.

Real-world GPTs use subword tokenizers (like BPE) with vocabularies of 50k+ tokens. Here, the vocabulary is just the 26 lowercase letters plus BOS — 27 tokens total. Same idea, smaller scale.

### 3. The Autograd Engine

```python
class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
```

This is the heart of the "no dependencies" claim. The `Value` class implements a scalar-level automatic differentiation engine. Every arithmetic operation (`+`, `*`, `**`, `exp`, `log`, `relu`) creates a node in a computation graph, recording:

- The result of the forward computation (`data`)
- Which values produced it (`children`)
- The local derivative of the operation with respect to each child (`local_grads`)

When you call `loss.backward()`, it walks the graph in reverse topological order and applies the chain rule:

```python
def backward(self):
    # Build topological ordering
    # Then propagate gradients backward
    for v in reversed(topo):
        for child, local_grad in zip(v._children, v._local_grads):
            child.grad += local_grad * v.grad
```

This is exactly what PyTorch's `autograd` does — just on scalars instead of tensors. Every parameter gets a `.grad` that tells the optimizer how to nudge it to reduce the loss.

### 4. The Parameters

```python
n_embd = 16      # embedding dimension
n_head = 4       # number of attention heads
n_layer = 1      # number of layers
block_size = 8   # maximum sequence length
```

The model's knowledge lives in its parameters — matrices of `Value` objects initialized with small random numbers (Gaussian, std=0.02). The `state_dict` contains:

| Parameter | Shape | Purpose |
|-----------|-------|---------|
| `wte` | (vocab_size, 16) | Token embeddings — a learned vector for each token |
| `wpe` | (8, 16) | Position embeddings — a learned vector for each position |
| `attn_wq/wk/wv` | (16, 16) | Query, Key, Value projections for attention |
| `attn_wo` | (16, 16) | Output projection after attention |
| `mlp_fc1` | (64, 16) | First MLP layer (expands 4x) |
| `mlp_fc2` | (16, 64) | Second MLP layer (contracts back) |
| `lm_head` | (vocab_size, 16) | Final projection to vocabulary logits |

All parameters are flattened into a single list for the optimizer. The total count is printed at startup.

### 5. The Model (Forward Pass)

The `gpt()` function is a stateless function: tokens and parameters go in, logits come out. It follows the GPT-2 architecture with a few simplifications:


#### a) Embedding

```python
tok_emb = state_dict['wte'][token_id]
pos_emb = state_dict['wpe'][pos_id]
x = [t + p for t, p in zip(tok_emb, pos_emb)]
```

Each token is represented as the sum of two learned vectors: *what* the token is (token embedding) and *where* it sits in the sequence (position embedding). This is how the model knows that "a" in position 0 is different from "a" in position 5.

#### b) RMSNorm

```python
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]
```

Instead of LayerNorm (used in the original GPT-2), this implementation uses RMSNorm — a simpler variant that normalizes by the root mean square of the activations. It stabilizes training by keeping activation magnitudes in check, without needing to compute the mean.

#### c) Multi-Head Self-Attention

This is the mechanism that lets the model look at all previous tokens when deciding what comes next.

1. The current token's embedding is projected into **Query** (Q), **Key** (K), and **Value** (V) vectors
2. Q and K are split across 4 attention heads (each head sees a 4-dimensional slice)
3. For each head, attention scores are computed: `score = Q · K / √d`
4. Scores are passed through softmax to get attention weights
5. The weighted sum of V vectors becomes the head's output
6. All heads are concatenated and projected back to the embedding dimension

The KV cache (`keys` and `values` lists) stores previous tokens' K and V vectors so they don't need to be recomputed — the same optimization used in production LLM inference.

#### d) MLP Block (Feed-Forward Network)

```python
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # expand: 16 → 64
x = [xi.relu() ** 2 for xi in x]                    # squared ReLU activation
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # contract: 64 → 16
```

After attention, each token passes through a two-layer feed-forward network. The hidden dimension is 4× the embedding dimension (a standard GPT convention). The activation function is **squared ReLU** (`max(0, x)²`) instead of GeLU — simpler and still effective.

#### e) Residual Connections

Both the attention and MLP blocks use residual connections (`x = output + x_residual`). This lets gradients flow directly through the network during backpropagation, making deeper models trainable.

#### f) Output Logits

```python
logits = linear(x, state_dict['lm_head'])
```

The final embedding is projected to a vector of size `vocab_size`. Each element is a raw score (logit) representing how likely the model thinks each token is to come next.

### 6. The Training Loop

```python
for step in range(500):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

Training processes one name at a time. Each name is wrapped with BOS tokens: `[BOS, c, h, a, r, l, o, t, t, e, BOS]`. The model learns to predict each next character given all previous ones.

#### Loss Computation

```python
probs = softmax(logits)
loss_t = -probs[target_id].log()
```

For each position, the model outputs a probability distribution over the vocabulary. The loss is the **negative log-likelihood** of the correct next token — the standard language modeling objective. If the model assigns probability 1.0 to the right answer, the loss is 0. If it assigns 0.01, the loss is 4.6. The model is incentivized to put as much probability mass as possible on the correct next token.

The final loss is the average across all positions in the sequence.

#### Backpropagation

```python
loss.backward()
```

One call walks the entire computation graph backward, computing gradients for every parameter. This is the chain rule applied recursively — the same algorithm that powers all of deep learning.

#### Adam Optimizer

```python
lr_t = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
for i, p in enumerate(params):
    m[i] = beta1 * m[i] + (1 - beta1) * p.grad
    v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
    m_hat = m[i] / (1 - beta1 ** (step + 1))
    v_hat = v[i] / (1 - beta2 ** (step + 1))
    p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
    p.grad = 0
```

Adam maintains two running averages per parameter:
- **m**: the mean of recent gradients (momentum — keeps updates moving in a consistent direction)
- **v**: the mean of recent squared gradients (adapts the learning rate per-parameter)

The bias correction (`m_hat`, `v_hat`) compensates for the fact that `m` and `v` are initialized at zero.

The learning rate follows a **cosine decay** schedule — starting at `0.01` and smoothly decreasing to 0 over the 500 steps. This is a common trick: large steps early for fast progress, small steps later for fine-tuning.

### 7. Inference (Text Generation)

```python
temperature = 0.5
logits = gpt(token_id, pos_id, keys, values)
probs = softmax([l / temperature for l in logits])
token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
```

Generation is autoregressive: start with BOS, predict the next token, feed it back in, repeat until BOS appears again (signaling end of name) or the maximum length is reached.

The **temperature** parameter controls randomness:
- `temperature → 0`: always pick the most likely token (deterministic, repetitive)
- `temperature = 1.0`: sample from the raw distribution (creative, sometimes chaotic)
- `temperature = 0.5`: a middle ground — diverse but coherent

---

## How GPT-2 Differs From This

This is a faithful miniature of the GPT architecture. Here's what changes when you scale up:

| | microGPT | GPT-2 (small) |
|---|---|---|
| Parameters | ~7,000 | 124,000,000 |
| Layers | 1 | 12 |
| Embedding dim | 16 | 768 |
| Attention heads | 4 | 12 |
| Context length | 8 | 1024 |
| Vocabulary | 27 (characters) | 50,257 (BPE subwords) |
| Normalization | RMSNorm | LayerNorm |
| Activation | Squared ReLU | GeLU |
| Compute | Scalar autograd | Tensor operations (CUDA) |

The *algorithm* is the same. The difference is scale and efficiency.

---

## Running It

```bash
python microgpt.py
```

That's it. No `pip install` needed. The script will download the names dataset if it's not present, train for 500 steps (takes a few minutes on a laptop), and print 20 generated names.

---

## Why This Matters

Most GPT tutorials either stay at a high level ("attention is all you need") or immediately dive into PyTorch tensors. This project occupies a rare middle ground: every operation is explicit, every gradient is computed from first principles, and the entire thing fits in a single readable file.

If you understand this code, you understand the core algorithm behind ChatGPT, Claude, and every other transformer language model. The rest is engineering.

---

## Credits

Created by [Andrej Karpathy](https://github.com/karpathy). The original code is published at [karpathy.ai/microgpt.html](https://karpathy.ai/microgpt.html). The autograd engine is a spiritual descendant of [micrograd](https://github.com/karpathy/micrograd), and the model architecture follows [nanoGPT](https://github.com/karpathy/nanoGPT) — both excellent companion projects.
