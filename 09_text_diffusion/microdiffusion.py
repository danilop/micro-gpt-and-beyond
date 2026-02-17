"""
A masked diffusion language model trained and run in pure, dependency-free Python.
Instead of generating names left-to-right like a GPT, names emerge from pure noise
— all [MASK] tokens — through iterative unmasking. Same autograd, same transformer,
fundamentally different generative paradigm.

Autograd engine and transformer architecture from @karpathy's microGPT.
Diffusion framework from MDLM (Sahoo et al., 2024) and LLaDA (Nie et al., 2025).
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be an input dataset `docs`: list[str] of documents (e.g. a dataset of names)
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'input.txt')
if not os.path.exists(input_path):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, input_path)
docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer — same characters, but MASK replaces BOS and PAD handles fixed-length sequences
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
MASK = len(uchars)     # token id for [MASK] — the "noise" state that the model learns to denoise
PAD = len(uchars) + 1  # token id for [PAD] — fills unused positions in fixed-length sequences
vocab_size = len(uchars) + 2 # total number of unique tokens
print(f"vocab size: {vocab_size}")

# Let there be Autograd, to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model.
n_embd = 16     # embedding dimension
n_head = 4      # number of attention heads
n_layer = 2     # number of layers (diffusion needs depth to gather scattered clues)
block_size = 16 # maximum sequence length (names are padded/truncated to this)
head_dim = n_embd // n_head # dimension of each head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd)}
state_dict['lm_head'] = state_dict['wte'] # weight tying — same matrix for input embeddings and output projection
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
print(f"num params: {len(params)}")

# Define the model architecture: a bidirectional transformer that predicts clean tokens from masked input.
# Same building blocks as GPT-2 (rmsnorm, multi-head attention, MLP), but every position sees every other.
# No causal mask, no KV cache — the model looks in all directions to fill in the blanks.
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def mask_predictor(token_ids):
    """Takes a (partially masked) sequence, returns logits for every position."""
    n = len(token_ids)

    # Embed all tokens: what each token is + where it sits
    xs = [rmsnorm([t + p for t, p in zip(state_dict['wte'][tid], state_dict['wpe'][pos])])
          for pos, tid in enumerate(token_ids)]

    for li in range(n_layer):
        # 1) Multi-head bidirectional self-attention — every position attends to every other
        residuals = xs
        xs_norm = [rmsnorm(x) for x in xs]
        qs = [linear(x, state_dict[f'layer{li}.attn_wq']) for x in xs_norm]
        ks = [linear(x, state_dict[f'layer{li}.attn_wk']) for x in xs_norm]
        vs = [linear(x, state_dict[f'layer{li}.attn_wv']) for x in xs_norm]
        xs_attn = []
        for i in range(n):
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = qs[i][hs:hs+head_dim]
                attn_logits = [sum(q_h[j] * ks[t][hs+j] for j in range(head_dim)) / head_dim**0.5 for t in range(n)]
                attn_weights = softmax(attn_logits)
                head_out = [sum(attn_weights[t] * vs[t][hs+j] for t in range(n)) for j in range(head_dim)]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
            xs_attn.append([a + b for a, b in zip(x, residuals[i])])
        # 2) MLP block
        xs = []
        for x in xs_attn:
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
            xs.append([a + b for a, b in zip(x, x_residual)])

    return [linear(x, state_dict['lm_head']) for x in xs] # logits for every position

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer

# Repeat in sequence — but now with noise instead of next-token prediction
num_steps = 2000 # number of training steps
for step in range(num_steps):

    # Take a single document, tokenize it, pad to fixed length with PAD
    doc = docs[step % len(docs)]
    clean = [uchars.index(ch) for ch in doc] + [PAD] * (block_size - len(doc))
    clean = clean[:block_size] # truncate if longer than block_size

    # Forward process: corrupt the clean sequence by masking each token with probability t
    # Log-uniform t sampling: t ∝ 1/t, which cancels the 1/t ELBO weight (importance sampling)
    t = math.exp(random.uniform(math.log(0.2), 0))
    noisy = [MASK if random.random() < t else c for c in clean]
    n_masked = sum(1 for x in noisy if x == MASK)
    if n_masked == 0:
        continue # nothing to learn from if nothing was masked

    # Forward the noisy sequence through the model, compute loss only on masked positions
    all_logits = mask_predictor(noisy)
    losses = []
    for i in range(block_size):
        if noisy[i] == MASK:
            logits_i = all_logits[i][:]           # copy logits for this position
            logits_i[MASK] = logits_i[MASK] - 1e6 # never predict MASK (same as MDLM subs parameterization)
            probs = softmax(logits_i)
            losses.append(-probs[clean[i]].log())
    loss = (1.0 / n_masked) * sum(losses) # the 1/t cancels — just average cross-entropy. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters.
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients.
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# Inference: may names emerge from the noise
num_denoise_steps = 64 # how many steps to go from all-MASK to a clean name
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    seq = [MASK] * block_size # start from pure noise: every position is [MASK]

    for step_i in range(num_denoise_steps, 0, -1):
        t = math.cos(math.pi / 2 * (1 - step_i / num_denoise_steps))       # cosine schedule: more time in the middle
        s = math.cos(math.pi / 2 * (1 - (step_i - 1) / num_denoise_steps)) # next noise level (less noisy)
        temperature = 0.3 + 0.5 * t           # anneal: explore early (0.8), commit late (0.3)

        # Predict what each masked position should be
        all_logits = mask_predictor(seq)
        predicted = list(seq)
        confidences = [] # (confidence, position) — how sure the model is about each prediction
        for i in range(block_size):
            if seq[i] == MASK:
                logits_i = all_logits[i][:]           # copy logits for this position
                logits_i[MASK] = logits_i[MASK] - 1e6 # never predict MASK
                probs = softmax([l / temperature for l in logits_i])
                probs_data = [p.data for p in probs]
                predicted[i] = random.choices(range(vocab_size), weights=probs_data)[0]
                confidences.append((max(probs_data), i))

        # Remask: re-mask the least confident predictions (unless this is the final step)
        if s > 0 and confidences:
            n_to_remask = int(len(confidences) * s / t)
            confidences.sort() # lowest confidence first
            for _, i in confidences[:n_to_remask]:
                predicted[i] = MASK
        seq = predicted

    # Decode: strip PAD tokens and print the name
    name = ''.join(uchars[c] for c in seq if c < len(uchars))
    print(f"sample {sample_idx+1:2d}: {name}")
