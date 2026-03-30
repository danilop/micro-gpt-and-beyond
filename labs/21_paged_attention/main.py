"""
microGPT — PagedAttention edition.

Same architecture as the pure-Python version (01), but demonstrating two
KV cache strategies for inference: contiguous (wasteful) and paged (efficient).
PagedAttention is the core innovation in vLLM -- it applies the operating
system's virtual memory paging concept to KV caches, achieving near-perfect
memory utilization.

Zero dependencies. Pure Python. The algorithms are pure data structures.

Based on "Efficient Memory Management for Large Language Model Serving with
PagedAttention" (Kwon et al., 2023), https://arxiv.org/abs/2309.06180, the
core innovation behind vLLM. The concept directly parallels virtual memory
paging in operating systems, as described in standard texts like "Operating
System Concepts" (Silberschatz et al.). This lab implements the memory
management side of PagedAttention (block tables, on-demand allocation,
copy-on-write prefix sharing) -- not the fused PagedAttention CUDA attention
kernel itself. The pure-Python scalar implementation makes the data structure
logic fully transparent.
"""

import math
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model — reuse Lab 01's Value autograd + GPT architecture
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


# fmt: off
class Value:
    __slots__ = ('_children', '_local_grads', 'data', 'grad') # Python optimization for memory usage

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


def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    for name in ("attn_wq", "attn_wk", "attn_wv", "attn_wo"):
        state_dict[f"layer{i}.{name}"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    if isinstance(logits[0], Value):
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
    else:
        max_val = max(logits)
        exps = [math.exp(l - max_val) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    scale = (sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def relu(x):
    if isinstance(x[0], Value):
        return [xi.relu() for xi in x]
    return [max(0.0, xi) for xi in x]


# ---------------------------------------------------------------------------
# KV Cache Managers
# ---------------------------------------------------------------------------


class ContiguousKVCache:
    """Standard KV cache: pre-allocate max_seq_len slots per layer."""

    def __init__(self, max_seq_len, n_layers, dims):
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.dims = dims
        self.k_cache = [[[0.0] * dims for _ in range(max_seq_len)] for _ in range(n_layers)]
        self.v_cache = [[[0.0] * dims for _ in range(max_seq_len)] for _ in range(n_layers)]
        self.length = 0
        self.allocated = n_layers * max_seq_len * dims * 2
        self.label = "contiguous"

    def new_sequence(self, seq_id=0):
        self.length = 0

    def append(self, seq_id, layer, k, v):
        self.k_cache[layer][self.length] = k
        self.v_cache[layer][self.length] = v
        if layer == self.n_layers - 1:
            self.length += 1

    def get_kv(self, seq_id, layer):
        return self.k_cache[layer][: self.length], self.v_cache[layer][: self.length]

    def utilization(self):
        used = self.length * self.n_layers * self.dims * 2
        return used / self.allocated if self.allocated > 0 else 0


BLOCK_SIZE_TOKENS = 4


class PagedKVCache:
    """
    Paged KV cache (PagedAttention / vLLM concept).
    Allocate fixed-size blocks on demand. A block table maps logical positions
    to physical block IDs — just like page tables in an operating system.
    """

    def __init__(self, total_blocks, n_layers, dims, block_size_tokens=BLOCK_SIZE_TOKENS):
        self.block_size = block_size_tokens
        self.n_layers = n_layers
        self.dims = dims
        self.total_blocks = total_blocks
        self.k_blocks = [[[0.0] * dims for _ in range(block_size_tokens)] for _ in range(total_blocks)]
        self.v_blocks = [[[0.0] * dims for _ in range(block_size_tokens)] for _ in range(total_blocks)]
        self.free_blocks = list(range(total_blocks))
        self.refcounts = {i: 0 for i in range(total_blocks)}
        self.block_tables = {}
        self.seq_lengths = {}
        self.alloc_events = []
        self.label = "paged"

    def _alloc_block(self):
        if not self.free_blocks:
            raise RuntimeError("Out of physical blocks!")
        blk = self.free_blocks.pop()
        self.refcounts[blk] = 1
        return blk

    def new_sequence(self, seq_id):
        self.block_tables[seq_id] = {li: [] for li in range(self.n_layers)}
        self.seq_lengths[seq_id] = 0

    def append(self, seq_id, layer, k, v):
        length = self.seq_lengths[seq_id]
        logical_block = length // self.block_size
        slot_in_block = length % self.block_size
        if logical_block >= len(self.block_tables[seq_id][layer]):
            phys_id = self._alloc_block()
            self.block_tables[seq_id][layer].append(phys_id)
            self.alloc_events.append(
                f"  seq {seq_id}, layer {layer}, logical block {logical_block} -> physical block {phys_id}"
            )
        phys_id = self.block_tables[seq_id][layer][logical_block]
        # Copy-on-write: clone shared block before writing
        if self.refcounts[phys_id] > 1:
            new_id = self._alloc_block()
            self.k_blocks[new_id] = [row[:] for row in self.k_blocks[phys_id]]
            self.v_blocks[new_id] = [row[:] for row in self.v_blocks[phys_id]]
            self.refcounts[phys_id] -= 1
            self.block_tables[seq_id][layer][logical_block] = new_id
            phys_id = new_id
        self.k_blocks[phys_id][slot_in_block] = k
        self.v_blocks[phys_id][slot_in_block] = v
        if layer == self.n_layers - 1:
            self.seq_lengths[seq_id] = length + 1

    def get_kv(self, seq_id, layer):
        length = self.seq_lengths[seq_id]
        ks, vs = [], []
        for pos in range(length):
            logical_block = pos // self.block_size
            slot = pos % self.block_size
            phys_id = self.block_tables[seq_id][layer][logical_block]
            ks.append(self.k_blocks[phys_id][slot])
            vs.append(self.v_blocks[phys_id][slot])
        return ks, vs

    def free_sequence(self, seq_id):
        for layer_blocks in self.block_tables[seq_id].values():
            for blk in layer_blocks:
                self.refcounts[blk] -= 1
                if self.refcounts[blk] == 0:
                    self.free_blocks.append(blk)
        del self.block_tables[seq_id]
        del self.seq_lengths[seq_id]

    def utilization(self):
        used_blocks = self.total_blocks - len(self.free_blocks)
        if used_blocks <= 0:
            return 0.0

        occupied_slots = set()
        for seq_id, length in self.seq_lengths.items():
            for layer in range(self.n_layers):
                for pos in range(length):
                    logical_block = pos // self.block_size
                    slot = pos % self.block_size
                    phys_id = self.block_tables[seq_id][layer][logical_block]
                    occupied_slots.add((phys_id, slot))

        return len(occupied_slots) / (used_blocks * self.block_size)

    def share_prefix(self, src_seq_id, dst_seq_id, prefix_len):
        """Copy-on-write prefix sharing: point dst's block table at src's physical blocks."""
        self.new_sequence(dst_seq_id)
        n_shared_blocks = -(-prefix_len // self.block_size)  # ceiling division
        shared = 0
        for layer in range(self.n_layers):
            for b in range(min(n_shared_blocks, len(self.block_tables[src_seq_id][layer]))):
                phys_id = self.block_tables[src_seq_id][layer][b]
                self.block_tables[dst_seq_id][layer].append(phys_id)
                self.refcounts[phys_id] += 1
                shared += 1
        self.seq_lengths[dst_seq_id] = prefix_len
        return shared


# ---------------------------------------------------------------------------
# GPT forward pass (unified for training and inference)
# ---------------------------------------------------------------------------


def gpt(token_id, pos_id, cache, seq_id=0, W=None):
    if W is None:
        W = state_dict
    x = [t + p for t, p in zip(W["wte"][token_id], W["wpe"][pos_id])]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q, k, v = (linear(x, W[f"layer{li}.attn_w{n}"]) for n in ("q", "k", "v"))
        cache.append(seq_id, li, k, v)
        all_k, all_v = cache.get_kv(seq_id, li)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in all_k]
            v_h = [vi[hs : hs + head_dim] for vi in all_v]
            scores = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            weights = softmax(scores)
            head_out = [sum(weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, W[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_res)]
        x_res = x
        x = rmsnorm(x)
        x = linear(x, W[f"layer{li}.mlp_fc1"])
        x = relu(x)
        x = linear(x, W[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_res)]
    return linear(x, W["lm_head"])


# ---------------------------------------------------------------------------
# Training (identical to Lab 01, but using ContiguousKVCache)
# ---------------------------------------------------------------------------
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    cache = ContiguousKVCache(block_size, n_layer, n_embd)
    cache.new_sequence()
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, cache)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())
    loss = (1 / n) * sum(losses)
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0
    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

# ---------------------------------------------------------------------------
# Inference setup
# ---------------------------------------------------------------------------
W = {name: [[v.data for v in row] for row in mat] for name, mat in state_dict.items()}


def generate(cache, seq_id=0, max_len=block_size, temperature=0.5):
    cache.new_sequence(seq_id)
    token_id, sample = BOS, []
    for pos_id in range(max_len):
        logits = gpt(token_id, pos_id, cache, seq_id=seq_id, W=W)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        sample.append(token_id)
    return sample, cache


# ---------------------------------------------------------------------------
# Compare contiguous vs paged KV cache
# ---------------------------------------------------------------------------
print("\n--- contiguous vs paged KV cache ---\n")

cache_c = ContiguousKVCache(block_size, n_layer, n_embd)
random.seed(100)
sample_c, _ = generate(cache_c)
name_c = "".join(uchars[t] for t in sample_c)
used_c = cache_c.length * n_layer * n_embd * 2
print(f"contiguous: '{name_c}' ({cache_c.length} tokens)")
print(f"  allocated {cache_c.allocated} slots, used {used_c} ({cache_c.utilization():.0%} slot utilization)")

total_blocks = 32
paged = PagedKVCache(total_blocks, n_layer, n_embd)
random.seed(100)
sample_p, _ = generate(paged, seq_id=0)
name_p = "".join(uchars[t] for t in sample_p)
print(f"paged:      '{name_p}' ({paged.seq_lengths[0]} tokens)")
print(
    f"  {total_blocks - len(paged.free_blocks)} blocks allocated on demand "
    f"({paged.utilization():.0%} slot utilization)"
)
assert name_c == name_p, f"Outputs differ: {name_c} vs {name_p}"
print("  -> identical output, paged allocates only what's needed")

# ---------------------------------------------------------------------------
# Prefix sharing — shared system prompt
# ---------------------------------------------------------------------------
print("\n--- prefix sharing (shared system prompt) ---\n")

prefix_cache = PagedKVCache(total_blocks=64, n_layers=n_layer, dims=n_embd)
prefix_tokens = [BOS] + [uchars.index(ch) for ch in "an"]
prefix_cache.new_sequence("prefix")
for pos_id, token_id in enumerate(prefix_tokens):
    gpt(token_id, pos_id, prefix_cache, seq_id="prefix", W=W)

prefix_len = prefix_cache.seq_lengths["prefix"]
used_blk = prefix_cache.total_blocks - len(prefix_cache.free_blocks)
print(f"prefix 'an' cached: {prefix_len} tokens, {used_blk} blocks")

n_shared = 3
for i in range(n_shared):
    shared = prefix_cache.share_prefix("prefix", f"req_{i}", prefix_len)
    print(f"  req_{i}: shared {shared} blocks (zero-copy)")
print(f"blocks: {used_blk} total — sharing adds zero new allocations")
print(f"without sharing: {used_blk * (1 + n_shared)} blocks ({used_blk} x {1 + n_shared})")

print("""
At production scale (Llama 3 70B, 8K context, 100 requests):
contiguous needs ~200 GB, paged needs ~30-60 GB, prefix sharing saves
another 30-50%. This is why vLLM achieves 2-4x higher throughput.
""")

# ---------------------------------------------------------------------------
# Inference — generate names (using paged KV cache)
# ---------------------------------------------------------------------------
print("--- inference (paged KV cache) ---")
final_cache = PagedKVCache(total_blocks=128, n_layers=n_layer, dims=n_embd)
for sample_idx in range(20):
    random.seed(42 + sample_idx)
    sample, _ = generate(final_cache, seq_id=f"sample_{sample_idx}")
    name = "".join(uchars[t] for t in sample)
    print(f"sample {sample_idx + 1:2d}: {name}")
    final_cache.free_sequence(f"sample_{sample_idx}")
