"""
microGPT: Grouped-Query Attention (GQA/MQA) edition.

Shows the progression from Multi-Head Attention (MHA) → Multi-Query Attention
(MQA) → Grouped-Query Attention (GQA). The key idea: KV heads can be shared
across multiple query heads. At production scale this often preserves most of
the quality while dramatically reducing memory for the KV cache during inference.

Same base architecture as lab 03 (PyTorch), but the attention module is
parameterised by n_kv_head so a single class covers all three variants.

The standard multi-head attention mechanism was introduced in "Attention Is All
You Need" by Vaswani et al. (2017) (https://arxiv.org/abs/1706.03762). Shazeer
(2019) later proposed Multi-Query Attention in "Fast Transformer Decoding: One
Write-Head is All You Need" (https://arxiv.org/abs/1911.02150), which reduces
the KV cache to a single head. Ainslie et al. (2023) generalised this idea in
"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
Checkpoints" (https://arxiv.org/abs/2305.13245), showing that an intermediate
number of KV heads offers a favourable trade-off between quality and efficiency.

This implementation demonstrates the core concept of KV head sharing as
described in Ainslie et al., using a simplified single-file transformer so the
mechanism is easy to study in isolation.
"""

import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

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
# Model config
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class FlexAttention(nn.Module):
    """Unified attention that covers MHA, GQA, and MQA via n_kv_head."""

    def __init__(self, n_kv_head):
        super().__init__()
        self.n_kv_head = n_kv_head
        assert n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.repeats = n_head // n_kv_head

        # Q projection: always full heads
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        # K, V projections: only n_kv_head heads
        self.wk = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.wv = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        # Output projection
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        # Project queries (full heads), keys and values (fewer heads)
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)  # (B, n_head, T, hd)
        k = self.wk(x).view(B, T, self.n_kv_head, head_dim).transpose(1, 2)  # (B, n_kv_head, T, hd)
        v = self.wv(x).view(B, T, self.n_kv_head, head_dim).transpose(1, 2)  # (B, n_kv_head, T, hd)

        # Expand KV heads to match Q heads by repeating. This is the clearest way
        # to write it and the worst way to run it: the expansion materialises
        # `repeats` identical copies of K and V, throwing away the memory saving
        # for the duration of the forward pass. Production kernels never do this:
        # PyTorch's scaled_dot_product_attention with enable_gqa=True, and
        # FlashAttention's GQA support, index the shared KV head directly from
        # each query head instead, with the same maths and no copies.
        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=1)  # (B, n_head, T, hd)
            v = v.repeat_interleave(self.repeats, dim=1)  # (B, n_head, T, hd)

        # Standard scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_kv_head):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = FlexAttention(n_kv_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, n_kv_head):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block(n_kv_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training + comparison
# ---------------------------------------------------------------------------
device = "cpu"
num_steps = 1000
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high

variants = [
    ("MHA", n_head),  # 4 KV heads, standard multi-head attention
    ("GQA", 2),  # 2 KV heads, grouped-query attention
    ("MQA", 1),  # 1 KV head, multi-query attention
]


def generate(model, label, num_samples=10):
    """Generate names from a trained model."""
    model.eval()
    print(f"\n  --- {label} generated names ---")
    with torch.no_grad():
        for sample_idx in range(num_samples):
            tokens = [BOS]
            for _ in range(block_size):
                idx = torch.tensor([tokens[-block_size:]], device=device)
                logits = model(idx)
                logits = logits[0, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                if token_id == BOS:
                    break
                tokens.append(token_id)
            name = "".join(uchars[t] for t in tokens[1:])
            print(f"  sample {sample_idx + 1:2d}: {name}")


docs_snapshot = list(docs)
results = []  # (label, n_kv_head, params, kv_cache_bytes, trailing avg loss)

for variant_name, n_kv_head in variants:
    print(f"\n{'=' * 60}")
    print(f"  {variant_name}: n_head={n_head}, n_kv_head={n_kv_head}")
    print(f"{'=' * 60}")

    # Reset seed and restore original doc order for fair comparison
    random.seed(42)
    torch.manual_seed(42)
    docs = list(docs_snapshot)

    model = MicroGPT(n_kv_head).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num params: {num_params}")

    # KV cache size. This is arithmetic, not a measurement: there is no KV cache
    # in this lab at all (that is lab 12). It is the size the cache *would* have
    # at fp16, batch 1, a full block_size context, which is the number that
    # matters in production, where the cache is what fills the GPU.
    kv_cache_bytes = n_kv_head * head_dim * block_size * 2 * 2 * n_layer
    print(
        f"KV cache size if cached (fp16, batch 1, analytic): {kv_cache_bytes} bytes "
        f"({n_kv_head} kv_heads x {head_dim} head_dim x {block_size} seq_len x 2 tensors x 2 bytes x {n_layer} layers)"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)

    losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        input_ids = torch.tensor([tokens[:n]], device=device)
        targets = torch.tensor([tokens[1 : n + 1]], device=device)

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        lr_t = 1e-2 * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

    # Report a trailing average, never the last step's loss. Batch size is one
    # name, so a single step's loss is dominated by which name it happened to
    # land on: a short name and a long one differ by more than these three
    # variants do. Averaging the tail is still not a quality ranking (see the
    # note at the end), but at least it is not pure sampling noise.
    avg_window = 100
    avg_loss = sum(losses[-avg_window:]) / avg_window
    print(f"\n  {variant_name} trailing avg loss (last {avg_window} steps): {avg_loss:.4f}")
    results.append((variant_name, n_kv_head, num_params, kv_cache_bytes, avg_loss))
    generate(model, variant_name)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("  Summary: KV head sharing vs parameter count")
print(f"{'=' * 60}")
mha_cache = n_head * head_dim * block_size * 2 * n_layer * 2  # MHA is the reference point
print(f"  {'':3s} {'kv_heads':>8} {'KV proj':>8} {'total':>7} {'cache B':>8} {'vs MHA':>7} {'avg loss':>9}")
for variant_name, n_kv_head, num_params, kv_cache, avg_loss in results:
    kv_params = 2 * n_kv_head * head_dim * n_embd * n_layer  # wk + wv params
    print(
        f"  {variant_name:3s} {n_kv_head:>8} {kv_params:>8} {num_params:>7} "
        f"{kv_cache:>8} {kv_cache / mha_cache:>6.2f}x {avg_loss:>9.4f}"
    )
print("""
  The cache ratio is the transferable number. Raw bytes here are absurd, since this
  model is 16-dimensional, but 1.00x / 0.50x / 0.25x is exactly what you get at
  any scale, because the cache is linear in n_kv_head. LLaMA 2 70B runs 8 KV
  heads against 64 query heads: 0.125x. And note this is arithmetic, not a
  measurement: nothing in this lab actually caches anything (that is lab 12).

  What this lab does NOT show is quality. Three 1-layer, 16-dim models trained
  for 1000 single-name steps produce losses separated by less than the run-to-run
  noise, and whichever variant happens to come out lowest, that is not evidence.
  Read the loss column as "all three still learn to spell names", nothing more.
  The claim that GQA holds quality while cutting the cache is real, and it comes
  from Ainslie et al. (2023) uptraining 64-head checkpoints on real corpora,
  not from anything printed above.""")
