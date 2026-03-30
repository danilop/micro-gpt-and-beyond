"""
microGPT — Sampling Strategies.

Same microGPT architecture as the PyTorch edition (03), but focused on
demonstrating how different sampling strategies shape model output.
Five strategies: greedy, temperature, top-k, top-p (nucleus), min-p.
Same model, completely different outputs.

Top-p (nucleus) sampling is from "The Curious Case of Neural Text Degeneration"
(Holtzman et al., 2020), https://arxiv.org/abs/1904.09751. Min-p sampling is
from "Min P Sampling: Balancing Creativity and Coherence at High Temperature"
(Nguyen et al., 2024), https://arxiv.org/abs/2407.01082. Temperature scaling
and top-k are standard techniques. Top-p and min-p are the most widely used in
production -- top-p dynamically adjusts the candidate set based on distribution
shape, while min-p offers a simpler adaptive threshold.
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
# Model
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

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
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
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
# Training
# ---------------------------------------------------------------------------
device = "cpu"
model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

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

    if (step + 1) % 200 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# ---------------------------------------------------------------------------
# Sampling strategies
# ---------------------------------------------------------------------------


def sample_greedy(logits):
    """Always pick the highest probability token. Deterministic but repetitive."""
    return logits.argmax().item()


def sample_temperature(logits, T=1.0):
    """Scale logits by 1/T before softmax.
    T->0: approaches greedy (peaked). T=1: model's learned distribution.
    T->inf: approaches uniform random."""
    probs = F.softmax(logits / T, dim=-1)
    return torch.multinomial(probs, 1).item()


def sample_top_k(logits, k=5, T=1.0):
    """Keep only top-k logits, set rest to -inf, then sample with temperature.
    Prevents sampling from the long tail of unlikely tokens."""
    scaled = logits / T
    topk_vals, _ = torch.topk(scaled, k)
    scaled[scaled < topk_vals[-1]] = float("-inf")
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, 1).item()


def sample_top_p(logits, p=0.9, T=1.0):
    """Nucleus sampling: sort by probability, keep smallest set whose
    cumulative probability >= p, zero out rest, renormalize, sample.
    Dynamically adjusts candidates based on distribution shape."""
    probs = F.softmax(logits / T, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens with cumulative probability above p (keep first token always)
    mask = cumsum - sorted_probs > p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    # Scatter back to original order
    out = torch.zeros_like(probs)
    out.scatter_(0, sorted_idx, sorted_probs)
    return torch.multinomial(out, 1).item()


def sample_min_p(logits, min_p=0.05, T=1.0):
    """Keep tokens with probability >= min_p * max_probability.
    Simpler than top-p, adapts to distribution shape naturally.
    Peaked distributions: few tokens pass. Flat: many pass."""
    probs = F.softmax(logits / T, dim=-1)
    threshold = min_p * probs.max()
    probs[probs < threshold] = 0.0
    probs /= probs.sum()
    return torch.multinomial(probs, 1).item()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate(model, sampler_fn, n=20):
    """Generate n names using the given sampling function."""
    names = []
    for _ in range(n):
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], device=device)
            logits = model(idx)
            raw_logits = logits[0, -1]
            token_id = sampler_fn(raw_logits.clone())
            if token_id == BOS:
                break
            tokens.append(token_id)
        name = "".join(uchars[t] for t in tokens[1:])
        names.append(name)
    return names


# ---------------------------------------------------------------------------
# Compare all strategies
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SAMPLING STRATEGY COMPARISON")
print("=" * 60)

model.eval()
with torch.no_grad():
    strategies = [
        ("Greedy (T→0)", lambda lg: sample_greedy(lg)),
        ("Temperature 0.3 (conservative)", lambda lg: sample_temperature(lg, T=0.3)),
        ("Temperature 1.0 (model dist)", lambda lg: sample_temperature(lg, T=1.0)),
        ("Temperature 2.0 (chaotic)", lambda lg: sample_temperature(lg, T=2.0)),
        ("Top-k=3, T=0.8", lambda lg: sample_top_k(lg, k=3, T=0.8)),
        ("Top-k=10, T=0.8", lambda lg: sample_top_k(lg, k=10, T=0.8)),
        ("Top-p=0.5, T=0.8", lambda lg: sample_top_p(lg, p=0.5, T=0.8)),
        ("Top-p=0.95, T=0.8", lambda lg: sample_top_p(lg, p=0.95, T=0.8)),
        ("Min-p=0.1, T=0.8", lambda lg: sample_min_p(lg, min_p=0.1, T=0.8)),
        ("Min-p=0.01, T=0.8", lambda lg: sample_min_p(lg, min_p=0.01, T=0.8)),
    ]

    for label, sampler in strategies:
        # Reset RNG for reproducibility within each strategy
        torch.manual_seed(42)
        names = generate(model, sampler, n=20)
        print(f"\n--- {label} ---")
        for i, name in enumerate(names):
            print(f"  {i + 1:2d}. {name}")

# ---------------------------------------------------------------------------
# Visualization: probability distributions after applying each filter
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PROBABILITY DISTRIBUTION VISUALIZATION")
print("=" * 60)

# Show how each strategy reshapes the probability distribution for
# the same input. We encode "mar" and look at next-token probabilities.
prefix_str = "mar"
prefix_tokens = [BOS] + [uchars.index(ch) for ch in prefix_str]

with torch.no_grad():
    idx = torch.tensor([prefix_tokens], device=device)
    logits = model(idx)
    raw_logits = logits[0, -1]  # logits for next token after "mar"

    def bar_chart(label, probs, top_n=10):
        """Print a horizontal bar chart of the top-n token probabilities."""
        n_active = (probs > 0).sum().item()
        print(f"\n  {label}  ({n_active} active tokens)")
        n = min(top_n, n_active)
        vals, idxs = torch.topk(probs, n)
        bar_width = 30  # max bar length in characters
        for v, i in zip(vals, idxs):
            tok = uchars[i.item()] if i.item() < len(uchars) else "."
            p = v.item()
            bar = "#" * int(p * bar_width + 0.5)
            print(f"    {tok} {bar:<{bar_width}} {p:.0%}")

    print(f'\nNext token after "{prefix_str}":')

    # Raw distribution (T=1.0)
    raw_probs = F.softmax(raw_logits, dim=-1)
    bar_chart("Raw (T=1.0)", raw_probs)

    # Temperature 0.3
    t03_probs = F.softmax(raw_logits / 0.3, dim=-1)
    bar_chart("T=0.3 (conservative)", t03_probs)

    # Temperature 2.0
    t20_probs = F.softmax(raw_logits / 2.0, dim=-1)
    bar_chart("T=2.0 (chaotic)", t20_probs)

    # Top-k=3
    topk_logits = raw_logits.clone() / 0.8
    topk_vals, _ = torch.topk(topk_logits, 3)
    topk_logits[topk_logits < topk_vals[-1]] = float("-inf")
    topk_probs = F.softmax(topk_logits, dim=-1)
    bar_chart("Top-k=3, T=0.8", topk_probs)

    # Top-p=0.9
    topp_probs = F.softmax(raw_logits / 0.8, dim=-1)
    sorted_probs, sorted_idx = torch.sort(topp_probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > 0.9
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()
    topp_out = torch.zeros_like(topp_probs)
    topp_out.scatter_(0, sorted_idx, sorted_probs)
    bar_chart("Top-p=0.9, T=0.8", topp_out)

    # Min-p=0.1
    minp_probs = F.softmax(raw_logits / 0.8, dim=-1)
    threshold = 0.1 * minp_probs.max()
    minp_probs[minp_probs < threshold] = 0.0
    minp_probs /= minp_probs.sum()
    bar_chart("Min-p=0.1, T=0.8", minp_probs)

print()
