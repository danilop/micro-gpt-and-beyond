"""
microGPT — Rotary Position Embeddings (RoPE).

Same architecture as version 03 (PyTorch), but learned positional embeddings are
replaced with Rotary Position Embeddings. RoPE encodes position by rotating query
and key vectors in complex space, so the dot product q·k naturally depends on the
*relative* distance (m-n) rather than absolute positions m and n separately.
Every modern LLM (LLaMA, Mistral, GPT-NeoX) uses RoPE. This lab shows why.

The RoPE technique is from "RoFormer: Enhanced Transformer with Rotary Position
Embedding" (Su et al., 2021), https://arxiv.org/abs/2104.09864. The implementation
uses the real-valued rotation form (pairs of adjacent dimensions), matching the
approach used in production by LLaMA ("LLaMA: Open and Efficient Foundation
Language Models", Touvron et al., 2023, https://arxiv.org/abs/2302.13971). Note
that the base frequency of 10000 follows the original paper; later work like
"YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al.,
2023, https://arxiv.org/abs/2309.00071) explores different base frequencies for
length extension.
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
# RoPE helpers
# ---------------------------------------------------------------------------
def precompute_freqs(dim, max_len):
    """Precompute rotation frequencies for RoPE.

    theta_i = 1 / (10000 ^ (2i / dim))  for i in 0..dim//2
    Returns cos and sin of shape (max_len, dim//2).
    """
    i = torch.arange(0, dim, 2, dtype=torch.float32)  # (dim//2,)
    theta = 1.0 / (10000.0 ** (i / dim))  # (dim//2,)
    positions = torch.arange(max_len, dtype=torch.float32)  # (max_len,)
    angles = torch.outer(positions, theta)  # (max_len, dim//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos_freqs, sin_freqs):
    """Apply rotary embeddings to x of shape (B, n_head, T, head_dim).

    Split head_dim into pairs, rotate each pair:
        [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    """
    T = x.shape[2]
    cos_t = cos_freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim//2)
    sin_t = sin_freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim//2)
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t
    return torch.stack((out1, out2), dim=-1).flatten(-2)


# Precompute once for the whole model
rope_cos, rope_sin = precompute_freqs(head_dim, block_size)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            # V is NOT rotated — RoPE only affects Q and K

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
    def __init__(self, use_rope=False):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(use_rope=use_rope)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.wte = nn.Embedding(vocab_size, n_embd)
        if not use_rope:
            self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block(use_rope=use_rope) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        if not self.use_rope:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training + comparison
# ---------------------------------------------------------------------------
def train(model, label, num_steps=1000):
    """Train a model and return the loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
    losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        input_ids = torch.tensor([tokens[:n]])
        targets = torch.tensor([tokens[1 : n + 1]])

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        lr_t = 1e-2 * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  [{label}] step {step + 1:4d}/{num_steps} | loss {loss.item():.4f}")

    return losses


def generate(model, label, num_samples=10, temperature=0.5):
    """Generate names from a trained model."""
    model.eval()
    print(f"\n--- {label}: generated names ---")
    with torch.no_grad():
        for i in range(num_samples):
            tokens = [BOS]
            for _ in range(block_size):
                idx = torch.tensor([tokens[-block_size:]])
                logits = model(idx)
                logits = logits[0, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                if token_id == BOS:
                    break
                tokens.append(token_id)
            name = "".join(uchars[t] for t in tokens[1:])
            print(f"  sample {i + 1:2d}: {name}")
    model.train()


# Train both variants from the same random initialization
print("\n=== Training: Learned Positional Embeddings (baseline) ===")
torch.manual_seed(42)
model_learned = MicroGPT(use_rope=False)
n_params_learned = sum(p.numel() for p in model_learned.parameters())
print(f"  params: {n_params_learned}")
losses_learned = train(model_learned, "learned-pos")

print("\n=== Training: Rotary Position Embeddings (RoPE) ===")
torch.manual_seed(42)
model_rope = MicroGPT(use_rope=True)
n_params_rope = sum(p.numel() for p in model_rope.parameters())
print(f"  params: {n_params_rope} (no wpe — {n_params_learned - n_params_rope} fewer)")
losses_rope = train(model_rope, "rope")

# Compare final losses
print("\n=== Results ===")
avg_window = 50
avg_learned = sum(losses_learned[-avg_window:]) / avg_window
avg_rope = sum(losses_rope[-avg_window:]) / avg_window
print(f"  learned-pos  final avg loss (last {avg_window}): {avg_learned:.4f}")
print(f"  rope         final avg loss (last {avg_window}): {avg_rope:.4f}")

generate(model_learned, "Learned Positional Embeddings")
generate(model_rope, "RoPE")
