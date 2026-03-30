"""
microGPT — Soft thinking edition.

Same architecture as the PyTorch version (03), but with soft decoding at
inference time. Instead of collapsing to a single token at each step, soft
thinking passes a "concept token" -- a probability-weighted blend of all
token embeddings -- to the next step. The full distribution flows forward,
preserving information that hard decoding discards.

  Hard:  logits -> sample -> embed(token)            -> next input
  Soft:  logits -> softmax(logits/T) @ embed_table   -> next input

Based on "Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous
Concept Space" (Zhang et al., 2025), https://arxiv.org/abs/2505.15778. The
concept token computation follows the paper's formulation. Note that this is an
inference-only technique -- the model is trained with standard teacher forcing on
hard tokens, which creates a train-test distribution mismatch that Lab 18 (soft
training) addresses.
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
# Model (same as Lab 03, with inputs_embeds support for concept tokens)
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head
device = "cpu"


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

    def forward(self, idx=None, inputs_embeds=None):
        tok_emb = self.wte(idx) if idx is not None else inputs_embeds
        assert tok_emb is not None, "provide idx or inputs_embeds"
        T = tok_emb.shape[1]
        pos_emb = self.wpe(torch.arange(T, device=tok_emb.device))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Shared utilities (also used by Lab 18)
# ---------------------------------------------------------------------------
num_steps = 1000


@torch.no_grad()
def generate(model, mode="hard", soft_temp=1.0, temperature=0.5):
    """Generate a name using hard (discrete) or soft (concept token) decoding."""
    model.eval()
    tokens, entropies = [], []
    embeds = model.wte(torch.tensor([[BOS]], device=device))

    for _ in range(block_size):
        logits = model(inputs_embeds=embeds)[0, -1]

        # Sample a discrete token (for output and stopping)
        probs = F.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        if token_id == BOS:
            break
        tokens.append(token_id)

        # Track entropy: how spread is the distribution?
        entropies.append(-(probs * probs.clamp(min=1e-10).log()).sum().item())

        # Next input: discrete embedding or concept token
        if mode == "hard":
            next_emb = model.wte(torch.tensor([[token_id]], device=device))
        else:
            # Concept token: probability-weighted blend of ALL token embeddings
            soft_probs = F.softmax(logits / soft_temp, dim=-1)
            next_emb = (soft_probs @ model.wte.weight).view(1, 1, -1)

        embeds = torch.cat([embeds, next_emb], dim=1)[:, -block_size:]

    return tokens, entropies


# ---------------------------------------------------------------------------
# Training & Inference (runs only when executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = MicroGPT().to(device)
    print(f"num params: {sum(p.numel() for p in model.parameters())}")

    # Training — identical to Lab 03
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)

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
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

    # Inference — hard vs. soft decoding at different temperatures
    print("\n--- soft thinking comparison ---\n")
    n_samples = 20
    max_H = math.log(vocab_size)

    for mode, soft_temp, label in [
        ("hard", 1.0, "hard (standard decoding)"),
        ("soft", 0.5, "soft T=0.5 (mild blend)"),
        ("soft", 1.0, "soft T=1.0 (moderate blend)"),
        ("soft", 2.0, "soft T=2.0 (diffuse blend)"),
    ]:
        print(f"{label}:")
        all_H = []
        for i in range(n_samples):
            toks, ents = generate(model, mode, soft_temp)
            name = "".join(uchars[t] for t in toks)
            avg_H = sum(ents) / len(ents) if ents else 0
            all_H.append(avg_H)
            if i < 10:
                print(f"  {i + 1:2d}: {name:<15s} entropy {avg_H:.2f}/{max_H:.2f}")
        print(f"  -> mean entropy: {sum(all_H) / len(all_H):.2f}/{max_H:.2f}\n")

    print(f"""--- what's happening ---

Standard (hard) decoding collapses the model's rich output distribution to a
single token ID at every step. The next step sees only one embedding — all
information about what the model "almost said" is discarded.

Soft thinking preserves this information:

  concept_token = softmax(logits / T) @ embedding_table

The concept token is a {n_embd}-dimensional vector blending all {vocab_size} token
embeddings, weighted by their probability. It lives in the same embedding space
as regular tokens but encodes the model's full uncertainty.

Temperature (T) controls the softness:
  T -> 0:  concept token = argmax embedding (hard, no benefit)
  T = 1:   standard softmax (moderate blending)
  T -> inf: uniform weights (noise — all tokens equally blended)

The entropy column shows how "spread" each distribution is. Higher entropy
means more tokens contribute to the concept token — richer information, but
higher risk of out-of-distribution drift. Max entropy = ln({vocab_size}) = {max_H:.2f}.

This is training-free — no model weights change. The Soft Thinking paper
(Zhang et al., 2025) shows this improves reasoning in large models by
+2.5 pass@1 while using 22% fewer tokens.
""")
