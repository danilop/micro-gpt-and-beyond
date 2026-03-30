"""
microGPT — Soft training edition.

Builds on Lab 17 (soft thinking): instead of only using concept tokens at
inference time, this version also uses them during training. A curriculum
gradually replaces ground-truth token embeddings with the model's own soft
predictions, closing the train-test gap that limits inference-only soft thinking.

  Standard training:  input = embed(ground_truth_token)
  Soft training:      input = (1-mix) * embed(ground_truth) + mix * concept_token
                      where mix increases from 0 -> 1 during training

Extends the concept from "Soft Thinking" (Zhang et al., 2025),
https://arxiv.org/abs/2505.15778 by also using concept tokens during training.
The curriculum schedule (linear ramp from hard to soft inputs) is inspired by
"Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
(Bengio et al., 2015), https://arxiv.org/abs/1506.03099. Note that soft training
is an educational extension exploring how to close the train-test gap -- the
original Soft Thinking paper focuses on inference-time techniques.
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
# Model (same as Lab 17 — supports both token IDs and raw embeddings)
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
# Generation — hard (discrete) or soft (concept token) decoding
# ---------------------------------------------------------------------------
num_steps = 1000
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
soft_temp = 1.0


@torch.no_grad()
def generate(model, mode="hard", soft_temp=1.0):
    """Generate a name using hard (discrete) or soft (concept token) decoding."""
    model.eval()
    tokens, entropies = [], []
    embeds = model.wte(torch.tensor([[BOS]], device=device))

    for _ in range(block_size):
        logits = model(inputs_embeds=embeds)[0, -1]

        probs = F.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        if token_id == BOS:
            break
        tokens.append(token_id)

        entropies.append(-(probs * probs.clamp(min=1e-10).log()).sum().item())

        if mode == "hard":
            next_emb = model.wte(torch.tensor([[token_id]], device=device))
        else:
            soft_probs = F.softmax(logits / soft_temp, dim=-1)
            next_emb = (soft_probs @ model.wte.weight).view(1, 1, -1)

        embeds = torch.cat([embeds, next_emb], dim=1)[:, -block_size:]

    return tokens, entropies


# ---------------------------------------------------------------------------
# Training — standard vs. soft input mixing
# ---------------------------------------------------------------------------
def train_model(model, name, soft_mix=False):
    """Train with standard teacher forcing, or with soft input curriculum."""
    print(f"\n--- training {name} ({sum(p.numel() for p in model.parameters())} params) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        input_ids = torch.tensor([tokens[:n]], device=device)
        targets = torch.tensor([tokens[1 : n + 1]], device=device)

        if soft_mix and n > 1:
            mix = step / num_steps  # curriculum: 0 -> 1

            # Model's soft predictions (detached — no gradient through this path)
            with torch.no_grad():
                pred_logits = model(input_ids)
                soft_embeds = F.softmax(pred_logits / soft_temp, dim=-1) @ model.wte.weight

            # BOS stays ground truth; rest blended with concept tokens
            # soft_embeds[:, i] predicts position i+1, so shift left by one
            gt_embeds = model.wte(input_ids)
            mixed = torch.cat([gt_embeds[:, :1], (1 - mix) * gt_embeds[:, 1:] + mix * soft_embeds[:, :-1]], dim=1)

            logits = model(inputs_embeds=mixed)
        else:
            logits = model(input_ids)

        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        lr_t = 1e-2 * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        optimizer.step()

        if (step + 1) % 10 == 0 or step == 0:
            extra = f" | mix {mix:.2f}" if soft_mix and n > 1 else ""
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}{extra}")


# ---------------------------------------------------------------------------
# Train both models from the same initial weights (fair comparison)
# ---------------------------------------------------------------------------
standard_model = MicroGPT().to(device)
soft_model = MicroGPT().to(device)
soft_model.load_state_dict(standard_model.state_dict())

train_model(standard_model, "standard")
train_model(soft_model, "soft-trained", soft_mix=True)

# ---------------------------------------------------------------------------
# Compare: standard-trained vs. soft-trained, hard vs. soft decoding
# ---------------------------------------------------------------------------
print("\n--- comparison: standard-trained vs. soft-trained ---\n")
n_samples = 20
max_H = math.log(vocab_size)

configs = [
    (standard_model, "standard-trained", "hard", 1.0),
    (standard_model, "standard-trained", "soft", 1.0),
    (soft_model, "soft-trained", "hard", 1.0),
    (soft_model, "soft-trained", "soft", 1.0),
]

with torch.no_grad():
    for model_obj, model_name, mode, st in configs:
        model_obj.eval()
        print(f"{model_name} + {mode} decoding:")
        all_H = []
        for i in range(n_samples):
            toks, ents = generate(model_obj, mode, st)
            name = "".join(uchars[t] for t in toks)
            avg_H = sum(ents) / len(ents) if ents else 0
            all_H.append(avg_H)
            if i < 10:
                print(f"  {i + 1:2d}: {name:<15s} entropy {avg_H:.2f}/{max_H:.2f}")
        print(f"  -> mean entropy: {sum(all_H) / len(all_H):.2f}/{max_H:.2f}\n")

print("""--- what's happening ---

Standard training uses teacher forcing: the model always sees perfect
ground-truth embeddings as input. At inference with soft decoding (Lab 17),
it encounters concept tokens — weighted blends it never trained on.
This train-test mismatch limits soft decoding's effectiveness.

Soft training fixes this with a curriculum:

  input = (1 - mix) * embed(ground_truth) + mix * concept_token
  mix: 0 -> 1 over training (start standard, end fully soft)

Each training step has two forward passes:
  1. Standard forward (detached) -> logits -> compute concept tokens
  2. Mixed-input forward -> loss -> backprop

The concept token at position i comes from the model's prediction at
position i-1, replacing the ground-truth embedding that would normally
occupy that slot. Position 0 (BOS) always stays ground truth.

This is scheduled sampling with soft tokens: the model gradually learns
to work with continuous inputs, closing the distribution gap between
training and soft inference.
""")
