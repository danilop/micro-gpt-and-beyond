"""
microGPT — PyTorch batched edition.

Same architecture as 03_pytorch, but with proper mini-batch training:
  - Batches of 32 sequences per step
  - Padding + attention masking
  - Scaled-up model (2 layers, 64-dim embeddings, context 16)
  - 1000 training steps

This is what "engineering for efficiency" looks like on top of the same algorithm.
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'input.txt')
if not os.path.exists(input_path):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
PAD = vocab_size  # padding token (not in vocab, ignored in loss)
vocab_size_with_pad = vocab_size + 1
print(f"vocab size: {vocab_size} (+1 pad = {vocab_size_with_pad})")

# ---------------------------------------------------------------------------
# Hyperparameters (scaled up from the single-sample version)
# ---------------------------------------------------------------------------
n_embd = 64
n_head = 4
n_layer = 2
block_size = 16
head_dim = n_embd // n_head
batch_size = 32
num_steps = 1000

# ---------------------------------------------------------------------------
# Model (same architecture, just bigger)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
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

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal, float('-inf'))
        # Padding mask: don't attend to PAD positions
        if pad_mask is not None:
            # pad_mask: (B, T), True where padded
            att = att.masked_fill(pad_mask[:, None, None, :], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att)  # handle all-masked rows

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)) ** 2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.norm1(x), pad_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size_with_pad, n_embd, padding_idx=PAD)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)  # output over real vocab only
        self.apply(self._init_weights)
        # Zero-init output projections (wo, fc2) to match original std=0
        for layer in self.layers:
            nn.init.zeros_(layer.attn.wo.weight)
            nn.init.zeros_(layer.mlp.fc2.weight)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, pad_mask=None):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Batching utilities
# ---------------------------------------------------------------------------
def make_batch(docs, step, batch_size):
    """Create a padded batch of token sequences."""
    batch_docs = [docs[(step * batch_size + i) % len(docs)] for i in range(batch_size)]
    sequences = []
    for doc in batch_docs:
        toks = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        toks = toks[:block_size + 1]  # truncate to max length
        sequences.append(toks)

    max_len = max(len(s) for s in sequences)
    input_ids = []
    target_ids = []
    pad_masks = []
    for s in sequences:
        n = len(s) - 1
        inp = s[:n] + [PAD] * (max_len - 1 - n)
        tgt = s[1:n+1] + [-100] * (max_len - 1 - n)  # -100 = ignore in cross_entropy
        mask = [False] * n + [True] * (max_len - 1 - n)
        input_ids.append(inp)
        target_ids.append(tgt)
        pad_masks.append(mask)

    return (
        torch.tensor(input_ids),
        torch.tensor(target_ids),
        torch.tensor(pad_masks),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
device = 'cpu'
model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8)

for step in range(num_steps):
    input_ids, targets, pad_mask = make_batch(docs, step, batch_size)
    input_ids = input_ids.to(device)
    targets = targets.to(device)
    pad_mask = pad_mask.to(device)

    logits = model(input_ids, pad_mask)  # (B, T, V)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)

    optimizer.zero_grad()
    loss.backward()

    lr_t = 1e-2 * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    for pg in optimizer.param_groups:
        pg['lr'] = lr_t

    optimizer.step()

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
temperature = 0.5
print("\n--- inference ---")
model.eval()
with torch.no_grad():
    for sample_idx in range(20):
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
        name = ''.join(uchars[t] for t in tokens[1:])
        print(f"sample {sample_idx+1:2d}: {name}")
