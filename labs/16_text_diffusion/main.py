"""
microGPT — Masked Diffusion Language Model (PyTorch).

Instead of generating names left-to-right like a GPT, names emerge from pure
noise -- all [MASK] tokens -- through iterative unmasking. Same transformer
architecture, fundamentally different generative paradigm.

Built on the image-domain foundation of "Denoising Diffusion Probabilistic
Models" (Ho et al., 2020), https://arxiv.org/abs/2006.11239. The discrete-text
diffusion framework follows "Simple and Effective Masked Diffusion Language
Models" (Sahoo et al., 2024), https://arxiv.org/abs/2406.07524, and "Large
Language Diffusion Models" (Nie et al., 2025), https://arxiv.org/abs/2502.09992.
The denoising schedule and confidence-based unmasking in this lab are
illustrative simplifications -- the original MDLM uses a continuous-time ELBO
with an absorbing-state forward process.
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

# MASK replaces BOS, PAD handles fixed-length sequences
uchars = sorted(set("".join(docs)))
MASK = len(uchars)  # [MASK] — the "noise" state that the model learns to denoise
PAD = len(uchars) + 1  # [PAD] — fills unused positions in fixed-length sequences
vocab_size = len(uchars) + 2
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model — bidirectional transformer (no causal mask)
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 2  # number of layers (diffusion needs depth to gather scattered clues)
block_size = 16  # maximum sequence length (names are padded/truncated to this)
head_dim = n_embd // n_head


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class BidirectionalSelfAttention(nn.Module):
    """Every position attends to every other — no causal mask."""

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

        # Bidirectional: no causal mask, every position sees every other
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
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
        self.attn = BidirectionalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying — same matrix for input embeddings and output projection
        self.lm_head.weight = self.wte.weight
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
model = MicroDiffusion().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 3000
batch_size = 32  # diffusion needs batching — single-sample gradients are too noisy
print(f"batch size: {batch_size} (diffusion needs batching — single-sample gradients are too noisy)")

for step in range(num_steps):
    # Build a batch of (clean, noisy) pairs, each with its own random masking
    clean_batch = []
    noisy_batch = []
    mask_batch = []  # True where masked

    for b in range(batch_size):
        doc = docs[(step * batch_size + b) % len(docs)]
        clean = [uchars.index(ch) for ch in doc] + [PAD] * (block_size - len(doc))
        clean = clean[:block_size]

        # Forward process: corrupt by masking each token with probability t
        # Log-uniform t sampling: t ∝ 1/t, cancels the 1/t ELBO weight
        t = math.exp(random.uniform(math.log(0.2), 0))
        noisy = [MASK if random.random() < t else c for c in clean]
        is_masked = [n == MASK for n in noisy]

        if not any(is_masked):
            # Ensure at least one masked token for a training signal
            pos = random.randrange(block_size)
            noisy[pos] = MASK
            is_masked[pos] = True

        clean_batch.append(clean)
        noisy_batch.append(noisy)
        mask_batch.append(is_masked)

    # Forward pass — predict clean tokens from noisy input
    input_ids = torch.tensor(noisy_batch, device=device)  # (B, block_size)
    targets = torch.tensor(clean_batch, device=device)  # (B, block_size)
    mask = torch.tensor(mask_batch, device=device)  # (B, block_size)

    logits = model(input_ids)  # (B, block_size, vocab_size)
    logits[:, :, MASK] = logits[:, :, MASK] - 1e6  # never predict MASK

    # Loss only on masked positions, averaged across the batch
    loss = F.cross_entropy(logits[mask], targets[mask])

    optimizer.zero_grad()
    loss.backward()
    lr_t = 1e-2 * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t
    optimizer.step()

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# ---------------------------------------------------------------------------
# Inference — iterative denoising from all-MASK to clean names
# ---------------------------------------------------------------------------
num_denoise_steps = 64
print("\n--- inference (new, hallucinated names) ---")

model.eval()
with torch.no_grad():
    for sample_idx in range(20):
        seq = [MASK] * block_size  # start from pure noise

        for step_i in range(num_denoise_steps, 0, -1):
            t = math.cos(math.pi / 2 * (1 - step_i / num_denoise_steps))
            s = math.cos(math.pi / 2 * (1 - (step_i - 1) / num_denoise_steps))
            temperature = 0.3 + 0.5 * t  # explore early, commit late

            input_ids = torch.tensor([seq], device=device)
            logits = model(input_ids)[0]  # (block_size, vocab_size)

            predicted = list(seq)
            confidences = []
            for i in range(block_size):
                if seq[i] == MASK:
                    logits_i = logits[i].clone()
                    logits_i[MASK] = logits_i[MASK] - 1e6  # never predict MASK
                    probs = F.softmax(logits_i / temperature, dim=-1)
                    predicted[i] = torch.multinomial(probs, 1).item()
                    confidences.append((probs.max().item(), i))

            # Remask least confident predictions (unless final step)
            if s > 0 and confidences:
                n_to_remask = int(len(confidences) * s / t)
                confidences.sort()
                for _, i in confidences[:n_to_remask]:
                    predicted[i] = MASK
            seq = predicted

        name = "".join(uchars[c] for c in seq if c < len(uchars))
        print(f"sample {sample_idx + 1:2d}: {name}")
