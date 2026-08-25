"""
microGPT: Masked Diffusion Language Model (PyTorch).

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
MASK = len(uchars)  # [MASK], the "noise" state that the model learns to denoise
PAD = len(uchars) + 1  # [PAD], which fills unused positions in fixed-length sequences
vocab_size = len(uchars) + 2
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model: bidirectional transformer (no causal mask)
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
    """Every position attends to every other, with no causal mask."""

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
        # Weight tying: same matrix for input embeddings and output projection
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
batch_size = 32  # diffusion needs batching, since single-sample gradients are too noisy
print(f"batch size: {batch_size} (diffusion needs batching, since single-sample gradients are too noisy)")

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

    # Forward pass: predict clean tokens from noisy input
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
# Inference: iterative denoising from all-MASK to clean names
# ---------------------------------------------------------------------------
# The step count is a dial, not a constant, and it is the whole reason to care
# about diffusion: with fewer steps than tokens, several positions commit in the
# same forward pass. Left-to-right decoding has no such dial, and it always costs
# one forward pass per token. Note that the first arm below (16 steps for 16
# positions) is not that saving: it cannot commit more than one position per
# step on average, and the run reports that it does not.


def denoise(num_denoise_steps):
    """Generate one name by iterative unmasking.

    Returns (name, passes, commits, trace), where trace holds the temperature
    and the full sequence after every step, so a caller can print the run
    rather than only its result.
    """
    seq = [MASK] * block_size  # start from pure noise
    passes = 0
    commits = []  # positions committed per step, the schedule as measured
    trace = []  # (temperature, sequence) after each step, for the step-by-step view

    for step_i in range(num_denoise_steps, 0, -1):
        # Cosine schedule: `t` is the fraction of positions masked going into
        # this step, `s` the fraction the schedule wants masked after it.
        t = math.cos(math.pi / 2 * (1 - step_i / num_denoise_steps))
        s = math.cos(math.pi / 2 * (1 - (step_i - 1) / num_denoise_steps))
        temperature = 0.3 + 0.5 * t  # explore early, commit late

        logits = model(torch.tensor([seq], device=device))[0]  # (block_size, vocab_size)
        passes += 1

        predicted = list(seq)
        confidences = []
        for i in range(block_size):
            if seq[i] == MASK:
                logits_i = logits[i].clone()
                logits_i[MASK] = logits_i[MASK] - 1e6  # never predict MASK
                probs = F.softmax(logits_i / temperature, dim=-1)
                predicted[i] = torch.multinomial(probs, 1).item()
                confidences.append((probs.max().item(), i))

        # Commit the most confident predictions and re-mask the rest, so the
        # number left masked is what the schedule asked for. Anchoring on
        # block_size, rather than on how many happen to be masked right now, is what
        # lets the schedule set the pace; the `- 1` floor guarantees every step
        # commits at least one position. Which of the two binds depends on the
        # step count: with as many steps as positions the floor wins every time,
        # and the per-step commit counts printed below say which happened.
        n_to_remask = 0
        if confidences:
            n_to_remask = min(int(block_size * s), len(confidences) - 1)
            confidences.sort()
            for _, i in confidences[:n_to_remask]:
                predicted[i] = MASK
        commits.append(len(confidences) - n_to_remask)
        seq = predicted
        trace.append((temperature, list(seq)))

    return "".join(uchars[c] for c in seq if c < len(uchars)), passes, commits, trace


# A sequence mid-denoise is not text, so it cannot be printed as text. Two of
# the sixteen positions hold no character at all: MASK is a position the model
# has not committed yet, PAD a position it has decided the name does not reach.
# Both need a glyph exactly one column wide, or the rows stop lining up and the
# whole point of the display -- reading down a column to see when a position
# settled -- is lost. `_` reads as a blank waiting to be filled, and it sits low
# enough that the letters appearing around it are what your eye lands on.
GLYPH_MASK = "_"  # still noise
GLYPH_PAD = " "  # committed [PAD]: the name ends before this position


def render(seq):
    """One character per position, so every row is exactly block_size columns."""
    return "".join(GLYPH_MASK if c == MASK else GLYPH_PAD if c == PAD else uchars[c] for c in seq)


model.eval()
with torch.no_grad():
    # Watch a single name emerge. The arms below report where denoising ends up
    # after N steps; this reports how it gets there. Every row is the same 16
    # positions, so a column that changes is a position committing, and a column
    # never changes twice: a commit is final.
    trace_steps = 8
    torch.manual_seed(17)
    name, _, trace_commits, trace = denoise(trace_steps)
    print(f"\n--- one name, denoised in {trace_steps} steps ---")
    print(f"  `{GLYPH_MASK}` = still masked (noise), blank = [PAD], letters = committed\n")
    print(f"  {'step':>4}  {'masked':>6}  {'commit':>6}  {'temp':>4}   sequence")
    print(f"  {0:>4}  {block_size:>6}  {'-':>6}  {'-':>4}   |{render([MASK] * block_size)}|  pure noise")
    for i, (temp, seq) in enumerate(trace, start=1):
        print(f"  {i:>4}  {seq.count(MASK):>6}  {trace_commits[i - 1]:>6}  {temp:>4.2f}   |{render(seq)}|")
    print(f"\n  result: {name!r}")
    # The commit counts are 1,1,1,2,3,2,3,3 here and that shape is the cosine
    # schedule, not the sample: it barely commits at first, when every position
    # is still noise and there is nothing to condition on, then accelerates once
    # enough letters are fixed that the rest are nearly determined. Temperature
    # falls alongside it, so the early guesses explore and the late ones do not.
    assert sum(trace_commits) == block_size, "every position must be committed exactly once"

    for num_steps_denoise in (16, 8, 4):
        torch.manual_seed(1234)  # same noise every time, so only the schedule differs
        print(f"\n--- inference: {num_steps_denoise} denoising steps ---")
        total_passes = 0
        commits = []
        for sample_idx in range(10):
            name, passes, commits, _ = denoise(num_steps_denoise)
            total_passes += passes
            print(f"sample {sample_idx + 1:2d}: {name}")
        print(f"  {total_passes / 10:.1f} forward passes per name (left-to-right needs up to {block_size})")
        # The commit counts depend only on block_size and the schedule, not on
        # sampling, so one sample's pattern is every sample's pattern.
        print(f"  positions committed per step: {','.join(str(c) for c in commits)}")
        if max(commits) == 1:
            print("  Every step committed exactly one position: with as many steps as")
            print("  positions the `- 1` floor is what sets the pace, not the cosine")
            print("  schedule, and this arm costs the same as left-to-right decoding.")

print(
    "\nFewer steps means more positions committed per pass, so generation gets cheaper"
    "\nand, at this scale, visibly worse. How few steps you can get away with is the"
    "\ncentral question in diffusion language models."
)
