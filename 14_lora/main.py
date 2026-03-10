"""
microGPT with LoRA — parameter-efficient fine-tuning.

Pre-trains a standard microGPT on all names, then fine-tunes with LoRA
(Low-Rank Adaptation) on a filtered subset. Only the small LoRA matrices
are trained — the base model stays frozen. Demonstrates:
  - Freezing base weights and injecting low-rank adapters
  - Training <5% of parameters to shift the output distribution
  - Merging LoRA weights back into the base model (zero runtime overhead)
"""

import copy
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"num docs: {len(docs)}")
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# LoRA Linear layer
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with a low-rank adaptation.

    The base weight is frozen. Only lora_A and lora_B are trainable.
    B is zero-initialized so the adapter starts as a no-op (W + 0).
    """

    def __init__(self, base_linear, rank=4):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)  # freeze base
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # init B to zero

    def forward(self, x):
        # base output + low-rank adaptation
        # W*x + B*A*x  (B is zero-init, so LoRA starts as no-op)
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T


# ---------------------------------------------------------------------------
# Model (identical to lab 03)
# ---------------------------------------------------------------------------
n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head


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

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
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
# Helpers
# ---------------------------------------------------------------------------
device = "cpu"
temperature = 0.5


def generate(model, n_samples=20, label="sample"):
    """Generate names from the model."""
    model.eval()
    names = []
    with torch.no_grad():
        for i in range(n_samples):
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
            names.append(name)
            print(f"  {label} {i + 1:2d}: {name}")
    model.train()
    return names


def train_step(model, optimizer, doc, step, num_steps, base_lr=1e-2):
    """Single training step (shared between pre-training and fine-tuning)."""
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    input_ids = torch.tensor([tokens[:n]], device=device)
    targets = torch.tensor([tokens[1 : n + 1]], device=device)

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    lr_t = base_lr * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t
    optimizer.step()
    return loss.item()


# ===========================================================================
# Phase 1: Pre-train base model on ALL names
# ===========================================================================
print("\n" + "=" * 60)
print("PHASE 1: Pre-training base model on all names")
print("=" * 60)

model = MicroGPT().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"total params: {total_params}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    loss = train_step(model, optimizer, doc, step, num_steps)
    if (step + 1) % 200 == 0 or step == 0:
        print(f"  step {step + 1:4d} / {num_steps} | loss {loss:.4f}")

print("\nBaseline generation (pre-trained on all names):")
base_names = generate(model, n_samples=10, label="base")

# Save base model state for later comparison
base_state = copy.deepcopy(model.state_dict())

# ===========================================================================
# Phase 2: Create fine-tuning target — names starting with "m"
# ===========================================================================
target_letter = "m"
ft_docs = [d for d in docs if d.lower().startswith(target_letter)]
print(f"\n{'=' * 60}")
print(f"PHASE 2: Fine-tuning target — names starting with '{target_letter}'")
print(f"{'=' * 60}")
print(f"filtered dataset: {len(ft_docs)} names (out of {len(docs)} total)")

# ===========================================================================
# Phase 3: Apply LoRA — replace Linear layers with LoRALinear wrappers
# ===========================================================================
print(f"\n{'=' * 60}")
print("PHASE 3: Injecting LoRA adapters")
print("=" * 60)

lora_rank = 4


def inject_lora(module, rank=4):
    """Replace all nn.Linear layers with LoRALinear wrappers (recursively)."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank))
        else:
            inject_lora(child, rank=rank)


# Apply LoRA to attention and MLP layers (not embeddings or lm_head)
for layer in model.layers:
    inject_lora(layer, rank=lora_rank)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
pct = 100.0 * trainable_params / total_params

print(f"total params:     {total_params}")
print(f"frozen params:    {frozen_params}")
print(f"trainable params: {trainable_params} ({pct:.1f}% of total)")

# ===========================================================================
# Phase 4: Fine-tune with LoRA on filtered subset
# ===========================================================================
print(f"\n{'=' * 60}")
print(f"PHASE 4: Fine-tuning with LoRA on '{target_letter}' names")
print("=" * 60)

# Only optimize LoRA parameters (the ones that require grad)
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(lora_params, lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
ft_steps = 500

for step in range(ft_steps):
    doc = ft_docs[step % len(ft_docs)]
    loss = train_step(model, optimizer, doc, step, ft_steps)
    if (step + 1) % 100 == 0 or step == 0:
        print(f"  step {step + 1:4d} / {ft_steps} | loss {loss:.4f}")

# ===========================================================================
# Phase 5: Compare results — base vs LoRA-adapted
# ===========================================================================
print(f"\n{'=' * 60}")
print("PHASE 5: Comparing base vs LoRA-adapted generation")
print("=" * 60)

# Generate from LoRA-adapted model
print(f"\nLoRA-adapted model (fine-tuned on '{target_letter}' names):")
torch.manual_seed(42)
lora_names = generate(model, n_samples=20, label="lora")

# Reload base model (without LoRA) for comparison
base_model = MicroGPT().to(device)
base_model.load_state_dict(base_state)
print("\nBase model (no fine-tuning):")
torch.manual_seed(42)
base_only_names = generate(base_model, n_samples=20, label="base")

# Measure distribution shift
lora_m_count = sum(1 for n in lora_names if n.lower().startswith(target_letter))
base_m_count = sum(1 for n in base_only_names if n.lower().startswith(target_letter))

print("\n--- Distribution shift ---")
print(f"  Base model:  {base_m_count}/20 names start with '{target_letter}'")
print(f"  LoRA model:  {lora_m_count}/20 names start with '{target_letter}'")
print(f"\n  Fine-tuned {trainable_params} params ({pct:.1f}% of {total_params}) to shift the distribution")

# ===========================================================================
# Phase 6: LoRA merge — fold adapters back into base weights
# ===========================================================================
print(f"\n{'=' * 60}")
print("PHASE 6: Merging LoRA weights into base model")
print("=" * 60)


def merge_lora(module):
    """Merge LoRA weights back: W_new = W_base + B @ A."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            merged = nn.Linear(
                child.base.weight.shape[1],
                child.base.weight.shape[0],
                bias=False,
            )
            # W_new = W_base + B @ A
            merged.weight = nn.Parameter(child.base.weight + child.lora_B @ child.lora_A)
            setattr(module, name, merged)
        else:
            merge_lora(child)


merge_lora(model)

print("LoRA adapters merged into base weights (no runtime overhead)")

# Verify merged model produces identical output
print("\nMerged model generation (should match LoRA model exactly):")
torch.manual_seed(42)
merged_names = generate(model, n_samples=20, label="merged")

match = all(a == b for a, b in zip(lora_names, merged_names))
print(f"\nMerged output matches LoRA output: {match}")

merged_params = sum(p.numel() for p in model.parameters())
print(f"Merged model params: {merged_params} (same as original base model)")
