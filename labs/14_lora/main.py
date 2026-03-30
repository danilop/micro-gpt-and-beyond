"""
microGPT with LoRA — parameter-efficient fine-tuning.

Pre-trains a standard microGPT on all names, then fine-tunes with LoRA
(Low-Rank Adaptation) on a filtered subset — names whose consonants are
all drawn from {m, n, r} (e.g., emma, aria, naomi, aurora). This creates a
distinct phonetic style that is easy to inspect in the generated samples. Only the small
LoRA matrices are trained — the base model stays frozen. Demonstrates:
  - Freezing base weights and injecting low-rank adapters
  - Training a small fraction of parameters to shift the output distribution
  - Merging LoRA weights back into the base model (zero runtime overhead)

Reference:
  "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
  https://arxiv.org/abs/2106.09685

This implementation follows the core LoRA algorithm from Hu et al. (2021),
including the alpha/rank scaling factor. For simplicity, adapters are
injected only on the query and value projections (wq, wv) — the original
paper found this subset sufficient for most tasks.
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
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
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
    The scaling factor alpha/rank controls the magnitude of the adaptation
    (see Hu et al., 2021, Section 4.1).
    """

    def __init__(self, base_linear, rank=4, alpha=None):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)  # freeze base
        d_out, d_in = base_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))  # init B to zero
        self.scaling = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        # base output + scaled low-rank adaptation
        # W*x + (alpha/rank) * B*A*x  (B is zero-init, so LoRA starts as no-op)
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling


# ---------------------------------------------------------------------------
# Model (identical to lab 03)
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
# Helpers
# ---------------------------------------------------------------------------
device = "cpu"
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high


def generate(model, n_samples=20, label="sample", check_fn=None, quiet=False):
    """Generate names from the model, optionally marking matches."""
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
            if not quiet:
                mark = f"  {'<-- soft' if check_fn(name) else ''}" if check_fn else ""
                print(f"  {label} {i + 1:2d}: {name}{mark}")
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
# Phase 2: Create fine-tuning target — "soft" names (consonants ⊆ {m, n, r})
# ===========================================================================
vowels = set("aeiou")
soft_consonants = set("mnr")
ft_docs = [d for d in docs if (set(d.lower()) - vowels).issubset(soft_consonants) and len(set(d.lower()) - vowels) >= 1]
print(f"\n{'=' * 60}")
print("PHASE 2: Fine-tuning target — 'soft' names (consonants ⊆ {m, n, r})")
print(f"{'=' * 60}")
print(f"filtered dataset: {len(ft_docs)} names (out of {len(docs)} total)")
print(f"examples: {', '.join(ft_docs[:8])}")

# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------


def is_soft(name):
    """Check if a name uses only 'soft' consonants (m, n, r)."""
    return (set(name.lower()) - vowels).issubset(soft_consonants) and len(name) > 0


def inject_lora(module, rank, alpha=None, targets=("wq", "wv")):
    """Replace targeted nn.Linear layers with LoRALinear wrappers."""
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank=rank, alpha=alpha, targets=targets)


def merge_lora(module):
    """Merge LoRA weights back: W_new = W_base + scaling * B @ A."""
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            merged = nn.Linear(
                child.base.weight.shape[1],
                child.base.weight.shape[0],
                bias=False,
            )
            merged.weight = nn.Parameter(child.base.weight + child.scaling * child.lora_B @ child.lora_A)
            setattr(module, name, merged)
        else:
            merge_lora(child)


# Generate baseline scores (same seed for fair comparison)
base_model = MicroGPT().to(device)
base_model.load_state_dict(base_state)
print("\nBase model (no fine-tuning):")
torch.manual_seed(42)
base_only_names = generate(base_model, n_samples=20, label="base", check_fn=is_soft)
base_soft = sum(1 for n in base_only_names if is_soft(n))
print(f"  -> {base_soft}/20 soft names")

# ===========================================================================
# Phase 3: Rank ablation — LoRA with rank 1, 2, 4
# ===========================================================================

base_total = sum(p.numel() for p in base_model.parameters())
n_samples = 20
ft_steps = 500
results = []  # (rank, adapter_params, pct, soft_count, soft_pct, merge_ok)

for lora_rank in [1, 2, 4]:
    print(f"\n{'=' * 60}")
    print(f"RANK {lora_rank}: Inject → Fine-tune → Evaluate → Merge")
    print("=" * 60)

    # Restore base model and freeze all weights
    model = MicroGPT().to(device)
    model.load_state_dict(base_state)
    for p in model.parameters():
        p.requires_grad_(False)

    # Inject LoRA on query and value projections (standard recipe)
    for layer in model.layers:
        inject_lora(layer, rank=lora_rank)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable
    assert frozen == base_total, f"freeze error: {frozen} != {base_total}"
    pct = 100.0 * trainable / total_params

    # Show adapter dimensions: A(rank×d_in) @ B(d_out×rank) adapts W(d_out×d_in)
    for name, child in model.named_modules():
        if isinstance(child, LoRALinear):
            d_out, d_in = child.base.weight.shape
            r = child.lora_A.shape[0]
            a_size = r * d_in
            b_size = d_out * r
            short = name.split(".")[-1]
            print(
                f"  {short}: W({d_out}x{d_in})={d_out * d_in} params"
                f" → A({r}x{d_in})={a_size} + B({d_out}x{r})={b_size}"
                f" = {a_size + b_size} adapter params"
            )
    print(f"  total: {trainable} adapter params ({pct:.1f}% of {total_params}, base frozen: {frozen})")

    # Fine-tune on soft names
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
    for step in range(ft_steps):
        doc = ft_docs[step % len(ft_docs)]
        loss = train_step(model, optimizer, doc, step, ft_steps)
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:4d} / {ft_steps} | loss {loss:.4f}")

    # Generate and evaluate
    torch.manual_seed(42)
    lora_names = generate(model, n_samples=n_samples, label=f"r{lora_rank}", check_fn=is_soft)
    soft_count = sum(1 for n in lora_names if is_soft(n))
    soft_pct = 100.0 * soft_count / n_samples
    print(f"  -> {soft_count}/{n_samples} soft names ({soft_pct:.0f}%)")

    # Merge LoRA weights into base and verify identical output
    merge_lora(model)
    torch.manual_seed(42)
    merged_names = generate(model, n_samples=n_samples, quiet=True)
    merge_ok = all(a == b for a, b in zip(lora_names, merged_names))
    merged_total = sum(p.numel() for p in model.parameters())
    print(f"  merge: {'ok' if merge_ok else 'MISMATCH'}, params back to {merged_total}")

    results.append((lora_rank, trainable, pct, soft_count, soft_pct, merge_ok))

# ===========================================================================
# Summary table
# ===========================================================================
print(f"\n{'=' * 60}")
print("RANK ABLATION SUMMARY")
print("=" * 60)
print(f"  Base model: {base_soft}/{n_samples} soft ({100.0 * base_soft / n_samples:.0f}%)")
print(f"  Base params: {base_total}")
print()
print(f"  {'rank':>4}  {'adapter':>7}  {'% total':>7}  {'soft':>4}  {'soft%':>5}  {'merge':>5}")
print(f"  {'----':>4}  {'-------':>7}  {'-------':>7}  {'----':>4}  {'-----':>5}  {'-----':>5}")
for rank, ap, pct, sc, sp, mo in results:
    print(f"  {rank:>4}  {ap:>7}  {pct:>6.1f}%  {sc:>2}/{n_samples}  {sp:>4.0f}%  {'ok' if mo else 'FAIL':>5}")
