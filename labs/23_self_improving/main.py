"""
microGPT — Self-improving model edition.

A tiny character-level GPT that iteratively improves itself by generating
candidate outputs, scoring them with a verifiable quality function, and
retraining on its own best generations alongside the original data.

This implements a simplified form of filtered self-training, closest in spirit
to "STaR: Bootstrapping Reasoning with Reasoning" (Zelikman et al., 2022,
https://arxiv.org/abs/2203.14465), which keeps only verified-correct
self-generated rationales. Karpathy's autoresearch project
(https://github.com/karpathy/autoresearch) applies the same
generate-evaluate-keep loop to hyperparameter optimization. Broader related
work includes "Self-Rewarding Language Models" (Yuan et al., 2024,
https://arxiv.org/abs/2401.10020), which uses a learned self-judge rather
than a fixed scorer, and "SPIN" (Chen et al., 2024,
https://arxiv.org/abs/2401.01335), which uses distribution-matching self-play.

This lab uses a fixed handcrafted scorer (bigram statistics) as the verifier,
which is simpler than the learned judges in Self-Rewarding LMs or the
distribution-matching in SPIN. No external LLM is needed.
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
char_to_id = {ch: i for i, ch in enumerate(uchars)}
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Quality scorer — the "verifier" that decides which self-generations to keep
# ---------------------------------------------------------------------------
# We build a bigram model from the REAL training data. This serves as a
# ground-truth quality signal: generated names that match real-name statistics
# score higher. In STaR, the verifier checks logical correctness; here it
# checks "does this look like a real name?"


def build_bigram_scorer(names):
    """Build a log-probability scorer from bigram statistics of real names."""
    counts = {}
    for name in names:
        seq = [BOS] + [char_to_id[ch] for ch in name] + [BOS]
        for a, b in zip(seq, seq[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    # Add-1 (Laplace) smoothing: P(b|a) = (count(a,b) + 1) / (count(a) + V)
    total_per_context = {}
    for (a, _), c in counts.items():
        total_per_context[a] = total_per_context.get(a, 0) + c
    log_probs = {}
    for (a, b), c in counts.items():
        log_probs[(a, b)] = math.log((c + 1) / (total_per_context[a] + vocab_size))
    # Default for unseen bigrams: use per-context denominator when available
    default_lp_per_context = {a: math.log(1 / (t + vocab_size)) for a, t in total_per_context.items()}
    # Fallback for completely unseen contexts
    default_lp_unknown = math.log(1 / vocab_size)
    return log_probs, default_lp_per_context, default_lp_unknown


def score_name(name, log_probs, default_lp_ctx, default_lp_unk):
    """Score a generated name by average bigram log-probability with length penalty."""
    if len(name) < 3 or len(name) > 15:
        return -10.0  # reject too-short or too-long names
    seq = [BOS] + [char_to_id.get(ch, 0) for ch in name if ch in char_to_id] + [BOS]
    total = 0.0
    for a, b in zip(seq, seq[1:]):
        total += log_probs.get((a, b), default_lp_ctx.get(a, default_lp_unk))
    avg_lp = total / len(seq)
    # Penalize names far from the typical length (mean ~6 chars in the dataset)
    length_penalty = -0.1 * abs(len(name) - 6)
    return avg_lp + length_penalty


bigram_lp, default_lp_ctx, default_lp_unk = build_bigram_scorer(docs)

# ---------------------------------------------------------------------------
# Model (same architecture as lab 03)
# ---------------------------------------------------------------------------
n_embd = 32
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
# Training and generation helpers
# ---------------------------------------------------------------------------
device = "cpu"
temperature = 0.5


def train_on_data(model, train_docs, num_steps, base_lr=1e-2):
    """Train the model on a list of name strings."""
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.85, 0.99), eps=1e-8)
    total_loss = 0
    for step in range(num_steps):
        doc = train_docs[step % len(train_docs)]
        tokens = [BOS] + [char_to_id[ch] for ch in doc] + [BOS]
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
        total_loss += loss.item()
        if (step + 1) % 200 == 0 or step == 0:
            print(f"  step {step + 1:4d} / {num_steps} | loss {loss.item():.4f}")
    return total_loss / num_steps


@torch.no_grad()
def generate_names(model, n_samples=100):
    """Generate n_samples names from the model."""
    model.eval()
    names = []
    for _ in range(n_samples):
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
        if name:
            names.append(name)
    model.train()
    return names


def evaluate_quality(names):
    """Compute average quality score for a list of names."""
    scores = [score_name(n, bigram_lp, default_lp_ctx, default_lp_unk) for n in names]
    return sum(scores) / len(scores) if scores else -10.0


# ===========================================================================
# Phase 1: Initial training on real data
# ===========================================================================
print("=" * 60)
print("PHASE 1: Initial training on real data")
print("=" * 60)

model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

initial_steps = 800
avg_loss = train_on_data(model, docs, initial_steps)
print(f"initial training: {initial_steps} steps, avg loss {avg_loss:.4f}")

# Generate baseline samples
baseline_names = generate_names(model, 200)
baseline_quality = evaluate_quality(baseline_names)
print(f"baseline quality: {baseline_quality:.4f}")
print(f"baseline samples: {', '.join(baseline_names[:10])}")

# ===========================================================================
# Phase 2: Self-improvement loop
# ===========================================================================
print(f"\n{'=' * 60}")
print("PHASE 2: Self-improvement loop")
print("=" * 60)

NUM_ROUNDS = 6
SAMPLES_PER_ROUND = 300
KEEP_TOP_FRACTION = 0.2  # keep top 20% of self-generated names
RETRAIN_STEPS = 300
MIX_RATIO = 0.15  # 15% self-generated, 85% real data in retraining

round_stats = []

for round_idx in range(NUM_ROUNDS):
    # Step 1: Generate candidates
    candidates = generate_names(model, SAMPLES_PER_ROUND)

    # Step 2: Score and filter — keep only the best
    scored = [(name, score_name(name, bigram_lp, default_lp_ctx, default_lp_unk)) for name in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(len(scored) * KEEP_TOP_FRACTION))
    kept = [name for name, _ in scored[:n_keep]]
    kept_quality = evaluate_quality(kept)
    all_quality = evaluate_quality(candidates)

    # Step 3: Mix self-generated data with real data for retraining
    n_self = int(RETRAIN_STEPS * MIX_RATIO)
    n_real = RETRAIN_STEPS - n_self
    mixed_data = []
    for i in range(RETRAIN_STEPS):
        if i < n_self:
            mixed_data.append(kept[i % len(kept)])
        else:
            mixed_data.append(docs[random.randint(0, len(docs) - 1)])
    random.shuffle(mixed_data)

    # Step 4: Retrain on mixed data
    avg_loss = train_on_data(model, mixed_data, RETRAIN_STEPS, base_lr=5e-3)

    # Step 5: Evaluate post-retraining
    post_names = generate_names(model, 200)
    post_quality = evaluate_quality(post_names)

    round_stats.append({
        "round": round_idx + 1,
        "candidates": len(candidates),
        "kept": n_keep,
        "kept_quality": kept_quality,
        "all_quality": all_quality,
        "post_quality": post_quality,
        "avg_loss": avg_loss,
    })

    print(
        f"round {round_idx + 1:2d}: "
        f"generated {len(candidates)}, kept {n_keep} (top {KEEP_TOP_FRACTION:.0%}), "
        f"quality {all_quality:.4f} -> {post_quality:.4f}, "
        f"loss {avg_loss:.4f}"
    )

# ===========================================================================
# Results
# ===========================================================================
print(f"\n{'=' * 60}")
print("SELF-IMPROVEMENT RESULTS")
print("=" * 60)

print(f"\n{'Round':>5s}  {'Generated':>9s}  {'Kept':>4s}  {'Quality':>8s}  {'Post-Quality':>12s}  {'Improvement':>11s}")
print("-" * 60)
for s in round_stats:
    imp = s["post_quality"] - baseline_quality
    print(
        f"{s['round']:>5d}  {s['candidates']:>9d}  {s['kept']:>4d}  "
        f"{s['all_quality']:>8.4f}  {s['post_quality']:>12.4f}  "
        f"{'+' if imp >= 0 else ''}{imp:>10.4f}"
    )

final_quality = round_stats[-1]["post_quality"]
total_improvement = final_quality - baseline_quality
print(f"\nbaseline quality:  {baseline_quality:.4f}")
print(f"final quality:     {final_quality:.4f}")
print(f"total improvement: {'+' if total_improvement >= 0 else ''}{total_improvement:.4f}")

# Show final generated names
print(f"\n--- baseline names (before self-improvement) ---")
for i, name in enumerate(baseline_names[:10]):
    s = score_name(name, bigram_lp, default_lp_ctx, default_lp_unk)
    print(f"  {i + 1:2d}: {name:<15s} (score {s:.3f})")

final_names = generate_names(model, 20)
print(f"\n--- final names (after {NUM_ROUNDS} rounds of self-improvement) ---")
for i, name in enumerate(final_names):
    s = score_name(name, bigram_lp, default_lp_ctx, default_lp_unk)
    print(f"  {i + 1:2d}: {name:<15s} (score {s:.3f})")

# ===========================================================================
# Explanation
# ===========================================================================
print(f"""
{'=' * 60}
HOW SELF-IMPROVEMENT WORKS
{'=' * 60}

The self-improvement loop has four steps per round:

  1. GENERATE: Model produces {SAMPLES_PER_ROUND} candidate names
  2. SCORE:    Bigram quality function + length penalty rates each candidate
  3. FILTER:   Keep only the top {KEEP_TOP_FRACTION:.0%} (verified good outputs)
  4. RETRAIN:  Mix {MIX_RATIO:.0%} self-generated + {1 - MIX_RATIO:.0%} real data, train {RETRAIN_STEPS} steps

The key insight: only keeping VERIFIED good outputs reduces the risk of
collapse. Naive self-training (keeping all outputs) causes the model to
amplify its own errors. The quality filter mitigates this — though it
does not fully prevent mode collapse (the model can still converge toward
a narrow set of high-scoring patterns). Diversity controls, quality
thresholds, and revert-on-regression would strengthen the loop further.

  Without filtering: model drifts toward its own mistakes (collapse)
  With filtering:    model preferentially learns from good outputs

This is a simplified version of the pattern behind:
  - STaR: keep self-generated rationales only if the answer is correct
  - Karpathy's autoresearch: keep code changes only if loss improves
  - Self-Rewarding LMs use a learned judge (not a fixed scorer like here)
  - SPIN uses distribution-matching self-play (a different mechanism)
""")
