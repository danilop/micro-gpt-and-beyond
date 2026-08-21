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
distribution-matching in SPIN. No external LLM is needed. The verifier is
built from a held-out split of the corpus, so it scores names the model has
never trained on.

The interesting result here is a failure, and the lab is built to show it.
The loop runs three arms from identical starting weights and identical seeds:
filtered (keep the top 20%), unfiltered (keep everything), and a control that
retrains on real data only with the same step budget. Every round prints the
quality score AND two things the quality score cannot see: the fraction of
unique names and the most-repeated name with its count. Watch the quality
metric rise while diversity falls. That is Goodhart's law, on screen, in a
loop small enough to read in one sitting.
"""

import copy
import itertools
import math
import os
import random
from collections import Counter

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
# Held-out split — the verifier must not read the model's training data
# ---------------------------------------------------------------------------
# The original version of this lab built the bigram verifier from the same
# corpus the model trained on, so a high "quality" score could simply mean the
# model had memorized the training set. Splitting the corpus fixes that: the
# model only ever sees train_docs, the verifier only ever sees heldout_docs.
# A generated name now has to look like names the verifier has seen and the
# model has not.
HELDOUT_SIZE = 3000
heldout_docs = docs[:HELDOUT_SIZE]
train_docs = docs[HELDOUT_SIZE:]
train_set = set(train_docs)  # used to measure novelty (is the name actually new?)
print(f"train docs: {len(train_docs)}, held-out docs (verifier only): {len(heldout_docs)}")

# ---------------------------------------------------------------------------
# Quality scorer — the "verifier" that decides which self-generations to keep
# ---------------------------------------------------------------------------
# We build a bigram model from REAL names the model never trains on (the
# held-out split). This serves as a ground-truth quality signal: generated
# names that match real-name statistics score higher. In STaR, the verifier
# checks logical correctness; here it checks "does this look like a real name?"
#
# Important limitation to keep in mind while reading the results: this verifier
# scores each name in isolation. It has no opinion about the *distribution* of
# names, so a model that emits one very high-scoring name over and over gets a
# great score. Phase 2 measures exactly that failure.


def build_bigram_scorer(names):
    """Build a log-probability scorer from bigram statistics of real names."""
    counts = {}
    for name in names:
        seq = [BOS] + [char_to_id[ch] for ch in name] + [BOS]
        for a, b in itertools.pairwise(seq):
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
    for a, b in itertools.pairwise(seq):
        total += log_probs.get((a, b), default_lp_ctx.get(a, default_lp_unk))
    # A sequence of N tokens contains N-1 bigrams, so that is what we average
    # over. Dividing by len(seq) (the token count) would quietly shrink every
    # score toward zero, more so for short names.
    avg_lp = total / (len(seq) - 1)
    # Penalize names far from the typical length (mean ~6 chars in the dataset)
    length_penalty = -0.1 * abs(len(name) - 6)
    return avg_lp + length_penalty


bigram_lp, default_lp_ctx, default_lp_unk = build_bigram_scorer(heldout_docs)

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


def train_on_data(model, data, num_steps, base_lr=1e-2, verbose=True):
    """Train the model on a list of name strings."""
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.85, 0.99), eps=1e-8)
    total_loss = 0
    for step in range(num_steps):
        doc = data[step % len(data)]
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
        if verbose and ((step + 1) % 200 == 0 or step == 0):
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


# ---------------------------------------------------------------------------
# The metrics the quality score cannot see
# ---------------------------------------------------------------------------
# Average quality is a per-name statistic, so it is blind to two failure modes
# that matter more than the score itself:
#   diversity — a collapsed model emits the same high-scoring string forever
#   novelty   — a model can score well by parroting its training corpus
# Both are one line of Python each. Printing them next to the quality score is
# the difference between "the loop worked" and "the loop found a shortcut".


def diversity_stats(names):
    """Return (unique fraction, most-repeated name, its count)."""
    if not names:
        return 0.0, "", 0
    counts = Counter(names)
    top_name, top_count = counts.most_common(1)[0]
    return len(counts) / len(names), top_name, top_count


def novelty(names):
    """Fraction of generated names that do not appear verbatim in the training corpus."""
    if not names:
        return 0.0
    return sum(1 for n in names if n not in train_set) / len(names)


# ===========================================================================
# Phase 1: Initial training on real data
# ===========================================================================
print("=" * 60)
print("PHASE 1: Initial training on real data")
print("=" * 60)

model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

initial_steps = 800
avg_loss = train_on_data(model, train_docs, initial_steps)
print(f"initial training: {initial_steps} steps, avg loss {avg_loss:.4f}")

# Snapshot the freshly trained weights. Every arm of the experiment below
# starts from exactly these weights, so the arms differ only in what data they
# retrain on — not in where they started.
initial_state = copy.deepcopy(model.state_dict())

# Generate baseline samples
EVAL_SAMPLES = 200
baseline_names = generate_names(model, EVAL_SAMPLES)
baseline_quality = evaluate_quality(baseline_names)
baseline_unique, baseline_top, baseline_top_n = diversity_stats(baseline_names)
baseline_novelty = novelty(baseline_names)
print(f"baseline quality:   {baseline_quality:.4f}")
print(f"baseline diversity: {baseline_unique:.1%} unique, most repeated '{baseline_top}' x{baseline_top_n}")
print(f"baseline novelty:   {baseline_novelty:.1%} of names are not in the training corpus")
print(f"baseline samples:   {', '.join(baseline_names[:10])}")

# ===========================================================================
# Phase 2: Self-improvement loop
# ===========================================================================
print(f"\n{'=' * 60}")
print("PHASE 2: Self-improvement loop")
print("=" * 60)

NUM_ROUNDS = 6
SAMPLES_PER_ROUND = 300
KEEP_TOP_FRACTION = 0.2  # filtered arm: keep the top 20% of self-generated names
RETRAIN_STEPS = 300
MIX_RATIO = 0.15  # 15% self-generated, 85% real data in retraining


def run_arm(keep_fraction, mix_ratio, label):
    """
    Run the generate -> score -> filter -> retrain loop for NUM_ROUNDS and
    measure quality, diversity and novelty after every round.

    Three settings of the two knobs give us three arms of one experiment:
      keep_fraction=0.2, mix_ratio=0.15 -> the filtered loop this lab teaches
      keep_fraction=1.0, mix_ratio=0.15 -> the naive loop (keep everything)
      keep_fraction=0.2, mix_ratio=0.0  -> control: real data only, same budget
    The control matters. Without it, we cannot tell whether the self-generated
    data helped or whether 1800 extra training steps would have done the same.
    """
    # Same seeds and the same starting weights for every arm, so the arms
    # differ only in what data they retrain on.
    random.seed(4242)
    torch.manual_seed(4242)
    arm_model = MicroGPT().to(device)
    arm_model.load_state_dict(copy.deepcopy(initial_state))

    stats = []
    for round_idx in range(NUM_ROUNDS):
        # Step 1: Generate candidates
        candidates = generate_names(arm_model, SAMPLES_PER_ROUND)

        # Step 2: Score and filter — keep the best `keep_fraction` of them
        scored = [(name, score_name(name, bigram_lp, default_lp_ctx, default_lp_unk)) for name in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(scored) * keep_fraction))
        kept = [name for name, _ in scored[:n_keep]]
        all_quality = evaluate_quality(candidates)

        # Step 3: Mix self-generated data with real data for retraining.
        # mix_ratio sets how many retraining steps see a self-generated name
        # (45 of 300 with the defaults), and that is usually FEWER than the
        # number of names we kept (60). So we sample the self-generated slots
        # uniformly from the whole kept set. Walking `kept` in score order
        # instead would silently use only the top 15% and discard the rest.
        n_self = int(RETRAIN_STEPS * mix_ratio)
        if n_self <= len(kept):
            self_slots = random.sample(kept, n_self)
        else:
            # More slots than kept names: reuse them, so a few names influence
            # many steps. This is what happens if you raise MIX_RATIO.
            self_slots = [random.choice(kept) for _ in range(n_self)]
        mixed_data = self_slots + [random.choice(train_docs) for _ in range(RETRAIN_STEPS - n_self)]
        random.shuffle(mixed_data)

        # Step 4: Retrain on mixed data
        avg_loss = train_on_data(arm_model, mixed_data, RETRAIN_STEPS, base_lr=5e-3, verbose=False)

        # Step 5: Evaluate — quality, plus the two things quality cannot see
        post_names = generate_names(arm_model, EVAL_SAMPLES)
        post_quality = evaluate_quality(post_names)
        unique_frac, top_name, top_count = diversity_stats(post_names)
        novel_frac = novelty(post_names)

        stats.append({
            "round": round_idx + 1,
            "candidates": len(candidates),
            "kept": n_keep,
            "slots": n_self,
            "distinct_fed": len(set(self_slots)),
            "all_quality": all_quality,
            "post_quality": post_quality,
            "unique": unique_frac,
            "top_name": top_name,
            "top_count": top_count,
            "novel": novel_frac,
            "avg_loss": avg_loss,
        })

        # The "slots" figure is how many of the RETRAIN_STEPS steps see a
        # self-generated name, and "distinct" is how many different names those
        # slots contain. When the kept set itself fills up with copies of one
        # high-scoring name, "distinct" drops even though "slots" does not.
        print(
            f"  {label} round {round_idx + 1}: "
            f"kept {n_keep:3d}/{len(candidates)}, self slots {n_self:3d} ({len(set(self_slots)):2d} distinct), "
            f"quality {all_quality:.4f} -> {post_quality:.4f}, "
            f"unique {unique_frac:5.1%}, top '{top_name}' x{top_count:<3d} "
            f"novel {novel_frac:5.1%}, loss {avg_loss:.4f}"
        )

    return arm_model, stats


print(f"\nthree arms, {NUM_ROUNDS} rounds each, identical seeds and identical step budgets")
print(f"every round: generate {SAMPLES_PER_ROUND}, filter, retrain {RETRAIN_STEPS} steps, evaluate {EVAL_SAMPLES}\n")

print(f"arm 1 — FILTERED: keep top {KEEP_TOP_FRACTION:.0%} of self-generations, mix {MIX_RATIO:.0%} into retraining")
filtered_model, filtered_stats = run_arm(KEEP_TOP_FRACTION, MIX_RATIO, "filtered  ")

print("\narm 2 — UNFILTERED: keep ALL self-generations (the naive loop), same mix ratio")
unfiltered_model, unfiltered_stats = run_arm(1.0, MIX_RATIO, "unfiltered")

print("\narm 3 — CONTROL: real data only (MIX_RATIO = 0), same seeds and same step budget")
control_model, control_stats = run_arm(KEEP_TOP_FRACTION, 0.0, "control   ")

# ===========================================================================
# Results
# ===========================================================================
print(f"\n{'=' * 60}")
print("SELF-IMPROVEMENT RESULTS (filtered arm)")
print("=" * 60)

print(
    f"\n{'Round':>5s}  {'Kept':>4s}  {'Quality':>8s}  {'Post-Q':>8s}  {'vs base':>8s}"
    f"  {'Unique':>7s}  {'Most repeated':>18s}  {'Novel':>6s}"
)
print("-" * 82)
for s in filtered_stats:
    imp = s["post_quality"] - baseline_quality
    top = f"'{s['top_name']}' x{s['top_count']}"
    print(
        f"{s['round']:>5d}  {s['kept']:>4d}  {s['all_quality']:>8.4f}  {s['post_quality']:>8.4f}  "
        f"{imp:>+8.4f}  {s['unique']:>7.1%}  {top:>18s}  {s['novel']:>6.1%}"
    )
baseline_top_label = f"'{baseline_top}' x{baseline_top_n}"
print(
    f"{'base':>5s}  {'':>4s}  {'':>8s}  {baseline_quality:>8.4f}  {0.0:>+8.4f}  "
    f"{baseline_unique:>7.1%}  {baseline_top_label:>18s}  {baseline_novelty:>6.1%}"
)

final_quality = filtered_stats[-1]["post_quality"]
total_improvement = final_quality - baseline_quality
final_unique = filtered_stats[-1]["unique"]
print(f"\nbaseline quality:  {baseline_quality:.4f}   (unique {baseline_unique:.1%})")
print(f"final quality:     {final_quality:.4f}   (unique {final_unique:.1%})")
print(f"total improvement: {total_improvement:+.4f}   (unique fraction {final_unique - baseline_unique:+.1%})")

# ---------------------------------------------------------------------------
# The honest comparison: three arms, quality AND diversity
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("THREE ARMS, SIDE BY SIDE — quality and diversity")
print("=" * 60)
print("\nquality after each round (higher is better):")
print(f"{'Round':>5s}  {'filtered':>10s}  {'unfiltered':>10s}  {'control':>10s}")
print("-" * 42)
print(f"{'base':>5s}  {baseline_quality:>10.4f}  {baseline_quality:>10.4f}  {baseline_quality:>10.4f}")
for f, u, c in zip(filtered_stats, unfiltered_stats, control_stats):
    print(f"{f['round']:>5d}  {f['post_quality']:>10.4f}  {u['post_quality']:>10.4f}  {c['post_quality']:>10.4f}")

print("\nunique fraction of 200 samples after each round (higher is better):")
print(f"{'Round':>5s}  {'filtered':>10s}  {'unfiltered':>10s}  {'control':>10s}")
print("-" * 42)
print(f"{'base':>5s}  {baseline_unique:>10.1%}  {baseline_unique:>10.1%}  {baseline_unique:>10.1%}")
for f, u, c in zip(filtered_stats, unfiltered_stats, control_stats):
    print(f"{f['round']:>5d}  {f['unique']:>10.1%}  {u['unique']:>10.1%}  {c['unique']:>10.1%}")

print(f"\n{'arm':>12s}  {'quality':>9s}  {'vs base':>8s}  {'unique':>7s}  {'novel':>6s}  most repeated")
print("-" * 72)
for name, st in (("filtered", filtered_stats), ("unfiltered", unfiltered_stats), ("control", control_stats)):
    last = st[-1]
    print(
        f"{name:>12s}  {last['post_quality']:>9.4f}  {last['post_quality'] - baseline_quality:>+8.4f}  "
        f"{last['unique']:>7.1%}  {last['novel']:>6.1%}  '{last['top_name']}' x{last['top_count']}"
    )

# Show final generated names
print("\n--- baseline names (before self-improvement) ---")
for i, name in enumerate(baseline_names[:10]):
    s = score_name(name, bigram_lp, default_lp_ctx, default_lp_unk)
    print(f"  {i + 1:2d}: {name:<15s} (score {s:.3f})")

final_names = generate_names(filtered_model, 20)
print(f"\n--- final names, filtered arm (after {NUM_ROUNDS} rounds of self-improvement) ---")
for i, name in enumerate(final_names):
    s = score_name(name, bigram_lp, default_lp_ctx, default_lp_unk)
    print(f"  {i + 1:2d}: {name:<15s} (score {s:.3f})")
fn_unique, fn_top, fn_top_n = diversity_stats(final_names)
print(f"  -> {len(set(final_names))}/{len(final_names)} distinct; most repeated '{fn_top}' x{fn_top_n}")

unfiltered_names = generate_names(unfiltered_model, 20)
uf_unique, uf_top, uf_top_n = diversity_stats(unfiltered_names)
print(f"\n--- final names, unfiltered arm (after {NUM_ROUNDS} rounds) ---")
print("  " + ", ".join(unfiltered_names))
print(f"  -> {len(set(unfiltered_names))}/{len(unfiltered_names)} distinct; most repeated '{uf_top}' x{uf_top_n}")

# ---------------------------------------------------------------------------
# Read the two numbers together, not one at a time
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("WHAT THE NUMBERS ACTUALLY SAY")
print("=" * 60)
control_quality = control_stats[-1]["post_quality"]
control_unique = control_stats[-1]["unique"]
unfiltered_quality = unfiltered_stats[-1]["post_quality"]
unfiltered_unique = unfiltered_stats[-1]["unique"]
# Which arm ended up least diverse? Compute it instead of asserting it: the
# answer depends on the seed, the scorer and the number of rounds.
arm_unique = {"filtered": final_unique, "unfiltered": unfiltered_unique, "control": control_unique}
least_diverse = min(arm_unique, key=arm_unique.get)
final_top_name = filtered_stats[-1]["top_name"]
final_top_count = filtered_stats[-1]["top_count"]
top_share = final_top_count / EVAL_SAMPLES
print(f"""
The filtered loop moved the quality metric by {total_improvement:+.4f} and the
unique fraction by {final_unique - baseline_unique:+.1%} ({baseline_unique:.1%} -> {final_unique:.1%}).
One name, '{final_top_name}', accounts for {final_top_count} of {EVAL_SAMPLES} samples ({top_share:.1%}).

That is Goodhart's law in 400 lines: the metric can improve while the model
dies. The verifier scores names one at a time, so a model that discovers a
single high-scoring string and emits it over and over gets a better average
score than a model that produces many plausible different names. The loop
optimized what we measured, and we measured the wrong thing.

Note which arm ended up least diverse in this run: {least_diverse}. The
folklore says the UNFILTERED loop is the dangerous one, and at large mix ratios
it is. But filtering is itself a distribution-narrowing operation: taking the
top {KEEP_TOP_FRACTION:.0%} by a per-name score deliberately throws away the tails, so with
only {MIX_RATIO:.0%} of the retraining stream coming from the model, the filtered arm is the
one that concentrates. Selection pressure toward a narrow optimum is not a bug
in the filter, it is what the filter does.

The control arm (real data only, same seeds, same step budget) reached
{control_quality:.4f} vs the filtered arm's {final_quality:.4f}, so the
self-generated data is doing real work on the metric — the problem is not that
the loop does nothing, it is that the metric it improves is incomplete.

The per-round quality trajectory is also not monotonic. Do not expect a clean
staircase: each round retrains on a fresh random sample, and the loop has no
revert-on-regression rule, so rounds can and do go backwards.

Fixes that would make this loop trustworthy, none of them implemented here:
  - score the distribution, not just each name (penalize repeats)
  - deduplicate the kept set before retraining
  - keep a diversity floor as a hard constraint, not a hope
  - revert a round when any tracked metric regresses
""")

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
     (the verifier is built from {len(heldout_docs)} held-out names the model never trains on)
  3. FILTER:   Keep only the top {KEEP_TOP_FRACTION:.0%} (verified good outputs)
  4. RETRAIN:  Mix {MIX_RATIO:.0%} self-generated + {1 - MIX_RATIO:.0%} real data, train {RETRAIN_STEPS} steps

Keeping only VERIFIED good outputs reduces the risk of collapse. Naive
self-training (keeping all outputs) lets the model amplify its own errors.
The quality filter mitigates this — though it does not fully prevent mode
collapse, and the numbers above show it did not prevent it here either.
Diversity controls, quality thresholds, and revert-on-regression would
strengthen the loop further.

Measured in this run:
  filtered    quality {final_quality:.4f}, unique {final_unique:.1%}
  unfiltered  quality {unfiltered_quality:.4f}, unique {unfiltered_unique:.1%}
  control     quality {control_quality:.4f}, unique {control_unique:.1%}

Do not memorize the ordering of those three numbers: it depends on the seed,
the scorer, and how many rounds you run. Memorize the habit of printing all
three columns, because "quality went up" on its own cannot tell you whether
the loop learned or narrowed.

This is a simplified version of the pattern behind:
  - STaR: keep self-generated rationales only if the answer is correct
  - Karpathy's autoresearch: keep code changes only if loss improves
  - Self-Rewarding LMs use a learned judge (not a fixed scorer like here)
  - SPIN uses distribution-matching self-play (a different mechanism)
""")
