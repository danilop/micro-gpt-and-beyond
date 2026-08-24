"""
microGPT: Evolutionary self-improvement edition.

Instead of improving a single model's outputs (Lab 23), this lab improves the
model itself through population-based evolution. A population of tiny GPT
variants -- each with different hyperparameters -- compete for survival.
The fittest are selected, mutated, and the cycle repeats across generations.

This implements Population Based Training (PBT) as described in "Population
Based Training of Neural Networks" (Jaderberg et al., 2017,
https://arxiv.org/abs/1711.09846). PBT combines random hyperparameter search
with online selection: instead of training one model to completion, train many
in parallel, periodically replacing the worst with mutated copies of the best.

Related approaches include "FunSearch: Making new discoveries in mathematical
sciences using large language models" (Romera-Paredes et al., 2023,
https://www.nature.com/articles/s41586-023-06924-6), which uses an evolutionary
loop over LLM-generated programs, and Karpathy's autoresearch
(https://github.com/karpathy/autoresearch), which runs a sequential version
of the same pattern: mutate config, train, evaluate, keep or discard.

This lab uses no external LLM. Mutations are random perturbations of
hyperparameters (learning rate, embedding dimension, number of heads). The
fitness function is validation loss on held-out names. Evolution discovers
good configurations that a single random guess would likely miss.

Note: this is a simplified evolutionary hyperparameter search inspired by
PBT, not a full PBT implementation. Canonical PBT copies weights into
underperformers in-place, preserves optimizer state across generations, and
perturbs continuous schedules. Here we use a discrete search space, fresh
optimizers per generation, and deepcopy survivors.

Three things this lab measures that toy evolution demos usually skip, because
each one deflates the headline number:
  - Global elitism. The best model of the run is archived when it is found.
    The best member of the FINAL population is often worse, because the
    population regresses after its peak generation.
  - The compute ledger. The evolutionary run spends thousands of training
    steps; the single random baseline spends 500. Beating it is mostly budget.
    So we also compare the two configurations at matched per-model budget,
    and add a random-search arm with the same total budget as evolution.
  - Diversity. Distinct architectures per generation is printed, because
    top-k truncation on a population of 8 collapses the gene pool quickly,
    and a population with one architecture left is not searching any more.
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

# Split into train/val
val_size = 1000
val_docs = docs[:val_size]
train_docs = docs[val_size:]

# Fitness is evaluated on a fixed prefix of the validation split, not all of it.
# 200 names is enough to rank models while keeping the search affordable: this
# lab evaluates ~150 models, so every extra sample costs 150 forward passes.
VAL_SAMPLES = 200
print(f"train: {len(train_docs)}, val: {len(val_docs)} (fitness uses the first {VAL_SAMPLES})")

# ---------------------------------------------------------------------------
# Flexible model (hyperparameters are configurable)
# ---------------------------------------------------------------------------
device = "cpu"
block_size = 16


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.relu(self.fc1(self.norm2(x))))
        return x


class MicroGPT(nn.Module):
    def __init__(self, n_embd, n_head, n_layer):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx):
        B, T = idx.shape
        x = self.norm_in(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Evolution primitives
# ---------------------------------------------------------------------------

# Hyperparameter search space
HP_SPACE = {
    "n_embd": [16, 24, 32, 48, 64],
    "n_head": [1, 2, 4],
    "n_layer": [1, 2],
    "lr": [5e-3, 8e-3, 1e-2, 1.5e-2, 2e-2],
    "beta1": [0.8, 0.85, 0.9, 0.95],
}


def random_config():
    """Sample a random hyperparameter configuration."""
    cfg = {k: random.choice(v) for k, v in HP_SPACE.items()}
    # Ensure n_embd is divisible by n_head
    while cfg["n_embd"] % cfg["n_head"] != 0:
        cfg["n_head"] = random.choice(HP_SPACE["n_head"])
    return cfg


def mutate_config(cfg):
    """Mutate one or two hyperparameters of a configuration."""
    new_cfg = dict(cfg)
    n_mutations = random.choice([1, 1, 2])  # usually 1, sometimes 2
    keys_to_mutate = random.sample(list(HP_SPACE.keys()), min(n_mutations, len(HP_SPACE)))
    for key in keys_to_mutate:
        new_cfg[key] = random.choice(HP_SPACE[key])
    # Ensure n_embd divisible by n_head
    while new_cfg["n_embd"] % new_cfg["n_head"] != 0:
        new_cfg["n_head"] = random.choice(HP_SPACE["n_head"])
    return new_cfg


def build_model(cfg):
    """Build a model from a configuration."""
    return MicroGPT(cfg["n_embd"], cfg["n_head"], cfg["n_layer"])


def train_model(model, cfg, train_data, num_steps, verbose=False, start_step=0):
    """Train a model for a fixed number of steps, return avg loss.

    `start_step` is where in `train_data` this call picks up. A model that has
    already seen 400 names must continue at name 400, not restart at name 0 --
    otherwise every generation re-reads the same first STEPS_PER_GEN names and
    "more steps" buys memorisation of a tiny slice rather than more data.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.99), eps=1e-8)
    total_loss = 0
    for step in range(num_steps):
        doc = train_data[(start_step + step) % len(train_data)]
        tokens = [BOS] + [char_to_id[ch] for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        input_ids = torch.tensor([tokens[:n]], device=device)
        targets = torch.tensor([tokens[1 : n + 1]], device=device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if verbose and ((step + 1) % 100 == 0 or step == 0):
            print(f"  step {step + 1:4d} / {num_steps} | loss {loss.item():.4f}")
    return total_loss / num_steps


@torch.no_grad()
def evaluate_model(model, val_data, max_samples=VAL_SAMPLES):
    """Evaluate validation loss (fitness = negative val loss)."""
    model.eval()
    total_loss = 0
    n_samples = min(max_samples, len(val_data))
    for i in range(n_samples):
        doc = val_data[i]
        tokens = [BOS] + [char_to_id[ch] for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        input_ids = torch.tensor([tokens[:n]])
        targets = torch.tensor([tokens[1 : n + 1]])
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / n_samples


@torch.no_grad()
def generate_names(model, n_samples=10, temperature=0.5):
    """Generate sample names from a model."""
    model.eval()
    names = []
    for _ in range(n_samples):
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]])
            logits = model(idx)[0, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
        names.append("".join(uchars[t] for t in tokens[1:]))
    model.train()
    return names


# ===========================================================================
# Phase 1: Random baseline, train one random configuration
# ===========================================================================
print(f"\n{'=' * 70}")
print("PHASE 1: Random baseline (single random configuration)")
print("=" * 70)

BASELINE_STEPS = 500

baseline_cfg = random_config()
print(f"config: {baseline_cfg}")
baseline_model = build_model(baseline_cfg)
n_params = sum(p.numel() for p in baseline_model.parameters())
print(f"params: {n_params}")

# 500 steps is a small budget, and that matters for every comparison below.
# The evolutionary run further down spends thousands of steps across its whole
# population, so "evolved beats this baseline" would mostly be a statement
# about compute. The results section therefore also compares the two configs
# at matched per-model budget, and adds a random-search arm with the same total
# budget as evolution.
train_model(baseline_model, baseline_cfg, train_docs, BASELINE_STEPS, verbose=True)
baseline_val_loss = evaluate_model(baseline_model, val_docs)
print(f"val loss: {baseline_val_loss:.4f}")
print(f"samples: {', '.join(generate_names(baseline_model, 5))}")

# ===========================================================================
# Phase 2: Population-Based Evolution
# ===========================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Population-Based Training (evolutionary self-improvement)")
print("=" * 70)

POP_SIZE = 8
NUM_GENERATIONS = 6
STEPS_PER_GEN = 200
TOP_K = 3  # keep top-k, replace the rest

# Initialize population with random configurations
population: list[dict] = []
for i in range(POP_SIZE):
    cfg = random_config()
    model = build_model(cfg)
    # "steps" is the cumulative training this member's weights have received.
    # It is not the same for every member: survivors keep their weights and
    # keep accumulating, while a child whose architecture changed starts from
    # scratch at 0. Without this counter, within-generation fitness silently
    # confounds configuration quality with lineage age.
    population.append({"id": i, "cfg": cfg, "model": model, "val_loss": float("inf"), "steps": 0})

print(f"population size: {POP_SIZE}")
print(f"generations: {NUM_GENERATIONS}")
print(f"steps/generation: {STEPS_PER_GEN}")
print(f"selection: top-{TOP_K} survive, rest are replaced by mutated copies\n")

generation_stats: list[dict] = []
evolution_steps = 0  # total training steps spent by the whole population

# Global elitism: an archive of the single best model ever evaluated. Without
# it the "evolved best" is just whatever happens to be in the final
# population, which is not the same thing as the best the search found,
# in this lab the population regularly peaks in a middle generation and then
# regresses.
best_ever: dict = {"val_loss": float("inf"), "cfg": None, "model": None, "gen": 0, "steps": 0}

for gen in range(NUM_GENERATIONS):
    # Train each member for STEPS_PER_GEN
    for member in population:
        # start_step: this member resumes at the name after the last one it saw.
        train_model(member["model"], member["cfg"], train_docs, STEPS_PER_GEN, start_step=member["steps"])
        member["steps"] += STEPS_PER_GEN
        evolution_steps += STEPS_PER_GEN
        member["val_loss"] = evaluate_model(member["model"], val_docs)

    # Sort by fitness (lower val loss = better)
    population.sort(key=lambda m: m["val_loss"])

    best = population[0]
    worst = population[-1]
    avg_loss = sum(m["val_loss"] for m in population) / len(population)

    # How much variety is actually left in the population? Two counts: distinct
    # architectures (what changes the parameter count) and distinct full
    # configs (architecture plus lr and beta1).
    n_archs = len({(m["cfg"]["n_embd"], m["cfg"]["n_head"], m["cfg"]["n_layer"]) for m in population})
    n_cfgs = len({tuple(sorted(m["cfg"].items())) for m in population})

    if best["val_loss"] < best_ever["val_loss"]:
        best_ever = {
            "val_loss": best["val_loss"],
            "cfg": dict(best["cfg"]),
            "model": copy.deepcopy(best["model"]),
            "gen": gen + 1,
            "steps": best["steps"],
        }

    gen_stat = {
        "gen": gen + 1,
        "best_loss": best["val_loss"],
        "worst_loss": worst["val_loss"],
        "avg_loss": avg_loss,
        "best_cfg": dict(best["cfg"]),
        "best_params": sum(p.numel() for p in best["model"].parameters()),
        "best_steps": best["steps"],
        "n_archs": n_archs,
        "n_cfgs": n_cfgs,
    }
    generation_stats.append(gen_stat)

    print(
        f"gen {gen + 1:2d}: "
        f"best {best['val_loss']:.4f} (id={best['id']}, {best['steps']} steps), "
        f"worst {worst['val_loss']:.4f}, "
        f"avg {avg_loss:.4f}, "
        f"distinct archs {n_archs}/{POP_SIZE}, configs {n_cfgs}/{POP_SIZE}, "
        f"best cfg: embd={best['cfg']['n_embd']}, heads={best['cfg']['n_head']}, "
        f"layers={best['cfg']['n_layer']}, lr={best['cfg']['lr']:.4f}"
    )
    # Cumulative steps next to val loss, so you can see the confound directly.
    for rank, m in enumerate(population):
        c = m["cfg"]
        print(
            f"       rank {rank + 1}: val {m['val_loss']:.4f}  steps {m['steps']:5d}  "
            f"embd={c['n_embd']:2d}, heads={c['n_head']}, layers={c['n_layer']}, "
            f"lr={c['lr']:.4f}, beta1={c['beta1']}"
        )

    # Evolution: replace bottom members with mutated copies of top members
    survivors = population[:TOP_K]
    new_population = copy.deepcopy(survivors)  # keep survivors as-is (with their trained weights)

    for i in range(POP_SIZE - TOP_K):
        # Pick a random survivor as parent
        parent = random.choice(survivors)
        child_cfg = mutate_config(parent["cfg"])

        # If architecture changed, need a new model (can't inherit weights)
        arch_changed = (
            child_cfg["n_embd"] != parent["cfg"]["n_embd"]
            or child_cfg["n_head"] != parent["cfg"]["n_head"]
            or child_cfg["n_layer"] != parent["cfg"]["n_layer"]
        )

        child_model = build_model(child_cfg)
        if not arch_changed:
            # Same architecture: inherit parent's weights (the PBT "exploit"
            # step), and inherit the parent's accumulated step count with them.
            child_model.load_state_dict(parent["model"].state_dict())

        new_population.append(
            {
                "id": POP_SIZE * (gen + 1) + i,
                "cfg": child_cfg,
                "model": child_model,
                "val_loss": float("inf"),
                # A child with a new architecture is untrained: 0 steps. A child
                # that inherited weights starts where its parent left off.
                "steps": 0 if arch_changed else parent["steps"],
            }
        )

    population = new_population

# ===========================================================================
# Results
# ===========================================================================
print(f"\n{'=' * 70}")
print("EVOLUTION RESULTS")
print("=" * 70)

# Final evaluation: train and evaluate all members of the last population
# (children spawned in the last generation haven't been trained yet)
for member in population:
    if member["val_loss"] == float("inf"):
        train_model(member["model"], member["cfg"], train_docs, STEPS_PER_GEN, start_step=member["steps"])
        member["steps"] += STEPS_PER_GEN
        evolution_steps += STEPS_PER_GEN
        member["val_loss"] = evaluate_model(member["model"], val_docs)
        if member["val_loss"] < best_ever["val_loss"]:
            best_ever = {
                "val_loss": member["val_loss"],
                "cfg": dict(member["cfg"]),
                "model": copy.deepcopy(member["model"]),
                "gen": NUM_GENERATIONS,  # spawned in the last generation, trained here
                "steps": member["steps"],
            }

population.sort(key=lambda m: m["val_loss"])
final_best = population[0]  # best in the final population
best_evolved = best_ever  # best ever seen, which is what we report
evolved_val_loss = best_evolved["val_loss"]

print(
    f"\n{'Gen':>4s}  {'Best Loss':>9s}  {'Avg Loss':>8s}  {'Worst Loss':>10s}  "
    f"{'Steps':>6s}  {'Archs':>5s}  {'Cfgs':>4s}  {'Best Config'}"
)
print("-" * 100)
for s in generation_stats:
    c = s["best_cfg"]
    print(
        f"{s['gen']:>4d}  {s['best_loss']:>9.4f}  {s['avg_loss']:>8.4f}  {s['worst_loss']:>10.4f}  "
        f"{s['best_steps']:>6d}  {s['n_archs']:>5d}  {s['n_cfgs']:>4d}  "
        f"embd={c['n_embd']}, heads={c['n_head']}, layers={c['n_layer']}, lr={c['lr']:.4f}"
    )
print("(Steps = cumulative training steps of that generation's best member.")
print(" Archs/Cfgs = distinct architectures and distinct full configs in the population.)")

# The best of the final population is not the best of the run. Say so with numbers.
print(f"\nbest in final population: {final_best['val_loss']:.4f} ({final_best['steps']} steps)")
print(
    f"best ever seen:           {best_ever['val_loss']:.4f} (generation {best_ever['gen']}, {best_ever['steps']} steps)"
)
gen_best_curve = ", ".join(f"{s['best_loss']:.4f}" for s in generation_stats)
print(f"per-generation best:      {gen_best_curve}")
if best_ever["gen"] < NUM_GENERATIONS:
    print(
        f"  -> the search peaked at generation {best_ever['gen']} and then regressed. "
        "Global elitism is why we can still report the peak."
    )

# ===========================================================================
# Was any of this fair? The compute ledger
# ===========================================================================
print(f"\n{'=' * 70}")
print("COMPUTE LEDGER: the headline number is mostly budget")
print("=" * 70)

# Arm A: the same 500-step baseline, but let its config keep training so we can
# compare configurations at matched per-model budget instead of comparing
# 500 steps against a whole population's worth of training.
MATCHED_STEPS = 1200
CHECK_EVERY = 200


def budget_curve(cfg, total_steps=MATCHED_STEPS, check_every=CHECK_EVERY, seed=7):
    """
    Train one config from scratch and record val loss every `check_every` steps.

    Each chunk builds a fresh optimizer, exactly as a generation of evolution
    does, so this curve is the like-for-like comparison rather than a single
    long run with persistent Adam state.
    """
    torch.manual_seed(seed)
    m = build_model(cfg)
    curve = []
    for chunk in range(total_steps // check_every):
        train_model(m, cfg, train_docs, check_every, start_step=chunk * check_every)
        curve.append(evaluate_model(m, val_docs))
    return curve


print(f"\nSame two configs, trained from scratch, val loss every {CHECK_EVERY} steps:")
base_curve = budget_curve(baseline_cfg)
eval_curve = budget_curve(best_evolved["cfg"])
header = "  ".join(f"{(i + 1) * CHECK_EVERY:>7d}" for i in range(MATCHED_STEPS // CHECK_EVERY))
print(f"\n{'config':>16s}  {header}")
print("-" * (18 + 9 * (MATCHED_STEPS // CHECK_EVERY)))
print(f"{'baseline':>16s}  " + "  ".join(f"{v:>7.4f}" for v in base_curve))
print(f"{'evolved':>16s}  " + "  ".join(f"{v:>7.4f}" for v in eval_curve))
matched_delta = min(base_curve) - min(eval_curve)
print(f"\nbest-of-curve: baseline {min(base_curve):.4f}, evolved {min(eval_curve):.4f}, delta {matched_delta:+.4f}")
print("That delta is the configuration advantage at matched per-model budget.")

# Arm B: random search at (almost) the same total training budget as evolution.
# This is the control that tells you whether selection and mutation earned
# anything, or whether spending the same compute on independent random configs
# would have done as well. POP_SIZE x NUM_GENERATIONS configs at STEPS_PER_GEN
# steps each is slightly less compute than the evolutionary run (which also
# trains the children spawned in the last generation), so the comparison is
# conservative in evolution's favour.
# This is the most expensive block in the lab. It is also the only reason the
# headline number below can be trusted.
rs_configs = POP_SIZE * NUM_GENERATIONS
print(
    f"\nRandom search control: {rs_configs} random configs x {STEPS_PER_GEN} steps = {rs_configs * STEPS_PER_GEN} steps"
)
rs_best = {"val_loss": float("inf"), "cfg": None, "model": None}
for i in range(rs_configs):
    cfg = random_config()
    m = build_model(cfg)
    train_model(m, cfg, train_docs, STEPS_PER_GEN)
    vl = evaluate_model(m, val_docs)
    if vl < rs_best["val_loss"]:
        rs_best = {"val_loss": vl, "cfg": cfg, "model": m}
    if (i + 1) % 10 == 0:
        print(f"  {i + 1:3d}/{rs_configs} configs sampled, best so far {rs_best['val_loss']:.4f}")
print(f"random search best: {rs_best['val_loss']:.4f} (config: {rs_best['cfg']})")

print("\n--- training steps spent ---")
print(f"  single random baseline:      {BASELINE_STEPS:>7,d}")
print(f"  evolution (whole run):       {evolution_steps:>7,d}  ({evolution_steps / BASELINE_STEPS:.0f}x the baseline)")
print(f"  random search (matched):     {rs_configs * STEPS_PER_GEN:>7,d}")
print(f"  matched-budget head-to-head: {MATCHED_STEPS:>7,d}  per config")

print("\n--- comparison ---")
print(f"random baseline ({BASELINE_STEPS} steps):  val_loss={baseline_val_loss:.4f} (config: {baseline_cfg})")
print(f"evolved best (best ever):     val_loss={evolved_val_loss:.4f} (config: {best_evolved['cfg']})")
print(f"random search (equal budget): val_loss={rs_best['val_loss']:.4f}")
improvement = baseline_val_loss - evolved_val_loss
print(
    f"\nevolved vs {BASELINE_STEPS}-step baseline:   {improvement:+.4f}  <- NOT a fair comparison, {evolution_steps / BASELINE_STEPS:.0f}x the compute"
)
print(
    f"evolved vs equal-budget search: {rs_best['val_loss'] - evolved_val_loss:+.4f}  <- what evolution bought over random search"
)
print(f"config advantage, matched budget: {matched_delta:+.4f}  <- the honest per-config number")

# Generate from both
print("\n--- random baseline samples ---")
for i, name in enumerate(generate_names(baseline_model, 10)):
    print(f"  {i + 1:2d}: {name}")

print("\n--- evolved best samples ---")
for i, name in enumerate(generate_names(best_evolved["model"], 10)):
    print(f"  {i + 1:2d}: {name}")

# Show the evolutionary tree
print("\n--- population diversity (final generation) ---")
for i, member in enumerate(population):
    c = member["cfg"]
    n_p = sum(p.numel() for p in member["model"].parameters())
    print(
        f"  rank {i + 1}: val_loss={member['val_loss']:.4f}, steps={member['steps']:5d}, "
        f"params={n_p:,}, "
        f"embd={c['n_embd']}, heads={c['n_head']}, layers={c['n_layer']}, "
        f"lr={c['lr']:.4f}, beta1={c['beta1']}"
    )

final_archs = len({(m["cfg"]["n_embd"], m["cfg"]["n_head"], m["cfg"]["n_layer"]) for m in population})
final_cfgs = len({tuple(sorted(m["cfg"].items())) for m in population})
final_sizes = {sum(p.numel() for p in m["model"].parameters()) for m in population}
final_shapes = len({(m["cfg"]["n_embd"], m["cfg"]["n_layer"]) for m in population})
arch_curve = ", ".join(str(s["n_archs"]) for s in generation_stats)
avg_curve = ", ".join(f"{s['avg_loss']:.4f}" for s in generation_stats)
print(f"\ndistinct architectures per generation: {arch_curve} (final population: {final_archs}/{POP_SIZE})")
print(f"distinct full configs in final population: {final_cfgs}/{POP_SIZE}")
print(f"distinct model sizes in final population: {len(final_sizes)}/{POP_SIZE} ({sorted(final_sizes)} params)")
print(f"population average val loss per generation: {avg_curve}")
if len(final_sizes) == 1 or final_shapes == 1:
    print(
        "\n  -> Read those lines again: every surviving member is the same size.\n"
        "     Whatever variety is left is in lr, beta1 and the head count, none of\n"
        "     which changes the parameter count. There is effectively no\n"
        "     architectural diversity left to select from. Top-k truncation with a\n"
        "     low mutation rate collapses a gene pool this small very fast.\n"
        "     A real search needs a diversity floor: forbid duplicate genomes, keep\n"
        "     a fraction of random immigrants, or raise the mutation rate when\n"
        "     variance drops. None of that is implemented here, and measuring it first\n"
        "     is the point."
    )
if generation_stats[-1]["avg_loss"] > generation_stats[0]["avg_loss"]:
    print(
        f"\n  -> The population AVERAGE got worse over the run "
        f"({generation_stats[0]['avg_loss']:.4f} -> {generation_stats[-1]['avg_loss']:.4f}).\n"
        "     Children with fresh architectures are untrained, so a population that\n"
        "     keeps mutating keeps paying a restart cost. 'Evolution improved the\n"
        "     population' is not a claim this run supports."
    )

# ===========================================================================
# Explanation
# ===========================================================================
print(f"""
{"=" * 70}
HOW EVOLUTIONARY SELF-IMPROVEMENT WORKS
{"=" * 70}

Population Based Training (PBT) evolves model configurations:

  1. INITIALIZE: Create {POP_SIZE} models with random hyperparameters
  2. TRAIN:      Each model trains for {STEPS_PER_GEN} steps
  3. EVALUATE:   Measure validation loss (fitness)
  4. SELECT:     Keep the top {TOP_K} models (survivors)
  5. MUTATE:     Create {POP_SIZE - TOP_K} children by mutating survivors' configs
  6. INHERIT:    If architecture unchanged, children inherit parent weights
  7. REPEAT:     Go to step 2 for {NUM_GENERATIONS} generations

Key differences from Lab 23 (self-improving model):
  - Lab 23 improves the MODEL'S OUTPUTS (better training data)
  - Lab 24 improves the MODEL ITSELF (better architecture/hyperparameters)
  - Lab 23 uses one model; Lab 24 uses a competing population
  - Lab 23 is like STaR; Lab 24 is like natural selection

The "exploit + explore" balance:
  - EXPLOIT: children inherit trained weights from successful parents
  - EXPLORE: hyperparameter mutations introduce diversity

This is the same pattern behind:
  - PBT (Jaderberg et al.): evolve learning rates and augmentation during training
  - Karpathy's autoresearch: sequential mutate-train-evaluate-keep loop
  - FunSearch: evolve programs scored by a fitness function
  - Neural Architecture Search: evolve network topologies

THREE THINGS THIS RUN MEASURED, WHICH ARE EASY TO GET WRONG:

  1. The headline is mostly compute. Evolution spent {evolution_steps:,} training
     steps against the baseline's 500. At matched per-config budget the
     configuration advantage was {matched_delta:+.4f}, not {improvement:+.4f}.

  2. Fitness within a generation is not a clean config comparison. Survivors
     keep their weights and keep accumulating steps; children with a new
     architecture restart at zero. The "steps" column shows the confound.
     Each generation gives every member the same NUMBER of new steps, which
     is not the same as giving them the same total training.

  3. Diversity dies quietly. Distinct architectures per generation:
     {arch_curve}. Once that hits 1, selection has nothing left to select.

The reported winner comes from a best-ever archive, not from the final
population, because the two are frequently not the same model.

At production scale, PBT discovers training schedules that would take
human researchers weeks of manual tuning. It also runs with populations large
enough, and mutation rates high enough, that the gene pool survives, which is
exactly the part a toy run at this size cannot show you for free.
""")
