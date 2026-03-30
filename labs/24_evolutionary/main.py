"""
microGPT — Evolutionary self-improvement edition.

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
print(f"train: {len(train_docs)}, val: {len(val_docs)}")

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


def train_model(model, cfg, train_data, num_steps, verbose=False):
    """Train a model for a fixed number of steps, return avg loss."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], 0.99), eps=1e-8
    )
    total_loss = 0
    for step in range(num_steps):
        doc = train_data[step % len(train_data)]
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
def evaluate_model(model, val_data, max_samples=200):
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
# Phase 1: Random baseline — train one random configuration
# ===========================================================================
print(f"\n{'=' * 70}")
print("PHASE 1: Random baseline (single random configuration)")
print("=" * 70)

baseline_cfg = random_config()
print(f"config: {baseline_cfg}")
baseline_model = build_model(baseline_cfg)
n_params = sum(p.numel() for p in baseline_model.parameters())
print(f"params: {n_params}")

train_model(baseline_model, baseline_cfg, train_docs, 500, verbose=True)
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
population = []
for i in range(POP_SIZE):
    cfg = random_config()
    model = build_model(cfg)
    population.append({"id": i, "cfg": cfg, "model": model, "val_loss": float("inf")})

print(f"population size: {POP_SIZE}")
print(f"generations: {NUM_GENERATIONS}")
print(f"steps/generation: {STEPS_PER_GEN}")
print(f"selection: top-{TOP_K} survive, rest are replaced by mutated copies\n")

generation_stats = []

for gen in range(NUM_GENERATIONS):
    # Train each member for STEPS_PER_GEN
    for member in population:
        train_model(member["model"], member["cfg"], train_docs, STEPS_PER_GEN)
        member["val_loss"] = evaluate_model(member["model"], val_docs)

    # Sort by fitness (lower val loss = better)
    population.sort(key=lambda m: m["val_loss"])

    best = population[0]
    worst = population[-1]
    avg_loss = sum(m["val_loss"] for m in population) / len(population)

    gen_stat = {
        "gen": gen + 1,
        "best_loss": best["val_loss"],
        "worst_loss": worst["val_loss"],
        "avg_loss": avg_loss,
        "best_cfg": dict(best["cfg"]),
        "best_params": sum(p.numel() for p in best["model"].parameters()),
    }
    generation_stats.append(gen_stat)

    print(
        f"gen {gen + 1:2d}: "
        f"best {best['val_loss']:.4f} (id={best['id']}), "
        f"worst {worst['val_loss']:.4f}, "
        f"avg {avg_loss:.4f}, "
        f"best cfg: embd={best['cfg']['n_embd']}, heads={best['cfg']['n_head']}, "
        f"layers={best['cfg']['n_layer']}, lr={best['cfg']['lr']:.4f}"
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

        if arch_changed:
            child_model = build_model(child_cfg)
        else:
            # Same architecture: inherit parent's weights (the PBT "exploit" step)
            child_model = build_model(child_cfg)
            child_model.load_state_dict(parent["model"].state_dict())

        new_population.append({
            "id": POP_SIZE * (gen + 1) + i,
            "cfg": child_cfg,
            "model": child_model,
            "val_loss": float("inf"),
        })

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
        train_model(member["model"], member["cfg"], train_docs, STEPS_PER_GEN)
        member["val_loss"] = evaluate_model(member["model"], val_docs)

population.sort(key=lambda m: m["val_loss"])
best_evolved = population[0]
evolved_val_loss = best_evolved["val_loss"]

print(f"\n{'Gen':>4s}  {'Best Loss':>9s}  {'Avg Loss':>8s}  {'Worst Loss':>10s}  {'Best Config'}")
print("-" * 70)
for s in generation_stats:
    c = s["best_cfg"]
    print(
        f"{s['gen']:>4d}  {s['best_loss']:>9.4f}  {s['avg_loss']:>8.4f}  {s['worst_loss']:>10.4f}  "
        f"embd={c['n_embd']}, heads={c['n_head']}, layers={c['n_layer']}, lr={c['lr']:.4f}"
    )

print(f"\n--- comparison ---")
print(f"random baseline:   val_loss={baseline_val_loss:.4f} (config: {baseline_cfg})")
print(f"evolved best:      val_loss={evolved_val_loss:.4f} (config: {best_evolved['cfg']})")
improvement = baseline_val_loss - evolved_val_loss
print(f"improvement:       {improvement:+.4f} ({'better' if improvement > 0 else 'worse'})")

# Generate from both
print(f"\n--- random baseline samples ---")
for i, name in enumerate(generate_names(baseline_model, 10)):
    print(f"  {i + 1:2d}: {name}")

print(f"\n--- evolved best samples ---")
for i, name in enumerate(generate_names(best_evolved["model"], 10)):
    print(f"  {i + 1:2d}: {name}")

# Show the evolutionary tree
print(f"\n--- population diversity (final generation) ---")
for i, member in enumerate(population):
    c = member["cfg"]
    n_p = sum(p.numel() for p in member["model"].parameters())
    print(
        f"  rank {i + 1}: val_loss={member['val_loss']:.4f}, "
        f"params={n_p:,}, "
        f"embd={c['n_embd']}, heads={c['n_head']}, layers={c['n_layer']}, "
        f"lr={c['lr']:.4f}, beta1={c['beta1']}"
    )

# ===========================================================================
# Explanation
# ===========================================================================
print(f"""
{'=' * 70}
HOW EVOLUTIONARY SELF-IMPROVEMENT WORKS
{'=' * 70}

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

At our tiny scale, evolution finds good configurations in {NUM_GENERATIONS} generations.
At production scale, PBT discovers training schedules that would take
human researchers weeks of manual tuning.
""")
