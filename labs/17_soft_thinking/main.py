"""
microGPT — Soft thinking edition.

Same architecture as the PyTorch version (03), but with soft decoding at
inference time. Instead of collapsing to a single token at each step, soft
thinking passes a "concept token" -- a probability-weighted blend of all
token embeddings -- to the next step. The full distribution flows forward,
preserving information that hard decoding discards.

  Hard:  logits -> sample -> embed(token)            -> next input
  Soft:  logits -> softmax(logits/T) @ embed_table   -> next input

Based on "Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous
Concept Space" (Zhang et al., 2025), https://arxiv.org/abs/2505.15778. The
concept token computation follows the paper's formulation. Note that this is an
inference-only technique -- the model is trained with standard teacher forcing on
hard tokens, which creates a train-test distribution mismatch that Lab 18 (soft
training) addresses.
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
# Model (same as Lab 03, with inputs_embeds support for concept tokens)
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
# Lab 18 duplicates the model and this generation function rather than importing
# them: each lab is meant to run standalone, from a single file, with no
# cross-lab imports to trace.
num_steps = 1000


def entropy_of(p):
    """Shannon entropy in nats of a probability vector."""
    return -(p * p.clamp(min=1e-10).log()).sum().item()


@torch.no_grad()
def generate(model, mode="hard", soft_temp=1.0, temperature=0.5):
    """Generate a name using hard (discrete) or soft (concept token) decoding."""
    # No dropout and no batch norm anywhere in this model, so eval() changes
    # nothing here. Kept because it is the habit you want everywhere else.
    model.eval()
    tokens, entropies = [], []
    embeds = model.wte(torch.tensor([[BOS]], device=device))

    for _ in range(block_size):
        logits = model(inputs_embeds=embeds)[0, -1]

        # Sample a discrete token (for output and stopping)
        probs = F.softmax(logits / temperature, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        if token_id == BOS:
            break
        tokens.append(token_id)

        # Next input: discrete embedding or concept token.
        #
        # The entropy we report is the entropy of the distribution that actually
        # BUILDS the next input, which is the only one soft thinking changes.
        # Measuring the sampling distribution softmax(logits/temperature) instead
        # would be measuring the wrong thing: it is the same for every row of the
        # table below, so it would read flat no matter what soft_temp does.
        if mode == "hard":
            next_emb = model.wte(torch.tensor([[token_id]], device=device))
            # Hard decoding feeds exactly one embedding. Its input distribution
            # is a one-hot delta, so its entropy is exactly 0 — that IS the
            # information bottleneck, stated as a number.
            entropies.append(0.0)
        else:
            # Concept token: probability-weighted blend of ALL token embeddings
            soft_probs = F.softmax(logits / soft_temp, dim=-1)
            next_emb = (soft_probs @ model.wte.weight).view(1, 1, -1)
            entropies.append(entropy_of(soft_probs))

        embeds = torch.cat([embeds, next_emb], dim=1)[:, -block_size:]

    return tokens, entropies


# ---------------------------------------------------------------------------
# The cost side: is the output still in distribution?
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_nll(model, names):
    """Mean per-token NLL (nats) of `names` under the model, with hard inputs.

    This asks "would the model itself have written this?". A string the model
    considers unlikely on its normal hard path scores badly here, which is what
    out-of-distribution drift looks like from the outside. Compare any row
    against the real-names reference at the bottom of the table.
    """
    total, count = 0.0, 0
    for name in names:
        if not name:
            continue
        tokens = [BOS] + [uchars.index(ch) for ch in name] + [BOS]
        n = min(block_size, len(tokens) - 1)
        logits = model(torch.tensor([tokens[:n]], device=device))
        targets = torch.tensor([tokens[1 : n + 1]], device=device)
        total += F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="sum").item()
        count += n
    return total / max(count, 1)


def dup_rate(names):
    """Fraction of adjacent character pairs that repeat the same character.

    Real names do this a little (emma, aaron). Soft decoding at high temperature
    does it constantly. The tempting explanation is that a diffuse concept token
    carries little information about which token was just emitted, so the model
    loses track and stutters -- plausible, but this lab measures the stutter, not
    the mechanism. One number, and it catches the specific way this fails.
    """
    pairs = dups = 0
    for name in names:
        for i in range(1, len(name)):
            pairs += 1
            dups += name[i] == name[i - 1]
    return dups / max(pairs, 1)


# ---------------------------------------------------------------------------
# Training & Inference (runs only when executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = MicroGPT().to(device)
    print(f"num params: {sum(p.numel() for p in model.parameters())}")

    # Training — identical to Lab 03
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        input_ids = torch.tensor([tokens[:n]], device=device)
        targets = torch.tensor([tokens[1 : n + 1]], device=device)

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        lr_t = 1e-2 * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        optimizer.step()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

    # Inference — hard vs. soft decoding at different temperatures
    print("\n--- soft thinking comparison ---\n")
    n_samples = 50
    max_H = math.log(vocab_size)
    rows = []

    for mode, soft_temp, label in [
        ("hard", 1.0, "hard (standard decoding)"),
        ("soft", 0.5, "soft T=0.5 (mild blend)"),
        ("soft", 1.0, "soft T=1.0 (moderate blend)"),
        ("soft", 2.0, "soft T=2.0 (diffuse blend)"),
    ]:
        print(f"{label}:")
        all_H, names = [], []
        for i in range(n_samples):
            toks, ents = generate(model, mode, soft_temp)
            name = "".join(uchars[t] for t in toks)
            names.append(name)
            avg_H = sum(ents) / len(ents) if ents else 0
            all_H.append(avg_H)
            if i < 10:
                print(f"  {i + 1:2d}: {name:<15s} concept-token entropy {avg_H:.3f}/{max_H:.2f}")
        mean_H = sum(all_H) / len(all_H)
        print(f"  -> mean concept-token entropy: {mean_H:.3f}/{max_H:.2f}\n")
        rows.append((label, mean_H, score_nll(model, names), dup_rate(names)))

    # Reference row: names the model never trained on (training walked docs[0:1000]).
    real = docs[num_steps : num_steps + 500]
    real_nll, real_dup = score_nll(model, real), dup_rate(real)

    # ---------------------------------------------------------------------
    # Benefit and cost, side by side
    # ---------------------------------------------------------------------
    # Entropy alone only measures the BENEFIT (how much of the distribution
    # survives into the next step). On its own it would make T=2.0 look best.
    # The other two columns are the price.
    hot_dup = rows[-1][3]  # the T=2.0 row, the one the prose points at
    print("--- benefit vs. cost ---\n")
    print(f"  {'mode':<28s} {'concept H':>9s}  {'sample NLL':>10s}  {'dup rate':>8s}")
    print(f"  {'-' * 28} {'-' * 9}  {'-' * 10}  {'-' * 8}")
    for label, mean_H, nll, dup in rows:
        print(f"  {label:<28s} {mean_H:>9.3f}  {nll:>10.4f}  {dup:>7.1%}")
    print(f"  {'real held-out names (' + str(len(real)) + ')':<28s} {'-':>9s}  {real_nll:>10.4f}  {real_dup:>7.1%}")
    print(f"""
  concept H   entropy of the distribution that builds the next input (max {max_H:.2f}).
              Hard decoding is exactly 0: one token in, everything else discarded.
  sample NLL  per-token NLL of the generated names under the model itself, hard
              inputs. Read it as a trend down the column: it climbs with soft
              temperature, which is the output drifting away from what the model
              was trained on. Do not read it as pass/fail against the real-names
              row — the sampling temperature of 0.5 sharpens every generated row,
              so they all score below real names, drift included.
  dup rate    adjacent repeated characters, of any kind — the stutter visible in
              the T=2.0 block above. Real names do this {real_dup:.1%} of the time;
              T=2.0 soft decoding does it {hot_dup / real_dup:.1f}x as often.
""")

    print(f"""--- what's happening ---

Standard (hard) decoding collapses the model's rich output distribution to a
single sampled token ID at every step (sampled, not argmax — see the
torch.multinomial call above). The next step sees only one embedding — all
information about what the model "almost said" is discarded.

Soft thinking preserves this information:

  concept_token = softmax(logits / T) @ embedding_table

The concept token is a {n_embd}-dimensional vector blending all {vocab_size} token
embeddings, weighted by their probability. It lives in the same embedding space
as regular tokens but encodes the model's full uncertainty.

Temperature (T) controls the softness:
  T -> 0:  concept token = argmax embedding (hard, no benefit)
  T = 1:   standard softmax (moderate blending)
  T -> inf: uniform weights (noise — all tokens equally blended)

The entropy column shows how "spread" the next-input distribution is. Higher
entropy means more tokens contribute to the concept token — richer information,
but higher risk of out-of-distribution drift. Max entropy = ln({vocab_size}) = {max_H:.2f}.

The tradeoff in that sentence is not rhetorical: the table above measures both
halves of it. Entropy rises monotonically with T, and so do both costs: the NLL
of what comes out, and the rate of repeated adjacent characters. Why the damage
takes the form of repetition in particular is not something this lab establishes
-- the natural guess is that a flat blend says little about which token was just
emitted -- but that the output degrades, and degrades in that specific way, is
measured.

This is training-free — no model weights change. The Soft Thinking paper
(Zhang et al., 2025) reports +2.5 pass@1 while using 22% fewer tokens on large
reasoning models. That is their measurement at their scale, not this lab's.
""")
