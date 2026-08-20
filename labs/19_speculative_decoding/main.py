"""
microGPT — Speculative decoding edition.

Same architecture as the PyTorch version (03/16), but demonstrating
speculative decoding: a small "draft" model guesses multiple tokens ahead,
then the larger "target" model verifies them all in a single forward pass.
The output distribution is mathematically identical to the target model alone.

This is the #1 technique used in production inference systems (vLLM,
TensorRT-LLM, SGLang) because autoregressive decoding is memory-bound —
each token requires reading all model weights but does very little compute.
"""

import math
import os
import random
import time

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
# Model — two sizes: a small draft model and a larger target model
# ---------------------------------------------------------------------------
block_size = 16  # maximum sequence length


class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
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


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
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

# Draft model: small, fast, less accurate
draft_cfg = {"n_embd": 32, "n_head": 4, "n_layer": 1}
# Target model: larger and better quality. It is 7.1x the draft's parameter count
# (102,784 vs 14,528), but do NOT translate that into a per-forward-pass cost
# ratio — the two are unrelated at this scale, where Python and dispatch overhead
# dominate the arithmetic. The measured ratio is printed further down.
target_cfg = {"n_embd": 64, "n_head": 4, "n_layer": 2}


def train_model(model, name, num_steps=1000):
    print(f"\n--- training {name} ({sum(p.numel() for p in model.parameters())} params) ---")
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
        if (step + 1) % 200 == 0 or step == 0:
            print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")


draft_model = MicroGPT(**draft_cfg).to(device)
target_model = MicroGPT(**target_cfg).to(device)

train_model(draft_model, "draft")
train_model(target_model, "target")

# ---------------------------------------------------------------------------
# Inference methods
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high


@torch.no_grad()
def get_probs(model, tokens):
    """Get probability distribution for the next token(s) given a token sequence."""
    idx = torch.tensor([tokens[-block_size:]], device=device)
    logits = model(idx)[0] / temperature  # (T, V)
    return F.softmax(logits, dim=-1)  # (T, V)


# --- 1) Naive autoregressive decoding (baseline) ---


def generate_autoregressive(model, max_len=block_size):
    """Standard decoding: one token at a time, one forward pass per token."""
    tokens = [BOS]
    with torch.no_grad():
        for _ in range(max_len):
            probs = get_probs(model, tokens)[-1]  # last position
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
    return tokens[1:]  # strip BOS


# --- 2) Speculative decoding ---


def generate_speculative(draft, target, max_len=block_size, K=4):
    """
    Speculative decoding: the draft model proposes K tokens, the target
    model verifies them all in one forward pass.

    The acceptance/rejection algorithm guarantees the output distribution
    is IDENTICAL to target-only generation:

    For each drafted token at position i with draft probability q(x):
      - Target model gives probability p(x) for the same token
      - Accept with probability min(1, p(x) / q(x))
      - On rejection: sample from adjusted distribution max(0, p(x) - q(x))
        (normalized), which corrects for the draft model's bias

    This is mathematically proven to sample from p(x) exactly.

    This technique was introduced in "Fast Inference from Transformers via
    Speculative Decoding" by Leviathan, Kalman, and Matias (2023),
    available at https://arxiv.org/abs/2211.17192. A concurrent and closely
    related approach was presented in "Accelerating Large Language Model
    Decoding with Speculative Sampling" by Chen, Borgeaud, Irving et al.
    (2023), available at https://arxiv.org/abs/2302.01318. The implementation
    here faithfully follows the accept/reject algorithm from Leviathan et al.
    (2023), which provably preserves the target model's output distribution.
    """
    tokens = [BOS]
    # drafted   = every token the draft model proposed (all K per round)
    # evaluated = tokens the target actually made an accept/reject decision on.
    #             These differ: after the first rejection the rest of the window
    #             is thrown away unexamined, so accepted/drafted is "how much of
    #             the draft window survived" (a K-dependent quantity) and NOT the
    #             per-token acceptance rate alpha from the literature.
    # accepted / evaluated is alpha. accepted / target_fwd is the mean accepted
    # run length per round, which is what determines the speedup.
    stats = {"drafted": 0, "evaluated": 0, "accepted": 0, "target_fwd": 0, "draft_fwd": 0}

    with torch.no_grad():
        while len(tokens) - 1 < max_len:
            # Step 1: Draft model generates K candidate tokens
            draft_tokens = list(tokens)
            draft_probs_list = []  # (token_id, q(x), full distribution) for each drafted token
            for _ in range(K):
                if len(draft_tokens) - 1 >= max_len:
                    break
                probs_d = get_probs(draft, draft_tokens)[-1]
                stats["draft_fwd"] += 1
                token_id = torch.multinomial(probs_d, 1).item()
                draft_probs_list.append((token_id, probs_d[token_id].item(), probs_d))
                draft_tokens.append(token_id)

            if not draft_probs_list:
                break

            stats["drafted"] += len(draft_probs_list)

            # Step 2: Target model scores ALL positions in one forward pass
            # This is the key efficiency win: one forward pass instead of K
            target_probs = get_probs(target, draft_tokens)  # (T, V)
            stats["target_fwd"] += 1

            # Step 3: Accept/reject each drafted token
            n_accepted = 0
            # Snapshot the prefix length BEFORE the loop: `tokens` grows as we
            # accept, so reading len(tokens) inside the loop would drift by +i
            # and verify token i against the wrong target row.
            base_len = len(tokens)
            for i, (drafted_token, q_x, draft_dist) in enumerate(draft_probs_list):
                # Target model's probability for the drafted token
                # Position in target_probs: tokens before draft + i
                pos = base_len - 1 + i
                if pos >= target_probs.shape[0]:
                    break
                p_x = target_probs[pos, drafted_token].item()

                # Accept with probability min(1, p(x) / q(x))
                if q_x == 0:
                    break
                # From here on a real accept/reject decision happens, so this
                # token counts towards the denominator of alpha.
                stats["evaluated"] += 1
                accept_prob = min(1.0, p_x / q_x)

                if random.random() < accept_prob:
                    # Accept this token
                    tokens.append(drafted_token)
                    n_accepted += 1
                    if drafted_token == BOS:
                        tokens.pop()  # remove BOS, end generation
                        # The BOS was accepted by the verification step even
                        # though it is not emitted, so it counts in `accepted`.
                        # `accepted` measures verification outcomes here, not
                        # emitted tokens.
                        stats["accepted"] += n_accepted
                        return tokens[1:], stats
                else:
                    # Reject: sample from adjusted distribution max(0, p - q)
                    pos_probs = target_probs[pos]
                    adjusted = torch.clamp(pos_probs - draft_dist, min=0)
                    adj_sum = adjusted.sum()
                    if adj_sum > 0:
                        adjusted = adjusted / adj_sum
                    else:
                        adjusted = pos_probs
                    token_id = torch.multinomial(adjusted, 1).item()
                    if token_id == BOS:
                        stats["accepted"] += n_accepted
                        return tokens[1:], stats
                    tokens.append(token_id)
                    break  # restart drafting from the new position

            else:
                # All K tokens accepted — sample one bonus token from target
                bonus_pos = len(tokens) - 1
                if bonus_pos < target_probs.shape[0]:
                    bonus_probs = target_probs[bonus_pos]
                    token_id = torch.multinomial(bonus_probs, 1).item()
                    if token_id == BOS:
                        # BOS from the target means end of sequence, exactly as in
                        # the accept and correction paths above. Returning here
                        # matters for losslessness: falling through instead would
                        # silently drop the target's own stop decision and redraft
                        # from the same prefix, biasing generated names longer.
                        stats["accepted"] += n_accepted
                        return tokens[1:], stats
                    tokens.append(token_id)

            stats["accepted"] += n_accepted

    return tokens[1:], stats


# ---------------------------------------------------------------------------
# Compare the three approaches
# ---------------------------------------------------------------------------
print("\n--- speculative decoding comparison ---\n")

draft_model.eval()
target_model.eval()
n_samples = 50


def timed_generate(name, gen_fn):
    print(f"generating {n_samples} samples with {name}...")
    # perf_counter, not time(): monotonic and high resolution. This lab exists to
    # report a timing measurement, so it should not be reading a wall clock that
    # NTP is allowed to step underneath it.
    start = time.perf_counter()
    samples = [gen_fn() for _ in range(n_samples)]
    elapsed = time.perf_counter() - start
    print(f"  time: {elapsed:.3f}s ({elapsed / n_samples * 1000:.1f} ms/sample)")
    return samples, elapsed


def time_forward(model, reps=300, seq_len=8):
    """Mean wall-clock time of one forward pass, in milliseconds.

    The draft/target cost ratio is the whole premise of speculative decoding, so
    measure it instead of guessing it from parameter counts.
    """
    idx = torch.randint(0, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        model(idx)  # warm up: first call pays for lazy allocation
        start = time.perf_counter()
        for _ in range(reps):
            model(idx)
        return (time.perf_counter() - start) / reps * 1000


auto_samples, auto_time = timed_generate(
    "autoregressive decoding (target)", lambda: generate_autoregressive(target_model)
)

# Speculative: also collect accept/reject stats. timed_generate only wants the
# token list back, so stash the stats on the side. (This used to be a lambda
# abusing the fact that list.append returns None; a named function says what it
# does.)
total_stats = {"drafted": 0, "evaluated": 0, "accepted": 0, "target_fwd": 0, "draft_fwd": 0}
spec_results = []


def speculative_sample():
    tokens, stats = generate_speculative(draft_model, target_model, K=4)
    spec_results.append((tokens, stats))
    return tokens


spec_samples, spec_time = timed_generate("speculative decoding (K=4)", speculative_sample)
for _, stats in spec_results:
    for k in total_stats:
        total_stats[k] += stats[k]

draft_samples, draft_time = timed_generate(
    "autoregressive decoding (draft)", lambda: generate_autoregressive(draft_model)
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
alpha = total_stats["accepted"] / max(total_stats["evaluated"], 1)
window_use = total_stats["accepted"] / max(total_stats["drafted"], 1)
accepted_per_round = total_stats["accepted"] / max(total_stats["target_fwd"], 1)
avg_target_fwd = total_stats["target_fwd"] / n_samples
avg_draft_fwd = total_stats["draft_fwd"] / n_samples
avg_auto_tokens = sum(len(s) for s in auto_samples) / n_samples
avg_spec_tokens = sum(len(s) for s in spec_samples) / n_samples

draft_ms = time_forward(draft_model)
target_ms = time_forward(target_model)

print("\n--- results ---\n")

# Lead with the forward-pass count. That is the metric that transfers: on a GPU
# with a real model the target forward dominates everything, so cutting target
# forwards from 5.0 to 2.3 per name IS the win. Wall clock on a CPU at this
# scale measures Python overhead, not the algorithm.
print(f"target forward passes per name: {avg_target_fwd:.1f} (speculative) vs {avg_auto_tokens:.1f} (autoregressive)")
print(f"  -> {avg_auto_tokens / avg_target_fwd:.2f}x fewer target forward passes")
print(f"draft forward passes per name:  {avg_draft_fwd:.1f} (the extra work that buys the reduction)")
print()
print("acceptance, three ways (they are not the same number):")
print(f"  per-token acceptance rate alpha: {alpha:.1%}  (accepted / evaluated)")
print(f"  mean accepted run per round:     {accepted_per_round:.2f} tokens of K=4")
print(f"  draft window utilisation:        {window_use:.1%}  (accepted / all K proposals)")
print("  alpha is the literature's metric and is K-independent. Window utilisation")
print("  falls as K grows simply because more of the window goes unexamined after")
print("  the first rejection — it is not a property of the draft model alone.")
print()
print(f"measured cost per forward pass: draft {draft_ms:.3f} ms, target {target_ms:.3f} ms")
print(f"  -> target is {target_ms / draft_ms:.2f}x the draft's cost, against a {102784 / 14528:.1f}x parameter ratio.")
print("  At this size the two do not track each other: per-call Python and dispatch")
print("  overhead dominates the arithmetic, which is exactly why the wall-clock")
print("  ratio below is not the number to take away from this lab.")
print()
print(f"wall clock: {auto_time / spec_time:.2f}x ratio (autoregressive time / speculative time)")
print("  Below 1.00 means speculative decoding is SLOWER here. It is: three Python")
print("  loop iterations of draft work plus one target forward costs more than one")
print("  target forward, when a target forward is this cheap.")
print(f"avg tokens generated: {avg_auto_tokens:.1f} (auto) vs {avg_spec_tokens:.1f} (spec)")

# ---------------------------------------------------------------------------
# Distributional check — is it actually lossless?
# ---------------------------------------------------------------------------
# The headline claim of this lab is that speculative decoding samples from the
# TARGET model's distribution exactly. A speedup number does not test that at
# all, so test it: draw a large sample autoregressively, a large sample
# speculatively, and compare summary statistics.
#
# The essential part is the noise floor. Two independent autoregressive runs also
# differ from each other, because each is a finite sample from the same
# distribution. Without that reference number, "the distributions match" is
# unfalsifiable — any small difference can be waved away. So run AR twice.
print("\n--- distributional check: is it really lossless? ---\n")
n_dist = 500


def char_unigram(samples):
    """Character frequency distribution over a list of token-id lists."""
    counts = torch.zeros(vocab_size)
    for sample in samples:
        for t in sample:
            counts[t] += 1
    return counts / counts.sum().clamp(min=1)


def tv_distance(p, q):
    """Total variation distance. 0 = identical, 1 = disjoint support."""
    return 0.5 * (p - q).abs().sum().item()


ar_a = [generate_autoregressive(target_model) for _ in range(n_dist)]
ar_b = [generate_autoregressive(target_model) for _ in range(n_dist)]
spec_dist = [generate_speculative(draft_model, target_model, K=4)[0] for _ in range(n_dist)]

print(f"  {'run':<22s} {'n':>5s}  {'mean length':>11s}")
for label, samples in [
    ("autoregressive A", ar_a),
    ("autoregressive B", ar_b),
    ("speculative (K=4)", spec_dist),
]:
    print(f"  {label:<22s} {len(samples):>5d}  {sum(len(s) for s in samples) / len(samples):>11.3f}")

u_a, u_b, u_spec = char_unigram(ar_a), char_unigram(ar_b), char_unigram(spec_dist)
floor = tv_distance(u_a, u_b)
measured = tv_distance(u_a, u_spec)
print()
print(f"  character-unigram TV distance, AR vs AR:          {floor:.4f}   <- noise floor")
print(f"  character-unigram TV distance, AR vs speculative:  {measured:.4f}")
print(f"  ratio to noise floor: {measured / max(floor, 1e-9):.2f}x")
print()
if measured <= 2.0 * floor:
    print("  Within twice the noise floor, so at this sample size the two are")
    print("  indistinguishable on this statistic. That is what 'lossless' looks like")
    print("  numerically: not zero, just no bigger than sampling error.")
else:
    print("  LARGER than twice the noise floor. On this statistic the speculative")
    print("  output is measurably different from target-only sampling, which the")
    print("  accept/reject proof says should not happen. Printed rather than hidden:")
    print("  a check you only trust when it passes is not a check.")
print()
print("  Caveats worth stating. This compares two summary statistics (mean length")
print("  and a 27-bin character histogram), not the full joint distribution over")
print("  strings, and it does so at n=500. A test like this can fail to detect a")
print("  real difference; it cannot prove there is none. It is a sanity check on")
print("  the implementation, and the proof is in Leviathan et al. (2023).")

print(f"""
--- why this matters ---

Autoregressive decoding is memory-bound: each token requires reading ALL model
weights from memory, but only computes a single vector-matrix product per layer.
The GPU's compute units sit mostly idle, waiting for data from slow memory.

Speculative decoding exploits this: the target model can verify K tokens in
nearly the same time as generating 1, because the bottleneck is reading weights
(which happens once regardless of sequence length), not computing attention.

In production (7B-70B models on GPUs), speculative decoding achieves:
  - 2-3x speedup with a well-matched draft model
  - Higher acceptance rates with better draft models (EAGLE, Medusa)
  - Zero quality loss — the output distribution is mathematically identical

None of that happens here, and the reason is in the numbers above rather than in
a hand-wave. The whole technique is an arbitrage on the draft/target cost ratio,
and at this scale that ratio has collapsed: the target forward costs only
{target_ms / draft_ms:.1f}x the draft forward, because both are dominated by per-call Python and
dispatch overhead rather than by arithmetic. Fewer target forwards therefore
buys almost nothing, and the extra draft work more than eats it. In production
the same ratio is 10-100x, which is where the 2-3x comes from.

So take the forward-pass count away from this lab, not the wall clock. The
forward-pass reduction is a property of the algorithm and transfers directly.
The wall clock is a property of this CPU and this interpreter and does not.
""")

# ---------------------------------------------------------------------------
# Show sample outputs
# ---------------------------------------------------------------------------
print("--- sample outputs ---\n")
for label, samples in [
    ("autoregressive (target)", auto_samples),
    ("speculative decoding", spec_samples),
    ("draft model only", draft_samples),
]:
    print(f"{label}:")
    for i, s in enumerate(samples[:10]):
        print(f"  {i + 1:2d}: {''.join(uchars[t] for t in s)}")
    print()
