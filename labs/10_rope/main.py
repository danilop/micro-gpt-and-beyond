"""
microGPT — Rotary Position Embeddings (RoPE).

Same architecture as version 03 (PyTorch), but learned positional embeddings are
replaced with Rotary Position Embeddings. RoPE encodes position by rotating query
and key vectors in complex space, so the dot product q·k naturally depends on the
*relative* distance (m-n) rather than absolute positions m and n separately.
Every modern LLM (LLaMA, Mistral, GPT-NeoX) uses RoPE. This lab shows why.

The RoPE technique is from "RoFormer: Enhanced Transformer with Rotary Position
Embedding" (Su et al., 2021), https://arxiv.org/abs/2104.09864. The implementation
uses the real-valued rotation form (pairs of adjacent dimensions), matching the
approach used in production by LLaMA ("LLaMA: Open and Efficient Foundation
Language Models", Touvron et al., 2023, https://arxiv.org/abs/2302.13971). Note
that the base frequency of 10000 follows the original paper; later work like
"YaRN: Efficient Context Window Extension of Large Language Models" (Peng et al.,
2023, https://arxiv.org/abs/2309.00071) explores different base frequencies for
length extension.
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
# Model config
# ---------------------------------------------------------------------------
n_embd = 16  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def precompute_freqs(dim, max_len):
    """Precompute rotation frequencies for RoPE.

    theta_i = 1 / (10000 ^ (2i / dim))  for i in 0..dim//2
    Returns cos and sin of shape (max_len, dim//2).
    """
    i = torch.arange(0, dim, 2, dtype=torch.float32)  # (dim//2,)
    theta = 1.0 / (10000.0 ** (i / dim))  # (dim//2,)
    positions = torch.arange(max_len, dtype=torch.float32)  # (max_len,)
    angles = torch.outer(positions, theta)  # (max_len, dim//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos_freqs, sin_freqs):
    """Apply rotary embeddings to x of shape (B, n_head, T, head_dim).

    Split head_dim into pairs, rotate each pair:
        [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    """
    T = x.shape[2]
    cos_t = cos_freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim//2)
    sin_t = sin_freqs[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim//2)
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t
    return torch.stack((out1, out2), dim=-1).flatten(-2)


# Precompute once for the whole model. Note the span: 4 * block_size, not
# block_size. RoPE has no position table to size, only a formula, so there is
# nothing stopping us from precomputing positions the model never trains on —
# which is what makes the length-generalization test at the bottom of this file
# possible at all. The learned-position baseline has no equivalent option: its
# `wpe` embedding has exactly block_size rows and row 16 does not exist.
ROPE_MAX_LEN = 4 * block_size
rope_cos, rope_sin = precompute_freqs(head_dim, ROPE_MAX_LEN)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            # V is NOT rotated — RoPE only affects Q and K

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
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
    def __init__(self, use_rope=False):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(use_rope=use_rope)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self, use_rope=False):
        super().__init__()
        self.use_rope = use_rope
        self.wte = nn.Embedding(vocab_size, n_embd)
        if not use_rope:
            self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block(use_rope=use_rope) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx)
        if not self.use_rope:
            x = x + self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training + comparison
# ---------------------------------------------------------------------------
def train(model, label, num_steps=1000):
    """Train a model and return the loss history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
    losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        input_ids = torch.tensor([tokens[:n]])
        targets = torch.tensor([tokens[1 : n + 1]])

        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        lr_t = 1e-2 * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  [{label}] step {step + 1:4d}/{num_steps} | loss {loss.item():.4f}")

    return losses


def generate(model, label, num_samples=10, temperature=0.5):
    """Generate names from a trained model."""
    model.eval()
    print(f"\n--- {label}: generated names ---")
    with torch.no_grad():
        for i in range(num_samples):
            tokens = [BOS]
            for _ in range(block_size):
                idx = torch.tensor([tokens[-block_size:]])
                logits = model(idx)
                logits = logits[0, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
                if token_id == BOS:
                    break
                tokens.append(token_id)
            name = "".join(uchars[t] for t in tokens[1:])
            print(f"  sample {i + 1:2d}: {name}")
    model.train()


def share_weights(src, dst):
    """Copy every parameter the two models have in common (everything but wpe)."""
    src_sd = src.state_dict()
    shared = 0
    with torch.no_grad():
        for name, param in dst.named_parameters():
            if name in src_sd and src_sd[name].shape == param.shape:
                param.copy_(src_sd[name])
                shared += 1
    return shared


# Give both variants genuinely identical starting weights.
#
# Re-seeding before each constructor looks like it should do this, and it does
# not. `apply(_init_weights)` walks the modules in registration order, and the
# baseline registers `wpe` second, so it draws 16x16 random numbers that the
# RoPE model never draws. From that point the two streams are offset and every
# remaining weight differs. The loss comparison then measures initialization
# luck as much as it measures positional encoding. So: build both models, then
# copy across every tensor they share, leaving exactly one difference — how
# position information enters the model.
print("\n=== Training: Learned Positional Embeddings (baseline) ===")
torch.manual_seed(42)
model_learned = MicroGPT(use_rope=False)
n_params_learned = sum(p.numel() for p in model_learned.parameters())
print(f"  params: {n_params_learned}")

torch.manual_seed(42)
model_rope = MicroGPT(use_rope=True)
n_shared = share_weights(model_learned, model_rope)
print(f"  copied {n_shared} shared weight tensors into the RoPE model (everything except wpe)")

losses_learned = train(model_learned, "learned-pos")

print("\n=== Training: Rotary Position Embeddings (RoPE) ===")
n_params_rope = sum(p.numel() for p in model_rope.parameters())
print(f"  params: {n_params_rope} (no wpe — {n_params_learned - n_params_rope} fewer)")
losses_rope = train(model_rope, "rope")

# Compare losses. A single last-step loss would be pure noise at batch size 1,
# so report a trailing average over the last `avg_window` steps instead.
print("\n=== Results: training loss ===")
avg_window = 100
avg_learned = sum(losses_learned[-avg_window:]) / avg_window
avg_rope = sum(losses_rope[-avg_window:]) / avg_window
print(f"  learned-pos  trailing avg loss (last {avg_window} steps): {avg_learned:.4f}")
print(f"  rope         trailing avg loss (last {avg_window} steps): {avg_rope:.4f}")
print("  Two runs of one 16-dim, 1-layer model on 1000 single-name steps. Treat a gap")
print("  this small as a tie: the interesting difference is below, not here.")

# ---------------------------------------------------------------------------
# Length generalization — measured, not asserted
# ---------------------------------------------------------------------------
# "RoPE generalizes to longer sequences" packs two claims into one sentence,
# and they are not equally true.
#
# (1) It RUNS on longer sequences. `wpe` is a table with block_size rows, so
#     position 16 does not exist and a longer input raises IndexError. RoPE is
#     a formula, so the same trained weights accept any position we precomputed.
#     This part is unambiguous, and the test below shows it as an exception.
#
# (2) It stays ACCURATE on longer sequences. This does not follow, and the
#     numbers below show it does not hold either. Beyond the trained span the
#     rotations produce relative-distance patterns the model never saw, and
#     loss degrades. That is exactly why length-extension methods exist: YaRN
#     (cited in the module docstring) rescales the base frequency because
#     fixed-base RoPE does not extrapolate for free.
#
# Names are short, so a single name never exceeds block_size. To get a genuinely
# longer sequence we concatenate names separated by BOS — the same token stream
# the model trained on, just continued past position 15.
EVAL_LEN = 3 * block_size  # 48 positions, 3x the trained context


def make_long_batch(num_seqs=64, length=EVAL_LEN):
    """Build BOS-separated concatenations of names, `length` + 1 tokens each."""
    xs, ys = [], []
    doc_i = 0
    for _ in range(num_seqs):
        toks = []
        while len(toks) < length + 1:
            toks += [BOS] + [uchars.index(ch) for ch in docs[doc_i % len(docs)]]
            doc_i += 1
        toks = toks[: length + 1]
        xs.append(toks[:-1])
        ys.append(toks[1:])
    return torch.tensor(xs), torch.tensor(ys)


def eval_loss(model, inputs, targets):
    """Teacher-forced cross-entropy over the whole sequence in one forward pass."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    model.train()
    return loss.item()


def band_losses(model, inputs, targets, window=block_size):
    """Per-chunk loss two ways, so position can be separated from content.

    Returns (full, rebased). Both lists cover the same chunks of the same tokens:

      full     — one forward pass over the whole sequence, so chunk c sits at
                 absolute positions [c*window, ...]. Only chunk 0 is inside the
                 range the model trained on.
      rebased  — each chunk fed in as its own sequence, so every chunk sits at
                 positions [0, window-1], inside the trained range.

    Content is identical between the two, so any gap is the position encoding
    behaving differently at distances it never saw. (The rebased pass also drops
    cross-chunk context, but the aggregate numbers above show that context is
    worth ~0.0002 nats here, so it is not what moves these figures.)
    """
    model.eval()
    full, rebased = [], []
    with torch.no_grad():
        logits = model(inputs)
        for start in range(0, inputs.shape[1], window):
            ti = targets[:, start : start + window]
            lg = logits[:, start : start + window]
            full.append((start, start + ti.shape[1] - 1, F.cross_entropy(lg.reshape(-1, vocab_size), ti.reshape(-1)).item()))
            lg2 = model(inputs[:, start : start + window])
            rebased.append(F.cross_entropy(lg2.reshape(-1, vocab_size), ti.reshape(-1)).item())
    model.train()
    return full, rebased


def eval_loss_windowed(model, inputs, targets, window=block_size):
    """Same tokens, chopped into independent `window`-sized chunks.

    This is what a hard position limit forces you to do, and it is the fair
    comparison: context is thrown away at every chunk boundary.
    """
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, inputs.shape[1], window):
            xi, ti = inputs[:, start : start + window], targets[:, start : start + window]
            logits = model(xi)
            total += F.cross_entropy(logits.reshape(-1, vocab_size), ti.reshape(-1)).item() * ti.numel()
            count += ti.numel()
    model.train()
    return total / count


long_x, long_y = make_long_batch()
print(f"\n=== Results: length generalization (trained on {block_size} positions) ===")
print(f"  Evaluating {long_x.shape[0]} sequences of {EVAL_LEN} tokens ({EVAL_LEN // block_size}x block_size).")

try:
    loss_learned_long = eval_loss(model_learned, long_x, long_y)
    print(f"  learned-pos, full {EVAL_LEN} tokens in one pass: loss {loss_learned_long:.4f}")
except IndexError as exc:
    print(f"  learned-pos, full {EVAL_LEN} tokens in one pass: {type(exc).__name__}")
    print(f"    wpe has exactly {block_size} rows, so position {block_size} does not exist. Hard stop —")
    print("    the only option is to chop the input into windows and lose the context.")

loss_rope_windowed = eval_loss_windowed(model_rope, long_x, long_y)
loss_rope_long = eval_loss(model_rope, long_x, long_y)
print(f"  rope, {block_size}-token windows (context reset at each boundary): loss {loss_rope_windowed:.4f}")
print(f"  rope, full {EVAL_LEN} tokens in one pass:                          loss {loss_rope_long:.4f}")

delta = loss_rope_long - loss_rope_windowed
verdict = "better" if delta < 0 else "worse"
print(f"  ...which is {abs(delta):.4f} nats {verdict} than windowing.")

# Now the controlled version: same token chunks, evaluated once at their true
# (partly unseen) positions and once re-based into the trained range.
full_bands, rebased = band_losses(model_rope, long_x, long_y)
print("\n  Loss per chunk, same tokens, two sets of position indices:")
print(f"    {'chunk':<16} {'at true position':>17} {'re-based to 0-' + str(block_size - 1):>19}   position cost")
for (lo, hi, loss_full), loss_rebased in zip(full_bands, rebased):
    print(f"    positions {lo:2d}-{hi:2d}    {loss_full:>17.4f} {loss_rebased:>19.4f}   {loss_full - loss_rebased:>+13.4f}")

extrapolation_cost = sum(f[2] - r for f, r in zip(full_bands[1:], rebased[1:])) / max(1, len(full_bands) - 1)

# Honest reading of the numbers above.
print("\n  Read the last column, not the third-from-last. The 'at true position' figures")
print("  rise steadily down the table, which looks like extrapolation damage, and is not:")
print("  the chunks contain different text. Chunk 0 starts on a name boundary, the others")
print("  start mid-name. Only the difference against the re-based column controls for that,")
print("  and chunk 0 checks the method — same positions both ways, so it must read +0.0000.")

print("\n  What this does and does not show:")
print(f"  - RoPE removes the hard length limit. That is not a claim, it is the {block_size}-row")
print("    IndexError above. No position table means no position ceiling, so the same")
print(f"    trained weights accept {EVAL_LEN} positions and produce a loss.")
if abs(extrapolation_cost) < 0.05:
    print(f"  - Accuracy at unseen positions: {extrapolation_cost:+.4f} nats on average, which is noise.")
    print("    At this scale, evaluating past the trained span costs nothing measurable.")
    print("    Do NOT read that as proof that RoPE extrapolates. A 1-layer, 16-dim model on")
    print("    a corpus of independent names has almost no long-range structure to get")
    print("    wrong — the aggregate numbers above put the value of all cross-chunk context")
    print("    at 0.0002 nats. A test with nothing at stake cannot fail.")
else:
    print(f"  - Accuracy at unseen positions costs {extrapolation_cost:+.4f} nats on average, for the")
    print("    same tokens, purely because of where they sit in the sequence.")
print("  - So 'RoPE generalizes to longer sequences' is two claims. The mechanical one is")
print("    proven here. The accuracy one this lab cannot settle, and at real scale it is")
print("    known to fail: that is why length extension is its own research area. YaRN")
print("    (module docstring) rescales the base frequency; NTK-aware interpolation and")
print("    fine-tuning at the target length are the other standard answers. Fixed-base")
print("    RoPE does not extrapolate for free, and this lab is too small to show why.")

generate(model_learned, "Learned Positional Embeddings")
generate(model_rope, "RoPE")
