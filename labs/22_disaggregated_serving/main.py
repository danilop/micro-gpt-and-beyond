"""
microGPT — Disaggregated serving edition.

Same model as lab 03/12, but demonstrating disaggregated (split) inference:
the prefill phase (process the prompt, compute-bound) and the decode phase
(generate tokens one at a time, memory-bound) run on separate workers.

Production systems like Splitwise (Microsoft), DistServe, and TetriInfer
use this to avoid head-of-line blocking: a long prompt prefill on a shared
GPU stalls all the decode requests waiting behind it. Disaggregation lets
each phase run on hardware optimized for its bottleneck.

Based on "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
Large Language Model Serving" (Zhong et al., 2024),
https://arxiv.org/abs/2401.09670, and "Splitwise: Efficient generative LLM
inference using phase splitting" (Patel et al., 2024),
https://arxiv.org/abs/2311.18677. Also relevant: "TetriInfer: Disaggregated
LLM Inference on Heterogeneous GPUs" (2024). This lab simulates disaggregation
with threads and queues -- production systems transfer KV caches over
NVLink/RDMA between physical GPU workers.

What the simulation is honest about, because the numbers do not survive the
alternative:
  - It compares ONE colocated worker against TWO disaggregated workers, and
    the colocated arm is strict FIFO with no continuous batching. Some of the
    TTFT win is extra hardware and a weak baseline, not scheduling.
  - The KV handoff here is a tuple of the same tensors on a queue: no copy at
    all. So the handoff is charged an explicit KV_TRANSFER_COST_MS per prompt
    token, and the break-even value is printed with the results.
  - Prefill scaling is measured at prompt lengths of 8 to 1024, not 2 to 12.
    At a dozen tokens the wall clock is dispatch overhead and shows no trend.
  - Each request samples from its own torch.Generator, so both strategies
    produce identical text and only latency differs. Sampling from the global
    RNG across two threads made the disaggregated output change every run.
"""

import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field

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
# Model
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

    def forward(self, x, kv_cache=None):
        B, T_new, C = x.shape
        q = self.wq(x).view(B, T_new, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T_new, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T_new, n_head, head_dim).transpose(1, 2)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        T_total = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(T_new, T_total, device=x.device, dtype=torch.bool), diagonal=T_total - T_new + 1)
        att = F.softmax(att.masked_fill(mask, float("-inf")), dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T_new, C)
        return self.wo(out), (k, v)


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

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class MicroGPT(nn.Module):
    # max_pos is the size of the position table. The serving simulation only
    # ever needs block_size positions, but Part 1 profiles prefill at prompt
    # lengths far beyond that, so it builds a second instance with a wider
    # table. Everything else about the two models is identical.
    def __init__(self, max_pos=block_size):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_pos, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx, past_caches=None, start_pos=0):
        _, T = idx.shape
        x = self.norm_in(self.wte(idx) + self.wpe(torch.arange(start_pos, start_pos + T, device=idx.device)))
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, kv_cache=past_caches[i] if past_caches else None)
            new_caches.append(new_cache)
        return self.lm_head(x), new_caches

    def prefill(self, idx):
        return self.forward(idx)

    def decode_step(self, token_id, past_caches, pos):
        return self.forward(torch.tensor([[token_id]]), past_caches, pos)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
device = "cpu"
model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    input_ids = torch.tensor([tokens[:n]], device=device)
    targets = torch.tensor([tokens[1 : n + 1]], device=device)
    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    optimizer.zero_grad()
    loss.backward()
    for pg in optimizer.param_groups:
        pg["lr"] = 1e-2 * (1 - step / num_steps)
    optimizer.step()
    if (step + 1) % 200 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

model.eval()
torch.set_grad_enabled(False)  # inference only from here

# Part 1: Measure prefill vs decode profiles
print("\n" + "=" * 70)
print("PART 1: PREFILL vs DECODE — different compute profiles")
print("=" * 70)


# The serving model's position table only holds block_size = 16 positions, and
# at prompt lengths of 2-12 tokens the wall clock is pure Python and dispatch
# overhead: prefill takes about the same time at 2 tokens as at 12, which says
# nothing at all about the O(T^2) attention term. To see the scaling we need
# real prompt lengths, so we clone the trained weights into a model with a
# wider position table. The extra position rows are untrained, which does not
# matter here because we are measuring time, not quality.
PROFILE_MAX_T = 1152
N_DECODE = 5

profile_model = MicroGPT(max_pos=PROFILE_MAX_T)
profile_sd = dict(model.state_dict())
profile_sd["wpe.weight"] = profile_model.state_dict()["wpe.weight"]  # keep the wider table
profile_model.load_state_dict(profile_sd)
profile_model.eval()


def timed(fn, n_runs=50):
    """Median of n_runs timings. The median, not the mean: on a shared CPU a
    single scheduling hiccup can inflate a mean by several times over."""
    for _ in range(5):
        fn()
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def count_flops(plen, n_dec):
    """
    Multiply-accumulate counts for prefill and decode, attention broken out.

    Per token, per layer, the DENSE work is:
      4 projections (q, k, v, o):  4 * 2 * n_embd^2
      MLP (two matmuls, 4x hidden): 2 * 2 * 4 * n_embd^2
    plus the output head, 2 * n_embd * vocab_size per token. All of that is
    LINEAR in the number of tokens.

    Attention over T tokens is the only quadratic term: causal masking leaves
    T^2/2 (query, key) pairs, and each pair costs 2 * n_embd for the score plus
    2 * n_embd for the value-weighted sum, giving 2 * n_embd * T^2 per layer.

    Counting only attention (the older version of this table did) understates
    prefill by ~16x at T=12 AND reports the wrong scaling exponent, because at
    small T the dense terms dominate completely.
    """
    dense_per_token = n_layer * (8 * n_embd**2 + 16 * n_embd**2) + 2 * n_embd * vocab_size
    attn_prefill = n_layer * 2 * n_embd * plen**2
    prefill_total = dense_per_token * plen + attn_prefill
    # Each decode step attends to the whole context so far: 4 * n_embd per key.
    attn_decode = sum(n_layer * 4 * n_embd * (plen + s) for s in range(n_dec))
    decode_total = dense_per_token * n_dec + attn_decode
    return prefill_total, attn_prefill, decode_total


def measure_profile(plen):
    idx = torch.tensor([[BOS] + [random.randint(0, vocab_size - 2) for _ in range(plen - 1)]])
    t_pre = timed(lambda: profile_model.prefill(idx))
    _, init_caches = profile_model.prefill(idx)

    def _decode():
        past, pos = init_caches, plen
        for _ in range(N_DECODE):
            _, past = profile_model.decode_step(BOS, past, pos)
            pos += 1

    t_dec = timed(_decode)
    fp, fa, fd = count_flops(plen, N_DECODE)
    return t_pre, t_dec, fp, fa, fd


print(
    f"\n{'Prompt':>8s} | {'Prefill (ms)':>12s} | {'us/token':>9s} | {f'Decode {N_DECODE}tok':>13s}"
    f" | {'Prefill FLOPs':>14s} | {'of which attn':>13s} | {'Decode FLOPs':>12s}"
)
print("-" * 105)
prefill_times = {}
for prompt_len in [8, 64, 256, 1024]:
    tp, td, fp, fa, fd = measure_profile(prompt_len)
    prefill_times[prompt_len] = tp
    print(
        f"  {prompt_len:6d} | {tp * 1000:10.3f}  | {tp * 1e6 / prompt_len:9.2f} | {td * 1000:11.3f}  "
        f"| {fp:14,} | {fa:12,} ({fa / fp:.0%}) | {fd:12,}"
    )

# The per-token curve is a U, not a slide: fixed per-call overhead dominates at
# short prompts and is amortised away, then the quadratic attention term takes
# over. Reporting only the endpoints would make it look monotone, so find the
# actual minimum and print it.
per_token = {t: prefill_times[t] * 1e6 / t for t in prefill_times}
best_T = min(per_token, key=per_token.get)
time_growth = prefill_times[1024] / prefill_times[8]
flop_growth = count_flops(1024, N_DECODE)[0] / count_flops(8, N_DECODE)[0]
attn_share_8 = count_flops(8, N_DECODE)[1] / count_flops(8, N_DECODE)[0]
attn_share_1024 = count_flops(1024, N_DECODE)[1] / count_flops(1024, N_DECODE)[0]

print(f"""
Read the us/token column, not the total. From T=8 to T=1024 the prefill FLOP
count grows {flop_growth:.0f}x while the wall clock grows only {time_growth:.0f}x, and the per-token
cost traces a U: {per_token[8]:.1f} us/token at T=8, down to a minimum of
{per_token[best_T]:.1f} us/token at T={best_T}, then back up to {per_token[1024]:.1f} us/token at T=1024.
Quoting only the endpoints ({per_token[8]:.1f} -> {per_token[1024]:.1f}) would make that look monotone.
The left arm falls because at short prompts almost all of the time is fixed
per-call overhead — Python, dispatch, kernel launch — not arithmetic, and that
overhead is amortised over more tokens as T grows. The right arm rises because
the O(T^2) attention term overtakes the amortisation. That is why the earlier version of this table, measured at prompt
lengths of 2 to 12 tokens, came out flat and could not support any claim about
O(T^2) scaling. The effect is real; you have to measure it where it exists.

The attention share of prefill FLOPs grows from {attn_share_8:.1%} at T=8 to {attn_share_1024:.1%} at
T=1024. That is the quadratic term taking over, and it is the honest version of
"prefill is compute-bound": true at production context lengths, not at the
length of a name.

Prefill processes T tokens in ONE pass (compute-bound, high parallelism).
Decode processes 1 token per step (memory-bound, reads all weights each time).
On real GPUs, prefill saturates compute; decode wastes it waiting on memory.
At this toy scale neither statement is visible in the wall clock: the model has
{sum(p.numel() for p in model.parameters()):,} parameters and fits in L1 cache. The FLOP columns are arithmetic,
not measurements.
""")

# Part 2: Simulate colocated vs disaggregated serving
print("=" * 70)
print("PART 2: COLOCATED vs DISAGGREGATED SERVING")
print("=" * 70)

temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
# The simulated phase costs. They are deliberately in the milliseconds: on a
# loaded host time.sleep() overshoots by around a millisecond, so a simulation
# built on 0.3 ms sleeps measures OS scheduling noise instead of the cost model
# it wrote down. Only the RATIOS between these three numbers matter.
PREFILL_COST_MS = 5.0  # ms per prompt token (simulated)
DECODE_COST_MS = 3.0  # ms per decode step (simulated)
# The one cost disaggregation adds that colocated serving does not pay: the KV
# cache has to move from the prefill worker to the decode worker. In this lab
# the handoff is a tuple of the same tensors on a queue — no copy at all — so
# without an explicit charge the simulation would tell you the transfer is
# free, and then the closing text would tell you it costs 1.1 ms. Charge it.
# Sweep this value upward and disaggregation eventually stops paying; the
# break-even point is computed and printed below.
KV_TRANSFER_COST_MS = 0.5  # ms per prompt token of KV cache shipped to the decode worker


@dataclass
class Request:
    id: int
    prompt: list
    arrival_time: float
    max_decode_steps: int = 10
    first_token_time: float = 0.0
    finish_time: float = 0.0
    generated_tokens: list = field(default_factory=list)
    # Each request carries its OWN sampling RNG. The global torch RNG is not
    # usable here: the disaggregated arm samples from two threads, so the
    # interleaving decides who draws which number and the generated names
    # change from run to run. With a per-request generator, both strategies
    # draw the same numbers in the same order for the same request, so the
    # names are identical by construction and only latency differs.
    rng: torch.Generator = None


def make_workload(n_requests=12, seed=999):
    rng = random.Random(seed)
    reqs = []
    for i in range(n_requests):
        plen = rng.randint(8, 12) if rng.random() < 0.3 else rng.randint(2, 3)
        prompt = [BOS] + [rng.randint(0, vocab_size - 2) for _ in range(plen - 1)]
        reqs.append(
            Request(
                id=i,
                prompt=prompt,
                arrival_time=i * 0.3e-3,
                max_decode_steps=block_size - plen,
                rng=torch.Generator().manual_seed(2000 + i),
            )
        )
    return reqs


def sim_prefill(prompt_tokens):
    logits, caches = model.prefill(torch.tensor([prompt_tokens]))
    time.sleep(len(prompt_tokens) * PREFILL_COST_MS / 1000)
    return logits, caches


def sim_kv_transfer(prompt_tokens):
    """Model the prefill -> decode KV cache handoff. Cost scales with prompt length."""
    time.sleep(len(prompt_tokens) * KV_TRANSFER_COST_MS / 1000)
    return len(prompt_tokens) * KV_TRANSFER_COST_MS


def sim_decode(token_id, caches, pos):
    logits, caches = model.decode_step(token_id, caches, pos)
    time.sleep(DECODE_COST_MS / 1000)
    return logits, caches


def sample_token(logits, rng):
    return torch.multinomial(F.softmax(logits[0, -1] / temperature, dim=-1), 1, generator=rng).item()


# Strategy 1: Colocated — one worker does both prefill and decode (FIFO).
#
# Be clear about what this baseline is and is not. It is ONE worker running
# strict FIFO: a request's prefill and its entire decode finish before the next
# request is looked at. It has no continuous batching, so it is the weakest
# reasonable baseline, and part of the speedup measured below is simply the
# second worker the disaggregated arm gets. See the notes after the results.
def serve_colocated(requests):
    results, clock = [], time.perf_counter()
    for req in requests:
        dt = clock + req.arrival_time - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        logits, caches = sim_prefill(req.prompt)
        req.first_token_time = time.perf_counter() - clock
        tok, pos, gen = sample_token(logits, req.rng), len(req.prompt), []
        for _ in range(req.max_decode_steps):
            if tok == BOS or pos >= block_size:
                break
            gen.append(tok)
            logits, caches = sim_decode(tok, caches, pos)
            tok, pos = sample_token(logits, req.rng), pos + 1
        req.generated_tokens, req.finish_time = gen, time.perf_counter() - clock
        results.append(req)
    return results


# Strategy 2: Disaggregated — separate prefill and decode workers
def serve_disaggregated(requests):
    decode_queue, lock = deque(), threading.Lock()
    results, results_lock = [], threading.Lock()
    prefill_done, clock = threading.Event(), time.perf_counter()
    transfer_ms = []

    def prefill_worker():
        for req in requests:
            dt = clock + req.arrival_time - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            logits, caches = sim_prefill(req.prompt)
            # Ship the KV cache to the decode worker. In production this is an
            # NVLink or RDMA transfer; here it is a modelled delay. The first
            # token is only useful once the request can actually continue on
            # the decode worker, so the transfer is inside TTFT.
            transfer_ms.append(sim_kv_transfer(req.prompt))
            req.first_token_time = time.perf_counter() - clock
            with lock:
                decode_queue.append((req, caches, sample_token(logits, req.rng), len(req.prompt)))
        prefill_done.set()

    def decode_worker():
        # active maps request id -> that request's in-flight decode state.
        active = {}
        while True:
            # Sample the completion flag BEFORE draining the queue. If we read
            # it afterwards, prefill could append its last request and set the
            # flag in the window between the drain and the check, and we would
            # exit having silently dropped that request — which would then make
            # zip(results_coloc, results_disagg) misalign every pair after it.
            prefill_finished = prefill_done.is_set()
            with lock:
                while decode_queue:
                    req, caches, tok, pos = decode_queue.popleft()
                    active[req.id] = (req, caches, tok, pos, [])
                queue_empty = not decode_queue
            if not active:
                if prefill_finished and queue_empty:
                    break
                time.sleep(0.1e-3)
                continue
            still_active = {}
            for req, caches, tok, pos, gen in active.values():
                if tok == BOS or len(gen) >= req.max_decode_steps or pos >= block_size:
                    req.generated_tokens, req.finish_time = gen, time.perf_counter() - clock
                    with results_lock:
                        results.append(req)
                else:
                    gen.append(tok)
                    logits, caches = sim_decode(tok, caches, pos)
                    still_active[req.id] = (req, caches, sample_token(logits, req.rng), pos + 1, gen)
            active = still_active

    threads = [threading.Thread(target=f) for f in (prefill_worker, decode_worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results, key=lambda r: r.id), sum(transfer_ms)


# Run both strategies
print("\nWorkload: 12 requests (mix of short and long prompts)\n")
workload = make_workload()
for req in workload:
    print(f"  req {req.id:2d}: {'LONG ' if len(req.prompt) > 5 else 'short'} prompt ({len(req.prompt):2d} tokens)")

print("\n--- Colocated (single worker: prefill + decode, FIFO) ---")
results_coloc = serve_colocated(make_workload())

print("--- Disaggregated (two workers: prefill + decode, with KV transfer cost) ---")
results_disagg, total_transfer_ms = serve_disaggregated(make_workload())
assert len(results_disagg) == len(results_coloc), "a request was dropped in the disaggregated arm"

# Part 3: Compare results
print("\n" + "=" * 70)
print("PART 3: RESULTS")
print("=" * 70)

print(f"\n{'Req':>4s} {'Prompt':>6s}  {'TTFT coloc':>10s} {'TTFT disagg':>11s} {'Speedup':>8s}  Name")
print("-" * 70)
mismatches = 0
for rc, rd in zip(results_coloc, results_disagg):
    plen = len(rc.prompt)
    speedup = rc.first_token_time / rd.first_token_time if rd.first_token_time > 0 else 0
    name_c = "".join(uchars[t] for t in rc.generated_tokens) or "(empty)"
    name_d = "".join(uchars[t] for t in rd.generated_tokens) or "(empty)"
    mismatches += name_c != name_d
    tag = " LONG" if plen > 5 else ""
    print(
        f"  {rc.id:2d}  {plen:2d}tok{tag:>5s}"
        f" {rc.first_token_time * 1000:6.1f} ms"
        f" {rd.first_token_time * 1000:8.1f} ms"
        f"  {speedup:5.1f}x"
        f"  {name_c}{'' if name_c == name_d else f'  != {name_d}'}"
    )

# One Name column, not two. Each request samples from its own torch.Generator,
# so both strategies produce byte-identical output and there is nothing to
# compare there — which is the point: disaggregation is a scheduling change,
# not a change to what the model says. Two "Name" columns side by side would
# invite the reader to compare thread-scheduling noise.
print(f"\n  Output identical across strategies: {mismatches == 0} ({mismatches} mismatches)")
assert mismatches == 0, "per-request RNG should make the two strategies produce identical text"

avg_c = sum(r.first_token_time for r in results_coloc) / len(results_coloc)
avg_d = sum(r.first_token_time for r in results_disagg) / len(results_disagg)
print(f"  Avg TTFT:  colocated {avg_c * 1000:.1f} ms  |  disaggregated {avg_d * 1000:.1f} ms  ({avg_c / avg_d:.1f}x)")
t_c = max(r.finish_time for r in results_coloc)
t_d = max(r.finish_time for r in results_disagg)
print(f"  Total:     colocated {t_c * 1000:.1f} ms  |  disaggregated {t_d * 1000:.1f} ms")

# What did the KV handoff cost, and how much more could it cost before
# disaggregation stopped being worth it?
avg_plen = sum(len(r.prompt) for r in results_coloc) / len(results_coloc)
gain_ms = (avg_c - avg_d) * 1000
# The handoff runs inline on the prefill worker, so raising its per-token cost
# delays a request by every prompt token queued AHEAD of it, not just its own.
# The right denominator is therefore the mean cumulative prompt length, which is
# far larger than the mean prompt length -- and makes the break-even far closer.
_cum, _cums = 0, []
for _r in results_coloc:
    _cum += len(_r.prompt)
    _cums.append(_cum)
mean_cum_plen = sum(_cums) / len(_cums)
breakeven_per_token = KV_TRANSFER_COST_MS + gain_ms / mean_cum_plen
print(f"\n  KV transfer: {KV_TRANSFER_COST_MS} ms/token, {total_transfer_ms:.1f} ms total across 12 requests")
print("               already included in the disaggregated TTFT above")
print(f"  Break-even:  ~{breakeven_per_token:.2f} ms/token would erase the {gain_ms:.1f} ms average TTFT gain")
print(f"               (mean prompt {avg_plen:.1f} tokens, but mean prompt-tokens-ahead-of-you {mean_cum_plen:.1f};")
print("                that second number is the one that sets the break-even. Raise KV_TRANSFER_COST_MS and re-run.)")

# Honest accounting of what this comparison is.
print(f"""
  Caveats, so the {avg_c / avg_d:.1f}x is not read as more than it is:
    - 1 worker vs 2 workers. The disaggregated arm has twice the hardware, so
      part of the gain is capacity, not scheduling. A fair test would give the
      colocated arm two workers too.
    - The colocated arm is strict FIFO with no continuous batching, which is
      the weakest reasonable baseline. Production colocated serving interleaves
      decode steps of many requests and would close much of this gap.
    - Both arms run on one CPU with sleeps standing in for GPU time. What is
      being demonstrated is the scheduling structure, not a benchmark.""")

# Part 4: Why disaggregation matters
print("\n" + "=" * 70)
print("WHY DISAGGREGATED SERVING MATTERS")
print("=" * 70)
print("""
The two phases of LLM inference have different hardware profiles:
  Prefill: compute-bound (T tokens in parallel, O(T^2)). Wants more ALUs, FP8.
  Decode:  memory-bound (1 token/step, reads all weights). Wants high BW, HBM3e.

PROBLEM — colocated serving (one GPU does both):
  [====PREFILL req0====][dec 0][dec 0]...[==PREFILL req1==][dec 1]...
                                          ^ decode 1 blocked by prefill 1!

SOLUTION — disaggregate prefill and decode onto separate workers:
  Prefill GPU:  [==PREFILL 0==][==PREFILL 1==][PREFILL 2]
  Decode GPU:   [dec 0][dec 0][dec 1][dec 0]...  <- starts right after handoff

Benefits: lower TTFT, better throughput, reduced tail latency, cost-efficient
(fewer compute GPUs for prefill, more memory-BW GPUs for decode).

Production systems:
  - Splitwise (Microsoft, ISCA'24): 1.4x throughput via prefill/decode split
  - DistServe (OSDI'24): goodput-based placement across workers
  - TetriInfer (2024): prefill/decode on different SMs within one GPU
  - Mooncake (Moonshot AI, 2024): KV cache transfer over RDMA

The KV cache transfer is the key engineering challenge. For Llama 3 70B at
8K context, ~1 GB transfers over NVLink (900 GB/s) in ~1.1 ms — small next to
the prefill time it unblocks, but not free, and it is the cost colocated
serving never pays. This lab charges KV_TRANSFER_COST_MS per prompt token in
the handoff so the cost appears in the disaggregated TTFT instead of only in
this paragraph. Raise it until the advantage disappears; the break-even value
is printed with the results.

What this lab does NOT show:
  - a fair worker count (1 vs 2) or continuous batching in the baseline
  - real KV transfer over a real interconnect (it is a tuple on a queue)
  - GPU behaviour of any kind; the phase costs are sleeps
""")
