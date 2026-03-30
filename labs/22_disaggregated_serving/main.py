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


def timed(fn, n_runs=50):
    for _ in range(5):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        fn()
    return (time.perf_counter() - t0) / n_runs


def measure_profile(plen):
    idx = torch.tensor([[BOS] + [random.randint(0, vocab_size - 2) for _ in range(plen - 1)]])
    n_dec = min(5, block_size - plen)
    t_pre = timed(lambda: model.prefill(idx))
    _, init_caches = model.prefill(idx)

    def _decode():
        past, pos = init_caches, plen
        for _ in range(n_dec):
            _, past = model.decode_step(BOS, past, pos)
            pos += 1

    t_dec = timed(_decode) if n_dec > 0 else 0.0
    c = 2 * n_head * head_dim * n_layer
    return t_pre, t_dec, c * plen**2, sum(c * (plen + s) for s in range(n_dec))


print(
    f"\n{'Prompt':>8s} | {'Prefill (ms)':>12s} | {'Decode 5tok':>14s} | {'Prefill FLOPs':>13s} | {'Decode FLOPs':>12s}"
)
print("-" * 72)
for prompt_len in [2, 4, 8, 12]:
    tp, td, fp, fd = measure_profile(prompt_len)
    print(f"  {prompt_len:6d} | {tp * 1000:10.3f}  | {td * 1000:12.3f}  | {fp:13,} | {fd:12,}")

print("""
Prefill processes T tokens in ONE pass (compute-bound, high parallelism).
Decode processes 1 token per step (memory-bound, reads all weights each time).
On real GPUs, prefill saturates compute; decode wastes it waiting on memory.
""")

# Part 2: Simulate colocated vs disaggregated serving
print("=" * 70)
print("PART 2: COLOCATED vs DISAGGREGATED SERVING")
print("=" * 70)

temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
PREFILL_COST_MS = 0.5  # ms per prompt token (simulated)
DECODE_COST_MS = 0.3  # ms per decode step (simulated)


@dataclass
class Request:
    id: int
    prompt: list
    arrival_time: float
    max_decode_steps: int = 10
    first_token_time: float = 0.0
    finish_time: float = 0.0
    generated_tokens: list = field(default_factory=list)


def make_workload(n_requests=12, seed=999):
    rng = random.Random(seed)
    reqs = []
    for i in range(n_requests):
        plen = rng.randint(8, 12) if rng.random() < 0.3 else rng.randint(2, 3)
        prompt = [BOS] + [rng.randint(0, vocab_size - 2) for _ in range(plen - 1)]
        reqs.append(Request(id=i, prompt=prompt, arrival_time=i * 0.3e-3, max_decode_steps=block_size - plen))
    return reqs


def sim_prefill(prompt_tokens):
    logits, caches = model.prefill(torch.tensor([prompt_tokens]))
    time.sleep(len(prompt_tokens) * PREFILL_COST_MS / 1000)
    return logits, caches


def sim_decode(token_id, caches, pos):
    logits, caches = model.decode_step(token_id, caches, pos)
    time.sleep(DECODE_COST_MS / 1000)
    return logits, caches


def sample_token(logits):
    return torch.multinomial(F.softmax(logits[0, -1] / temperature, dim=-1), 1).item()


# Strategy 1: Colocated — one worker does both prefill and decode (FIFO)
def serve_colocated(requests):
    results, clock = [], time.perf_counter()
    for req in requests:
        dt = clock + req.arrival_time - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        logits, caches = sim_prefill(req.prompt)
        req.first_token_time = time.perf_counter() - clock
        tok, pos, gen = sample_token(logits), len(req.prompt), []
        for _ in range(req.max_decode_steps):
            if tok == BOS or pos >= block_size:
                break
            gen.append(tok)
            logits, caches = sim_decode(tok, caches, pos)
            tok, pos = sample_token(logits), pos + 1
        req.generated_tokens, req.finish_time = gen, time.perf_counter() - clock
        results.append(req)
    return results


# Strategy 2: Disaggregated — separate prefill and decode workers
def serve_disaggregated(requests):
    decode_queue, lock = deque(), threading.Lock()
    results, results_lock = [], threading.Lock()
    prefill_done, clock = threading.Event(), time.perf_counter()

    def prefill_worker():
        for req in requests:
            dt = clock + req.arrival_time - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            logits, caches = sim_prefill(req.prompt)
            req.first_token_time = time.perf_counter() - clock
            with lock:
                decode_queue.append((req, caches, sample_token(logits), len(req.prompt)))
        prefill_done.set()

    def decode_worker():
        active = []
        while True:
            with lock:
                while decode_queue:
                    req, caches, tok, pos = decode_queue.popleft()
                    active.append((req, caches, tok, pos, []))
            if not active:
                if prefill_done.is_set():
                    break
                time.sleep(0.1e-3)
                continue
            still_active = []
            for req, caches, tok, pos, gen in active:
                if tok == BOS or len(gen) >= req.max_decode_steps or pos >= block_size:
                    req.generated_tokens, req.finish_time = gen, time.perf_counter() - clock
                    with results_lock:
                        results.append(req)
                else:
                    gen.append(tok)
                    logits, caches = sim_decode(tok, caches, pos)
                    still_active.append((req, caches, sample_token(logits), pos + 1, gen))
            active = still_active

    threads = [threading.Thread(target=f) for f in (prefill_worker, decode_worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results, key=lambda r: r.id)


# Run both strategies
print("\nWorkload: 12 requests (mix of short and long prompts)\n")
workload = make_workload()
for req in workload:
    print(f"  req {req.id:2d}: {'LONG ' if len(req.prompt) > 5 else 'short'} prompt ({len(req.prompt):2d} tokens)")

print("\n--- Colocated (single worker) ---")
torch.manual_seed(2000)
results_coloc = serve_colocated(make_workload())

print("\n--- Disaggregated (prefill + decode workers) ---")
torch.manual_seed(2000)
results_disagg = serve_disaggregated(make_workload())

# Part 3: Compare results
print("\n" + "=" * 70)
print("PART 3: RESULTS")
print("=" * 70)

print(
    f"\n{'Req':>4s} {'Prompt':>6s}  {'TTFT coloc':>10s} {'TTFT disagg':>11s} {'Speedup':>8s}  {'Name coloc':>12s}  Name disagg"
)
print("-" * 80)
for rc, rd in zip(results_coloc, results_disagg):
    plen = len(rc.prompt)
    speedup = rc.first_token_time / rd.first_token_time if rd.first_token_time > 0 else 0
    name_c = "".join(uchars[t] for t in rc.generated_tokens) or "(empty)"
    name_d = "".join(uchars[t] for t in rd.generated_tokens) or "(empty)"
    tag = " LONG" if plen > 5 else ""
    print(
        f"  {rc.id:2d}  {plen:2d}tok{tag:>5s}"
        f" {rc.first_token_time * 1000:6.1f} ms"
        f" {rd.first_token_time * 1000:8.1f} ms"
        f"  {speedup:5.1f}x"
        f"  {name_c:>12s}  {name_d}"
    )

avg_c = sum(r.first_token_time for r in results_coloc) / len(results_coloc)
avg_d = sum(r.first_token_time for r in results_disagg) / len(results_disagg)
print(f"\n  Avg TTFT:  colocated {avg_c * 1000:.1f} ms  |  disaggregated {avg_d * 1000:.1f} ms  ({avg_c / avg_d:.1f}x)")
t_c = max(r.finish_time for r in results_coloc)
t_d = max(r.finish_time for r in results_disagg)
print(f"  Total:     colocated {t_c * 1000:.1f} ms  |  disaggregated {t_d * 1000:.1f} ms")

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
8K context, ~1 GB transfers over NVLink (900 GB/s) in ~1.1 ms — negligible
vs. the prefill time it unblocks.
""")
