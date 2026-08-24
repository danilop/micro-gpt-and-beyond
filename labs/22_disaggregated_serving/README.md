# Understanding LLMs by Building One: Disaggregated Serving

Same model as lab 03/12, but demonstrating disaggregated inference: the prefill phase (process the prompt, compute-bound) and the decode phase (generate tokens, memory-bound) run on separate workers. This is how production systems like Splitwise, DistServe, and TetriInfer avoid head-of-line blocking and improve both throughput and latency.

## Why this version exists

LLM inference has two phases with fundamentally different hardware profiles. Prefill processes the entire prompt in parallel and is compute-bound, saturating GPU ALUs. Decode generates tokens one at a time and is memory-bandwidth-bound, with the GPU mostly idle waiting to read weights from memory. When both phases share the same GPU, a long prefill (e.g., a 32K-token document) blocks all decode requests, causing latency spikes for users waiting for their next token. Disaggregation fixes this by running each phase on separate, optimized hardware.

## What makes it interesting

### The interference problem

On a shared GPU, prefill and decode compete for the same resources:

```
Colocated (one GPU):
  [====PREFILL req0====][dec 0][dec 0][==PREFILL req1==][dec 1][dec 0][dec 1]...
                                      ^ req1 decode is blocked while req1 prefills
                                        AND req0 decode is also blocked!
```

A long prompt prefill can take 200-500ms. Every decode request on that GPU stalls for the duration. This causes unpredictable latency spikes, the "time between tokens" that users feel.

### The disaggregated solution

Separate the phases onto different workers:

```
Prefill worker:  [====PREFILL req0====][==PREFILL req1==][PREFILL req2]...
                           |                    |              |
                     (KV cache transfer)  (KV cache)     (KV cache)
                           v                    v              v
Decode worker:   [dec 0][dec 0][dec 0][dec 1][dec 0][dec 1][dec 2]...
```

The prefill worker processes prompts and ships the KV caches to the decode worker. The decode worker iterates over all active requests in round-robin, never blocked by a prefill.

### Why the phases have different hardware needs

| Phase | Bottleneck | Arithmetic Intensity | What matters |
|-------|-----------|---------------------|-------------|
| Prefill | Compute | High: T tokens processed in parallel, O(T^2) attention | More ALUs, FP8 support, tensor cores |
| Decode | Memory BW | Low: 1 token, reads all weights each step | Higher memory bandwidth (HBM3e), more memory capacity |

Be careful with "O(T^2)". Attention is quadratic, but the projections and the
MLP are linear and they dominate until T gets large. In this model attention is
**4% of prefill FLOPs at T=8** and **82% at T=1024**. The lab prints both
columns, total and attention-only, because a table that counts attention alone
understates prefill by about 16x at short prompts and reports the wrong scaling
exponent.

The per-token prefill cost is **not** a curve that just falls. It is a U. One
run prints 46.21, 7.26, 4.60 and 27.45 us/token at T=8, 64, 256 and 1024: the
minimum is at T=256, and by T=1024 the cost has climbed back to 27.45 us/token,
several times the minimum.
Reading only the endpoints ("46.2 down to 27.4") makes it look like a monotone
improvement, and it is not. The left arm falls because the fixed per-call
overhead (Python, dispatch, kernel launch) is amortised over more tokens. The
right arm rises because the quadratic attention term has taken over: attention
is 3.5% of prefill FLOPs at T=8 but 82.4% at T=1024, so past a few hundred
tokens every extra token costs more than the one before it. Those two effects
are what make the curve turn, and the turning point is where this model prefills
most efficiently, a number the endpoints cannot tell you.

This means you can use different GPU types: compute-dense GPUs for prefill, bandwidth-optimized GPUs for decode. Or even different numbers, since one prefill GPU can feed several decode GPUs.

### What this lab measures

1. **Compute profiles**: Wall-clock time and FLOPs for prefill vs decode at prompt lengths of 8, 64, 256 and 1024. Measured at 2-12 tokens the wall clock is flat (about 0.5 ms regardless of length) because it is all dispatch overhead; the scaling only appears at real prompt lengths, so that is where the lab measures it.
2. **Head-of-line blocking**: With colocated serving, later requests wait behind earlier ones' full prefill+decode cycles
3. **TTFT improvement**: Average time-to-first-token drops from about 206 ms to about 158 ms (1.3x) because decode requests start immediately after their prefill completes
4. **Throughput**: Total time to serve all requests is lower when the phases don't block each other
5. **The cost of the handoff**: the KV transfer is charged per prompt token, and the lab prints the per-token cost that would erase the TTFT gain entirely

### What the comparison is not

The colocated arm is **one** worker; the disaggregated arm is **two**. Part of
the gain is simply twice the hardware. The colocated arm also runs strict FIFO
with no continuous batching, which is the weakest reasonable baseline: a
production colocated server interleaves decode steps across requests and would
close much of the gap. The lab prints these caveats next to the speedup, and
both arms produce byte-identical text (each request samples from its own
`torch.Generator`), so the only thing that differs is latency.

### The KV cache transfer

The key engineering challenge is transferring KV caches from prefill to decode workers. At scale:

- Llama 3 70B, 8K context: ~1 GB KV cache per request
- NVLink bandwidth: 900 GB/s, so ~1.1 ms transfer time
- Small compared to prefill time (50-500 ms), but not free, and it is a cost colocated serving never pays

In this lab the handoff is a thread-safe queue carrying the same tensors, so it
would otherwise be free. `KV_TRANSFER_COST_MS` charges 0.5 ms per prompt token
inside the handoff, which lands in the disaggregated TTFT (23 ms total across
the 12 requests). Raise it and the advantage shrinks; the lab prints the value
that would erase it completely, **~2.41 ms/token**, against the 0.5 ms/token
it currently charges. That is a much narrower margin than the naive arithmetic
suggests, and the reason is the denominator. Dividing the 47.6 ms average TTFT
gain by the mean prompt length (3.8 tokens) would put the break-even in the
low tens of ms/token, but the handoff runs *inline on the prefill worker*, so
raising its per-token cost delays a request by every prompt token queued ahead
of it, not just its own. The right denominator is the mean cumulative prompt
length, the mean prompt-tokens-ahead-of-you of 24.9 tokens here, and dividing by
that much larger number is what brings the break-even down to 2.41 ms/token.

The simulated phase costs (5 ms per prompt token for prefill, 3 ms per decode
step, 0.5 ms per token for the transfer) are larger than they look like they
should be. That is deliberate: `time.sleep` overshoots by about a millisecond
on a loaded host, so a simulation built on 0.3 ms sleeps would be measuring OS
scheduling noise instead of its own cost model. Only the ratios matter.

## What you learn here

- Why prefill and decode have different compute profiles (compute-bound vs memory-bound), and at what prompt length that difference becomes measurable
- Why a two-worker result should never be compared against a one-worker baseline without saying so
- Why a simulation has to charge for the thing it is simulating (the KV transfer), or it will prove whatever you hoped
- Why a per-request cost that runs inline on a shared worker has to be divided by the work queued ahead of a request, not by the request's own size, before you can call anything a break-even
- Why a per-token cost curve needs its whole shape read, not its endpoints: this one is a U, and the minimum is the answer
- How head-of-line blocking degrades decode latency in colocated serving
- How disaggregation eliminates interference between phases
- The TTFT vs throughput tradeoffs in serving system design
- How a prefill→decode handoff works via KV cache transfer

## What's not covered (but exists in practice)

- **Splitwise** (Patel et al., ISCA 2024): First to demonstrate 1.4x throughput improvement by splitting phases across GPU pairs on the same node. Uses NVLink for KV cache transfer.
- **DistServe** (Zhong et al., OSDI 2024): Extends disaggregation across nodes with goodput-based scheduling. Achieves 1.5-2.3x goodput improvement.
- **TetriInfer** (Hu et al., 2024): Instead of separate GPUs, runs prefill and decode on different Streaming Multiprocessors (SMs) within the same GPU using CUDA MPS.
- **Mooncake** (Moonshot AI, 2024): KV cache-centric disaggregated architecture. Stores KV caches in a distributed pool, transfers via RDMA. Powers Moonshot's Kimi service.
- **Continuous batching**: Iteration-level scheduling where finished requests are immediately replaced. Orthogonal to disaggregation, and production systems combine both.
- **Chunked prefill** (Sarathi, Agrawal et al., OSDI 2024): Instead of full disaggregation, break long prefills into chunks interleaved with decode steps. Reduces TTFT without separate hardware.
- **KV cache compression**: Quantize caches to FP8/FP4 before transfer to reduce bandwidth requirements.
- **Key papers**: Patel et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (ISCA 2024), Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" (OSDI 2024).

## Run

```bash
uv run python main.py
```

Trains for 1000 steps, then:
1. Measures prefill vs decode compute profiles at prompt lengths of 8 to 1024, with FLOPs counted properly (projections and MLP included, attention broken out)
2. Serves 12 mixed requests with colocated serving (single worker, FIFO)
3. Serves the same requests with disaggregated serving (prefill + decode workers, with a KV transfer cost)
4. Compares TTFT and completion time, checks that the generated text is identical, and prints the caveats and the break-even transfer cost

## Why serving architecture matters

The fastest model in the world is useless if the serving system can't deliver tokens at consistent latency. Disaggregated serving is the architectural insight that each phase of inference deserves its own resource allocation, the same principle behind every microservice decomposition. Recognize that your workload has distinct phases with different resource profiles, and separate them.
