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

This means you can use different GPU types: compute-dense GPUs for prefill, bandwidth-optimized GPUs for decode. Or even different numbers, since one prefill GPU can feed several decode GPUs.

### What this lab measures

1. **Compute profiles**: Wall-clock time and FLOPs for prefill vs decode at different prompt lengths, showing the different scaling behavior
2. **Head-of-line blocking**: With colocated serving, later requests wait behind earlier ones' full prefill+decode cycles
3. **TTFT improvement**: Time-to-first-token drops significantly with disaggregation because decode requests start immediately after their prefill completes
4. **Throughput**: Total time to serve all requests is lower when the phases don't block each other

### The KV cache transfer

The key engineering challenge is transferring KV caches from prefill to decode workers. At scale:

- Llama 3 70B, 8K context: ~1 GB KV cache per request
- NVLink bandwidth: 900 GB/s, so ~1.1 ms transfer time
- This is negligible compared to prefill time (50-500ms)

In this lab we simulate the transfer with a thread-safe queue.

## What you learn here

- Why prefill and decode have different compute profiles (compute-bound vs memory-bound)
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
1. Measures prefill vs decode compute profiles at different prompt lengths
2. Serves 12 mixed requests with colocated serving (single worker)
3. Serves the same requests with disaggregated serving (prefill + decode workers)
4. Compares TTFT, completion time, and generated output

## Why serving architecture matters

The fastest model in the world is useless if the serving system can't deliver tokens at consistent latency. Disaggregated serving is the architectural insight that each phase of inference deserves its own resource allocation, the same principle behind every microservice decomposition. Recognize that your workload has distinct phases with different resource profiles, and separate them.
