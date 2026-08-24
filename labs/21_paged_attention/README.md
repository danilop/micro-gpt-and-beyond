# Understanding LLMs by Building One: Paged KV Cache (PagedAttention)

Same architecture as the pure-Python version (01), but with two KV cache implementations for inference: contiguous (wasteful pre-allocation) and paged (on-demand block allocation). PagedAttention is the core innovation in vLLM. It applies the operating system's virtual memory paging concept to KV caches, replacing unbounded over-allocation with waste bounded by one partially-filled block per sequence.

Zero dependencies. Pure Python. The model is scaffolding borrowed from lab 01, needed only to produce weights. The paging itself is plain lists and dicts.

## Why this version exists

When serving LLMs to many users simultaneously, the KV cache, not model weights, becomes the memory bottleneck. Each active request needs its own KV cache, and the naive approach (pre-allocate for maximum sequence length) wastes most of the GPU memory. PagedAttention solves this with an idea borrowed from operating systems: virtual memory paging.

## What makes it interesting

### The KV cache memory problem

During autoregressive generation, the model caches the key and value vectors from all previous tokens to avoid recomputing them. For each new token, only the new K/V pair is computed, and attention uses the full cached history.

For a model like Llama 3 70B serving 100 concurrent requests:
- KV cache per token: 80 layers × 8 KV heads × 128 head dim × 2 (K+V) × 2 bytes (fp16) = **320 KiB**
- Max context length: 8,192 tokens
- **Per request (contiguous)**: 8,192 × 320 KiB = **2.5 GiB**
- **100 requests**: **250 GiB**, far beyond any single GPU

But most requests use far less than max context. If the average length is 500 tokens, 94% of that pre-allocated memory is never written, and paging the same workload needs 100 × 500 × 320 KiB = **~15 GiB**. `main.py` prints the same arithmetic at the end of the prefix-sharing section, so the two agree.

### Virtual memory for KV caches

PagedAttention applies the same solution that operating systems use for process memory:

| OS Concept | KV Cache Equivalent |
|---|---|
| Virtual page | Logical block (e.g., 4 tokens of KV data) |
| Physical frame | Physical block in GPU memory pool |
| Page table | Block table: `(seq_id, layer, logical_block) → physical_block_id` |
| Demand paging | Allocate blocks only when tokens are generated |
| Free list | Pool of available physical blocks |
| Copy-on-write | Share blocks between sequences with common prefixes |

### Block table data structure

The block table maps logical positions to physical memory:

`main.py` prints the real thing. Two sequences allocating at the same time,
7 and 9 tokens, block size 4:

```
allocation events, in the order they happened:
  seq A, layer 0, logical block 0 -> physical block 7
  seq B, layer 0, logical block 0 -> physical block 6
  seq A, layer 0, logical block 1 -> physical block 5
  seq B, layer 0, logical block 1 -> physical block 4
  seq B, layer 0, logical block 2 -> physical block 3

block tables:
  seq 'A', layer 0: 7 tokens in 2 block(s)
    logical block 0 -> physical block  7  (tokens 0-3, refcount 1)
    logical block 1 -> physical block  5  (tokens 4-6, refcount 1, 1 unused slot(s))
  seq 'B', layer 0: 9 tokens in 3 block(s)
    logical block 0 -> physical block  6  (tokens 0-3, refcount 1)
    logical block 1 -> physical block  4  (tokens 4-7, refcount 1)
    logical block 2 -> physical block  3  (tokens 8-8, refcount 1, 3 unused slot(s))
```

A's logical blocks 0 and 1 live in physical blocks 7 and 5, with B's block
sitting between them. Physical blocks can be anywhere in memory and don't need
to be contiguous. The attention computation gathers K/V vectors by following
the block table (scattered reads).

Free A and its physical blocks go straight back on the free list, where the
next sequence picks them up regardless of how long that sequence is:

```
free 'A' (its physical blocks were [5, 7]): free list [0, 1, 2] -> [0, 1, 2, 5, 7]
allocate 'C' (5 tokens):
  seq C, layer 0, logical block 0 -> physical block 5
  seq C, layer 0, logical block 1 -> physical block 7
```

That is the whole fragmentation story: with contiguous allocation, a freed
region is only reusable by a request that fits it.

### Prefix sharing (copy-on-write)

When multiple requests share a common prefix (e.g., a system prompt), their block tables can point to the **same physical blocks** for the shared portion:

No data is copied; only block IDs are shared. When a sequence needs to modify a
shared block (diverges from the prefix), a new physical block is allocated and
the data is copied (copy-on-write). This is exactly how `fork()` works in Unix.

The lab makes that branch actually execute, and prints the refcounts on both
sides of it. A 3-token prefix shared by three requests, then `req_0` generates
one more token into the shared block:

```
physical block 63 is now referenced by 4 sequences:
  prefix->[63], req_0->[63], req_1->[63], req_2->[63]

before write:  req_0 layer 0 block table [63]
               refcount of physical block 63: 4
after write:   req_0 layer 0 block table [62]  <- cloned
               refcount of physical block 63: 3 (was 4)
               refcount of physical block 62: 1 (req_0's private copy)
               blocks allocated by the write: 1
req_1 and req_2 still share block 63: [63]
```

Only the writer allocated. Without that write, the copy-on-write branch in
`append()` never runs, which is exactly what used to happen: the mechanism was
implemented, advertised, and never exercised.

### Memory utilization comparison

The lab demonstrates the efficiency difference:
- **Contiguous**: allocates `max_seq_len` slots per sequence, wastes unused space
- **Paged**: allocates blocks on demand, only wastes space in the last partially-filled block

Measured in the run, with `BLOCK_SIZE_TOKENS = 4`:
- one 5-token sequence in 2 blocks: **62.5% utilization**, so 37.5% waste
- a 7-token and a 9-token sequence allocating at the same time: **80.0% utilization**, so 20.0% waste (4 of 20 slots)
- the same 5-token sequence with contiguous pre-allocation of 16 slots: **31% utilization**

"Near zero" would be wrong at this scale. Paging does not eliminate internal fragmentation, it *bounds* it: at most `block_size - 1` tokens per sequence per layer, no matter how long the sequence gets or how wrong your length guess was. With `block_size = 4` and 5-token names that bound is loose. With vLLM's default of 16 tokens per block and requests of hundreds of tokens, the same bound is a rounding error, which is where "near zero" comes from.

## What you learn here

- How operating system concepts (virtual memory, page tables, copy-on-write) transfer to ML inference
- Why memory fragmentation is the #1 bottleneck in LLM serving (not compute)
- How vLLM achieves 2-4× throughput improvement through better memory management alone
- The block table data structure and how scattered memory access enables efficient allocation
- Why prefix caching reduces time-to-first-token for repeated system prompts

## What's not covered (but exists in practice)

- **vLLM** (Kwon et al., SOSP 2023): The production system that introduced PagedAttention. Handles continuous batching, preemption, and distributed serving on top of paged KV caches.
- **Continuous batching**: Instead of waiting for all sequences in a batch to finish, immediately replace finished sequences with new ones. Iteration-level scheduling for maximum GPU utilization.
- **SGLang's RadixAttention**: Uses a radix tree (prefix tree) for KV cache reuse, enabling automatic prefix matching across requests. More flexible than vLLM's hash-based caching.
- **KV cache compression**: Quantize cached values to FP8 or FP4, reducing memory by 2-4× with minimal quality loss. Used in production by Anthropic and others.
- **KV cache eviction**: When memory is full, intelligently evict less-important tokens. H2O (Heavy Hitter Oracle) keeps tokens with high attention scores. StreamingLLM keeps the first few tokens plus a sliding window.
- **Disaggregated serving** (lab 22): Separate the prefill phase (compute-bound, processes the full prompt) from the decode phase (memory-bound, generates tokens). Different hardware is optimal for each.
- **GQA / MQA / MLA**: Architectural changes that reduce KV cache size at the model level. Grouped-Query Attention (Llama 3) uses fewer KV heads than query heads. Multi-Latent Attention (DeepSeek-V3) compresses the KV cache with learned projections.
- **Prefix caching at scale**: Anthropic reported ~90% cost reduction on repetitive workloads through prefix caching. OpenAI offers ~50% input token cost savings for cached prefixes.
- **Key papers**: Kwon et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023), Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs" (NeurIPS 2024).

## Run

```bash
python main.py
```

Trains for 350 steps (pure Python, about 20 seconds), then:
1. Compares contiguous and paged KV caches on the same sequence, asserting identical output
2. Prints the allocation events and the full block table for two interleaved sequences
3. Frees one sequence and shows its physical blocks being recycled by the next one
4. Shares a prefix across three requests, then diverges one of them so copy-on-write fires, printing refcounts before and after
5. Generates 20 sample names using the paged KV cache, freeing each sequence

## Why memory management matters

The most impactful LLM serving optimization isn't a faster attention kernel or a better quantization scheme. It's better memory management. vLLM's PagedAttention showed that applying a 50-year-old OS concept (virtual memory) to a new domain (KV caches) can double or triple serving throughput. The algorithms are simple data structures. The insight is knowing where to apply them.
