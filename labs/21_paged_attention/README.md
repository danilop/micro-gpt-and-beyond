# Understanding LLMs by Building One: Paged KV Cache (PagedAttention)

Same architecture as the pure-Python version (01), but with two KV cache implementations for inference: contiguous (wasteful pre-allocation) and paged (on-demand block allocation). PagedAttention is the core innovation in vLLM. It applies the operating system's virtual memory paging concept to KV caches, reducing memory waste from 60-80% to near zero.

Zero dependencies. Pure Python. The algorithms are pure data structures.

## Why this version exists

When serving LLMs to many users simultaneously, the KV cache, not model weights, becomes the memory bottleneck. Each active request needs its own KV cache, and the naive approach (pre-allocate for maximum sequence length) wastes most of the GPU memory. PagedAttention solves this with an idea borrowed from operating systems: virtual memory paging.

## What makes it interesting

### The KV cache memory problem

During autoregressive generation, the model caches the key and value vectors from all previous tokens to avoid recomputing them. For each new token, only the new K/V pair is computed, and attention uses the full cached history.

For a model like Llama 3 70B serving 100 concurrent requests:
- KV cache per token: ~1 MB (80 layers × 8 KV heads × 128 dim × 2 (K+V) × 2 bytes)
- Max context length: 8,192 tokens
- **Per request (contiguous)**: 8,192 × 1 MB = **8 GB**
- **100 requests**: 800 GB, far beyond any GPU

But most requests use far less than max context. Average length might be 500 tokens, meaning 94% of pre-allocated memory is wasted.

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

```
Sequence "req_0", Layer 0:
  Logical block 0 → Physical block 7   (tokens 0-3)
  Logical block 1 → Physical block 12  (tokens 4-7)

Sequence "req_1", Layer 0:
  Logical block 0 → Physical block 3   (tokens 0-3)
  Logical block 1 → Physical block 19  (tokens 4-7)
  Logical block 2 → Physical block 5   (tokens 8-11)
```

Physical blocks can be anywhere in memory and don't need to be contiguous. The attention computation gathers K/V vectors by following the block table (scattered reads).

### Prefix sharing (copy-on-write)

When multiple requests share a common prefix (e.g., a system prompt), their block tables can point to the **same physical blocks** for the shared portion:

```
"prefix" (system prompt):
  Block 0 → Physical 2, Block 1 → Physical 4

"req_0": [Physical 2, Physical 4, Physical 8]  ← shares prefix blocks
"req_1": [Physical 2, Physical 4, Physical 11] ← shares prefix blocks
"req_2": [Physical 2, Physical 4, Physical 15] ← shares prefix blocks
```

No data is copied; only block IDs are shared. When a sequence needs to modify a shared block (diverges from the prefix), a new physical block is allocated and the data is copied (copy-on-write). This is exactly how `fork()` works in Unix.

### Memory utilization comparison

The lab demonstrates the efficiency difference:
- **Contiguous**: allocates `max_seq_len` slots per sequence, wastes unused space
- **Paged**: allocates blocks on demand, only wastes space in the last partially-filled block

For short sequences (which are common in practice), paged allocation uses a fraction of the memory.

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

Trains for 1000 steps (pure Python, ~2 min), then demonstrates contiguous vs. paged KV cache with memory utilization comparison, multiple concurrent sequences, and prefix sharing. Generates 20 sample names using the paged KV cache.

## Why memory management matters

The most impactful LLM serving optimization isn't a faster attention kernel or a better quantization scheme. It's better memory management. vLLM's PagedAttention showed that applying a 50-year-old OS concept (virtual memory) to a new domain (KV caches) can double or triple serving throughput. The algorithms are simple data structures. The insight is knowing where to apply them.
