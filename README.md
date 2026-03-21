# microGPT and Beyond

A progressive series of labs exploring tiny language models, inspired by Andrej Karpathy's [microGPT](https://karpathy.ai/microgpt.html) — a GPT trained and run in a single file of pure Python. Each lab teaches something different about how neural networks are built, trained, and served. The models learn to generate human names from a dataset of ~32,000 real ones.

**[Interactive Web Tutorial](https://danilop.github.io/micro-gpt-and-beyond/)** — browse the code with line-by-line explanations.

The progression goes from raw first principles to framework-powered GPU code, modern architecture upgrades, inference optimization, and alternative paradigms:

```
01_pure_python/            Karpathy's original code. Zero dependencies. Scalar autograd.
02_numpy_manual_backprop/  NumPy arrays, but every gradient still written by hand.
03_pytorch/                PyTorch autograd takes over. Same model, ~30 lines shorter.
04_pytorch_batched/        Mini-batches, padding, masking — training at scale.
05_jax/                    Functional style. Pure functions, explicit state, JIT compilation.
06_jax_batched/            jax.vmap — write for one example, run for a batch automatically.
07_mlx/                    Apple Silicon GPU via MLX. Unified memory, lazy evaluation.
08_mlx_batched/            Batched MLX — same PyTorch-style padding, but on Apple GPU.
09_bpe_tokenizer/          Byte-Pair Encoding from scratch. The algorithm behind GPT's tokenizer.
10_rope/                   Rotary Position Embeddings. How modern LLMs encode position.
11_gqa/                    Grouped-Query Attention (MHA → GQA → MQA). KV head sharing.
12_kv_cache/               KV cache for inference. THE fundamental decoding optimization.
13_sampling/               Sampling strategies: greedy, temperature, top-k, top-p, min-p.
14_lora/                   LoRA — parameter-efficient fine-tuning with low-rank adapters.
15_text_diffusion/         Masked diffusion model (MDLM/LLaDA). Names emerge from noise.
16_pytorch_quantized/      INT8 quantization for inference. FP32 → INT8, 4× smaller, faster.
17_speculative_decoding/   Draft model guesses, target model verifies. Lossless speedup.
18_tiled_attention/        FlashAttention algorithm from scratch. Tiling beats the memory wall.
19_paged_attention/        PagedAttention (vLLM). OS-style virtual memory for KV caches.
20_soft_thinking/          Concept tokens preserve the full output distribution at inference.
21_soft_training/          Soft input curriculum closes the train-test gap for concept tokens.
22_disaggregated_serving/  Split prefill/decode onto separate workers. No more head-of-line blocking.
data/                      Shared dataset (auto-downloaded on first run if not present).
```

## The idea

Every version trains the same architecture — a character-level transformer with RMSNorm, ReLU, multi-head attention, and linear LR decay — on the same names dataset. What changes is *how* the computation is expressed:

| Version | Autograd | Tensors | Batching | Hardware |
|---------|----------|---------|----------|----------|
| 01 pure Python | hand-built scalar engine | no | no | CPU (slow) |
| 02 NumPy | none — manual backward pass | yes | no | CPU |
| 03 PyTorch | automatic | yes | no | CPU |
| 04 PyTorch batched | automatic | yes | yes (32) | CPU |
| 05 JAX | automatic (functional) | yes | no | CPU/GPU |
| 06 JAX batched | automatic (functional) | yes | yes (32, vmap) | CPU/GPU |
| 07 MLX | automatic | yes | no | Apple GPU |
| 08 MLX batched | automatic | yes | yes (32) | Apple GPU |
| 09 BPE tokenizer | — (bigram model) | no | no | CPU |
| 10 RoPE | automatic | yes | no | CPU |
| 11 GQA | automatic | yes | no | CPU |
| 12 KV cache | automatic | yes | no | CPU |
| 13 sampling | automatic | yes | no | CPU |
| 14 LoRA | automatic | yes | no | CPU |
| 15 text diffusion | hand-built scalar engine | no | no | CPU (slow) |
| 16 PyTorch quantized | automatic | yes | no | CPU (INT8) |
| 17 speculative decoding | automatic | yes | no | CPU |
| 18 tiled attention | automatic | yes | no | CPU |
| 19 paged attention | hand-built scalar engine | no | no | CPU (slow) |
| 20 soft thinking | automatic | yes | no | CPU |
| 21 soft training | automatic | yes | no | CPU |
| 22 disaggregated serving | automatic | yes | no | CPU |

Beyond the framework comparisons, the labs extend the base model with **modern architecture and techniques**: BPE tokenization, rotary position embeddings, grouped-query attention, KV caching, sampling strategies, and LoRA fine-tuning.

The text diffusion lab takes a different path: instead of autoregressive (left-to-right) generation, it uses a **masked diffusion model** (MDLM/LLaDA). Names emerge from pure noise — all [MASK] tokens — through iterative unmasking. Same scalar autograd engine as the pure Python version, same zero dependencies, but a fundamentally different generative paradigm.

The quantization lab shows how to deploy efficiently: **INT8 quantization** compresses a trained model from 32-bit floats to 8-bit integers, reducing size by ~4× and speeding up inference. This is how production models run on edge devices and servers.

Several labs explore **inference optimization** — the techniques used by production systems like vLLM, FlashAttention, and TensorRT-LLM. Speculative decoding uses a small draft model to propose tokens while a larger target model verifies them in one pass. FlashAttention restructures attention to stay in fast on-chip memory. PagedAttention applies OS-style virtual memory paging to KV caches. Disaggregated serving splits prefill and decode onto separate workers, eliminating head-of-line blocking.

The soft thinking and soft training labs explore **preserving the full output distribution** instead of collapsing to a single token — passing "concept tokens" (probability-weighted blends of all embeddings) forward, and training the model to work with its own uncertain outputs.

## Who is this for?

- **ML students** learning transformers from first principles — start with pure Python, see the full algorithm, then watch frameworks simplify it
- **Engineers** comparing framework tradeoffs — see how PyTorch, JAX, and MLX express the same model differently
- **Practitioners** scaling to production — learn batching, GPU optimization, modern architectures, and deployment techniques
- **Researchers** exploring new paradigms — see how diffusion and soft thinking change the generative process

## Learning paths

**New to transformers?** Start here:
- `01_pure_python` → understand the complete algorithm
- `03_pytorch` → see how frameworks simplify it
- `04_pytorch_batched` → learn production engineering (batching, padding, masking)

**Comparing frameworks?** Jump to:
- `03_pytorch` (object-oriented, imperative)
- `05_jax` (functional, pure functions)
- `07_mlx` (Apple Silicon, unified memory)

**Modern architecture?** See what changed since the original transformer:
- `09_bpe_tokenizer` → how real tokenizers work (BPE from scratch)
- `10_rope` → rotary position embeddings (used by LLaMA, Mistral)
- `11_gqa` → grouped-query attention (KV head sharing)

**Understanding inference?** Learn why decoding is slow and how to make it fast:
- `12_kv_cache` → the fundamental inference optimization
- `13_sampling` → how sampling strategies shape output
- `17_speculative_decoding` → lossless speedup via draft-and-verify
- `18_tiled_attention` → the memory wall and how FlashAttention beats it
- `19_paged_attention` → OS-style memory management for serving at scale
- `22_disaggregated_serving` → split prefill/decode for production serving

**Fine-tuning and deployment?** Check out:
- `14_lora` → parameter-efficient fine-tuning with low-rank adapters
- `16_pytorch_quantized` → compress models 4× for production

**Interested in diffusion?** Go straight to:
- `15_text_diffusion` — a fundamentally different generative paradigm

**Exploring soft thinking?** See what happens when you stop discarding information:
- `20_soft_thinking` → concept tokens preserve the full distribution at inference
- `21_soft_training` → train the model to expect soft inputs (scheduled curriculum)

## Running

**Browse online:** Visit the **[Interactive Web Tutorial](https://danilop.github.io/micro-gpt-and-beyond/)** to read the code with line-by-line explanations — no installation needed.

**Run locally with the tutorial:** Start the web tutorial server to browse code AND run labs from the browser:

```bash
./start_tutorial.sh
```

**Quick start:** Use the `run.py` helper script from the project root:

```bash
python run.py 01          # Run lab 01 (pure Python)
python run.py 09          # Run lab 09 (BPE tokenizer)
python run.py --list      # List all available labs
```

**Direct execution:**

`01_pure_python`, `09_bpe_tokenizer`, `15_text_diffusion`, and `19_paged_attention` have no dependencies — run them with plain Python:

```bash
python 01_pure_python/microgpt.py
python 09_bpe_tokenizer/main.py
python 15_text_diffusion/microdiffusion.py
python 19_paged_attention/main.py
```

All other versions (02–08, 10–14, 16–18, 20–22) are managed with [uv](https://docs.astral.sh/uv/) and have their own `pyproject.toml`:

```bash
cd 02_numpy_manual_backprop
uv run python main.py
```

Each subfolder has its own README with a deeper look at what makes that implementation interesting.

## Credits

This project is inspired by [microGPT](https://karpathy.ai/microgpt.html) by [Andrej Karpathy](https://github.com/karpathy) — a GPT trained and run in a single file of pure, dependency-free Python. The `01_pure_python` folder contains his original, unmodified code (the only change is the file path for the dataset to fit the project structure). The framework labs reimplement the same algorithm in NumPy, PyTorch, JAX, and MLX to show how the same ideas translate across tools. Further labs extend the architecture with modern techniques and explore inference optimization, alternative generation paradigms, and production serving — showing that the core ideas behind today's LLM systems are accessible at any scale.
