# Understanding LLMs by Building One

A progressive series of labs exploring tiny language models, inspired by Andrej Karpathy's [microGPT](https://karpathy.ai/microgpt.html), a GPT trained and run in a single file of pure Python. Each lab teaches something different about how neural networks are built, trained, and served. The models learn to generate human names from a dataset of ~32,000 real ones.

**[Interactive Web Tutorial](https://danilop.github.io/micro-gpt-and-beyond/)** — browse the code with line-by-line explanations, Mermaid diagrams with per-line node highlighting, and lab descriptions. The 24 labs are organized into 8 chapters (Foundations, Frameworks, Tokenization & Architecture, Inference Optimization, Fine-tuning & Deployment, Alternative Paradigms, Production Serving, Self-Improvement), each with an overview and a conceptual diagram. Run labs directly from the browser when using the local server.

The progression goes from raw first principles to framework-powered GPU code, modern architecture upgrades, inference optimization, and alternative paradigms:

```
labs/
  01_pure_python/            Karpathy's original code. Zero dependencies. Scalar autograd.
  02_numpy_manual_backprop/  NumPy arrays, but every gradient still written by hand.
  03_pytorch/                PyTorch autograd takes over. Same model, ~30 lines shorter.
  04_pytorch_batched/        Mini-batches, padding, masking. Training at scale.
  05_jax/                    Functional style. Pure functions, explicit state, JIT compilation.
  06_jax_batched/            jax.vmap: write for one example, run for a batch automatically.
  07_mlx/                    Apple Silicon GPU via MLX. Unified memory, lazy evaluation.
  08_mlx_batched/            Batched MLX. Same PyTorch-style padding, but on Apple GPU.
  09_bpe_tokenizer/          Byte-Pair Encoding from scratch. The algorithm behind GPT's tokenizer.
  10_rope/                   Rotary Position Embeddings. How modern LLMs encode position.
  11_gqa/                    Grouped-Query Attention (MHA to GQA to MQA). KV head sharing.
  12_kv_cache/               KV cache for inference. THE fundamental decoding optimization.
  13_sampling/               Sampling strategies: greedy, temperature, top-k, top-p, min-p.
  14_lora/                   LoRA. Parameter-efficient fine-tuning with low-rank adapters.
  15_pytorch_quantized/      INT8 quantization for inference. FP32 to INT8, 4x smaller, faster.
  16_text_diffusion/         Masked diffusion model (MDLM/LLaDA). Names emerge from noise.
  17_soft_thinking/          Concept tokens preserve the full output distribution at inference.
  18_soft_training/          Soft input curriculum closes the train-test gap for concept tokens.
  19_speculative_decoding/   Draft model guesses, target model verifies. Lossless speedup.
  20_tiled_attention/        FlashAttention algorithm from scratch. Tiling beats the memory wall.
  21_paged_attention/        PagedAttention (vLLM). OS-style virtual memory for KV caches.
  22_disaggregated_serving/  Split prefill and decode onto separate workers. No more head-of-line blocking.
  23_self_improving/         Generate, score, filter, retrain. Improve the training data.
  24_evolutionary/           Population-based training. Improve the model itself.
data/                        Shared dataset (auto-downloaded on first run if not present).
```

## Chapters

The [interactive web tutorial](https://danilop.github.io/micro-gpt-and-beyond/) organizes the labs into 8 chapters, each with a description and conceptual diagram:

| Chapter | Labs | Theme |
|---------|------|-------|
| **Foundations** | 01, 02 | Scalar autograd and manual backprop — see every gradient |
| **Frameworks** | 03–08 | PyTorch, JAX, MLX — same model, three paradigms, single and batched |
| **Tokenization & Architecture** | 09, 10, 11 | BPE, RoPE, GQA — the upgrades behind LLaMA and Mistral |
| **Inference Optimization** | 12, 13 | KV cache and sampling strategies — the basics of fast decoding |
| **Fine-tuning & Deployment** | 14, 15 | LoRA and INT8 quantization — adapt and compress |
| **Alternative Paradigms** | 16, 17, 18 | Text diffusion and soft thinking — beyond left-to-right |
| **Production Serving** | 19, 20, 21, 22 | Speculative decoding, FlashAttention, PagedAttention, disaggregated serving |
| **Self-Improvement** | 23, 24 | Filtered self-training and population-based evolution |

## The idea

Every version trains the same architecture (a character-level transformer with RMSNorm, ReLU, multi-head attention, and linear LR decay) on the same names dataset. What changes is *how* the computation is expressed:

| Version | Autograd | Tensors | Batching | Hardware |
|---------|----------|---------|----------|----------|
| **Foundations** | | | | |
| 01 pure Python | hand-built scalar engine | no | no | CPU (slow) |
| 02 NumPy | none, manual backward pass | yes | no | CPU |
| **Frameworks** | | | | |
| 03 PyTorch | automatic | yes | no | CPU |
| 04 PyTorch batched | automatic | yes | yes (32) | CPU |
| 05 JAX | automatic (functional) | yes | no | CPU/GPU |
| 06 JAX batched | automatic (functional) | yes | yes (32, vmap) | CPU/GPU |
| 07 MLX | automatic | yes | no | Apple GPU |
| 08 MLX batched | automatic | yes | yes (32) | Apple GPU |
| **Tokenization & Architecture** | | | | |
| 09 BPE tokenizer | n/a (bigram model) | no | no | CPU |
| 10 RoPE | automatic | yes | no | CPU |
| 11 GQA | automatic | yes | no | CPU |
| **Inference Optimization** | | | | |
| 12 KV cache | automatic | yes | no | CPU |
| 13 sampling | automatic | yes | no | CPU |
| **Fine-tuning & Deployment** | | | | |
| 14 LoRA | automatic | yes | no | CPU |
| 15 PyTorch quantized | automatic | yes | no | CPU (INT8) |
| **Alternative Paradigms** | | | | |
| 16 text diffusion | automatic | yes | no | CPU |
| 17 soft thinking | automatic | yes | no | CPU |
| 18 soft training | automatic | yes | no | CPU |
| **Production Serving** | | | | |
| 19 speculative decoding | automatic | yes | no | CPU |
| 20 tiled attention | automatic | yes | no | CPU |
| 21 paged attention | hand-built scalar engine | no | no | CPU (slow) |
| 22 disaggregated serving | automatic | yes | no | CPU |
| **Self-Improvement** | | | | |
| 23 self-improving | automatic | yes | no | CPU |
| 24 evolutionary | automatic | yes | no | CPU |

Beyond the framework comparisons, the labs extend the base model with **modern architecture and techniques**: BPE tokenization, rotary position embeddings, grouped-query attention, KV caching, sampling strategies, and LoRA fine-tuning.

The text diffusion lab takes a different path: instead of autoregressive (left-to-right) generation, it uses a **masked diffusion model** (MDLM/LLaDA). Names emerge from pure noise, all [MASK] tokens, through iterative unmasking. It keeps the same small-transformer scale as the PyTorch labs, but swaps in bidirectional attention and a different training objective.

The quantization lab shows how to deploy efficiently: **INT8 quantization** compresses a trained model from 32-bit floats to 8-bit integers, reducing size by ~4x. The educational implementation demonstrates the size tradeoff directly and explains why production kernels also translate that compression into inference speedups.

Several labs explore **inference optimization**, the techniques used by production systems like vLLM, FlashAttention, and TensorRT-LLM. Speculative decoding uses a small draft model to propose tokens while a larger target model verifies them in one pass. FlashAttention restructures attention to stay in fast on-chip memory. PagedAttention applies OS-style virtual memory paging to KV caches. Disaggregated serving splits prefill and decode onto separate workers, eliminating head-of-line blocking.

The soft thinking and soft training labs explore **preserving the full output distribution** instead of collapsing to a single token. They pass "concept tokens" (probability-weighted blends of all embeddings) forward, and train the model to work with its own uncertain outputs.

## Who is this for?

- **ML students** learning transformers from first principles. Start with pure Python, see the full algorithm, then watch frameworks simplify it.
- **Engineers** comparing framework tradeoffs. See how PyTorch, JAX, and MLX express the same model differently.
- **Practitioners** scaling to production. Learn batching, GPU optimization, modern architectures, and deployment techniques.
- **Researchers** exploring new paradigms. See how diffusion and soft thinking change the generative process.

## Learning paths

**New to transformers?** Start here:
- `01_pure_python` to understand the complete algorithm
- `03_pytorch` to see how frameworks simplify it
- `04_pytorch_batched` to learn production engineering (batching, padding, masking)

**Comparing frameworks?** Jump to:
- `03_pytorch` (object-oriented, imperative)
- `05_jax` (functional, pure functions)
- `07_mlx` (Apple Silicon, unified memory)

**Modern architecture?** See what changed since the original transformer:
- `09_bpe_tokenizer` for how real tokenizers work (BPE from scratch)
- `10_rope` for rotary position embeddings (used by LLaMA, Mistral)
- `11_gqa` for grouped-query attention (KV head sharing)

**Understanding inference?** Learn why decoding is slow and how to make it fast:
- `12_kv_cache` for the fundamental inference optimization
- `13_sampling` for how sampling strategies shape output
- `19_speculative_decoding` for lossless speedup via draft-and-verify
- `20_tiled_attention` for the memory wall and how FlashAttention beats it
- `21_paged_attention` for OS-style memory management for serving at scale
- `22_disaggregated_serving` for splitting prefill and decode in production

**Fine-tuning and deployment?** Check out:
- `14_lora` for parameter-efficient fine-tuning with low-rank adapters
- `15_pytorch_quantized` to compress models 4x for production

**Interested in diffusion?** Go straight to:
- `16_text_diffusion` for a fundamentally different generative paradigm

**Exploring soft thinking?** See what happens when you stop discarding information:
- `17_soft_thinking` for concept tokens that preserve the full distribution at inference
- `18_soft_training` to train the model to expect soft inputs (scheduled curriculum)

**Curious about self-improvement?** See the two complementary loops:
- `23_self_improving` for generate-score-filter-retrain on the model's own outputs
- `24_evolutionary` for population-based training over architecture and optimizer choices

## How-to guides

### How to run a specific lab

1. From the project root, use the helper script: `python run_lab.py <number>` (e.g., `python run_lab.py 01`).
2. Or run directly: `cd labs/<lab_dir>` and use `uv run python main.py`.
3. Labs 01, 09, and 21 are pure Python — run them with `python3 main.py` (no `uv` needed).
4. Use `python run_lab.py --list` to see all available labs.
5. The dataset is auto-downloaded on first run if not already present in `data/`.

### How to compare framework implementations

1. Run the three unbatched framework labs side by side: `03_pytorch`, `05_jax`, `07_mlx`.
2. Compare model definition: PyTorch uses `nn.Module` classes, JAX uses pure functions with explicit state, MLX uses a PyTorch-like API on Apple GPU.
3. Compare training loops: look at how gradients are computed and parameters are updated in each.
4. Run the batched variants (`04`, `06`, `08`) to see how each framework handles padding, masking, and vectorization.
5. All six labs train the same architecture on the same data — differences in output come from framework semantics and hardware.

### How to run workspace smoke checks

1. From the shared workspace root, run `python3 smoke_check.py`.
2. This always validates the tutorial config, runs the `walk-the-code` core unit subset, and executes the pure-Python labs `09` and `21`.
3. Optional framework-heavy labs are reported separately based on whether `numpy`, `torch`, `jax`, and `mlx` are installed in the interpreter you used.
4. To include socket-binding server tests for `walk-the-code`, run `python3 smoke_check.py --include-server-tests`.

### How to explore the interactive tutorial

1. Run `./start_tutorial.sh` from the project root to start the local web server.
2. Open the URL printed in the terminal (usually `http://localhost:8000`).
3. Navigate chapters using the sidebar; click any code line to see its annotation.
4. Annotated lines show Mermaid diagrams with highlighted nodes that update as you navigate.
5. Use keyboard shortcuts: arrow keys to move between annotations, `Escape` to deselect.
6. Run labs directly from the browser using the "Run" button when the local server is active.

### How to add your own experiments

1. Copy an existing lab directory (e.g., `cp -r labs/03_pytorch labs/99_my_experiment`).
2. Edit `main.py` in your new directory — modify the model, hyperparameters, or training loop.
3. Run with `cd labs/99_my_experiment && uv run python main.py` to test changes.
4. To add your lab to the interactive tutorial, create a YAML annotation file and add an entry to `config.json`.
5. Use the walk-the-code Edit mode to annotate your code line by line.

## Reference

### Lab reference table

| # | Lab | Framework | Dependencies | Pure Python |
|---|-----|-----------|--------------|-------------|
| 01 | Pure Python | None | None | Yes |
| 02 | NumPy manual backprop | NumPy | numpy | No |
| 03 | PyTorch | PyTorch | numpy, torch | No |
| 04 | PyTorch batched | PyTorch | numpy, torch | No |
| 05 | JAX | JAX | jax | No |
| 06 | JAX batched | JAX | jax | No |
| 07 | MLX | MLX | mlx | No |
| 08 | MLX batched | MLX | mlx | No |
| 09 | BPE tokenizer | None | None | Yes |
| 10 | RoPE | PyTorch | numpy, torch | No |
| 11 | GQA | PyTorch | numpy, torch | No |
| 12 | KV cache | PyTorch | numpy, torch | No |
| 13 | Sampling | PyTorch | numpy, torch | No |
| 14 | LoRA | PyTorch | numpy, torch | No |
| 15 | PyTorch quantized | PyTorch | numpy, torch | No |
| 16 | Text diffusion | PyTorch | numpy, torch | No |
| 17 | Soft thinking | PyTorch | numpy, torch | No |
| 18 | Soft training | PyTorch | numpy, torch | No |
| 19 | Speculative decoding | PyTorch | numpy, torch | No |
| 20 | Tiled attention | PyTorch | numpy, torch | No |
| 21 | Paged attention | None | None | Yes |
| 22 | Disaggregated serving | PyTorch | numpy, torch | No |
| 23 | Self-improving | PyTorch | numpy, torch | No |
| 24 | Evolutionary | PyTorch | numpy, torch | No |

### Hyperparameter defaults

All labs (except 09) share these training hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_embd` | 16 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `block_size` | 16 | Maximum sequence length (context window) |
| Training steps | 1000 | Total optimization steps |
| Dataset | names | ~32,000 real human names from `data/` |

### Prerequisites per learning path

| Path | Prerequisites |
|------|---------------|
| New to transformers | Python 3 |
| Comparing frameworks | Python 3, uv; macOS for MLX labs |
| Modern architecture | Python 3, uv, basic PyTorch familiarity |
| Inference optimization | Python 3, uv, understanding of attention mechanism |
| Fine-tuning & deployment | Python 3, uv, PyTorch basics |
| Alternative paradigms | Python 3, uv, understanding of autoregressive models |

### Runtime environments

| Environment | What should work |
|-------------|------------------|
| Minimal Python | `walk-the-code` validation/tests, labs `01`, `09`, `21`, and the workspace smoke checks |
| Full PyTorch env | All PyTorch-based labs (`03`, `04`, `10`–`24`) plus the minimal set |
| Full research env | PyTorch labs + JAX labs (`05`, `06`) + MLX labs (`07`, `08`) on Apple Silicon |

## Running

**Browse online:** Visit the **[Interactive Web Tutorial](https://danilop.github.io/micro-gpt-and-beyond/)** to read the code with line-by-line explanations — no installation needed. Labs are grouped into 8 chapters, each with a description and conceptual diagram. Click any code line to see what it does; annotated lines include Mermaid diagrams with highlighted nodes that change as you navigate.

**Run locally with the tutorial:** Start the [walk-the-code](https://github.com/danilop/walk-the-code) web tutorial server to browse code, read explanations with diagrams, and run labs from the browser:

```bash
./start_tutorial.sh
```

If a sibling `../walk-the-code` checkout is present, `start_tutorial.sh` uses that local repo directly. That is the preferred setup when developing the two repos together because it avoids `uv tool install` and exercises your local `walk-the-code` changes end to end. If no sibling repo is present, the script falls back to an installed `wtc-serve`, and only installs it from GitHub as a last resort.

The tutorial is powered by [walk-the-code](https://github.com/danilop/walk-the-code), a standalone project you can also install as a CLI tool:

```bash
uv tool install "walk-the-code @ git+https://github.com/danilop/walk-the-code"
cd walk-the-code
wtc-serve --config config.json
```

**Quick start:** Use the `run_lab.py` helper script from the project root:

```bash
python run_lab.py 01          # Run lab 01 (pure Python)
python run_lab.py 09          # Run lab 09 (BPE tokenizer)
python run_lab.py --list      # List all available labs
```

**Direct execution:**

`01_pure_python`, `09_bpe_tokenizer`, and `21_paged_attention` have no dependencies and can be run with plain Python:

```bash
python labs/01_pure_python/microgpt.py
python labs/09_bpe_tokenizer/main.py
python labs/21_paged_attention/main.py
```

All other labs are managed with [uv](https://docs.astral.sh/uv/) and have their own `pyproject.toml`:

```bash
cd labs/02_numpy_manual_backprop
uv run python main.py
```

Each subfolder has its own README with a deeper look at what makes that implementation interesting.

## Credits

This project is inspired by [microGPT](https://karpathy.ai/microgpt.html) by [Andrej Karpathy](https://github.com/karpathy), a GPT trained and run in a single file of pure, dependency-free Python. The `labs/01_pure_python` folder contains his original, unmodified code (the only change is the file path for the dataset to fit the project structure). The framework labs reimplement the same algorithm in NumPy, PyTorch, JAX, and MLX to show how the same ideas translate across tools. Further labs extend the architecture with modern techniques and explore inference optimization, alternative generation paradigms, and production serving, showing that the core ideas behind today's LLM systems are accessible at any scale.

The interactive web tutorial is powered by [walk-the-code](https://github.com/danilop/walk-the-code), a standalone line-by-line code tutorial viewer with multi-language support, chapters, Mermaid diagrams with per-line node highlighting, and stale annotation detection.
