# microGPT and Beyond

Nine implementations exploring tiny language models, inspired by Andrej Karpathy's [microGPT](https://karpathy.ai/microgpt.html) — a GPT trained and run in a single file of pure Python. Each version teaches something different about how neural networks are built, trained, and run. The models learn to generate human names from a dataset of ~32,000 real ones.

The progression goes from raw first principles to framework-powered GPU code, and beyond autoregressive generation:

```
01_pure_python/            Karpathy's original code. Zero dependencies. Scalar autograd.
02_numpy_manual_backprop/  NumPy arrays, but every gradient still written by hand.
03_pytorch/                PyTorch autograd takes over. Same model, ~30 lines shorter.
04_pytorch_batched/        Mini-batches, padding, masking — training at scale.
05_jax/                    Functional style. Pure functions, explicit state, JIT compilation.
06_jax_batched/            jax.vmap — write for one example, run for a batch automatically.
07_mlx/                    Apple Silicon GPU via MLX. Unified memory, lazy evaluation.
08_mlx_batched/            Batched MLX — same PyTorch-style padding, but on Apple GPU.
09_text_diffusion/         Masked diffusion model (MDLM/LLaDA). Names emerge from noise.
data/                      Shared dataset (auto-downloaded on first run).
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
| 09 text diffusion | hand-built scalar engine | no | no | CPU (slow) |

Version 09 is different: instead of autoregressive (left-to-right) generation, it uses a **masked diffusion model** (MDLM/LLaDA). Names emerge from pure noise — all [MASK] tokens — through iterative unmasking. Same scalar autograd engine as version 01, same zero dependencies, but a fundamentally different generative paradigm.

## Running

`01_pure_python` and `09_text_diffusion` have no dependencies — run them with plain Python:

```bash
python 01_pure_python/microgpt.py
python 09_text_diffusion/microdiffusion.py
```

All other versions (02–08) are managed with [uv](https://docs.astral.sh/uv/) and have their own `pyproject.toml`:

```bash
cd 02_numpy_manual_backprop
uv run python main.py
```

Each subfolder has its own README with a deeper look at what makes that implementation interesting.

## Credits

This project is inspired by [microGPT](https://karpathy.ai/microgpt.html) by [Andrej Karpathy](https://github.com/karpathy) — a GPT trained and run in a single file of pure, dependency-free Python. The `01_pure_python` folder contains his original, unmodified code (the only change is the file path for the dataset to fit the project structure). Versions 02–08 reimplement the same algorithm in different frameworks to show how the same ideas translate across tools. Version 09 uses the same autograd engine to explore a different generative paradigm — masked diffusion.
