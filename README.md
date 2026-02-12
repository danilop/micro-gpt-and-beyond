# microGPT and Beyond

Six implementations of the same tiny GPT language model, inspired by Andrej Karpathy's [microGPT](https://karpathy.ai/microgpt.html) — a GPT trained and run in a single file of pure Python. Each version teaches something different about how neural networks are built, trained, and run. The model learns to generate human names from a dataset of ~32,000 real ones.

The progression goes from raw first principles to framework-powered GPU code:

```
01_pure_python/          Zero dependencies. Scalar autograd. The whole algorithm laid bare.
02_numpy_manual_backprop/  NumPy arrays, but every gradient still written by hand.
03_pytorch/              PyTorch autograd takes over. Same model, ~30 lines shorter.
04_pytorch_batched/      Mini-batches, padding, masking — training at scale.
05_jax/                  Functional style. Pure functions, explicit state, JIT compilation.
06_mlx/                  Apple Silicon GPU via MLX. Unified memory, lazy evaluation.
data/                    Shared dataset (auto-downloaded on first run).
```

## The idea

Every version trains the same architecture — a character-level transformer with RMSNorm, squared ReLU, multi-head attention, and cosine LR decay — on the same names dataset. What changes is *how* the computation is expressed:

| Version | Autograd | Tensors | Batching | Hardware |
|---------|----------|---------|----------|----------|
| 01 pure Python | hand-built scalar engine | no | no | CPU (slow) |
| 02 NumPy | none — manual backward pass | yes | no | CPU |
| 03 PyTorch | automatic | yes | no | CPU |
| 04 PyTorch batched | automatic | yes | yes (32) | CPU |
| 05 JAX | automatic (functional) | yes | no | CPU/GPU |
| 06 MLX | automatic | yes | no | Apple GPU |

## Running

`01_pure_python` has no dependencies — run it with plain Python:

```bash
python 01_pure_python/microgpt.py
```

All other versions (02–06) are managed with [uv](https://docs.astral.sh/uv/) and have their own `pyproject.toml`:

```bash
cd 02_numpy_manual_backprop
uv run python main.py
```

Each subfolder has its own README with a deeper look at what makes that implementation interesting.

## Credits

This project is inspired by [microGPT](https://karpathy.ai/microgpt.html) by [Andrej Karpathy](https://github.com/karpathy) — a GPT trained and run in a single file of pure, dependency-free Python. The `01_pure_python` folder contains his original code. The other five versions reimplement the same algorithm in different frameworks to show how the same ideas translate across tools.
