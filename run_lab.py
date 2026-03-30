#!/usr/bin/env python3
"""
Run any lab from the project root.

Usage:
    python run_lab.py 01          # Run lab 01 (pure Python)
    python run_lab.py 03          # Run lab 03 (PyTorch)
    python run_lab.py 09          # Run lab 09 (BPE tokenizer)
    python run_lab.py --list      # List all available labs
"""

import os
import subprocess
import sys

# Use the same Python interpreter that's running this script
PYTHON = sys.executable

LABS = {
    "01": ("labs/01_pure_python", [PYTHON, "microgpt.py"]),
    "02": ("labs/02_numpy_manual_backprop", ["uv", "run", "python", "main.py"]),
    "03": ("labs/03_pytorch", ["uv", "run", "python", "main.py"]),
    "04": ("labs/04_pytorch_batched", ["uv", "run", "python", "main.py"]),
    "05": ("labs/05_jax", ["uv", "run", "python", "main.py"]),
    "06": ("labs/06_jax_batched", ["uv", "run", "python", "main.py"]),
    "07": ("labs/07_mlx", ["uv", "run", "python", "main.py"]),
    "08": ("labs/08_mlx_batched", ["uv", "run", "python", "main.py"]),
    "09": ("labs/09_bpe_tokenizer", [PYTHON, "main.py"]),
    "10": ("labs/10_rope", ["uv", "run", "python", "main.py"]),
    "11": ("labs/11_gqa", ["uv", "run", "python", "main.py"]),
    "12": ("labs/12_kv_cache", ["uv", "run", "python", "main.py"]),
    "13": ("labs/13_sampling", ["uv", "run", "python", "main.py"]),
    "14": ("labs/14_lora", ["uv", "run", "python", "main.py"]),
    "15": ("labs/15_pytorch_quantized", ["uv", "run", "python", "main.py"]),
    "16": ("labs/16_text_diffusion", ["uv", "run", "python", "main.py"]),
    "17": ("labs/17_soft_thinking", ["uv", "run", "python", "main.py"]),
    "18": ("labs/18_soft_training", ["uv", "run", "python", "main.py"]),
    "19": ("labs/19_speculative_decoding", ["uv", "run", "python", "main.py"]),
    "20": ("labs/20_tiled_attention", ["uv", "run", "python", "main.py"]),
    "21": ("labs/21_paged_attention", [PYTHON, "main.py"]),
    "22": ("labs/22_disaggregated_serving", ["uv", "run", "python", "main.py"]),
    "23": ("labs/23_self_improving", ["uv", "run", "python", "main.py"]),
    "24": ("labs/24_evolutionary", ["uv", "run", "python", "main.py"]),
}


def list_labs():
    print("Available labs:\n")
    print("  01  pure_python            Zero dependencies, scalar autograd")
    print("  02  numpy_manual_backprop  NumPy arrays, hand-written gradients")
    print("  03  pytorch                PyTorch autograd")
    print("  04  pytorch_batched        Mini-batches, padding, masking")
    print("  05  jax                    Functional style, pure functions")
    print("  06  jax_batched            jax.vmap automatic vectorization")
    print("  07  mlx                    Apple Silicon GPU (M1/M2/M3/M4)")
    print("  08  mlx_batched            Batched MLX on Apple GPU")
    print("  09  bpe_tokenizer          Byte-Pair Encoding from scratch")
    print("  10  rope                   Rotary Position Embeddings")
    print("  11  gqa                    Grouped-Query Attention (MHA/GQA/MQA)")
    print("  12  kv_cache               KV cache for fast inference")
    print("  13  sampling               Sampling strategies (greedy/top-k/top-p/min-p)")
    print("  14  lora                   LoRA parameter-efficient fine-tuning")
    print("  15  pytorch_quantized      INT8 quantization for inference")
    print("  16  text_diffusion         Masked diffusion model (MDLM/LLaDA)")
    print("  17  soft_thinking          Concept tokens preserve full distribution")
    print("  18  soft_training          Train with soft inputs (scheduled curriculum)")
    print("  19  speculative_decoding   Draft-and-verify lossless speedup")
    print("  20  tiled_attention        FlashAttention algorithm (memory wall)")
    print("  21  paged_attention        PagedAttention (vLLM-style KV cache)")
    print("  22  disaggregated_serving  Split prefill/decode onto separate workers")
    print("  23  self_improving         Self-improvement via generate-score-retrain")
    print("  24  evolutionary           Population-based training (evolutionary)")
    print("\nUsage: python run_lab.py <lab_number>")


def run_lab(lab_num):
    if lab_num not in LABS:
        print(f"Error: Lab '{lab_num}' not found.")
        print("Run 'python run_lab.py --list' to see available labs.")
        sys.exit(1)

    lab_dir, command = LABS[lab_num]
    lab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lab_dir)

    if not os.path.exists(lab_path):
        print(f"Error: Directory '{lab_dir}' not found.")
        sys.exit(1)

    print(f"Running lab {lab_num}: {lab_dir}")
    print(f"Command: {' '.join(command)}\n")
    print("-" * 79)

    result = subprocess.run(command, cwd=lab_path)
    sys.exit(result.returncode)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] in ["-l", "--list"]:
        list_labs()
        sys.exit(0)

    lab_num = sys.argv[1].zfill(2)  # '1' -> '01'
    run_lab(lab_num)
