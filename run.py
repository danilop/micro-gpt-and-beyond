#!/usr/bin/env python
"""
Run any lab from the project root.

Usage:
    python run.py 01          # Run lab 01 (pure Python)
    python run.py 03          # Run lab 03 (PyTorch)
    python run.py 09          # Run lab 09 (text diffusion)
    python run.py --list      # List all available labs
"""

import os
import sys
import subprocess

# Use the same Python interpreter that's running this script
PYTHON = sys.executable

LABS = {
    '01': ('01_pure_python', f'{PYTHON} microgpt.py'),
    '02': ('02_numpy_manual_backprop', 'uv run python main.py'),
    '03': ('03_pytorch', 'uv run python main.py'),
    '04': ('04_pytorch_batched', 'uv run python main.py'),
    '05': ('05_jax', 'uv run python main.py'),
    '06': ('06_jax_batched', 'uv run python main.py'),
    '07': ('07_mlx', 'uv run python main.py'),
    '08': ('08_mlx_batched', 'uv run python main.py'),
    '09': ('09_text_diffusion', f'{PYTHON} microdiffusion.py'),
    '10': ('10_pytorch_quantized', 'uv run python main.py'),
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
    print("  09  text_diffusion         Masked diffusion model (MDLM/LLaDA)")
    print("  10  pytorch_quantized      INT8 quantization for inference")
    print("\nUsage: python run.py <lab_number>")

def run_lab(lab_num):
    if lab_num not in LABS:
        print(f"Error: Lab '{lab_num}' not found.")
        print("Run 'python run.py --list' to see available labs.")
        sys.exit(1)
    
    lab_dir, command = LABS[lab_num]
    lab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), lab_dir)
    
    if not os.path.exists(lab_path):
        print(f"Error: Directory '{lab_dir}' not found.")
        sys.exit(1)
    
    print(f"Running lab {lab_num}: {lab_dir}")
    print(f"Command: {command}\n")
    print("-" * 79)
    
    # Run the command in the lab directory
    result = subprocess.run(command, shell=True, cwd=lab_path)
    sys.exit(result.returncode)

if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        sys.exit(0)
    
    if sys.argv[1] in ['-l', '--list']:
        list_labs()
        sys.exit(0)
    
    lab_num = sys.argv[1].zfill(2)  # '1' -> '01'
    run_lab(lab_num)
