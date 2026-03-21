#!/usr/bin/env python3
"""Add content hashes to explanation JSON files for sync detection.

Transforms: {"12": "explanation text"}
Into:        {"12": {"text": "explanation text", "hash": "a1b2c3d4"}}
"""

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLANATIONS_DIR = Path(__file__).resolve().parent / "explanations"

LABS = [
    ("01_pure_python", "microgpt.py"),
    ("02_numpy_manual_backprop", "main.py"),
    ("03_pytorch", "main.py"),
    ("04_pytorch_batched", "main.py"),
    ("05_jax", "main.py"),
    ("06_jax_batched", "main.py"),
    ("07_mlx", "main.py"),
    ("08_mlx_batched", "main.py"),
    ("09_bpe_tokenizer", "main.py"),
    ("10_rope", "main.py"),
    ("11_gqa", "main.py"),
    ("12_kv_cache", "main.py"),
    ("13_sampling", "main.py"),
    ("14_lora", "main.py"),
    ("15_text_diffusion", "microdiffusion.py"),
    ("16_pytorch_quantized", "main.py"),
    ("17_speculative_decoding", "main.py"),
    ("18_tiled_attention", "main.py"),
    ("19_paged_attention", "main.py"),
    ("20_soft_thinking", "main.py"),
    ("21_soft_training", "main.py"),
    ("22_disaggregated_serving", "main.py"),
]


def line_hash(text):
    """Short hash of a code line's content (stripped)."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:8]


for lab_dir, code_file in LABS:
    exp_path = EXPLANATIONS_DIR / f"{lab_dir}.json"
    code_path = PROJECT_ROOT / lab_dir / code_file

    if not exp_path.exists() or not code_path.exists():
        continue

    code_lines = code_path.read_text().split("\n")
    explanations = json.loads(exp_path.read_text())

    # Check if already converted (has nested objects)
    sample_val = next(iter(explanations.values()), None)
    if isinstance(sample_val, dict):
        print(f"  {lab_dir}: already has hashes, skipping")
        continue

    new_explanations = {}
    for line_num_str, text in explanations.items():
        line_idx = int(line_num_str) - 1
        if 0 <= line_idx < len(code_lines):
            h = line_hash(code_lines[line_idx])
        else:
            h = ""
        new_explanations[line_num_str] = {"text": text, "hash": h}

    exp_path.write_text(json.dumps(new_explanations, indent=2))
    print(f"  {lab_dir}: {len(new_explanations)} entries updated with hashes")

print("Done!")
