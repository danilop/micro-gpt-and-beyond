#!/usr/bin/env python3
"""Generate line-by-line explanations for all labs using Claude API.

Run: ANTHROPIC_API_KEY=... python generate_explanations.py
"""

import json
import os
import sys
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "explanations"
OUTPUT_DIR.mkdir(exist_ok=True)

LABS = [
    ("01_pure_python", "microgpt.py", "Pure Python — the complete GPT algorithm with manual autograd"),
    ("02_numpy_manual_backprop", "main.py", "NumPy with hand-written backpropagation"),
    ("03_pytorch", "main.py", "PyTorch — framework handles autograd"),
    ("04_pytorch_batched", "main.py", "PyTorch with mini-batch training, padding, and masking"),
    ("05_jax", "main.py", "JAX — purely functional style with JIT"),
    ("06_jax_batched", "main.py", "JAX batched training using vmap"),
    ("07_mlx", "main.py", "MLX on Apple Silicon"),
    ("08_mlx_batched", "main.py", "MLX with mini-batch training"),
    ("09_bpe_tokenizer", "main.py", "Byte-Pair Encoding tokenizer from scratch"),
    ("10_rope", "main.py", "Rotary Position Embeddings (RoPE)"),
    ("11_gqa", "main.py", "Grouped-Query Attention (GQA/MQA)"),
    ("12_kv_cache", "main.py", "KV Cache for efficient inference"),
    ("13_sampling", "main.py", "Sampling strategies: greedy, temperature, top-k, top-p, min-p"),
    ("14_lora", "main.py", "LoRA — low-rank adaptation for fine-tuning"),
    ("15_text_diffusion", "microdiffusion.py", "Text diffusion — generation through iterative unmasking"),
    ("16_pytorch_quantized", "main.py", "INT8 quantization for inference"),
    ("17_speculative_decoding", "main.py", "Speculative decoding with draft and target models"),
    ("18_tiled_attention", "main.py", "Tiled attention (FlashAttention algorithm)"),
    ("19_paged_attention", "main.py", "Paged KV cache (PagedAttention)"),
    ("20_soft_thinking", "main.py", "Soft thinking — concept tokens at inference"),
    ("21_soft_training", "main.py", "Soft training — concept tokens during training"),
    ("22_disaggregated_serving", "main.py", "Disaggregated serving — separate prefill and decode"),
]


def generate_for_lab(client, lab_dir, code_file, lab_description):
    code_path = PROJECT_ROOT / lab_dir / code_file
    code = code_path.read_text()
    lines = code.split("\n")
    n_lines = len(lines)

    print(f"  {lab_dir}: {n_lines} lines...")

    prompt = f"""You are annotating a Python source file for an interactive code tutorial about building GPT language models from scratch.

Lab context: {lab_description}

For each CODE line (not blank lines, not pure comment lines), write a short explanation (1-2 sentences) of what that line does in the context of this lab. The explanation should be:
- Clear and precise
- Non-verbose (aim for 15-30 words)
- Focused on what and why, not restating the code
- Using <code>name</code> tags for identifiers
- Accessible to someone learning ML

For comment-only lines (starting with #) and blank lines, skip them (do not include in output).
For section separator lines (like # ---), skip them.
For docstrings (triple quotes), you can briefly explain the function's purpose.

Return a JSON object mapping line numbers (as strings) to HTML explanation strings.
Example: {{"3": "Set the random seed for reproducibility.", "5": "Load training data from <code>input.txt</code>, one name per line."}}

Only include lines that have meaningful code. Aim for every code line to have an explanation.

Here is the complete source file:

```python
{code}
```

Return ONLY the JSON object, no markdown fencing."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Strip markdown fencing if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        explanations = json.loads(text)
    except json.JSONDecodeError:
        print(f"    WARNING: Failed to parse JSON for {lab_dir}, saving raw")
        explanations = {}

    return explanations


def main():
    client = anthropic.Anthropic()

    for lab_dir, code_file, description in LABS:
        output_path = OUTPUT_DIR / f"{lab_dir}.json"

        # Skip if already generated
        if output_path.exists() and "--force" not in sys.argv:
            print(f"  {lab_dir}: already exists, skipping (use --force to regenerate)")
            continue

        explanations = generate_for_lab(client, lab_dir, code_file, description)
        output_path.write_text(json.dumps(explanations, indent=2))
        n = len(explanations)
        print(f"    -> {n} explanations saved to {output_path.name}")


if __name__ == "__main__":
    print("Generating line-by-line explanations for all labs...")
    main()
    print("Done!")
