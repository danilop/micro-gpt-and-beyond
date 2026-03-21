#!/usr/bin/env python3
"""Audit annotations for quality issues. Flags lines that need improvement."""

import json
import os
import re
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

# Patterns that suggest a mechanical/lazy annotation
MECHANICAL_STARTS = [
    r"^(Set|Store|Create|Define|Initialize|Return|Compute|Extract|Get|Apply|Call|Pass|Assign|Declare|Import|Load|Save|Check|Build|Run|Use|Add|Make|Put|Move)\s",
]

# Words that suggest restating the code rather than explaining it
CODE_RESTATEMENT_WORDS = [
    "to a variable", "into a variable", "the variable", "the value of",
    "a new", "an instance of", "a list of", "a dict of",
]

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text)

def audit():
    flagged = []

    for lab_dir, code_file in LABS:
        exp_path = EXPLANATIONS_DIR / f"{lab_dir}.json"
        code_path = PROJECT_ROOT / lab_dir / code_file
        if not exp_path.exists() or not code_path.exists():
            continue

        code_lines = code_path.read_text().split("\n")
        explanations = json.loads(exp_path.read_text())

        for line_str, entry in explanations.items():
            text = entry["text"] if isinstance(entry, dict) else entry
            clean = strip_html(text)
            line_idx = int(line_str) - 1
            code = code_lines[line_idx].strip() if 0 <= line_idx < len(code_lines) else ""
            reasons = []

            # 1. Too short
            if len(clean) < 25:
                reasons.append("SHORT")

            # 2. Starts with mechanical verb
            for pat in MECHANICAL_STARTS:
                if re.match(pat, clean):
                    # Only flag if the annotation is also short-ish or generic
                    if len(clean) < 60:
                        reasons.append("MECHANICAL")
                    break

            # 3. Restates code
            for phrase in CODE_RESTATEMENT_WORDS:
                if phrase in clean.lower():
                    reasons.append("RESTATES")
                    break

            # 4. Ends with just a period after one short phrase (no real explanation)
            words = clean.split()
            if len(words) <= 4 and not any(c in clean for c in ['?', 'because', 'so ', 'since']):
                reasons.append("TERSE")

            if reasons:
                flagged.append({
                    "lab": lab_dir,
                    "line": line_str,
                    "code": code[:80],
                    "annotation": clean,
                    "reasons": reasons,
                })

    # Summary
    by_reason = {}
    by_lab = {}
    for f in flagged:
        for r in f["reasons"]:
            by_reason[r] = by_reason.get(r, 0) + 1
        by_lab[f["lab"]] = by_lab.get(f["lab"], 0) + 1

    print(f"Total flagged: {len(flagged)} / {sum(len(json.loads(open(EXPLANATIONS_DIR / f'{l}.json').read())) for l, _ in LABS)} annotations\n")
    print("By reason:")
    for r, n in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"  {r}: {n}")
    print("\nBy lab (top 10):")
    for lab, n in sorted(by_lab.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:3d} | {lab}")

    print("\n=== Sample flagged annotations ===\n")
    for f in flagged[:30]:
        print(f"[{','.join(f['reasons'])}] {f['lab']}:{f['line']}")
        print(f"  code: {f['code']}")
        print(f"  ann:  {f['annotation']}")
        print()

    return flagged

if __name__ == "__main__":
    audit()
