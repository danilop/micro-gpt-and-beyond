#!/usr/bin/env python3
"""Build a static labs.json bundle for GitHub Pages deployment.

Bundles all lab metadata, code, and explanations into a single JSON file
so the tutorial can run without a backend server.

Run: python build_static.py
Output: data/labs.json
"""

import json
from pathlib import Path

# Import lab metadata from server
from server import LABS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPLANATIONS_DIR = Path(__file__).resolve().parent / "explanations"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

bundle = []
for lab_dir, code_file, title, tagline in LABS:
    code_path = PROJECT_ROOT / lab_dir / code_file
    exp_path = EXPLANATIONS_DIR / f"{lab_dir}.json"

    code = code_path.read_text() if code_path.exists() else ""
    explanations = json.loads(exp_path.read_text()) if exp_path.exists() else {}

    bundle.append({
        "id": lab_dir,
        "title": title,
        "tagline": tagline,
        "file": code_file,
        "code": code,
        "explanations": explanations,
    })

output_path = OUTPUT_DIR / "labs.json"
output_path.write_text(json.dumps(bundle))
size_kb = output_path.stat().st_size / 1024
print(f"Built {output_path} ({len(bundle)} labs, {size_kb:.0f} KB)")
