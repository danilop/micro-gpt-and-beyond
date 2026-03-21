#!/usr/bin/env python3
"""Local web server for the MicroGPT interactive tutorial."""

import json
import os
import shutil
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

PORT = 8000
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

# Lab metadata: (directory, code_file, title, tagline)
LABS = [
    ("01_pure_python", "microgpt.py",
     "microGPT — Pure Python",
     "Karpathy's original microGPT — the complete algorithm in one file. Everything else is just efficiency."),
    ("02_numpy_manual_backprop", "main.py",
     "NumPy with Manual Backpropagation",
     "Same architecture, but NumPy handles the math. Every gradient is derived by hand — you are the autograd."),
    ("03_pytorch", "main.py",
     "PyTorch",
     "PyTorch handles tensors and gradients. See how much boilerplate disappears when a framework does the differentiation."),
    ("04_pytorch_batched", "main.py",
     "PyTorch Batched",
     "Mini-batch training bridges the gap between educational toy and how real models are trained."),
    ("05_jax", "main.py",
     "JAX",
     "Purely functional style. No classes, no hidden state, no mutation — every function takes inputs and returns outputs."),
    ("06_jax_batched", "main.py",
     "JAX Batched",
     "Use jax.vmap to automatically vectorize single-example code across a batch — no rewriting needed."),
    ("07_mlx", "main.py",
     "MLX (Apple Silicon)",
     "Running on Apple Silicon GPU via MLX — NumPy-like API with automatic differentiation and unified memory."),
    ("08_mlx_batched", "main.py",
     "MLX Batched",
     "Mini-batch training on Apple Silicon, following the same padding/masking approach as PyTorch batched."),
    ("09_bpe_tokenizer", "main.py",
     "BPE Tokenizer",
     "Byte-Pair Encoding from scratch — pure Python, zero dependencies."),
    ("10_rope", "main.py",
     "Rotary Position Embeddings (RoPE)",
     "Position encoded by rotating Q/K vectors in complex space — relative position without learned parameters."),
    ("11_gqa", "main.py",
     "Grouped-Query Attention (GQA/MQA)",
     "From MHA to MQA to GQA: sharing KV heads reduces memory during inference while preserving quality."),
    ("12_kv_cache", "main.py",
     "KV Cache",
     "The single most important inference optimization: cache K/V tensors so each new token processes one position."),
    ("13_sampling", "main.py",
     "Sampling Strategies",
     "Five strategies — greedy, temperature, top-k, top-p, min-p — producing completely different output from the same model."),
    ("14_lora", "main.py",
     "LoRA (Low-Rank Adaptation)",
     "Freeze the model, inject tiny low-rank matrices, train only those. The base model never changes."),
    ("15_text_diffusion", "microdiffusion.py",
     "Text Diffusion",
     "Names emerge from pure noise through iterative unmasking — generation without left-to-right decoding."),
    ("16_pytorch_quantized", "main.py",
     "PyTorch Quantized",
     "Compress a trained model from FP32 to INT8 — the technique behind deploying models on phones and edge devices."),
    ("17_speculative_decoding", "main.py",
     "Speculative Decoding",
     "A small draft model guesses ahead, a larger target model verifies in one pass. Same distribution, faster."),
    ("18_tiled_attention", "main.py",
     "Tiled Attention (FlashAttention)",
     "Three attention implementations, identical outputs — the difference is memory trips. This is why FlashAttention matters."),
    ("19_paged_attention", "main.py",
     "Paged KV Cache (PagedAttention)",
     "Virtual memory paging applied to KV caches — reducing memory waste from 60-80% to near zero."),
    ("20_soft_thinking", "main.py",
     "Soft Thinking",
     "Instead of collapsing to one token per step, pass a probability-weighted blend forward — preserving information."),
    ("21_soft_training", "main.py",
     "Soft Training",
     "Use concept tokens during training too — a curriculum closes the train-test gap from inference-only soft thinking."),
    ("22_disaggregated_serving", "main.py",
     "Disaggregated Serving",
     "Prefill and decode on separate workers — how production systems avoid head-of-line blocking."),
]

# Labs that run with plain python3 (no dependencies / no pyproject.toml)
PLAIN_PYTHON_LABS = {"01_pure_python", "09_bpe_tokenizer", "15_text_diffusion", "19_paged_attention"}

# Track running processes so we can stop them
_running: dict[str, subprocess.Popen] = {}


class TutorialHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/api/labs":
            self._json_response([
                {"id": d, "title": t, "tagline": tg, "file": f}
                for d, f, t, tg in LABS
            ])
        elif self.path.startswith("/api/code/"):
            lab_id = self.path[len("/api/code/"):]
            self._serve_code(lab_id)
        elif self.path.startswith("/api/explanations/"):
            lab_id = self.path[len("/api/explanations/"):]
            self._serve_explanations(lab_id)
        elif self.path.startswith("/api/run/"):
            lab_id = self.path[len("/api/run/"):]
            self._run_lab(lab_id)
        elif self.path.startswith("/api/stop/"):
            lab_id = self.path[len("/api/stop/"):]
            self._stop_lab(lab_id)
        elif self.path.startswith("/lab/") or self.path.startswith("/lab.html"):
            # Serve lab.html for any /lab/<id> or /lab.html?lab=<id> route
            self._serve_file("lab.html")
        else:
            super().do_GET()

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_code(self, lab_id):
        for d, f, _, _ in LABS:
            if d == lab_id:
                code_path = PROJECT_ROOT / d / f
                if code_path.exists():
                    text = code_path.read_text()
                    self._json_response({"code": text, "filename": f})
                    return
        self._json_response({"error": "not found"}, 404)

    def _serve_explanations(self, lab_id):
        exp_path = ROOT / "explanations" / f"{lab_id}.json"
        if exp_path.exists():
            data = json.loads(exp_path.read_text())
            self._json_response(data)
        else:
            self._json_response({})

    def _run_lab(self, lab_id):
        """Stream lab execution output via Server-Sent Events."""
        # Find the lab
        lab_entry = None
        for d, f, _, _ in LABS:
            if d == lab_id:
                lab_entry = (d, f)
                break
        if not lab_entry:
            self._json_response({"error": "not found"}, 404)
            return

        lab_dir, code_file = lab_entry
        work_dir = PROJECT_ROOT / lab_dir

        # Kill any previous run of this lab
        if lab_id in _running:
            try:
                _running[lab_id].kill()
            except OSError:
                pass
            del _running[lab_id]

        # Build command
        if lab_id in PLAIN_PYTHON_LABS:
            cmd = ["python3", code_file]
        else:
            uv = shutil.which("uv") or "uv"
            cmd = [uv, "run", "python", code_file]

        # Start process
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except Exception as e:
            self._json_response({"error": str(e)}, 500)
            return

        _running[lab_id] = proc

        # SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send_event(event, data):
            try:
                self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                proc.kill()

        send_event("status", {"state": "running", "cmd": " ".join(cmd)})

        try:
            for line in proc.stdout:
                send_event("output", {"text": line})
            proc.wait()
            exit_code = proc.returncode
            send_event("status", {"state": "done", "exit_code": exit_code})
        except (BrokenPipeError, ConnectionResetError):
            proc.kill()
        finally:
            _running.pop(lab_id, None)

    def _stop_lab(self, lab_id):
        """Stop a running lab process."""
        if lab_id in _running:
            try:
                _running[lab_id].kill()
                _running[lab_id].wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
            _running.pop(lab_id, None)
            self._json_response({"status": "stopped"})
        else:
            self._json_response({"status": "not_running"})

    def _serve_file(self, filename):
        filepath = ROOT / filename
        if filepath.exists():
            content = filepath.read_bytes()
            ct = "text/html" if filename.endswith(".html") else "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    server = ThreadedHTTPServer(("localhost", PORT), TutorialHandler)
    print(f"MicroGPT Tutorial running at http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
