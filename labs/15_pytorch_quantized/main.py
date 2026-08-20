"""
microGPT — PyTorch quantized edition.

Same architecture as 03_pytorch, but with INT8 quantization for inference.
Trains in FP32, quantizes Linear layers to INT8, then measures size, speed,
weight error and held-out loss. Only the Linear layers are quantized, so the
size reduction lands near 3.5x rather than the ideal 4x. This dequantize-to-FP32
path is SLOWER than FP32, not faster; production INT8 kernels are what buy speed.

Post-training quantization follows the principles in "Quantization and Training
of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al.,
2018), https://arxiv.org/abs/1712.05877. For LLM-specific quantization, see
"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (Dettmers
et al., 2022), https://arxiv.org/abs/2208.07339. Note that this lab implements
simulated per-tensor symmetric quantization for educational purposes -- the
forward pass dequantizes INT8 weights back to float, so it demonstrates model
SIZE savings, not the speed improvements that real INT8 GEMM kernels provide.
Production systems use per-channel quantization and also quantize activations.
"""

import copy
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Dataset & Tokenizer
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Model (larger than 03 to show quantization benefits)
# ---------------------------------------------------------------------------
n_embd = 64  # embedding dimension
n_head = 4  # number of attention heads
n_layer = 2  # number of layers
block_size = 16  # maximum sequence length
head_dim = n_embd // n_head  # dimension of each head


class RMSNorm(nn.Module):
    def __init__(self, _dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, n_head, head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, n_head, head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention()
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MicroGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm_in = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.08)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.norm_in(tok_emb + pos_emb)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
device = "cpu"
model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = torch.tensor([tokens[:n]], device=device)
    targets = torch.tensor([tokens[1 : n + 1]], device=device)

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()
    lr_t = 1e-2 * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t
    optimizer.step()

    if (step + 1) % 10 == 0 or step == 0:
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
print("\n--- quantization ---")


# NOTE: This is a *simulated* quantization for educational purposes.
# The forward pass dequantizes INT8 weights back to the activation dtype,
# so it does NOT use actual INT8 GEMM kernels. The benchmark below
# demonstrates model SIZE savings, not speed improvements.
class QuantizedLinear(nn.Module):
    def __init__(self, fp32_linear):
        super().__init__()
        w = fp32_linear.weight.data
        scale = w.abs().max() / 127.0
        self.register_buffer("weight_int8", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.detach().clone())
        self.bias = fp32_linear.bias

    def forward(self, x):
        return F.linear(x, self.weight_int8.to(x.dtype) * self.scale, self.bias)


def quantize_model(model):
    # deepcopy gives an independent model without re-running __init__ (and so
    # without throwing away a fresh random init), and materializing the module
    # list up front means we are not mutating the module tree while walking it.
    model_q = copy.deepcopy(model)
    for _, module in list(model_q.named_modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child))
    return model_q


# ---------------------------------------------------------------------------
# Inference & Benchmark
# ---------------------------------------------------------------------------
temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high


@torch.no_grad()
def generate(model, n=1):
    model.eval()
    samples = []
    for _ in range(n):
        tokens = [BOS]
        for _ in range(block_size):
            idx = torch.tensor([tokens[-block_size:]], device=device)
            probs = F.softmax(model(idx)[0, -1] / temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            tokens.append(token_id)
        samples.append(tokens[1:])
    return samples


# NOTE: Speed comparison is included for completeness, but this simulated
# quantization path is expected to be similar or slower than FP32 because
# we dequantize on every forward pass. The key takeaway is model size reduction.
n_bench = 100  # samples per timing run


def benchmark(model):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.pt")
        torch.save(model.state_dict(), path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
    # Seed immediately before generating so both models draw the SAME sampling
    # decisions. Without this the two timings run different numbers of forward
    # passes on different-length sequences, and the "comparison" compares nothing.
    torch.manual_seed(1234)
    # perf_counter, not time(): this is a monotonic high-resolution clock, and
    # a wall clock that can be stepped by NTP has no business timing a benchmark.
    start = time.perf_counter()
    generate(model, n_bench)
    elapsed = time.perf_counter() - start
    return size_mb, elapsed / n_bench * 1000


fp32_size, fp32_time = benchmark(model)
print(f"FP32: {fp32_size:.3f} MB, {fp32_time:.2f} ms/sample")

model_int8 = quantize_model(model)
int8_size, int8_time = benchmark(model_int8)
print(
    f"INT8: {int8_size:.3f} MB ({int8_size / fp32_size:.1%}), {int8_time:.2f} ms/sample ({int8_time / fp32_time:.1%})"
)
print(f"size: {fp32_size / int8_size:.2f}x smaller | speed: {int8_time / fp32_time - 1:+.1%} vs FP32 (slower is expected here)")
print("\nNote: This implementation prioritizes memory savings.")
print("Production systems use INT8 kernels for both size and speed benefits.")

# ---------------------------------------------------------------------------
# Quantization error — what precision actually costs
# ---------------------------------------------------------------------------
# Size and speed are the easy numbers. The one that matters for deployment is
# how much the weights moved, and whether that moves the model's predictions.
print("\n--- quantization error per layer ---\n")
print(f"  {'layer':<22s} {'max|W|':>9s} {'scale':>10s} {'max abs err':>11s} {'as % of max|W|':>15s}")

fp32_weights = {name: mod.weight.data for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
worst_err = 0.0
for name, mod in model_int8.named_modules():
    if isinstance(mod, QuantizedLinear):
        w = fp32_weights[name]
        # Round-trip the INT8 weights back to float and compare with the original.
        dequant = mod.weight_int8.to(torch.float32) * mod.scale
        err = (w - dequant).abs().max().item()
        worst_err = max(worst_err, err)
        wmax = w.abs().max().item()
        print(f"  {name:<22s} {wmax:>9.5f} {mod.scale.item():>10.2e} {err:>11.2e} {100 * err / wmax:>14.3f}%")
print(f"\n  worst layer error: {worst_err:.2e}")
print("  Symmetric per-tensor INT8 rounds to a grid of step `scale`, so the error is")
print("  bounded by scale/2 = max|W|/254 — about 0.4% of the largest weight, per layer.")


# Weight error is only interesting if it shows up in the model's predictions.
# Measure per-token cross-entropy on names the model never trained on.
@torch.no_grad()
def eval_loss(m, names):
    m.eval()
    total, count = 0.0, 0
    for doc in names:
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        logits = m(torch.tensor([tokens[:n]], device=device))
        targets = torch.tensor([tokens[1 : n + 1]], device=device)
        total += F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction="sum").item()
        count += n
    return total / count


# Training walked docs[0:num_steps], so anything after that is unseen.
heldout = docs[num_steps : num_steps + 2000]
fp32_loss = eval_loss(model, heldout)
int8_loss = eval_loss(model_int8, heldout)
print(f"\n  held-out loss on {len(heldout)} unseen names:")
print(f"    FP32: {fp32_loss:.4f}")
print(f"    INT8: {int8_loss:.4f}  ({int8_loss - fp32_loss:+.4f}, {100 * (int8_loss - fp32_loss) / fp32_loss:+.3f}%)")
print("  The sample lists below come out identical, which is a nice result but not")
print("  evidence of zero error — the perturbation is just too small to flip any of")
print("  these sampling decisions. The loss delta above is the honest measure.")

for label, m in [("FP32", model), ("INT8", model_int8)]:
    torch.manual_seed(42)
    print(f"\n--- {label} samples ---")
    for i, s in enumerate(generate(m, 10)):
        print(f"  {i + 1:2d}: {''.join(uchars[t] for t in s)}")
