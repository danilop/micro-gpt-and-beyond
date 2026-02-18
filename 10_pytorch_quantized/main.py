"""
microGPT — PyTorch quantized edition.

Same architecture as 03_pytorch, but with INT8 quantization for inference.
Trains in FP32, quantizes Linear layers to INT8, compares size and speed.
Shows ~4× memory reduction. Production systems also get speed improvements.
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

# Dataset
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'input.txt')
if not os.path.exists(input_path):
    import urllib.request
    url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer (character-level, identical to the original)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# Model (larger than 03 to show quantization benefits)
n_embd = 64
n_head = 4
n_layer = 2
block_size = 16
head_dim = n_embd // n_head


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
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
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


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


# Training
device = 'cpu'
model = MicroGPT().to(device)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)
num_steps = 1000

for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    input_ids = torch.tensor([tokens[:n]], device=device)
    targets = torch.tensor([tokens[1:n+1]], device=device)

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

    optimizer.zero_grad()
    loss.backward()

    lr_t = 1e-2 * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr_t

    optimizer.step()

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.item():.4f}")

# Quantization
print("\n--- quantization ---")

class QuantizedLinear(nn.Module):
    def __init__(self, fp32_linear):
        super().__init__()
        w = fp32_linear.weight.data
        scale = w.abs().max() / 127.0
        self.register_buffer('weight_int8', torch.round(w / scale).to(torch.int8))
        self.register_buffer('scale', torch.tensor(scale))
        self.bias = fp32_linear.bias
        
    def forward(self, x):
        return F.linear(x, self.weight_int8.to(x.dtype) * self.scale, self.bias)

def quantize_model(model):
    model_q = type(model)()
    model_q.load_state_dict(model.state_dict())
    for name, module in model_q.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, child_name, QuantizedLinear(child))
    return model_q

def benchmark(model, name):
    model.eval()
    torch.save(model.state_dict(), "/tmp/temp.pt")
    size_mb = os.path.getsize("/tmp/temp.pt") / (1024 * 1024)
    os.remove("/tmp/temp.pt")
    
    with torch.no_grad():
        start = time.time()
        for _ in range(100):
            tokens, idx = [BOS], torch.tensor([[BOS]], device=device)
            for _ in range(block_size):
                logits = model(idx)[0, -1] / 0.5
                token_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                if token_id == BOS: break
                tokens.append(token_id)
                idx = torch.tensor([tokens[-block_size:]], device=device)
        time_ms = (time.time() - start) * 10
    
    return size_mb, time_ms

fp32_size, fp32_time = benchmark(model, "FP32")
print(f"FP32: {fp32_size:.3f} MB, {fp32_time:.2f} ms/sample")

model_int8 = quantize_model(model)
int8_size, int8_time = benchmark(model_int8, "INT8")
print(f"INT8: {int8_size:.3f} MB ({int8_size/fp32_size:.1%}), {int8_time:.2f} ms/sample ({int8_time/fp32_time:.1%})")
print("\nNote: This implementation prioritizes memory savings.")
print("Production systems use INT8 kernels for both size and speed benefits.")

# Inference comparison
def generate(model, n=10):
    model.eval()
    with torch.no_grad():
        for i in range(n):
            tokens, idx = [BOS], torch.tensor([[BOS]], device=device)
            for _ in range(block_size):
                logits = model(idx)[0, -1] / 0.5
                token_id = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                if token_id == BOS: break
                tokens.append(token_id)
                idx = torch.tensor([tokens[-block_size:]], device=device)
            print(f"  {i+1:2d}: {''.join(uchars[t] for t in tokens[1:])}")

print("\n--- FP32 samples ---")
generate(model)
print("\n--- INT8 samples ---")
generate(model_int8)
