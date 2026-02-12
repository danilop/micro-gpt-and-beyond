# microGPT and Beyond — PyTorch Batched

Same PyTorch architecture as version 03, but scaled up with mini-batch training. This is the version that bridges the gap between "educational toy" and "how real models are trained."

## Why this version exists

Versions 01–03 all train on one name at a time. That's simple to understand, but wasteful — the GPU (or CPU) could process dozens of sequences in parallel. This version adds batching, padding, and attention masking, which are the engineering pieces you need to train efficiently.

## What's different

| | 03 (single) | 04 (batched) |
|---|---|---|
| Batch size | 1 | 32 |
| Embedding dim | 16 | 64 |
| Layers | 1 | 2 |
| Context length | 8 | 16 |
| Training steps | 500 | 1000 |

The model is 4x wider, 2x deeper, sees 32x more data per step, and trains for 2x as many steps. The generated names are noticeably better.

## What makes it interesting

### Padding and masking

Names have different lengths, but tensors need uniform dimensions. The `make_batch` function pads shorter sequences and creates a mask so the model knows what's real and what's filler:

```python
def make_batch(docs, step, batch_size):
    sequences = []
    for doc in batch_docs:
        toks = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        toks = toks[:block_size + 1]
        sequences.append(toks)

    max_len = max(len(s) for s in sequences)
    for s in sequences:
        n = len(s) - 1
        inp = s[:n] + [PAD] * (max_len - 1 - n)
        tgt = s[1:n+1] + [-100] * (max_len - 1 - n)  # -100 = ignore in loss
        mask = [False] * n + [True] * (max_len - 1 - n)
```

The `-100` target value is PyTorch's convention for `ignore_index` in `F.cross_entropy` — padded positions don't contribute to the loss.

### Attention mask stacking

The attention layer now combines two masks — the causal mask (can't look ahead) and the padding mask (can't attend to PAD tokens):

```python
def forward(self, x, pad_mask=None):
    att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    att = att.masked_fill(causal, float('-inf'))
    if pad_mask is not None:
        att = att.masked_fill(pad_mask[:, None, None, :], float('-inf'))
    att = F.softmax(att, dim=-1)
    att = torch.nan_to_num(att)  # handle all-masked rows
```

The `nan_to_num` call handles edge cases where a row is entirely masked (all `-inf` → softmax produces NaN). This is a practical detail that doesn't come up in single-sequence training.

### Separate embedding for PAD

The embedding table has `vocab_size + 1` entries, with `padding_idx=PAD` so the PAD embedding stays at zero and doesn't receive gradients:

```python
self.wte = nn.Embedding(vocab_size_with_pad, n_embd, padding_idx=PAD)
```

But the output head only projects to `vocab_size` — the model can never predict PAD as a next token.

## What you learn here

- How variable-length sequences are batched with padding
- The interplay between causal masks and padding masks in attention
- Why `ignore_index=-100` exists in cross-entropy loss
- The practical engineering that sits between "model works on one example" and "model trains efficiently"

## Run

```bash
uv run python main.py
```

Trains for 1000 steps (prints every 10) and generates 20 names. Noticeably better output than the single-sample versions thanks to the larger model and more training.
