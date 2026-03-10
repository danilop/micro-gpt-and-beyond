"""
microGPT — Soft training edition.

Builds on Lab 14 (soft thinking): instead of only using concept tokens at
inference time, this version also uses them during training. A curriculum
gradually replaces ground-truth token embeddings with the model's own soft
predictions, closing the train-test gap that limits inference-only soft thinking.

  Standard training:  input = embed(ground_truth_token)
  Soft training:      input = (1-mix) * embed(ground_truth) + mix * concept_token
                      where mix increases from 0 -> 1 during training
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

# Build on Lab 14: reuse model architecture and soft generation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "14_soft_thinking"))
from main import BOS, MicroGPT, block_size, device, docs, generate, num_steps, uchars, vocab_size

# ---------------------------------------------------------------------------
# Training — standard vs. soft input mixing
# ---------------------------------------------------------------------------
soft_temp = 1.0


def train_model(model, name, soft_mix=False):
    """Train with standard teacher forcing, or with soft input curriculum."""
    print(f"\n--- training {name} ({sum(p.numel() for p in model.parameters())} params) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.85, 0.99), eps=1e-8)

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        input_ids = torch.tensor([tokens[:n]], device=device)
        targets = torch.tensor([tokens[1 : n + 1]], device=device)

        if soft_mix and n > 1:
            mix = step / num_steps  # curriculum: 0 -> 1

            # Model's soft predictions (detached — no gradient through this path)
            with torch.no_grad():
                pred_logits = model(input_ids)
                soft_embeds = F.softmax(pred_logits / soft_temp, dim=-1) @ model.wte.weight

            # BOS stays ground truth; rest blended with concept tokens
            # soft_embeds[:, i] predicts position i+1, so shift left by one
            gt_embeds = model.wte(input_ids)
            mixed = torch.cat([gt_embeds[:, :1], (1 - mix) * gt_embeds[:, 1:] + mix * soft_embeds[:, :-1]], dim=1)

            logits = model(inputs_embeds=mixed)
        else:
            logits = model(input_ids)

        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()

        lr_t = 1e-2 * (1 - step / num_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t

        optimizer.step()

        extra = f" | mix {mix:.2f}" if soft_mix and n > 1 else ""
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.item():.4f}{extra}")


# ---------------------------------------------------------------------------
# Train both models from the same initial weights (fair comparison)
# ---------------------------------------------------------------------------
standard_model = MicroGPT().to(device)
soft_model = MicroGPT().to(device)
soft_model.load_state_dict(standard_model.state_dict())

train_model(standard_model, "standard")
train_model(soft_model, "soft-trained", soft_mix=True)

# ---------------------------------------------------------------------------
# Compare: standard-trained vs. soft-trained, hard vs. soft decoding
# ---------------------------------------------------------------------------
print("\n--- comparison: standard-trained vs. soft-trained ---\n")
n_samples = 20
max_H = math.log(vocab_size)

configs = [
    (standard_model, "standard-trained", "hard", 1.0),
    (standard_model, "standard-trained", "soft", 1.0),
    (soft_model, "soft-trained", "hard", 1.0),
    (soft_model, "soft-trained", "soft", 1.0),
]

with torch.no_grad():
    for model_obj, model_name, mode, st in configs:
        model_obj.eval()
        print(f"{model_name} + {mode} decoding:")
        all_H = []
        for i in range(n_samples):
            toks, ents = generate(model_obj, mode, st)
            name = "".join(uchars[t] for t in toks)
            avg_H = sum(ents) / len(ents) if ents else 0
            all_H.append(avg_H)
            if i < 10:
                print(f"  {i + 1:2d}: {name:<15s} entropy {avg_H:.2f}/{max_H:.2f}")
        print(f"  -> mean entropy: {sum(all_H) / len(all_H):.2f}/{max_H:.2f}\n")

print("""--- what's happening ---

Standard training uses teacher forcing: the model always sees perfect
ground-truth embeddings as input. At inference with soft decoding (Lab 14),
it encounters concept tokens — weighted blends it never trained on.
This train-test mismatch limits soft decoding's effectiveness.

Soft training fixes this with a curriculum:

  input = (1 - mix) * embed(ground_truth) + mix * concept_token
  mix: 0 -> 1 over training (start standard, end fully soft)

Each training step has two forward passes:
  1. Standard forward (detached) -> logits -> compute concept tokens
  2. Mixed-input forward -> loss -> backprop

The concept token at position i comes from the model's prediction at
position i-1, replacing the ground-truth embedding that would normally
occupy that slot. Position 0 (BOS) always stays ground truth.

This is scheduled sampling with soft tokens: the model gradually learns
to work with continuous inputs, closing the distribution gap between
training and soft inference.
""")
