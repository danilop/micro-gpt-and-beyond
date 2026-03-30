"""
Understanding LLMs by Building One — BPE Tokenizer

Byte-Pair Encoding from scratch, then a side-by-side comparison of
character-level vs BPE tokenization on the names dataset. The BPE algorithm
was originally introduced for data compression in "A New Algorithm for Data
Compression" (Gage, 1994) and adapted for NLP in "Neural Machine Translation
of Rare Words with Subword Units" (Sennrich, Haddow & Birch, 2016),
https://arxiv.org/abs/1508.07909. GPT-2 uses byte-level BPE as described in
"Language Models are Unsupervised Multitask Learners" (Radford et al., 2019).

This lab implements character-level BPE for simplicity — production tokenizers
like GPT's operate on bytes, not characters, and use a much larger vocabulary
(50K-100K merges). Zero dependencies. Pure Python. Just counting and merging.
"""

import math
import os
import random

random.seed(42)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "input.txt")
if not os.path.exists(input_path):
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(url, input_path)

docs = [l.strip() for l in open(input_path).read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ---------------------------------------------------------------------------
# Character-level tokenizer (baseline — same as lab 01)
# ---------------------------------------------------------------------------
# Every unique character gets an integer id. BOS marks start/end of sequence.
uchars = sorted(set("".join(docs)))
char_to_id = {ch: i for i, ch in enumerate(uchars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}
CHAR_BOS = len(uchars)
char_vocab_size = len(uchars) + 1  # +1 for BOS

print(f"\n{'=' * 60}")
print("CHARACTER-LEVEL TOKENIZER (baseline)")
print(f"{'=' * 60}")
print(f"vocab size: {char_vocab_size} ({len(uchars)} chars + BOS)")


def char_encode(name):
    """Encode a name as a list of character token ids."""
    return [CHAR_BOS] + [char_to_id[ch] for ch in name] + [CHAR_BOS]


def char_decode(ids):
    """Decode token ids back to a string (strip BOS)."""
    return "".join(id_to_char[i] for i in ids if i != CHAR_BOS)


# ---------------------------------------------------------------------------
# BPE tokenizer — trained from scratch
# ---------------------------------------------------------------------------
# BPE starts with the character vocabulary and iteratively merges the most
# frequent adjacent pair into a new token. This is EXACTLY how GPT-2/3/4,
# LLaMA, etc. build their tokenizers (at much larger scale with ~50K merges).
#
# The tradeoff: larger vocab = shorter sequences = faster training/inference,
# but more parameters in the embedding table. At our tiny scale (names),
# character-level works great. At GPT scale (internet text), BPE with
# 50K-100K tokens is essential — without it, sequences would be absurdly long.
print(f"\n{'=' * 60}")
print("BPE TOKENIZER (training from scratch)")
print(f"{'=' * 60}")

NUM_MERGES = 200  # number of merge operations (new tokens to learn)

# Step 1: Start with character-level tokens for each name.
# We work with lists of integers. Initially each integer = one character.
# We reserve ids 0..25 for a-z and 26 for BOS, same as the char tokenizer.
BPE_BOS = len(uchars)
base_vocab_size = len(uchars) + 1  # characters + BOS

# Build the initial corpus: each name becomes a list of char token ids.
# We wrap with BOS just like the char tokenizer.
corpus = []
for name in docs:
    tokens = [BPE_BOS] + [char_to_id[ch] for ch in name] + [BPE_BOS]
    corpus.append(tokens)


def count_pairs(corpus):
    """Count frequency of each adjacent token pair across all sequences."""
    counts = {}
    for seq in corpus:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(corpus, pair, new_id):
    """Replace every occurrence of `pair` with `new_id` in all sequences."""
    new_corpus = []
    for seq in corpus:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        new_corpus.append(new_seq)
    return new_corpus


# Step 2: Iteratively merge the most frequent pair.
# Each merge creates a new token and shrinks all sequences by 1 wherever
# that pair appeared. The merge table records what was merged.
merges = {}  # (pair) -> new_token_id
vocab_size = base_vocab_size


def bpe_decode_token(tid):
    """Decode a single BPE token id to its string representation."""
    if tid == BPE_BOS:
        return ""
    if tid < len(uchars):
        return id_to_char[tid]
    # This is a merged token — find its components
    for (a, b), merged_id in merges.items():
        if merged_id == tid:
            return bpe_decode_token(a) + bpe_decode_token(b)
    return "?"


def bpe_display_token(tid):
    """Decode a BPE token for display, keeping BOS visible as '|'."""
    if tid == BPE_BOS:
        return "|"
    if tid < len(uchars):
        return id_to_char[tid]
    for (a, b), merged_id in merges.items():
        if merged_id == tid:
            return bpe_display_token(a) + bpe_display_token(b)
    return "?"


def token_str(tid):
    """Human-readable token string for display."""
    if tid == BPE_BOS:
        return "<BOS>"
    if tid < len(uchars):
        return id_to_char[tid]
    return bpe_decode_token(tid)


print(f"\nStarting vocab: {vocab_size} tokens ({len(uchars)} chars + BOS)")
print(f"Training {NUM_MERGES} merges...\n")
print(f"{'Merge':>5}  {'Pair':>20}  {'Visual':>12}  {'Freq':>6}  {'Vocab':>5}")
print("-" * 60)

for i in range(NUM_MERGES):
    counts = count_pairs(corpus)
    if not counts:
        print(f"No more pairs to merge after {i} merges.")
        break
    # Find the most frequent pair
    best_pair = max(counts, key=counts.get)
    best_count = counts[best_pair]
    new_id = vocab_size
    merges[best_pair] = new_id

    if i < 20 or i % 10 == 0 or i == NUM_MERGES - 1:
        visual = f"'{token_str(best_pair[0])}'+'{token_str(best_pair[1])}'"
        print(f"{i + 1:>5}  {best_pair!s:>20}  {visual:>12}  {best_count:>6}  {vocab_size + 1:>5}")

    corpus = merge_pair(corpus, best_pair, new_id)
    vocab_size += 1

print(f"\nFinal BPE vocab size: {vocab_size} ({base_vocab_size} base + {NUM_MERGES} merges)")


def bpe_decode(ids):
    """Decode a list of BPE token ids back to a string."""
    return "".join(bpe_decode_token(tid) for tid in ids)


# ---------------------------------------------------------------------------
# BPE encode: string -> token ids
# ---------------------------------------------------------------------------
def bpe_encode(name):
    """Encode a name using the learned BPE merges."""
    # Start with character-level tokens
    tokens = [BPE_BOS] + [char_to_id[ch] for ch in name] + [BPE_BOS]
    # Apply merges in the order they were learned (greedy, left-to-right)
    for pair, new_id in merges.items():
        tokens = merge_pair([tokens], pair, new_id)[0]
    return tokens


# ---------------------------------------------------------------------------
# Show encoding examples: character-level vs BPE
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("ENCODING EXAMPLES: Character vs BPE")
print(f"{'=' * 60}")
example_names = ["emma", "olivia", "charlotte", "alexander", "ann", "lee"]
# Also pick some random names from the dataset
example_names += random.sample(docs, 4)
# Deduplicate while preserving order
example_names = list(dict.fromkeys(example_names))[:10]

print("\nToken counts include BOS start/end markers (shown as '|' in BPE encoding).\n")
print(f"{'Name':<14} {'Char tokens':>11} {'BPE tokens':>10}  {'Compression':>11}  BPE encoding")
print("-" * 75)
for name in example_names:
    char_toks = char_encode(name)
    bpe_toks = bpe_encode(name)
    ratio = len(char_toks) / len(bpe_toks) if bpe_toks else 0
    bpe_visual = [f"'{bpe_display_token(t)}'" for t in bpe_toks]
    print(f"{name:<14} {len(char_toks):>11} {len(bpe_toks):>10}  {ratio:>10.2f}x  {' '.join(bpe_visual)}")


# ---------------------------------------------------------------------------
# Corpus-wide compression statistics
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("COMPRESSION STATISTICS (full corpus)")
print(f"{'=' * 60}")

total_char_tokens = sum(len(char_encode(name)) for name in docs)
total_bpe_tokens = sum(len(bpe_encode(name)) for name in docs)
avg_char = total_char_tokens / len(docs)
avg_bpe = total_bpe_tokens / len(docs)

print(f"Total names:          {len(docs):>10}")
print(f"Char vocab size:      {char_vocab_size:>10}")
print(f"BPE vocab size:       {vocab_size:>10}")
print(f"Total char tokens:    {total_char_tokens:>10}")
print(f"Total BPE tokens:     {total_bpe_tokens:>10}")
print(f"Avg tokens/name char: {avg_char:>10.2f}")
print(f"Avg tokens/name BPE:  {avg_bpe:>10.2f}")
print(f"Overall compression:  {total_char_tokens / total_bpe_tokens:>10.2f}x")

# ---------------------------------------------------------------------------
# BPE vocabulary: show what each merged token represents
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("LEARNED BPE VOCABULARY (merged tokens)")
print(f"{'=' * 60}")
print(f"\nBase tokens (0-{len(uchars) - 1}): {' '.join(uchars)}")
print(f"BOS token: {BPE_BOS}")
print("\nMerged tokens:")
print(f"{'ID':>5}  {'Represents':>12}  {'Components'}")
print("-" * 40)
for (a, b), new_id in merges.items():
    merged_str = bpe_display_token(new_id)
    comp_a = bpe_display_token(a).replace("|", "<BOS>")
    comp_b = bpe_display_token(b).replace("|", "<BOS>")
    print(f"{new_id:>5}  {merged_str!r:>12}  '{comp_a}' + '{comp_b}'")

# ---------------------------------------------------------------------------
# Bigram language model — trained with BOTH tokenizations
# ---------------------------------------------------------------------------
# A bigram model is the simplest language model: P(next_token | current_token).
# It's just a lookup table of transition counts, normalized to probabilities.
# No neural network, no autograd — pure counting.
#
# We train one bigram model on character tokens and one on BPE tokens to show
# how tokenization affects sequence modeling. With BPE, each "step" covers
# more text, so the bigram captures longer-range patterns.
print(f"\n{'=' * 60}")
print("BIGRAM LANGUAGE MODEL COMPARISON")
print(f"{'=' * 60}")


def train_bigram(encoded_corpus, v_size):
    """Train a bigram model: count transitions, add-1 smooth, normalize."""
    # counts[a][b] = number of times token b follows token a
    counts = [[0] * v_size for _ in range(v_size)]
    for seq in encoded_corpus:
        for i in range(len(seq) - 1):
            counts[seq[i]][seq[i + 1]] += 1
    # Normalize to probabilities with add-1 (Laplace) smoothing
    probs = []
    for row in counts:
        total = sum(row) + v_size  # add-1 smoothing
        probs.append([(c + 1) / total for c in row])
    return probs


def sample_bigram(probs, bos_token, max_len=20):
    """Sample a sequence from the bigram model."""
    tokens = [bos_token]
    for _ in range(max_len):
        # Sample next token from probability distribution
        p = probs[tokens[-1]]
        r = random.random()
        cumsum = 0
        for tid, prob in enumerate(p):
            cumsum += prob
            if r < cumsum:
                tokens.append(tid)
                break
        if tokens[-1] == bos_token and len(tokens) > 1:
            break
    return tokens


def bigram_log_likelihood(probs, encoded_corpus):
    """Compute average log-likelihood per token (higher = better)."""
    total_ll = 0.0
    total_tokens = 0
    for seq in encoded_corpus:
        for i in range(len(seq) - 1):
            p = probs[seq[i]][seq[i + 1]]
            if p > 0:
                total_ll += math.log(p)
            total_tokens += 1
    return total_ll / total_tokens if total_tokens > 0 else 0.0


# Train character-level bigram
print("\nTraining character-level bigram...")
char_corpus = [char_encode(name) for name in docs]
char_bigram = train_bigram(char_corpus, char_vocab_size)
char_ll = bigram_log_likelihood(char_bigram, char_corpus)
print(f"  Vocab size: {char_vocab_size}")
print(f"  Avg log-likelihood per token: {char_ll:.4f}")
print(f"  Perplexity per token: {math.exp(-char_ll):.2f}")

# Train BPE bigram
print("\nTraining BPE bigram...")
bpe_corpus = [bpe_encode(name) for name in docs]
bpe_bigram = train_bigram(bpe_corpus, vocab_size)
bpe_ll = bigram_log_likelihood(bpe_bigram, bpe_corpus)
print(f"  Vocab size: {vocab_size}")
print(f"  Avg log-likelihood per token: {bpe_ll:.4f}")
print(f"  Perplexity per token: {math.exp(-bpe_ll):.2f}")

# ---------------------------------------------------------------------------
# Generate names from both models
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("GENERATED NAMES")
print(f"{'=' * 60}")

print("\nCharacter-level bigram (27 tokens, ~6 tokens/name):")
for i in range(10):
    toks = sample_bigram(char_bigram, CHAR_BOS)
    name = char_decode(toks)
    print(f"  sample {i + 1:>2}: {name}")

print(f"\nBPE bigram ({vocab_size} tokens, ~{avg_bpe:.1f} tokens/name):")
for i in range(10):
    toks = sample_bigram(bpe_bigram, BPE_BOS)
    name = bpe_decode(toks)
    print(f"  sample {i + 1:>2}: {name}")

# ---------------------------------------------------------------------------
# Key takeaway
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("KEY TAKEAWAY")
print(f"{'=' * 60}")
print(f"""
BPE compresses the corpus from {total_char_tokens} to {total_bpe_tokens} tokens ({total_char_tokens / total_bpe_tokens:.1f}x shorter)
by learning {NUM_MERGES} merge rules that capture common character sequences.
Vocab grows from {char_vocab_size} to {vocab_size} tokens (more embedding params, but shorter sequences).

This is the algorithm behind GPT-2, LLaMA, and most modern LLM tokenizers.
At GPT scale, BPE with 50K-100K merges is essential to keep sequences tractable.
""")
