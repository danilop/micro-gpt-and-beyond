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
like GPT's operate on bytes, not characters, use a pre-tokenization regex to
stop merges crossing word boundaries, and learn a much larger vocabulary
(50K-100K merges). What this lab does share with them is the core loop (count
pairs, merge the most frequent) and one rule that is easy to get wrong:
special tokens are never merge candidates. Zero dependencies. Pure Python.
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
# frequent adjacent pair into a new token. That counting-and-merging loop is
# the same one GPT-2/3/4 and LLaMA use, at much larger scale (~50K merges) and
# over bytes rather than characters, with a pre-tokenization step we skip here.
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
    """Count frequency of each adjacent token pair, skipping the BOS delimiter.

    This skip is the whole reason the tokenizer works. BOS is a structural
    marker, not content, and it is the most frequent token in the corpus — two
    per name. Let it into the merge candidates and the top merges become
    'n'+'<BOS>', 'a'+'<BOS>', '<BOS>'+'a', ... which glues the end-of-name
    marker onto letters. A standalone BOS then almost never survives encoding,
    the model can no longer represent "the name ends here", and generation runs
    forever. Real tokenizers (GPT-2, LLaMA) exclude special tokens from merges
    for exactly this reason.
    """
    counts = {}
    for seq in corpus:
        for i in range(len(seq) - 1):
            if seq[i] == BPE_BOS or seq[i + 1] == BPE_BOS:
                continue  # never merge across or into a special token
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

# The text each token id stands for. We fill this in as merges are learned:
# a merged token's text is just the concatenation of its two components, and
# both are already in the table. Decoding is then a dict lookup. Searching the
# merge table on every decode instead would cost O(vocab) per token, which is
# the difference between this lab running in seconds and running in minutes.
token_text = dict(id_to_char)  # 0..25 -> 'a'..'z'
token_text[BPE_BOS] = ""  # BOS is a delimiter — it decodes to no characters


def bpe_decode_token(tid):
    """Decode a single BPE token id to its string representation."""
    return token_text[tid]


def bpe_display_token(tid):
    """Decode a BPE token for display, keeping BOS visible as '|'."""
    return "|" if tid == BPE_BOS else token_text[tid]


def token_str(tid):
    """Human-readable token string for display."""
    return "<BOS>" if tid == BPE_BOS else token_text[tid]


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
    token_text[new_id] = token_text[best_pair[0]] + token_text[best_pair[1]]

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
    # Apply merges in the order they were learned (greedy, left-to-right).
    # A merge can only fire if both of its components are currently present, so
    # check that first: a six-letter name skips almost all 200 merge rules and
    # encoding the corpus gets several times cheaper for free.
    for pair, new_id in merges.items():
        if pair[0] in tokens and pair[1] in tokens:
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

# The BOS delimiters are shown as '|' but excluded from the counts: they are
# framing, not content, and counting them would flatter the compression number.
# For the character tokenizer, content tokens == characters, so "Chars" is also
# the char-level token count.
print("\nBOS delimiters are shown as '|' but excluded from the token counts.\n")
print(f"{'Name':<14} {'Chars':>7} {'BPE tokens':>10}  {'Chars/token':>11}  BPE encoding")
print("-" * 75)
for name in example_names:
    bpe_toks = bpe_encode(name)
    n_content = len(bpe_toks) - 2  # drop the two BOS markers
    ratio = len(name) / n_content if n_content else 0
    bpe_visual = [f"'{bpe_display_token(t)}'" for t in bpe_toks]
    print(f"{name:<14} {len(name):>7} {n_content:>10}  {ratio:>10.2f}x  {' '.join(bpe_visual)}")


# ---------------------------------------------------------------------------
# Corpus-wide compression statistics
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("COMPRESSION STATISTICS (full corpus)")
print(f"{'=' * 60}")

# Encode the whole corpus once with each tokenizer and reuse it below for the
# bigram models — encoding is the expensive part of this lab, so do it once.
char_corpus = [char_encode(name) for name in docs]
bpe_corpus = [bpe_encode(name) for name in docs]

total_chars = sum(len(name) for name in docs)
total_char_tokens = sum(len(seq) for seq in char_corpus)
total_bpe_tokens = sum(len(seq) for seq in bpe_corpus)
# Content tokens exclude the two BOS delimiters every name carries. Compression
# is a statement about the text, so the delimiters must not be counted as text.
char_content_tokens = total_char_tokens - 2 * len(docs)
bpe_content_tokens = total_bpe_tokens - 2 * len(docs)
avg_char = total_char_tokens / len(docs)
avg_bpe = total_bpe_tokens / len(docs)
chars_per_bpe_token = total_chars / bpe_content_tokens

print(f"Total names:              {len(docs):>10}")
print(f"Total characters:         {total_chars:>10}")
print(f"Char vocab size:          {char_vocab_size:>10}")
print(f"BPE vocab size:           {vocab_size:>10}")
print("\nRaw token counts (each name also carries 2 BOS delimiters):")
print(f"  Total char tokens:      {total_char_tokens:>10}")
print(f"  Total BPE tokens:       {total_bpe_tokens:>10}")
print(f"  Avg tokens/name char:   {avg_char:>10.2f}")
print(f"  Avg tokens/name BPE:    {avg_bpe:>10.2f}")
print("\nContent tokens (BOS delimiters excluded):")
print(f"  Char content tokens:    {char_content_tokens:>10}")
print(f"  BPE content tokens:     {bpe_content_tokens:>10}")
print("\nCharacters per token — the honest compression number:")
print(f"  Char-level:             {total_chars / char_content_tokens:>10.2f}x  (one token per character, by definition)")
print(f"  BPE:                    {chars_per_bpe_token:>10.2f}x  ({NUM_MERGES} merges over a {vocab_size}-token vocab)")

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
# how tokenization affects sequence modeling. With BPE each "step" covers more
# characters, so a fixed one-step-of-history model reaches further back into
# the string — but it also has to choose from a 227-token vocabulary using
# counts collected from the same 32K names, so the table is far sparser. Which
# effect wins is an empirical question, and the numbers below answer it.
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


# HOW TO COMPARE TWO TOKENIZERS — and how not to.
#
# The usual language-model metric is perplexity per token. Across two different
# tokenizations it is meaningless. The two models are answering different
# questions: "which of 27 letters comes next?" versus "which of 227 chunks
# comes next?". A coarser vocabulary predicts rarer events, so its per-token
# perplexity is larger even when it describes the text better. The denominators
# do not match, so the ratio says nothing.
#
# Bits per character fixes this. Both models assign a probability to the *same*
# string, so take the total number of bits each one needs to encode the corpus
# and divide by the number of characters in the corpus. Same numerator units,
# same denominator, fair comparison — and directly interpretable as "how many
# bits of surprise per letter". This is why papers comparing models with
# different tokenizers report bits-per-character or bits-per-byte.
def bigram_bits_per_char(probs, encoded_corpus, n_chars):
    """Total bits the model needs for the corpus, divided by its character count."""
    total_bits = 0.0
    for seq in encoded_corpus:
        for i in range(len(seq) - 1):
            total_bits -= math.log2(probs[seq[i]][seq[i + 1]])
    return total_bits / n_chars


def bigram_log_likelihood(probs, encoded_corpus):
    """Average log-likelihood per token (shown only to make the trap visible)."""
    total_ll = 0.0
    total_tokens = 0
    for seq in encoded_corpus:
        for i in range(len(seq) - 1):
            total_ll += math.log(probs[seq[i]][seq[i + 1]])
            total_tokens += 1
    return total_ll / total_tokens if total_tokens > 0 else 0.0


# Train character-level bigram (char_corpus was built in the section above)
print("\nTraining character-level bigram...")
char_bigram = train_bigram(char_corpus, char_vocab_size)
char_bpc = bigram_bits_per_char(char_bigram, char_corpus, total_chars)
char_ll = bigram_log_likelihood(char_bigram, char_corpus)
print(f"  Vocab size: {char_vocab_size}")
print(f"  Bits per character:  {char_bpc:.4f}   <- comparable across tokenizers")
print(f"  Perplexity per token: {math.exp(-char_ll):.2f}   (NOT comparable — different vocab)")

# Train BPE bigram
print("\nTraining BPE bigram...")
bpe_bigram = train_bigram(bpe_corpus, vocab_size)
bpe_bpc = bigram_bits_per_char(bpe_bigram, bpe_corpus, total_chars)
bpe_ll = bigram_log_likelihood(bpe_bigram, bpe_corpus)
print(f"  Vocab size: {vocab_size}")
print(f"  Bits per character:  {bpe_bpc:.4f}   <- comparable across tokenizers")
print(f"  Perplexity per token: {math.exp(-bpe_ll):.2f}   (NOT comparable — different vocab)")

better = "BPE" if bpe_bpc < char_bpc else "character-level"
print(f"\n  On bits per character, {better} wins ({min(char_bpc, bpe_bpc):.4f} vs {max(char_bpc, bpe_bpc):.4f}).")
print("  Note how the per-token perplexities point the other way. That column is a trap:")
print("  a BPE token covers more characters, so it is a harder thing to predict, and its")
print("  perplexity is bigger regardless of whether the model is any good.")

# ---------------------------------------------------------------------------
# Generate names from both models
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("GENERATED NAMES")
print(f"{'=' * 60}")

# Both samplers stop when they emit a standalone BOS. That only works because
# BOS was kept out of the merges — otherwise the "name ends here" token would
# have been absorbed into letter pairs and every sample would run to max_len.
print(f"\nCharacter-level bigram ({char_vocab_size} tokens, {avg_char:.1f} tokens/name incl. BOS):")
for i in range(10):
    toks = sample_bigram(char_bigram, CHAR_BOS)
    name = char_decode(toks)
    print(f"  sample {i + 1:>2}: {name}")

print(f"\nBPE bigram ({vocab_size} tokens, {avg_bpe:.1f} tokens/name incl. BOS):")
print("  (a bigram is a one-step-of-history model either way, so neither list is great;")
print("   the point is that both terminate and produce name-shaped strings)")
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
{NUM_MERGES} merge rules turn {total_chars} characters into {bpe_content_tokens} content tokens:
{chars_per_bpe_token:.2f} characters per token. Vocab grows from {char_vocab_size} to {vocab_size} tokens, so the
embedding table gets bigger while every sequence gets shorter. That is the
whole tradeoff, and it is why GPT-scale models spend 50K-100K vocab slots.

Quality, measured in bits per character (the only figure comparable across
tokenizations): character-level {char_bpc:.2f}, BPE {bpe_bpc:.2f}. BPE really does model the
text better here — one step of BPE history spans {chars_per_bpe_token:.2f} characters, so a
one-step model effectively sees further back, and that outweighs its sparser
{vocab_size}x{vocab_size} transition table estimated from the same {len(docs)} names. Per-token
perplexity claims the opposite ({math.exp(-char_ll):.1f} vs {math.exp(-bpe_ll):.1f}), which is precisely why that
metric must never be used to compare two tokenizations.

The counting-and-merging loop here is the one GPT-2 and LLaMA use. What they
add: bytes instead of characters, a pre-tokenization regex, and orders of
magnitude more merges. What they share with this lab: special tokens are never
merge candidates.
""")
