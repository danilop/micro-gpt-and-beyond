# Understanding LLMs by Building One: BPE Tokenizer

Byte-Pair Encoding from scratch, then a side-by-side comparison of character-level vs BPE tokenization on the names dataset. Zero dependencies, pure Python.

## Why this version exists

Every lab so far uses character-level tokenization: 26 letters + BOS (Beginning of Sequence) = 27 tokens. This works fine for short names, but real LLMs tokenize text with BPE (50K–100K tokens). This lab implements the exact algorithm behind GPT-2's, LLaMA's, and most modern tokenizers, just at names-dataset scale.

## What makes it interesting

### The BPE algorithm

BPE starts with a character vocabulary and iteratively merges the most frequent adjacent pair into a new token:

```python
for i in range(NUM_MERGES):
    counts = count_pairs(corpus)       # count all adjacent pairs
    best_pair = max(counts, key=counts.get)  # find most frequent
    corpus = merge_pair(corpus, best_pair, new_id)  # replace everywhere
```

Each merge creates one new token and shortens every sequence where that pair appeared. After 50 merges, common substrings like "an", "er", "on" become single tokens.

### Compression vs vocabulary size

The fundamental tradeoff: larger vocabulary = shorter sequences = faster training/inference, but more parameters in the embedding table. The lab measures this directly on the names corpus.

### Bigram model comparison

To show how tokenization affects sequence modeling, the lab trains a simple bigram model (just a transition probability table, no neural network) with both tokenizations. With BPE, each bigram step spans multiple characters, capturing longer-range patterns.

## What you learn here

- The BPE training algorithm (pair counting + merging), the same algorithm behind GPT-2/3/4
- BPE encoding and decoding (applying learned merges to new text)
- The vocabulary size vs sequence length tradeoff
- How tokenization choices affect downstream modeling
- Why character-level works for names but not for internet-scale text

## Run

```bash
python main.py
```

Trains BPE with 50 merges on ~32,000 names, shows the merge table, compression statistics, encoding examples, and generates names from both character-level and BPE bigram models.
