# Understanding LLMs by Building One: BPE Tokenizer

Byte-Pair Encoding from scratch, then a side-by-side comparison of character-level vs BPE tokenization on the names dataset. Zero dependencies, pure Python.

## Why this version exists

Every lab so far uses character-level tokenization: 26 letters + BOS (Beginning of Sequence) = 27 tokens. This works fine for short names, but real LLMs tokenize text with BPE (50K–100K tokens). This lab implements the counting-and-merging loop at the heart of GPT-2's and LLaMA's tokenizers, at names-dataset scale.

It is not equivalent to a production tokenizer, and it is worth being precise about the gap. GPT-2 runs BPE over *bytes*, so it can encode any input including emoji and other scripts; this lab runs it over the 26 lowercase characters in the dataset. GPT-2 also applies a pre-tokenization regex first, so merges never cross word boundaries; this lab has no such step because a name is one word. And GPT-2 learns ~50K merges against this lab's 200. What does carry over unchanged: the merge loop, the greedy encoder, and the rule that special tokens are never merge candidates.

## What makes it interesting

### The BPE algorithm

BPE starts with a character vocabulary and iteratively merges the most frequent adjacent pair into a new token:

```python
for i in range(NUM_MERGES):
    counts = count_pairs(corpus)  # count all adjacent pairs
    best_pair = max(counts, key=counts.get)  # find most frequent
    corpus = merge_pair(corpus, best_pair, new_id)  # replace everywhere
```

Each merge creates one new token and shortens every sequence where that pair appeared. The lab runs 200 merges, and the table it prints is readable end to end: `a`+`n`, `a`+`r`, `e`+`l`, `l`+`e` early on, then `el`+`la`, `lei`+`gh`, `st`+`on` once the earlier merges have something to build on.

### Special tokens are not merge candidates

`count_pairs` skips any pair touching BOS, and that one filter is what makes the rest of the lab work. BOS is the most frequent token in the corpus, at two per name, so leaving it in the candidate set makes the top merges `n`+`<BOS>`, `a`+`<BOS>`, `<BOS>`+`a` and so on. The end-of-name marker gets glued onto letters, a standalone BOS almost never survives encoding, and the model loses its only way to say "the name ends here": every sample then runs to the length cap and comes out as a 30-character run-on. Real tokenizers exclude special tokens from merges for exactly this reason.

### Compression, measured honestly

The headline number is **characters per token**, computed with the BOS delimiters excluded. Counting the two markers per name as content would inflate the figure, because they are framing rather than text. The lab prints the raw token counts too, clearly labelled, so you can see the difference.

### Bigram model comparison, and how to compare two tokenizers

To show how tokenization affects sequence modeling, the lab trains a simple bigram model (a transition probability table, no neural network) with both tokenizations. With BPE, each bigram step spans more than one character, so a one-step-of-history model effectively reaches further back into the string. Against that, the BPE table is 227x227 estimated from the same 32,033 names, so it is much sparser. The lab measures which effect wins.

The measurement is **bits per character**, and the choice of metric is the real lesson. Per-token perplexity, the usual language-model number, is not comparable across two different tokenizations. The two models are answering different questions: "which of 27 letters comes next?" versus "which of 227 chunks comes next?". A coarser vocabulary predicts rarer events, so its per-token perplexity is larger even when it describes the text better. The denominators do not match.

Bits per character shares a denominator. Both models assign a probability to the same underlying string, so divide the total bits each needs for the corpus by the corpus's character count. The lab prints both figures and they disagree in direction, which is the point: this is why papers comparing models with different tokenizers report bits-per-character or bits-per-byte.

## What you learn here

- The BPE training algorithm (pair counting + merging), and why special tokens must be held out of it
- BPE encoding and decoding (applying learned merges to new text)
- The vocabulary size vs sequence length tradeoff, measured as characters per token
- Why per-token perplexity cannot compare two tokenizations, and what to use instead
- Why character-level works for names but not for internet-scale text

## Run

```bash
python main.py
```

Trains BPE with 200 merges on ~32,000 names, shows the merge table, compression statistics, encoding examples, bits-per-character for both tokenizations, and generates names from both character-level and BPE bigram models.
