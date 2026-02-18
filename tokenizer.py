#!/usr/bin/env python3
"""BPE tokenizer for Mason transformer.

Optimized for larger vocabularies (16K+) with batch encoding.

Train:   python tokenizer.py --train
Test:    python tokenizer.py --test "Estimate 2000 LF of 8-inch sewer"
"""

import json
import re
import argparse
import time
from collections import Counter
from pathlib import Path

from config import Config

SCRIPT_DIR = Path(__file__).resolve().parent
CFG = Config()


class BPETokenizer:
    """Byte-pair encoding tokenizer with special token support."""

    def __init__(self):
        self.merges = {}          # (a, b) -> merged token
        self.vocab = {}           # token_id -> bytes/string
        self.inverse_vocab = {}   # bytes/string -> token_id
        self.special_tokens = {}  # string -> token_id
        self.inverse_special = {} # token_id -> string
        self._merge_list = []     # ordered list of merges for fast encoding
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+""",
            re.UNICODE
        )

    # ---- training ----------------------------------------------------------

    def train(self, text: str, num_merges: int):
        """Train BPE on the given text."""
        t0 = time.time()
        words = re.findall(self.pat, text)
        word_freqs = Counter(words)

        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = list(word.encode("utf-8"))

        vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        for tok in CFG.special_tokens:
            self.special_tokens[tok] = next_id
            self.inverse_special[next_id] = tok
            vocab[next_id] = tok.encode("utf-8")
            next_id += 1

        merges = {}
        merge_list = []

        for step in range(num_merges):
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                tokens = splits[word]
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            new_token = next_id

            for word in word_freqs:
                tokens = splits[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                splits[word] = new_tokens

            merges[best_pair] = new_token
            merge_list.append((best_pair[0], best_pair[1], new_token))
            a_bytes = vocab[best_pair[0]] if isinstance(vocab[best_pair[0]], bytes) else vocab[best_pair[0]]
            b_bytes = vocab[best_pair[1]] if isinstance(vocab[best_pair[1]], bytes) else vocab[best_pair[1]]
            vocab[new_token] = a_bytes + b_bytes
            next_id += 1

            if (step + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  merge {step+1}/{num_merges} ({elapsed:.1f}s)")

        self.merges = merges
        self._merge_list = merge_list
        self.vocab = vocab
        self.inverse_vocab = {
            v if isinstance(v, str) else v.decode("utf-8", errors="replace"): k
            for k, v in vocab.items()
        }
        elapsed = time.time() - t0
        print(f"Tokenizer trained: {len(vocab)} tokens ({num_merges} merges + 256 bytes + {len(self.special_tokens)} special) in {elapsed:.1f}s")

    # ---- encode / decode ---------------------------------------------------

    def _apply_merges_to_word(self, word_tokens: list[int]) -> list[int]:
        """Apply all merges to a single word's token list (ordered)."""
        for a, b, merged in self._merge_list:
            new_tokens = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == a and word_tokens[i+1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(word_tokens[i])
                    i += 1
            word_tokens = new_tokens
            if len(word_tokens) == 1:
                break
        return word_tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        tokens = []
        i = 0
        while i < len(text):
            matched = False
            for sp_tok, sp_id in sorted(self.special_tokens.items(), key=lambda x: -len(x[0])):
                if text[i:i+len(sp_tok)] == sp_tok:
                    tokens.append(sp_id)
                    i += len(sp_tok)
                    matched = True
                    break
            if matched:
                continue

            next_special = len(text)
            for sp_tok in self.special_tokens:
                pos = text.find(sp_tok, i)
                if pos != -1 and pos < next_special:
                    next_special = pos

            chunk = text[i:next_special]
            if chunk:
                words = re.findall(self.pat, chunk)
                for word in words:
                    word_tokens = list(word.encode("utf-8"))
                    word_tokens = self._apply_merges_to_word(word_tokens)
                    tokens.extend(word_tokens)
            i = next_special

        return tokens

    def encode_batch(self, texts: list[str], show_progress: bool = True) -> list[list[int]]:
        """Encode a batch of texts with optional progress reporting."""
        results = []
        total = len(texts)
        t0 = time.time()
        for idx, text in enumerate(texts):
            results.append(self.encode(text))
            if show_progress and (idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                remaining = (total - idx - 1) / rate if rate > 0 else 0
                print(f"  Encoded {idx+1}/{total} ({rate:.0f}/s, ~{remaining:.0f}s remaining)")
        return results

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        parts = []
        for tid in ids:
            if tid in self.inverse_special:
                parts.append(self.inverse_special[tid].encode("utf-8"))
            elif tid in self.vocab:
                val = self.vocab[tid]
                if isinstance(val, bytes):
                    parts.append(val)
                else:
                    parts.append(val.encode("utf-8"))
            else:
                parts.append(b"?")
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---- save / load -------------------------------------------------------

    def save(self, path: str):
        """Save tokenizer to JSON."""
        data = {
            "merges": [[int(a), int(b), int(m)] for (a, b), m in self.merges.items()],
            "special_tokens": self.special_tokens,
            "vocab_size": len(self.vocab),
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)

        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256

        self.special_tokens = {k: int(v) for k, v in data["special_tokens"].items()}
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        for tok, tid in self.special_tokens.items():
            self.vocab[tid] = tok.encode("utf-8")
            next_id = max(next_id, tid + 1)

        self.merges = {}
        self._merge_list = []
        for a, b, m in data["merges"]:
            a, b, m = int(a), int(b), int(m)
            self.merges[(a, b)] = m
            self._merge_list.append((a, b, m))
            a_bytes = self.vocab[a] if isinstance(self.vocab.get(a), bytes) else self.vocab.get(a, b"?")
            b_bytes = self.vocab[b] if isinstance(self.vocab.get(b), bytes) else self.vocab.get(b, b"?")
            self.vocab[m] = a_bytes + b_bytes

        self.inverse_vocab = {
            v if isinstance(v, str) else v.decode("utf-8", errors="replace"): k
            for k, v in self.vocab.items()
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_tokens["<|pad|>"]

    @property
    def end_id(self) -> int:
        return self.special_tokens["<|end|>"]


# ---- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train tokenizer on corpus")
    parser.add_argument("--test", type=str, help="Test encoding on a string")
    args = parser.parse_args()

    tok_path = SCRIPT_DIR / CFG.tokenizer_path

    if args.train:
        corpus_file = SCRIPT_DIR / CFG.corpus_dir / "full_corpus.txt"
        if not corpus_file.exists():
            print(f"Corpus not found at {corpus_file}. Run build_corpus.py first.")
            return

        text = corpus_file.read_text()
        print(f"Training BPE tokenizer on {len(text):,} chars with {CFG.bpe_merges} merges...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, CFG.bpe_merges)
        tokenizer.save(str(tok_path))

        # Quick sanity check
        test = "Estimate 2000 LF of 8-inch sewer in clay soil."
        ids = tokenizer.encode(test)
        decoded = tokenizer.decode(ids)
        print(f"\nSanity check:")
        print(f"  Input:   {test}")
        print(f"  Tokens:  {len(ids)} IDs")
        print(f"  Decoded: {decoded}")
        print(f"  Match:   {test == decoded}")

        # Compression ratio
        sample = text[:10000]
        sample_ids = tokenizer.encode(sample)
        print(f"\nCompression: {len(sample)} chars -> {len(sample_ids)} tokens ({len(sample)/len(sample_ids):.1f} chars/token)")

    elif args.test:
        tokenizer = BPETokenizer()
        tokenizer.load(str(tok_path))
        ids = tokenizer.encode(args.test)
        decoded = tokenizer.decode(ids)
        print(f"Input:   {args.test}")
        print(f"Tokens:  {ids}")
        print(f"Count:   {len(ids)}")
        print(f"Decoded: {decoded}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
