"""Dataset class for the Mason transformer.

Tokenizes the corpus and serves sliding-window chunks for training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path

from config import Config
from tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Sliding-window dataset over tokenized text."""

    def __init__(self, tokens: list[int], block_size: int):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def load_dataset(cfg: Config, tokenizer: BPETokenizer, split_ratio=0.9):
    """Load and tokenize the corpus, return train and val datasets."""
    corpus_path = Path(__file__).resolve().parent / cfg.corpus_dir / "full_corpus.txt"
    text = corpus_path.read_text()

    print(f"Tokenizing {len(text):,} chars...")
    tokens = tokenizer.encode(text)
    print(f"  {len(tokens):,} tokens (avg {len(text)/len(tokens):.1f} chars/token)")

    # Train/val split
    split_idx = int(len(tokens) * split_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    train_ds = TextDataset(train_tokens, cfg.block_size)
    val_ds = TextDataset(val_tokens, cfg.block_size)

    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val:   {len(val_ds):,} samples")

    return train_ds, val_ds


if __name__ == "__main__":
    cfg = Config()
    tok = BPETokenizer()
    tok.load(str(Path(__file__).resolve().parent / cfg.tokenizer_path))
    train_ds, val_ds = load_dataset(cfg, tok)
    x, y = train_ds[0]
    print(f"Sample x shape: {x.shape}, y shape: {y.shape}")
