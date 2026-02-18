"""Mason Transformer -- hyperparameters and configuration.

~125M parameter GPT-2 Small equivalent with modern architecture choices.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Model architecture (125M params)
    n_layers: int = 12
    n_heads: int = 12
    n_embd: int = 768
    block_size: int = 2048          # context window (max sequence length)
    vocab_size: int = 16384         # set after tokenizer training
    dropout: float = 0.1
    use_flash_attn: bool = True     # use scaled_dot_product_attention when available

    # Training
    batch_size: int = 4
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    max_steps: int = 20000
    warmup_steps: int = 500
    eval_interval: int = 500
    eval_steps: int = 20
    grad_accum_steps: int = 16      # effective batch = batch_size * grad_accum_steps = 64
    lr_min: float = 6e-5
    use_amp: bool = True            # mixed precision training
    grad_checkpoint: bool = True    # gradient checkpointing for memory savings

    # Tokenizer
    bpe_merges: int = 16000
    special_tokens: tuple = (
        "<|pad|>",
        "<|end|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool_call|>",
        "<|tool_result|>",
    )

    # Paths (relative to transformer/ directory)
    corpus_dir: str = "corpus"
    tokenizer_path: str = "tokenizer.json"
    checkpoint_dir: str = "checkpoints"
    best_model_path: str = "checkpoints/best.pt"
    latest_model_path: str = "checkpoints/latest.pt"
    data_dir: str = "../../data"   # points to estimator/data/
