"""Mason Transformer -- decoder-only GPT-style architecture (125M params).

Modern architecture with RoPE, RMSNorm, SwiGLU, and flash attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from config import Config


# ---------------------------------------------------------------------------
# RMSNorm (faster than LayerNorm, used in LLaMA / Mistral)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the complex exponentials for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embedding to query or key tensor.

    x: (B, n_heads, T, head_dim)
    cos, sin: (T, head_dim // 2)
    """
    B, H, T, D = x.shape
    half = D // 2

    cos = cos[:T, :half].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    sin = sin[:T, :half].unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    return torch.cat([out1, out2], dim=-1)


# ---------------------------------------------------------------------------
# Attention with RoPE and optional flash attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.use_flash = cfg.use_flash_attn

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(self.head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Causal mask fallback (only used when flash attention is unavailable)
        if not self.use_flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                     .view(1, 1, cfg.block_size, cfg.block_size)
            )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Attention
        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_drop(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward (used in LLaMA / PaLM)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward: gate * swish(linear(x)) then project down."""

    def __init__(self, cfg: Config):
        super().__init__()
        hidden = int(4 * cfg.n_embd * 2 / 3)  # SwiGLU convention
        # Round up to nearest multiple of 64 for GPU efficiency
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm, RoPE attention, SwiGLU FFN."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Mason Transformer
# ---------------------------------------------------------------------------

class MasonTransformer(nn.Module):
    """125M-param decoder-only transformer for Mason's personal AI."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpoint = cfg.grad_checkpoint

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # No learned positional embedding -- RoPE handles positions in attention
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # Scale residual projections per GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        print(f"MasonTransformer: {self.count_params()/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) token indices
        targets: (B, T) target indices for loss computation (optional)
        Returns: logits (B, T, vocab_size), loss (scalar or None)
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"Sequence length {T} > block_size {self.cfg.block_size}"

        x = self.drop(self.tok_emb(idx))

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss


# ---- quick test ------------------------------------------------------------

if __name__ == "__main__":
    cfg = Config()
    model = MasonTransformer(cfg)

    # Dummy forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")

    # With targets
    targets = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, targets)
    print(f"Loss: {loss.item():.4f}")
