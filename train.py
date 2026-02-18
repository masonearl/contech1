#!/usr/bin/env python3
"""Training loop for the Mason transformer (125M params).

Supports mixed precision, gradient checkpointing, and cosine LR schedule.
Designed to work on MPS (Apple Silicon), CUDA (Colab/cloud), and CPU.

Usage:
    python train.py                    # train from scratch
    python train.py --resume           # resume from latest checkpoint
    python train.py --steps 5000       # override max steps
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config
from tokenizer import BPETokenizer
from model import MasonTransformer
from data import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent


def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Device: {dev} ({name}, {mem:.1f} GB)")
        return dev
    elif torch.backends.mps.is_available():
        print("Device: mps (Apple Silicon)")
        return torch.device("mps")
    print("Device: cpu")
    return torch.device("cpu")


def get_lr(step, cfg: Config):
    """Cosine learning rate schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr_min
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.lr_min + 0.5 * (cfg.learning_rate - cfg.lr_min) * (1 + math.cos(math.pi * progress))


def get_amp_dtype(device):
    """Get the appropriate AMP dtype for the device."""
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        return torch.float16
    return torch.float32


@torch.no_grad()
def estimate_loss(model, val_loader, device, cfg, amp_dtype):
    """Estimate loss on validation set."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_steps:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.use_amp and device.type != "cpu"):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


def train(cfg: Config, resume=False):
    device = get_device()
    amp_dtype = get_amp_dtype(device)

    # Enable TF32 on Ampere+ GPUs (A100, RTX 3090+) -- free ~1.5x speedup
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg.use_amp:
        print(f"Mixed precision: {amp_dtype}")
    if cfg.grad_checkpoint:
        print("Gradient checkpointing: enabled")

    # Load tokenizer
    tok = BPETokenizer()
    tok.load(str(SCRIPT_DIR / cfg.tokenizer_path))
    cfg.vocab_size = tok.vocab_size

    # Load data
    train_ds, val_ds = load_dataset(cfg, tok)
    num_workers = 2 if device.type == "cuda" else 0
    pin_mem = device.type == "cuda"
    prefetch = 4 if num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin_mem, prefetch_factor=prefetch)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
                            num_workers=num_workers, pin_memory=pin_mem, prefetch_factor=prefetch)

    # Model
    model = MasonTransformer(cfg).to(device)

    # torch.compile() -- ~1.5-2x speedup on CUDA (PyTorch 2.0+)
    if hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # GradScaler for float16 (not needed for bfloat16 or MPS)
    use_scaler = cfg.use_amp and amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if use_scaler else None

    # Checkpointing setup
    ckpt_dir = SCRIPT_DIR / cfg.checkpoint_dir
    ckpt_dir.mkdir(exist_ok=True)

    start_step = 0
    best_val_loss = float("inf")

    if resume:
        latest_path = SCRIPT_DIR / cfg.latest_model_path
        if latest_path.exists():
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            if scaler and "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            print(f"Resumed from step {start_step}")
        else:
            print("No checkpoint found, starting fresh")

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    running_loss = 0.0
    tokens_processed = 0

    effective_batch = cfg.batch_size * cfg.grad_accum_steps
    print(f"\nTraining config:")
    print(f"  Steps: {start_step} -> {cfg.max_steps}")
    print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accum_steps} grad accum = {effective_batch} effective")
    print(f"  Block size: {cfg.block_size}")
    print(f"  Learning rate: {cfg.learning_rate} -> {cfg.lr_min}")
    print(f"  Tokens per step: {effective_batch * cfg.block_size:,}")
    print()

    for step in range(start_step, cfg.max_steps):
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Forward + backward with mixed precision
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.use_amp and device.type != "cpu"):
            _, loss = model(x, y)
            loss = loss / cfg.grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * cfg.grad_accum_steps
        tokens_processed += cfg.batch_size * cfg.block_size

        # Logging
        if (step + 1) % 50 == 0:
            avg_loss = running_loss / 50
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / elapsed
            lr = get_lr(step, cfg)
            print(
                f"step {step+1:>6d}/{cfg.max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tok_per_sec:,.0f} tok/s"
            )
            running_loss = 0.0
            tokens_processed = 0
            t0 = time.time()

        # Evaluation
        if (step + 1) % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device, cfg, amp_dtype)
            print(f"  val loss: {val_loss:.4f}")

            # Save latest
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
                "cfg": vars(cfg),
                "best_val_loss": best_val_loss,
            }
            if scaler:
                save_dict["scaler"] = scaler.state_dict()
            torch.save(save_dict, SCRIPT_DIR / cfg.latest_model_path)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "step": step + 1,
                    "cfg": vars(cfg),
                    "val_loss": val_loss,
                }, SCRIPT_DIR / cfg.best_model_path)
                print(f"  new best model saved (val_loss={val_loss:.4f})")

    # Final save
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": cfg.max_steps,
        "cfg": vars(cfg),
        "best_val_loss": best_val_loss,
    }
    if scaler:
        save_dict["scaler"] = scaler.state_dict()
    torch.save(save_dict, SCRIPT_DIR / cfg.latest_model_path)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable gradient checkpointing")
    args = parser.parse_args()

    cfg = Config()
    if args.steps:
        cfg.max_steps = args.steps
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.no_amp:
        cfg.use_amp = False
    if args.no_checkpoint:
        cfg.grad_checkpoint = False

    train(cfg, resume=args.resume)
