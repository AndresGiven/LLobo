import os
import sys
import time
import math
from dataclasses import asdict

import torch
import torch.nn as nn

from config import Hyperparameters
from data import create_dataloader
from data import load_training_tokens
from model import Model


if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8")


def _resolve_device(configured_device):
    if configured_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    return configured_device


def _build_optimizer(model, lr):
    try:
        return torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(model.parameters(), lr=lr)


def _build_scheduler(optimizer, params, total_steps):
    if params.scheduler_type == "none":
        return None
    if params.scheduler_type != "cosine":
        raise ValueError(f"Unsupported scheduler_type: {params.scheduler_type}")

    warmup_steps = max(0, min(params.warmup_steps, max(0, total_steps - 1)))

    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        decay_steps = total_steps - warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        min_lr_ratio = params.min_learn_rate / params.learn_rate
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def print_system_info(loader_len):
    print(f"1 epoch = {loader_len} batches")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN available?: {torch.backends.cudnn.is_available()}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
        print(
            f"Built in FlashAttention?: {torch.backends.cuda.is_flash_attention_available()}"
        )
        print(f"FlashAttention enabled? {torch.backends.cuda.flash_sdp_enabled()}")
        print(
            f"Memory effecient flashAttention enabled? {torch.backends.cuda.mem_efficient_sdp_enabled()}"
        )
        print(
            f"fp16/bf16 enabled?: {torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()}"
        )


def train(params=None):
    if params is None:
        params = Hyperparameters()

    params.device = _resolve_device(params.device)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)

    training_data = load_training_tokens(params.data_path, params.device)
    loader = create_dataloader(
        training_data, batch_size=params.batch_size, context_size=params.context_size
    )

    model = Model(params).to(params.device)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
    print_system_info(len(loader))

    optimizer = _build_optimizer(model, params.learn_rate)
    total_steps = len(loader) * params.epochs
    scheduler = _build_scheduler(optimizer, params, total_steps)
    os.makedirs(params.checkpoint_dir, exist_ok=True)

    for epoch in range(params.epochs):
        for index, (inputs, targets) in enumerate(loader):
            t0 = time.time()
            optimizer.zero_grad()
            inputs = inputs.to(params.device)
            targets = targets.to(params.device)

            with torch.autocast(device_type=params.device, dtype=torch.bfloat16):
                _, loss = model(inputs, targets)

            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if params.device == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            if index % params.log_every == 0:
                dt = (t1 - t0) * 1000
                tok_per_sec = (params.batch_size * params.context_size) / (t1 - t0)
                curr_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch {epoch}, iter {index}, norm {norm:.4f}, "
                    f"lr: {curr_lr:.6e}, loss: {loss.item():.4f}, "
                    f"time: {dt:.2f}ms, tok/sec: {tok_per_sec:.2f}"
                )

            should_save = (
                (index > 0 and index % params.save_every == 0)
                or (index == len(loader) - 1)
            )
            if should_save:
                ckpt_path = os.path.join(
                    params.checkpoint_dir, f"model8b_i{index}_l{loss.item():.4f}.pth"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "batch": index,
                        "config": asdict(params),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()
                        if scheduler is not None
                        else None,
                        "loss": loss,
                    },
                    ckpt_path,
                )


if __name__ == "__main__":
    train()
