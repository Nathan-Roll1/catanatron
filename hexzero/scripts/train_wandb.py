"""Training with Weights & Biases logging.

Full-featured training entry point that logs all metrics, gradients,
learning rates, and model stats to W&B in real time.

Usage:
    python -m hexzero.scripts.train_wandb \
        --replay-buffer replay_buffer/buffer.pt \
        --checkpoint-dir checkpoints/ \
        --epochs 20 \
        --wandb-project hexazero \
        --wandb-name "iter0-jagupard10" \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train_wandb")


def main() -> None:
    parser = argparse.ArgumentParser(description="HexaZero training with W&B")
    parser.add_argument("--replay-buffer", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="hexazero")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Imports (deferred so --help is fast) ──────────────────────────
    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.selfplay.replay_buffer import ReplayBuffer
    from hexzero.training.loss import HexaZeroLoss

    cfg = get_default_config()
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.lr
    cfg.training.num_epochs_per_iteration = args.epochs

    # ── Load data ─────────────────────────────────────────────────────
    buf = ReplayBuffer.load(args.replay_buffer)
    log.info("Replay buffer: %d positions", len(buf))

    if len(buf) < args.batch_size:
        log.error("Buffer too small: %d < batch_size %d. Run more self-play first.",
                  len(buf), args.batch_size)
        return

    # ── Build / load model ────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        log.info("Resuming from %s", args.resume)
        net = HexaZeroNet.load_checkpoint(args.resume, device=device)
    else:
        net = HexaZeroNet(cfg.network)
    net.to(device)

    # ── W&B init ──────────────────────────────────────────────────────
    import wandb

    gpu_name = "cpu"
    gpu_mem = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9

    run_name = args.wandb_name or f"iter{args.iteration}-{os.uname().nodename}"

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "iteration": args.iteration,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "buffer_size": len(buf),
            "model_params": net.num_parameters,
            "gpu": gpu_name,
            "gpu_memory_gb": round(gpu_mem, 1),
            "hostname": os.uname().nodename,
            "network": asdict(cfg.network),
            "mcts": asdict(cfg.mcts),
            "training": asdict(cfg.training),
        },
        tags=["training", f"iter{args.iteration}", gpu_name.split()[0]],
    )

    wandb.watch(net, log="gradients", log_freq=50, log_graph=False)

    # ── Optimizer / scheduler / scaler ────────────────────────────────
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = HexaZeroLoss(
        value_weight=cfg.training.value_loss_weight,
        policy_weight=cfg.training.policy_loss_weight,
    )

    use_amp = device == "cuda"
    scaler = GradScaler(enabled=use_amp)
    max_grad_norm = 1.0

    num_batches_per_epoch = max(len(buf) // args.batch_size, 1)
    total_steps = args.epochs * num_batches_per_epoch
    warmup_steps = min(cfg.training.lr_warmup_steps, total_steps // 4)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training loop ─────────────────────────────────────────────────
    global_step = 0
    best_loss = float("inf")

    log.info("Training: %d epochs, %d batches/epoch, %d total steps",
             args.epochs, num_batches_per_epoch, total_steps)
    log.info("Device: %s (%s, %.1f GB)", device, gpu_name, gpu_mem)

    t_start = time.time()

    for epoch in range(args.epochs):
        net.train()
        epoch_metrics = {
            "total_loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0,
            "value_accuracy": 0.0, "policy_entropy": 0.0, "gradient_norm": 0.0,
        }

        t_epoch = time.time()

        for batch_idx in range(num_batches_per_epoch):
            batch = buf.sample(args.batch_size)

            inputs = {
                "node_features": batch.node_features.to(device, non_blocking=True),
                "edge_index": batch.edge_index.to(device, non_blocking=True),
                "edge_features": batch.edge_features.to(device, non_blocking=True),
                "flat_features": batch.flat_features.to(device, non_blocking=True),
                "action_mask": batch.action_masks.to(device, non_blocking=True),
            }
            targets = {
                "policy_targets": batch.policy_targets.to(device, non_blocking=True),
                "value_targets": batch.value_targets.to(device, non_blocking=True),
                "action_masks": batch.action_masks.to(device, non_blocking=True),
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                predictions = net(inputs)
                losses = criterion(predictions, targets)

            total_loss: Tensor = losses["total_loss"]

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                log.warning("NaN/Inf loss at step %d, skipping", global_step)
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            step_metrics = {
                "total_loss": total_loss.item(),
                "value_loss": losses["value_loss"].item(),
                "policy_loss": losses["policy_loss"].item(),
                "value_accuracy": losses["value_accuracy"].item(),
                "policy_entropy": losses["policy_entropy"].item(),
                "gradient_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
            }

            for k, v in step_metrics.items():
                epoch_metrics[k] += v

            # Log every step to W&B
            wandb.log({
                "step/total_loss": step_metrics["total_loss"],
                "step/value_loss": step_metrics["value_loss"],
                "step/policy_loss": step_metrics["policy_loss"],
                "step/value_accuracy": step_metrics["value_accuracy"],
                "step/policy_entropy": step_metrics["policy_entropy"],
                "step/gradient_norm": step_metrics["gradient_norm"],
                "step/learning_rate": optimizer.param_groups[0]["lr"],
                "step/global_step": global_step,
            }, step=global_step)

        # ── Epoch summary ─────────────────────────────────────────────
        epoch_time = time.time() - t_epoch
        n = num_batches_per_epoch
        avg = {k: v / n for k, v in epoch_metrics.items()}

        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.max_memory_allocated(0) / 1e9
            torch.cuda.reset_peak_memory_stats(0)
        else:
            gpu_mem_used = 0.0

        wandb.log({
            "epoch/total_loss": avg["total_loss"],
            "epoch/value_loss": avg["value_loss"],
            "epoch/policy_loss": avg["policy_loss"],
            "epoch/value_accuracy": avg["value_accuracy"],
            "epoch/policy_entropy": avg["policy_entropy"],
            "epoch/gradient_norm": avg["gradient_norm"],
            "epoch/learning_rate": optimizer.param_groups[0]["lr"],
            "epoch/epoch_time_s": epoch_time,
            "epoch/samples_per_sec": (n * args.batch_size) / epoch_time,
            "epoch/gpu_mem_peak_gb": gpu_mem_used,
            "epoch/epoch": epoch,
        }, step=global_step)

        log.info(
            "Epoch %2d/%d | loss=%.4f vloss=%.4f ploss=%.4f vacc=%.3f "
            "entropy=%.2f gnorm=%.3f lr=%.2e | %.1fs (%.0f samp/s) | GPU %.1fGB",
            epoch + 1, args.epochs,
            avg["total_loss"], avg["value_loss"], avg["policy_loss"],
            avg["value_accuracy"], avg["policy_entropy"],
            avg["gradient_norm"], optimizer.param_groups[0]["lr"],
            epoch_time, (n * args.batch_size) / epoch_time,
            gpu_mem_used,
        )

        # Save checkpoint every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt = {
                "iteration": args.iteration,
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": asdict(cfg.network),
                "metrics": avg,
            }

            ckpt_path = ckpt_dir / f"iter{args.iteration:04d}_epoch{epoch+1:03d}.pt"
            torch.save(ckpt, ckpt_path)

            latest_path = ckpt_dir / "latest.pt"
            net.save_checkpoint(str(latest_path), metadata={
                "iteration": args.iteration,
                "epoch": epoch + 1,
                "global_step": global_step,
                "metrics": avg,
            })

            if avg["total_loss"] < best_loss:
                best_loss = avg["total_loss"]
                best_path = ckpt_dir / "best.pt"
                net.save_checkpoint(str(best_path), metadata={
                    "iteration": args.iteration,
                    "best_loss": best_loss,
                })
                log.info("  New best loss: %.4f -> %s", best_loss, best_path)

            # Log model as W&B artifact
            artifact = wandb.Artifact(
                f"hexazero-model-iter{args.iteration}",
                type="model",
                metadata={"epoch": epoch + 1, "loss": avg["total_loss"]},
            )
            artifact.add_file(str(latest_path))
            wandb.log_artifact(artifact)

    # ── Final summary ─────────────────────────────────────────────────
    total_time = time.time() - t_start
    wandb.summary["total_training_time_s"] = total_time
    wandb.summary["total_steps"] = global_step
    wandb.summary["best_loss"] = best_loss
    wandb.summary["final_value_accuracy"] = avg["value_accuracy"]

    log.info("Training complete: %d steps in %.1fs (%.1f steps/s)",
             global_step, total_time, global_step / total_time)
    log.info("Best loss: %.4f", best_loss)
    log.info("W&B run: %s", wandb.run.url)

    wandb.finish()


if __name__ == "__main__":
    main()
