"""Training entry point for Slurm jobs.

Loads the replay buffer from disk, trains the network for N epochs,
and saves a checkpoint.

Usage:
    python -m hexzero.scripts.train \
        --replay-buffer replay_buffer/buffer.pt \
        --checkpoint-dir checkpoints/ \
        --epochs 10 \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description="HexaZero training")
    parser.add_argument("--replay-buffer", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from this checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--iteration", type=int, default=0,
                        help="Current iteration number for checkpointing")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.selfplay.replay_buffer import ReplayBuffer
    from hexzero.training.trainer import Trainer

    cfg = get_default_config()
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.lr
    cfg.training.num_epochs_per_iteration = args.epochs

    buf = ReplayBuffer.load(args.replay_buffer)
    log.info("Replay buffer: %d positions", len(buf))

    if len(buf) < args.batch_size:
        log.error("Not enough data: %d < batch_size %d", len(buf), args.batch_size)
        return

    if args.resume and Path(args.resume).exists():
        log.info("Resuming from %s", args.resume)
        net = HexaZeroNet.load_checkpoint(args.resume, device=device)
        trainer = Trainer(net, cfg.training, device=device)
    else:
        net = HexaZeroNet(cfg.network)
        trainer = Trainer(net, cfg.training, device=device)

    t0 = time.time()
    metrics = trainer.train_iteration(buf, num_epochs=args.epochs)
    elapsed = time.time() - t0

    log.info(
        "Training done: loss=%.4f vloss=%.4f ploss=%.4f vacc=%.3f (%.1fs)",
        metrics["total_loss"],
        metrics["value_loss"],
        metrics["policy_loss"],
        metrics["value_accuracy"],
        elapsed,
    )

    ckpt_path = ckpt_dir / f"iteration_{args.iteration:06d}.pt"
    latest_path = ckpt_dir / "latest.pt"

    trainer.save_checkpoint(str(ckpt_path), args.iteration)
    trainer.save_checkpoint(str(latest_path), args.iteration)
    log.info("Saved checkpoints: %s, %s", ckpt_path, latest_path)


if __name__ == "__main__":
    main()
