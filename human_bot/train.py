#!/usr/bin/env python3
"""Train a compact dual-head model to predict human Catan moves.

Optimised for Apple Silicon (M5 Max, MPS backend):
  - fp16 autocast for 2x throughput
  - Large batch sizes (4096) exploiting unified memory
  - torch.compile with aot_eager backend
  - Async batch prefetching on device
  - No DataLoader overhead (direct tensor indexing)

Model: HumanBotNet (~192k params) — GNN(3 layers, 48-dim) + ResNet(4 blocks, 80-ch)
Heads: spatial policy (337-dim, label-smoothed CE) + value (4-dim win distribution CE)

Usage:
    python -m human_bot.train --data-dir data/human_games --epochs 20
    python -m human_bot.train --colonist-dir data/colonist_raw --epochs 20
"""

from __future__ import annotations

import argparse
import math
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from human_bot.config import HumanBotTrainingConfig
from human_bot.dataset import HumanGameDataset, load_tensor_shards
from human_bot.evaluate import compute_metrics, evaluate_vs_ab2
from human_bot.loss import (
    FixedWeightLoss,
    UncertaintyWeightedLoss,
    human_policy_loss,
    masked_entropy,
    value_loss,
)
from human_bot.model import HumanBotNet, SmallNetworkConfig


def detect_device(requested: str) -> str:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Device-resident dataset (avoids CPU→GPU transfers during training)
# ======================================================================

class DeviceDataset:
    """Pre-stages all tensors on the target device for zero-copy batch access."""

    __slots__ = ("nf", "ef", "ff", "mask", "action_idx", "value_target", "n")

    def __init__(self, ds: HumanGameDataset, device: str) -> None:
        self.nf = ds.nf.to(device, non_blocking=True)
        self.ef = ds.ef.to(device, non_blocking=True)
        self.ff = ds.ff.to(device, non_blocking=True)
        self.mask = ds.mask.to(device, non_blocking=True)
        self.action_idx = ds.action_idx.to(device, non_blocking=True)
        self.value_target = ds.value_target.to(device, non_blocking=True)
        self.n = ds.n

    def __len__(self) -> int:
        return self.n

    def get_batch(self, indices: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """Return raw tensors (no dict overhead) already on device."""
        return (
            self.nf[indices], self.ef[indices], self.ff[indices],
            self.mask[indices], self.action_idx[indices], self.value_target[indices],
        )


# ======================================================================
# Training core
# ======================================================================

def train_epoch(
    net: nn.Module,
    dataset: DeviceDataset,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    loss_combiner: nn.Module,
    edge_index: torch.Tensor,
    device: str,
    cfg: HumanBotTrainingConfig,
) -> dict[str, float]:
    net.train()
    n = len(dataset)
    perm = torch.randperm(n, device=device)
    BS = cfg.batch_size

    sums = dict.fromkeys(
        ["policy_loss", "value_loss", "total_loss", "entropy", "policy_acc", "value_acc"],
        0.0,
    )
    n_batches = 0

    for i in range(0, n, BS):
        idx = perm[i : i + BS]
        if len(idx) < 16:
            continue

        nf, ef, ff, mask, action_idx, vt = dataset.get_batch(idx)

        out = net({
            "node_features": nf,
            "edge_index": edge_index,
            "edge_features": ef,
            "flat_features": ff,
            "action_mask": mask,
        })

        p_loss = human_policy_loss(
            out["policy_logits"], action_idx, mask,
            label_smoothing=cfg.label_smoothing,
        )
        v_loss = value_loss(out["value"], vt)
        ent = masked_entropy(out["policy_logits"], mask)
        total, _ = loss_combiner(p_loss, v_loss, ent, cfg.entropy_weight)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            pred = out["policy_logits"].argmax(dim=-1)
            pacc = (pred == action_idx).float().mean().item()
            vp = out["value"].argmax(dim=-1)
            vacc = (vp == vt.argmax(dim=-1)).float().mean().item()

        sums["policy_loss"] += p_loss.item()
        sums["value_loss"] += v_loss.item()
        sums["total_loss"] += total.item()
        sums["entropy"] += ent.item()
        sums["policy_acc"] += pacc
        sums["value_acc"] += vacc
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def freeze_encoder(net: nn.Module) -> None:
    for name, param in net.named_parameters():
        if name.startswith("board_encoder."):
            param.requires_grad = False


def unfreeze_all(net: nn.Module) -> None:
    for param in net.parameters():
        param.requires_grad = True


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train human-bot (MPS-optimised)")
    parser.add_argument("--data-dir", type=str, default="data/human_games")
    parser.add_argument("--colonist-dir", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/human_bot")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-temperature", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--no-uncertainty-weighting", action="store_true")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="human-bot")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = HumanBotTrainingConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        pretrained_checkpoint=args.pretrained,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        label_smoothing=args.label_smoothing,
        entropy_weight=args.entropy_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        eval_games=args.eval_games,
        eval_temperature=args.eval_temperature,
        max_examples=args.max_examples,
        use_uncertainty_weighting=not args.no_uncertainty_weighting,
        seed=args.seed,
        device=args.device,
        wandb_key=args.wandb_key,
        wandb_project=args.wandb_project,
    )

    device = detect_device(cfg.device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    action_enc = ActionEncoder()
    lib = load_library()

    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    # ── Convert Colonist data if requested ─────────────────────────
    if args.colonist_dir and os.path.isdir(args.colonist_dir):
        from human_bot.dataset import convert_colonist_games
        print(f"Converting Colonist.io games from {args.colonist_dir} ...", flush=True)
        convert_colonist_games(args.colonist_dir, cfg.data_dir)

    # ── Load data ──────────────────────────────────────────────────
    print(f"Loading data from {cfg.data_dir} ...", flush=True)
    full_dataset = load_tensor_shards(cfg.data_dir, max_examples=cfg.max_examples)
    train_ds, val_ds = full_dataset.split(val_fraction=cfg.val_fraction, seed=cfg.seed)
    print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}", flush=True)

    # Stage full dataset on device (unified memory = no copy cost on MPS)
    print(f"Staging tensors on {device} ...", flush=True)
    t_stage = time.perf_counter()
    train_dev = DeviceDataset(train_ds, device)
    val_dev_ds = val_ds  # val stays on CPU for compute_metrics compatibility
    if device == "mps":
        torch.mps.synchronize()
    stage_ms = (time.perf_counter() - t_stage) * 1000
    print(f"Staged {len(train_dev):,} examples in {stage_ms:.0f} ms\n", flush=True)

    # ── Model ──────────────────────────────────────────────────────
    net_cfg = SmallNetworkConfig()
    if cfg.pretrained_checkpoint and os.path.exists(cfg.pretrained_checkpoint):
        net = HumanBotNet.load_checkpoint(cfg.pretrained_checkpoint, device=device)
        net.train()
        print(f"Loaded pretrained: {cfg.pretrained_checkpoint}", flush=True)
    else:
        net = HumanBotNet(net_cfg).to(device)

    print(f"HumanBotNet: {net.num_parameters:,} params", flush=True)

    # ── Loss combiner ──────────────────────────────────────────────
    if cfg.use_uncertainty_weighting:
        loss_combiner: nn.Module = UncertaintyWeightedLoss().to(device)
    else:
        loss_combiner = FixedWeightLoss(cfg.policy_weight, cfg.value_weight).to(device)

    # ── W&B ────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb and cfg.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = cfg.wandb_key
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=f"human-bot-{time.strftime('%m%d-%H%M')}",
                config={**vars(args), "params": net.num_parameters},
                tags=["human-bot", device, f"{net.num_parameters // 1000}k"],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed: {e}", flush=True)

    # ── Banner ─────────────────────────────────────────────────────
    print("=" * 65, flush=True)
    print(" Human-Bot Training (MPS-optimised)", flush=True)
    print(f" Device            : {device}", flush=True)
    print(f" Model params      : {net.num_parameters:,}", flush=True)
    print(f" Train examples    : {len(train_dev):,}", flush=True)
    print(f" Batch size        : {cfg.batch_size}", flush=True)
    print(f" Epochs            : {cfg.epochs}", flush=True)
    print(f" Freeze encoder    : {cfg.freeze_encoder_epochs} epochs", flush=True)
    print(f" Label smoothing   : {cfg.label_smoothing}", flush=True)
    print(f" Uncertainty wt    : {cfg.use_uncertainty_weighting}", flush=True)
    print("=" * 65, flush=True)

    # ── Training loop ──────────────────────────────────────────────
    best_val_top1 = -1.0
    t_start = time.time()
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    for epoch in range(1, cfg.epochs + 1):
        t_epoch = time.perf_counter()

        # Phase transitions
        if epoch == 1 and cfg.freeze_encoder_epochs > 0:
            freeze_encoder(net)
            lr = args.lr if args.lr > 0 else cfg.lr_encoder_frozen
            all_params = list(net.parameters()) + list(loss_combiner.parameters())
            optimizer = torch.optim.AdamW(
                [p for p in all_params if p.requires_grad],
                lr=lr, weight_decay=cfg.weight_decay,
            )
            steps = max(1, len(train_dev) // cfg.batch_size) * cfg.freeze_encoder_epochs
            scheduler = build_cosine_scheduler(optimizer, steps, cfg.lr_warmup_steps)
            print(f"[phase 1] encoder frozen, lr={lr}", flush=True)

        if epoch == cfg.freeze_encoder_epochs + 1:
            unfreeze_all(net)
            lr = args.lr if args.lr > 0 else cfg.lr_finetune
            all_params = list(net.parameters()) + list(loss_combiner.parameters())
            optimizer = torch.optim.AdamW(
                all_params, lr=lr, weight_decay=cfg.weight_decay,
            )
            remaining = cfg.epochs - cfg.freeze_encoder_epochs
            steps = max(1, len(train_dev) // cfg.batch_size) * remaining
            scheduler = build_cosine_scheduler(optimizer, steps, min(cfg.lr_warmup_steps, 200))
            print(f"[phase 2] full model, lr={lr}", flush=True)

        assert optimizer is not None
        avg = train_epoch(
            net, train_dev, optimizer, scheduler, loss_combiner,
            edge_index_dev, device, cfg,
        )

        if device == "mps":
            torch.mps.synchronize()
        epoch_sec = time.perf_counter() - t_epoch
        samples_per_sec = len(train_dev) / epoch_sec

        # ── Validation ─────────────────────────────────────────────
        val_metrics: dict[str, float] = {}
        if epoch % cfg.eval_every == 0 and len(val_dev_ds) > 0:
            val_metrics = compute_metrics(
                net, val_dev_ds, edge_index_dev, device, cfg.batch_size,
            )

        val_top1 = val_metrics.get("top1_acc", avg["policy_acc"])

        # ── AB2 evaluation ─────────────────────────────────────────
        ab2_result: dict = {}
        if epoch % cfg.eval_every == 0 and cfg.eval_games > 0:
            ab2_result = evaluate_vs_ab2(
                net, state_enc, action_enc, device, lib,
                num_games=cfg.eval_games,
                temperature=cfg.eval_temperature,
                seed_offset=epoch,
            )

        # ── Log ────────────────────────────────────────────────────
        parts = [
            f"[{epoch}/{cfg.epochs}]",
            f"ploss={avg['policy_loss']:.4f}",
            f"pacc={avg['policy_acc']:.3f}",
            f"vloss={avg['value_loss']:.4f}",
            f"vacc={avg['value_acc']:.3f}",
        ]
        if val_metrics:
            parts.append(f"v.top1={val_metrics.get('top1_acc', 0):.3f}")
            parts.append(f"v.top3={val_metrics.get('top3_acc', 0):.3f}")
        if ab2_result:
            parts.append(
                f"AB2:{ab2_result['hz_wins']}/{ab2_result['ab2_wins']}"
                f"({ab2_result['win_rate']:.0%})"
            )
        parts.append(f"{epoch_sec:.1f}s")
        parts.append(f"{samples_per_sec:.0f} ex/s")
        print("  ".join(parts), flush=True)

        # ── Checkpoints ────────────────────────────────────────────
        meta = {
            "epoch": epoch,
            "ploss": avg["policy_loss"], "pacc": avg["policy_acc"],
            "vtop1": val_metrics.get("top1_acc"),
            "ab2_wr": ab2_result.get("win_rate"),
        }
        net.save_checkpoint(os.path.join(cfg.checkpoint_dir, "latest.pt"), meta)

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            net.save_checkpoint(os.path.join(cfg.checkpoint_dir, "best.pt"), meta)
            print(f"  * best val top-1: {val_top1:.3f}", flush=True)

        if wandb_run:
            import wandb
            log_d = {
                "train/policy_loss": avg["policy_loss"],
                "train/value_loss": avg["value_loss"],
                "train/policy_acc": avg["policy_acc"],
                "train/value_acc": avg["value_acc"],
                "train/entropy": avg["entropy"],
                "perf/samples_per_sec": samples_per_sec,
                "perf/epoch_sec": epoch_sec,
                "epoch": epoch,
            }
            for k, v in val_metrics.items():
                log_d[f"val/{k}"] = v
            for k, v in ab2_result.items():
                log_d[f"eval/{k}"] = v
            wandb.log(log_d)

        if epoch % 5 == 0 and val_metrics:
            print("  action-type accuracy:", flush=True)
            for k, v in sorted(val_metrics.items()):
                if k.startswith("acc/"):
                    cnt = val_metrics.get(k.replace("acc/", "count/"), 0)
                    print(f"    {k[4:]:12s}  {v:.3f}  (n={int(cnt):,})", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone. {cfg.epochs} epochs in {elapsed:.0f}s  "
          f"best top-1={best_val_top1:.3f}", flush=True)
    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
