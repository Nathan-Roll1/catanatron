"""Incremental training on human games with cached dataset.

Loads ALL shards once, caches the preprocessed tensors.  Each run
trains on the next N unseen games (tracked via a cursor file).

Usage:
    python -m human_bot.run_experiment                  # default 20k new games
    python -m human_bot.run_experiment --games 5000     # 5k new games
    python -m human_bot.run_experiment --reset          # start from scratch
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn

from human_bot.config import HumanBotTrainingConfig
from human_bot.dataset import HumanGameDataset, fix_action_masks, load_tensor_shards
from human_bot.eval_search import evaluate_search_vs_ab2
from human_bot.loss import UncertaintyWeightedLoss
from human_bot.model import HumanBotNet, SmallNetworkConfig
from human_bot.train import DeviceDataset, build_cosine_scheduler, train_epoch

DATA_DIR = "data/human_games"
CKPT_DIR = "checkpoints/human_bot_experiment"
CACHE_PATH = os.path.join(CKPT_DIR, "all_data_cache.pt")
CURSOR_PATH = os.path.join(CKPT_DIR, "cursor.json")
MODEL_PATH = os.path.join(CKPT_DIR, "latest.pt")
EXAMPLES_PER_GAME = 269


def load_or_cache_all(device: str) -> HumanGameDataset:
    """Load all shards, or reload from a single cached .pt file."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached dataset from {CACHE_PATH} ...")
        t0 = time.perf_counter()
        c = torch.load(CACHE_PATH, weights_only=False, map_location="cpu")
        mask = fix_action_masks(c["ff"], c["mask"])
        mask[torch.arange(len(c["action_idx"])), c["action_idx"]] = 1.0
        ds = HumanGameDataset(
            c["nf"], c["ef"], c["ff"], mask, c["action_idx"], c["value_target"],
        )
        print(f"Loaded {len(ds):,} examples in {time.perf_counter()-t0:.1f}s (cached, masks fixed)")
        return ds

    print(f"First run: loading ALL shards from {DATA_DIR} (this is slow once) ...")
    t0 = time.perf_counter()
    ds = load_tensor_shards(DATA_DIR, max_examples=0)
    load_sec = time.perf_counter() - t0
    print(f"Loaded {len(ds):,} examples in {load_sec:.1f}s")

    print(f"Caching to {CACHE_PATH} ...")
    t0 = time.perf_counter()
    torch.save({
        "nf": ds.nf, "ef": ds.ef, "ff": ds.ff,
        "mask": ds.mask, "action_idx": ds.action_idx,
        "value_target": ds.value_target,
    }, CACHE_PATH)
    print(f"Cached in {time.perf_counter()-t0:.1f}s")
    return ds


def load_cursor() -> dict:
    if os.path.exists(CURSOR_PATH):
        with open(CURSOR_PATH) as f:
            return json.load(f)
    return {"examples_seen": 0, "epoch": 0}


def save_cursor(cursor: dict) -> None:
    with open(CURSOR_PATH, "w") as f:
        json.dump(cursor, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20_000)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--pretrained", type=str, default="")
    args = parser.parse_args()

    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    SEED = 42
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.reset:
        for p in [CURSOR_PATH, MODEL_PATH]:
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    action_enc = ActionEncoder()
    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    # ── Load all data (cached) ────────────────────────────────────
    full_ds = load_or_cache_all(device)
    total_available = len(full_ds)

    # ── Slice out unseen examples ─────────────────────────────────
    cursor = load_cursor()
    start = cursor["examples_seen"]
    new_examples = args.games * EXAMPLES_PER_GAME
    end = min(start + new_examples, total_available)

    if start >= total_available:
        print(f"\nAll {total_available:,} examples already seen. Use --reset to start over.")
        return

    actual_new = end - start
    actual_games = actual_new // EXAMPLES_PER_GAME

    print(f"\nCursor: {start:,} examples seen previously (epoch {cursor['epoch']})")
    print(f"Training on examples [{start:,} .. {end:,}] "
          f"(~{actual_games:,} new games, {actual_new:,} examples)")
    print(f"Remaining after this: {total_available - end:,} examples\n")

    idx = torch.arange(start, end)
    train_ds = HumanGameDataset(
        full_ds.nf[idx], full_ds.ef[idx], full_ds.ff[idx],
        full_ds.mask[idx], full_ds.action_idx[idx], full_ds.value_target[idx],
    )

    # Free the full dataset from CPU memory
    del full_ds

    train_dev = DeviceDataset(train_ds, device)
    if device == "mps":
        torch.mps.synchronize()

    # ── Model ─────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH) and not args.reset:
        net = HumanBotNet.load_checkpoint(MODEL_PATH, device=device)
        net.train()
        print(f"Resumed from {MODEL_PATH} ({net.num_parameters:,} params)")
    elif args.pretrained and os.path.exists(args.pretrained):
        net = HumanBotNet.load_checkpoint(args.pretrained, device=device)
        net.train()
        print(f"Starting from pretrained: {args.pretrained} ({net.num_parameters:,} params)")
    else:
        net = HumanBotNet(SmallNetworkConfig()).to(device)
        print(f"New HumanBotNet: {net.num_parameters:,} params")

    # ── Train 1 epoch on the new slice ────────────────────────────
    cfg = HumanBotTrainingConfig(
        batch_size=args.batch_size,
        epochs=1,
        freeze_encoder_epochs=0,
        label_smoothing=0.05,
        entropy_weight=0.01,
    )

    loss_combiner = UncertaintyWeightedLoss().to(device)
    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    lr = cfg.lr_finetune
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(train_dev) // args.batch_size)
    scheduler = build_cosine_scheduler(optimizer, steps, min(cfg.lr_warmup_steps, 200))

    epoch = cursor["epoch"] + 1
    print(f"\n{'='*60}")
    print(f"  Epoch {epoch}: training on ~{actual_games:,} NEW games ({actual_new:,} examples)")
    print(f"  LR={lr}  batch_size={args.batch_size}  device={device}")
    print(f"{'='*60}")

    t_train = time.perf_counter()
    avg = train_epoch(
        net, train_dev, optimizer, scheduler, loss_combiner,
        edge_index_dev, device, cfg,
    )
    if device == "mps":
        torch.mps.synchronize()
    train_sec = time.perf_counter() - t_train

    print(f"  ploss={avg['policy_loss']:.4f}  pacc={avg['policy_acc']:.3f}  "
          f"vloss={avg['value_loss']:.4f}  vacc={avg['value_acc']:.3f}")
    print(f"  Time: {train_sec:.1f}s  ({actual_new/train_sec:.0f} ex/s)\n")

    net.save_checkpoint(MODEL_PATH, {"epoch": epoch, **avg})
    cursor["examples_seen"] = end
    cursor["epoch"] = epoch
    save_cursor(cursor)
    print(f"Saved checkpoint + cursor (total seen: {end:,} examples)\n")

    # ── Evaluation ────────────────────────────────────────────────
    for depth in [0, 1, 2]:
        label = f"{depth}-ply"
        print("=" * 60)
        print(f"  Evaluating {label} vs AB2  ({args.eval_games} games, 2v2 seats)")
        print("=" * 60)

        t_eval = time.perf_counter()
        result = evaluate_search_vs_ab2(
            net, state_enc, action_enc, device, lib,
            num_games=args.eval_games,
            search_depth=depth,
            seed_offset=epoch * 1000 + depth * 100,
        )
        eval_sec = time.perf_counter() - t_eval

        print(f"  NN wins: {result['hz_wins']}   AB2 wins: {result['ab2_wins']}   "
              f"draws: {result['draws']}")
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  NN forward calls: {result['nn_fwd_calls']:,}")
        print(f"  Time: {eval_sec:.1f}s  ({eval_sec/max(args.eval_games,1):.2f}s/game)\n")

    print("Done.")


if __name__ == "__main__":
    main()
