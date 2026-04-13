"""Full pipeline: 16k AB2 pretrain -> 2 human epochs -> 500-game eval.

Saves checkpoints after each stage.

Usage:
    python3 -u -m human_bot.run_full_pipeline
"""

from __future__ import annotations

import json
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

from human_bot.config import HumanBotTrainingConfig
from human_bot.dataset import HumanGameDataset, load_tensor_shards
from human_bot.eval_search import evaluate_search_vs_ab2
from human_bot.loss import UncertaintyWeightedLoss
from human_bot.model import HumanBotNet, SmallNetworkConfig
from human_bot.train import DeviceDataset, build_cosine_scheduler, train_epoch

CKPT_DIR = "checkpoints/human_bot_pipeline"
AB2_DIR = "data/ab2_pretrain"
HUMAN_DIR = "data/human_games"
BATCH_SIZE = 4096
SEED = 42
EVAL_GAMES = 500


def detect_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_on_shards(net, shard_dir, device, edge_index_dev, tag, max_examples=0):
    """Train 1 epoch on shards from a directory. Returns avg metrics."""
    print(f"\nLoading {tag} data from {shard_dir} ...", flush=True)
    t0 = time.perf_counter()
    ds = load_tensor_shards(shard_dir, max_examples=max_examples)
    print(f"Loaded {len(ds):,} examples in {time.perf_counter()-t0:.1f}s", flush=True)

    train_dev = DeviceDataset(ds, device)
    del ds
    if device == "mps":
        torch.mps.synchronize()

    cfg = HumanBotTrainingConfig(
        batch_size=BATCH_SIZE, epochs=1, freeze_encoder_epochs=0,
        label_smoothing=0.05, entropy_weight=0.01,
    )
    loss_combiner = UncertaintyWeightedLoss().to(device)
    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    lr = cfg.lr_finetune
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(train_dev) // BATCH_SIZE)
    scheduler = build_cosine_scheduler(optimizer, steps, min(cfg.lr_warmup_steps, 200))

    print(f"\n{'='*60}", flush=True)
    print(f"  {tag}: {len(train_dev):,} examples, LR={lr}", flush=True)
    print(f"{'='*60}", flush=True)

    t_train = time.perf_counter()
    avg = train_epoch(
        net, train_dev, optimizer, scheduler, loss_combiner,
        edge_index_dev, device, cfg,
    )
    if device == "mps":
        torch.mps.synchronize()
    sec = time.perf_counter() - t_train

    print(f"  ploss={avg['policy_loss']:.4f}  pacc={avg['policy_acc']:.3f}  "
          f"vloss={avg['value_loss']:.4f}  vacc={avg['value_acc']:.3f}", flush=True)
    print(f"  Time: {sec:.0f}s  ({len(train_dev)/sec:.0f} ex/s)", flush=True)
    del train_dev
    return avg


def run_eval(net, state_enc, action_enc, device, lib, tag, eval_games=EVAL_GAMES):
    """Run 0/1/2-ply evaluation."""
    edge_index_dev = state_enc._edge_index.to(device)
    epoch_num = int(tag.split("_")[-1]) if "_" in tag else 0

    for depth in [0, 1, 2]:
        print(f"\n{'='*60}", flush=True)
        print(f"  [{tag}] {depth}-ply vs AB2  ({eval_games} games)", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.perf_counter()
        result = evaluate_search_vs_ab2(
            net, state_enc, action_enc, device, lib,
            num_games=eval_games, search_depth=depth,
            seed_offset=epoch_num * 1000 + depth * 100 + 50000,
        )
        sec = time.perf_counter() - t0
        print(f"  NN wins: {result['hz_wins']}   AB2 wins: {result['ab2_wins']}   "
              f"draws: {result['draws']}", flush=True)
        print(f"  Win rate: {result['win_rate']:.1%}  ({sec:.0f}s)", flush=True)


def main():
    device = detect_device()
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    action_enc = ActionEncoder()
    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    # ── Stage 1: AB2 Pretrain ─────────────────────────────────────
    net = HumanBotNet(SmallNetworkConfig()).to(device)
    print(f"HumanBotNet: {net.num_parameters:,} params on {device}", flush=True)

    train_on_shards(net, AB2_DIR, device, edge_index_dev, "AB2_pretrain")

    ckpt_ab2 = os.path.join(CKPT_DIR, "stage1_ab2_pretrain.pt")
    net.save_checkpoint(ckpt_ab2, {"stage": "ab2_pretrain"})
    print(f"\nSaved: {ckpt_ab2}", flush=True)

    # ── Stage 2: Human Epoch 1 ────────────────────────────────────
    train_on_shards(net, HUMAN_DIR, device, edge_index_dev, "Human_epoch_1")

    ckpt_h1 = os.path.join(CKPT_DIR, "stage2_human_epoch1.pt")
    net.save_checkpoint(ckpt_h1, {"stage": "human_epoch_1"})
    print(f"\nSaved: {ckpt_h1}", flush=True)

    # ── Stage 3: Human Epoch 2 ────────────────────────────────────
    train_on_shards(net, HUMAN_DIR, device, edge_index_dev, "Human_epoch_2")

    ckpt_h2 = os.path.join(CKPT_DIR, "stage3_human_epoch2.pt")
    net.save_checkpoint(ckpt_h2, {"stage": "human_epoch_2"})
    print(f"\nSaved: {ckpt_h2}", flush=True)

    # ── Stage 4: 500-game Eval ────────────────────────────────────
    print(f"\n\n{'#'*60}", flush=True)
    print(f"  FINAL EVALUATION (500 games per depth)", flush=True)
    print(f"{'#'*60}", flush=True)

    run_eval(net, state_enc, action_enc, device, lib, "final_3", eval_games=EVAL_GAMES)

    print(f"\n\nPipeline complete.", flush=True)
    print(f"Checkpoints saved in {CKPT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
