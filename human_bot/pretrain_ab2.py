"""Pretrain HumanBotNet on AB2 self-play data.

Loads shards in small groups to stay within memory.

Usage:
    python -m human_bot.pretrain_ab2
"""

from __future__ import annotations

import os
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

from human_bot.config import HumanBotTrainingConfig
from human_bot.dataset import HumanGameDataset
from human_bot.loss import UncertaintyWeightedLoss
from human_bot.model import HumanBotNet, SmallNetworkConfig
from human_bot.train import DeviceDataset, build_cosine_scheduler, train_epoch

AB2_DIR = "data/ab2_pretrain"
CKPT_DIR = "checkpoints/human_bot_experiment"
CKPT_PATH = os.path.join(CKPT_DIR, "ab2_pretrained.pt")
BATCH_SIZE = 4096
SEED = 42
SHARDS_PER_GROUP = 20


def load_shard_group(fnames):
    all_nf, all_ef, all_ff, all_mask, all_act, all_vt = [], [], [], [], [], []
    for fname in fnames:
        data = torch.load(
            os.path.join(AB2_DIR, fname), weights_only=False, map_location="cpu",
        )
        players = data["player"].numpy()
        rv = data["reward_vec"].numpy()
        S = players.shape[0]

        winners = rv.argmax(axis=1)
        vt = np.zeros((S, 4), dtype=np.float32)
        vt[np.arange(S), winners] = 1.0
        bad = rv.max(axis=1) < 1e-8
        vt[bad] = 0.25
        shifts = (-players % 4).astype(np.int32)
        idx_arr = (np.arange(4)[None, :] + shifts[:, None]) % 4
        vt = np.take_along_axis(vt, idx_arr, axis=1)

        mask = data["action_mask"]
        if mask.shape[-1] < 397:
            pad = torch.zeros(S, 397 - mask.shape[-1], dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=-1)

        all_nf.append(data["node_features"])
        all_ef.append(data["edge_features"])
        all_ff.append(data["flat_features"])
        all_mask.append(mask)
        all_act.append(data["action_idx"])
        all_vt.append(torch.from_numpy(vt))

    return HumanGameDataset(
        torch.cat(all_nf), torch.cat(all_ef), torch.cat(all_ff),
        torch.cat(all_mask), torch.cat(all_act), torch.cat(all_vt),
    )


def main():
    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from hexzero.game.interface import CatanGame
    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    net = HumanBotNet(SmallNetworkConfig()).to(device)
    print(f"HumanBotNet: {net.num_parameters:,} params on {device}", flush=True)

    cfg = HumanBotTrainingConfig(
        batch_size=BATCH_SIZE, epochs=1, freeze_encoder_epochs=0,
        label_smoothing=0.05, entropy_weight=0.01,
    )

    shard_files = sorted(
        f for f in os.listdir(AB2_DIR)
        if f.endswith(".pt") and f != "metadata.pt"
    )
    total_shards = len(shard_files)
    n_groups = (total_shards + SHARDS_PER_GROUP - 1) // SHARDS_PER_GROUP
    print(f"{total_shards} shards in {n_groups} groups of {SHARDS_PER_GROUP}", flush=True)

    loss_combiner = UncertaintyWeightedLoss().to(device)
    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    lr = cfg.lr_finetune
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=cfg.weight_decay)

    total_ex = 0
    t_start = time.perf_counter()

    for gi in range(n_groups):
        s0 = gi * SHARDS_PER_GROUP
        s1 = min(s0 + SHARDS_PER_GROUP, total_shards)
        group_files = shard_files[s0:s1]

        ds = load_shard_group(group_files)
        dev_ds = DeviceDataset(ds, device)
        n = len(ds)
        del ds
        if device == "mps":
            torch.mps.synchronize()

        steps = max(1, n // BATCH_SIZE)
        scheduler = build_cosine_scheduler(optimizer, steps, min(30, steps))

        avg = train_epoch(
            net, dev_ds, optimizer, scheduler, loss_combiner,
            edge_index_dev, device, cfg,
        )
        del dev_ds
        total_ex += n

        if (gi + 1) % 10 == 0 or gi == 0 or gi + 1 == n_groups:
            elapsed = time.perf_counter() - t_start
            print(f"  [{gi+1}/{n_groups}] {total_ex:,} ex  "
                  f"pacc={avg['policy_acc']:.3f} vacc={avg['value_acc']:.3f}  "
                  f"ploss={avg['policy_loss']:.3f}  ({elapsed:.0f}s)", flush=True)
            sys.stdout.flush()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone: {total_ex:,} examples in {elapsed:.0f}s", flush=True)
    net.save_checkpoint(CKPT_PATH, {"stage": "ab2_pretrain_100k", "total_examples": total_ex})
    print(f"Saved: {CKPT_PATH}", flush=True)


if __name__ == "__main__":
    main()
