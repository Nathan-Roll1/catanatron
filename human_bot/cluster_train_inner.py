#!/usr/bin/env python3
"""Cluster training script for Human Bot.

Designed for jagupard machines on the Stanford NLP cluster.
Streams data in shard groups to stay within RAM limits.
Single-GPU training (model is ~600k params, too small for DataParallel benefit).
Logs to Weights & Biases.

Usage:
    python3 -u human_bot/cluster_train_inner.py \
        --ab2-dir data/ab2_100k \
        --human-dir data/human_games_fixed \
        --ckpt-dir checkpoints/cluster_run \
        --batch-size 8192 --shards-per-group 20
"""

from __future__ import annotations

import argparse
import gc
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn

from human_bot.config import HumanBotTrainingConfig
from human_bot.dataset import HumanGameDataset, rotate_value_targets_to_cp
from human_bot.loss import UncertaintyWeightedLoss
from human_bot.model import HumanBotNet, SmallNetworkConfig
from human_bot.train import DeviceDataset, build_cosine_scheduler, train_epoch

WANDB_KEY = "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"


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_shard_group(data_dir: str, shard_files: list[str], device: str,
                     ) -> tuple[DeviceDataset, int]:
    """Load a group of shards into a DeviceDataset."""
    nfs, efs, ffs, masks, acts, vts = [], [], [], [], [], []
    for fn in shard_files:
        d = torch.load(
            os.path.join(data_dir, fn), weights_only=False, map_location="cpu",
        )
        players = d["player"].numpy()
        rv = d["reward_vec"].numpy()
        S = players.shape[0]

        winners = rv.argmax(axis=1)
        vt = np.zeros((S, 4), dtype=np.float32)
        vt[np.arange(S), winners] = 1.0
        vt[rv.max(axis=1) < 1e-8] = 0.25
        n_p = d.get("num_players")
        n_p_arr = n_p.numpy() if n_p is not None else None
        vt = rotate_value_targets_to_cp(vt, players, n_p_arr)

        mask = d["action_mask"]
        if mask.shape[-1] < 397:
            mask = torch.cat(
                [mask, torch.zeros(S, 397 - mask.shape[-1], dtype=mask.dtype)],
                dim=-1,
            )

        nfs.append(d["node_features"])
        efs.append(d["edge_features"])
        ffs.append(d["flat_features"])
        masks.append(mask)
        acts.append(d["action_idx"])
        vts.append(torch.from_numpy(vt))

    ds = HumanGameDataset(
        torch.cat(nfs), torch.cat(efs), torch.cat(ffs),
        torch.cat(masks), torch.cat(acts), torch.cat(vts),
    )
    n = len(ds)
    dd = DeviceDataset(ds, device)
    del ds, nfs, efs, ffs, masks, acts, vts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return dd, n


def train_on_dir(
    net: nn.Module,
    loss_combiner: nn.Module,
    data_dir: str,
    label: str,
    device: str,
    edge_index: torch.Tensor,
    cfg: HumanBotTrainingConfig,
    shards_per_group: int = 20,
    ckpt_dir: str = "",
    ckpt_milestones: list[int] | None = None,
    wandb_run=None,
    global_step_offset: int = 0,
    on_milestone=None,
) -> tuple[int, dict]:
    """Train one epoch over all shards in a directory, streaming in groups.

    Saves checkpoints at specified example-count milestones.
    """
    shard_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".pt") and f != "metadata.pt"
    )

    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=cfg.lr_finetune, weight_decay=cfg.weight_decay,
    )

    total_ex = 0
    t_start = time.perf_counter()
    n_groups = (len(shard_files) + shards_per_group - 1) // shards_per_group
    milestones_hit: set[int] = set()
    avg: dict = {"policy_loss": 0, "value_loss": 0, "policy_acc": 0,
                 "value_acc": 0, "entropy": 0}

    for gi in range(n_groups):
        s0 = gi * shards_per_group
        s1 = min(s0 + shards_per_group, len(shard_files))
        group = shard_files[s0:s1]

        dd, n = load_shard_group(data_dir, group, device)
        steps = max(1, n // cfg.batch_size)
        scheduler = build_cosine_scheduler(optimizer, steps, min(50, steps))

        avg = train_epoch(
            net, dd, optimizer, scheduler, loss_combiner,
            edge_index, device, cfg,
        )

        del dd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_ex += n
        global_step = global_step_offset + total_ex

        # W&B logging
        if wandb_run is not None:
            import wandb
            wandb.log({
                f"{label}/policy_loss": avg["policy_loss"],
                f"{label}/value_loss": avg["value_loss"],
                f"{label}/policy_acc": avg["policy_acc"],
                f"{label}/value_acc": avg["value_acc"],
                f"{label}/entropy": avg["entropy"],
                "global_step": global_step,
                f"{label}/examples_seen": total_ex,
            })

        # Milestone checkpoints
        if ckpt_milestones and ckpt_dir:
            for milestone in ckpt_milestones:
                if total_ex >= milestone and milestone not in milestones_hit:
                    milestones_hit.add(milestone)
                    ckpt_path = os.path.join(
                        ckpt_dir,
                        f"{label.lower()}_{milestone//1_000_000}M.pt",
                    )
                    net.save_checkpoint(ckpt_path, {
                        "stage": label,
                        "total_examples": total_ex,
                        **avg,
                    })
                    print(f"  ** Saved milestone checkpoint: {ckpt_path}",
                          flush=True)
                    if on_milestone:
                        on_milestone(f"{label}_{milestone // 1_000_000}M")

        if (gi + 1) % 10 == 0 or gi + 1 == n_groups:
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / (gi + 1)) * (n_groups - gi - 1)
            print(
                f"  {label} [{gi+1}/{n_groups}] {total_ex:,} ex  "
                f"pacc={avg['policy_acc']:.3f}  vacc={avg['value_acc']:.3f}  "
                f"ploss={avg['policy_loss']:.3f}  "
                f"({elapsed:.0f}s, ETA {eta/60:.0f}min)",
                flush=True,
            )

    return total_ex, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab2-dir", type=str, required=True)
    parser.add_argument("--human-dir", type=str, default=None,
                        help="Human game shard dir (omit to skip human finetune)")
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--shards-per-group", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device if args.device != "auto" else detect_device()
    if device == "cuda":
        device = "cuda:0"

    t_total_start = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Shards per group: {args.shards_per_group}", flush=True)

    # ── W&B ───────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = WANDB_KEY
            wandb_run = wandb.init(
                project="human-bot",
                name=f"cluster-{time.strftime('%m%d-%H%M')}",
                config={
                    "batch_size": args.batch_size,
                    "shards_per_group": args.shards_per_group,
                    "device": device,
                    "seed": args.seed,
                },
            )
            print(f"W&B: {wandb_run.url}", flush=True)
            # Save run ID so parallel eval processes can resume this run
            id_path = os.path.join(args.ckpt_dir, ".wandb_id")
            with open(id_path, "w") as f:
                f.write(wandb_run.id)
        except Exception as e:
            print(f"W&B init failed: {e}", flush=True)

    from hexzero.game.interface import CatanGame

    g = CatanGame(seed=0)
    g.reset()
    se = g.make_state_encoder()
    edge_index = se._edge_index.to(device)

    # ── Model ─────────────────────────────────────────────────────
    model_cfg = SmallNetworkConfig(
        gnn_hidden_dim=int(os.environ.get("GNN_HIDDEN", 64)),
        trunk_channels=int(os.environ.get("TRUNK_CHANNELS", 128)),
        mask_as_input=True,
    )
    net = HumanBotNet(model_cfg).to(device)
    print(f"Model: {net.num_parameters:,} params on {device}", flush=True)

    if wandb_run:
        import wandb
        wandb.config.update({"params": net.num_parameters})

    loss_combiner = UncertaintyWeightedLoss().to(device)

    cfg = HumanBotTrainingConfig(
        batch_size=args.batch_size,
        epochs=1,
        freeze_encoder_epochs=0,
        label_smoothing=0.05,
        entropy_weight=0.01,
    )

    # ── Helpers: val + self-play probe ──────────────────────────
    from hexzero.bindings.lib_loader import load_library
    from hexzero.encoder.action_encoder import ActionEncoder
    from collections import Counter

    lib = load_library()
    ae = ActionEncoder()
    AD = 337
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    RESOURCES = ["Lu", "Br", "Sh", "Wh", "Or"]

    # Load human shards for validation (if human dir provided)
    human_shards_all = []
    val_ds = None
    if args.human_dir and os.path.isdir(args.human_dir):
        human_shards_all = sorted(
            f for f in os.listdir(args.human_dir)
            if f.endswith(".pt") and f != "metadata.pt"
        )
        if len(human_shards_all) >= 5:
            val_shard_files = human_shards_all[-5:]
            _nfs, _efs, _ffs, _masks, _acts, _vts = [], [], [], [], [], []
            for fn in val_shard_files:
                d = torch.load(os.path.join(args.human_dir, fn), weights_only=False, map_location="cpu")
                p = d["player"].numpy(); rv = d["reward_vec"].numpy(); S = p.shape[0]
                w = rv.argmax(axis=1); vt = np.zeros((S, 4), dtype=np.float32)
                vt[np.arange(S), w] = 1.0; vt[rv.max(axis=1) < 1e-8] = 0.25
                n_p_eval = d.get("num_players")
                vt = rotate_value_targets_to_cp(
                    vt, p, n_p_eval.numpy() if n_p_eval is not None else None)
                m = d["action_mask"]
                if m.shape[-1] < 397:
                    m = torch.cat([m, torch.zeros(S, 397 - m.shape[-1], dtype=m.dtype)], dim=-1)
                _nfs.append(d["node_features"]); _efs.append(d["edge_features"])
                _ffs.append(d["flat_features"]); _masks.append(m)
                _acts.append(d["action_idx"]); _vts.append(torch.from_numpy(vt))
            val_ds = HumanGameDataset(
                torch.cat(_nfs), torch.cat(_efs), torch.cat(_ffs),
                torch.cat(_masks), torch.cat(_acts), torch.cat(_vts),
            )
            del _nfs, _efs, _ffs, _masks, _acts, _vts
            print(f"Validation set: {len(val_ds):,} examples (last 5 human shards)\n", flush=True)
    if val_ds is None:
        print(f"No human val set (human_dir={'none' if not args.human_dir else str(len(human_shards_all)) + ' shards'})\n",
              flush=True)

    def desc(act):
        t, v = act.type, act.value
        if t == 4: return f"SETT(n{v[0]})"
        if t == 5: return f"CITY(n{v[0]})"
        if t == 3: return f"ROAD({v[0]}-{v[1]})"
        if t == 1: return f"ROB(|{'P' + str(v[3]) if v[3] >= 0 else 'X'})"
        if t == 11: return f"TR({RESOURCES[v[0]]}->{RESOURCES[v[4]]})"
        if t == 6: return "DEV"
        if t == 7: return "KNIGHT"
        if t == 9: return f"MONO({RESOURCES[v[0]]})"
        if t == 10: return "RD_BUILD"
        if t == 17: return "END"
        if t == 0: return "ROLL"
        if t == 2: return f"DISC({RESOURCES[v[0]]})"
        return f"t{t}"

    def run_probe(stage_name: str, seed: int = 12345):
        """Play 1 self-play game, print every action; compute val metrics."""
        net.eval()

        # ── Val metrics ──
        if val_ds is not None:
            from human_bot.evaluate import compute_metrics
            val_m = compute_metrics(net, val_ds, edge_index, device, batch_size=4096)
            print(f"\n  [{stage_name}] Val: top1={val_m.get('top1_acc', 0):.3f}  "
                  f"top3={val_m.get('top3_acc', 0):.3f}  "
                  f"vloss={val_m.get('value_loss', 0):.4f}  "
                  f"vwinner={val_m.get('value_winner_acc', 0):.3f}", flush=True)

            if wandb_run:
                import wandb
                wandb.log({
                    f"val/top1_acc": val_m.get("top1_acc", 0),
                    f"val/top3_acc": val_m.get("top3_acc", 0),
                    f"val/value_loss": val_m.get("value_loss", 0),
                    f"val/value_winner_acc": val_m.get("value_winner_acc", 0),
                    f"val/stage": stage_name,
                })

        # ── Self-play game ──
        print(f"\n  [{stage_name}] Self-play game (seed={seed}):", flush=True)
        game = CatanGame(seed=seed)
        game.reset()
        st = {p: Counter() for p in range(4)}

        while not game.is_terminal() and game.turn_number < 1000:
            cp = game.current_player()
            le = game.get_legal_actions()
            if not le:
                break
            if len(le) == 1:
                game.step(0)
                continue

            nf = np.zeros((1, N, NF), dtype=np.float32)
            ef = np.zeros((1, E, EF), dtype=np.float32)
            ff = np.zeros((1, FFD), dtype=np.float32)
            se.encode_into(game.get_state_view(), nf[0], ef[0], ff[0])
            mn = ae.get_action_mask(le).numpy()

            with torch.no_grad():
                mt = torch.from_numpy(mn).unsqueeze(0).to(device)
                m3 = torch.cat([mt, torch.zeros(1, 397 - AD, device=device)], dim=1)
                out = net({
                    "node_features": torch.from_numpy(nf.copy()).to(device),
                    "edge_index": edge_index,
                    "edge_features": torch.from_numpy(ef.copy()).to(device),
                    "flat_features": torch.from_numpy(ff.copy()).to(device),
                    "action_mask": m3,
                })
                lo = out["policy_logits"][0, :AD]
                lo = lo.masked_fill(mt[0] == 0, -1e9)
                pr = torch.exp(lo)
                pr = pr / pr.sum()
                pr = pr.cpu().numpy()

            ai = int(pr.argmax())
            ch = next((i for i, a in enumerate(le) if ae.encode(a) == ai), 0)
            act = le[ch]

            res = (ff[0, 1:6] * 19).round().astype(int)
            vp = int(ff[0, 0] * 10)
            print(f"    T{game.turn_number:>3d} P{cp} VP={vp} "
                  f"[{res[0]}L {res[1]}B {res[2]}S {res[3]}W {res[4]}O] "
                  f"-> {desc(act)}", flush=True)

            for t, k in [(4, "sett"), (5, "city"), (3, "road"), (6, "dev"),
                         (7, "knight"), (11, "trade")]:
                if act.type == t:
                    st[cp][k] += 1
            if act.type == 1:
                st[cp]["steal" if act.value[3] >= 0 else "no_steal"] += 1

            game.step(ch)

        w = game.winner()
        tc = sum(st[p]["city"] for p in range(4))
        td = sum(st[p]["dev"] for p in range(4))
        ts = sum(st[p]["steal"] for p in range(4))
        tn = sum(st[p]["no_steal"] for p in range(4))
        print(f"  [{stage_name}] Result: P{w} wins T{game.turn_number} "
              f"Cities={tc} Dev={td} Steals={ts}/{ts + tn}", flush=True)
        for p in range(4):
            s = st[p]
            print(f"    P{p}: s={s['sett']} c={s['city']} r={s['road']} "
                  f"d={s['dev']} k={s['knight']} t={s['trade']}", flush=True)

        net.train()

    # ── Stage 1: AB2 Pretrain ─────────────────────────────────────
    ab2_shards = [
        f for f in os.listdir(args.ab2_dir)
        if f.endswith(".pt") and f != "metadata.pt"
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"  Stage 1: AB2 pretrain ({len(ab2_shards)} shards)", flush=True)
    print(f"{'='*60}", flush=True)

    # ~31 examples/game => 100k games ≈ 3.1M ex
    AB2_MILESTONES = [1_500_000]

    n_ab2, avg_ab2 = train_on_dir(
        net, loss_combiner, args.ab2_dir, "AB2",
        device, edge_index, cfg,
        shards_per_group=args.shards_per_group,
        ckpt_dir=args.ckpt_dir,
        ckpt_milestones=AB2_MILESTONES,
        wandb_run=wandb_run,
        on_milestone=lambda name: run_probe(name, seed=54321),
    )

    ckpt_ab2 = os.path.join(args.ckpt_dir, "ab2_final.pt")
    net.save_checkpoint(ckpt_ab2, {
        "stage": "ab2_pretrain",
        "total_examples": n_ab2,
        **avg_ab2,
    })
    print(f"AB2 done: {n_ab2:,} examples  pacc={avg_ab2['policy_acc']:.3f}",
          flush=True)
    run_probe("after_ab2", seed=54321)

    # ── Stage 2: Human Finetune (exclude last 5 val shards) ─────
    if len(human_shards_all) > 5:
        MAX_HUMAN_SHARDS = int(os.environ.get("MAX_HUMAN_SHARDS", len(human_shards_all) - 5))
        human_train_shards = human_shards_all[:min(MAX_HUMAN_SHARDS, len(human_shards_all) - 5)]
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage 2: Human finetune ({len(human_train_shards)} train shards, "
              f"5 val shards held out)", flush=True)
        print(f"{'='*60}", flush=True)

        import tempfile
        human_train_dir = tempfile.mkdtemp(prefix="human_train_")
        for fn in human_train_shards:
            os.symlink(
                os.path.abspath(os.path.join(args.human_dir, fn)),
                os.path.join(human_train_dir, fn),
            )

        n_human, avg_human = train_on_dir(
            net, loss_combiner, human_train_dir, "Human",
            device, edge_index, cfg,
            shards_per_group=args.shards_per_group,
            ckpt_dir=args.ckpt_dir,
            wandb_run=wandb_run,
            global_step_offset=n_ab2,
        )

        import shutil
        shutil.rmtree(human_train_dir, ignore_errors=True)

        ckpt_final = os.path.join(args.ckpt_dir, "final.pt")
        net.save_checkpoint(ckpt_final, {
            "stage": "ab2+human",
            "ab2_examples": n_ab2,
            "human_examples": n_human,
            **avg_human,
        })
        print(
            f"Human done: {n_human:,} examples  "
            f"pacc={avg_human['policy_acc']:.3f}  vacc={avg_human['value_acc']:.3f}",
            flush=True,
        )
        run_probe("after_human_ft", seed=54321)
    else:
        reason = "no --human-dir" if not args.human_dir else f"only {len(human_shards_all)} shards"
        print(f"\n  Skipping human finetune ({reason})", flush=True)
        ckpt_final = os.path.join(args.ckpt_dir, "final.pt")
        net.save_checkpoint(ckpt_final, {
            "stage": "ab2_only",
            "ab2_examples": n_ab2,
            **avg_ab2,
        })

    # ── Stage 3: Benchmark vs AB2 ────────────────────────────────
    if args.eval_games > 0:
        print(f"\n{'='*60}", flush=True)
        print(f"  Stage 3: Benchmark ({args.eval_games} games per depth)",
              flush=True)
        print(f"{'='*60}", flush=True)

        net.eval()

        from human_bot.eval_search import evaluate_search_vs_ab2

        for depth in [0, 1, 2]:
            t0 = time.perf_counter()
            result = evaluate_search_vs_ab2(
                net, se, ae, device, lib,
                num_games=args.eval_games,
                search_depth=depth,
                seed_offset=depth * 100 + 8888,
            )
            sec = time.perf_counter() - t0
            print(
                f"  {depth}-ply: NN={result['hz_wins']}  "
                f"AB2={result['ab2_wins']}  "
                f"WR={result['win_rate']:.1%}  ({sec:.0f}s)",
                flush=True,
            )
            if wandb_run:
                import wandb
                wandb.log({
                    f"eval/{depth}ply_win_rate": result["win_rate"],
                    f"eval/{depth}ply_nn_wins": result["hz_wins"],
                    f"eval/{depth}ply_ab2_wins": result["ab2_wins"],
                })

    total_wall = time.perf_counter() - t_total_start
    print(f"\nAll done in {total_wall/60:.1f} min. Checkpoints in {args.ckpt_dir}/",
          flush=True)

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
