#!/usr/bin/env python3
"""Phase 3 Expert Iteration learner.

Polls a shared trajectory directory (local SSD) for .pt files written by
selfplay_actor processes running ExIt (1-ply NN search). Trains the policy
via cross-entropy toward the search-improved actions and the value head on
terminal game outcomes.

No REINFORCE, no AB2 behavioral cloning.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

from hexzero.scripts.selfplay_train import (
    detect_device,
    train_step_exit,
    eval_batch,
    eval_vs_model,
    _save_checkpoint,
)


def _collect_trajectories(
    traj_dir: str, processed_dir: str, max_files: int = 200,
) -> dict[str, torch.Tensor] | None:
    """Read ExIt trajectory files, tensorize, move originals to processed_dir.

    Returns dict of CPU tensors {nf, ef, ff, mask, action_idx, value_target}
    or None if no files available.
    """
    pattern = os.path.join(traj_dir, "actor*_batch*.pt")
    files = sorted(glob.glob(pattern))
    if not files:
        return None

    files = files[:max_files]

    all_nf, all_ef, all_ff, all_mask = [], [], [], []
    all_act, all_vt = [], []

    for fpath in files:
        try:
            payload = torch.load(fpath, weights_only=False, map_location="cpu")
        except Exception:
            continue

        steps = payload["steps"]
        if not steps:
            continue

        nf = np.stack([s["nf"] for s in steps])
        ef = np.stack([s["ef"] for s in steps])
        ff = np.stack([s["ff"] for s in steps])
        mask = np.stack([s["mask"] for s in steps])
        act = np.array([s["action_idx"] for s in steps], dtype=np.int64)
        vt = np.stack([s["value_target"] for s in steps])

        all_nf.append(torch.from_numpy(nf))
        all_ef.append(torch.from_numpy(ef))
        all_ff.append(torch.from_numpy(ff))
        all_mask.append(torch.from_numpy(mask))
        all_act.append(torch.from_numpy(act))
        all_vt.append(torch.from_numpy(vt))

        dst = os.path.join(processed_dir, os.path.basename(fpath))
        try:
            os.replace(fpath, dst)
        except OSError:
            try:
                os.remove(fpath)
            except OSError:
                pass

    if not all_nf:
        return None

    return {
        "nf": torch.cat(all_nf),
        "ef": torch.cat(all_ef),
        "ff": torch.cat(all_ff),
        "mask": torch.cat(all_mask),
        "action_idx": torch.cat(all_act),
        "value_target": torch.cat(all_vt),
    }


def _prune_processed(processed_dir: str, keep: int = 100) -> None:
    files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
    if len(files) > keep:
        for f in files[: len(files) - keep]:
            try:
                os.remove(f)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expert Iteration learner (cross-entropy on search-improved targets)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pretrained checkpoint from Phase 1 / 2")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--trajectory-dir", type=str, required=True)

    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--max-files-per-step", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)

    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--eval-temperature", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="hexazero-exit")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)

    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.elo.rating import EloRating, MatchResult
    from hexzero.bindings.lib_loader import load_library

    action_enc = ActionEncoder()
    lib = load_library()

    tmp = CatanGame(seed=0)
    tmp.reset()
    state_enc = tmp.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    processed_dir = os.path.join(args.trajectory_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    net = HexaZeroNet.load_checkpoint(args.checkpoint, device=device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    _save_checkpoint(net, args.checkpoint_dir, 0, {})

    seed_copy = os.path.join(args.checkpoint_dir, "seed.pt")
    if not os.path.exists(seed_copy):
        shutil.copy2(args.checkpoint, seed_copy)

    print("=" * 64, flush=True)
    print("  Expert Iteration Learner", flush=True)
    print(f"  Seed ckpt   : {args.checkpoint}", flush=True)
    print(f"  Device      : {device} ({gpu_name})", flush=True)
    print(f"  Parameters  : {net.num_parameters:,}", flush=True)
    print(f"  Traj dir    : {args.trajectory_dir}", flush=True)
    print(f"  Ckpt dir    : {args.checkpoint_dir}", flush=True)
    print(f"  BS={args.batch_size}  LR={args.lr}  Ent={args.entropy_weight}", flush=True)
    print(f"  Eval every  : {args.eval_every} steps ({args.eval_games} games)", flush=True)
    print("=" * 64, flush=True)

    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"exit-learner-{os.uname().nodename}",
                config=vars(args),
                tags=["exit", "learner", device],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed: {e}", flush=True)

    global_step = 0
    total_positions = 0
    best_eval_wr = -1.0
    elo = EloRating(k_factor=32.0)
    elo.register_player("AB2", 1000.0, pinned=True)
    elo.register_player("HexaZero", 1000.0)
    t_start = time.time()
    last_prune = time.time()

    while args.max_steps == 0 or global_step < args.max_steps:
        data = _collect_trajectories(
            args.trajectory_dir, processed_dir, args.max_files_per_step)

        if data is None:
            time.sleep(args.poll_interval)
            continue

        n_pos = data["nf"].shape[0]
        gpu_data = {k: v.to(device) for k, v in data.items()}
        del data

        t0 = time.time()
        metrics, n_mb = train_step_exit(
            net, optimizer, gpu_data, edge_index_dev, device,
            args.batch_size, args.entropy_weight)
        t_train = time.time() - t0

        del gpu_data

        global_step += 1
        total_positions += n_pos
        elapsed = time.time() - t_start
        pps = total_positions / max(elapsed, 0.01)

        print(
            f"[learn {global_step:4d}] "
            f"ploss={metrics.get('policy_loss', 0):.4f} "
            f"vloss={metrics.get('value_loss', 0):.4f} "
            f"pacc={metrics.get('policy_accuracy', 0):.3f} "
            f"vacc={metrics.get('value_accuracy', 0):.3f} "
            f"ent={metrics.get('entropy', 0):.3f} "
            f"| {n_pos} pos {n_mb} mb "
            f"| {t_train:.1f}s | {pps:.0f} pos/s",
            flush=True,
        )

        _save_checkpoint(net, args.checkpoint_dir, global_step, metrics)

        if wandb_run:
            import wandb
            wandb.log({
                "train/policy_loss": metrics.get("policy_loss", 0),
                "train/value_loss": metrics.get("value_loss", 0),
                "train/total_loss": metrics.get("total_loss", 0),
                "train/policy_accuracy": metrics.get("policy_accuracy", 0),
                "train/value_accuracy": metrics.get("value_accuracy", 0),
                "train/entropy": metrics.get("entropy", 0),
                "train/positions": n_pos,
                "train/positions_per_sec": pps,
                "step": global_step,
            })

        # ── Evaluate ───────────────────────────────────────────────
        if global_step % args.eval_every == 0:
            from hexzero.model.network import HexaZeroNet as HZNet

            t_ev = time.time()
            eval_log: dict[str, float] = {}

            print(f"[eval] vs AB2 ({args.eval_games} games) ...", flush=True)
            e_hz, e_ab2 = eval_batch(
                net, state_enc, action_enc, lib, device, edge_index_dev,
                args.eval_games, global_step, args.eval_temperature)
            for _ in range(e_hz):
                elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2"], "HexaZero", 0, 0, 0, time.time()))
            for _ in range(e_ab2):
                elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2"], "AB2", 0, 0, 0, time.time()))
            ab2_wr = e_hz / max(e_hz + e_ab2, 1)
            eval_log["eval/vs_ab2_wins"] = e_hz
            eval_log["eval/vs_ab2_losses"] = e_ab2
            eval_log["eval/vs_ab2_winrate"] = ab2_wr
            eval_log["eval/hz_elo"] = elo.get_rating("HexaZero")
            print(f"  vs AB2: {e_hz}-{e_ab2} ({ab2_wr:.0%})", flush=True)

            seed_path = os.path.join(args.checkpoint_dir, "seed.pt")
            if os.path.exists(seed_path):
                try:
                    opp = HZNet.load_checkpoint(seed_path, device=device)
                    print(f"[eval] vs seed ({args.eval_games} games) ...",
                          flush=True)
                    cw, ow = eval_vs_model(
                        net, opp, state_enc, action_enc, device,
                        edge_index_dev, args.eval_games,
                        global_step + 10000, args.eval_temperature)
                    seed_wr = cw / max(cw + ow, 1)
                    eval_log["eval/vs_seed_wins"] = cw
                    eval_log["eval/vs_seed_losses"] = ow
                    eval_log["eval/vs_seed_winrate"] = seed_wr
                    print(f"  vs seed: {cw}-{ow} ({seed_wr:.0%})", flush=True)
                    del opp
                except Exception as e:
                    print(f"  vs seed: skip ({e})", flush=True)

            past_25 = os.path.join(
                args.checkpoint_dir, f"step_{max(global_step - 25, 0):06d}.pt")
            if os.path.exists(past_25) and global_step > 25:
                try:
                    opp = HZNet.load_checkpoint(past_25, device=device)
                    print(f"[eval] vs step-25 ({args.eval_games} games) ...",
                          flush=True)
                    cw, ow = eval_vs_model(
                        net, opp, state_enc, action_enc, device,
                        edge_index_dev, args.eval_games,
                        global_step + 20000, args.eval_temperature)
                    p25_wr = cw / max(cw + ow, 1)
                    eval_log["eval/vs_past25_wins"] = cw
                    eval_log["eval/vs_past25_losses"] = ow
                    eval_log["eval/vs_past25_winrate"] = p25_wr
                    print(f"  vs step-25: {cw}-{ow} ({p25_wr:.0%})", flush=True)
                    del opp
                except Exception as e:
                    print(f"  vs step-25: skip ({e})", flush=True)

            t_ev = time.time() - t_ev
            hz_elo_val = elo.get_rating("HexaZero")

            print(f"[eval] total {t_ev:.1f}s | AB2 WR={ab2_wr:.0%} | "
                  f"ELO={hz_elo_val:.0f}", flush=True)

            meta = {"eval_vs_ab2_wr": ab2_wr, "eval_elo": hz_elo_val}
            _save_checkpoint(net, args.checkpoint_dir, global_step, meta)

            if ab2_wr > best_eval_wr:
                best_eval_wr = ab2_wr
                net.save_checkpoint(
                    os.path.join(args.checkpoint_dir, "best.pt"),
                    metadata={**meta, "step": global_step})
                print(f"[eval] *** New best: WR={ab2_wr:.1%} ELO={hz_elo_val:.0f}",
                      flush=True)

            eval_log["step"] = global_step
            if wandb_run:
                import wandb
                wandb.log(eval_log)

        if time.time() - last_prune > 120:
            _prune_processed(processed_dir, keep=100)
            last_prune = time.time()

    elapsed = time.time() - t_start
    print(f"\n[learner] Done: {global_step} steps, {total_positions} positions, "
          f"{elapsed:.0f}s", flush=True)
    print(f"[learner] Best eval WR: {best_eval_wr:.1%}", flush=True)
    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
