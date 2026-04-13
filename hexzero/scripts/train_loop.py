#!/usr/bin/env python3
"""Continuous training loop for distributed HexaZero.

Scans a games/ directory for new .pt game files written by self-play
workers. Ingests them into a local replay buffer, trains, saves
checkpoints, and evaluates against AB2. Logs everything to W&B.

No NFS locking needed -- workers write files, trainer reads them.
Processed files are moved to a processed/ subdirectory.

Usage:
    python -m hexzero.scripts.train_loop \
        --games-dir /nlp/scr/nroll/catanatron/games \
        --checkpoint-dir /nlp/scr/nroll/catanatron/checkpoints \
        --wandb-key <key>
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
import shutil
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description="HexaZero continuous trainer")
    parser.add_argument("--games-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs-per-cycle", type=int, default=5,
                        help="Training epochs each time new data arrives")
    parser.add_argument("--min-new-games", type=int, default=10,
                        help="Wait for this many new game files before training")
    parser.add_argument("--eval-every", type=int, default=3,
                        help="Evaluate every N training cycles")
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--max-cycles", type=int, default=100,
                        help="Max training cycles (0=infinite)")
    parser.add_argument("--poll-interval", type=int, default=15,
                        help="Seconds between checking for new games")
    parser.add_argument("--buffer-capacity", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb-project", type=str, default="hexazero")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = _detect_device(args.device)
    games_dir = Path(args.games_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    processed_dir = games_dir / "processed"
    games_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.selfplay.replay_buffer import ReplayBuffer
    from hexzero.training.loss import HexaZeroLoss
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.elo.rating import EloRating, MatchResult
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()
    crit = HexaZeroLoss()

    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()

    # Model
    if args.resume and Path(args.resume).exists():
        net = HexaZeroNet.load_checkpoint(args.resume, device=device)
        print(f"[trainer] Resumed from {args.resume}", flush=True)
    else:
        net = HexaZeroNet(cfg.network)
        net.to(device)
    print(f"[trainer] {net.num_parameters:,} params on {device}", flush=True)

    buf = ReplayBuffer(capacity=args.buffer_capacity)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # ELO
    elo = EloRating(k_factor=32.0)
    elo.register_player("AB2", 100.0, pinned=True)
    elo.register_player("HexaZero", 100.0)
    elo.register_player("Random", 100.0)

    # W&B
    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            gpu_name = "cpu"
            if device == "cuda" and torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_properties(0).name
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"trainer-{os.uname().nodename}",
                config=vars(args),
                tags=["trainer", gpu_name.split()[0]],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] Failed: {e}", flush=True)

    # Main loop
    cycle = 0
    total_games_ingested = 0
    t_start = time.time()

    while args.max_cycles == 0 or cycle < args.max_cycles:
        # --- Wait for new game files ---
        new_files = _find_new_games(games_dir)
        while len(new_files) < args.min_new_games:
            if new_files:
                print(f"[trainer] {len(new_files)} new games, "
                      f"waiting for {args.min_new_games}...", flush=True)
            time.sleep(args.poll_interval)
            new_files = _find_new_games(games_dir)

        # --- Ingest game files ---
        n_ingested, n_positions = _ingest_games(new_files, buf, processed_dir)
        total_games_ingested += n_ingested
        print(f"[trainer] Cycle {cycle}: ingested {n_ingested} games "
              f"({n_positions} positions), buffer={len(buf)}", flush=True)

        if len(buf) < args.batch_size:
            print(f"[trainer] Buffer too small ({len(buf)}), waiting...", flush=True)
            continue

        # --- Train ---
        net.train()
        nb = max(len(buf) // args.batch_size, 1)
        use_amp = (device == "cuda")

        for ep in range(args.epochs_per_cycle):
            acc = {}
            t_ep = time.time()
            for _ in range(nb):
                b = buf.sample(args.batch_size)
                inp = {
                    "node_features": b.node_features.to(device, non_blocking=True),
                    "edge_index": b.edge_index.to(device, non_blocking=True),
                    "edge_features": b.edge_features.to(device, non_blocking=True),
                    "flat_features": b.flat_features.to(device, non_blocking=True),
                    "action_mask": b.action_masks.to(device, non_blocking=True),
                }
                tgt = {
                    "policy_targets": b.policy_targets.to(device, non_blocking=True),
                    "value_targets": b.value_targets.to(device, non_blocking=True),
                    "action_masks": b.action_masks.to(device, non_blocking=True),
                }
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.split(":")[0], enabled=use_amp):
                    L = crit(net(inp), tgt)
                L["total_loss"].backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                for k in ["total_loss", "value_loss", "policy_loss",
                           "value_accuracy", "policy_entropy"]:
                    acc[k] = acc.get(k, 0) + L[k].item()

            avg = {k: v / nb for k, v in acc.items()}
            sps = (nb * args.batch_size) / (time.time() - t_ep)
            print(f"  E{ep}: loss={avg['total_loss']:.4f} "
                  f"ploss={avg['policy_loss']:.4f} "
                  f"vacc={avg['value_accuracy']:.3f} "
                  f"{sps:.0f} s/s", flush=True)

        # --- Save checkpoint ---
        net.save_checkpoint(
            str(ckpt_dir / "latest.pt"),
            metadata={"cycle": cycle, "games_ingested": total_games_ingested})
        net.save_checkpoint(
            str(ckpt_dir / f"cycle_{cycle:04d}.pt"),
            metadata={"cycle": cycle})

        # --- Log to W&B ---
        if wandb_run:
            import wandb
            wandb.log({
                "train/total_loss": avg["total_loss"],
                "train/policy_loss": avg["policy_loss"],
                "train/value_loss": avg["value_loss"],
                "train/value_accuracy": avg["value_accuracy"],
                "train/policy_entropy": avg["policy_entropy"],
                "train/buffer_size": len(buf),
                "train/total_games": total_games_ingested,
                "train/samples_per_sec": sps,
                "train/cycle": cycle,
            })

        # --- Evaluate ---
        if (cycle + 1) % args.eval_every == 0:
            print(f"[eval] {args.eval_games} games vs AB2...", flush=True)
            net.eval()
            hz_w, ab2_w, rand_w = _evaluate(
                net, state_enc, action_enc, device, lib, args.eval_games, cycle)
            total = hz_w + ab2_w + rand_w

            for _ in range(hz_w):
                elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2", "Random", "Random"],
                    "HexaZero", 0, 0, 0, time.time()))
            for _ in range(ab2_w):
                elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2", "Random", "Random"],
                    "AB2", 1, 0, 0, time.time()))
            for _ in range(rand_w):
                elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2", "Random", "Random"],
                    "Random", 2, 0, 0, time.time()))

            hz_elo = elo.get_rating("HexaZero")
            print(f"  HZ={hz_w} AB2={ab2_w} Rand={rand_w} | "
                  f"ELO: HZ={hz_elo:.0f} AB2=100", flush=True)

            if wandb_run:
                import wandb
                wandb.log({
                    "eval/hz_wins": hz_w,
                    "eval/ab2_wins": ab2_w,
                    "eval/hz_win_rate": hz_w / max(total, 1),
                    "eval/hz_elo": hz_elo,
                    "eval/cycle": cycle,
                })

        cycle += 1

    if wandb_run:
        import wandb
        wandb.finish()
    print(f"[trainer] Done: {cycle} cycles, {total_games_ingested} games, "
          f"{time.time()-t_start:.0f}s", flush=True)


def _find_new_games(games_dir: Path) -> list[Path]:
    return sorted(games_dir.glob("w*_g*.pt"))


def _ingest_games(files: list[Path], buf, processed_dir: Path):
    from hexzero.selfplay.replay_buffer import TrainingExample
    n_games = 0
    n_pos = 0
    for f in files:
        try:
            examples = torch.load(f, weights_only=False, map_location="cpu")
            for ex in examples:
                buf.push(ex["state"], ex["policy"], ex["value"])
                n_pos += 1
            n_games += 1
            shutil.move(str(f), str(processed_dir / f.name))
        except Exception as e:
            print(f"  Warning: failed to load {f.name}: {e}", flush=True)
    return n_games, n_pos


def _evaluate(net, state_enc, action_enc, device, lib, num_games, cycle):
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    hz_w = ab2_w = rand_w = 0
    for gi in range(num_games):
        g = CatanGame(seed=80000 + cycle * 1000 + gi)
        g.reset()
        hz_s, ab2_s = gi % 4, (gi + 1) % 4

        while not g.is_terminal() and g.turn_number < 1000:
            cp = g.current_player()
            le = g.get_legal_actions()
            if not le:
                break

            if cp == hz_s:
                bi, bv = 0, -1.0
                for i in range(len(le)):
                    c = g.clone(); c.step(i)
                    if c.is_terminal():
                        v = 1.0 if c.winner() == hz_s else 0.0
                    else:
                        enc = state_enc.encode(c.get_state_view())
                        bb = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
                        cl = c.get_legal_actions()
                        if cl:
                            bb["action_mask"] = action_enc.get_action_mask(cl).unsqueeze(0).to(device)
                        with torch.no_grad():
                            v = F.softmax(net(bb)["value"], dim=-1)[0, 0].item()
                    if v > bv:
                        bv = v; bi = i
                g.step(bi)

            elif cp == ab2_s:
                cg = g._game
                bc = cg.state.colors[cg.state.current_player_index]
                bi, bv = 0, -math.inf
                ch = CGame(); ca = (Action * MAX_ACTIONS)(); cn = ctypes.c_int(0)
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
                    if v > bv:
                        bv = v; bi = i
                g.step(bi)
            else:
                g.step(random.randrange(len(le)))

        w = g.winner()
        if w == hz_s: hz_w += 1
        elif w == ab2_s: ab2_w += 1
        elif w is not None: rand_w += 1

    return hz_w, ab2_w, rand_w


def _detect_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


if __name__ == "__main__":
    main()
