#!/usr/bin/env python3
"""HexaZero GPU training pipeline with W&B logging.

Single self-contained script. No bash wrapper needed.
Handles: C library build, data generation, training, evaluation, ELO tracking.
Works on CUDA (jag cluster) and MPS (Apple Silicon) and CPU.

Usage on jag cluster (from sc):
    nlprun -q jag -g 1 -r 60G -c 16 -p standard -n hexazero-v1 \
        'cd /nlp/scr/nroll/catanatron && python -m hexzero.scripts.gpu_train'

Usage locally:
    python -m hexzero.scripts.gpu_train --device mps --iterations 2 --games 100
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# Unbuffered output for slurm logs
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def gpu_info(device: str) -> dict:
    info = {"device": device, "name": "cpu", "vram_gb": 0.0}
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["name"] = props.name
        info["vram_gb"] = round(props.total_memory / 1e9, 1)
    elif device == "mps":
        info["name"] = "Apple Silicon MPS"
    return info


def ensure_c_library():
    """Build the C shared library if it doesn't exist for this platform."""
    from hexzero.bindings.build_lib import build, _lib_name, _LIB_DIR
    lib_path = _LIB_DIR / _lib_name()
    if not lib_path.exists():
        print(f"[build] Compiling {_lib_name()}...")
        build()
    else:
        print(f"[build] {lib_path} exists, OK")


def smoke_test():
    from hexzero.game.interface import CatanGame
    g = CatanGame(seed=1)
    g.reset()
    assert g.get_legal_action_count() > 0
    print(f"[smoke] {g}")


def generate_data(net, state_enc, action_enc, device, num_games, iteration,
                   mcts_sims=50):
    """MCTS self-play: the real AlphaZero data generation loop.

    Iteration 0 first half uses random play for bootstrap diversity.
    All other games use MCTS with Dirichlet noise for exploration.
    """
    from hexzero.game.interface import CatanGame
    from hexzero.selfplay.replay_buffer import ReplayBuffer
    from hexzero.mcts.search import MCTSSearch
    from hexzero.config import MCTSConfig

    buf = ReplayBuffer(capacity=max(num_games * 800, 100_000))
    net.eval()
    t0 = time.time()
    winners = [0, 0, 0, 0, 0]  # p0,p1,p2,p3,timeout

    mcts_cfg = MCTSConfig(
        num_simulations=mcts_sims,
        num_determinizations=1,
        c_puct=2.5,
        dirichlet_alpha=0.15,
        dirichlet_epsilon=0.25,
        temperature_threshold=30,
        temperature_init=1.0,
        temperature_final=0.01,
        virtual_loss=0.0,
    )
    mcts = MCTSSearch(
        network=net, encoder=state_enc, action_encoder=action_enc,
        config=mcts_cfg, device=device,
    )

    for gi in range(num_games):
        game = CatanGame(seed=iteration * 10000 + gi)
        game.reset()
        hist = []
        use_random = (iteration == 0 and gi < num_games // 3)

        while not game.is_terminal() and game.turn_number < 1000:
            cp = game.current_player()
            st = state_enc.encode(game.get_state_view())
            st = {k: v.detach() for k, v in st.items()}
            le = game.get_legal_actions()
            mask = action_enc.get_action_mask(le)
            st["action_masks"] = mask

            if use_random:
                policy = mask / mask.sum()
                chosen = random.randrange(len(le))
            else:
                result = mcts.search(game)
                # Mask-align MCTS policy to legal actions
                pol = torch.from_numpy(result.action_probs).float() * mask
                s = pol.sum()
                policy = pol / s if s > 0 else mask / mask.sum()

                temp = 1.0 if game.turn_number < mcts_cfg.temperature_threshold else 0.01
                aidx = mcts.select_action(result.action_probs, temp, game.turn_number)
                chosen = next((i for i, a in enumerate(le)
                               if action_enc.encode(a) == aidx), 0)

            hist.append((st, policy, cp))
            game.step(chosen)

        w = game.winner()
        w = w if w is not None else -1
        if 0 <= w < 4:
            winners[w] += 1
        else:
            winners[4] += 1

        for st, pol, p in hist:
            vt = torch.zeros(4)
            if w >= 0:
                vt[(w - p) % 4] = 1.0
            buf.push(st, pol, vt)

        if (gi + 1) % max(num_games // 5, 1) == 0:
            elapsed = time.time() - t0
            print(f"  {gi+1}/{num_games} games | {len(buf)} pos | "
                  f"{elapsed:.0f}s | {(gi+1)/elapsed:.1f} g/s | "
                  f"w={winners[:4]} to={winners[4]}",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Done: {len(buf)} positions, {elapsed:.0f}s, "
          f"winners={winners[:4]} timeouts={winners[4]}", flush=True)
    return buf


def train_epoch(net, buf, optimizer, criterion, batch_size, device):
    net.train()
    nb = max(len(buf) // batch_size, 1)
    acc = {}
    use_amp = (device == "cuda")

    for bi in range(nb):
        b = buf.sample(batch_size)
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
            L = criterion(net(inp), tgt)

        L["total_loss"].backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        for k in ["total_loss", "value_loss", "policy_loss", "value_accuracy", "policy_entropy"]:
            acc[k] = acc.get(k, 0) + L[k].item()

    return {k: v / nb for k, v in acc.items()}, nb


def evaluate_vs_ab2(net, state_enc, action_enc, device, lib, num_games):
    """HexaZero (greedy value) vs AB2 (greedy C heuristic) vs 2 random."""
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    net.eval()
    hz_w = ab2_w = rand_w = 0

    for gi in range(num_games):
        g = CatanGame(seed=50000 + gi)
        g.reset()
        hz_s = gi % 4
        ab2_s = (gi + 1) % 4

        while not g.is_terminal() and g.turn_number < 1000:
            cp = g.current_player()
            le = g.get_legal_actions()
            if not le:
                break

            if cp == hz_s:
                bi, bv = 0, -1.0
                for i in range(len(le)):
                    c = g.clone()
                    c.step(i)
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
                        bv = v
                        bi = i
                g.step(bi)

            elif cp == ab2_s:
                cg = g._game
                bc = cg.state.colors[cg.state.current_player_index]
                bi, bv = 0, -math.inf
                ch = CGame()
                ca = (Action * MAX_ACTIONS)()
                cn = ctypes.c_int(0)
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
                    if v > bv:
                        bv = v
                        bi = i
                g.step(bi)
            else:
                g.step(random.randrange(len(le)))

        w = g.winner()
        if w == hz_s:
            hz_w += 1
        elif w == ab2_s:
            ab2_w += 1
        elif w is not None:
            rand_w += 1

    return hz_w, ab2_w, rand_w


def main():
    parser = argparse.ArgumentParser(description="HexaZero GPU Training Pipeline")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=150,
                        help="Self-play games per iteration")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--mcts-sims", type=int, default=50,
                        help="MCTS simulations per move during self-play")
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="hexzero/checkpoints")
    parser.add_argument("--wandb-project", type=str, default="hexazero")
    parser.add_argument("--wandb-key", type=str,
                        default="wandb_v1_5Wm7tx6uj1GvNyXjt5ogWR8WJyO")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)
    info = gpu_info(device)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print(f" HexaZero GPU Training Pipeline", flush=True)
    print(f" Device: {device} ({info['name']}, {info['vram_gb']} GB)", flush=True)
    print(f" Host:   {os.uname().nodename}", flush=True)
    print(f" Plan:   {args.iterations} iters x {args.games} games x {args.epochs} epochs", flush=True)
    print("=" * 60, flush=True)

    # ── Build & smoke test ────────────────────────────────────────────
    ensure_c_library()
    smoke_test()

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"hexazero-{os.uname().nodename}-{int(time.time())%10000}",
                config={
                    "iterations": args.iterations,
                    "games_per_iter": args.games,
                    "epochs_per_iter": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "mcts_sims": args.mcts_sims,
                    "device": device,
                    "gpu": info["name"],
                    "gpu_vram_gb": info["vram_gb"],
                    "hostname": os.uname().nodename,
                },
                tags=[info["name"].split()[0], device],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] Failed: {e}, continuing without W&B", flush=True)
            wandb_run = None

    # ── Load modules ──────────────────────────────────────────────────
    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.training.loss import HexaZeroLoss
    from hexzero.game.interface import CatanGame
    from hexzero.elo.rating import EloRating, MatchResult
    from hexzero.bindings.lib_loader import load_library

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()
    crit = HexaZeroLoss()

    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()

    # ── Model ─────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        print(f"[model] Resuming from {args.resume}", flush=True)
        net = HexaZeroNet.load_checkpoint(args.resume, device=device)
    else:
        net = HexaZeroNet(cfg.network)
        net.to(device)
    print(f"[model] {net.num_parameters:,} params on {device}", flush=True)

    if wandb_run:
        import wandb
        wandb.config.update({"model_params": net.num_parameters})

    # ── ELO tracker ───────────────────────────────────────────────────
    elo = EloRating(k_factor=32.0)
    elo.register_player("AB2", 100.0, pinned=True)
    elo.register_player("HexaZero", 100.0)
    elo.register_player("Random", 100.0)

    # ── Main loop ─────────────────────────────────────────────────────
    t_total = time.time()

    for iteration in range(args.iterations):
        print(f"\n{'='*60}", flush=True)
        print(f" ITERATION {iteration}", flush=True)
        print(f"{'='*60}", flush=True)

        # ── Data generation ───────────────────────────────────────────
        n_games = args.games if iteration > 0 else max(args.games, 200)
        print(f"[data] {n_games} games (iter {iteration})...", flush=True)
        t_data = time.time()
        buf = generate_data(net, state_enc, action_enc, device, n_games, iteration,
                            mcts_sims=args.mcts_sims)
        data_time = time.time() - t_data

        # ── Training ──────────────────────────────────────────────────
        lr = args.lr if iteration == 0 else args.lr * 0.5
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        n_epochs = args.epochs

        print(f"[train] {n_epochs} epochs, bs={args.batch_size}, lr={lr}", flush=True)
        t_train = time.time()

        for ep in range(n_epochs):
            t_ep = time.time()
            avg, nb = train_epoch(net, buf, optimizer, crit, args.batch_size, device)
            ep_time = time.time() - t_ep
            sps = (nb * args.batch_size) / ep_time

            if (ep + 1) % max(n_epochs // 5, 1) == 0 or ep == n_epochs - 1:
                print(f"  E{ep}: loss={avg['total_loss']:.4f} "
                      f"vloss={avg['value_loss']:.4f} ploss={avg['policy_loss']:.4f} "
                      f"vacc={avg['value_accuracy']:.3f} ent={avg['policy_entropy']:.3f} "
                      f"{sps:.0f}s/s ({ep_time:.0f}s)", flush=True)

            if wandb_run:
                import wandb
                wandb.log({
                    "train/total_loss": avg["total_loss"],
                    "train/value_loss": avg["value_loss"],
                    "train/policy_loss": avg["policy_loss"],
                    "train/value_accuracy": avg["value_accuracy"],
                    "train/policy_entropy": avg["policy_entropy"],
                    "train/learning_rate": lr,
                    "train/samples_per_sec": sps,
                    "train/epoch": ep,
                    "train/iteration": iteration,
                })

        train_time = time.time() - t_train

        # ── Evaluation ────────────────────────────────────────────────
        print(f"[eval] {args.eval_games} games vs AB2...", flush=True)
        t_eval = time.time()
        hz_w, ab2_w, rand_w = evaluate_vs_ab2(
            net, state_enc, action_enc, device, lib, args.eval_games)
        eval_time = time.time() - t_eval
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
        hz_wr = hz_w / max(total, 1)

        print(f"  HZ={hz_w} AB2={ab2_w} Rand={rand_w} | "
              f"HZ={hz_wr:.0%} | ELO: HZ={hz_elo:.0f} AB2=100 | "
              f"{eval_time:.0f}s", flush=True)

        if wandb_run:
            import wandb
            wandb.log({
                "eval/hz_wins": hz_w,
                "eval/ab2_wins": ab2_w,
                "eval/rand_wins": rand_w,
                "eval/hz_win_rate": hz_wr,
                "eval/hz_elo": hz_elo,
                "eval/ab2_elo": 100.0,
                "eval/games_played": total,
                "timing/data_gen_s": data_time,
                "timing/training_s": train_time,
                "timing/eval_s": eval_time,
                "iteration": iteration,
            })

        # ── Checkpoint ────────────────────────────────────────────────
        net.save_checkpoint(
            str(ckpt_dir / f"iter{iteration:04d}.pt"),
            metadata={"iteration": iteration, "elo": hz_elo,
                       "policy_loss": avg["policy_loss"], "hz_wins": hz_w})
        net.save_checkpoint(
            str(ckpt_dir / "latest.pt"),
            metadata={"iteration": iteration, "elo": hz_elo})
        print(f"  Saved iter{iteration} checkpoint", flush=True)

    # ── Final summary ─────────────────────────────────────────────────
    total_time = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f" DONE  ({total_time/60:.1f} min total)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Final policy loss: {avg['policy_loss']:.4f}", flush=True)
    print(f"  Final value acc:   {avg['value_accuracy']:.3f}", flush=True)
    print(f"  ELO ratings:", flush=True)
    for r in elo.get_ratings_table():
        pin = " [PINNED]" if r["pinned"] else ""
        print(f"    {r['name']:12s} {r['rating']:7.1f} ({r['games_played']} games){pin}",
              flush=True)

    if wandb_run:
        import wandb
        wandb.summary["final_elo"] = hz_elo
        wandb.summary["final_policy_loss"] = avg["policy_loss"]
        wandb.summary["total_time_min"] = total_time / 60
        wandb.finish()
        print(f"  W&B: {wandb_run.url}", flush=True)


if __name__ == "__main__":
    main()
