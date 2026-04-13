#!/usr/bin/env python3
"""R-NaD training pipeline for HexaZero.

No MCTS. Direct policy sampling at C engine speed.
83x faster than AlphaZero MCTS self-play.

Architecture:
    Outer loop: reset anchor every ANCHOR_INTERVAL steps.
    Inner loop: play N concurrent games -> reward transform -> V-trace -> NERD update.

Usage (single GPU):
    python -m hexzero.scripts.rnad_train --device cuda --outer-steps 50
    python -m hexzero.scripts.rnad_train --device mps --concurrent 16 --outer-steps 10

Usage (jag cluster):
    nlprun -q jag -g 1 -r 60G -c 16 -p standard -n rnad-v1 \
        'eval "$(/nlp/scr/nroll/miniconda3/bin/conda shell.bash hook)" && \
         conda activate hexazero && cd /nlp/scr/nroll/catanatron && \
         python -m hexzero.scripts.rnad_train --outer-steps 50 --concurrent 64'
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import math
import os
import random
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def main():
    parser = argparse.ArgumentParser(description="HexaZero R-NaD Training")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--concurrent", type=int, default=32,
                        help="Concurrent games per self-play batch (GPU batch size)")
    parser.add_argument("--outer-steps", type=int, default=50,
                        help="Number of anchor resets")
    parser.add_argument("--inner-steps", type=int, default=200,
                        help="Training steps per anchor (inner loop)")
    parser.add_argument("--eta", type=float, default=0.2,
                        help="R-NaD regularization strength")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate every N outer steps")
    parser.add_argument("--checkpoint-dir", type=str, default="hexzero/checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Seed offset for game diversity across agents")
    parser.add_argument("--auto-resume", type=str, default=None,
                        help="Path to shared best.pt; auto-reload if newer at each outer step")
    parser.add_argument("--agent-id", type=int, default=0,
                        help="Agent ID for multi-agent PBT runs")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="hexazero")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.training.rnad_loss import RNaDLoss
    from hexzero.training.vtrace import compute_vtrace, transform_rewards
    from hexzero.elo.rating import EloRating, MatchResult
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()

    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    print("=" * 60, flush=True)
    print(" HexaZero R-NaD Training (DeepNash-style)", flush=True)
    print(f" Device: {device} ({gpu_name})", flush=True)
    print(f" Concurrent: {args.concurrent} games/batch", flush=True)
    print(f" Schedule: {args.outer_steps} outer x {args.inner_steps} inner", flush=True)
    print(f" eta={args.eta} lr={args.lr} bs={args.batch_size}", flush=True)
    print("=" * 60, flush=True)

    # ── Model ─────────────────────────────────────────────────────────
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[model] Loading checkpoint: {args.resume}", flush=True)
            net = HexaZeroNet.load_checkpoint(args.resume, device=device)
            print(f"[model] Checkpoint loaded successfully", flush=True)
        else:
            print(f"[model] WARNING: --resume path does not exist: {args.resume}", flush=True)
            print(f"[model] Starting from random initialization instead", flush=True)
            net = HexaZeroNet(cfg.network).to(device)
    else:
        print(f"[model] No --resume, starting from random initialization", flush=True)
        net = HexaZeroNet(cfg.network).to(device)
    print(f"[model] {net.num_parameters:,} params on {device}", flush=True)

    print(f"[model] Creating anchor network...", flush=True)
    anchor = HexaZeroNet(cfg.network).to(device)
    anchor.load_state_dict(net.state_dict())
    anchor.eval()
    print(f"[model] Anchor created and synced", flush=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = RNaDLoss(eta=args.eta, clip_bound=10_000.0, value_weight=0.5)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            run_name = args.wandb_name or f"rnad-agent{args.agent_id}-{os.uname().nodename}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                tags=["rnad", device, f"agent{args.agent_id}"],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] Failed: {e}", flush=True)

    # ── ELO ───────────────────────────────────────────────────────────
    elo = EloRating(k_factor=32.0)
    elo.register_player("AB2", 1000.0, pinned=True)
    elo.register_player("HexaZero-RNaD", 1000.0)

    # ── Main loop ─────────────────────────────────────────────────────
    global_step = 0
    t_total = time.time()

    _last_auto_resume_mtime = 0.0

    for outer in range(args.outer_steps):
        # ── Auto-resume from shared best.pt if newer ──────────────────
        if args.auto_resume and os.path.exists(args.auto_resume):
            try:
                mtime = os.path.getmtime(args.auto_resume)
                if mtime > _last_auto_resume_mtime + 10:
                    print(f"[outer {outer}] Auto-resuming from {args.auto_resume}", flush=True)
                    best_sd = torch.load(args.auto_resume, map_location=device, weights_only=False)
                    net.load_state_dict(best_sd["model_state_dict"])
                    _last_auto_resume_mtime = mtime
                    print(f"[outer {outer}] Loaded shared best.pt", flush=True)
            except Exception as e:
                print(f"[outer {outer}] Auto-resume failed: {e}", flush=True)

        # ── Reset anchor ──────────────────────────────────────────────
        anchor.load_state_dict(net.state_dict())
        anchor.eval()
        print(f"\n[outer {outer}] Anchor reset (agent {args.agent_id})", flush=True)

        for inner in range(args.inner_steps):
            t_step = time.time()
            print(f"  [{outer}.{inner:3d}] Starting self-play...", flush=True)

            # ── Self-play: N concurrent games ─────────────────────────
            net.eval()
            seed_base = args.seed_offset + global_step * 10000
            trajectories = _play_games(
                net, anchor, state_enc, action_enc, device,
                args.concurrent, seed_base,
            )
            net.train()
            t_sp = time.time() - t_step

            n_pos = sum(len(t) for t in trajectories)
            n_fin = sum(1 for t in trajectories if t and t[-1]["terminal"])
            print(f"  [{outer}.{inner:3d}] Self-play done: {n_pos} pos, "
                  f"{n_fin}/{len(trajectories)} finished, {t_sp:.1f}s", flush=True)

            # ── Process trajectories into training data ───────────────
            t_proc = time.time()
            train_data = _process_trajectories(
                trajectories, args.eta, args.batch_size, device,
            )
            print(f"  [{outer}.{inner:3d}] Processed: {len(train_data)} training examples "
                  f"({time.time()-t_proc:.1f}s)", flush=True)

            if not train_data:
                print(f"  [{outer}.{inner:3d}] No training data, skipping", flush=True)
                global_step += 1
                continue

            # ── NERD + value update ───────────────────────────────────
            step_metrics = _train_on_data(
                net, anchor, optimizer, loss_fn, train_data, device,
                args.batch_size, state_enc._edge_index.to(device),
            )
            global_step += 1

            # ── Logging ───────────────────────────────────────────────
            step_time = time.time() - t_step
            n_positions = sum(len(t) for t in trajectories)
            n_games = len(trajectories)
            games_finished = sum(1 for t in trajectories if t and t[-1]["terminal"])

            if (inner + 1) % 5 == 0 or inner <= 2 or inner == args.inner_steps - 1:
                print(
                    f"  [{outer}.{inner:3d}] "
                    f"nerd={step_metrics['nerd_loss']:.4f} "
                    f"vloss={step_metrics['value_loss']:.4f} "
                    f"ent={step_metrics['policy_entropy']:.3f} "
                    f"pos={n_positions} "
                    f"fin={games_finished}/{n_games} "
                    f"{step_time:.1f}s",
                    flush=True,
                )

            if wandb_run:
                import wandb
                wandb.log({
                    "train/nerd_loss": step_metrics["nerd_loss"],
                    "train/value_loss": step_metrics["value_loss"],
                    "train/total_loss": step_metrics["total_loss"],
                    "train/policy_entropy": step_metrics["policy_entropy"],
                    "train/mean_advantage": step_metrics["mean_advantage"],
                    "train/positions": n_positions,
                    "train/games_finished": games_finished,
                    "train/step_time": step_time,
                    "train/global_step": global_step,
                    "train/outer_step": outer,
                })

        # ── Checkpoint ────────────────────────────────────────────────
        ckpt_path = os.path.join(args.checkpoint_dir, f"rnad_outer{outer:04d}.pt")
        latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
        net.save_checkpoint(ckpt_path, metadata={"outer": outer, "step": global_step})
        net.save_checkpoint(latest_path, metadata={"outer": outer, "step": global_step})

        # ── Evaluate ──────────────────────────────────────────────────
        if (outer + 1) % args.eval_every == 0 or outer == args.outer_steps - 1:
            print(f"[eval] {args.eval_games} games vs AB2...", flush=True)
            hz_w, ab2_w, rand_w = _evaluate(
                net, state_enc, action_enc, device, lib, args.eval_games, outer,
            )
            total = hz_w + ab2_w + rand_w
            for _ in range(hz_w):
                elo.update_ratings(MatchResult(
                    ["HexaZero-RNaD", "AB2", "Random", "Random"],
                    "HexaZero-RNaD", 0, 0, 0, time.time()))
            for _ in range(ab2_w):
                elo.update_ratings(MatchResult(
                    ["HexaZero-RNaD", "AB2", "Random", "Random"],
                    "AB2", 1, 0, 0, time.time()))

            hz_elo = elo.get_rating("HexaZero-RNaD")
            print(f"  HZ={hz_w} AB2={ab2_w} Rand={rand_w} | "
                  f"ELO={hz_elo:.0f} (AB2=1000)", flush=True)

            if wandb_run:
                import wandb
                wandb.log({
                    "eval/hz_wins": hz_w,
                    "eval/ab2_wins": ab2_w,
                    "eval/hz_win_rate": hz_w / max(total, 1),
                    "eval/hz_elo": hz_elo,
                    "eval/outer": outer,
                })

    elapsed = time.time() - t_total
    print(f"\nDone: {global_step} steps in {elapsed/60:.1f} min", flush=True)
    print(f"ELO: HexaZero-RNaD={elo.get_rating('HexaZero-RNaD'):.0f} AB2=1000", flush=True)

    if wandb_run:
        import wandb
        wandb.finish()


# ======================================================================
# Self-play: no MCTS, direct policy sampling, batched inference
# ======================================================================

def _play_games(net, anchor, state_enc, action_enc, device, n_games, seed_base):
    """Play n_games concurrently with batched forward passes.

    Optimizations over naive approach:
    - Pre-allocated numpy buffers for state encoding (no torch.zeros per step)
    - Only run anchor forward pass, don't store full 337-dim log-probs per step
      (recompute during training instead)
    - Single GPU transfer per batch, keep action sampling on CPU
    - Minimal per-step dict (scalars only, no tensor copies)
    """
    from hexzero.game.interface import CatanGame

    N = state_enc.num_nodes
    E = state_enc.num_edges
    NF_DIM = state_enc.NODE_FEATURE_DIM
    EF_DIM = state_enc.EDGE_FEATURE_DIM
    FF_DIM = state_enc.FLAT_FEATURE_DIM
    ACTION_DIM = 337

    print(f"    [selfplay] Creating {n_games} games...", flush=True)
    games = [CatanGame(seed=seed_base + i) for i in range(n_games)]
    for g in games:
        g.reset()
    trajectories = [[] for _ in range(n_games)]
    active = list(range(n_games))
    move_count = 0
    t_start = time.time()
    last_log = t_start
    games_done = 0
    batch_count = 0

    # Pre-allocate numpy buffers for batch encoding
    nf_buf = np.zeros((n_games, N, NF_DIM), dtype=np.float32)
    ef_buf = np.zeros((n_games, E, EF_DIM), dtype=np.float32)
    ff_buf = np.zeros((n_games, FF_DIM), dtype=np.float32)
    mask_buf = np.zeros((n_games, ACTION_DIM), dtype=np.float32)

    # Edge index on device (static, upload once)
    edge_index_dev = state_enc._edge_index.to(device)

    print(f"    [selfplay] Starting game loop ({n_games} active)...", flush=True)

    while active:
        B = 0
        index_map = []
        for idx in active:
            g = games[idx]
            if g.is_terminal() or g.turn_number >= 750 or len(trajectories[idx]) >= 2000:
                continue
            sv = g.get_state_view()
            state_enc.encode_into(sv, nf_buf[B], ef_buf[B], ff_buf[B])
            le = g.get_legal_actions()
            mask_buf[B] = action_enc.get_action_mask(le).numpy()
            index_map.append((idx, le))
            B += 1

        if B == 0:
            break

        # Single CPU->GPU transfer for the whole batch
        batch = {
            "node_features": torch.from_numpy(nf_buf[:B]).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(ef_buf[:B]).to(device),
            "flat_features": torch.from_numpy(ff_buf[:B]).to(device),
            "action_mask": torch.from_numpy(mask_buf[:B]).to(device),
        }

        if batch_count == 0:
            print(f"    [selfplay] First forward pass (batch={B})...", flush=True)

        with torch.no_grad():
            out = net(batch)
            anchor_out = anchor(batch)

        if batch_count == 0:
            print(f"    [selfplay] First forward pass complete", flush=True)

        batch_count += 1

        # Pull results to CPU in one shot
        pi_probs_cpu = out["policy_probs"].cpu().numpy()
        values_cpu = F.softmax(out["value"], dim=-1)[:, 0].cpu().numpy()
        log_pi_cpu = F.log_softmax(out["policy_logits"], dim=-1).cpu().numpy()
        anchor_log_pi_cpu = F.log_softmax(anchor_out["policy_logits"], dim=-1).cpu().numpy()

        still_active = []
        for b, (idx, le) in enumerate(index_map):
            g = games[idx]
            probs = pi_probs_cpu[b]

            if probs.sum() < 1e-6:
                probs = mask_buf[b] / mask_buf[b].sum()

            action_idx = int(np.random.choice(ACTION_DIM, p=probs))

            # Q-values: use V(s) as Q for all actions (simple baseline)
            # The distributed actor does proper 1-ply lookahead
            q_all = np.full(ACTION_DIM, float(values_cpu[b]), dtype=np.float32)
            q_all = q_all * mask_buf[b]

            trajectories[idx].append({
                "nf": nf_buf[b].copy(),
                "ef": ef_buf[b].copy(),
                "ff": ff_buf[b].copy(),
                "mask": mask_buf[b].copy(),
                "q_all": q_all,
                "action_idx": action_idx,
                "player": g.current_player(),
                "value": float(values_cpu[b]),
                "terminal": False,
            })

            chosen = next((i for i, a in enumerate(le)
                           if action_enc.encode(a) == action_idx), 0)
            g.step(chosen)

            if not g.is_terminal() and g.turn_number < 750 and len(trajectories[idx]) < 2000:
                still_active.append(idx)
            elif trajectories[idx]:
                trajectories[idx][-1]["terminal"] = True
                games_done += 1

        move_count += B
        active = still_active

        now = time.time()
        if now - last_log > 10.0:
            elapsed = now - t_start
            avg_turns = sum(g.turn_number for g in games) / n_games
            print(f"    [selfplay] {games_done}/{n_games} done | "
                  f"{len(active)} active | {move_count} moves | "
                  f"avg_turn={avg_turns:.0f} | {elapsed:.0f}s",
                  flush=True)
            last_log = now

    # Assign terminal rewards with graded VP-based scoring for timeouts
    winners = [0, 0, 0, 0, 0]
    for idx in range(n_games):
        g = games[idx]
        winner = g.winner()
        n_players = g.num_players

        vps = [g._game.state.player_state[i][0] for i in range(n_players)]
        ranked = sorted(range(n_players), key=lambda i: vps[i], reverse=True)
        timed_out = winner is None

        if timed_out:
            grade = {ranked[0]: 0.1, ranked[1]: 0.05,
                     ranked[2]: 0.02, ranked[3]: 0.0}
            winners[4] += 1
        else:
            grade = {ranked[0]: 1.0, ranked[1]: 0.3,
                     ranked[2]: 0.1, ranked[3]: 0.0}
            winners[winner] += 1
            grade[winner] = 1.0

        for step in trajectories[idx]:
            step["reward"] = grade.get(step["player"], 0.0)

    elapsed = time.time() - t_start
    total_pos = sum(len(t) for t in trajectories)
    print(f"    [selfplay] Complete: {games_done}/{n_games} games, "
          f"{total_pos} pos, winners={winners[:4]} timeouts={winners[4]}, "
          f"{elapsed:.1f}s ({n_games/max(elapsed,0.01):.1f} g/s)", flush=True)

    return trajectories


# ======================================================================
# Trajectory processing: reward transform + V-trace
# ======================================================================

def _process_trajectories(trajectories, eta, batch_size, device):
    """Convert raw trajectories into training batches. Value target = 4-dim distribution."""

    all_data = []

    for traj in trajectories:
        if len(traj) < 2:
            continue

        if "reward_vec" in traj[0]:
            reward_vec = traj[0]["reward_vec"]
        else:
            reward_vec = np.zeros(4, dtype=np.float32)
            last_steps = {}
            for step in traj:
                last_steps[step["player"]] = step
            for pid, step in last_steps.items():
                reward_vec[pid] = step.get("reward", 0.0)

        for player_id in range(4):
            steps = [s for s in traj if s["player"] == player_id]
            if len(steps) < 2:
                continue

            rot_reward = np.roll(reward_vec, -player_id).copy()
            rsum = rot_reward.sum()
            if rsum > 1e-8:
                rot_reward_dist = rot_reward / rsum
            else:
                rot_reward_dist = np.ones(4, dtype=np.float32) / 4.0

            for t in range(len(steps)):
                all_data.append({
                    "nf": steps[t]["nf"],
                    "ef": steps[t]["ef"],
                    "ff": steps[t]["ff"],
                    "mask": steps[t]["mask"],
                    "q_all": steps[t].get("q_all", np.zeros(337, dtype=np.float32)),
                    "value_target": rot_reward_dist,
                })

    return all_data


# ======================================================================
# Training step
# ======================================================================

def _train_on_data(net, anchor, optimizer, loss_fn, data, device, batch_size,
                    edge_index_dev):
    """Run NERD gradient updates on processed trajectory data.

    Recomputes anchor log-probs during training (cheaper than storing
    337-dim vectors per step during self-play).
    """
    random.shuffle(data)
    net.train()

    total_metrics = {}
    n_batches = 0

    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        if len(chunk) < 8:
            continue

        B = len(chunk)
        batch_input = {
            "node_features": torch.from_numpy(np.stack([d["nf"] for d in chunk])).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(np.stack([d["ef"] for d in chunk])).to(device),
            "flat_features": torch.from_numpy(np.stack([d["ff"] for d in chunk])).to(device),
            "action_mask": torch.from_numpy(np.stack([d["mask"] for d in chunk])).to(device),
        }

        q_all = torch.from_numpy(np.stack([d["q_all"] for d in chunk])).to(device)
        vt_tgt = torch.from_numpy(
            np.stack([d["value_target"] for d in chunk])).to(device)

        optimizer.zero_grad(set_to_none=True)

        out = net(batch_input)

        with torch.no_grad():
            anchor_out = anchor(batch_input)
            anchor_lp = F.log_softmax(anchor_out["policy_logits"], dim=-1)

        losses = loss_fn(
            policy_logits=out["policy_logits"],
            anchor_log_probs=anchor_lp,
            action_mask=batch_input["action_mask"],
            q_all=q_all,
            value_targets=vt_tgt,
            value_logits=out["value"],
        )

        losses["total_loss"].backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        for k, v in losses.items():
            val = v.item() if isinstance(v, Tensor) else float(v)
            total_metrics[k] = total_metrics.get(k, 0.0) + val
        n_batches += 1

    if n_batches == 0:
        return {"total_loss": 0, "nerd_loss": 0, "value_loss": 0,
                "policy_entropy": 0, "mean_advantage": 0}

    return {k: v / n_batches for k, v in total_metrics.items()}


# ======================================================================
# Evaluation
# ======================================================================

def _evaluate(net, state_enc, action_enc, device, lib, num_games, seed_offset):
    """HexaZero (greedy value) vs AB2 (greedy heuristic) vs 2 random."""
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    net.eval()
    hz_w = ab2_w = rand_w = 0

    for gi in range(num_games):
        g = CatanGame(seed=60000 + seed_offset * 1000 + gi)
        g.reset()
        hz_s, ab2_s = gi % 4, (gi + 1) % 4

        while not g.is_terminal() and g.turn_number < 1000:
            cp = g.current_player()
            le = g.get_legal_actions()
            if not le:
                break

            if cp == hz_s:
                bi, bv = 0, -1e9
                for i in range(len(le)):
                    c = g.clone(); c.step(i)
                    if c.is_terminal():
                        v = 10.0 if c.winner() == hz_s else -10.0
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
        if w == hz_s:
            hz_w += 1
        elif w == ab2_s:
            ab2_w += 1
        elif w is not None:
            rand_w += 1

    return hz_w, ab2_w, rand_w


if __name__ == "__main__":
    main()
