#!/usr/bin/env python3
"""R-NaD learner: continuously ingests trajectory files and trains.

Watches a trajectory directory for new .pt files written by actors.
Processes V-trace + reward transformation, runs NERD updates,
saves checkpoints, evaluates vs AB2, logs to W&B.

Usage:
    python -m hexzero.scripts.rnad_learner \
        --trajectory-dir /nlp/scr/nroll/catanatron/trajectories \
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
    parser = argparse.ArgumentParser(description="R-NaD continuous learner")
    parser.add_argument("--trajectory-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--anchor-interval", type=int, default=50,
                        help="Reset anchor every N training steps")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate vs AB2 every N training steps")
    parser.add_argument("--eval-games", type=int, default=24)
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between checking for new trajectories")
    parser.add_argument("--min-trajectories", type=int, default=2,
                        help="Min new trajectory files before training")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max training steps (0=infinite)")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="hexazero")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)
    traj_dir = args.trajectory_dir
    ckpt_dir = args.checkpoint_dir
    processed_dir = os.path.join(traj_dir, "processed")
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

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
    loss_fn = RNaDLoss(eta=args.eta, clip_bound=100.0, value_weight=1.0, entropy_weight=2.0)

    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    print("=" * 60, flush=True)
    print(" R-NaD Learner", flush=True)
    print(f" Device: {device} ({gpu_name})", flush=True)
    print(f" Trajectory dir: {traj_dir}", flush=True)
    print(f" Anchor interval: {args.anchor_interval} steps", flush=True)
    print(f" LR: {args.lr}  BS: {args.batch_size}  eta: {args.eta}", flush=True)
    print("=" * 60, flush=True)

    # ── Model ─────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        net = HexaZeroNet.load_checkpoint(args.resume, device=device)
        print(f"[learner] Resumed from {args.resume}", flush=True)
    else:
        net = HexaZeroNet(cfg.network).to(device)
        print("[learner] Random initialization", flush=True)
    print(f"[learner] {net.num_parameters:,} params", flush=True)

    anchor = HexaZeroNet(cfg.network).to(device)
    anchor.load_state_dict(net.state_dict())
    anchor.eval()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    # Save initial checkpoint so actors can start
    _save_checkpoint(net, ckpt_dir, 0)
    print("[learner] Saved initial checkpoint for actors", flush=True)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"rnad-learner-{os.uname().nodename}",
                config=vars(args),
                tags=["learner", "rnad", device],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] Failed: {e}", flush=True)

    # ── Main loop ─────────────────────────────────────────────────────
    global_step = 0
    total_games_ingested = 0
    total_positions = 0
    t_start = time.time()
    consecutive_above_60 = 0  # track consecutive steps with HZ WR >= 60%

    print("[learner] Waiting for trajectory files...", flush=True)

    while args.max_steps == 0 or global_step < args.max_steps:
        # ── Wait for new trajectory files ─────────────────────────────
        new_files = _find_new_trajectories(traj_dir)
        while len(new_files) < args.min_trajectories:
            time.sleep(args.poll_interval)
            new_files = _find_new_trajectories(traj_dir)

        # ── Ingest trajectories ───────────────────────────────────────
        all_trajectories = []
        # Ingest enough files for a meaningful gradient (~100 games)
        MAX_FILES_PER_STEP = 100
        n_files = 0
        for f in new_files[:MAX_FILES_PER_STEP]:
            try:
                trajs = torch.load(f, weights_only=False, map_location="cpu")
                all_trajectories.extend(trajs)
                n_files += 1
                shutil.move(f, os.path.join(processed_dir, os.path.basename(f)))
            except Exception as e:
                print(f"[learner] Failed to load {f}: {e}", flush=True)

        n_games = len(all_trajectories)
        total_games_ingested += n_games

        # ── Process V-trace ───────────────────────────────────────────
        train_data = _process_trajectories(all_trajectories, args.eta)
        n_pos = len(train_data)
        total_positions += n_pos

        if n_pos < 64:
            print(f"[learner] Too few positions ({n_pos}), skipping", flush=True)
            continue

        print(f"[learner] Ingested {n_files} files, {n_games} games, "
              f"{n_pos} positions (total: {total_games_ingested} games, "
              f"{total_positions} pos)", flush=True)

        # ── Determine training mode ───────────────────────────────────
        selfplay_flag = os.path.exists(os.path.join(ckpt_dir, "SELFPLAY_MODE"))

        if selfplay_flag and global_step > 0 and global_step % args.anchor_interval == 0:
            anchor.load_state_dict(net.state_dict())
            anchor.eval()
            print(f"[learner] Anchor reset at step {global_step}", flush=True)

        # ── Training update ───────────────────────────────────────────
        random.shuffle(train_data)
        net.train()
        step_metrics = {}
        n_batches = 0

        for i in range(0, len(train_data), args.batch_size):
            chunk = train_data[i:i + args.batch_size]
            if len(chunk) < 8:
                continue

            batch_input = {
                "node_features": torch.from_numpy(
                    np.stack([d["nf"] for d in chunk])).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(
                    np.stack([d["ef"] for d in chunk])).to(device),
                "flat_features": torch.from_numpy(
                    np.stack([d["ff"] for d in chunk])).to(device),
                "action_mask": torch.from_numpy(
                    np.stack([d["mask"] for d in chunk])).to(device),
            }
            q_all = torch.from_numpy(
                np.stack([d["q_all"] for d in chunk])).to(device)
            vt_tgt = torch.from_numpy(
                np.stack([d["value_target"] for d in chunk])).to(device)

            optimizer.zero_grad(set_to_none=True)
            out = net(batch_input)

            if selfplay_flag:
                # ── NERD mode (self-play) ─────────────────────────────
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
            else:
                # ── Supervised mode (AB2 imitation) ───────────────────
                mask = batch_input["action_mask"]
                logits = out["policy_logits"].masked_fill(mask == 0, -1e9)
                log_probs = F.log_softmax(logits, dim=-1)

                q_masked = q_all * mask
                q_shifted = q_masked - q_masked.max(dim=-1, keepdim=True).values
                q_shifted = q_shifted.masked_fill(mask == 0, -1e9)
                target_dist = F.softmax(q_shifted / 0.5, dim=-1)

                policy_loss = -(target_dist.detach() * log_probs).sum(dim=-1).mean()
                policy_loss = torch.nan_to_num(policy_loss, nan=0.0)

                # Value: cross-entropy with 4-dim target distribution
                vt_norm = vt_tgt.detach().clamp(min=0.0)
                vt_sum = vt_norm.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                vt_dist = vt_norm / vt_sum
                value_log_probs = F.log_softmax(out["value"], dim=-1)
                value_loss = -(vt_dist * value_log_probs).sum(dim=-1).mean()
                value_loss = torch.nan_to_num(value_loss, nan=0.0)

                entropy = -(log_probs.exp() * log_probs * mask).sum(dim=-1)
                entropy = torch.nan_to_num(entropy, nan=0.0).mean()

                total_loss = policy_loss + value_loss - 0.3 * entropy

                with torch.no_grad():
                    vp_winner = F.softmax(out["value"], dim=-1).argmax(dim=-1)
                    vt_winner = vt_dist.argmax(dim=-1)
                    vacc = (vp_winner == vt_winner).float().mean()

                losses = {
                    "total_loss": total_loss,
                    "nerd_loss": policy_loss,
                    "value_loss": value_loss,
                    "policy_entropy": entropy,
                    "mean_advantage": torch.tensor(0.0),
                    "value_accuracy": vacc,
                }

            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                val = v.item() if isinstance(v, Tensor) else float(v)
                step_metrics[k] = step_metrics.get(k, 0.0) + val
            n_batches += 1

        global_step += 1

        if n_batches > 0:
            avg = {k: v / n_batches for k, v in step_metrics.items()}
        else:
            avg = {"total_loss": 0, "nerd_loss": 0, "value_loss": 0,
                   "policy_entropy": 0, "mean_advantage": 0, "value_accuracy": 0}

        # ── Save checkpoint ───────────────────────────────────────────
        _save_checkpoint(net, ckpt_dir, global_step)

        # Count wins per player from trajectory rewards
        # Each trajectory = one game. Winner has reward 1.0.
        # AB2 seat's steps have q_all with 0.8 values (imitation signal).
        step_ab2_w = 0
        step_hz_w = 0
        step_games = 0
        for traj in all_trajectories:
            if not traj:
                continue
            step_games += 1
            # Find the winner: the player whose last step has highest reward
            last_rewards = {}
            for s in traj:
                last_rewards[s["player"]] = s.get("reward", 0)
            if not last_rewards:
                continue
            winner_player = max(last_rewards, key=last_rewards.get)
            winner_reward = last_rewards[winner_player]
            if winner_reward < 0.5:
                continue  # timeout, no clear winner
            # AB2 seat is identified by having q_all with 0.8 peak
            winner_steps = [s for s in traj if s["player"] == winner_player]
            if winner_steps:
                q = winner_steps[0].get("q_all")
                if q is not None and float(np.max(q)) > 0.75:
                    step_ab2_w += 1
                else:
                    step_hz_w += 1

        step_total = step_ab2_w + step_hz_w
        step_hz_wr = step_hz_w / max(step_total, 1)

        # Auto-switch: 20% per-seat HZ WR for 3 consecutive steps
        # (25% = matching AB2, 20% = close enough to switch)
        if not os.path.exists(os.path.join(ckpt_dir, "SELFPLAY_MODE")):
            if step_total >= 10 and step_hz_wr >= 0.20:
                consecutive_above_60 += 1
                if consecutive_above_60 >= 3:
                    switch_file = os.path.join(ckpt_dir, "SELFPLAY_MODE")
                    with open(switch_file, "w") as f:
                        f.write(f"Switched at step {global_step}, hz_wr={step_hz_wr:.1%}\n")
                    print(f"[learner] *** SWITCHING TO SELF-PLAY + NERD *** "
                          f"(HZ per-seat WR >= 20% for 3 consecutive steps)", flush=True)
            else:
                consecutive_above_60 = 0

        elapsed = time.time() - t_start
        gps = total_games_ingested / max(elapsed, 0.01)
        print(f"[learner] Step {global_step}: nerd={avg.get('nerd_loss',0):.4f} "
              f"vloss={avg.get('value_loss',0):.4f} "
              f"vacc={avg.get('value_accuracy',0):.3f} "
              f"ent={avg.get('policy_entropy',0):.3f} "
              f"| {n_pos} pos, {n_batches} batches "
              f"| AB2={step_ab2_w} HZ={step_hz_w} ({step_hz_wr:.0%}) "
              f"| total {total_games_ingested} games ({gps:.1f} g/s)",
              flush=True)

        # ── W&B logging ───────────────────────────────────────────────
        if wandb_run:
            import wandb
            wandb.log({
                "train/nerd_loss": avg.get("nerd_loss", 0),
                "train/value_loss": avg.get("value_loss", 0),
                "train/total_loss": avg.get("total_loss", 0),
                "train/policy_entropy": avg.get("policy_entropy", 0),
                "train/mean_advantage": avg.get("mean_advantage", 0),
                "train/value_accuracy": avg.get("value_accuracy", 0),
                "train/positions_per_step": n_pos,
                "train/total_games": total_games_ingested,
                "train/total_positions": total_positions,
                "train/games_per_sec": gps,
                "train/global_step": global_step,
                "train/step_ab2_wins": step_ab2_w,
                "train/step_hz_wins": step_hz_w,
                "train/step_hz_winrate": step_hz_wr,
            })

        # ── Evaluate ──────────────────────────────────────────────────
        if global_step % args.eval_every == 0:
            print(f"[learner] Evaluating vs AB2 ({args.eval_games} games)...",
                  flush=True)
            hz_w, ab2_w, rand_w = _evaluate(
                net, state_enc, action_enc, device, lib,
                args.eval_games, global_step)
            total = hz_w + ab2_w + rand_w

            # Fresh ELO per eval round (not cumulative)
            eval_elo = EloRating(k_factor=32.0)
            eval_elo.register_player("AB2", 1000.0, pinned=True)
            eval_elo.register_player("HexaZero-RNaD", 1000.0)
            for _ in range(hz_w):
                eval_elo.update_ratings(MatchResult(
                    ["HexaZero-RNaD", "AB2", "Random", "Random"],
                    "HexaZero-RNaD", 0, 0, 0, time.time()))
            for _ in range(ab2_w):
                eval_elo.update_ratings(MatchResult(
                    ["HexaZero-RNaD", "AB2", "Random", "Random"],
                    "AB2", 1, 0, 0, time.time()))
            for _ in range(rand_w):
                eval_elo.update_ratings(MatchResult(
                    ["HexaZero-RNaD", "AB2", "Random", "Random"],
                    "Random", 2, 0, 0, time.time()))

            hz_elo = eval_elo.get_rating("HexaZero-RNaD")
            hz_wr = hz_w / max(total, 1)
            ab2_wr = ab2_w / max(total, 1)

            # Also compute AB2 win rate from selfplay games (actors report in filenames)
            # by counting recent trajectory game outcomes
            sp_ab2_wins = 0
            sp_hz_wins = 0
            for traj in all_trajectories[-100:]:  # last ~100 games
                if not traj:
                    continue
                last = traj[-1]
                if last.get("terminal"):
                    reward = last.get("reward", 0)
                    if reward >= 0.9:
                        sp_hz_wins += 1
                    elif reward <= 0.05:
                        sp_ab2_wins += 1
            sp_total = sp_ab2_wins + sp_hz_wins
            sp_hz_wr = sp_hz_wins / max(sp_total, 1)

            print(f"[learner] EVAL: HZ={hz_w} AB2={ab2_w} Rand={rand_w} | "
                  f"eval_wr={hz_wr:.1%} | ELO={hz_elo:.0f} | "
                  f"selfplay_hz_wr={sp_hz_wr:.1%}", flush=True)

            # Eval-based auto-switch removed; using training WR instead

            if wandb_run:
                import wandb
                wandb.log({
                    "eval/hz_wins": hz_w,
                    "eval/ab2_wins": ab2_w,
                    "eval/rand_wins": rand_w,
                    "eval/hz_win_rate": hz_wr,
                    "eval/ab2_win_rate": ab2_wr,
                    "eval/hz_elo": hz_elo,
                    "eval/selfplay_hz_wr": sp_hz_wr,
                })

    print(f"[learner] Done: {global_step} steps, {total_games_ingested} games, "
          f"{time.time() - t_start:.0f}s", flush=True)
    if wandb_run:
        import wandb
        wandb.finish()


def _save_checkpoint(net, ckpt_dir, step):
    path = os.path.join(ckpt_dir, "latest.pt")
    net.save_checkpoint(path, metadata={"step": step})


def _find_new_trajectories(traj_dir):
    files = []
    for f in os.listdir(traj_dir):
        if f.startswith("actor") and f.endswith(".pt"):
            files.append(os.path.join(traj_dir, f))
    return sorted(files)


def _process_trajectories(all_trajectories, eta):
    """Process trajectories. Value target = 4-dim win distribution (AlphaZero-style)."""

    all_data = []
    for traj in all_trajectories:
        if len(traj) < 2:
            continue

        # Get per-game reward vector (4 floats, one per seat)
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

            # Rotate reward_vec so index 0 = current player
            rot_reward = np.roll(reward_vec, -player_id).copy()
            rsum = rot_reward.sum()
            if rsum > 1e-8:
                rot_reward_dist = rot_reward / rsum
            else:
                rot_reward_dist = np.ones(4, dtype=np.float32) / 4.0

            for t in range(len(steps)):
                q_all = steps[t].get("q_all", np.zeros(337, dtype=np.float32))
                all_data.append({
                    "nf": steps[t]["nf"],
                    "ef": steps[t]["ef"],
                    "ff": steps[t]["ff"],
                    "mask": steps[t]["mask"],
                    "q_all": q_all,
                    "value_target": rot_reward_dist,
                })
    return all_data


def _evaluate(net, state_enc, action_enc, device, lib, num_games, step):
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    net.eval()
    hz_w = ab2_w = rand_w = 0
    for gi in range(num_games):
        g = CatanGame(seed=70000 + step * 100 + gi)
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
                            bb["action_mask"] = action_enc.get_action_mask(
                                cl).unsqueeze(0).to(device)
                        with torch.no_grad():
                            v = F.softmax(net(bb)["value"], dim=-1)[0, 0].item()
                    if v > bv:
                        bv = v; bi = i
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
                        bv = v; bi = i
                g.step(bi)
            else:
                g.step(random.randrange(len(le)))
        w = g.winner()
        if w == hz_s: hz_w += 1
        elif w == ab2_s: ab2_w += 1
        elif w is not None: rand_w += 1
    return hz_w, ab2_w, rand_w


if __name__ == "__main__":
    main()
