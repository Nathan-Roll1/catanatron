#!/usr/bin/env python3
"""R-NaD self-play actor: generates trajectory data for the learner.

No training, no optimizer, no backward pass. Pure inference + C engine.
Writes trajectory files to a shared directory for the learner to ingest.
Reloads latest.pt periodically to stay in sync with the learner.

Supports --num-processes to spawn multiple actor sub-processes on the
same GPU, each pinned to its own CPU core. This saturates both GPU
(via CUDA time-slicing) and CPU (multi-core encoding).

Usage:
    # Single process (8 concurrent games):
    python -m hexzero.scripts.rnad_actor --actor-id 0 ...

    # 4 sub-processes on one GPU (32 concurrent games, 4 CPU cores):
    python -m hexzero.scripts.rnad_actor --actor-id 0 --num-processes 4 ...
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import random
import numpy as np
import torch
import torch.nn.functional as F


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def main():
    parser = argparse.ArgumentParser(description="R-NaD self-play actor")
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--trajectory-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Sub-processes per GPU (each gets own CPU core)")
    parser.add_argument("--reload-every", type=int, default=50,
                        help="Reload checkpoint every N games")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Stop after N games (0=infinite)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.num_processes > 1:
        _run_multi_process(args)
        return

    device = detect_device(args.device)
    traj_dir = args.trajectory_dir
    ckpt_dir = args.checkpoint_dir
    os.makedirs(traj_dir, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    import ctypes

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()
    tmp = CatanGame(seed=0); tmp.reset()
    state_enc = tmp.make_state_encoder()

    net, ckpt_mtime = _load_checkpoint(ckpt_dir, cfg, device)
    anchor = HexaZeroNet(cfg.network).to(device)
    anchor.load_state_dict(net.state_dict())
    anchor.eval()

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    print(f"[actor {args.actor_id}] {device} ({gpu_name}) | "
          f"{net.num_parameters:,} params | concurrent={args.concurrent}",
          flush=True)

    N = state_enc.num_nodes
    E = state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM
    AD = 337

    # Pre-allocate buffers
    nf_buf = np.zeros((args.concurrent, N, NF), dtype=np.float32)
    ef_buf = np.zeros((args.concurrent, E, EF), dtype=np.float32)
    ff_buf = np.zeros((args.concurrent, FF), dtype=np.float32)
    mask_buf = np.zeros((args.concurrent, AD), dtype=np.float32)
    edge_index_dev = state_enc._edge_index.to(device)

    total_games = 0
    batch_num = 0
    seed_base = args.actor_id * 10_000_000 + int(time.time()) % 10_000_000
    t_start = time.time()

    # AB2 plays one seat per game, rotating across games
    from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS

    def _ab2_pick(game, le):
        """AB2 greedy 1-ply: pick action maximizing base_value_fn."""
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        ch = CGame(); ca = (CAction * MAX_ACTIONS)(); cn = ctypes.c_int(0)
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv:
                bv = v; bi = i
        return bi

    while args.max_games == 0 or total_games < args.max_games:
        # ── Play a batch of concurrent games ──────────────────────────
        games = [CatanGame(seed=seed_base + total_games + i)
                 for i in range(args.concurrent)]
        for g in games:
            g.reset()
        trajectories = [[] for _ in range(args.concurrent)]
        active = list(range(args.concurrent))
        games_done = 0
        t_batch = time.time()
        last_log = t_batch
        move_count = 0

        # Supervised: 3 AB2 + 1 HZ. Self-play: 1 AB2 + 3 HZ.
        selfplay_mode = os.path.exists(os.path.join(ckpt_dir, "SELFPLAY_MODE"))
        hz_seat = random.randrange(4)
        if selfplay_mode:
            ab2_seats = set([hz_seat])  # only 1 AB2, rest are HZ
            # (reusing hz_seat var -- in self-play, HZ is everyone except this one)
            ab2_seats = set([random.randrange(4)])
        else:
            all_seats = set([0, 1, 2, 3])
            hz_seat = random.randrange(4)
            ab2_seats = all_seats - set([hz_seat])  # 3 AB2 + 1 HZ

        net.eval()
        anchor.eval()

        while active:
            # ── Handle AB2 seats ──────────────────────────────────────
            ab2_stepped = True
            while ab2_stepped:
                ab2_stepped = False
                for idx in list(active):
                    g = games[idx]
                    if g.is_terminal() or g.turn_number >= 750 or len(trajectories[idx]) >= 2000:
                        continue
                    if g.current_player() in ab2_seats:
                        le = g.get_legal_actions()
                        if le:
                                # Record AB2's state as training data
                                state_enc.encode_into(g.get_state_view(),
                                                      nf_buf[0], ef_buf[0], ff_buf[0])
                                mask = action_enc.get_action_mask(le).numpy()
                                chosen = _ab2_pick(g, le)

                                # AB2's action as one-hot Q target
                                # (sharp: 1.0 for chosen, V(s) for rest)
                                q_ab2 = np.full(AD, 0.3, dtype=np.float32) * mask
                                q_ab2[action_enc.encode(le[chosen])] = 0.8

                                trajectories[idx].append({
                                    "nf": nf_buf[0].copy(),
                                    "ef": ef_buf[0].copy(),
                                    "ff": ff_buf[0].copy(),
                                    "mask": mask.copy(),
                                    "q_all": q_ab2,
                                    "action_idx": action_enc.encode(le[chosen]),
                                    "player": g.current_player(),
                                    "value": 0.5,
                                    "terminal": False,
                                })
                                g.step(chosen)
                                ab2_stepped = True

            # ── Encode current states for HexaZero seats ──────────────
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

            # ── Forward pass on current states ────────────────────────
            batch = {
                "node_features": torch.from_numpy(nf_buf[:B]).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B]).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B]).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B]).to(device),
            }

            with torch.no_grad():
                out = net(batch)

            pi_probs_cpu = out["policy_probs"].cpu().numpy()
            values_cpu = F.softmax(out["value"], dim=-1)[:, 0].cpu().numpy()

            TOP_K = 5
            child_nfs = []
            child_efs = []
            child_ffs = []
            child_masks_list = []
            child_map = []  # (game_b_idx, action_enc_idx, is_terminal, terminal_q)

            for b, (idx, le) in enumerate(index_map):
                g = games[idx]
                cp = g.current_player()

                # Pick top-K actions by policy probability
                n_le = len(le)
                if n_le <= TOP_K:
                    eval_indices = list(range(n_le))
                else:
                    action_probs = [float(pi_probs_cpu[b, action_enc.encode(a)])
                                    for a in le]
                    eval_indices = sorted(range(n_le),
                                          key=lambda i: action_probs[i],
                                          reverse=True)[:TOP_K]

                for i in eval_indices:
                    action = le[i]
                    enc_idx = action_enc.encode(action)
                    child = g.clone()
                    child.step(i)
                    if child.is_terminal():
                        w = child.winner()
                        tq = 1.0 if w == cp else 0.0
                        child_map.append((b, enc_idx, True, tq))
                    else:
                        c_nf = np.zeros((N, NF), dtype=np.float32)
                        c_ef = np.zeros((E, EF), dtype=np.float32)
                        c_ff = np.zeros(FF, dtype=np.float32)
                        state_enc.encode_into(child.get_state_view(),
                                              c_nf, c_ef, c_ff)
                        cl = child.get_legal_actions()
                        c_mask = action_enc.get_action_mask(cl).numpy()
                        child_nfs.append(c_nf)
                        child_efs.append(c_ef)
                        child_ffs.append(c_ff)
                        child_masks_list.append(c_mask)
                        child_map.append((b, enc_idx, False,
                                          len(child_nfs) - 1))

            # Batched forward pass on all non-terminal children
            q_all_per_game = [np.full(AD, float(values_cpu[b_i]),
                                       dtype=np.float32) * mask_buf[b_i]
                               for b_i in range(B)]
            if child_nfs:
                c_batch = {
                    "node_features": torch.from_numpy(
                        np.stack(child_nfs)).to(device),
                    "edge_index": edge_index_dev,
                    "edge_features": torch.from_numpy(
                        np.stack(child_efs)).to(device),
                    "flat_features": torch.from_numpy(
                        np.stack(child_ffs)).to(device),
                    "action_mask": torch.from_numpy(
                        np.stack(child_masks_list)).to(device),
                }
                with torch.no_grad():
                    c_out = net(c_batch)
                c_values = F.softmax(c_out["value"], dim=-1)[:, 0].cpu().numpy()

            # Fill in Q-values (top-K get real values, rest keep V(s))
            for b_idx, enc_idx, is_term, payload in child_map:
                if is_term:
                    q_all_per_game[b_idx][enc_idx] = payload
                else:
                    q_all_per_game[b_idx][enc_idx] = float(c_values[payload])

            # ── Sample actions and step games ─────────────────────────
            still_active = []
            for b, (idx, le) in enumerate(index_map):
                g = games[idx]
                probs = pi_probs_cpu[b]
                if probs.sum() < 1e-6:
                    probs = mask_buf[b] / mask_buf[b].sum()

                uniform = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
                probs = 0.75 * probs + 0.25 * uniform
                probs = probs / probs.sum()

                action_idx = int(np.random.choice(AD, p=probs))

                trajectories[idx].append({
                    "nf": nf_buf[b].copy(),
                    "ef": ef_buf[b].copy(),
                    "ff": ff_buf[b].copy(),
                    "mask": mask_buf[b].copy(),
                    "q_all": q_all_per_game[b].copy(),
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
            if now - last_log > 15.0:
                avg_turns = sum(g.turn_number for g in games) / args.concurrent
                print(f"  [actor {args.actor_id}] {games_done}/{args.concurrent} done | "
                      f"{move_count} moves | avg_turn={avg_turns:.0f} | "
                      f"{now - t_batch:.0f}s", flush=True)
                last_log = now

        # ── Assign terminal rewards (graded by VP) ───────────────────
        for idx in range(args.concurrent):
            g = games[idx]
            n_p = g.num_players
            winner = g.winner()
            timed_out = winner is None
            vps = [g._game.state.player_state[i][0] for i in range(n_p)]
            ranked = sorted(range(n_p), key=lambda i: vps[i], reverse=True)
            if timed_out:
                grade = {ranked[0]: 0.1, ranked[1]: 0.05,
                         ranked[2]: 0.02, ranked[3]: 0.0}
            else:
                grade = {ranked[0]: 1.0, ranked[1]: 0.3,
                         ranked[2]: 0.1, ranked[3]: 0.0}
                grade[winner] = 1.0
            reward_vec = np.array([grade.get(p, 0.0) for p in range(4)],
                                  dtype=np.float32)
            for step in trajectories[idx]:
                step["reward"] = grade.get(step["player"], 0.0)
                step["reward_vec"] = reward_vec

        # Count per-game: did HZ seat win?
        # HZ has 1 seat in supervised, 3 seats in self-play
        hz_wins_batch = 0
        ab2_wins_batch = 0
        for idx in range(args.concurrent):
            w = games[idx].winner()
            if w is None:
                continue
            if w in ab2_seats:
                ab2_wins_batch += 1
            else:
                hz_wins_batch += 1

        # ── Write trajectory file ─────────────────────────────────────
        traj_file = os.path.join(traj_dir,
                                 f"actor{args.actor_id}_batch{batch_num:06d}.pt")
        torch.save(trajectories, traj_file)
        total_games += args.concurrent
        batch_num += 1

        elapsed = time.time() - t_batch
        total_elapsed = time.time() - t_start
        gps = total_games / total_elapsed
        print(f"[actor {args.actor_id}] Batch {batch_num}: "
              f"{args.concurrent} games in {elapsed:.1f}s | "
              f"total={total_games} | {gps:.2f} g/s | "
              f"AB2={ab2_wins_batch} HZ={hz_wins_batch}", flush=True)

        # ── Reload checkpoint if newer ────────────────────────────────
        if total_games % args.reload_every < args.concurrent:
            net, anchor, ckpt_mtime = _maybe_reload(
                ckpt_dir, cfg, device, net, anchor, ckpt_mtime)

    print(f"[actor {args.actor_id}] Done: {total_games} games in "
          f"{time.time() - t_start:.0f}s", flush=True)


def _run_multi_process(args):
    """Spawn N sub-processes on the same GPU, each with its own CPU core."""
    device = detect_device(args.device)
    n = args.num_processes

    print(f"[actor {args.actor_id}] Spawning {n} sub-processes on {device}",
          flush=True)

    processes = []
    ctx = mp.get_context("spawn")
    for sub_id in range(n):
        # Each sub-process gets a unique actor_id: base * 100 + sub_id
        sub_actor_id = args.actor_id * 100 + sub_id
        p = ctx.Process(
            target=_sub_process_entry,
            args=(sub_actor_id, args.trajectory_dir, args.checkpoint_dir,
                  args.concurrent, args.reload_every, args.max_games,
                  args.device),
            daemon=True,
        )
        p.start()
        processes.append(p)
        print(f"  Sub-process {sub_id} started (actor_id={sub_actor_id}, pid={p.pid})",
              flush=True)

    for p in processes:
        p.join()


def _sub_process_entry(actor_id, traj_dir, ckpt_dir, concurrent,
                        reload_every, max_games, device_str):
    """Entry point for each sub-process actor."""
    import sys
    sys.argv = ["rnad_actor"]  # reset argv for argparse in imports

    # Re-run main() with modified args
    device = detect_device(device_str)
    os.makedirs(traj_dir, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    cfg = get_default_config()
    action_enc = ActionEncoder()
    tmp = CatanGame(seed=0); tmp.reset()
    state_enc = tmp.make_state_encoder()

    net, ckpt_mtime = _load_checkpoint(ckpt_dir, cfg, device)
    anchor = HexaZeroNet(cfg.network).to(device)
    anchor.load_state_dict(net.state_dict())
    anchor.eval()

    N = state_enc.num_nodes
    E = state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM
    AD = 337

    nf_buf = np.zeros((concurrent, N, NF), dtype=np.float32)
    ef_buf = np.zeros((concurrent, E, EF), dtype=np.float32)
    ff_buf = np.zeros((concurrent, FF), dtype=np.float32)
    mask_buf = np.zeros((concurrent, AD), dtype=np.float32)
    edge_index_dev = state_enc._edge_index.to(device)

    total_games = 0
    batch_num = 0
    seed_base = actor_id * 10_000_000 + int(time.time()) % 10_000_000
    t_start = time.time()

    print(f"[actor {actor_id}] Sub-process ready on {device}", flush=True)

    while max_games == 0 or total_games < max_games:
        games = [CatanGame(seed=seed_base + total_games + i) for i in range(concurrent)]
        for gg in games: gg.reset()
        trajectories = [[] for _ in range(concurrent)]
        active = list(range(concurrent))
        games_done = 0; t_batch = time.time(); last_log = t_batch; move_count = 0

        net.eval(); anchor.eval()

        while active:
            B = 0; index_map = []
            for idx in active:
                gg = games[idx]
                if gg.is_terminal() or gg.turn_number >= 750 or len(trajectories[idx]) >= 2000: continue
                sv = gg.get_state_view()
                state_enc.encode_into(sv, nf_buf[B], ef_buf[B], ff_buf[B])
                le = gg.get_legal_actions()
                mask_buf[B] = action_enc.get_action_mask(le).numpy()
                index_map.append((idx, le)); B += 1
            if B == 0: break

            batch = {
                "node_features": torch.from_numpy(nf_buf[:B]).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B]).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B]).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B]).to(device),
            }
            with torch.no_grad():
                out = net(batch); anchor_out = anchor(batch)

            pi_cpu = out["policy_probs"].cpu().numpy()
            val_cpu = F.softmax(out["value"], dim=-1)[:, 0].cpu().numpy()
            lp_cpu = F.log_softmax(out["policy_logits"], dim=-1).cpu().numpy()
            alp_cpu = F.log_softmax(anchor_out["policy_logits"], dim=-1).cpu().numpy()

            still_active = []
            for b, (idx, le) in enumerate(index_map):
                gg = games[idx]; probs = pi_cpu[b]
                if probs.sum() < 1e-6: probs = mask_buf[b] / mask_buf[b].sum()
                uniform = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
                probs = 0.75 * probs + 0.25 * uniform
                probs = probs / probs.sum()
                aidx = int(np.random.choice(AD, p=probs))
                trajectories[idx].append({
                    "nf": nf_buf[b].copy(), "ef": ef_buf[b].copy(),
                    "ff": ff_buf[b].copy(), "mask": mask_buf[b].copy(),
                    "action_idx": aidx, "player": gg.current_player(),
                    "log_pi_taken": float(lp_cpu[b, aidx]),
                    "anchor_log_pi_taken": float(alp_cpu[b, aidx]),
                    "value": float(val_cpu[b]), "terminal": False,
                })
                chosen = next((i for i, a in enumerate(le)
                               if action_enc.encode(a) == aidx), 0)
                gg.step(chosen)
                if not gg.is_terminal() and gg.turn_number < 750 and len(trajectories[idx]) < 2000:
                    still_active.append(idx)
                elif trajectories[idx]:
                    trajectories[idx][-1]["terminal"] = True; games_done += 1
            move_count += B; active = still_active

            now = time.time()
            if now - last_log > 30.0:
                avg_t = sum(gg.turn_number for gg in games) / concurrent
                print(f"  [actor {actor_id}] {games_done}/{concurrent} done | "
                      f"{move_count} moves | avg_turn={avg_t:.0f}", flush=True)
                last_log = now

        for idx in range(concurrent):
            gg = games[idx]; winner = gg.winner(); n_p = gg.num_players
            vps = [gg._game.state.player_state[i][0] for i in range(n_p)]
            ranked = sorted(range(n_p), key=lambda i: vps[i], reverse=True)
            grade = {ranked[0]:1.0, ranked[1]:0.3, ranked[2]:0.1, ranked[3]:0.0}
            if winner is not None and 0 <= winner < 4: grade[winner] = 1.0
            reward_vec = np.array([grade.get(p, 0.0) for p in range(4)],
                                  dtype=np.float32)
            for step in trajectories[idx]:
                step["reward"] = grade.get(step["player"], 0.0)
                step["reward_vec"] = reward_vec

        traj_file = os.path.join(traj_dir, f"actor{actor_id}_batch{batch_num:06d}.pt")
        torch.save(trajectories, traj_file)
        total_games += concurrent; batch_num += 1
        elapsed = time.time() - t_batch
        gps = total_games / (time.time() - t_start)
        print(f"[actor {actor_id}] Batch {batch_num}: {concurrent} games "
              f"in {elapsed:.1f}s | total={total_games} | {gps:.2f} g/s", flush=True)

        if total_games % reload_every < concurrent:
            net, anchor, ckpt_mtime = _maybe_reload(
                ckpt_dir, cfg, device, net, anchor, ckpt_mtime)


def _load_checkpoint(ckpt_dir, cfg, device):
    from hexzero.model.network import HexaZeroNet
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        try:
            net = HexaZeroNet.load_checkpoint(latest, device=device)
            mtime = os.path.getmtime(latest)
            print(f"  Loaded checkpoint: {latest}", flush=True)
            return net, mtime
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}, random init", flush=True)
    net = HexaZeroNet(cfg.network).to(device)
    return net, 0.0


def _maybe_reload(ckpt_dir, cfg, device, net, anchor, last_mtime):
    latest = os.path.join(ckpt_dir, "latest.pt")
    if not os.path.exists(latest):
        return net, anchor, last_mtime
    try:
        mtime = os.path.getmtime(latest)
        if mtime > last_mtime + 5:
            from hexzero.model.network import HexaZeroNet
            state = torch.load(latest, map_location=device, weights_only=False)
            net.load_state_dict(state["model_state_dict"])
            anchor.load_state_dict(state["model_state_dict"])
            net.eval()
            anchor.eval()
            print(f"  [actor] Reloaded weights (age={time.time()-mtime:.0f}s)",
                  flush=True)
            return net, anchor, mtime
    except Exception:
        pass
    return net, anchor, last_mtime


if __name__ == "__main__":
    main()
