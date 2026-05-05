#!/usr/bin/env python3
"""Stream AB2 imitation games to a shard directory for the GPU learner.

CPU-only actors that play full AB2-vs-AB2 games using the *proper*
alphabeta_search (full chance-node expectimax — matches Python catanatron
``AlphaBetaPlayer(depth=2)``), encode every decision into the same shard
format c_selfplay produces, and write to ``shard_dir/pending/`` with
backpressure so the learner can consume them in real time.

Pairs with::

    python3 -u human_bot/c_selfplay.py --role learner \\
        --shard-dir <same dir> --ckpt-dir <ckpt dir> ...

The learner trains on the streaming AB2 imitation shards exactly as if
they had come from a self-play actor — same step_weight semantics, same
graded value targets, same num_players field.

Usage::

    # Single node, max actors:
    python3 -u human_bot/ab2_stream.py \\
        --shard-dir data/ab2_mp_v1 \\
        --num-workers 80 --max-pending 200 --player-counts 2,3,4

    # Multi-node: pass --actor-id-offset N on each additional node so
    # filenames don't collide.
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import time
import traceback

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

# Match c_selfplay's shard format constants
GAMES_PER_SHARD = 25
MAX_TURNS = 1000
MAX_STEPS_PER_GAME = 2000
MASK_DIM = 397

# Per-action-type policy-loss weight modifier (same table as c_selfplay)
_ACT_MOD = np.ones(MASK_DIM, dtype=np.float32)
_ACT_MOD[0] = 0.2
_ACT_MOD[1] = 0.5
_ACT_MOD[2:5] = 1.5
_ACT_MOD[5:113] = 1.5
_ACT_MOD[113:185] = 1.5
_ACT_MOD[185:280] = 1.5
_ACT_MOD[280:285] = 0.3
_ACT_MOD[285:310] = 1.5
_ACT_MOD[310:397] = 1.3


def compute_policy_weights(steps):
    """Action-type-only policy-loss weights.

    Deliberately NO winner-boost: AB2 targets are correct regardless of
    game outcome, so upweighting "winner moves" would just concentrate
    gradient on lucky trajectories instead of teaching AB2's policy. See
    plan: robust-improvement-run.
    """
    S = len(steps)
    weights = np.ones(S, dtype=np.float32)
    for i, s in enumerate(steps):
        weights[i] = _ACT_MOD[min(s["action_idx"], MASK_DIM - 1)]
    return weights


def atomic_torch_save(data, path):
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.rename(tmp, path)


SOURCE_TAG = "ab2"  # shard provenance for source-aware learner sampling


def save_shard(games_data, output_dir, shard_id):
    """Writes a shard in the shared c_selfplay format with:
      - policy_weight: action-type-only weights (no winner boost)
      - step_weight: legacy field, kept identical to policy_weight for
        backwards compatibility with older learners
      - source: shard provenance (bytes) so the learner can sample by source
      - num_players: per-row real player count"""
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player, all_reward, all_pw = [], [], [], [], []
    all_np = []
    for steps, rv, pw, n_players in games_data:
        for i, s in enumerate(steps):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_pw.append(pw[i])
            all_np.append(n_players)
    if not all_nf:
        return 0
    pw_t = torch.tensor(all_pw, dtype=torch.float32)
    data = {
        "node_features": torch.from_numpy(np.stack(all_nf)),
        "edge_features": torch.from_numpy(np.stack(all_ef)),
        "flat_features": torch.from_numpy(np.stack(all_ff)),
        "action_mask": torch.from_numpy(np.stack(all_mask)),
        "action_idx": torch.tensor(all_act, dtype=torch.int64),
        "player": torch.tensor(all_player, dtype=torch.int64),
        "reward_vec": torch.from_numpy(np.stack(all_reward)),
        "policy_weight": pw_t,
        "step_weight": pw_t.clone(),  # legacy alias
        "num_players": torch.tensor(all_np, dtype=torch.int64),
        "source": SOURCE_TAG,
    }
    atomic_torch_save(data, os.path.join(output_dir, f"{shard_id}.pt"))
    return len(all_nf)


# ---------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------

def run_actor(actor_id, shard_dir, seed_base, ab_depth, max_pending,
              player_counts):
    try:
        _run_actor(actor_id, shard_dir, seed_base, ab_depth, max_pending,
                   player_counts)
    except Exception:
        print(f"!!! [ab2 actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_actor(actor_id, shard_dir, seed_base, ab_depth, max_pending,
               player_counts):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
    )

    lib = load_library()
    ae = ActionEncoder()
    eval_fn = ValueFn(lib.base_value_fn)  # required by alphabeta_search

    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)

    ctx = SearchCtx()
    action_buf = (CAction * MAX_ACTIONS)()

    pc_arr = np.asarray(player_counts, dtype=np.int64)
    rng = np.random.default_rng((seed_base + actor_id) ^ 0xCAFE)

    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    def ab2_pick(game):
        """Proper alpha-beta minimax with full chance-node expectimax.

        Returns the index of the chosen action within game.get_legal_actions().
        """
        le = game.get_legal_actions()
        n = len(le)
        if n == 0:
            return None, le
        if n == 1:
            return 0, le
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le):
            action_buf[i] = a
        res = lib.alphabeta_search(
            ctypes.byref(ctx), ctypes.byref(cg), action_buf,
            ctypes.c_int(n), ctypes.c_int(ab_depth),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), eval_fn,
        )
        cb = ctypes.string_at(ctypes.byref(res.action),
                               ctypes.sizeof(res.action))
        for i, a in enumerate(le):
            if ctypes.string_at(ctypes.byref(a),
                                ctypes.sizeof(a)) == cb:
                return i, le
        return 0, le

    def play_game(seed, num_players):
        """One full AB2 game; returns (steps, reward_vec, step_weight, winner)."""
        cfg = GameConfig(num_players=num_players)
        game = CatanGame(seed=seed, random_board=True, config=cfg)
        game.reset()
        steps = []
        while (not game.is_terminal()
               and game.turn_number < MAX_TURNS
               and len(steps) < MAX_STEPS_PER_GAME):
            chosen, le = ab2_pick(game)
            if chosen is None:
                break

            # Encode state BEFORE stepping
            sv = game.get_state_view()
            se.encode_into(sv, nf, ef, ff)
            mask = ae.get_action_mask(le).numpy()

            try:
                enc_action = ae.encode(le[chosen])
            except ValueError:
                # Action not encodable (rare edge case) — just step, skip step
                game.step(chosen)
                continue

            # Save snapshot (only for multi-action decisions to keep shards
            # focused on real choices, matching c_selfplay's behavior).
            if len(le) > 1:
                steps.append({
                    "nf": nf.copy(),
                    "ef": ef.copy(),
                    "ff": ff.copy(),
                    "mask": mask.copy(),
                    "action_idx": enc_action,
                    "player": game.current_player(),
                })
            game.step(chosen)

        # M2-style reward (matches c_selfplay.play_game):
        winner = game.winner()
        reward_vec = np.zeros(4, dtype=np.float32)
        if winner is not None:
            turns = game.turn_number
            speed_bonus = max(0.0, min(0.5, (300 - turns) / 300.0))
            reward_vec[winner] = 1.0 + speed_bonus
            for seat in range(num_players):
                if seat == winner:
                    continue
                vp = game._game.state.player_state[seat][0]
                reward_vec[seat] = vp / 20.0
        pw = compute_policy_weights(steps)
        return steps, reward_vec, pw, winner

    game_batch = []
    shard_idx = 0
    total_games = 0
    total_steps = 0
    wins = np.zeros(4, dtype=np.int64)
    games_by_pc = {int(pc): 0 for pc in pc_arr}
    t_start = time.time()

    stop_file = os.path.join(shard_dir, ".stop")
    ckpt_stop = os.path.join(os.path.dirname(shard_dir), "checkpoints", "exit_v2", ".stop")
    print(f"[ab2 actor {actor_id}] Started, depth={ab_depth}, "
          f"seed_base={seed_base}, player_counts={list(pc_arr)}, "
          f"max_pending={max_pending}", flush=True)

    while not (os.path.exists(stop_file) or os.path.exists(ckpt_stop)):
        seed = seed_base + total_games
        n_players = int(rng.choice(pc_arr))
        games_by_pc[n_players] = games_by_pc.get(n_players, 0) + 1
        steps, rv, pw, winner = play_game(seed, n_players)

        if not steps:
            # Game ended without any recorded multi-action steps (very rare).
            total_games += 1
            continue

        game_batch.append((steps, rv, pw, n_players))
        total_games += 1
        total_steps += len(steps)
        if winner is not None:
            wins[winner] += 1

        if len(game_batch) >= GAMES_PER_SHARD:
            sid = f"ab2_a{actor_id:03d}_{shard_idx:06d}"
            save_shard(game_batch, pending_dir, sid)
            game_batch = []
            shard_idx += 1

            # Backpressure: pause until pending shards drop below cap
            while True:
                try:
                    n_pending = len([f for f in os.listdir(pending_dir)
                                     if f.endswith(".pt")
                                     and not f.endswith(".tmp")])
                except FileNotFoundError:
                    n_pending = 0
                if n_pending <= max_pending:
                    break
                time.sleep(2)

        if total_games % 25 == 0:
            elapsed = time.time() - t_start
            gps = total_games / elapsed if elapsed > 0 else 0
            avg_s = total_steps / total_games if total_games else 0
            pc_summary = " ".join(
                f"{k}p={v}" for k, v in sorted(games_by_pc.items()))
            print(f"[ab2 actor {actor_id}] {total_games} games, "
                  f"{shard_idx} shards, {gps:.2f} g/s, "
                  f"~{avg_s:.0f} steps/g, wins={wins.tolist()} | {pc_summary}",
                  flush=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Streaming AB2 imitation actors (CPU). Pairs with "
                    "c_selfplay.py --role learner.")
    parser.add_argument("--shard-dir", type=str, required=True,
                        help="Output dir; shards land in <shard-dir>/pending/")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of CPU actors (0 = auto = cpu_count - 2)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Alpha-beta search depth (2 matches Python AB2)")
    parser.add_argument("--max-pending", type=int, default=200,
                        help="Max .pt shards in pending/ before actors block")
    parser.add_argument("--seed", type=int, default=9_000_000)
    parser.add_argument("--actor-id-offset", type=int, default=0,
                        help="First actor id (use distinct values per node "
                             "in multi-node deployments)")
    parser.add_argument("--player-counts", type=str, default="2,3,4",
                        help="Comma-separated player counts to sample "
                             "uniformly per game (e.g. '2,3,4' or '4').")
    args = parser.parse_args()

    try:
        args.player_counts = tuple(int(x) for x in args.player_counts.split(",")
                                    if x.strip())
    except ValueError:
        parser.error(f"--player-counts must be comma-separated ints, "
                     f"got: {args.player_counts}")
    if not args.player_counts:
        parser.error("--player-counts cannot be empty")
    for n in args.player_counts:
        if n not in (2, 3, 4):
            parser.error(f"--player-counts values must be 2, 3 or 4 (got {n})")

    if args.num_workers <= 0:
        args.num_workers = max(1, (os.cpu_count() or 2) - 2)

    pending_dir = os.path.join(args.shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    print(f"AB2 stream", flush=True)
    print(f"  Shard dir:    {args.shard_dir}", flush=True)
    print(f"  Workers:      {args.num_workers} (CPU)", flush=True)
    print(f"  AB depth:     {args.depth}", flush=True)
    print(f"  Max pending:  {args.max_pending}", flush=True)
    print(f"  Player cnts:  {list(args.player_counts)} (uniform per game)",
          flush=True)
    print(f"  Seed base:    {args.seed}", flush=True)
    print(f"  Actor offset: {args.actor_id_offset}", flush=True)
    print(flush=True)

    ctx = mp.get_context("spawn")
    per_actor_seeds = 1_000_000  # leave room for many games per actor

    procs = []
    for i in range(args.num_workers):
        aid = args.actor_id_offset + i
        seed_base = args.seed + aid * per_actor_seeds
        p = ctx.Process(
            target=run_actor,
            args=(aid, args.shard_dir, seed_base, args.depth,
                  args.max_pending, args.player_counts),
            daemon=True,
        )
        p.start()
        procs.append(p)

    print(f"[main] {len(procs)} AB2 actors started "
          f"(ids {args.actor_id_offset}..{args.actor_id_offset + args.num_workers - 1})",
          flush=True)

    try:
        while True:
            time.sleep(60)
            alive = sum(1 for p in procs if p.is_alive())
            if alive == 0:
                print("[main] All actors died; exiting.", flush=True)
                break
            if alive < len(procs):
                print(f"[main] {len(procs) - alive}/{len(procs)} actors died, "
                      f"{alive} still running.", flush=True)
    except KeyboardInterrupt:
        print("[main] Interrupted.", flush=True)

    for p in procs:
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
