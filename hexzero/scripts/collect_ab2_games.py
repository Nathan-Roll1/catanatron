#!/usr/bin/env python3
"""Phase 1 data collection: AB2 vs AB2 games for supervised pretraining.

Plays games where every seat is AB2 (proper alpha-beta minimax with full
chance-node expectimax — equivalent to Python catanatron's
``AlphaBetaPlayer(depth=2)``). For each game, records every decision
state's encoding, action mask, and chosen action. After each game,
assigns graded rewards based on VP ranking (1.0 / 0.3 / 0.1 / 0.0).

By default games are sampled uniformly from {2, 3, 4} player counts so
the resulting model handles all variants. Pass ``--player-counts 4`` for
4-player-only data.

Saves concatenated tensors per file batch — directly loadable by a
PyTorch Dataset without per-step dict overhead.

No neural network involvement. Pure AB2 heuristic gameplay with state
recording for offline supervised learning.

Usage:
    python -m hexzero.scripts.collect_ab2_games \\
        --output-dir data/ab2_games --num-games 2000 --num-workers 8 \\
        --depth 2 --player-counts 2,3,4
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

ACTION_DIM = 337
MAX_TURNS = 750
MAX_STEPS = 2000


# ── AB2 action selection ─────────────────────────────────────────────


def _ab2_pick(cg, legal_actions, lib, ctx, action_buf, eval_fn, depth=2):
    """Proper alpha-beta minimax with full chance-node expectimax.

    Calls into the strengthened ``alphabeta_search`` in ``csrc/search.c``,
    which expands ROLL (11 dice outcomes), BUY_DEVELOPMENT_CARD (deck
    composition), and MOVE_ROBBER-with-steal (5 resources) as chance
    nodes — semantically equivalent to Python catanatron's
    ``tree_search_utils.execute_spectrum``.

    Returns the index of the chosen action within ``legal_actions``.
    """
    n = len(legal_actions)
    if n == 0:
        return 0
    if n == 1:
        return 0
    bc = cg.state.colors[cg.state.current_player_index]

    # Copy legal actions into a contiguous buffer (alphabeta_search may
    # reorder them internally for move ordering).
    for i, act in enumerate(legal_actions):
        action_buf[i] = act

    res = lib.alphabeta_search(
        ctypes.byref(ctx),
        ctypes.byref(cg),
        action_buf,
        ctypes.c_int(n),
        ctypes.c_int(depth),
        ctypes.c_double(-1e30),
        ctypes.c_double(1e30),
        ctypes.c_int(bc),
        eval_fn,
    )

    # Find the matching action in the original order. We compare by
    # bytewise equality of the Action struct since that's what the C side
    # produced.
    chosen_bytes = ctypes.string_at(ctypes.byref(res.action),
                                     ctypes.sizeof(res.action))
    for i, act in enumerate(legal_actions):
        if ctypes.string_at(ctypes.byref(act),
                            ctypes.sizeof(act)) == chosen_bytes:
            return i
    return 0


# ── Reward computation ────────────────────────────────────────────────

def _compute_rewards(game) -> np.ndarray:
    """Graded reward vector (always length 4 — pad unused seats with 0).

    Works for 2/3/4-player games. The model's value head is fixed at 4
    outputs; downstream training rotates so the current player is slot 0
    and ignores trailing slots beyond ``num_players``.
    """
    winner = game.winner()
    n_p = game.num_players
    vps = [game._game.state.player_state[i][0] for i in range(n_p)]
    ranked = sorted(range(n_p), key=lambda i: vps[i], reverse=True)

    # Position-based grades: 1st/2nd/3rd/4th finishers
    if winner is None:
        ladder = [0.1, 0.05, 0.02, 0.0]
    else:
        ladder = [1.0, 0.3, 0.1, 0.0]
    grade = {ranked[i]: ladder[i] for i in range(n_p)}
    if winner is not None:
        grade[winner] = 1.0

    out = np.zeros(4, dtype=np.float32)
    for p in range(n_p):
        out[p] = grade.get(p, 0.0)
    return out


# ── Per-file serialisation ────────────────────────────────────────────

def _save_batch(games_data, output_dir, file_id):
    """Flatten a batch of games into concatenated tensors and write one .pt file.

    Each element of *games_data* is a tuple:
        (nf_list, ef_list, ff_list, mask_list, act_list, player_list, reward_vec)

    Saved tensor dict keys (S = total steps across all games in file):
        node_features   (S, N, 18)   float32
        edge_features   (S, E, 5)    float32
        flat_features   (S, 115)     float32
        action_mask     (S, 337)     float32
        action_idx      (S,)         int64
        player          (S,)         int64
        reward_vec      (S, 4)       float32
        game_id         (S,)         int64   (local id within this file)
    """
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player, all_reward, all_gid = [], [], [], [], []
    all_np = []

    for gid, (nf_l, ef_l, ff_l, mk_l, ac_l, pl_l, rv, n_p) in enumerate(games_data):
        n = len(nf_l)
        if n == 0:
            continue
        all_nf.extend(nf_l)
        all_ef.extend(ef_l)
        all_ff.extend(ff_l)
        all_mask.extend(mk_l)
        all_act.extend(ac_l)
        all_player.extend(pl_l)
        all_reward.extend([rv] * n)
        all_gid.extend([gid] * n)
        all_np.extend([n_p] * n)

    if not all_nf:
        return 0

    data = {
        "node_features": torch.from_numpy(np.stack(all_nf)),
        "edge_features": torch.from_numpy(np.stack(all_ef)),
        "flat_features": torch.from_numpy(np.stack(all_ff)),
        "action_mask": torch.from_numpy(np.stack(all_mask)),
        "action_idx": torch.tensor(all_act, dtype=torch.int64),
        "player": torch.tensor(all_player, dtype=torch.int64),
        "reward_vec": torch.from_numpy(np.stack(all_reward)),
        "game_id": torch.tensor(all_gid, dtype=torch.int64),
        "num_players": torch.tensor(all_np, dtype=torch.int64),
    }

    path = os.path.join(output_dir, f"{file_id}.pt")
    torch.save(data, path)
    return len(all_nf)


def _save_metadata(state_enc, output_dir):
    """Write a small metadata.pt so the training script can reconstruct
    edge_index and feature dims without importing game modules."""
    meta = {
        "edge_index": state_enc._edge_index.clone(),
        "num_nodes": state_enc.num_nodes,
        "num_edges": state_enc.num_edges,
        "node_feature_dim": state_enc.NODE_FEATURE_DIM,
        "edge_feature_dim": state_enc.EDGE_FEATURE_DIM,
        "flat_feature_dim": state_enc.FLAT_FEATURE_DIM,
        "action_space_size": ACTION_DIM,
    }
    torch.save(meta, os.path.join(output_dir, "metadata.pt"))


# ── Worker process ────────────────────────────────────────────────────

def _worker_fn(worker_id, num_games, output_dir, games_per_file,
               seed_base, counter, total_target, file_idx_start=0,
               ab_depth=2, player_counts=(2, 3, 4)):
    """Play *num_games* AB-only games, record every step, save batched .pt files."""
    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
    )

    lib = load_library()
    action_enc = ActionEncoder()
    # Wrap C base_value_fn into a CFUNCTYPE so ctypes accepts it as the
    # eval_fn callback parameter to alphabeta_search.
    eval_fn = ValueFn(lib.base_value_fn)

    tmp = CatanGame(seed=0)
    tmp.reset()
    state_enc = tmp.make_state_encoder()

    if worker_id == 0:
        _save_metadata(state_enc, output_dir)

    N = state_enc.num_nodes
    NF = state_enc.NODE_FEATURE_DIM
    E = state_enc.num_edges
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM

    nf_buf = np.zeros((N, NF), dtype=np.float32)
    ef_buf = np.zeros((E, EF), dtype=np.float32)
    ff_buf = np.zeros(FF, dtype=np.float32)

    # Single SearchCtx reused across all decisions in this worker.
    ctx = SearchCtx()
    action_buf = (CAction * MAX_ACTIONS)()

    rng = np.random.default_rng(seed_base ^ 0xCAFE)
    pc_arr = np.asarray(player_counts, dtype=np.int64)

    batch: list[tuple] = []
    file_idx = file_idx_start
    total_steps = 0
    wins = np.zeros(4, dtype=np.int64)
    games_by_pc = {int(pc): 0 for pc in pc_arr}
    timeouts = 0
    t_start = time.time()

    for game_num in range(num_games):
        seed = seed_base + game_num
        n_players = int(rng.choice(pc_arr))
        cfg = GameConfig(num_players=n_players)
        game = CatanGame(seed=seed, random_board=True, config=cfg)
        game.reset()
        games_by_pc[n_players] = games_by_pc.get(n_players, 0) + 1

        nf_list, ef_list, ff_list = [], [], []
        mask_list, act_list, player_list = [], [], []

        while (not game.is_terminal()
               and game.turn_number < MAX_TURNS
               and len(nf_list) < MAX_STEPS):
            le = game.get_legal_actions()
            if not le:
                break

            sv = game.get_state_view()
            state_enc.encode_into(sv, nf_buf, ef_buf, ff_buf)
            mask = action_enc.get_action_mask(le).numpy()

            chosen = _ab2_pick(game._game, le, lib, ctx, action_buf,
                               eval_fn, depth=ab_depth)

            try:
                enc_idx = action_enc.encode(le[chosen])
            except ValueError:
                game.step(chosen)
                continue

            nf_list.append(nf_buf.copy())
            ef_list.append(ef_buf.copy())
            ff_list.append(ff_buf.copy())
            mask_list.append(mask)
            act_list.append(enc_idx)
            player_list.append(game.current_player())

            game.step(chosen)

        reward_vec = _compute_rewards(game)
        batch.append((nf_list, ef_list, ff_list,
                       mask_list, act_list, player_list, reward_vec,
                       n_players))

        w = game.winner()
        if w is not None:
            wins[w] += 1
        else:
            timeouts += 1

        # ── Progress ──────────────────────────────────────────────
        if counter is not None:
            with counter.get_lock():
                counter.value += 1
                count = counter.value
        else:
            count = game_num + 1

        if count % 100 == 0:
            elapsed = time.time() - t_start
            local_gps = (game_num + 1) / elapsed if elapsed > 0 else 0
            avg_steps = (total_steps + sum(len(g[0]) for g in batch)) / count
            print(f"[progress] {count}/{total_target} games "
                  f"({count * 100 // total_target}%) | "
                  f"worker {worker_id}: {local_gps:.1f} g/s, "
                  f"~{avg_steps:.0f} steps/game",
                  flush=True)

        # ── Save file when batch is full ──────────────────────────
        if len(batch) >= games_per_file:
            fid = f"w{worker_id:02d}_{file_idx:04d}"
            n = _save_batch(batch, output_dir, fid)
            total_steps += n
            batch = []
            file_idx += 1

    if batch:
        fid = f"w{worker_id:02d}_{file_idx:04d}"
        n = _save_batch(batch, output_dir, fid)
        total_steps += n
        file_idx += 1

    elapsed = time.time() - t_start
    files_written = file_idx - file_idx_start
    gps = num_games / elapsed if elapsed > 0 else 0
    pc_summary = " ".join(f"{k}p={v}" for k, v in sorted(games_by_pc.items()))
    print(f"[worker {worker_id}] Done: {num_games} games, {total_steps:,} steps, "
          f"{files_written} files in {elapsed:.1f}s ({gps:.1f} g/s) | "
          f"wins={wins.tolist()} timeouts={timeouts} | counts: {pc_summary}",
          flush=True)


# ── Entry point ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Collect AB2-vs-AB2 game data for supervised pretraining"
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save .pt data files")
    parser.add_argument("--num-games", type=int, default=2000,
                        help="Total number of games to play")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Parallel workers (0 = auto-detect CPU count - 2)")
    parser.add_argument("--games-per-file", type=int, default=25,
                        help="Games batched into each output .pt file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (workers get non-overlapping ranges)")
    parser.add_argument("--depth", type=int, default=2,
                        help="AB search depth (1 = greedy, 2 = 2-ply lookahead)")
    parser.add_argument("--player-counts", type=str, default="2,3,4",
                        help="Comma-separated player counts to sample uniformly "
                             "(e.g. '2,3,4' or '4'). Each game picks one at random.")
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

    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_workers <= 0:
        args.num_workers = max(1, os.cpu_count() - 2)

    # Resume: count existing shards and only generate the gap
    existing_shards = [
        f for f in os.listdir(args.output_dir)
        if f.endswith(".pt") and f != "metadata.pt"
    ]
    existing_games = len(existing_shards) * args.games_per_file
    games_to_gen = max(0, args.num_games - existing_games)

    # Per-worker max file index so resumed workers don't overwrite old shards
    worker_max_file_idx: dict[int, int] = {}
    for f in existing_shards:
        stem = f.removesuffix(".pt")
        if "_" in stem and stem.startswith("w"):
            try:
                wid = int(stem.split("_")[0][1:])
                fid = int(stem.split("_")[1])
                worker_max_file_idx[wid] = max(worker_max_file_idx.get(wid, -1), fid)
            except ValueError:
                pass

    if games_to_gen == 0:
        print(f"Phase 1 data collection: {args.num_games} AB2 games requested, "
              f"{existing_games} already exist ({len(existing_shards)} shards). Nothing to do.")
        return

    seed_resume = args.seed + existing_games

    print(f"Phase 1 data collection: {args.num_games} AB2-only games")
    print(f"  Existing:    {existing_games} games ({len(existing_shards)} shards)")
    print(f"  To generate: {games_to_gen} games")
    print(f"  Workers:     {args.num_workers}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Games/file:  {args.games_per_file}")
    print(f"  Seed:        {seed_resume} (offset for resume)")
    print(f"  AB depth:    {args.depth} (proper alpha-beta + chance-node expectimax)")
    print(f"  Player cnts: {list(args.player_counts)} (uniform per game)")
    print(flush=True)

    t0 = time.time()

    if args.num_workers <= 1:
        fstart = worker_max_file_idx.get(0, -1) + 1
        _worker_fn(0, games_to_gen, args.output_dir, args.games_per_file,
                   seed_resume, None, games_to_gen, file_idx_start=fstart,
                   ab_depth=args.depth, player_counts=args.player_counts)
    else:
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)

        per_worker = games_to_gen // args.num_workers
        remainder = games_to_gen % args.num_workers

        procs = []
        seed_offset = 0
        for w in range(args.num_workers):
            n = per_worker + (1 if w < remainder else 0)
            if n == 0:
                continue
            fstart = worker_max_file_idx.get(w, -1) + 1
            p = ctx.Process(
                target=_worker_fn,
                args=(w, n, args.output_dir, args.games_per_file,
                      seed_resume + seed_offset, counter, games_to_gen,
                      fstart, args.depth, args.player_counts),
                daemon=True,
            )
            p.start()
            procs.append(p)
            seed_offset += n
            print(f"  Started worker {w} (pid={p.pid}, {n} games, "
                  f"seeds {seed_resume + seed_offset - n}..{seed_resume + seed_offset - 1})",
                  flush=True)

        print(flush=True)
        for p in procs:
            p.join()

    elapsed = time.time() - t0

    # ── Summary (lightweight: no torch.load, just count files + disk) ──
    files = [f for f in os.listdir(args.output_dir)
             if f.endswith(".pt") and f != "metadata.pt"]
    total_games = len(files) * args.games_per_file
    disk_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in files
    ) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"Collection complete")
    print(f"  Games:          ~{total_games} ({games_to_gen} new + {existing_games} existing)")
    print(f"  Files:          {len(files)}")
    print(f"  Disk usage:     {disk_mb:.1f} MB")
    print(f"  Wall time:      {elapsed:.1f}s ({games_to_gen / elapsed:.1f} g/s)")
    print(f"  Output:         {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
