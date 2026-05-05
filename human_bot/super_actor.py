"""super_actor: long-running expert-iteration actor.

Plays 4-way super_m2 self-play games (all 4 seats use deep recursive search),
records every super_m2 decision from every seat, and writes shards to
`shard_dir/pending/` for the learner to consume.

The shard format is the standard `human_bot.dataset` shard:
    node_features, edge_features, flat_features, action_mask,
    action_idx (super_m2's chosen action),
    player (acting seat), reward_vec, step_weight, num_players

The actor reloads weights from `weights_bin_path` whenever its mtime changes
(checked between shards). It exits gracefully when `ckpt_dir/.stop` exists.

NOTE on GPU: super_m2's inner loop is C (`csrc/nn.c` + `deep_search.c`) with
NEON / Accelerate / OpenBLAS — it cannot use CUDA. We still allocate a GPU
on each actor node so we get jagupard machines (which have many CPU cores);
the GPU sits idle. CPU is the actual super_m2 bottleneck.

Usage:
    python -m human_bot.super_actor \
        --actor-id 0 \
        --weights-bin checkpoints/super_exit/nn_weights_latest.bin \
        --shard-dir data/super_exit \
        --ckpt-dir checkpoints/super_exit \
        --parallel-games 4 \
        --games-per-shard 4 \
        --depth 6 --k-schedule 12,8,6,5,4,3 --time-ms 4000 \
        --max-pending 32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback

import numpy as np
import torch


def _worker_play(args):
    """Single-game worker: 4-way super_m2 self-play, all seats record.

    Re-uses the proven `_play_one_game` from collect_super_m2_dataset
    (which already supports `super_seat == -1` for all-seats mode).

    Args tuple: (game_idx, seed, weights_path, depth, k_schedule, time_ms, dense)

    Returns dict from _play_one_game.
    """
    (game_idx, seed, weights_path, depth, k_schedule, time_ms, dense) = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from human_bot.collect_super_m2_dataset import _play_one_game

    r = _play_one_game(
        (game_idx, seed, -1, weights_path, depth, k_schedule, time_ms, dense))
    return r


def _save_shard(games, output_dir, shard_id):
    """Aggregate per-game results into a single .pt shard.

    Delegates to `collect_super_m2_dataset.save_shard` which handles both
    the standard (sparse) format and the dense format (with policy_target
    and signal_kind). Format is fully compatible with c_selfplay.save_shard
    plus the optional dense fields the super_learner consumes.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from human_bot.collect_super_m2_dataset import save_shard as _canonical
    return _canonical(games, output_dir, shard_id)


def _count_pending(pending_dir):
    if not os.path.isdir(pending_dir):
        return 0
    return sum(1 for f in os.listdir(pending_dir)
               if f.endswith(".pt") and not f.endswith(".tmp"))


def run(actor_id, weights_bin_path, shard_dir, ckpt_dir,
        parallel_games, games_per_shard, depth, k_schedule, time_ms,
        max_pending, seed_base, dense):
    """Main actor loop. Runs until `ckpt_dir/.stop` exists."""
    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)
    stop_file = os.path.join(ckpt_dir, ".stop")

    print(f"[actor {actor_id}] starting", flush=True)
    print(f"  weights:        {weights_bin_path}", flush=True)
    print(f"  shard_dir:      {pending_dir}", flush=True)
    print(f"  parallel:       {parallel_games} games concurrently", flush=True)
    print(f"  games/shard:    {games_per_shard}", flush=True)
    print(f"  depth:          {depth} k={k_schedule} t={time_ms}ms", flush=True)
    print(f"  max_pending:    {max_pending}", flush=True)
    print(f"  stop file:      {stop_file}", flush=True)
    print(f"  dense:          {dense} "
          f"({'soft policy_target from search values' if dense else 'one-hot only'})",
          flush=True)

    # Wait for initial weights file
    wait_start = time.time()
    while not os.path.exists(weights_bin_path):
        if os.path.exists(stop_file):
            print(f"[actor {actor_id}] stop file present at startup; exiting",
                  flush=True)
            return
        if time.time() - wait_start > 1800:
            print(f"[actor {actor_id}] gave up waiting for {weights_bin_path}",
                  flush=True)
            return
        print(f"[actor {actor_id}] waiting for weights at {weights_bin_path}...",
              flush=True)
        time.sleep(10)

    weights_mtime = os.path.getmtime(weights_bin_path)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=parallel_games)

    shard_idx = 0
    total_games = 0
    total_steps = 0
    rng = np.random.default_rng((seed_base + actor_id) ^ 0xCA7AB07)
    t_actor_start = time.time()

    try:
        while not os.path.exists(stop_file):
            # Backpressure: don't pile up shards if learner is behind
            n_pending = _count_pending(pending_dir)
            if n_pending >= max_pending:
                if total_games == 0 or total_games % 10 == 0:
                    print(f"[actor {actor_id}] backpressure: "
                          f"{n_pending} pending shards (cap {max_pending}), "
                          f"sleeping 30s", flush=True)
                # Re-check stop file every 5s while sleeping
                for _ in range(6):
                    if os.path.exists(stop_file):
                        break
                    time.sleep(5)
                continue

            # Reload weights if file changed (between shards)
            try:
                cur_mtime = os.path.getmtime(weights_bin_path)
                if cur_mtime > weights_mtime:
                    print(f"[actor {actor_id}] weights changed "
                          f"(mtime {weights_mtime:.0f} -> {cur_mtime:.0f}); "
                          f"workers will pick up new weights on next game",
                          flush=True)
                    weights_mtime = cur_mtime
            except OSError:
                pass

            # Build job batch (one shard's worth of games)
            jobs = []
            for _ in range(games_per_shard):
                gi = total_games + len(jobs)
                seed = int(rng.integers(0, 2**31 - 1))
                jobs.append((gi, seed, weights_bin_path, depth,
                             k_schedule, time_ms, dense))

            t_shard_start = time.time()
            results = []
            try:
                # imap_unordered so we can detect stop file mid-shard
                for r in pool.imap_unordered(_worker_play, jobs):
                    results.append(r)
                    if os.path.exists(stop_file) and len(results) < len(jobs):
                        # Drain remaining workers but don't enqueue more
                        pass
            except KeyboardInterrupt:
                print(f"[actor {actor_id}] keyboard interrupt; saving partial "
                      f"shard ({len(results)} games)", flush=True)

            t_shard = time.time() - t_shard_start
            total_games += len(results)
            shard_steps = sum(r["n_steps"] for r in results)
            total_steps += shard_steps

            # Save shard
            sid = f"super_a{actor_id:03d}_{shard_idx:06d}"
            n_saved = _save_shard(results, pending_dir, sid)
            shard_idx += 1

            # Per-shard log
            elapsed = time.time() - t_actor_start
            wins = [0, 0, 0, 0]
            for r in results:
                if r["winner"] is not None:
                    wins[r["winner"]] += 1
            avg_turns = (sum(r["n_turns"] for r in results)
                         / max(len(results), 1))
            print(f"[actor {actor_id}] shard {sid}: "
                  f"{len(results)} games, {n_saved} steps, {t_shard:.0f}s "
                  f"(avg {t_shard/max(len(results),1):.0f}s/g, "
                  f"{avg_turns:.0f} turns/g, "
                  f"seat_wins=[{wins[0]} {wins[1]} {wins[2]} {wins[3]}])  "
                  f"| total: {total_games}g {total_steps}s "
                  f"({total_games / max(elapsed/60, 1e-9):.1f} g/min) "
                  f"wall={elapsed:.0f}s",
                  flush=True)

            # Reset pool periodically to free memory and force fresh worker
            # processes (each worker holds 4× super_m2 ctxs; long-lived workers
            # accumulate fragmented C heap).
            if shard_idx % 8 == 0:
                print(f"[actor {actor_id}] recycling worker pool",
                      flush=True)
                pool.close()
                pool.join()
                pool = ctx.Pool(processes=parallel_games)
    finally:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass

    print(f"[actor {actor_id}] stop file detected; exiting after "
          f"{total_games} games, {total_steps} steps "
          f"({(time.time()-t_actor_start)/60:.1f} min)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor-id", type=int, required=True)
    p.add_argument("--weights-bin", type=str, required=True,
                   help="Path to nn_weights_latest.bin produced by learner.")
    p.add_argument("--shard-dir", type=str, required=True,
                   help="Root for shards; pending/ subdir is created.")
    p.add_argument("--ckpt-dir", type=str, required=True,
                   help="Root for checkpoints; .stop file lives here.")
    p.add_argument("--parallel-games", type=int, default=4,
                   help="Concurrent games per actor (each uses ~4 CPU cores).")
    p.add_argument("--games-per-shard", type=int, default=4)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k-schedule", type=str, default="12,8,6,5,4,3")
    p.add_argument("--time-ms", type=int, default=4000)
    p.add_argument("--max-pending", type=int, default=32,
                   help="Pause if pending shard count exceeds this.")
    p.add_argument("--seed-base", type=int, default=400000)
    p.add_argument("--dense", action="store_true",
                   help="Record dense soft policy_target from per-candidate "
                        "search values (recommended).")
    args = p.parse_args()

    schedule = tuple(int(x) for x in args.k_schedule.split(","))

    try:
        run(actor_id=args.actor_id,
            weights_bin_path=os.path.abspath(args.weights_bin),
            shard_dir=os.path.abspath(args.shard_dir),
            ckpt_dir=os.path.abspath(args.ckpt_dir),
            parallel_games=args.parallel_games,
            games_per_shard=args.games_per_shard,
            depth=args.depth,
            k_schedule=schedule,
            time_ms=args.time_ms,
            max_pending=args.max_pending,
            seed_base=args.seed_base,
            dense=args.dense)
    except Exception:
        print(f"!!! [actor {args.actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
