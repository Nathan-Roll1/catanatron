#!/usr/bin/env python3
"""2v2 head-to-head between two Super M2 configurations."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexzero.bindings.structs import PS_ACTUAL_VICTORY_POINTS, PS_VICTORY_POINTS
from hexzero.game.interface import CatanGame
from human_bot.superbot_v3_c2 import SuperBotV3C2


def play_one(args):
    (game_idx, seed, weights, mod_depth, mod_schedule, mod_leaf_mode,
     mod_incremental_replay, mod_root_c_policy, mod_algo_policy,
     mod_algo_flags, mod_algo_value_tiebreak,
     mod_robust_model, mod_robust_weight,
     mod_leaf_pressure, mod_leaf_threat_bonus, mod_leaf_threat_vp,
     mod_endgame_extra, mod_threat_extra, mod_threat_opp_ab,
     mod_opp_ab_depth,
     mod_iterative, mod_iter_start_depth,
     mod_critical_vp, mod_critical_extra,
     orig_depth, orig_schedule, orig_leaf_mode,
     orig_incremental_replay, orig_root_c_policy, orig_algo_policy,
     orig_algo_flags, orig_algo_value_tiebreak,
     orig_robust_model, orig_robust_weight,
     orig_leaf_pressure, orig_leaf_threat_bonus, orig_leaf_threat_vp,
     orig_endgame_extra, orig_threat_extra, orig_threat_opp_ab,
     orig_opp_ab_depth,
     orig_iterative, orig_iter_start_depth,
     orig_critical_vp, orig_critical_extra,
     time_ms, parallel_workers) = args
    mod_parity = game_idx % 2
    mod_seats = {mod_parity, mod_parity + 2}

    if parallel_workers > 0:
        from human_bot.superbot_v3_parallel import ParallelSuperBot

        # Each team's workers are active only on that team's turns. This keeps
        # the per-decision fanout at `parallel_workers` for both leaf modes.
        mod_bot = ParallelSuperBot(
            weights,
            num_workers=parallel_workers,
            our_depth=mod_depth,
            top_k_schedule=mod_schedule,
            time_budget_ms=time_ms,
            backend="c2",
            leaf_mode=mod_leaf_mode,
            incremental_replay=mod_incremental_replay,
            root_c_policy=mod_root_c_policy,
            algo_policy=mod_algo_policy,
            algo_flags=mod_algo_flags,
            algo_value_tiebreak=mod_algo_value_tiebreak,
        )
        orig_bot = ParallelSuperBot(
            weights,
            num_workers=parallel_workers,
            our_depth=orig_depth,
            top_k_schedule=orig_schedule,
            time_budget_ms=time_ms,
            backend="c2",
            leaf_mode=orig_leaf_mode,
            incremental_replay=orig_incremental_replay,
            root_c_policy=orig_root_c_policy,
            algo_policy=orig_algo_policy,
            algo_flags=orig_algo_flags,
            algo_value_tiebreak=orig_algo_value_tiebreak,
        )
        mod_bot.reset_game(seed, 4)
        orig_bot.reset_game(seed, 4)
        bots = {seat: (mod_bot if seat in mod_seats else orig_bot) for seat in range(4)}
    else:
        bots = {}
        for seat in range(4):
            mod_seat = seat in mod_seats
            bots[seat] = SuperBotV3C2(
                weights,
                our_depth=mod_depth if mod_seat else orig_depth,
                top_k_schedule=mod_schedule if mod_seat else orig_schedule,
                entropy_fast_thresh=0.15,
                time_budget_ms=time_ms,
                opponent_ab_depth=mod_opp_ab_depth if mod_seat else orig_opp_ab_depth,
                leaf_mode=mod_leaf_mode if mod_seat else orig_leaf_mode,
                algo_policy=mod_algo_policy if mod_seat else orig_algo_policy,
                algo_flags=mod_algo_flags if mod_seat else orig_algo_flags,
                algo_value_tiebreak=(
                    mod_algo_value_tiebreak if mod_seat else orig_algo_value_tiebreak
                ),
                robust_opponent_model=(
                    mod_robust_model if mod_seat else orig_robust_model
                ),
                robust_penalty_weight=(
                    mod_robust_weight if mod_seat else orig_robust_weight
                ),
                leaf_pressure_weight=(
                    mod_leaf_pressure if mod_seat else orig_leaf_pressure
                ),
                leaf_threat_bonus=(
                    mod_leaf_threat_bonus if mod_seat else orig_leaf_threat_bonus
                ),
                leaf_threat_vp=(
                    mod_leaf_threat_vp if mod_seat else orig_leaf_threat_vp
                ),
                endgame_extra_depth=(
                    mod_endgame_extra if mod_seat else orig_endgame_extra
                ),
                threat_extra_depth=(
                    mod_threat_extra if mod_seat else orig_threat_extra
                ),
                threat_opp_ab_depth=(
                    mod_threat_opp_ab if mod_seat else orig_threat_opp_ab
                ),
                iterative_deepening=(
                    mod_iterative if mod_seat else orig_iterative
                ),
                iter_start_depth=(
                    mod_iter_start_depth if mod_seat else orig_iter_start_depth
                ),
                critical_vp_threshold=(
                    mod_critical_vp if mod_seat else orig_critical_vp
                ),
                critical_extra_depth=(
                    mod_critical_extra if mod_seat else orig_critical_extra
                ),
            )

    t0 = time.perf_counter()
    game = CatanGame(seed=seed)
    game.reset()

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        if len(legal) == 1:
            game.step(0)
            if parallel_workers > 0:
                mod_bot.record_action(0)
                orig_bot.record_action(0)
            continue
        seat = game.current_player()
        chosen = bots[seat].pick(game)
        game.step(chosen)
        if parallel_workers > 0:
            mod_bot.record_action(chosen)
            orig_bot.record_action(chosen)

    elapsed = time.perf_counter() - t0
    winner = game.winner()
    public_vps = [
        int(game._game.state.player_state[s][PS_VICTORY_POINTS])
        for s in range(4)
    ]
    actual_vps = [
        int(game._game.state.player_state[s][PS_ACTUAL_VICTORY_POINTS])
        for s in range(4)
    ]
    mod_win = winner in mod_seats
    ranks = sorted(range(4), key=lambda s: -actual_vps[s])
    mod_ranks = [ranks.index(s) + 1 for s in sorted(mod_seats)]
    orig_seats = sorted(set(range(4)) - mod_seats)
    orig_ranks = [ranks.index(s) + 1 for s in orig_seats]
    try:
        stats = {s: bots[s].stats_summary() for s in range(4)}
        return {
            "game": game_idx,
            "seed": seed,
            "mod_seats": sorted(mod_seats),
            "orig_seats": orig_seats,
            "winner": winner,
            "mod_win": mod_win,
            "vps": actual_vps,
            "public_vps": public_vps,
            "mod_avg_rank": sum(mod_ranks) / len(mod_ranks),
            "orig_avg_rank": sum(orig_ranks) / len(orig_ranks),
            "turns": game.turn_number,
            "seconds": elapsed,
            "stats": stats,
        }
    finally:
        if parallel_workers > 0:
            mod_bot.shutdown()
            orig_bot.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=96000)
    parser.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    parser.add_argument("--depth", type=int, default=6,
                        help="Experiment depth (alias for --mod-depth).")
    parser.add_argument("--k-schedule", default="12,8,6,5,4,3",
                        help="Experiment schedule (alias for --mod-k-schedule).")
    parser.add_argument("--mod-depth", type=int, default=None)
    parser.add_argument("--mod-k-schedule", default=None)
    parser.add_argument("--mod-leaf-mode", type=int, default=1)
    parser.add_argument("--mod-no-incremental-replay", action="store_true")
    parser.add_argument("--mod-root-c-policy", action="store_true")
    parser.add_argument("--mod-algo-policy", action="store_true")
    parser.add_argument("--mod-algo-flags", type=int, default=0)
    parser.add_argument("--mod-algo-value-tiebreak", action="store_true")
    parser.add_argument("--mod-robust-opponent-model", default=None)
    parser.add_argument("--mod-robust-penalty-weight", type=float, default=0.5)
    parser.add_argument("--mod-leaf-pressure", type=float, default=None)
    parser.add_argument("--mod-leaf-threat-bonus", type=float, default=None)
    parser.add_argument("--mod-leaf-threat-vp", type=int, default=8)
    parser.add_argument("--mod-endgame-extra-depth", type=int, default=0)
    parser.add_argument("--mod-threat-extra-depth", type=int, default=0)
    parser.add_argument("--mod-threat-opp-ab-depth", type=int, default=2)
    parser.add_argument("--mod-opp-ab-depth", type=int, default=2)
    parser.add_argument("--mod-iterative", action="store_true")
    parser.add_argument("--mod-iter-start-depth", type=int, default=2)
    parser.add_argument("--mod-critical-vp", type=int, default=100)
    parser.add_argument("--mod-critical-extra", type=int, default=0)
    parser.add_argument("--orig-depth", type=int, default=6)
    parser.add_argument("--orig-k-schedule", default="12,8,6,5,4,3")
    parser.add_argument("--orig-leaf-mode", type=int, default=0)
    parser.add_argument("--orig-no-incremental-replay", action="store_true")
    parser.add_argument("--orig-root-c-policy", action="store_true")
    parser.add_argument("--orig-algo-policy", action="store_true")
    parser.add_argument("--orig-algo-flags", type=int, default=0)
    parser.add_argument("--orig-algo-value-tiebreak", action="store_true")
    parser.add_argument("--orig-robust-opponent-model", default=None)
    parser.add_argument("--orig-robust-penalty-weight", type=float, default=0.5)
    parser.add_argument("--orig-leaf-pressure", type=float, default=None)
    parser.add_argument("--orig-leaf-threat-bonus", type=float, default=None)
    parser.add_argument("--orig-leaf-threat-vp", type=int, default=8)
    parser.add_argument("--orig-endgame-extra-depth", type=int, default=0)
    parser.add_argument("--orig-threat-extra-depth", type=int, default=0)
    parser.add_argument("--orig-threat-opp-ab-depth", type=int, default=2)
    parser.add_argument("--orig-opp-ab-depth", type=int, default=2)
    parser.add_argument("--orig-iterative", action="store_true")
    parser.add_argument("--orig-iter-start-depth", type=int, default=2)
    parser.add_argument("--orig-critical-vp", type=int, default=100)
    parser.add_argument("--orig-critical-extra", type=int, default=0)
    parser.add_argument("--time-ms", type=int, default=4000)
    parser.add_argument("--parallel-workers", type=int, default=0,
                        help="Use root-candidate parallel C2 Super M2 with this many workers per active team.")
    args = parser.parse_args()

    weights = os.path.abspath(args.weights)
    mod_depth = args.mod_depth if args.mod_depth is not None else args.depth
    mod_k = args.mod_k_schedule if args.mod_k_schedule is not None else args.k_schedule
    mod_schedule = tuple(int(x) for x in mod_k.split(","))
    orig_schedule = tuple(int(x) for x in args.orig_k_schedule.split(","))
    jobs = [
        (i, args.seed_base + i, weights,
         mod_depth, mod_schedule, args.mod_leaf_mode,
         not args.mod_no_incremental_replay,
         args.mod_root_c_policy, args.mod_algo_policy,
         args.mod_algo_flags, args.mod_algo_value_tiebreak,
         args.mod_robust_opponent_model, args.mod_robust_penalty_weight,
         args.mod_leaf_pressure, args.mod_leaf_threat_bonus, args.mod_leaf_threat_vp,
         args.mod_endgame_extra_depth, args.mod_threat_extra_depth,
         args.mod_threat_opp_ab_depth, args.mod_opp_ab_depth,
         args.mod_iterative, args.mod_iter_start_depth,
         args.mod_critical_vp, args.mod_critical_extra,
         args.orig_depth, orig_schedule, args.orig_leaf_mode,
         not args.orig_no_incremental_replay,
         args.orig_root_c_policy, args.orig_algo_policy,
         args.orig_algo_flags, args.orig_algo_value_tiebreak,
         args.orig_robust_opponent_model, args.orig_robust_penalty_weight,
         args.orig_leaf_pressure, args.orig_leaf_threat_bonus, args.orig_leaf_threat_vp,
         args.orig_endgame_extra_depth, args.orig_threat_extra_depth,
         args.orig_threat_opp_ab_depth, args.orig_opp_ab_depth,
         args.orig_iterative, args.orig_iter_start_depth,
         args.orig_critical_vp, args.orig_critical_extra,
         args.time_ms, args.parallel_workers)
        for i in range(args.games)
    ]

    print("=== Super M2 config 2v2 H2H ===")
    print(
        f"  modified: depth={mod_depth} k={mod_schedule} leaf_mode={args.mod_leaf_mode} "
        f"incr_replay={int(not args.mod_no_incremental_replay)} "
        f"root_c={int(args.mod_root_c_policy)} algo={int(args.mod_algo_policy)}"
        f" flags={args.mod_algo_flags} vt={int(args.mod_algo_value_tiebreak)}"
        f" robust={args.mod_robust_opponent_model}:{args.mod_robust_penalty_weight}"
    )
    print(
        f"  original: depth={args.orig_depth} k={orig_schedule} leaf_mode={args.orig_leaf_mode} "
        f"incr_replay={int(not args.orig_no_incremental_replay)} "
        f"root_c={int(args.orig_root_c_policy)} algo={int(args.orig_algo_policy)}"
        f" flags={args.orig_algo_flags} vt={int(args.orig_algo_value_tiebreak)}"
        f" robust={args.orig_robust_opponent_model}:{args.orig_robust_penalty_weight}"
    )
    print(f"  games={args.games} workers={args.workers}")
    if args.parallel_workers:
        print(f"  per-decision parallel workers={args.parallel_workers} (backend=c2)")
    print()

    t0 = time.perf_counter()
    mod_wins = 0
    mod_rank_sum = 0.0
    orig_rank_sum = 0.0
    completed = 0

    if args.workers == 1:
        result_iter = (play_one(job) for job in jobs)
    else:
        ctx = mp.get_context("spawn")
        result_iter = ctx.Pool(processes=args.workers).imap_unordered(play_one, jobs)

    try:
        for r in result_iter:
            completed += 1
            mod_wins += int(r["mod_win"])
            mod_rank_sum += r["mod_avg_rank"]
            orig_rank_sum += r["orig_avg_rank"]
            print(
                f"[{completed}/{args.games}] g{r['game']} seed={r['seed']} "
                f"mod={r['mod_seats']} orig={r['orig_seats']} "
                f"W={r['winner']} actualVP={r['vps']} publicVP={r['public_vps']} "
                f"mod_win={r['mod_win']} turns={r['turns']} "
                f"time={r['seconds']:.1f}s "
                f"rank mod/orig={r['mod_avg_rank']:.2f}/{r['orig_avg_rank']:.2f}",
                flush=True,
            )
    finally:
        if hasattr(result_iter, "close"):
            result_iter.close()

    elapsed = time.perf_counter() - t0
    print()
    print("===== RESULTS =====")
    print(f"modified wins: {mod_wins}/{args.games}")
    print(f"avg rank: modified={mod_rank_sum/args.games:.2f} original={orig_rank_sum/args.games:.2f}")
    print(f"wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
