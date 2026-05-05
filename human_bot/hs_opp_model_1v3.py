#!/usr/bin/env python3
"""1v3 H-S benchmark for opponent-model experiments.

Seat 0 runs the modified H-S configuration. Seats 1-3 run baseline H-S by
default. This isolates whether changing H-S's internal opponent model helps
against the current no-ML bot.
"""

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


def _make_bot(
    weights: str,
    depth: int,
    schedule: tuple[int, ...],
    time_ms: int,
    leaf_mode: int,
    opponent_model: str,
) -> SuperBotV3C2:
    return SuperBotV3C2(
        weights,
        our_depth=depth,
        top_k_schedule=schedule,
        time_budget_ms=time_ms,
        opponent_ab_depth=2,
        leaf_mode=leaf_mode,
        algo_policy=True,
        opponent_model=opponent_model,
    )


def play_one(args: tuple) -> dict:
    (
        game_idx,
        seed,
        weights,
        mod_depth,
        mod_schedule,
        mod_leaf_mode,
        mod_opponent_model,
        base_depth,
        base_schedule,
        base_leaf_mode,
        base_opponent_model,
        time_ms,
    ) = args

    bots = [
        _make_bot(weights, mod_depth, mod_schedule, time_ms, mod_leaf_mode, mod_opponent_model)
    ]
    for _ in range(3):
        bots.append(
            _make_bot(
                weights,
                base_depth,
                base_schedule,
                time_ms,
                base_leaf_mode,
                base_opponent_model,
            )
        )

    game = CatanGame(seed=seed)
    game.reset()
    t0 = time.perf_counter()

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        if len(legal) == 1:
            chosen = 0
        else:
            chosen = bots[game.current_player()].pick(game)
        game.step(chosen)

    elapsed = time.perf_counter() - t0
    public_vps = [
        int(game._game.state.player_state[s][PS_VICTORY_POINTS])
        for s in range(4)
    ]
    actual_vps = [
        int(game._game.state.player_state[s][PS_ACTUAL_VICTORY_POINTS])
        for s in range(4)
    ]
    ranks = sorted(range(4), key=lambda s: -actual_vps[s])
    mod_rank = ranks.index(0) + 1
    baseline_ranks = [ranks.index(s) + 1 for s in (1, 2, 3)]
    return {
        "game": game_idx,
        "seed": seed,
        "winner": game.winner(),
        "mod_win": game.winner() == 0,
        "actual_vps": actual_vps,
        "public_vps": public_vps,
        "mod_rank": mod_rank,
        "baseline_avg_rank": sum(baseline_ranks) / len(baseline_ranks),
        "turns": game.turn_number,
        "seconds": elapsed,
        "mod_stats": bots[0].stats_summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=5000)
    parser.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    parser.add_argument("--mod-depth", type=int, default=5)
    parser.add_argument("--mod-k-schedule", default="6,4,2,2,2")
    parser.add_argument("--mod-leaf-mode", type=int, default=0)
    opponent_choices = ["ab2", "hs", "h-s", "algo", "hs1", "hs-leaf", "h-s-leaf", "algo-leaf"]
    parser.add_argument("--mod-opponent-model", default="hs", choices=opponent_choices)
    parser.add_argument("--base-depth", type=int, default=5)
    parser.add_argument("--base-k-schedule", default="6,4,2,2,2")
    parser.add_argument("--base-leaf-mode", type=int, default=0)
    parser.add_argument("--base-opponent-model", default="ab2", choices=opponent_choices)
    parser.add_argument("--time-ms", type=int, default=5000)
    args = parser.parse_args()

    weights = os.path.abspath(args.weights)
    mod_schedule = tuple(int(x) for x in args.mod_k_schedule.split(","))
    base_schedule = tuple(int(x) for x in args.base_k_schedule.split(","))
    jobs = [
        (
            i,
            args.seed_base + i,
            weights,
            args.mod_depth,
            mod_schedule,
            args.mod_leaf_mode,
            args.mod_opponent_model,
            args.base_depth,
            base_schedule,
            args.base_leaf_mode,
            args.base_opponent_model,
            args.time_ms,
        )
        for i in range(args.games)
    ]

    print("=== H-S opponent-model 1v3 ===", flush=True)
    print(
        f"  modified seat0: depth={args.mod_depth} k={mod_schedule} "
        f"leaf={args.mod_leaf_mode} opp_model={args.mod_opponent_model}",
        flush=True,
    )
    print(
        f"  baseline seats1-3: depth={args.base_depth} k={base_schedule} "
        f"leaf={args.base_leaf_mode} opp_model={args.base_opponent_model}",
        flush=True,
    )
    print(f"  games={args.games} workers={args.workers} seed_base={args.seed_base}", flush=True)
    print(flush=True)

    t0 = time.perf_counter()
    wins = 0
    mod_rank_sum = 0.0
    base_rank_sum = 0.0
    mod_vp_sum = 0.0
    base_vp_sum = 0.0
    completed = 0

    if args.workers == 1:
        result_iter = (play_one(job) for job in jobs)
    else:
        ctx = mp.get_context("spawn")
        result_iter = ctx.Pool(processes=args.workers).imap_unordered(play_one, jobs)

    try:
        for r in result_iter:
            completed += 1
            wins += int(r["mod_win"])
            mod_rank_sum += r["mod_rank"]
            base_rank_sum += r["baseline_avg_rank"]
            mod_vp_sum += r["actual_vps"][0]
            base_vp_sum += sum(r["actual_vps"][1:]) / 3.0
            print(
                f"[{completed}/{args.games}] g{r['game']} seed={r['seed']} "
                f"W={r['winner']} mod_win={r['mod_win']} "
                f"actualVP={r['actual_vps']} publicVP={r['public_vps']} "
                f"rank mod/base={r['mod_rank']:.2f}/{r['baseline_avg_rank']:.2f} "
                f"turns={r['turns']} time={r['seconds']:.1f}s",
                flush=True,
            )
    finally:
        pass

    elapsed = time.perf_counter() - t0
    n = max(completed, 1)
    print("\n===== RESULTS =====", flush=True)
    print(f"modified wins: {wins}/{completed} ({100.0 * wins / n:.1f}%)", flush=True)
    print(f"avg VP: modified={mod_vp_sum / n:.2f} baseline={base_vp_sum / n:.2f}", flush=True)
    print(f"avg rank: modified={mod_rank_sum / n:.2f} baseline={base_rank_sum / n:.2f}", flush=True)
    print(f"wall time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
