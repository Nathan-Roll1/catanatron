#!/usr/bin/env python3
"""Serial C2 H-S vs AB2 evaluator for robust root experiments."""

from __future__ import annotations

import argparse
import ctypes as C
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexzero.bindings.lib_loader import load_library
from hexzero.bindings.structs import (
    Action as CAction,
    MAX_ACTIONS,
    PS_ACTUAL_VICTORY_POINTS,
    PS_VICTORY_POINTS,
    SearchCtx,
    ValueFn,
)
from hexzero.game.interface import CatanGame
from human_bot.superbot_v3_c2 import SuperBotV3C2


def play_one(args: tuple) -> dict:
    (
        game_idx,
        seed,
        weights,
        depth,
        schedule,
        leaf_mode,
        robust_model,
        robust_weight,
        algo_flags,
        algo_value_tiebreak,
        leaf_pressure,
        leaf_threat_bonus,
        leaf_threat_vp,
        endgame_extra,
        threat_extra,
        threat_opp_ab,
        opp_ab_depth,
        iterative,
        iter_start_depth,
        critical_vp,
        critical_extra,
        time_ms,
    ) = args

    lib = load_library()
    ab_ctx = SearchCtx()
    ab_buf = (CAction * MAX_ACTIONS)()
    ab_eval = ValueFn(lib.base_value_fn)

    bot = SuperBotV3C2(
        weights,
        our_depth=depth,
        top_k_schedule=schedule,
        time_budget_ms=time_ms,
        opponent_ab_depth=opp_ab_depth,
        leaf_mode=leaf_mode,
        algo_policy=True,
        algo_flags=algo_flags,
        algo_value_tiebreak=algo_value_tiebreak,
        robust_opponent_model=robust_model,
        robust_penalty_weight=robust_weight,
        leaf_pressure_weight=leaf_pressure,
        leaf_threat_bonus=leaf_threat_bonus,
        leaf_threat_vp=leaf_threat_vp,
        endgame_extra_depth=endgame_extra,
        threat_extra_depth=threat_extra,
        threat_opp_ab_depth=threat_opp_ab,
        iterative_deepening=iterative,
        iter_start_depth=iter_start_depth,
        critical_vp_threshold=critical_vp,
        critical_extra_depth=critical_extra,
    )

    def ab2_choose(game: CatanGame, legal: list) -> int:
        n = len(legal)
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, action in enumerate(legal):
            ab_buf[i] = action
        result = lib.alphabeta_search(
            C.byref(ab_ctx),
            C.byref(cg),
            ab_buf,
            C.c_int(n),
            C.c_int(2),
            C.c_double(-1e30),
            C.c_double(1e30),
            C.c_int(bc),
            ab_eval,
        )
        chosen_bytes = C.string_at(C.byref(result.action), C.sizeof(result.action))
        for i, action in enumerate(legal):
            if C.string_at(C.byref(action), C.sizeof(action)) == chosen_bytes:
                return i
        return 0

    game = CatanGame(seed=seed)
    game.reset()
    t0 = time.perf_counter()
    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        if len(legal) == 1:
            chosen = 0
        elif game.current_player() == 0:
            chosen = bot.pick(game)
        else:
            chosen = ab2_choose(game, legal)
        game.step(chosen)

    elapsed = time.perf_counter() - t0
    actual_vps = [
        int(game._game.state.player_state[s][PS_ACTUAL_VICTORY_POINTS])
        for s in range(4)
    ]
    public_vps = [
        int(game._game.state.player_state[s][PS_VICTORY_POINTS])
        for s in range(4)
    ]
    ranks = sorted(range(4), key=lambda s: -actual_vps[s])
    return {
        "game": game_idx,
        "seed": seed,
        "winner": game.winner(),
        "win": game.winner() == 0,
        "actual_vps": actual_vps,
        "public_vps": public_vps,
        "rank": ranks.index(0) + 1,
        "turns": game.turn_number,
        "seconds": elapsed,
        "stats": bot.stats_summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=7400)
    parser.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--k-schedule", default="6,4,2,2,2,2")
    parser.add_argument("--leaf-mode", type=int, default=4)
    parser.add_argument("--robust-opponent-model", default=None)
    parser.add_argument("--robust-penalty-weight", type=float, default=0.0)
    parser.add_argument("--algo-flags", type=int, default=0)
    parser.add_argument("--algo-value-tiebreak", action="store_true")
    parser.add_argument("--leaf-pressure", type=float, default=None)
    parser.add_argument("--leaf-threat-bonus", type=float, default=None)
    parser.add_argument("--leaf-threat-vp", type=int, default=8)
    parser.add_argument("--endgame-extra-depth", type=int, default=0)
    parser.add_argument("--threat-extra-depth", type=int, default=0)
    parser.add_argument("--threat-opp-ab-depth", type=int, default=2)
    parser.add_argument("--opp-ab-depth", type=int, default=2)
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--iter-start-depth", type=int, default=2)
    parser.add_argument("--critical-vp", type=int, default=100)
    parser.add_argument("--critical-extra", type=int, default=0)
    parser.add_argument("--time-ms", type=int, default=5000)
    args = parser.parse_args()

    weights = os.path.abspath(args.weights)
    schedule = tuple(int(x) for x in args.k_schedule.split(","))
    jobs = [
        (
            i,
            args.seed_base + i,
            weights,
            args.depth,
            schedule,
            args.leaf_mode,
            args.robust_opponent_model,
            args.robust_penalty_weight,
            args.algo_flags,
            args.algo_value_tiebreak,
            args.leaf_pressure,
            args.leaf_threat_bonus,
            args.leaf_threat_vp,
            args.endgame_extra_depth,
            args.threat_extra_depth,
            args.threat_opp_ab_depth,
            args.opp_ab_depth,
            args.iterative,
            args.iter_start_depth,
            args.critical_vp,
            args.critical_extra,
            args.time_ms,
        )
        for i in range(args.games)
    ]

    print("=== Robust H-S vs AB2 ===", flush=True)
    print(
        f"  depth={args.depth} k={schedule} leaf={args.leaf_mode} "
        f"robust={args.robust_opponent_model}:{args.robust_penalty_weight} "
        f"flags={args.algo_flags} vt={int(args.algo_value_tiebreak)}",
        flush=True,
    )
    print(f"  games={args.games} workers={args.workers} seed_base={args.seed_base}\n", flush=True)

    t0 = time.perf_counter()
    wins = 0
    rank_sum = 0.0
    vp_sum = 0.0
    completed = 0
    if args.workers == 1:
        result_iter = (play_one(job) for job in jobs)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            result_iter = pool.imap_unordered(play_one, jobs)
            for r in result_iter:
                completed += 1
                wins += int(r["win"])
                rank_sum += r["rank"]
                vp_sum += r["actual_vps"][0]
                print(
                    f"[{completed}/{args.games}] g{r['game']} seed={r['seed']} "
                    f"W={r['winner']} win={r['win']} actualVP={r['actual_vps']} "
                    f"publicVP={r['public_vps']} rank={r['rank']} "
                    f"turns={r['turns']} time={r['seconds']:.1f}s",
                    flush=True,
                )
            result_iter = None
    if result_iter is not None:
        for r in result_iter:
            completed += 1
            wins += int(r["win"])
            rank_sum += r["rank"]
            vp_sum += r["actual_vps"][0]
            print(
                f"[{completed}/{args.games}] g{r['game']} seed={r['seed']} "
                f"W={r['winner']} win={r['win']} actualVP={r['actual_vps']} "
                f"publicVP={r['public_vps']} rank={r['rank']} "
                f"turns={r['turns']} time={r['seconds']:.1f}s",
                flush=True,
            )

    n = max(completed, 1)
    elapsed = time.perf_counter() - t0
    print("\n===== RESULTS =====", flush=True)
    print(f"wins: {wins}/{completed} ({100.0*wins/n:.1f}%)", flush=True)
    print(f"avg VP: {vp_sum/n:.2f}", flush=True)
    print(f"avg rank: {rank_sum/n:.2f}/4", flush=True)
    print(f"wall time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
