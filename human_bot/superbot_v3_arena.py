"""Parallel arena: 100 games of SuperBotV3 d=5 vs AB2 in 1v3 mode.

Game-level parallelism: each worker process plays one full game start to
finish, then takes the next from the queue. M5 Max has 12 P-cores; using
~10 workers leaves headroom for OS + parent process.

Each worker independently loads:
  - libcatan (game engine)
  - libnn (NN inference) + M2 weights
  - SuperBotV3 instance

These are heavy first-time loads (~1-2s each) but happen once per worker.

Random seat assignment per game: NN seat = (game_idx + seed_perm) % 4
where seed_perm rotates per game for fairness across all 4 seats.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from typing import Optional


def _play_one_game(args):
    """Worker: play one full game, return (game_idx, nn_seat, winner, vps, turns, dt)."""
    (game_idx, seed, nn_seat, weights_path, our_depth, k_schedule,
     entropy_thresh, time_budget_ms) = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import ctypes as C
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.game.interface import CatanGame
    from human_bot.superbot_v3 import SuperBotV3

    lib = load_library()
    bot = SuperBotV3(weights_path,
                     our_depth=our_depth,
                     top_k_schedule=k_schedule,
                     entropy_fast_thresh=entropy_thresh,
                     use_leaf_cache=True,
                     leaf_cache_bits=20,
                     time_budget_ms=time_budget_ms)

    ab_ctx = SearchCtx()
    ab_buf = (CAction * MAX_ACTIONS)()
    ab_eval = ValueFn(lib.base_value_fn)

    def ab2_choose(game, le):
        n = len(le)
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le): ab_buf[i] = a
        res = lib.alphabeta_search(
            C.byref(ab_ctx), C.byref(cg), ab_buf,
            C.c_int(n), C.c_int(2),
            C.c_double(-1e30), C.c_double(1e30),
            C.c_int(bc), ab_eval)
        cb = C.string_at(C.byref(res.action), C.sizeof(res.action))
        for i, a in enumerate(le):
            if C.string_at(C.byref(a), C.sizeof(a)) == cb: return i
        return 0

    t0 = time.time()
    game = CatanGame(seed=seed); game.reset()
    nn_seats = {nn_seat}

    while not game.is_terminal() and game.turn_number < 500:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        if cp in nn_seats:
            game.step(bot.pick(game))
        else:
            game.step(ab2_choose(game, le))

    dt = time.time() - t0
    w = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    return game_idx, nn_seat, w, vps, game.turn_number, dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--k-schedule", type=str, default="10,7,5,4,3")
    parser.add_argument("--time-ms", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=80000)
    parser.add_argument("--entropy-thresh", type=float, default=0.15)
    args = parser.parse_args()

    k_schedule = tuple(int(x) for x in args.k_schedule.split(","))
    weights_path = os.path.abspath(args.weights)

    # Build job list: each game has a different NN seat for fairness.
    # Cycle seats 0,1,2,3,0,1,2,3,... so each seat gets ~25 games of 100.
    jobs = []
    for gi in range(args.games):
        nn_seat = gi % 4
        seed = args.seed_base + gi
        jobs.append((gi, seed, nn_seat, weights_path, args.depth, k_schedule,
                     args.entropy_thresh, args.time_ms))

    print(f"=== Arena: SuperBotV3 d={args.depth} k={k_schedule} ===")
    print(f"  Games:    {args.games} (1v3, random NN seat)")
    print(f"  Workers:  {args.workers}")
    print(f"  Weights:  {args.weights}")
    print(f"  Time/dec: {args.time_ms}ms cap")
    print()

    ctx = mp.get_context("spawn")
    t_start = time.time()

    nn_wins = ab2_wins = 0
    rank_sum = 0
    nn_vp_sum = 0
    ab_vp_sum = 0
    seat_results = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}  # [wins, total]
    game_results = []
    completed = 0

    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_play_one_game, jobs):
            gi, nn_seat, w, vps, turns, dt = result
            completed += 1
            game_results.append(result)

            # Stats
            nn_vp = vps[nn_seat]
            opp_vps = [vps[s] for s in range(4) if s != nn_seat]
            avg_opp_vp = sum(opp_vps) / 3
            nn_vp_sum += nn_vp
            ab_vp_sum += avg_opp_vp

            rank = sorted(range(4), key=lambda s: -vps[s]).index(nn_seat) + 1
            rank_sum += rank

            seat_results[nn_seat][1] += 1
            if w == nn_seat:
                nn_wins += 1
                seat_results[nn_seat][0] += 1
            elif w is not None:
                ab2_wins += 1

            elapsed = time.time() - t_start
            wr = nn_wins / max(nn_wins + ab2_wins, 1)
            print(f"  [{completed:>3d}/{args.games}] "
                  f"g{gi:>3d} seat={nn_seat} W={w} "
                  f"VP={nn_vp}/{int(avg_opp_vp)} rank={rank} "
                  f"({turns}t {dt:.0f}s) | "
                  f"WR={wr:.0%} ({nn_wins}/{nn_wins+ab2_wins}) "
                  f"avg_rank={rank_sum/completed:.2f} "
                  f"[{elapsed/60:.1f}min wall]",
                  flush=True)

    elapsed = time.time() - t_start
    total = nn_wins + ab2_wins
    wr = nn_wins / max(total, 1)

    print()
    print("=" * 70)
    print(f"FINAL RESULTS — {args.games} games, {args.workers} workers, {elapsed/60:.1f}min wall")
    print("=" * 70)
    print(f"  Wins:         {nn_wins} / {total} ({wr:.1%})")
    print(f"  Avg rank:     {rank_sum/args.games:.2f} / 4 (random=2.50, perfect=1.00)")
    print(f"  Avg VP:       NN={nn_vp_sum/args.games:.2f}  opp={ab_vp_sum/args.games:.2f}")
    print(f"  Wall time:    {elapsed:.0f}s ({args.games/elapsed*60:.1f} g/min)")
    print(f"  CPU time:     ~{elapsed*args.workers:.0f}s (×{args.workers} workers)")
    print()
    print(f"  Per-seat WR:")
    for s, (w, t) in sorted(seat_results.items()):
        if t > 0:
            print(f"    seat {s}: {w}/{t} = {w/t:.0%}")

    # Significance: Wilson 95% CI
    from math import sqrt
    n = args.games
    z = 1.96
    p_hat = wr
    denom = 1 + z**2/n
    centre = (p_hat + z**2/(2*n)) / denom
    half = z * sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
    ci_lo, ci_hi = max(0, centre - half), min(1, centre + half)
    print(f"\n  Wilson 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"  AB2 baseline (1v3 vs 3 AB2): expected ~25%")
    print(f"  Lift vs random seat: +{(wr - 0.25)*100:.0f} pts above random")


if __name__ == "__main__":
    main()
