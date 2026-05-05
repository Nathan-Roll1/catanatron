#!/usr/bin/env python3
"""Verify the AB-expert teacher is legitimately stronger than the AB2
eval opponent. Gates the ground-up redesign.

What it does:
  1. AB{TEACHER_DEPTH} vs AB2 in N games (alternating seats)
  2. Report win rate, confidence interval, avg game length
  3. Report AB{TEACHER_DEPTH} decision-time distribution so we can
     size the cluster throughput expectation

Pass criterion: teacher wins ≥55% (ideally ≥60%) with tight CI.
Fail means the teacher isn't materially stronger than the eval
opponent, so training on its targets won't help.

Usage:
    python3 -u human_bot/verify_ab_teacher.py \\
        --teacher-depth 4 --opponent-depth 2 --num-games 30
"""
from __future__ import annotations
import argparse
import ctypes
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-depth", type=int, default=4)
    p.add_argument("--opponent-depth", type=int, default=2)
    p.add_argument("--teacher-eval", choices=["base", "diff"], default="base",
                   help="base = lib.base_value_fn (1-sided); "
                        "diff = us - max_over_opps base_value_fn")
    p.add_argument("--opponent-eval", choices=["base", "diff"], default="base")
    p.add_argument("--teacher-search", choices=["ab", "sta"], default="ab",
                   help="ab = standard alphabeta_search; "
                        "sta = alphabeta_search_same_turn (Python "
                        "SameTurnAlphaBetaPlayer equivalent)")
    p.add_argument("--opponent-search", choices=["ab", "sta"], default="ab")
    p.add_argument("--num-games", type=int, default=30)
    p.add_argument("--num-players", type=int, default=4)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
    )

    lib = load_library()
    ctx = SearchCtx()
    action_buf = (CAction * MAX_ACTIONS)()

    # Build eval function variants. `base` is the 1-sided C heuristic.
    # `diff` wraps the same C function but returns us-minus-max-opponent —
    # this fixes the paranoid-minimax pathology in multi-player games,
    # letting opponents' moves actually track their threats to us.
    base_eval_cb = ValueFn(lib.base_value_fn)

    def _diff_py(g_ptr, color):
        g = g_ptr.contents
        us = lib.base_value_fn(g_ptr, color)
        n_players = g.state.num_players
        max_opp = -1e300
        for p in range(n_players):
            c = g.state.colors[p]
            if c == color:
                continue
            v = lib.base_value_fn(g_ptr, c)
            if v > max_opp:
                max_opp = v
        if max_opp <= -1e299:
            return us
        return us - max_opp
    diff_eval_cb = ValueFn(_diff_py)

    def _pick_eval(which):
        return diff_eval_cb if which == "diff" else base_eval_cb

    teacher_eval_cb = _pick_eval(args.teacher_eval)
    opp_eval_cb = _pick_eval(args.opponent_eval)

    teacher_search_fn = (lib.alphabeta_search_same_turn
                          if args.teacher_search == "sta"
                          else lib.alphabeta_search)
    opp_search_fn = (lib.alphabeta_search_same_turn
                      if args.opponent_search == "sta"
                      else lib.alphabeta_search)

    def ab_pick(game, depth, eval_cb, search_fn):
        le = game.get_legal_actions()
        n = len(le)
        if n == 0:
            return None, le, 0.0
        if n == 1:
            return 0, le, 0.0
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le):
            action_buf[i] = a
        t0 = time.time()
        res = search_fn(
            ctypes.byref(ctx), ctypes.byref(cg), action_buf,
            ctypes.c_int(n), ctypes.c_int(depth),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), eval_cb,
        )
        dt = time.time() - t0
        cb = ctypes.string_at(ctypes.byref(res.action),
                              ctypes.sizeof(res.action))
        for i, a in enumerate(le):
            if ctypes.string_at(ctypes.byref(a),
                                ctypes.sizeof(a)) == cb:
                return i, le, dt
        return 0, le, dt

    print(f"=== Verify teacher ({args.teacher_search}{args.teacher_depth}/"
          f"{args.teacher_eval}) vs "
          f"opponent ({args.opponent_search}{args.opponent_depth}/"
          f"{args.opponent_eval}) "
          f"({args.num_games} games, {args.num_players}p) ===")

    teacher_wins = 0
    losses = 0
    draws = 0
    teacher_times = []
    opp_times = []
    game_lengths = []
    t_run = time.time()

    for g in range(args.num_games):
        teacher_seat = g % args.num_players
        cfg = GameConfig(num_players=args.num_players)
        game = CatanGame(seed=args.seed + g, random_board=True, config=cfg)
        game.reset()

        while not game.is_terminal() and game.turn_number < 750:
            le = game.get_legal_actions()
            if not le: break
            cp = game.current_player()
            if cp == teacher_seat:
                depth, eval_cb, sfn = (args.teacher_depth, teacher_eval_cb,
                                         teacher_search_fn)
            else:
                depth, eval_cb, sfn = (args.opponent_depth, opp_eval_cb,
                                         opp_search_fn)
            chosen, le, dt = ab_pick(game, depth, eval_cb, sfn)
            if chosen is None: break
            if cp == teacher_seat: teacher_times.append(dt)
            else: opp_times.append(dt)
            game.step(chosen)

        w = game.winner()
        if w == teacher_seat: teacher_wins += 1
        elif w is None: draws += 1
        else: losses += 1
        game_lengths.append(game.turn_number)

        elapsed = time.time() - t_run
        gps = (g + 1) / elapsed if elapsed > 0 else 0
        print(f"  game {g+1}/{args.num_games}: winner={w} "
              f"(teacher seat={teacher_seat}) | "
              f"teacher_wr={teacher_wins/(g+1):.1%} "
              f"({gps:.2f} g/s)",
              flush=True)

    print()
    total = teacher_wins + losses + draws
    wr = teacher_wins / total if total else 0.0
    # Wilson CI (95%)
    from math import sqrt
    z = 1.96
    n = total
    if n > 0:
        p_hat = wr
        denom = 1 + z**2/n
        centre = (p_hat + z**2/(2*n)) / denom
        half = z * sqrt((p_hat*(1-p_hat) + z**2/(4*n))/n) / denom
        ci_lo, ci_hi = max(0, centre - half), min(1, centre + half)
    else:
        ci_lo = ci_hi = 0

    teacher_times = np.array(teacher_times) * 1000  # ms
    opp_times = np.array(opp_times) * 1000

    print(f"=== RESULTS ===")
    print(f"  teacher win rate: {wr:.1%} "
          f"(95% CI: [{ci_lo:.1%}, {ci_hi:.1%}])")
    print(f"  teacher wins / draws / losses: "
          f"{teacher_wins} / {draws} / {losses}")
    print(f"  avg game length: {np.mean(game_lengths):.1f} turns")
    print(f"  teacher decision ms: "
          f"median={np.median(teacher_times):.1f} "
          f"p90={np.percentile(teacher_times, 90):.1f} "
          f"p99={np.percentile(teacher_times, 99):.1f} "
          f"mean={np.mean(teacher_times):.1f}")
    print(f"  opponent decision ms: "
          f"median={np.median(opp_times):.1f} "
          f"mean={np.mean(opp_times):.1f}")

    decisions_per_game = len(teacher_times) / args.num_games
    sec_per_game = decisions_per_game * np.mean(teacher_times) / 1000
    print(f"  est. teacher-pure-game time: ~{sec_per_game:.1f}s")
    print(f"  est. throughput per actor: "
          f"~{3600/max(1, sec_per_game):.0f} games/hour")

    print()
    if wr >= 0.55:
        print("PASS: teacher is stronger than eval opponent "
              f"(WR {wr:.1%} >= 55%).")
    else:
        print(f"FAIL: teacher WR {wr:.1%} < 55% — "
              "depth gap too narrow to teach anything useful.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
