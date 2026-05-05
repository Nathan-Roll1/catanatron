#!/usr/bin/env python3
"""Four-seat arena: AB2, M2, M2-FT-1e-5, and M2-FT-1e-6.

Each game contains all four agents exactly once. Seats rotate so each agent
plays every position over four games.
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import sys
import time


AGENTS = [
    ("AB2", None),
    ("M2", "csrc/nn_weights_m2.bin"),
    ("M2-1e-5", "csrc/nn_weights_dense100_policyonly_lr1e5_freezebn.bin"),
    ("M2-1e-6", "csrc/nn_weights_dense100_policyonly_freezebn.bin"),
]


def _resolve(path: str | None) -> str | None:
    if path is None:
        return None
    return os.path.abspath(path)


def play_one(game_idx: int, seed: int, seat_agents: list[int], args):
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn
    from hexzero.game.interface import CatanGame
    from human_bot.superbot_v3_c2 import SuperBotV3C2

    lib = load_library()
    schedule = tuple(int(x) for x in args.k_schedule.split(","))

    bots = {}
    for agent_id in sorted(set(seat_agents)):
        name, wpath = AGENTS[agent_id]
        if wpath is None:
            continue
        bots[agent_id] = SuperBotV3C2(
            _resolve(wpath),
            our_depth=args.depth,
            top_k_schedule=schedule,
            entropy_fast_thresh=args.entropy_thresh,
            leaf_cache_bits=args.leaf_cache_bits,
            time_budget_ms=args.time_ms,
            opponent_ab_depth=2,
            leaf_mode=0,
        )

    ab_ctx = SearchCtx()
    ab_buf = (CAction * MAX_ACTIONS)()
    ab_eval = ValueFn(lib.base_value_fn)

    def ab2_choose(game, le):
        n = len(le)
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le):
            ab_buf[i] = a
        res = lib.alphabeta_search(
            C.byref(ab_ctx), C.byref(cg), ab_buf,
            C.c_int(n), C.c_int(2),
            C.c_double(-1e30), C.c_double(1e30),
            C.c_int(bc), ab_eval)
        cb = C.string_at(C.byref(res.action), C.sizeof(res.action))
        for i, a in enumerate(le):
            if C.string_at(C.byref(a), C.sizeof(a)) == cb:
                return i
        return 0

    game = CatanGame(seed=seed)
    game.reset()
    t0 = time.time()
    while not game.is_terminal() and game.turn_number < args.max_turns:
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            continue
        seat = game.current_player()
        agent_id = seat_agents[seat]
        if AGENTS[agent_id][1] is None:
            chosen = ab2_choose(game, le)
        else:
            chosen = bots[agent_id].pick(game)
        game.step(chosen)

    elapsed = time.time() - t0
    winner = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    ranks = [sorted(range(4), key=lambda s: (-vps[s], s)).index(s) + 1
             for s in range(4)]
    stats = {
        AGENTS[aid][0]: bots[aid].stats_summary()
        for aid in bots
    }
    return {
        "game_idx": game_idx,
        "seed": seed,
        "seat_agents": seat_agents,
        "winner": winner,
        "winner_agent": None if winner is None else seat_agents[winner],
        "vps": vps,
        "ranks": ranks,
        "turns": game.turn_number,
        "elapsed": elapsed,
        "stats": stats,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=360000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k-schedule", type=str, default="12,8,6,5,4,3")
    p.add_argument("--time-ms", type=int, default=4000)
    p.add_argument("--entropy-thresh", type=float, default=0.15)
    p.add_argument("--leaf-cache-bits", type=int, default=20)
    p.add_argument("--max-turns", type=int, default=500)
    args = p.parse_args()

    print("Mixed Super arena")
    print(f"  agents: {', '.join(name for name, _ in AGENTS)}")
    print(f"  games: {args.games}, depth={args.depth}, k={args.k_schedule}, time={args.time_ms}ms")
    print()

    totals = {
        i: {"wins": 0, "vp": 0.0, "rank": 0.0, "seats": [0, 0, 0, 0]}
        for i in range(len(AGENTS))
    }
    results = []
    t_start = time.time()
    for gi in range(args.games):
        seat_agents = [((seat - gi) % 4) for seat in range(4)]
        seed = args.seed_base + gi
        r = play_one(gi, seed, seat_agents, args)
        results.append(r)
        for seat, aid in enumerate(seat_agents):
            totals[aid]["vp"] += r["vps"][seat]
            totals[aid]["rank"] += r["ranks"][seat]
            totals[aid]["seats"][seat] += 1
        if r["winner_agent"] is not None:
            totals[r["winner_agent"]]["wins"] += 1
        seat_desc = ", ".join(f"P{s}={AGENTS[aid][0]}" for s, aid in enumerate(seat_agents))
        wname = "draw" if r["winner_agent"] is None else AGENTS[r["winner_agent"]][0]
        print(f"  {gi+1}/{args.games} seed={seed} W={wname} "
              f"VP={r['vps']} ranks={r['ranks']} turns={r['turns']} "
              f"({r['elapsed']:.0f}s) | {seat_desc}",
              flush=True)

    n = max(args.games, 1)
    print()
    print("===== RESULTS =====")
    for aid, (name, _) in enumerate(AGENTS):
        t = totals[aid]
        print(f"  {name:8s}: wins={t['wins']}/{args.games} "
              f"avg_vp={t['vp']/n:.2f} avg_rank={t['rank']/n:.2f} "
              f"seats={t['seats']}")
    print(f"  Wall: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    main()
