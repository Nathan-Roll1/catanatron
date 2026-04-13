#!/usr/bin/env python3
"""ELO arena: random 4-player matchups with replacement, Bradley-Terry ELO.

Usage:
  python -u human_bot/elo_arena.py --games 300 --workers 16 \
      --agents cl4k@10 cl4k@15 cl4k@20 cl4k@25 cl4k@30
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import random
import time

import numpy as np

CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc")


def _resolve_weights(name):
    if os.path.exists(name):
        return os.path.abspath(name)
    path = os.path.join(CSRC, f"nn_weights_{name}.bin")
    if os.path.exists(path):
        return os.path.abspath(path)
    raise FileNotFoundError(f"No weights for '{name}' (tried {path})")


def play_one_game(args):
    seed, players, depths, weight_paths = args

    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    lib_path = os.path.join(CSRC, "libnn.dylib")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(CSRC, "libnn.so")

    loaded = {}
    for wp in set(weight_paths):
        nn_lib = ctypes.CDLL(lib_path)
        nn_lib.nn_load.restype = ctypes.c_int
        nn_lib.nn_forward.restype = None
        nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
        nn_lib.nn_value_only.restype = None
        nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
        mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        assert nn_lib.nn_load(mptr, wp.encode()) == 0
        loaded[wp] = (nn_lib, mptr, mbuf)

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)

    def _enc(game): se.encode_into(game.get_state_view(), nf, ef, ff)
    def _mask(le):
        mk[:] = 0; mn = ae.get_action_mask(le).numpy(); mk[:len(mn)] = mn
        return mn

    def c_val(game, lib, ptr):
        _enc(game); _mask(game.get_legal_actions())
        lib.nn_value_only(ptr, nfp, efp, ffp, mkp, vlp)
        return vl.copy()

    def c_topk(game, le, k, lib, ptr):
        _enc(game); mn = _mask(le)
        out = np.zeros(4 + 397, dtype=np.float32)
        lib.nn_forward(ptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]
        a2i = {ae.encode(a): i for i, a in enumerate(le)}
        return [li for _, li in sorted([(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]

    def c_argmax(gc, lib, ptr):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        _enc(gc); mn = _mask(le)
        out = np.zeros(4 + 397, dtype=np.float32)
        lib.nn_forward(ptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        gc.step(next((i for i, a in enumerate(le) if ae.encode(a) == int(np.argmax(lo))), 0))

    def nnt_search(game, le, depth, lib, ptr, top_k=5):
        seat = game.current_player()
        candidates = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            candidates = c_topk(game, le, top_k, lib, ptr)
        bp, bv = 0, -1e30
        for p, ci in enumerate(candidates):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal(): break
                c_argmax(gc, lib, ptr)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
            else:
                vs = c_val(gc, lib, ptr)
                off = (seat - gc.current_player()) % 4
                v = float(vs[off])
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(candidates[bp], le)

    seat_libs = []
    for s in range(4):
        lib, ptr, _ = loaded[weight_paths[s]]
        seat_libs.append((lib, ptr, depths[s]))

    game = CatanGame(seed=seed); game.reset()
    t0 = time.perf_counter()
    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); continue
        cp = game.current_player()
        lib, ptr, depth = seat_libs[cp]
        game.step(nnt_search(game, le, depth, lib, ptr))
    el = time.perf_counter() - t0
    w = game.winner()
    winner_agent = players[w] if w is not None else None
    return (seed, w, winner_agent, players, game.turn_number, el)


def bradley_terry_elo(agents, pair_wins):
    from scipy.optimize import minimize
    p2i = {p: i for i, p in enumerate(agents)}
    NP = len(agents)

    def neg_ll(params):
        ll = 0.0
        for (a, b), (wa, wb) in pair_wins.items():
            if wa + wb == 0: continue
            ia, ib = p2i[a], p2i[b]
            ll -= wa * (params[ia] - np.logaddexp(params[ia], params[ib]))
            ll -= wb * (params[ib] - np.logaddexp(params[ia], params[ib]))
        ll += 0.01 * np.sum(params ** 2)
        return ll

    result = minimize(neg_ll, np.zeros(NP), method='L-BFGS-B')
    scale = 400.0 / np.log(10)
    return {p: 1500 + result.x[p2i[p]] * scale for p in agents}


def main():
    parser = argparse.ArgumentParser(description="ELO arena with random matchups")
    parser.add_argument("-n", "--games", type=int, default=300)
    parser.add_argument("-w", "--workers", type=int, default=16)
    parser.add_argument("--agents", nargs="+", required=True,
                        help="Agent specs: model@depth (e.g. cl4k@10 cl6k@15)")
    parser.add_argument("--seed-base", type=int, default=200000)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()

    agent_specs = []
    for spec in args.agents:
        if "@" in spec:
            model, depth = spec.rsplit("@", 1)
            agent_specs.append((spec, model, int(depth)))
        else:
            agent_specs.append((f"{spec}@10", spec, 10))

    agent_names = [s[0] for s in agent_specs]
    agent_weights = {s[0]: _resolve_weights(s[1]) for s in agent_specs}
    agent_depths = {s[0]: s[2] for s in agent_specs}

    rng = random.Random(args.rng_seed)
    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi
        players = [rng.choice(agent_names) for _ in range(4)]
        depths = [agent_depths[p] for p in players]
        wps = [agent_weights[p] for p in players]
        jobs.append((seed, players, depths, wps))

    print(f"ELO Arena: {args.games} games, {len(agent_names)} agents, {args.workers} workers")
    print(f"Agents: {', '.join(agent_names)}\n")

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one_game, jobs)
    wall = time.perf_counter() - t0

    from collections import defaultdict
    pair_wins = defaultdict(lambda: [0, 0])
    wins = defaultdict(int)
    games_count = defaultdict(int)

    for seed, w, winner_agent, players, turns, el in results:
        if winner_agent is None: continue
        wins[winner_agent] += 1
        for s in range(4):
            games_count[players[s]] += 1
            if s != w:
                loser = players[s]
                key = (min(winner_agent, loser), max(winner_agent, loser))
                if winner_agent < loser:
                    pair_wins[key][0] += 1
                else:
                    pair_wins[key][1] += 1

    elo = bradley_terry_elo(agent_names, pair_wins)

    print(f"\n{'='*60}")
    print(f"  ELO Rankings — {args.games} games, Bradley-Terry")
    print(f"{'='*60}")
    print(f"  {'#':>3s} {'Agent':<16s} {'ELO':>6s} {'Games':>6s} {'Wins':>5s} {'WR':>5s}")
    print(f"  {'-'*44}")
    for rank, name in enumerate(sorted(agent_names, key=lambda x: -elo[x])):
        g = games_count.get(name, 0)
        w = wins.get(name, 0)
        wr = 100 * w / max(g, 1)
        print(f"  {rank+1:>3d} {name:<16s} {elo[name]:>6.0f} {g:>6d} {w:>5d} {wr:>4.0f}%")

    print(f"\n  Wall time: {wall:.0f}s ({args.games/wall*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
