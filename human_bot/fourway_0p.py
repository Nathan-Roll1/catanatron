#!/usr/bin/env python3
"""4-way 0-ply matchup: each game has exactly one of each model, random seat order.

Usage:
    python -u human_bot/fourway_0p.py --games 1000 --workers 16
    python -u human_bot/fourway_0p.py --model original=csrc/nn_weights_m2.bin \
        --model incumbent=csrc/nn_weights_candidate.bin --model keepA=... \
        --model keepB=... --games 1000 --workers 16
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")

MODELS = [
    ("cl2k", os.path.join(CSRC, "nn_weights_cl2k.bin")),
    ("cl4k", os.path.join(CSRC, "nn_weights_cl4k.bin")),
    ("cl5k", os.path.join(CSRC, "nn_weights_cl5k.bin")),
    ("cl6k", os.path.join(CSRC, "nn_weights_cl6k.bin")),
]


def play_one(args):
    seed, seat_order = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    lib_path = os.path.join(CSRC, "..", "catan_player", "libcatan_nn.dylib")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(CSRC, "libnn.dylib")

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP)

    seat_nn = []
    for name, wpath in seat_order:
        nn_lib = ctypes.CDLL(lib_path)
        nn_lib.nn_load.restype = ctypes.c_int
        nn_lib.nn_forward.restype = None
        nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
        mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        assert nn_lib.nn_load(mptr, wpath.encode()) == 0
        seat_nn.append((name, nn_lib, mptr, mbuf))

    def nn_argmax(seat, game, le):
        _, nn_lib, mptr, _ = seat_nn[seat]
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        ai = int(np.argmax(lo))
        return next((i for i, a in enumerate(le) if ae.encode(a) == ai), 0)

    game = CatanGame(seed=seed); game.reset()
    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); continue
        cp = game.current_player()
        game.step(nn_argmax(cp, game, le))

    w = game.winner()
    winner_name = seat_nn[w][0] if w is not None else None
    return (seed, w, winner_name, [s[0] for s in seat_nn], game.turn_number)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model as label=weights.bin. Provide exactly four to override defaults.",
    )
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=150000)
    parser.add_argument("--rng-seed", type=int, default=42)
    args = parser.parse_args()

    models = list(MODELS)
    if args.model:
        if len(args.model) != 4:
            raise SystemExit("--model override requires exactly four label=path values")
        models = []
        for spec in args.model:
            if "=" not in spec:
                raise SystemExit(f"Bad --model value {spec!r}; expected label=path")
            label, path = spec.split("=", 1)
            label = label.strip()
            path = os.path.abspath(os.path.expanduser(path.strip()))
            if not label:
                raise SystemExit(f"Bad --model value {spec!r}; empty label")
            if not os.path.exists(path):
                raise SystemExit(f"Model path does not exist for {label}: {path}")
            models.append((label, path))

    rng = random.Random(args.rng_seed)
    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi
        order = list(models)
        rng.shuffle(order)
        jobs.append((seed, order))

    names = [m[0] for m in models]
    print(f"4-way 0-ply: {', '.join(names)} — {args.games} games, {args.workers} workers",
          flush=True)

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    wall = time.perf_counter() - t0

    wins = defaultdict(int)
    games_played = defaultdict(int)
    pair_wins = defaultdict(lambda: [0, 0])

    for seed, w, winner, players, turns in results:
        for p in players:
            games_played[p] += 1
        if winner is None:
            continue
        wins[winner] += 1
        for p in players:
            if p != winner:
                key = (min(winner, p), max(winner, p))
                if winner < p:
                    pair_wins[key][0] += 1
                else:
                    pair_wins[key][1] += 1

    from scipy.optimize import minimize
    p2i = {p: i for i, p in enumerate(names)}
    NP = len(names)

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
    elo = {p: 1500 + result.x[p2i[p]] * scale for p in names}

    print(f"\n{'='*55}")
    print(f"  0-ply ELO — {args.games} games, Bradley-Terry")
    print(f"{'='*55}")
    print(f"  {'#':>3s} {'Model':<8s} {'ELO':>6s} {'Games':>6s} {'Wins':>5s} {'WR':>5s}")
    print(f"  {'-'*36}")
    for rank, name in enumerate(sorted(names, key=lambda x: -elo[x])):
        g = games_played[name]
        w = wins[name]
        wr = 100 * w / max(g, 1)
        print(f"  {rank+1:>3d} {name:<8s} {elo[name]:>6.0f} {g:>6d} {w:>5d} {wr:>4.0f}%")

    print(f"\n  Wall time: {wall:.0f}s ({args.games/wall*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
