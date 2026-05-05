#!/usr/bin/env python3
"""4-model Elo arena: random seat assignments, Bradley-Terry ratings.

Usage:
    python -m human_bot.elo_arena_4way --games 2000 --workers 8
"""
import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time

import numpy as np

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")

MODELS = {
    0: ("M1_old", os.path.join(CSRC, "nn_weights_m1.bin")),
    1: ("M2_new", os.path.join(CSRC, "nn_weights_m2.bin")),
    2: ("AB1", None),
    3: ("AB2", None),
}


def play_one(args):
    seed, seat_assignments = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    nn_lib_path = os.path.join(CSRC, "libnn.dylib")
    if not os.path.exists(nn_lib_path):
        nn_lib_path = os.path.join(CSRC, "libnn.so")
    nn_lib = ctypes.CDLL(nn_lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [
        ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p
    ]

    nn_ptrs = {}
    for mid in set(seat_assignments):
        wpath = MODELS[mid][1]
        if wpath is None:
            continue
        if wpath in nn_ptrs:
            continue
        mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        assert nn_lib.nn_load(mptr, wpath.encode()) == 0
        nn_ptrs[wpath] = (mbuf, mptr)

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)
    ch2 = CGame()
    ca2 = (Action * MAX_ACTIONS)()
    cn2 = ctypes.c_int(0)

    def nn_argmax(game, le, wpath):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(
            nn_ptrs[wpath][1], nfp, efp, ffp, mkp,
            out.ctypes.data_as(ctypes.c_void_p),
        )
        lo = out[4:4 + AD]
        lo[mn[:AD] < 0.5] = -1e9
        ai = int(np.argmax(lo))
        for i, a in enumerate(le):
            try:
                if ae.encode(a) == ai:
                    return i
            except ValueError:
                continue
        return 0

    def ab_choose(game, le, depth):
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if (depth >= 2 and cn.value > 0
                    and lib.game_winning_color(ctypes.byref(ch)) < 0):
                if cn.value > 1:
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    brj, brv = 0, -1e30
                    for j in range(cn.value):
                        lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        lib.game_execute(
                            ctypes.byref(ch2), ca[j], ca2, ctypes.byref(cn2)
                        )
                        rv = lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > brv:
                            brv = rv
                            brj = j
                    lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    lib.game_execute(
                        ctypes.byref(ch2), ca[brj], ca2, ctypes.byref(cn2)
                    )
                    v = lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    lib.game_execute(
                        ctypes.byref(ch), ca[0], ca, ctypes.byref(cn)
                    )
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv:
                bv = v
                bi = i
        return bi

    game = CatanGame(seed=seed)
    game.reset()
    step = 0
    while not game.is_terminal() and game.turn_number < 500 and step < 2000:
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            step += 1
            continue
        cp = game.current_player()
        mid = seat_assignments[cp]
        wpath = MODELS[mid][1]
        if wpath is not None:
            chosen = nn_argmax(game, le, wpath)
        elif mid == 2:
            chosen = ab_choose(game, le, 1)
        else:
            chosen = ab_choose(game, le, 2)
        game.step(chosen)
        step += 1

    w = game.winner()
    vps = [game._game.state.player_state[s][0] for s in range(4)]
    return seed, w, list(seat_assignments), vps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=200000)
    args = parser.parse_args()

    rng = np.random.RandomState(42)
    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi
        seats = rng.choice(4, size=4, replace=True).tolist()
        jobs.append((seed, seats))

    model_names = [MODELS[i][0] for i in range(4)]
    print(f"Running {args.games} games, 4 models, random seats, "
          f"{args.workers} workers...", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    dt = time.time() - t0
    print(f"Done in {dt:.1f}s ({args.games / dt:.0f} games/s)\n", flush=True)

    wins = np.zeros((4, 4), dtype=int)
    model_wins = [0] * 4
    model_games = [0] * 4

    for seed, winner, seats, vps in results:
        if winner is None:
            continue
        wm = seats[winner]
        for s in range(4):
            model_games[seats[s]] += 1
            if s == winner:
                model_wins[seats[s]] += 1
            else:
                wins[wm][seats[s]] += 1

    print("Raw win rates:")
    for i in range(4):
        mg = max(model_games[i], 1)
        print(f"  {model_names[i]:10s}: {model_wins[i]:4d} wins / "
              f"{model_games[i]:4d} seats ({100 * model_wins[i] / mg:.1f}%)")

    import choix
    comparisons = []
    for i in range(4):
        for j in range(4):
            if i != j:
                for _ in range(wins[i][j]):
                    comparisons.append((i, j))

    params = choix.ilsr_pairwise(4, comparisons, alpha=0.01)
    elo = 400 * params / np.log(10)
    elo -= elo.min()

    order = np.argsort(-elo)
    print(f"\n{'=' * 36}")
    print(f"{'Elo Ratings (Bradley-Terry)':^36s}")
    print(f"{'=' * 36}")
    print(f"{'Model':>12s}  {'Elo':>6s}  {'Win%':>6s}")
    print(f"{'-' * 36}")
    for i in order:
        wp = 100 * model_wins[i] / max(model_games[i], 1)
        print(f"{model_names[i]:>12s}  {elo[i]:6.0f}  {wp:5.1f}%")

    print(f"\nPairwise wins matrix:")
    header = "          " + "".join(f"{model_names[j]:>10s}" for j in range(4))
    print(header)
    for i in range(4):
        row = f"{model_names[i]:>10s}"
        for j in range(4):
            if i == j:
                row += f"{'--':>10s}"
            else:
                row += f"{wins[i][j]:>10d}"
        print(row)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
