#!/usr/bin/env python3
"""catan_player 602k model at 0-ply vs AB2 2-ply."""
import ctypes, multiprocessing as mp, numpy as np, os, sys, time

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
CP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "catan_player")
CP_LIB = os.path.join(CP_DIR, "libnn_cp.dylib")
CP_WEIGHTS = os.path.join(CP_DIR, "weights", "model.bin")


def play_one(args):
    seed, nn_seats_list, ab_seats_list = args
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder(); lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    nn_lib = ctypes.CDLL(CP_LIB)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [
        ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, CP_WEIGHTS.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP)
    ch = CGame(); ca = (Action * MAX_ACTIONS)(); cn = ctypes.c_int(0)
    ch2 = CGame(); ca2 = (Action * MAX_ACTIONS)(); cn2 = ctypes.c_int(0)

    def nn_argmax(game, le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4 + AD]; lo[mn[:AD] < 0.5] = -1e9
        return next((i for i, a in enumerate(le)
                     if ae.encode(a) == int(np.argmax(lo))), 0)

    def ab2_choose(game, le):
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if (cn.value > 0
                    and lib.game_winning_color(ctypes.byref(ch)) < 0):
                if cn.value > 1:
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    brj, brv = 0, -1e30
                    for j in range(cn.value):
                        lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        lib.game_execute(ctypes.byref(ch2), ca[j], ca2,
                                         ctypes.byref(cn2))
                        rv = lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > brv: brv = rv; brj = j
                    lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    lib.game_execute(ctypes.byref(ch2), ca[brj], ca2,
                                     ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    lib.game_execute(ctypes.byref(ch), ca[0], ca,
                                     ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv: bv = v; bi = i
        return bi

    nn_seats = set(nn_seats_list)
    game = CatanGame(seed=seed); game.reset()
    step = 0
    while not game.is_terminal() and game.turn_number < 500 and step < 2000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); step += 1; continue
        cp = game.current_player()
        if cp in nn_seats:
            game.step(nn_argmax(game, le))
        else:
            game.step(ab2_choose(game, le))
        step += 1
    w = game.winner()
    tag = ("NN" if w is not None and w in nn_seats
           else ("AB2" if w is not None else "draw"))
    return seed, w, tag, game.turn_number


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=960000)
    args = parser.parse_args()

    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi
        nn_s = [gi % 4, (gi + 2) % 4]
        ab_s = [(gi + 1) % 4, (gi + 3) % 4]
        jobs.append((seed, nn_s, ab_s))

    print(f"Running {args.games} games: CP 0-ply vs AB2 2-ply, "
          f"{args.workers} workers...", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    dt = time.time() - t0

    nn_w = sum(1 for _, _, t, _ in results if t == "NN")
    ab_w = sum(1 for _, _, t, _ in results if t == "AB2")
    print(f"\n{'=' * 50}")
    print(f"  CP 0-ply vs AB2 2-ply — {args.games} games")
    print(f"{'=' * 50}")
    print(f"  {'CP 0-ply':>12s}: {nn_w} wins ({100*nn_w/args.games:.0f}%)")
    print(f"  {'AB2 2-ply':>12s}: {ab_w} wins ({100*ab_w/args.games:.0f}%)")
    print(f"  Wall time: {dt:.1f}s ({args.games/dt*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
