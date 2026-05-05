#!/usr/bin/env python3
"""M2 0-ply vs catan_player 0-ply, head-to-head."""
import ctypes, multiprocessing as mp, numpy as np, os, sys, time

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
CP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "catan_player")
M2_LIB = os.path.join(CSRC, "libnn.dylib")
M2_WEIGHTS = os.path.join(CSRC, "nn_weights_m2.bin")
CP_LIB = os.path.join(CP_DIR, "libnn_cp.dylib")
CP_WEIGHTS = os.path.join(CP_DIR, "weights", "model.bin")


def play_one(args):
    seed, m2_seats_list, cp_seats_list = args
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder(); load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    m2_nn = ctypes.CDLL(M2_LIB)
    m2_nn.nn_load.restype = ctypes.c_int
    m2_nn.nn_forward.restype = None
    m2_nn.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    m2_buf = (ctypes.c_char * (8 * 1024 * 1024))()
    m2_ptr = ctypes.cast(m2_buf, ctypes.c_void_p)
    assert m2_nn.nn_load(m2_ptr, M2_WEIGHTS.encode()) == 0

    cp_nn = ctypes.CDLL(CP_LIB)
    cp_nn.nn_load.restype = ctypes.c_int
    cp_nn.nn_forward.restype = None
    cp_nn.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    cp_buf = (ctypes.c_char * (8 * 1024 * 1024))()
    cp_ptr = ctypes.cast(cp_buf, ctypes.c_void_p)
    assert cp_nn.nn_load(cp_ptr, CP_WEIGHTS.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP)

    def nn_argmax(nn_lib, ptr, game, le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(ptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4 + AD]; lo[mn[:AD] < 0.5] = -1e9
        return next((i for i, a in enumerate(le)
                     if ae.encode(a) == int(np.argmax(lo))), 0)

    m2_seats = set(m2_seats_list)
    game = CatanGame(seed=seed); game.reset()
    step = 0
    while not game.is_terminal() and game.turn_number < 500 and step < 2000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); step += 1; continue
        cp = game.current_player()
        if cp in m2_seats:
            game.step(nn_argmax(m2_nn, m2_ptr, game, le))
        else:
            game.step(nn_argmax(cp_nn, cp_ptr, game, le))
        step += 1
    w = game.winner()
    tag = ("M2" if w is not None and w in m2_seats
           else ("CP" if w is not None else "draw"))
    return seed, w, tag, game.turn_number


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=970000)
    args = parser.parse_args()

    import itertools
    rng = np.random.RandomState(55)
    seat_pairs = list(itertools.combinations(range(4), 2))
    jobs = []
    for gi in range(args.games):
        m2s = list(seat_pairs[rng.randint(len(seat_pairs))])
        cps = [s for s in range(4) if s not in m2s]
        jobs.append((args.seed_base + gi, m2s, cps))

    print(f"Running {args.games} games: M2 0-ply vs CP 0-ply, "
          f"random seats, {args.workers} workers...", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    dt = time.time() - t0

    m2w = sum(1 for _, _, t, _ in results if t == "M2")
    cpw = sum(1 for _, _, t, _ in results if t == "CP")
    print(f"\n{'=' * 50}")
    print(f"  M2 0-ply vs CP 0-ply — {args.games} games")
    print(f"{'=' * 50}")
    print(f"  {'M2 0-ply':>12s}: {m2w} wins ({100*m2w/args.games:.0f}%)")
    print(f"  {'CP 0-ply':>12s}: {cpw} wins ({100*cpw/args.games:.0f}%)")
    print(f"  Wall time: {dt:.1f}s ({args.games/dt*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
