#!/usr/bin/env python3
"""M2 d60 k2 f1 ab_value vs catan_player default (ABt30 k5 f1 ab_value, 602k model)."""
import ctypes, multiprocessing as mp, numpy as np, os, sys, time

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
CP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "catan_player")

M2_WEIGHTS = os.path.join(CSRC, "nn_weights_m2.bin")
CP_WEIGHTS = os.path.join(CP_DIR, "weights", "model.bin")
M2_LIB = os.path.join(CSRC, "libnn.dylib")
CP_LIB = os.path.join(CP_DIR, "libnn_cp.dylib")


def play_one(args):
    seed, m2_seats_list, cp_seats_list = args
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder(); lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    # Load M2 (1M model)
    m2_nn = ctypes.CDLL(M2_LIB)
    m2_nn.nn_load.restype = ctypes.c_int
    m2_nn.nn_forward.restype = None
    m2_nn.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    m2_buf = (ctypes.c_char * (8 * 1024 * 1024))()
    m2_ptr = ctypes.cast(m2_buf, ctypes.c_void_p)
    assert m2_nn.nn_load(m2_ptr, M2_WEIGHTS.encode()) == 0

    # Load catan_player (602k model)
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
    ch = CGame(); ca = (Action * MAX_ACTIONS)(); cn = ctypes.c_int(0)

    def _forward(nn_lib, ptr, game, le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(ptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        return out

    def nn_argmax(nn_lib, ptr, game, le):
        out = _forward(nn_lib, ptr, game, le)
        lo = out[4:4 + AD]
        mn = ae.get_action_mask(le).numpy()
        lo[mn[:AD] < 0.5] = -1e9
        return next((i for i, a in enumerate(le)
                     if ae.encode(a) == int(np.argmax(lo))), 0)

    def c_topk(nn_lib, ptr, game, le, k):
        out = _forward(nn_lib, ptr, game, le)
        lo = out[4:4 + AD]
        a2i = {}
        for i, a in enumerate(le):
            try: a2i[ae.encode(a)] = i
            except ValueError: pass
        return [li for _, li in sorted(
            [(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]

    def ab_leaf(game, seat):
        cg = game._game; bc = cg.state.colors[seat]
        return float(lib.base_value_fn(ctypes.byref(cg), bc))

    def flat_eval(nn_lib, ptr, game, root_seat, plies, k):
        if plies == 0 or game.is_terminal():
            if game.is_terminal():
                w = game.winner()
                return (10.0 if w is not None and w == root_seat
                        else (-10.0 if w is not None else 0.0))
            return ab_leaf(game, root_seat)
        le2 = game.get_legal_actions()
        if not le2: return 0.0
        if len(le2) == 1:
            gc2 = game.clone(); gc2.step(0)
            return flat_eval(nn_lib, ptr, gc2, root_seat, plies - 1, k)
        cp = game.current_player()
        cands2 = (c_topk(nn_lib, ptr, game, le2, k)
                  if len(le2) > k else list(range(len(le2))))
        best_ci, best_own = cands2[0], -1e30
        for ci2 in cands2:
            gc2 = game.clone(); gc2.step(ci2)
            own_v = ab_leaf(gc2, cp)
            if own_v > best_own: best_own = own_v; best_ci = ci2
        gc2 = game.clone(); gc2.step(best_ci)
        return flat_eval(nn_lib, ptr, gc2, root_seat, plies - 1, k)

    def search(nn_lib, ptr, game, le, depth, top_k, flat_k):
        seat = game.current_player()
        cands = (c_topk(nn_lib, ptr, game, le, top_k)
                 if len(le) > top_k else list(range(len(le))))
        bp, bv = 0, -1e30
        for p, ci in enumerate(cands):
            gc = game.clone(); gc.step(ci)
            v = flat_eval(nn_lib, ptr, gc, seat, depth - 1, flat_k)
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(cands[bp], le)

    m2_seats = set(m2_seats_list)
    cp_seats = set(cp_seats_list)
    game = CatanGame(seed=seed); game.reset()
    step = 0
    while not game.is_terminal() and game.turn_number < 500 and step < 2000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); step += 1; continue
        cp_cur = game.current_player()
        if cp_cur in m2_seats:
            chosen = search(m2_nn, m2_ptr, game, le, 60, 2, 1)
        else:
            chosen = nn_argmax(cp_nn, cp_ptr, game, le)
        game.step(chosen); step += 1

    w = game.winner()
    tag = ("M2" if w is not None and w in m2_seats
           else ("CP" if w is not None else "draw"))
    return seed, w, tag, game.turn_number


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=950000)
    args = parser.parse_args()

    rng = np.random.RandomState(77)
    import itertools
    seat_pairs = list(itertools.combinations(range(4), 2))
    jobs = []
    for gi in range(args.games):
        m2s = list(seat_pairs[rng.randint(len(seat_pairs))])
        cps = [s for s in range(4) if s not in m2s]
        jobs.append((args.seed_base + gi, m2s, cps))

    print(f"Running {args.games} games: M2-d60k2f1 vs CP-ABt30k5f1, "
          f"random seats, {args.workers} workers...", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    dt = time.time() - t0

    m2w = sum(1 for _, _, t, _ in results if t == "M2")
    cpw = sum(1 for _, _, t, _ in results if t == "CP")
    print(f"\n{'=' * 50}")
    print(f"  M2 d60k2f1 vs catan_player ABt30k5f1 — {args.games} games")
    print(f"{'=' * 50}")
    print(f"  {'M2 d60k2f1':>20s}: {m2w} wins ({100*m2w/args.games:.0f}%)")
    print(f"  {'CP ABt30k5f1':>20s}: {cpw} wins ({100*cpw/args.games:.0f}%)")
    print(f"  Wall time: {dt:.1f}s ({args.games/dt*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
