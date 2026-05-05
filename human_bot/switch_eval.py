#!/usr/bin/env python3
"""Eval: 0-ply early → d60 k2 f1 ab_value late, with configurable switch turn."""
import ctypes, multiprocessing as mp, numpy as np, os, sys, time

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
WPATH = os.path.join(CSRC, "nn_weights_m2.bin")


def play_switch(args):
    seed, nn_seats_list, ab_seats_list, switch_turn = args
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

    nn_lib = ctypes.CDLL(os.path.join(CSRC, "libnn.dylib"))
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, WPATH.encode()) == 0

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

    def c_topk(game, le, k):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
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

    def flat_eval(game, root_seat, plies, k):
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
            return flat_eval(gc2, root_seat, plies - 1, k)
        cp = game.current_player()
        cands2 = c_topk(game, le2, k) if len(le2) > k else list(range(len(le2)))
        best_ci, best_own = cands2[0], -1e30
        for ci2 in cands2:
            gc2 = game.clone(); gc2.step(ci2)
            own_v = ab_leaf(gc2, cp)
            if own_v > best_own: best_own = own_v; best_ci = ci2
        gc2 = game.clone(); gc2.step(best_ci)
        return flat_eval(gc2, root_seat, plies - 1, k)

    def search_d60(game, le):
        seat = game.current_player()
        cands = c_topk(game, le, 2) if len(le) > 2 else list(range(len(le)))
        bp, bv = 0, -1e30
        for p, ci in enumerate(cands):
            gc = game.clone(); gc.step(ci)
            v = flat_eval(gc, seat, 59, 1)
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(cands[bp], le)

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
            if game.turn_number < switch_turn:
                game.step(nn_argmax(game, le))
            else:
                game.step(search_d60(game, le))
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
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=300000)
    args = parser.parse_args()

    configs = [
        ("0ply-only", 9999),
        ("0ply->d60 @T20", 20),
        ("0ply->d60 @T30", 30),
        ("0ply->d60 @T40", 40),
        ("0ply->d60 @T50", 50),
        ("0ply->d60 @T60", 60),
        ("0ply->d60 @T70", 70),
        ("d60-always", 0),
    ]

    for label, sw in configs:
        jobs = [(args.seed_base + gi,
                 [gi % 4, (gi + 2) % 4],
                 [(gi + 1) % 4, (gi + 3) % 4],
                 sw) for gi in range(args.games)]
        t0 = time.time()
        with mp.Pool(args.workers) as pool:
            results = pool.map(play_switch, jobs)
        dt = time.time() - t0
        nn_w = sum(1 for _, _, t, _ in results if t == "NN")
        ab_w = sum(1 for _, _, t, _ in results if t == "AB2")
        tot = nn_w + ab_w
        wr = nn_w / max(tot, 1)
        print(f"{label:20s}: NN={nn_w:2d} AB2={ab_w:2d} "
              f"WR={wr:.0%}  ({dt:.1f}s)", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
