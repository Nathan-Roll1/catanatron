#!/usr/bin/env python3
"""NN 0-ply vs AB2: parallel matchup.

Usage:
    python -u human_bot/nn_vs_ab2.py --model cl4k --games 100 --workers 16
    python -u human_bot/nn_vs_ab2.py --model cl1k --games 100 --weights csrc/nn_weights_exit_cluster.bin
"""

import argparse
import ctypes
import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")


def _load_heuristic(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"HPOL":
            raise ValueError(f"{path} is not an HPOL heuristic file")
        ver, fd, ad = struct.unpack("<III", f.read(12))
        if ver != 1 or fd != 115 or ad != 337:
            raise ValueError(f"bad HPOL header: ver={ver} fd={fd} ad={ad}")
        bias = np.frombuffer(f.read(ad * 4), dtype="<f4").copy()
        weight = np.frombuffer(f.read(ad * fd * 4), dtype="<f4").reshape(ad, fd).copy()
    return bias, weight


def play_one(args):
    (seed, nn_seats_list, ab2_seats_list, weights_path, depth, ab_depth,
     use_ab_value, top_k, flat_k_arg, opp_weights_path, opp_depth,
     heuristic_path) = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    lib_path = os.path.join(CSRC, "..", "catan_player", "libcatan_nn.dylib")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(CSRC, "..", "catan_player", "libcatan_nn.so")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(CSRC, "libnn.dylib")

    h_bias = h_weight = None
    if heuristic_path:
        h_bias, h_weight = _load_heuristic(heuristic_path)

    nn_lib = ctypes.CDLL(lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    if not heuristic_path or depth > 0:
        assert nn_lib.nn_load(mptr, weights_path.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    def nn_argmax(game, le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        ai = int(np.argmax(lo))
        return next((i for i, a in enumerate(le) if ae.encode(a) == ai), 0)

    def heuristic_argmax(game, le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        scores = h_bias + h_weight @ ff
        scores[mn[:AD] < 0.5] = -1e9
        ai = int(np.argmax(scores))
        return next((i for i, a in enumerate(le) if ae.encode(a) == ai), 0)

    def c_topk(game, le, k):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; a2i = {ae.encode(a): i for i, a in enumerate(le)}
        return [li for _, li in sorted([(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]

    def c_argmax(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        se.encode_into(gc.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        gc.step(next((i for i, a in enumerate(le) if ae.encode(a) == int(np.argmax(lo))), 0))

    def c_val(game):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(game.get_legal_actions()).numpy()
        mk[:] = 0; mk[:len(mn)] = mn
        nn_lib.nn_value_only(mptr, nfp, efp, ffp, mkp, vlp)
        return vl.copy()

    def ab_leaf_val(game, seat):
        """Evaluate leaf with hand-crafted base_value_fn for the given seat."""
        cg = game._game
        bc = cg.state.colors[seat]
        return float(lib.base_value_fn(ctypes.byref(cg), bc))

    def _flat_eval(game, root_seat, plies_left, k):
        """Recursive top-k search: each player picks their own best among top-k."""
        if plies_left == 0 or game.is_terminal():
            if game.is_terminal():
                w = game.winner()
                return 10.0 if (w is not None and w == root_seat) else (-10.0 if w is not None else 0.0)
            if use_ab_value:
                return ab_leaf_val(game, root_seat)
            vs = c_val(game); off = (root_seat - game.current_player()) % 4
            return float(vs[off])
        le2 = game.get_legal_actions()
        if not le2:
            return 0.0
        if len(le2) == 1:
            gc2 = game.clone(); gc2.step(0)
            return _flat_eval(gc2, root_seat, plies_left - 1, k)
        cp = game.current_player()
        cands2 = list(range(len(le2)))
        if len(le2) > k:
            cands2 = c_topk(game, le2, k)
        best_ci, best_own = cands2[0], -1e30
        for ci2 in cands2:
            gc2 = game.clone(); gc2.step(ci2)
            if use_ab_value:
                own_v = ab_leaf_val(gc2, cp)
            else:
                vs = c_val(gc2); off = (cp - gc2.current_player()) % 4
                own_v = float(vs[off])
            if own_v > best_own:
                best_own = own_v
                best_ci = ci2
        gc2 = game.clone(); gc2.step(best_ci)
        return _flat_eval(gc2, root_seat, plies_left - 1, k)

    flat_k = flat_k_arg

    def nnt_search(game, le, d, top_k=top_k):
        seat = game.current_player()
        cands = list(range(len(le)))
        if len(le) > top_k and d >= 2:
            cands = c_topk(game, le, top_k)
        if flat_k > 0:
            bp, bv = 0, -1e30
            for p, ci in enumerate(cands):
                gc = game.clone(); gc.step(ci)
                v = _flat_eval(gc, seat, d - 1, flat_k)
                v = apply_action_bonus(v, le[ci])
                if v > bv: bv = v; bp = p
            return fix_robber_steal(cands[bp], le)
        bp, bv = 0, -1e30
        for p, ci in enumerate(cands):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, d + 1):
                if gc.is_terminal(): break
                c_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
            elif use_ab_value:
                v = ab_leaf_val(gc, seat)
            else:
                vs = c_val(gc); off = (seat - gc.current_player()) % 4; v = float(vs[off])
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(cands[bp], le)

    ch2 = CGame()
    ca2 = (Action * MAX_ACTIONS)()
    cn2 = ctypes.c_int(0)

    def ab2_choose(game, le, ab_depth=1):
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if ab_depth >= 2 and cn.value > 0:
                if cn.value > 1:
                    best_resp, best_rv = 0, -1e30
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    for j in range(cn.value):
                        lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        lib.game_execute(ctypes.byref(ch2), ca[j], ca2, ctypes.byref(cn2))
                        rv = lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > best_rv: best_rv = rv; best_resp = j
                    lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    lib.game_execute(ctypes.byref(ch2), ca[best_resp], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    lib.game_execute(ctypes.byref(ch), ca[0], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv: bv = v; bi = i
        return bi

    # Optional second NN model for opponent seats
    opp_mptr = None
    if opp_weights_path:
        opp_mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
        opp_mptr = ctypes.cast(opp_mbuf, ctypes.c_void_p)
        assert nn_lib.nn_load(opp_mptr, opp_weights_path.encode()) == 0

    def opp_nn_argmax(game, le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(opp_mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        return next((i for i, a in enumerate(le) if ae.encode(a) == int(np.argmax(lo))), 0)

    def opp_c_topk(game, le, k):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(opp_mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; a2i = {ae.encode(a): i for i, a in enumerate(le)}
        return [li for _, li in sorted([(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]

    def opp_c_argmax(gc):
        le2 = gc.get_legal_actions()
        if not le2: return
        if len(le2) == 1: gc.step(0); return
        se.encode_into(gc.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le2).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(opp_mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        gc.step(next((i for i, a in enumerate(le2) if ae.encode(a) == int(np.argmax(lo))), 0))

    def opp_c_val(game):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(game.get_legal_actions()).numpy(); mk[:] = 0; mk[:len(mn)] = mn
        nn_lib.nn_value_only(opp_mptr, nfp, efp, ffp, mkp, vlp)
        return vl.copy()

    def opp_ab_leaf_val(game, seat):
        cg = game._game; bc = cg.state.colors[seat]
        return float(lib.base_value_fn(ctypes.byref(cg), bc))

    def _opp_flat_eval(game, root_seat, plies_left, k):
        if plies_left == 0 or game.is_terminal():
            if game.is_terminal():
                w = game.winner()
                return 10.0 if (w is not None and w == root_seat) else (-10.0 if w is not None else 0.0)
            if use_ab_value:
                return opp_ab_leaf_val(game, root_seat)
            vs = opp_c_val(game); off = (root_seat - game.current_player()) % 4
            return float(vs[off])
        le2 = game.get_legal_actions()
        if not le2: return 0.0
        if len(le2) == 1:
            gc2 = game.clone(); gc2.step(0)
            return _opp_flat_eval(gc2, root_seat, plies_left - 1, k)
        cp = game.current_player()
        cands2 = list(range(len(le2)))
        if len(le2) > k:
            cands2 = opp_c_topk(game, le2, k)
        best_ci, best_own = cands2[0], -1e30
        for ci2 in cands2:
            gc2 = game.clone(); gc2.step(ci2)
            own_v = opp_ab_leaf_val(gc2, cp)
            if own_v > best_own: best_own = own_v; best_ci = ci2
        gc2 = game.clone(); gc2.step(best_ci)
        return _opp_flat_eval(gc2, root_seat, plies_left - 1, k)

    def opp_nnt_search(game, le, d):
        seat = game.current_player()
        cands = list(range(len(le)))
        if len(le) > 5 and d >= 2:
            cands = opp_c_topk(game, le, 5)
        if flat_k_arg > 0:
            bp, bv = 0, -1e30
            for p, ci in enumerate(cands):
                gc = game.clone(); gc.step(ci)
                v = _opp_flat_eval(gc, seat, d - 1, flat_k_arg)
                v = apply_action_bonus(v, le[ci])
                if v > bv: bv = v; bp = p
            return fix_robber_steal(cands[bp], le)
        bp, bv = 0, -1e30
        for p, ci in enumerate(cands):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, d + 1):
                if gc.is_terminal(): break
                opp_c_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
            elif use_ab_value:
                v = opp_ab_leaf_val(gc, seat)
            else:
                vs = opp_c_val(gc); off = (seat - gc.current_player()) % 4; v = float(vs[off])
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(cands[bp], le)

    nn_seats = set(nn_seats_list); ab2_seats = set(ab2_seats_list)
    game = CatanGame(seed=seed); game.reset()
    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); continue
        cp = game.current_player()
        if cp in nn_seats:
            if heuristic_path and depth == 0:
                game.step(heuristic_argmax(game, le))
            elif depth == 0:
                game.step(nn_argmax(game, le))
            else:
                game.step(nnt_search(game, le, depth))
        elif opp_mptr:
            if opp_depth == 0:
                game.step(opp_nn_argmax(game, le))
            else:
                game.step(opp_nnt_search(game, le, opp_depth))
        else:
            game.step(ab2_choose(game, le, ab_depth))
    w = game.winner()
    tag = "NN" if (w is not None and w in nn_seats) else ("AB2" if w is not None else "draw")
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    nn_rank_sum = 0
    for s in nn_seats:
        nn_rank_sum += sorted(range(4), key=lambda p: (-vps[p], p)).index(s) + 1
    nn_avg_vp = sum(vps[s] for s in nn_seats) / max(len(nn_seats), 1)
    ab2_avg_vp = sum(vps[s] for s in ab2_seats) / max(len(ab2_seats), 1)
    return (seed, w, tag, game.turn_number, vps, nn_avg_vp, ab2_avg_vp,
            nn_rank_sum, len(nn_seats))


def resolve_weights(model):
    if os.path.exists(model):
        return os.path.abspath(model)
    p = os.path.join(CSRC, f"nn_weights_{model}.bin")
    if os.path.exists(p):
        return p
    p2 = os.path.join(CSRC, "..", "catan_player", "weights", f"{model}.bin")
    if os.path.exists(p2):
        return os.path.abspath(p2)
    raise FileNotFoundError(f"No weights for '{model}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="abv3")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--depth", type=int, default=0,
                        help="Search depth (0 = policy argmax, 10 = NNt10)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=120000)
    parser.add_argument("--mode", choices=["1v3", "2v2"], default="2v2")
    parser.add_argument("--ab-depth", type=int, default=1,
                        help="AB2 opponent search depth (1 = greedy, 2 = 2-ply)")
    parser.add_argument("--ab-value", action="store_true",
                        help="Use AB2 base_value_fn for leaf eval instead of NN value head")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-K policy pruning at root")
    parser.add_argument("--flat-k", type=int, default=0,
                        help="Flat search width per ply (0 = tapered/argmax rollout)")
    parser.add_argument("--opp-weights", type=str, default=None,
                        help="Opponent NN weights (if set, opponent uses NN instead of AB2)")
    parser.add_argument("--opp-depth", type=int, default=10,
                        help="Opponent NN search depth")
    parser.add_argument("--heuristic", type=str, default=None,
                        help="HPOL heuristic policy file. If set with --depth 0, uses it instead of NN.")
    args = parser.parse_args()

    wpath = args.weights if args.weights else resolve_weights(args.model)
    model_name = os.path.basename(wpath).replace("nn_weights_", "").replace(".bin", "") if args.weights else args.model
    if args.heuristic:
        model_name = os.path.basename(args.heuristic).replace("policy_heuristic_", "").replace(".bin", "")
    opp_wpath = args.opp_weights
    if opp_wpath and not os.path.exists(opp_wpath):
        opp_wpath = resolve_weights(opp_wpath)
    vfn = "ABt" if args.ab_value else "NNt"
    depth_label = f"{vfn}{args.depth}" if args.depth > 0 else "0-ply"
    if opp_wpath:
        ab_label = f"NNt{args.opp_depth}({os.path.basename(opp_wpath)})"
    else:
        ab_label = f"AB{args.ab_depth}" if args.ab_depth > 1 else "AB2"

    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi
        if args.mode == "1v3":
            nn_s = [gi % 4]
            ab2_s = [s for s in range(4) if s != gi % 4]
        else:
            nn_s = [gi % 4, (gi + 2) % 4]
            ab2_s = [(gi + 1) % 4, (gi + 3) % 4]
        jobs.append((seed, nn_s, ab2_s, wpath, args.depth, args.ab_depth,
                     args.ab_value, args.top_k, args.flat_k, opp_wpath,
                     args.opp_depth, args.heuristic))

    print(f"Running {args.games} games: {model_name} {depth_label} vs {ab_label} "
          f"({args.mode}, {args.workers} workers)...",
          flush=True)
    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    wall = time.perf_counter() - t0

    nn_w = sum(1 for r in results if r[2] == "NN")
    ab2_w = sum(1 for r in results if r[2] == "AB2")
    draws = args.games - nn_w - ab2_w
    nn_vp = sum(r[5] for r in results) / max(args.games, 1)
    ab2_vp = sum(r[6] for r in results) / max(args.games, 1)
    nn_rank = sum(r[7] for r in results) / max(sum(r[8] for r in results), 1)
    avg_turns = sum(r[3] for r in results) / max(args.games, 1)

    print(f"\n{'='*50}")
    print(f"  {model_name} {depth_label} vs {ab_label} — {args.games} games ({args.mode})")
    print(f"{'='*50}")
    print(f"  {model_name:>12s} {depth_label}: {nn_w} wins ({100*nn_w/args.games:.0f}%)")
    print(f"  {ab_label:>14s}: {ab2_w} wins ({100*ab2_w/args.games:.0f}%)")
    print(f"  Draws: {draws}")
    print(f"  Avg VP: NN={nn_vp:.2f} AB2={ab2_vp:.2f}")
    print(f"  Avg NN rank: {nn_rank:.2f}/4")
    print(f"  Avg turns: {avg_turns:.1f}")
    print(f"  Wall time: {wall:.1f}s ({args.games/wall*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
