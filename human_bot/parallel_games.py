#!/usr/bin/env python3
"""Parallel game runner. Supports different models + depths per agent.

Examples:
  # Same model, different depths:
  python -u human_bot/parallel_games.py --games 100 --workers 16 \
      --model-a cl4k --depth-a 10 --model-b cl4k --depth-b 20

  # Different models, same depth:
  python -u human_bot/parallel_games.py --games 100 --workers 16 \
      --model-a cl4k --model-b cl6k --depth-a 10 --depth-b 10

  # Short form (model names map to csrc/nn_weights_{name}.bin):
  python -u human_bot/parallel_games.py -n 100 -w 16 -A cl4k -B cl6k
"""

import argparse
import ctypes
import multiprocessing as mp
import os
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


def _make_nn(weights_path):
    """Load C NN library + weights, return (nn_lib, mptr, buffers)."""
    FP = ctypes.POINTER(ctypes.c_float)
    lib_path = os.path.join(CSRC, "libnn.dylib")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(CSRC, "libnn.so")
    nn_lib = ctypes.CDLL(lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    rc = nn_lib.nn_load(mptr, weights_path.encode())
    assert rc == 0, f"Failed to load {weights_path}"
    return nn_lib, mptr, mbuf


def play_one_game(args):
    (seed, s_a_list, s_b_list,
     depth_a, depth_b, weights_a, weights_b) = args

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

    same_model = (weights_a == weights_b)
    nn_a, mptr_a, _ba = _make_nn(weights_a)
    if same_model:
        nn_b, mptr_b, _bb = nn_a, mptr_a, _ba
    else:
        nn_b, mptr_b, _bb = _make_nn(weights_b)

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)

    def _enc(game):
        se.encode_into(game.get_state_view(), nf, ef, ff)

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

    s_a = set(s_a_list); s_b = set(s_b_list)
    game = CatanGame(seed=seed); game.reset()
    t0 = time.perf_counter()
    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1: game.step(0); continue
        cp = game.current_player()
        if cp in s_a:
            game.step(nnt_search(game, le, depth_a, nn_a, mptr_a))
        else:
            game.step(nnt_search(game, le, depth_b, nn_b, mptr_b))
    el = time.perf_counter() - t0
    w = game.winner()
    tag = "A" if (w is not None and w in s_a) else ("B" if w is not None else "draw")
    return (seed, w, tag, game.turn_number, el)


def main():
    parser = argparse.ArgumentParser(description="Parallel Catan game runner")
    parser.add_argument("-n", "--games", type=int, default=50)
    parser.add_argument("-w", "--workers", type=int, default=16)
    parser.add_argument("-A", "--model-a", type=str, default="cl4k")
    parser.add_argument("-B", "--model-b", type=str, default="cl4k")
    parser.add_argument("--depth-a", type=int, default=10)
    parser.add_argument("--depth-b", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=85000)
    args = parser.parse_args()

    weights_a = _resolve_weights(args.model_a)
    weights_b = _resolve_weights(args.model_b)
    la = f"{args.model_a}@t{args.depth_a}"
    lb = f"{args.model_b}@t{args.depth_b}"

    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi
        s_a = [gi % 4, (gi + 2) % 4]
        s_b = [(gi + 1) % 4, (gi + 3) % 4]
        jobs.append((seed, s_a, s_b, args.depth_a, args.depth_b, weights_a, weights_b))

    print(f"Running {args.games} games: {la} vs {lb} on {args.workers} workers...", flush=True)
    t0 = time.perf_counter()

    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one_game, jobs)

    wall = time.perf_counter() - t0

    wa = wb = 0
    total_time = 0.0
    for seed, w, tag, turns, el in results:
        total_time += el
        if tag == "A": wa += 1
        elif tag == "B": wb += 1

    print(f"\n{'='*55}")
    print(f"  {la} vs {lb} — {args.games} games, {args.workers} workers")
    print(f"{'='*55}")
    print(f"  {la:>16s}: {wa} wins ({100*wa/args.games:.0f}%)")
    print(f"  {lb:>16s}: {wb} wins ({100*wb/args.games:.0f}%)")
    print(f"  Wall time:       {wall:.1f}s ({args.games/wall*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
