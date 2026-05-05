"""super_eval: 1v3 evaluation of NN-0ply current weights.

Two evaluations supported, both 1v3 (1 NN seat, 3 opponent seats), both
with seat rotation across games for unbiased measurement:

    eval_1v3(our_weights_bin, opp_kind="ab2", n_games=20, ab_depth=2,
             num_workers=8) -> {wr, avg_rank, ...}
        Our seat: 0-ply NN argmax with `our_weights_bin`.
        Opp seats: full alpha-beta search at depth `ab_depth`,
        with full chance-node expectimax (matches catanatron's
        AlphaBetaPlayer).

    eval_1v3(our_weights_bin, opp_kind="nn", opp_weights_bin=baseline_bin,
             n_games=20, num_workers=8) -> {wr, avg_rank, ...}
        Our seat: 0-ply NN argmax with `our_weights_bin`.
        Opp seats: 0-ply NN argmax with `opp_weights_bin` (e.g. frozen
        baseline at training start).

Both return: {n_games, wins, winrate, avg_rank, vps_avg, elapsed,
              wins_per_seat}.

Designed to be called from the learner after every N training rounds.
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time

import numpy as np


AD = 337
MASK_DIM = 397
FP = ctypes.POINTER(ctypes.c_float)


def _resolve_libnn_path():
    """Find libnn.{so,dylib} in csrc/ — same logic as c_selfplay._load_c_nn."""
    import platform
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hostname = platform.node().split(".")[0]
    candidates = [
        os.path.join(project_root, "csrc", f"libnn_{hostname}.so"),
        os.path.join(project_root, "csrc", "libnn.so"),
        os.path.join(project_root, "csrc", "libnn.dylib"),
        os.path.join(project_root, "catan_player", "libcatan_nn.so"),
        os.path.join(project_root, "catan_player", "libcatan_nn.dylib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No libnn found, tried: {candidates}\n"
        "Build with `python -m hexzero.bindings.build_lib && "
        "make -C csrc libnn` (or equivalent).")


def _play_one(args):
    """Play one 1v3 game. Worker for the eval pool."""
    (game_idx, seed, our_seat, our_weights, opp_kind,
     opp_weights, ab_depth) = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    lib = load_library()
    ae = ActionEncoder()

    # Load our libnn (NN model lives in libnn's address space)
    libnn_path = _resolve_libnn_path()
    nn_lib = ctypes.CDLL(libnn_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP,
                                  ctypes.c_void_p]

    our_buf = (ctypes.c_char * (16 * 1024 * 1024))()
    our_ptr = ctypes.cast(our_buf, ctypes.c_void_p)
    rc = nn_lib.nn_load(our_ptr, our_weights.encode())
    if rc != 0:
        raise RuntimeError(f"nn_load(our) failed: {rc}")

    opp_ptr = None
    opp_buf = None
    if opp_kind == "nn":
        if not opp_weights:
            raise ValueError("opp_kind='nn' requires opp_weights")
        opp_buf = (ctypes.c_char * (16 * 1024 * 1024))()
        opp_ptr = ctypes.cast(opp_buf, ctypes.c_void_p)
        rc = nn_lib.nn_load(opp_ptr, opp_weights.encode())
        if rc != 0:
            raise RuntimeError(f"nn_load(opp) failed: {rc}")

    # Game + encoder
    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)
    out = np.zeros(4 + MASK_DIM, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)
    outp = out.ctypes.data_as(ctypes.c_void_p)

    def nn_argmax(g, le, model_ptr):
        se.encode_into(g.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        nn_lib.nn_forward(model_ptr, nfp, efp, ffp, mkp, outp)
        logits = out[4:4 + AD].copy()
        logits[mn[:AD] < 0.5] = -1e9
        a_idx = int(np.argmax(logits))
        for i, a in enumerate(le):
            try:
                if ae.encode(a) == a_idx:
                    return i
            except ValueError:
                continue
        return 0

    # AB2 setup (catanatron-equivalent: alpha-beta with chance-node expectimax)
    ab_ctx = SearchCtx()
    ab_buf = (Action * MAX_ACTIONS)()
    ab_eval = ValueFn(lib.base_value_fn)

    def ab_choose(g, le):
        n = len(le)
        cg = g._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le):
            ab_buf[i] = a
        res = lib.alphabeta_search(
            ctypes.byref(ab_ctx), ctypes.byref(cg), ab_buf,
            ctypes.c_int(n), ctypes.c_int(ab_depth),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), ab_eval)
        cb = ctypes.string_at(ctypes.byref(res.action),
                              ctypes.sizeof(res.action))
        for i, a in enumerate(le):
            if ctypes.string_at(ctypes.byref(a),
                                ctypes.sizeof(a)) == cb:
                return i
        return 0

    t0 = time.time()
    while not game.is_terminal() and game.turn_number < 500:
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        if cp == our_seat:
            game.step(nn_argmax(game, le, our_ptr))
        else:
            if opp_kind == "ab2":
                game.step(ab_choose(game, le))
            else:  # opp_kind == "nn"
                game.step(nn_argmax(game, le, opp_ptr))
    dt = time.time() - t0

    winner = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    rank = sorted(range(4), key=lambda s: -vps[s]).index(our_seat) + 1

    return {
        "game_idx": game_idx,
        "seed": seed,
        "our_seat": our_seat,
        "winner": winner,
        "vps": vps,
        "rank": rank,
        "n_turns": game.turn_number,
        "elapsed": dt,
    }


def eval_1v3(our_weights_bin, opp_kind, opp_weights_bin=None,
             n_games=20, ab_depth=2, num_workers=8, seed_base=900000):
    """Run a 1v3 evaluation.

    Args:
        our_weights_bin: Path to .bin (the weights under test).
        opp_kind: "ab2" or "nn".
        opp_weights_bin: Required if opp_kind == "nn".
        n_games: Number of games (will be rounded up so each seat is
            occupied an equal number of times if n_games % 4 != 0;
            we use exactly n_games and rotate by gi%4).
        ab_depth: Depth for AB opponent (only used if opp_kind == "ab2").
        num_workers: Parallel game count.
        seed_base: Reproducible seed base.

    Returns:
        dict with keys: n_games, wins, winrate, avg_rank, vps_avg,
        wins_per_seat, elapsed, opp_kind, ab_depth (or opp_weights).
    """
    assert opp_kind in {"ab2", "nn"}
    if opp_kind == "nn" and not opp_weights_bin:
        raise ValueError("opp_kind='nn' requires opp_weights_bin")

    our_weights_bin = os.path.abspath(our_weights_bin)
    if opp_weights_bin:
        opp_weights_bin = os.path.abspath(opp_weights_bin)

    jobs = []
    for gi in range(n_games):
        seed = seed_base + gi
        our_seat = gi % 4
        jobs.append((gi, seed, our_seat, our_weights_bin, opp_kind,
                     opp_weights_bin, ab_depth))

    ctx = mp.get_context("spawn")
    t_start = time.time()
    results = []
    with ctx.Pool(processes=num_workers) as pool:
        for r in pool.imap_unordered(_play_one, jobs):
            results.append(r)
    elapsed = time.time() - t_start

    wins = sum(1 for r in results if r["winner"] == r["our_seat"])
    rank_sum = sum(r["rank"] for r in results)
    vps_sum = sum(r["vps"][r["our_seat"]] for r in results)
    wins_per_seat = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        if r["winner"] == r["our_seat"]:
            wins_per_seat[r["our_seat"]] += 1

    return {
        "n_games": n_games,
        "wins": wins,
        "winrate": wins / max(n_games, 1),
        "avg_rank": rank_sum / max(n_games, 1),
        "vps_avg": vps_sum / max(n_games, 1),
        "wins_per_seat": wins_per_seat,
        "elapsed": elapsed,
        "opp_kind": opp_kind,
        "ab_depth": ab_depth if opp_kind == "ab2" else None,
        "opp_weights": opp_weights_bin if opp_kind == "nn" else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--our-weights", type=str, required=True)
    p.add_argument("--opp", type=str, choices=["ab2", "nn"], default="ab2")
    p.add_argument("--opp-weights", type=str, default=None,
                   help="Required if --opp nn")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=900000)
    args = p.parse_args()

    print(f"=== eval_1v3 ===")
    print(f"  our_weights:  {args.our_weights}")
    print(f"  opp:          {args.opp}"
          + (f" (depth={args.ab_depth})" if args.opp == "ab2"
             else f" weights={args.opp_weights}"))
    print(f"  games:        {args.games}")
    print(f"  workers:      {args.workers}")
    print(f"  seed_base:    {args.seed_base}")
    print(flush=True)

    res = eval_1v3(
        our_weights_bin=args.our_weights,
        opp_kind=args.opp,
        opp_weights_bin=args.opp_weights,
        n_games=args.games,
        ab_depth=args.ab_depth,
        num_workers=args.workers,
        seed_base=args.seed_base,
    )
    print(f"=== RESULTS ===")
    print(f"  WR:        {res['wins']}/{res['n_games']} "
          f"({100 * res['winrate']:.1f}%)")
    print(f"  avg rank:  {res['avg_rank']:.2f} / 4")
    print(f"  avg VP:    {res['vps_avg']:.2f}")
    print(f"  per-seat wins: {res['wins_per_seat']}")
    print(f"  elapsed:   {res['elapsed']:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
