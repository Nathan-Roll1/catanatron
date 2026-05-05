"""eval_2v2_nn: 2v2 NN-vs-NN evaluation, 0-ply argmax both sides.

Seats rotate every game so positional bias washes out:
    Game i: model A at {i%4, (i+2)%4}, model B at {(i+1)%4, (i+3)%4}
    Each model plays each seat-pair exactly num_games/4 times.

Usage:
    python -m human_bot.eval_2v2_nn \
        --a-weights csrc/nn_weights_m3.bin \
        --b-weights csrc/nn_weights_m2.bin \
        --games 100 --workers 8
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
    raise FileNotFoundError(f"No libnn found, tried: {candidates}")


def _play_one(args):
    (game_idx, seed, a_seats, b_seats, a_weights, b_weights) = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    libnn_path = _resolve_libnn_path()
    nn_lib = ctypes.CDLL(libnn_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP,
                                  ctypes.c_void_p]

    a_buf = (ctypes.c_char * (16 * 1024 * 1024))()
    a_ptr = ctypes.cast(a_buf, ctypes.c_void_p)
    assert nn_lib.nn_load(a_ptr, a_weights.encode()) == 0

    b_buf = (ctypes.c_char * (16 * 1024 * 1024))()
    b_ptr = ctypes.cast(b_buf, ctypes.c_void_p)
    assert nn_lib.nn_load(b_ptr, b_weights.encode()) == 0

    ae = ActionEncoder()
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

    while not game.is_terminal() and game.turn_number < 500:
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        if cp in a_seats:
            game.step(nn_argmax(game, le, a_ptr))
        else:
            game.step(nn_argmax(game, le, b_ptr))

    winner = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    a_won = winner in a_seats if winner is not None else False
    return {
        "game_idx": game_idx,
        "seed": seed,
        "a_seats": list(a_seats),
        "b_seats": list(b_seats),
        "winner": winner,
        "a_won": a_won,
        "vps": vps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-weights", type=str, required=True,
                   help="Path to side-A .bin (the 'new' model).")
    p.add_argument("--b-weights", type=str, required=True,
                   help="Path to side-B .bin (the 'old' model).")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=2000000)
    args = p.parse_args()

    a_weights = os.path.abspath(args.a_weights)
    b_weights = os.path.abspath(args.b_weights)

    # Build seat rotations: 4 distinct {a,b} pair patterns
    # Seats {0,2}vs{1,3}, {1,3}vs{0,2}, {2,0}vs{3,1}, {3,1}vs{2,0}
    # Rotate so each model plays each seat the same # of times.
    rotations = [
        ({0, 2}, {1, 3}),  # A=P0+P2, B=P1+P3
        ({1, 3}, {0, 2}),  # A=P1+P3, B=P0+P2
        ({0, 1}, {2, 3}),  # A=P0+P1, B=P2+P3
        ({2, 3}, {0, 1}),  # A=P2+P3, B=P0+P1
    ]
    jobs = []
    for gi in range(args.games):
        a_seats, b_seats = rotations[gi % len(rotations)]
        seed = args.seed_base + gi
        jobs.append((gi, seed, a_seats, b_seats, a_weights, b_weights))

    print(f"=== eval_2v2_nn ===")
    print(f"  A: {a_weights}")
    print(f"  B: {b_weights}")
    print(f"  games:    {args.games}  workers: {args.workers}")
    print(f"  baseline: 50% per side (random)")
    print(flush=True)

    ctx = mp.get_context("spawn")
    t0 = time.time()
    results = []
    with ctx.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_play_one, jobs):
            results.append(r)
    dt = time.time() - t0

    a_wins = sum(1 for r in results if r["a_won"])
    b_wins = sum(1 for r in results if (not r["a_won"]) and r["winner"] is not None)
    no_winner = sum(1 for r in results if r["winner"] is None)
    decisive = a_wins + b_wins

    a_seat_wins = {0: 0, 1: 0, 2: 0, 3: 0}
    b_seat_wins = {0: 0, 1: 0, 2: 0, 3: 0}
    seat_appearances_a = {0: 0, 1: 0, 2: 0, 3: 0}
    seat_appearances_b = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        for s in r["a_seats"]:
            seat_appearances_a[s] += 1
        for s in r["b_seats"]:
            seat_appearances_b[s] += 1
        if r["winner"] is not None:
            if r["winner"] in r["a_seats"]:
                a_seat_wins[r["winner"]] += 1
            else:
                b_seat_wins[r["winner"]] += 1

    print(f"=== RESULTS ===")
    print(f"  A wins: {a_wins}/{args.games}  ({100 * a_wins / args.games:.1f}%)")
    print(f"  B wins: {b_wins}/{args.games}  ({100 * b_wins / args.games:.1f}%)")
    if no_winner:
        print(f"  No winner: {no_winner}")
    if decisive:
        print(f"  A WR (decisive only): {a_wins}/{decisive} = "
              f"{100 * a_wins / decisive:.1f}%")
    print()
    print(f"  Per-seat wins (A occupied each seat ~{args.games//2} times):")
    print(f"    seat   A wins / appearances    B wins / appearances")
    for s in range(4):
        ap_a = seat_appearances_a[s]
        ap_b = seat_appearances_b[s]
        wa = a_seat_wins[s]
        wb = b_seat_wins[s]
        wr_a = 100 * wa / max(ap_a, 1)
        wr_b = 100 * wb / max(ap_b, 1)
        print(f"    P{s}    {wa:>3d} / {ap_a:>3d} ({wr_a:>5.1f}%)    "
              f"{wb:>3d} / {ap_b:>3d} ({wr_b:>5.1f}%)")
    print()
    # Bradley-Terry-ish significance via binomial
    if decisive > 0:
        from math import sqrt
        p_hat = a_wins / decisive
        se = sqrt(p_hat * (1 - p_hat) / decisive)
        ci_lo = max(0, p_hat - 1.96 * se)
        ci_hi = min(1, p_hat + 1.96 * se)
        print(f"  A 95% CI: [{100*ci_lo:.1f}%, {100*ci_hi:.1f}%]")
        if ci_lo > 0.5:
            print(f"  -> A is statistically stronger (CI excludes 50%)")
        elif ci_hi < 0.5:
            print(f"  -> B is statistically stronger (CI excludes 50%)")
        else:
            print(f"  -> No significant difference (CI includes 50%)")
    print(f"  elapsed: {dt:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
