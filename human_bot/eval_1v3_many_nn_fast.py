"""Batch no-search 1v3 free-for-all evaluation for many NN binaries.

For each seed, every candidate is played once from each seat against three
copies of the opponent policy.  If candidate and opponent weights are identical,
the full-seat sweep gives an exact 25% candidate win rate per seed, which makes
this a cleaner FFA metric than random single-seat sampling.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np


AD = 337
MASK_DIM = 397
MODEL_BYTES = 16 * 1024 * 1024
FP = ctypes.POINTER(ctypes.c_float)

_G = {}


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
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No libnn found, tried: {candidates}")


def _load_model(lib, weights):
    buf = (ctypes.c_char * MODEL_BYTES)()
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    rc = lib.nn_load(ptr, weights.encode())
    if rc != 0:
        raise RuntimeError(f"nn_load failed for {weights}: {rc}")
    return buf, ptr


def _init_worker(a_weights, b_weights):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    lib = ctypes.CDLL(_resolve_libnn_path())
    lib.nn_load.restype = ctypes.c_int
    lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nn_forward.restype = None
    lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]

    a_models = [_load_model(lib, path) for path in a_weights]
    b_buf, b_ptr = _load_model(lib, b_weights)
    _G.update({
        "lib": lib,
        "a_bufs": [buf for buf, _ptr in a_models],
        "a_ptrs": [ptr for _buf, ptr in a_models],
        "b_buf": b_buf,
        "b_ptr": b_ptr,
        "ae": ActionEncoder(),
        "CatanGame": CatanGame,
    })


def _play_one(job):
    cand_idx, seed_idx, seed, candidate_seat = job
    CatanGame = _G["CatanGame"]
    ae = _G["ae"]
    lib = _G["lib"]

    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)
    out = np.zeros(4 + MASK_DIM, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)
    outp = out.ctypes.data_as(ctypes.c_void_p)
    cand_ptr = _G["a_ptrs"][cand_idx]
    opp_ptr = _G["b_ptr"]

    def nn_argmax(le, model_ptr):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        lib.nn_forward(model_ptr, nfp, efp, ffp, mkp, outp)
        logits = out[4:4 + AD].copy()
        logits[mn[:AD] < 0.5] = -1e9
        action_idx = int(np.argmax(logits))
        for i, action in enumerate(le):
            try:
                if ae.encode(action) == action_idx:
                    return i
            except ValueError:
                continue
        return 0

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        if len(legal) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        game.step(nn_argmax(legal, cand_ptr if cp == candidate_seat else opp_ptr))

    winner = game.winner()
    return {
        "candidate": cand_idx,
        "seed_idx": seed_idx,
        "candidate_seat": candidate_seat,
        "winner": winner,
        "candidate_won": winner == candidate_seat if winner is not None else False,
    }


def run(a_weights, b_weights, games, workers, seed_base):
    jobs = []
    for ci in range(len(a_weights)):
        for gi in range(games):
            seed = seed_base + gi
            for seat in range(4):
                jobs.append((ci, gi, seed, seat))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=([os.path.abspath(p) for p in a_weights], os.path.abspath(b_weights)),
    ) as pool:
        raw = list(pool.imap_unordered(_play_one, jobs))
    elapsed = time.time() - t0

    results = []
    total_games = 4 * games
    for ci, path in enumerate(a_weights):
        rows = [r for r in raw if r["candidate"] == ci]
        cand_wins = sum(1 for r in rows if r["candidate_won"])
        opp_wins = sum(1 for r in rows if (not r["candidate_won"]) and r["winner"] is not None)
        no_winner = sum(1 for r in rows if r["winner"] is None)
        seat_wins = [
            sum(1 for r in rows if r["candidate_seat"] == seat and r["candidate_won"])
            for seat in range(4)
        ]
        results.append({
            "candidate": ci,
            "a_weights": os.path.abspath(path),
            "seed_count": games,
            "games": total_games,
            "a_wins": cand_wins,
            "b_wins": opp_wins,
            "no_winner": no_winner,
            "a_winrate": cand_wins / max(1, total_games),
            "candidate_seat_wins": seat_wins,
        })
    return {
        "seed_count": games,
        "games_per_candidate": total_games,
        "candidates": len(a_weights),
        "elapsed_sec": elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-weight", action="append", required=True)
    parser.add_argument("--b-weights", required=True)
    parser.add_argument("--games", type=int, default=64,
                        help="number of seeds; each candidate plays all 4 seats per seed")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=2000000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    res = run(args.a_weight, args.b_weights, args.games, args.workers,
              args.seed_base)
    if args.json:
        print(json.dumps(res, sort_keys=True))
    else:
        print(f"=== eval_1v3_many_nn_fast: {res['candidates']} candidates ===")
        for row in res["results"]:
            print(
                f"{row['candidate']:>3}: {row['a_wins']}/{row['games']} "
                f"({100 * row['a_winrate']:.1f}%)  {row['a_weights']}")
        print(f"elapsed: {res['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
