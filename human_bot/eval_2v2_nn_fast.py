"""Fast 2v2 NN-vs-NN evaluation with persistent worker model loads.

`eval_2v2_nn.py` is simple and robust, but each game reloads both models.
Eggroll-style candidate search evaluates many nearby models, so this variant
loads candidate/base weights once per worker and reuses them across games.
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
    for p in candidates:
        if os.path.exists(p):
            return p
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

    a_buf, a_ptr = _load_model(lib, a_weights)
    b_buf, b_ptr = _load_model(lib, b_weights)
    _G.update({
        "lib": lib,
        "a_buf": a_buf,
        "a_ptr": a_ptr,
        "b_buf": b_buf,
        "b_ptr": b_ptr,
        "ae": ActionEncoder(),
        "CatanGame": CatanGame,
    })


def _play_one(job):
    game_idx, seed, a_seats, b_seats = job
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

    def nn_argmax(le, model_ptr):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        lib.nn_forward(model_ptr, nfp, efp, ffp, mkp, outp)
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
        model_ptr = _G["a_ptr"] if cp in a_seats else _G["b_ptr"]
        game.step(nn_argmax(le, model_ptr))

    winner = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    return {
        "game_idx": game_idx,
        "seed": seed,
        "a_seats": list(a_seats),
        "b_seats": list(b_seats),
        "winner": winner,
        "a_won": winner in a_seats if winner is not None else False,
        "vps": vps,
    }


def run(a_weights, b_weights, games, workers, seed_base):
    rotations = [
        ({0, 2}, {1, 3}),
        ({1, 3}, {0, 2}),
        ({0, 1}, {2, 3}),
        ({2, 3}, {0, 1}),
    ]
    jobs = []
    for gi in range(games):
        a_seats, b_seats = rotations[gi % len(rotations)]
        jobs.append((gi, seed_base + gi, a_seats, b_seats))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(os.path.abspath(a_weights), os.path.abspath(b_weights)),
    ) as pool:
        results = list(pool.imap_unordered(_play_one, jobs))
    elapsed = time.time() - t0

    a_wins = sum(1 for r in results if r["a_won"])
    b_wins = sum(1 for r in results if (not r["a_won"]) and r["winner"] is not None)
    no_winner = sum(1 for r in results if r["winner"] is None)
    return {
        "games": games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "no_winner": no_winner,
        "a_winrate": a_wins / max(1, games),
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-weights", required=True)
    parser.add_argument("--b-weights", required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=2000000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print("=== eval_2v2_nn_fast ===")
    print(f"  A: {os.path.abspath(args.a_weights)}")
    print(f"  B: {os.path.abspath(args.b_weights)}")
    print(f"  games:    {args.games}  workers: {args.workers}")
    print(flush=True)

    res = run(args.a_weights, args.b_weights, args.games, args.workers, args.seed_base)
    print("=== RESULTS ===")
    print(f"  A wins: {res['a_wins']}/{args.games}  ({100 * res['a_winrate']:.1f}%)")
    print(f"  B wins: {res['b_wins']}/{args.games}  ({100 * res['b_wins'] / max(1, args.games):.1f}%)")
    if res["no_winner"]:
        print(f"  No winner: {res['no_winner']}")
    print(f"  elapsed: {res['elapsed_sec']:.1f}s")
    if args.json:
        print(json.dumps(res, sort_keys=True))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
