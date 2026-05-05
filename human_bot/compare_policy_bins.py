"""Compare two HBOT policy binaries on base-M2 roll-in states.

This is a cheap no-search proxy for Eggroll debugging. It answers:
  - does the perturbation change any legal top-1 choices?
  - how large is the legal-logit drift?
  - does the candidate still return legal actions?
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np


AD = 337
MASK_DIM = 397
MODEL_BYTES = 16 * 1024 * 1024
FP = ctypes.POINTER(ctypes.c_float)


def _resolve_libnn_path():
    import platform
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hostname = platform.node().split(".")[0]
    candidates = [
        os.path.join(project_root, "csrc", f"libnn_{hostname}.so"),
        os.path.join(project_root, "csrc", "libnn.so"),
        os.path.join(project_root, "csrc", "libnn.dylib"),
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


def _top_idx(lib, ptr, se, ae, game, nf, ef, ff, mk, out):
    le = game.get_legal_actions()
    se.encode_into(game.get_state_view(), nf, ef, ff)
    mn = ae.get_action_mask(le).numpy()
    mk[:] = 0
    mk[:len(mn)] = mn
    lib.nn_forward(
        ptr,
        nf.ctypes.data_as(FP),
        ef.ctypes.data_as(FP),
        ff.ctypes.data_as(FP),
        mk.ctypes.data_as(FP),
        out.ctypes.data_as(ctypes.c_void_p),
    )
    logits = out[4:4 + AD].copy()
    logits[mn[:AD] < 0.5] = -1e9
    top = int(np.argmax(logits))
    return top, logits, mn, le


def run(a_weights, b_weights, seeds, steps, seed_base):
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

    a_buf, a_ptr = _load_model(lib, os.path.abspath(a_weights))
    b_buf, b_ptr = _load_model(lib, os.path.abspath(b_weights))
    _keepalive = (a_buf, b_buf)
    ae = ActionEncoder()

    total = 0
    disagree = 0
    illegal = 0
    abs_diffs = []
    margin_diffs = []
    t0 = time.time()

    for si in range(seeds):
        game = CatanGame(seed=seed_base + si)
        game.reset()
        se = game.make_state_encoder()
        nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
        ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
        ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
        mk = np.zeros(MASK_DIM, dtype=np.float32)
        out_a = np.zeros(4 + MASK_DIM, dtype=np.float32)
        out_b = np.zeros(4 + MASK_DIM, dtype=np.float32)
        for _ in range(steps):
            if game.is_terminal():
                break
            le = game.get_legal_actions()
            if not le:
                break
            if len(le) == 1:
                game.step(0)
                continue
            a_top, a_logits, mn, _ = _top_idx(lib, a_ptr, se, ae, game, nf, ef, ff, mk, out_a)
            b_top, b_logits, _, _ = _top_idx(lib, b_ptr, se, ae, game, nf, ef, ff, mk, out_b)
            legal = np.flatnonzero(mn[:AD] > 0.5)
            total += 1
            disagree += int(a_top != b_top)
            illegal += int(a_top not in set(int(x) for x in legal))
            d = np.abs(a_logits[legal] - b_logits[legal])
            abs_diffs.append(float(d.mean()))
            top2_a = np.partition(a_logits[legal], -2)[-2:]
            top2_b = np.partition(b_logits[legal], -2)[-2:]
            margin_diffs.append(float(abs((top2_a[-1] - top2_a[-2]) - (top2_b[-1] - top2_b[-2]))))

            # Roll in with frozen base M2 so every candidate sees the same states.
            base_action_idx = b_top
            step_idx = 0
            for i, action in enumerate(le):
                try:
                    if ae.encode(action) == base_action_idx:
                        step_idx = i
                        break
                except ValueError:
                    continue
            game.step(step_idx)

    return {
        "states": total,
        "disagreement": disagree / max(1, total),
        "illegal_rate": illegal / max(1, total),
        "legal_abs_diff_mean": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
        "margin_diff_mean": float(np.mean(margin_diffs)) if margin_diffs else 0.0,
        "elapsed_sec": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-weights", required=True)
    parser.add_argument("--b-weights", required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed-base", type=int, default=6000000)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
