#!/usr/bin/env python3
"""Benchmark C NN compute modes and compare top legal moves."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hexzero.encoder.action_encoder import ActionEncoder
from hexzero.game.interface import CatanGame


MODEL_BYTES = 16 * 1024 * 1024


def load_model(lib: ctypes.CDLL, weights: str, mode: str) -> tuple[ctypes.Array, ctypes.c_void_p]:
    os.environ["CATAN_NN_COMPUTE"] = mode
    buf = (ctypes.c_char * MODEL_BYTES)()
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    rc = lib.nn_load(ptr, weights.encode())
    if rc != 0:
        raise RuntimeError(f"nn_load failed for {weights} in mode={mode}: {rc}")
    return buf, ptr


def collect_states(n_seeds: int, steps_per_seed: int):
    ae = ActionEncoder()
    states = []
    for seed in range(n_seeds):
        game = CatanGame(seed=seed)
        game.reset()
        se = game.make_state_encoder()
        for _ in range(steps_per_seed):
            legal = game.get_legal_actions()
            if not legal:
                break
            nf = np.zeros((54, 18), dtype=np.float32)
            ef = np.zeros((144, 5), dtype=np.float32)
            ff = np.zeros(115, dtype=np.float32)
            se.encode_into(game.get_state_view(), nf, ef, ff)
            mask_np = ae.get_action_mask(legal).numpy().astype(np.float32)
            mask = np.zeros(397, dtype=np.float32)
            mask[: len(mask_np)] = mask_np
            states.append((nf, ef, ff, mask))
            game.step(0)
    return states


def bench_forward(lib, ptr, states, reps: int, mode: str) -> float:
    out = np.zeros(401, dtype=np.float32)
    policy = np.zeros(397, dtype=np.float32)
    val = np.zeros(4, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(reps):
        nf, ef, ff, mask = states[i % len(states)]
        if mode == "value":
            lib.nn_value_only(ptr, nf, ef, ff, mask, val)
        elif mode == "policy":
            lib.nn_policy_only(ptr, nf, ef, ff, mask, policy)
        else:
            lib.nn_forward(ptr, nf, ef, ff, mask, out.ctypes.data_as(ctypes.c_void_p))
    return (time.perf_counter() - t0) * 1e6 / reps


def compare_moves(lib, fp_ptr, test_ptr, states) -> tuple[float, float, float]:
    out_fp = np.zeros(401, dtype=np.float32)
    out_test = np.zeros(401, dtype=np.float32)
    agree = 0
    max_abs = []
    mean_abs = []
    for nf, ef, ff, mask in states:
        lib.nn_forward(fp_ptr, nf, ef, ff, mask, out_fp.ctypes.data_as(ctypes.c_void_p))
        lib.nn_forward(test_ptr, nf, ef, ff, mask, out_test.ctypes.data_as(ctypes.c_void_p))
        p_fp = out_fp[4:]
        p_test = out_test[4:]
        legal = np.flatnonzero(mask > 0.5)
        if legal.size == 0:
            continue
        agree += int(legal[np.argmax(p_fp[legal])] == legal[np.argmax(p_test[legal])])
        diff = np.abs(p_fp[legal] - p_test[legal])
        max_abs.append(float(diff.max()))
        mean_abs.append(float(diff.mean()))
    total = max(1, len(max_abs))
    return agree / total, float(np.mean(max_abs)), float(np.mean(mean_abs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", default="csrc/libnn.dylib")
    parser.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--mode", default="int8", choices=("int8", "fp32"))
    args = parser.parse_args()

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nn_load.restype = ctypes.c_int
    lib.nn_forward.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, ctypes.c_void_p]
    lib.nn_value_only.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, fp]
    lib.nn_policy_only.argtypes = [ctypes.c_void_p, fp, fp, fp, fp, fp]

    states = collect_states(args.seeds, args.steps)
    fp_buf, fp_ptr = load_model(lib, args.weights, "fp32")
    test_buf, test_ptr = load_model(lib, args.weights, args.mode)
    _keepalive = (fp_buf, test_buf)

    fp_value = bench_forward(lib, fp_ptr, states, args.reps, mode="value")
    test_value = bench_forward(lib, test_ptr, states, args.reps, mode="value")
    fp_policy = bench_forward(lib, fp_ptr, states, args.reps, mode="policy")
    test_policy = bench_forward(lib, test_ptr, states, args.reps, mode="policy")
    fp_full = bench_forward(lib, fp_ptr, states, args.reps, mode="full")
    test_full = bench_forward(lib, test_ptr, states, args.reps, mode="full")
    agree, max_abs, mean_abs = compare_moves(lib, fp_ptr, test_ptr, states)

    print(f"states={len(states)} reps={args.reps} mode={args.mode}")
    print(f"value_only_us fp32={fp_value:.1f} {args.mode}={test_value:.1f}")
    print(f"policy_only_us fp32={fp_policy:.1f} {args.mode}={test_policy:.1f}")
    print(f"full_forward_us fp32={fp_full:.1f} {args.mode}={test_full:.1f}")
    print(f"top1_legal_agreement={agree:.4f}")
    print(f"policy_legal_abs max_mean={max_abs:.4f} mean_mean={mean_abs:.4f}")


if __name__ == "__main__":
    main()
