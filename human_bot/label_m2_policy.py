#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="csrc/data_super_m2/super_m2_4way_dense100_seed300000.pt")
    p.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    p.add_argument("--out-key", default="m2_action_idx")
    args = p.parse_args()

    d = torch.load(args.shard, map_location="cpu", weights_only=False)
    nf = d["node_features"].numpy().astype(np.float32, copy=False)
    ef = d["edge_features"].numpy().astype(np.float32, copy=False)
    ff = d["flat_features"].numpy().astype(np.float32, copy=False)
    mask = d["action_mask"].numpy().astype(np.float32, copy=False)
    n = nf.shape[0]

    lib_path = "csrc/libnn.dylib"
    if not os.path.exists(lib_path):
        lib_path = "csrc/libnn.so"
    lib = ctypes.CDLL(os.path.abspath(lib_path))
    FP = ctypes.POINTER(ctypes.c_float)
    lib.nn_load.restype = ctypes.c_int
    lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nn_policy_only.restype = None
    lib.nn_policy_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]

    mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    rc = lib.nn_load(mptr, os.path.abspath(args.weights).encode())
    if rc != 0:
        raise RuntimeError(f"nn_load failed: {rc}")

    out = np.zeros(397, dtype=np.float32)
    labels = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        lib.nn_policy_only(
            mptr,
            nf[i].ctypes.data_as(FP),
            ef[i].ctypes.data_as(FP),
            ff[i].ctypes.data_as(FP),
            mask[i].ctypes.data_as(FP),
            out.ctypes.data_as(FP),
        )
        logits = out[:337].copy()
        logits[mask[i, :337] < 0.5] = -1e9
        labels[i] = int(np.argmax(logits))
        if (i + 1) % 2000 == 0:
            print(f"  labeled {i+1}/{n}", flush=True)

    d[args.out_key] = torch.from_numpy(labels)
    tmp = args.shard + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, args.shard)
    print(f"Added {args.out_key} to {args.shard}")
    print("agreement with search action:",
          float((d["action_idx"] == d[args.out_key]).float().mean().item()))


if __name__ == "__main__":
    main()
