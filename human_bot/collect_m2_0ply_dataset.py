#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import sys

import numpy as np
import torch


def play_one(args):
    seed, nn_seat, weights_path, ab_depth, behavior_ckpt = args
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    ae = ActionEncoder()
    lib = load_library()
    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()

    lib_path = "csrc/libnn.dylib"
    if not os.path.exists(lib_path):
        lib_path = "csrc/libnn.so"
    nn_lib = ctypes.CDLL(os.path.abspath(lib_path))
    FP = ctypes.POINTER(ctypes.c_float)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nn_lib.nn_policy_only.restype = None
    nn_lib.nn_policy_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, os.path.abspath(weights_path).encode()) == 0

    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
    mask = np.zeros(397, dtype=np.float32)
    out = np.zeros(397, dtype=np.float32)
    behavior_model = None
    if behavior_ckpt:
        from human_bot.spatial_policy_heuristic import AD, SpatialPolicyHeuristic
        ckpt = torch.load(behavior_ckpt, map_location="cpu", weights_only=False)
        behavior_model = SpatialPolicyHeuristic(ckpt["tile_nodes"])
        behavior_model.load_state_dict(ckpt["state_dict"])
        behavior_model.eval()

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)
    ch2 = CGame()
    ca2 = (Action * MAX_ACTIONS)()
    cn2 = ctypes.c_int(0)

    rows = []

    def encode_and_label(le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mask[:] = 0
        mask[:len(mn)] = mn
        nn_lib.nn_policy_only(
            mptr, nf.ctypes.data_as(FP), ef.ctypes.data_as(FP),
            ff.ctypes.data_as(FP), mask.ctypes.data_as(FP),
            out.ctypes.data_as(FP))
        logits = out[:337].copy()
        logits[mn[:337] < 0.5] = -1e9
        aidx = int(np.argmax(logits))
        chosen = next((i for i, a in enumerate(le) if ae.encode(a) == aidx), 0)
        rows.append({
            "nf": nf.copy(),
            "ef": ef.copy(),
            "ff": ff.copy(),
            "mask": mask.copy(),
            "m2_action_idx": aidx,
        })
        return mn, aidx

    def behavior_choose(le):
        mn, m2_aidx = encode_and_label(le)
        if behavior_model is None:
            aidx = m2_aidx
        else:
            with torch.no_grad():
                logits = behavior_model(
                    torch.from_numpy(nf).unsqueeze(0),
                    torch.from_numpy(ef).unsqueeze(0),
                    torch.from_numpy(ff).unsqueeze(0),
                    torch.from_numpy((mn[:337] > 0.5)).unsqueeze(0),
                )[0].numpy()
            aidx = int(np.argmax(logits))
        return next((i for i, a in enumerate(le) if ae.encode(a) == aidx), 0)

    def ab_choose(le):
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if ab_depth >= 2 and cn.value > 0:
                if cn.value > 1:
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    brj, brv = 0, -1e30
                    for j in range(cn.value):
                        lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        lib.game_execute(ctypes.byref(ch2), ca[j], ca2, ctypes.byref(cn2))
                        rv = lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > brv:
                            brv, brj = rv, j
                    lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    lib.game_execute(ctypes.byref(ch2), ca[brj], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    lib.game_execute(ctypes.byref(ch), ca[0], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv:
                bv, bi = v, i
        return bi

    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            continue
        if game.current_player() == nn_seat:
            game.step(behavior_choose(le))
        else:
            game.step(ab_choose(le))

    w = game.winner()
    return {
        "seed": seed,
        "nn_seat": nn_seat,
        "winner": w,
        "vps": [int(game._game.state.player_state[s][0]) for s in range(4)],
        "rows": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=400000)
    p.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--out", default="csrc/data_super_m2/m2_0ply_1v3_100g_seed400000.pt")
    p.add_argument("--behavior-spatial-heuristic", type=str, default=None,
                   help="If set, play NN seat with this heuristic but label states with M2.")
    args = p.parse_args()

    jobs = [(args.seed_base + gi, gi % 4, args.weights, args.ab_depth,
             args.behavior_spatial_heuristic)
            for gi in range(args.games)]
    print(f"Collecting {args.games} M2 0-ply 1v3 games...", flush=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)

    all_nf, all_ef, all_ff, all_mask, all_act = [], [], [], [], []
    wins = 0
    for r in results:
        if r["winner"] == r["nn_seat"]:
            wins += 1
        for row in r["rows"]:
            all_nf.append(row["nf"])
            all_ef.append(row["ef"])
            all_ff.append(row["ff"])
            all_mask.append(row["mask"])
            all_act.append(row["m2_action_idx"])

    data = {
        "node_features": torch.from_numpy(np.stack(all_nf)),
        "edge_features": torch.from_numpy(np.stack(all_ef)),
        "flat_features": torch.from_numpy(np.stack(all_ff)),
        "action_mask": torch.from_numpy(np.stack(all_mask)),
        "m2_action_idx": torch.tensor(all_act, dtype=torch.int64),
        "action_idx": torch.tensor(all_act, dtype=torch.int64),
        "step_weight": torch.ones(len(all_act), dtype=torch.float32),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(data, args.out)
    print(f"Saved {len(all_act):,} rows to {args.out}")
    print(f"M2 WR in collection: {wins}/{args.games} ({100*wins/args.games:.1f}%)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
