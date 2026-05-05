#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import sys

import numpy as np
import torch

from human_bot.spatial_policy_heuristic import (
    AD, SpatialPolicyHeuristic, SpatialPolicyHeuristicMLP,
)


def play_one(args):
    seed, nn_seats, ab_seats, ckpt_path, ab_depth = args
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    if meta.get("model") == "mlp":
        model = SpatialPolicyHeuristicMLP(ckpt["tile_nodes"], hidden=int(meta.get("hidden", 128)))
    else:
        model = SpatialPolicyHeuristic(ckpt["tile_nodes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ae = ActionEncoder()
    lib = load_library()
    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)
    ch2 = CGame()
    ca2 = (Action * MAX_ACTIONS)()
    cn2 = ctypes.c_int(0)

    def h_choose(le):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mask = ae.get_action_mask(le).numpy()[:AD].astype(bool)
        with torch.no_grad():
            logits = model(
                torch.from_numpy(nf).unsqueeze(0),
                torch.from_numpy(ef).unsqueeze(0),
                torch.from_numpy(ff).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
            )[0].numpy()
        ai = int(np.argmax(logits))
        return next((i for i, a in enumerate(le) if ae.encode(a) == ai), 0)

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

    nn_set = set(nn_seats)
    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            continue
        if game.current_player() in nn_set:
            game.step(h_choose(le))
        else:
            game.step(ab_choose(le))

    w = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    nn_vp = sum(vps[s] for s in nn_seats) / len(nn_seats)
    ab_vp = sum(vps[s] for s in ab_seats) / len(ab_seats)
    rank_sum = sum(sorted(range(4), key=lambda p: (-vps[p], p)).index(s) + 1 for s in nn_seats)
    tag = "NN" if w is not None and w in nn_set else ("AB2" if w is not None else "draw")
    return seed, w, tag, game.turn_number, vps, nn_vp, ab_vp, rank_sum, len(nn_seats)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed-base", type=int, default=390000)
    p.add_argument("--mode", choices=["1v3", "2v2"], default="1v3")
    p.add_argument("--ab-depth", type=int, default=2)
    args = p.parse_args()

    jobs = []
    for gi in range(args.games):
        if args.mode == "1v3":
            nn_s = [gi % 4]
            ab_s = [s for s in range(4) if s != gi % 4]
        else:
            nn_s = [gi % 4, (gi + 2) % 4]
            ab_s = [(gi + 1) % 4, (gi + 3) % 4]
        jobs.append((args.seed_base + gi, nn_s, ab_s, args.checkpoint, args.ab_depth))

    print(f"Running {args.games} games: spatial heuristic vs AB2 ({args.mode})", flush=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)

    nn_w = sum(1 for r in results if r[2] == "NN")
    ab_w = sum(1 for r in results if r[2] == "AB2")
    nn_vp = sum(r[5] for r in results) / args.games
    ab_vp = sum(r[6] for r in results) / args.games
    rank = sum(r[7] for r in results) / max(sum(r[8] for r in results), 1)
    turns = sum(r[3] for r in results) / args.games
    print("\n===== RESULTS =====")
    print(f"  Wins: {nn_w}/{args.games} ({100*nn_w/args.games:.1f}%)")
    print(f"  AB2:  {ab_w}/{args.games}")
    print(f"  Avg VP: NN={nn_vp:.2f} AB2={ab_vp:.2f}")
    print(f"  Avg NN rank: {rank:.2f}/4")
    print(f"  Avg turns: {turns:.1f}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
