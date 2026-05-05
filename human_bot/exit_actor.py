#!/usr/bin/env python3
"""ExIt actor: generates training data using rollout search.

For each decision in a 4-player self-play game, uses top-4 policy candidates
and 60-ply rollout with AB value at leaves to find the best move. Records
the search-improved action as the training target.

Uses C inference (libnn) for fast rollout — runs on CPU, no GPU needed.

Usage:
    python -m human_bot.exit_actor \
        --weights csrc/nn_weights_m2.bin \
        --shard-dir data/c_selfplay_v4 \
        --num-games 1000 --workers 16
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

CSRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
AD = 337
MASK_DIM = 397
GAMES_PER_SHARD = 10


def play_one_exit(args):
    seed, weights_path, depth, top_k = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    _lib_path = os.path.join(CSRC, "libnn.so")
    if not os.path.exists(_lib_path):
        _lib_path = os.path.join(CSRC, "libnn.dylib")
    nn_lib = ctypes.CDLL(_lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [
        ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, weights_path.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)

    def _forward():
        out = np.zeros(4 + MASK_DIM, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        return out

    def _encode(game):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        le = game.get_legal_actions()
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        return le

    def _policy_argmax(game):
        le = _encode(game)
        if not le:
            return
        if len(le) == 1:
            game.step(0)
            return
        out = _forward()
        lo = out[4:4 + AD]
        lo[mk[:AD] < 0.5] = -1e9
        ai = int(np.argmax(lo))
        game.step(next((i for i, a in enumerate(le)
                        if ae.encode(a) == ai), 0))

    def _policy_topk(game, le, k):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        out = _forward()
        lo = out[4:4 + AD]
        scored = []
        for i, a in enumerate(le):
            try:
                scored.append((lo[ae.encode(a)], i))
            except ValueError:
                scored.append((-1e9, i))
        scored.sort(reverse=True)
        return [idx for _, idx in scored[:k]]

    def _ab_leaf(game, color):
        ch = CGame()
        lib.game_copy(ctypes.byref(ch), ctypes.byref(game._game))
        return float(lib.base_value_fn(ctypes.byref(ch), color))

    def _search_choose(game, le):
        """Top-k rollout search. Returns best action index."""
        seat = game.current_player()
        color = game._game.state.colors[game._game.state.current_player_index]

        if len(le) <= top_k:
            candidates = list(range(len(le)))
        else:
            candidates = _policy_topk(game, le, top_k)

        best_i, best_v = candidates[0], -1e30
        for ci in candidates:
            gc = game.clone()
            gc.step(ci)
            for _ in range(depth - 1):
                if gc.is_terminal():
                    break
                _policy_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (
                    -10.0 if w is not None else 0.0)
            else:
                v = _ab_leaf(gc, color)
            v = apply_action_bonus(v, le[ci])
            if v > best_v:
                best_v = v
                best_i = ci
        return fix_robber_steal(best_i, le)

    game = CatanGame(seed=seed, random_board=True)
    game.reset()

    steps = []
    while (not game.is_terminal()
           and game.turn_number < 500
           and len(steps) < 2000):
        le = game.get_legal_actions()
        if not le:
            break
        if len(le) == 1:
            game.step(0)
            continue

        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mask_copy = mn.copy()

        if game.turn_number <= 7:
            mk[:] = 0
            mk[:len(mn)] = mn
            out = _forward()
            lo = out[4:4 + AD]
            lo[mk[:AD] < 0.5] = -1e9
            chosen = next((i for i, a in enumerate(le)
                           if ae.encode(a) == int(np.argmax(lo))), 0)
        else:
            chosen = _search_choose(game, le)

        try:
            enc_action = ae.encode(le[chosen])
        except ValueError:
            game.step(chosen)
            continue

        steps.append({
            "nf": nf.copy(),
            "ef": ef.copy(),
            "ff": ff.copy(),
            "mask": mask_copy,
            "action_idx": enc_action,
            "player": game.current_player(),
        })
        game.step(chosen)

    winner = game.winner()
    reward_vec = np.zeros(4, dtype=np.float32)
    final_vp = np.array([game._game.state.player_state[s][0]
                         for s in range(4)], dtype=np.float32)
    if winner is not None:
        reward_vec[winner] = 1.0
        turns = game.turn_number
        speed_bonus = max(0.0, min(0.5, (300 - turns) / 300.0))
        reward_vec[winner] = 1.0 + speed_bonus
        for seat in range(4):
            if seat != winner:
                reward_vec[seat] = final_vp[seat] / 20.0

    from human_bot.selfplay import compute_step_weights
    sw = compute_step_weights(steps, reward_vec)
    return steps, reward_vec, sw, winner, final_vp, game.turn_number


def save_exit_shard(games_data, output_dir, shard_id):
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player, all_reward, all_sw = [], [], [], [], []
    all_final_vp = []

    for steps, rv, sw, fvp in games_data:
        for i, s in enumerate(steps):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_sw.append(sw[i])
            all_final_vp.append(fvp)

    if not all_nf:
        return 0

    data = {
        "node_features": torch.from_numpy(np.stack(all_nf)),
        "edge_features": torch.from_numpy(np.stack(all_ef)),
        "flat_features": torch.from_numpy(np.stack(all_ff)),
        "action_mask": torch.from_numpy(np.stack(all_mask)),
        "action_idx": torch.tensor(all_act, dtype=torch.int64),
        "player": torch.tensor(all_player, dtype=torch.int64),
        "reward_vec": torch.from_numpy(np.stack(all_reward)),
        "step_weight": torch.tensor(all_sw, dtype=torch.float32),
        "log_prob_old": torch.zeros(len(all_nf), dtype=torch.float32),
        "value_pred_old": torch.zeros(len(all_nf), 4, dtype=torch.float32),
        "final_vp": torch.from_numpy(np.stack(all_final_vp)),
    }
    path = os.path.join(output_dir, f"{shard_id}.pt")
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.rename(tmp, path)
    return len(all_nf)


def main():
    parser = argparse.ArgumentParser(
        description="ExIt actor: rollout search self-play")
    parser.add_argument("--weights", type=str, required=True,
                        help="C inference weights (.bin)")
    parser.add_argument("--shard-dir", type=str, required=True,
                        help="Output shard directory (writes to pending/)")
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--depth", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=1000000)
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoint dir to watch for new weights")
    args = parser.parse_args()

    weights_path = os.path.abspath(args.weights)
    pending_dir = os.path.join(args.shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    print(f"ExIt actor: {args.num_games} games, depth={args.depth}, "
          f"top_k={args.top_k}, {args.workers} workers", flush=True)

    jobs = [(args.seed_base + gi, weights_path, args.depth, args.top_k)
            for gi in range(args.num_games)]

    t0 = time.time()
    shard_idx = 0
    game_batch = []
    completed = 0

    with mp.Pool(args.workers) as pool:
        for result in pool.imap_unordered(play_one_exit, jobs):
            steps, rv, sw, winner, fvp, turns = result
            game_batch.append((steps, rv, sw, fvp))
            completed += 1

            if len(game_batch) >= GAMES_PER_SHARD:
                sid = f"exit_{shard_idx:06d}"
                n = save_exit_shard(game_batch, pending_dir, sid)
                game_batch = []
                shard_idx += 1

            if completed % 10 == 0:
                elapsed = time.time() - t0
                gps = completed / elapsed if elapsed > 0 else 0
                print(f"  {completed}/{args.num_games} games, "
                      f"{shard_idx} shards, {gps:.1f} g/s", flush=True)

    if game_batch:
        sid = f"exit_{shard_idx:06d}"
        save_exit_shard(game_batch, pending_dir, sid)
        shard_idx += 1

    dt = time.time() - t0
    print(f"\nDone: {completed} games, {shard_idx} shards, "
          f"{dt:.0f}s ({completed / dt:.1f} g/s)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
