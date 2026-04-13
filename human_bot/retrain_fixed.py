#!/usr/bin/env python3
"""Retrain clb on fixed engine: 1k ABt5 self-play games, 1 epoch.

Uses AB2 value function for leaf evaluation and opponent responses,
NN policy for candidate selection. Parallel collection with C engine.

Usage:
    python -u human_bot/retrain_fixed.py
"""

import ctypes
import gc
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(PROJECT_ROOT, "csrc")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

NUM_GAMES = 1000
SEARCH_DEPTH = 5
TOP_K = 5
WORKERS = 16
SEED_BASE = 600000
NN_WEIGHTS = os.path.join(CSRC, "nn_weights_cluster.bin")  # clb
CHECKPOINT_IN = os.path.join(CKPT_DIR, "cluster_run", "final.pt")  # clb
CHECKPOINT_OUT = os.path.join(CKPT_DIR, "v2_r1.pt")

EPOCHS = 1
LR = 3e-3
BATCH_SIZE = 4096


def collect_one_game(args):
    seed, nn_weights, depth, top_k = args

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    nn_lib_path = os.path.join(PROJECT_ROOT, "catan_player", "libcatan_nn.dylib")
    if not os.path.exists(nn_lib_path):
        nn_lib_path = os.path.join(CSRC, "libnn.dylib")
    nn_lib = ctypes.CDLL(nn_lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, nn_weights.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    def _enc(game): se.encode_into(game.get_state_view(), nf, ef, ff)
    def _mask(le):
        mk[:] = 0; mn = ae.get_action_mask(le).numpy(); mk[:len(mn)] = mn; return mn

    def c_topk(game, le, k):
        _enc(game); mn = _mask(le)
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; a2i = {ae.encode(a): i for i, a in enumerate(le)}
        return [li for _, li in sorted([(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]

    def c_argmax(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        _enc(gc); mn = _mask(le)
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        gc.step(next((i for i, a in enumerate(le) if ae.encode(a) == int(np.argmax(lo))), 0))

    def ab2_respond(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        cg = gc._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv: bv = v; bi = i
        gc.step(bi)

    def abt_search(game, le, nn_seats, ab2_seats):
        """ABt search: NN policy top-K, AB2 value + AB2 responses."""
        seat = game.current_player()
        cands = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            cands = c_topk(game, le, top_k)
        bp, bv = 0, -1e30
        for p, ci in enumerate(cands):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal(): break
                cp = gc.current_player()
                if cp in ab2_seats:
                    ab2_respond(gc)
                else:
                    c_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
            else:
                cg = gc._game
                bc = cg.state.colors[cg.state.current_player_index]
                v = float(lib.base_value_fn(ctypes.byref(cg), bc))
                seat_offset = (seat - gc.current_player()) % 4
                if seat_offset != 0:
                    v = -v
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(cands[bp], le)

    game = CatanGame(seed=seed); game.reset()
    gi = seed - SEED_BASE
    nn_seats = {gi % 4, (gi + 2) % 4}
    ab2_seats = {(gi + 1) % 4, (gi + 3) % 4}

    game_nf, game_ef, game_ff, game_mk = [], [], [], []
    game_action, game_player = [], []

    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        cp = game.current_player()

        if cp in ab2_seats:
            if len(le) == 1: game.step(0)
            else: ab2_respond(game)
            continue

        if len(le) == 1:
            game.step(0); continue

        nf_rec = np.zeros((N, NF), dtype=np.float32)
        ef_rec = np.zeros((E, EF), dtype=np.float32)
        ff_rec = np.zeros(FFD, dtype=np.float32)
        se.encode_into(game.get_state_view(), nf_rec, ef_rec, ff_rec)
        mn = ae.get_action_mask(le).numpy()
        mk_rec = np.zeros(397, dtype=np.float32)
        mk_rec[:len(mn)] = mn

        chosen = abt_search(game, le, nn_seats, ab2_seats)
        act = le[chosen]
        action_idx = ae.encode(act)

        game_nf.append(nf_rec)
        game_ef.append(ef_rec)
        game_ff.append(ff_rec)
        game_mk.append(mk_rec)
        game_action.append(action_idx)
        game_player.append(cp)
        game.step(chosen)

    winner = game.winner()
    n = len(game_action)
    if n == 0:
        return None

    rewards = []
    for di in range(n):
        p = game_player[di]
        rv = np.zeros(4, dtype=np.float32)
        if winner is not None:
            rv[winner] = 1.0
        else:
            rv[:] = 0.25
        rv = np.roll(rv, (-p) % 4)
        rewards.append(rv)

    nn_won = winner in nn_seats if winner is not None else False
    return {
        "nf": np.stack(game_nf), "ef": np.stack(game_ef),
        "ff": np.stack(game_ff), "mk": np.stack(game_mk),
        "action": np.array(game_action, dtype=np.int64),
        "reward": np.stack(rewards),
        "n": n, "nn_won": nn_won, "turns": game.turn_number,
    }


def main():
    mp.set_start_method("spawn", force=True)
    t_total = time.perf_counter()

    print(f"{'='*60}")
    print(f"  Phase 1: Collect {NUM_GAMES} ABt{SEARCH_DEPTH} self-play games")
    print(f"  Base model: clb (cluster_run/final.pt)")
    print(f"  {WORKERS} workers, fixed engine")
    print(f"{'='*60}\n")

    jobs = [(SEED_BASE + gi, NN_WEIGHTS, SEARCH_DEPTH, TOP_K)
            for gi in range(NUM_GAMES)]

    t0 = time.perf_counter()
    with mp.Pool(WORKERS) as pool:
        results = pool.map(collect_one_game, jobs)
    collect_time = time.perf_counter() - t0

    results = [r for r in results if r is not None]
    total_decisions = sum(r["n"] for r in results)
    nn_wr = sum(r["nn_won"] for r in results) / len(results)
    print(f"  Collected {total_decisions:,} decisions from {len(results)} games "
          f"in {collect_time:.0f}s ({len(results)/collect_time*60:.0f} games/min)")
    print(f"  NN win rate vs AB2: {nn_wr:.1%}")

    # Convert to numpy arrays, free result dicts immediately
    all_nf = np.concatenate([r["nf"] for r in results])
    all_ef = np.concatenate([r["ef"] for r in results])
    all_ff = np.concatenate([r["ff"] for r in results])
    all_mk = np.concatenate([r["mk"] for r in results])
    all_action = np.concatenate([r["action"] for r in results])
    all_reward = np.concatenate([r["reward"] for r in results])
    del results; gc.collect()

    S = all_nf.shape[0]
    print(f"  Dataset: {S:,} samples\n")

    # Phase 2: Train (batch-at-a-time to avoid OOM)
    print(f"{'='*60}")
    print(f"  Phase 2: Train ({EPOCHS} epoch, lr={LR})")
    print(f"{'='*60}\n")

    sys.path.insert(0, PROJECT_ROOT)
    from human_bot.model import HumanBotNet
    from human_bot.loss import UncertaintyWeightedLoss, human_policy_loss, value_loss, masked_entropy

    device = "cpu"
    net = HumanBotNet.load_checkpoint(CHECKPOINT_IN, device=device)
    print(f"  Loaded clb: {net.num_parameters:,} params")

    g0_tmp = __import__("hexzero.game.interface", fromlist=["CatanGame"]).CatanGame(seed=0)
    g0_tmp.reset()
    edge_index = g0_tmp.make_state_encoder()._edge_index.to(device)

    loss_combiner = UncertaintyWeightedLoss().to(device)
    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=1e-4)

    net.train()
    perm = np.random.permutation(S)
    sums = {"ploss": 0, "vloss": 0, "pacc": 0, "vacc": 0}
    nb = 0

    for i in range(0, S, BATCH_SIZE):
        idx = perm[i:i+BATCH_SIZE]
        if len(idx) < 16: continue

        nf_b = torch.from_numpy(all_nf[idx])
        ef_b = torch.from_numpy(all_ef[idx])
        ff_b = torch.from_numpy(all_ff[idx])
        mk_b = torch.from_numpy(all_mk[idx])
        act_b = torch.from_numpy(all_action[idx])
        vt_b = torch.from_numpy(all_reward[idx])

        out = net({
            "node_features": nf_b, "edge_index": edge_index,
            "edge_features": ef_b, "flat_features": ff_b,
            "action_mask": mk_b,
        })

        p_loss = human_policy_loss(out["policy_logits"], act_b, mk_b,
                                   label_smoothing=0.02)
        turn_progress = ff_b[:, 114] if ff_b.shape[1] > 114 else None
        v_loss = value_loss(out["value"], vt_b, turn_progress=turn_progress)
        ent = masked_entropy(out["policy_logits"], mk_b)
        total, _ = loss_combiner(p_loss, v_loss, ent, 0.01)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pacc = (out["policy_logits"].argmax(-1) == act_b).float().mean().item()
            vacc = (out["value"].argmax(-1) == vt_b.argmax(-1)).float().mean().item()

        sums["ploss"] += p_loss.item()
        sums["vloss"] += v_loss.item()
        sums["pacc"] += pacc
        sums["vacc"] += vacc
        nb += 1

        del nf_b, ef_b, ff_b, mk_b, act_b, vt_b, out
        if nb % 5 == 0: gc.collect()

    avg = {k: v / max(nb, 1) for k, v in sums.items()}
    print(f"  Epoch 1: ploss={avg['ploss']:.3f} pacc={avg['pacc']:.3f} "
          f"vloss={avg['vloss']:.3f} vacc={avg['vacc']:.3f}")

    # Save
    os.makedirs(CKPT_DIR, exist_ok=True)
    net.eval()
    net.save_checkpoint(CHECKPOINT_OUT, {
        "method": "exit_v2_r1",
        "base": "clb",
        "games": NUM_GAMES,
        "depth": SEARCH_DEPTH,
        "search": "ABt5",
        "epochs": EPOCHS,
        "lr": LR,
        "engine": "fixed_settlement_cost",
    })
    print(f"\n  Saved: {CHECKPOINT_OUT}")

    # Export
    print(f"\n{'='*60}")
    print(f"  Phase 3: Export")
    print(f"{'='*60}\n")
    from human_bot.export_nn import export
    export(CHECKPOINT_OUT, os.path.join(CSRC, "nn_weights_v2_r1.bin"))

    total_time = time.perf_counter() - t_total
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
