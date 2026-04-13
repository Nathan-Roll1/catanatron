#!/usr/bin/env python3
"""Train cl7k: cl6k policy (frozen) + retrained value head on deterministic self-play.

Phase 1: Collect 1k deterministic NNt10 games using cl6k policy (parallel, C engine)
Phase 2: Fine-tune value head only on real game outcomes (freeze everything else)
Phase 3: Export and quick-verify

Usage:
    python -u human_bot/train_cl7k.py
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
SEARCH_DEPTH = 10
TOP_K = 5
WORKERS = 16
SEED_BASE = 300000
WEIGHTS_PATH = os.path.join(CSRC, "nn_weights_cl6k.bin")
CHECKPOINT_IN = os.path.join(CKPT_DIR, "cl6k.pt")
CHECKPOINT_OUT = os.path.join(CKPT_DIR, "cl7k.pt")

EPOCHS = 3
LR = 5e-4
BATCH_SIZE = 4096


# ── Phase 1: Parallel data collection ────────────────────────────

def collect_one_game(args):
    seed, weights_path, depth, top_k = args

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    lib_path = os.path.join(PROJECT_ROOT, "catan_player", "libcatan_nn.dylib")
    if not os.path.exists(lib_path):
        lib_path = os.path.join(CSRC, "libnn.dylib")
    nn_lib = ctypes.CDLL(lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, weights_path.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)

    def _enc(game): se.encode_into(game.get_state_view(), nf, ef, ff)
    def _mask(le):
        mk[:] = 0; mn = ae.get_action_mask(le).numpy(); mk[:len(mn)] = mn; return mn
    def c_val(game):
        _enc(game); _mask(game.get_legal_actions())
        nn_lib.nn_value_only(mptr, nfp, efp, ffp, mkp, vlp); return vl.copy()
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

    def nnt_search(game, le):
        seat = game.current_player()
        candidates = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            candidates = c_topk(game, le, top_k)
        bp, bv = 0, -1e30
        for p, ci in enumerate(candidates):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal(): break
                c_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
            else:
                vs = c_val(gc); off = (seat - gc.current_player()) % 4; v = float(vs[off])
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(candidates[bp], le)

    game = CatanGame(seed=seed); game.reset()
    game_nf, game_ef, game_ff, game_mk = [], [], [], []
    game_action, game_player = [], []

    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        cp = game.current_player()
        if len(le) == 1:
            game.step(0); continue

        nf_rec = np.zeros((N, NF), dtype=np.float32)
        ef_rec = np.zeros((E, EF), dtype=np.float32)
        ff_rec = np.zeros(FFD, dtype=np.float32)
        se.encode_into(game.get_state_view(), nf_rec, ef_rec, ff_rec)
        mn = ae.get_action_mask(le).numpy()
        mk_rec = np.zeros(397, dtype=np.float32)
        mk_rec[:len(mn)] = mn

        chosen = nnt_search(game, le)
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

    return {
        "nf": np.stack(game_nf),
        "ef": np.stack(game_ef),
        "ff": np.stack(game_ff),
        "mk": np.stack(game_mk),
        "action": np.array(game_action, dtype=np.int64),
        "reward": np.stack(rewards),
        "player": np.array(game_player, dtype=np.int64),
        "winner": winner,
        "turns": game.turn_number,
        "n": n,
    }


# ── Phase 2: Value-only training ─────────────────────────────────

def train_value_only(net, data, edge_index, device, epochs, lr, batch_size):
    from human_bot.loss import value_loss

    for name, param in net.named_parameters():
        if not name.startswith("value_head."):
            param.requires_grad = False

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params (value_head only)")

    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4)

    S = data["nf"].shape[0]
    nf_t = torch.from_numpy(data["nf"]).to(device)
    ef_t = torch.from_numpy(data["ef"]).to(device)
    ff_t = torch.from_numpy(data["ff"]).to(device)
    mk_t = torch.from_numpy(data["mk"]).to(device)
    vt_t = torch.from_numpy(data["reward"]).to(device)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(S, device=device)
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for i in range(0, S, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 16: continue

            out = net({
                "node_features": nf_t[idx],
                "edge_index": edge_index,
                "edge_features": ef_t[idx],
                "flat_features": ff_t[idx],
                "action_mask": mk_t[idx],
            })

            turn_progress = ff_t[idx, 114] if ff_t.shape[1] > 114 else None
            v_loss = value_loss(out["value"], vt_t[idx], turn_progress=turn_progress)

            optimizer.zero_grad(set_to_none=True)
            v_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                vacc = (out["value"].argmax(-1) == vt_t[idx].argmax(-1)).float().mean().item()

            total_loss += v_loss.item()
            total_acc += vacc
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        print(f"  Epoch {ep+1}/{epochs}: vloss={avg_loss:.4f} vacc={avg_acc:.3f} "
              f"lr={optimizer.param_groups[0]['lr']:.1e}")

    for param in net.parameters():
        param.requires_grad = True


# ── Main ─────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)
    t_total = time.perf_counter()

    # Phase 1: Collect
    print(f"{'='*60}")
    print(f"  Phase 1: Collect {NUM_GAMES} deterministic NNt{SEARCH_DEPTH} games")
    print(f"  Model: cl6k, {WORKERS} workers")
    print(f"{'='*60}")

    jobs = [(SEED_BASE + gi, WEIGHTS_PATH, SEARCH_DEPTH, TOP_K)
            for gi in range(NUM_GAMES)]

    t0 = time.perf_counter()
    with mp.Pool(WORKERS) as pool:
        results = pool.map(collect_one_game, jobs)
    collect_time = time.perf_counter() - t0

    results = [r for r in results if r is not None]
    winners = [r["winner"] for r in results]
    total_decisions = sum(r["n"] for r in results)
    print(f"\n  Collected {total_decisions:,} decisions from {len(results)} games "
          f"in {collect_time:.0f}s ({len(results)/collect_time*60:.0f} games/min)")

    all_nf = np.concatenate([r["nf"] for r in results])
    all_ef = np.concatenate([r["ef"] for r in results])
    all_ff = np.concatenate([r["ff"] for r in results])
    all_mk = np.concatenate([r["mk"] for r in results])
    all_action = np.concatenate([r["action"] for r in results])
    all_reward = np.concatenate([r["reward"] for r in results])
    del results; gc.collect()

    data = {"nf": all_nf, "ef": all_ef, "ff": all_ff, "mk": all_mk,
            "action": all_action, "reward": all_reward}
    print(f"  Dataset: {all_nf.shape[0]:,} samples")

    # Phase 2: Train value head
    print(f"\n{'='*60}")
    print(f"  Phase 2: Fine-tune value head (freeze policy/trunk/GNN)")
    print(f"  {EPOCHS} epochs, lr={LR}, batch_size={BATCH_SIZE}")
    print(f"{'='*60}")

    sys.path.insert(0, PROJECT_ROOT)
    from human_bot.model import HumanBotNet

    device = "cpu"
    net = HumanBotNet.load_checkpoint(CHECKPOINT_IN, device=device)
    print(f"  Loaded cl6k: {net.num_parameters:,} params")

    g0_tmp = __import__("hexzero.game.interface", fromlist=["CatanGame"]).CatanGame(seed=0)
    g0_tmp.reset()
    edge_index = g0_tmp.make_state_encoder()._edge_index.to(device)

    train_value_only(net, data, edge_index, device, EPOCHS, LR, BATCH_SIZE)

    # Save
    os.makedirs(CKPT_DIR, exist_ok=True)
    net.eval()
    net.save_checkpoint(CHECKPOINT_OUT, {
        "method": "value_finetune",
        "base": "cl6k",
        "games": NUM_GAMES,
        "depth": SEARCH_DEPTH,
        "epochs": EPOCHS,
        "lr": LR,
    })
    print(f"\n  Saved: {CHECKPOINT_OUT}")

    # Phase 3: Export
    print(f"\n{'='*60}")
    print(f"  Phase 3: Export to C weights")
    print(f"{'='*60}")
    from human_bot.export_nn import export
    export(CHECKPOINT_OUT, os.path.join(CSRC, "nn_weights_cl7k.bin"))

    total_time = time.perf_counter() - t_total
    print(f"\nDone in {total_time:.0f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
