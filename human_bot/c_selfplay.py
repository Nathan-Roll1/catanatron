#!/usr/bin/env python3
"""C-inference self-play: CPU actors with C NN + GPU learner with PyTorch.

Actors use the pure-C nn.c inference library (no GPU, no PyTorch).
Learner trains on GPU and exports fresh C weights after each round.
All 48 CPUs drive actors; 1 GPU trains.

Usage:
    python3 -u human_bot/c_selfplay.py \
        --checkpoint checkpoints/pretrain_v4/final.pt \
        --weights-bin csrc/nn_weights_sp.bin \
        --num-actors 40 \
        --search-depth 10 \
        --shard-dir data/selfplay_v4 \
        --ckpt-dir checkpoints/selfplay_v4
"""
from __future__ import annotations

import argparse
import ctypes
import gc as gc_mod
import json
import multiprocessing as mp
import os
import time
import traceback

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

AD = 337
MASK_DIM = 397
GAMES_PER_SHARD = 25
MAX_TURNS = 1000
MAX_STEPS = 2000
MAX_PENDING = 200
FP = ctypes.POINTER(ctypes.c_float)

_ACT_MOD = np.ones(MASK_DIM, dtype=np.float32)
_ACT_MOD[0] = 0.2
_ACT_MOD[1] = 0.5
_ACT_MOD[2:5] = 1.5
_ACT_MOD[5:113] = 1.5
_ACT_MOD[113:185] = 1.5
_ACT_MOD[185:280] = 1.5
_ACT_MOD[280:285] = 0.3
_ACT_MOD[285:310] = 1.5
_ACT_MOD[310:397] = 1.3


def compute_step_weights(steps, reward_vec):
    winner = int(np.argmax(reward_vec)) if reward_vec.max() > 0 else -1
    S = len(steps)
    weights = np.ones(S, dtype=np.float32)
    # Short games = more decisive play, boost all winner steps
    speed_mult = 1.0 + max(0.0, min(0.5, (600 - S) / 600.0))
    for i, s in enumerate(steps):
        progress = i / max(S - 1, 1)
        if s["player"] == winner:
            base = (1.0 + progress) * speed_mult
        else:
            base = max(0.2, 0.6 - 0.4 * progress)
        weights[i] = base * _ACT_MOD[min(s["action_idx"], MASK_DIM - 1)]
    return weights


def temperature_for_round(round_num, start=1.0, end=0.2, anneal=200):
    if round_num >= anneal:
        return end
    return start + (end - start) * (round_num / anneal)


def atomic_torch_save(data, path):
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.rename(tmp, path)


def save_shard(games_data, output_dir, shard_id):
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player, all_reward, all_sw = [], [], [], [], []
    for steps, rv, sw in games_data:
        for i, s in enumerate(steps):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_sw.append(sw[i])
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
    }
    atomic_torch_save(data, os.path.join(output_dir, f"{shard_id}.pt"))
    return len(all_nf)


# =====================================================================
# C-inference actor (runs on CPU only, no PyTorch, no GPU)
# =====================================================================

def _load_c_nn(weights_path):
    import platform
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hostname = platform.node().split(".")[0]
    # Try hostname-specific build first, then generic
    candidates = [
        os.path.join(project_root, "csrc", f"libnn_{hostname}.so"),
        os.path.join(project_root, "csrc", "libnn.so"),
        os.path.join(project_root, "csrc", "libnn.dylib"),
        os.path.join(project_root, "catan_player", "libcatan_nn.so"),
        os.path.join(project_root, "catan_player", "libcatan_nn.dylib"),
    ]
    lib_path = None
    for p in candidates:
        if os.path.exists(p):
            lib_path = p
            break
    if lib_path is None:
        raise FileNotFoundError(f"No libnn found, tried: {candidates}")
    nn_lib = ctypes.CDLL(lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    rc = nn_lib.nn_load(mptr, weights_path.encode())
    if rc != 0:
        raise RuntimeError(f"nn_load failed for {weights_path}")
    return nn_lib, mptr, mbuf


def run_actor(actor_id, weights_path, shard_dir, ckpt_dir,
              search_depth, seed_base, max_pending,
              explore_setup_frac=0.2, dirichlet_alpha=0.3,
              dirichlet_frac=0.25):
    try:
        _run_actor(actor_id, weights_path, shard_dir, ckpt_dir,
                   search_depth, seed_base, max_pending,
                   explore_setup_frac, dirichlet_alpha, dirichlet_frac)
    except Exception:
        print(f"!!! [c_actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_actor(actor_id, weights_path, shard_dir, ckpt_dir,
               search_depth, seed_base, max_pending,
               explore_setup_frac, dirichlet_alpha, dirichlet_frac):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    load_library()
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    nn_lib, mptr, _mbuf = _load_c_nn(weights_path)

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)
    vlp = vl.ctypes.data_as(FP)

    def enc(game):
        se.encode_into(game.get_state_view(), nf, ef, ff)

    def mask_le(le):
        mk[:] = 0
        mn = ae.get_action_mask(le).numpy()
        mk[:len(mn)] = mn
        return mn

    def c_val(game):
        enc(game); mask_le(game.get_legal_actions())
        nn_lib.nn_value_only(mptr, nfp, efp, ffp, mkp, vlp)
        return vl.copy()

    def c_forward(game, le):
        enc(game); mn = mask_le(le)
        out = np.zeros(4 + MASK_DIM, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        return out[:4], out[4:4+AD], mn

    def c_argmax(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        _, lo, mn = c_forward(gc, le)
        lo[mn[:AD] < 0.5] = -1e9
        gc.step(next((i for i, a in enumerate(le)
                      if ae.encode(a) == int(np.argmax(lo))), 0))

    def c_topk(game, le, k):
        _, lo, _ = c_forward(game, le)
        a2i = {}
        for i, a in enumerate(le):
            try:
                a2i[ae.encode(a)] = i
            except ValueError:
                continue
        return [li for _, li in sorted(
            [(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]

    def nnt_search(game, le, depth, temperature=1.0, top_k=5):
        seat = game.current_player()
        candidates = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            candidates = c_topk(game, le, top_k)

        # Dirichlet noise on candidate values (exploration)
        noise = np.random.dirichlet([dirichlet_alpha] * len(candidates))

        bp, bv = 0, -1e30
        values = np.zeros(len(candidates), dtype=np.float32)
        for p, ci in enumerate(candidates):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal(): break
                c_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (
                    -10.0 if w is not None else 0.0)
            else:
                vs = c_val(gc)
                off = (seat - gc.current_player()) % 4
                v = float(vs[off])
            v = apply_action_bonus(v, le[ci])
            values[p] = v

        # Mix value with Dirichlet noise for exploration
        mixed = (1.0 - dirichlet_frac) * values + dirichlet_frac * noise * 10.0

        # Softmax selection with temperature
        top_k_sel = min(5, len(candidates))
        top_idx = np.argpartition(mixed, -top_k_sel)[-top_k_sel:]
        top_v = mixed[top_idx]
        top_v -= top_v.max()
        t = max(temperature, 0.01)
        probs = np.exp(top_v / t)
        probs /= probs.sum()
        chosen_p = int(top_idx[np.random.choice(top_k_sel, p=probs)])
        best = candidates[chosen_p]
        return fix_robber_steal(best, le)

    def play_game(seed, temperature):
        game = CatanGame(seed=seed); game.reset()
        explore_setup = np.random.random() < explore_setup_frac
        steps = []

        while (not game.is_terminal()
               and game.turn_number < MAX_TURNS
               and len(steps) < MAX_STEPS):
            le = game.get_legal_actions()
            if not le: break

            sv = game.get_state_view()
            se.encode_into(sv, nf, ef, ff)
            mask_arr = ae.get_action_mask(le).numpy()

            # Save state copies BEFORE search (search overwrites nf/ef/ff)
            nf_snap = nf.copy()
            ef_snap = ef.copy()
            ff_snap = ff.copy()

            if len(le) == 1:
                chosen = 0
            elif game.turn_number <= 7:
                if explore_setup:
                    _, lo, mn = c_forward(game, le)
                    lo[mn[:AD] < 0.5] = -1e9
                    scores = lo[:AD]
                    scores -= scores.max()
                    probs = np.exp(scores / 2.0)
                    enc_ids = []
                    legal_idx = []
                    for i, a in enumerate(le):
                        try:
                            enc_ids.append(ae.encode(a))
                            legal_idx.append(i)
                        except ValueError:
                            continue
                    if legal_idx:
                        p = np.array([probs[e] for e in enc_ids])
                        p = np.maximum(p, 1e-8)
                        p /= p.sum()
                        chosen = legal_idx[np.random.choice(len(legal_idx), p=p)]
                    else:
                        chosen = 0
                else:
                    _, lo, mn = c_forward(game, le)
                    lo[mn[:AD] < 0.5] = -1e9
                    chosen = next((i for i, a in enumerate(le)
                                   if ae.encode(a) == int(np.argmax(lo[:AD]))), 0)
            elif search_depth > 1:
                chosen = nnt_search(game, le, search_depth, temperature)
            else:
                _, lo, mn = c_forward(game, le)
                lo[mn[:AD] < 0.5] = -1e9
                chosen = next((i for i, a in enumerate(le)
                               if ae.encode(a) == int(np.argmax(lo[:AD]))), 0)

            try:
                enc_action = ae.encode(le[chosen])
            except ValueError:
                game.step(chosen)
                continue

            steps.append({
                "nf": nf_snap, "ef": ef_snap, "ff": ff_snap,
                "mask": mask_arr.copy(),
                "action_idx": enc_action, "player": game.current_player(),
            })
            game.step(chosen)

        winner = game.winner()
        reward_vec = np.zeros(4, dtype=np.float32)
        if winner is not None:
            reward_vec[winner] = 1.0
            # Speed bonus: fast wins get higher reward, slow wins get less.
            # Typical game is 150-300 turns. Scale winner reward by speed.
            turns = game.turn_number
            speed_bonus = max(0.0, min(0.5, (300 - turns) / 300.0))
            reward_vec[winner] = 1.0 + speed_bonus  # 1.0-1.5 for winner

            # VP-graded losers: closer to winning = less negative signal
            for seat in range(4):
                if seat == winner:
                    continue
                vp = game._game.state.player_state[seat][0]
                reward_vec[seat] = vp / 20.0  # 0.0-0.5 range for losers
        sw = compute_step_weights(steps, reward_vec)
        return steps, reward_vec, sw, winner

    # ── Actor main loop ──────────────────────────────────────────
    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    weights_mtime = os.path.getmtime(weights_path) if os.path.exists(weights_path) else 0
    round_file = os.path.join(ckpt_dir, ".round")
    current_round = 0
    if os.path.exists(round_file):
        try:
            current_round = int(open(round_file).read().strip())
        except Exception:
            pass

    game_batch = []
    shard_idx = 0
    total_games = 0
    total_steps = 0
    wins = np.zeros(4, dtype=np.int64)
    t_start = time.time()

    print(f"[c_actor {actor_id}] Started, depth={search_depth}, "
          f"seed_base={seed_base}", flush=True)

    while True:
        seed = seed_base + total_games
        temp = temperature_for_round(current_round)
        steps, rv, sw, winner = play_game(seed, temp)

        game_batch.append((steps, rv, sw))
        total_games += 1
        total_steps += len(steps)
        if winner is not None:
            wins[winner] += 1

        if len(game_batch) >= GAMES_PER_SHARD:
            sid = f"ca{actor_id:03d}_{shard_idx:06d}"
            save_shard(game_batch, pending_dir, sid)
            game_batch = []
            shard_idx += 1

            # Backpressure
            while True:
                n_pending = len([f for f in os.listdir(pending_dir)
                                 if f.endswith(".pt") and not f.endswith(".tmp")])
                if n_pending <= max_pending:
                    break
                time.sleep(2)

        if total_games % 10 == 0:
            elapsed = time.time() - t_start
            gps = total_games / elapsed if elapsed > 0 else 0
            avg_s = total_steps / total_games if total_games else 0
            print(f"[c_actor {actor_id}] {total_games} games, "
                  f"{shard_idx} shards, {gps:.2f} g/s, "
                  f"~{avg_s:.0f} steps/g, t={temp:.2f}, "
                  f"wins={wins.tolist()}", flush=True)

            # Write per-actor stats for the learner to aggregate
            stats_dir = os.path.join(ckpt_dir, ".actor_stats")
            os.makedirs(stats_dir, exist_ok=True)
            stats_path = os.path.join(stats_dir, f"actor_{actor_id:03d}.json")
            try:
                with open(stats_path + ".tmp", "w") as f:
                    json.dump({"games": total_games, "elapsed": elapsed,
                               "gps": round(gps, 3), "steps": total_steps,
                               "wins": wins.tolist()}, f)
                os.rename(stats_path + ".tmp", stats_path)
            except Exception:
                pass

        # Reload C weights if updated
        if total_games % 50 == 0:
            try:
                mt = os.path.getmtime(weights_path)
                if mt > weights_mtime:
                    nn_lib.nn_load(mptr, weights_path.encode())
                    weights_mtime = mt
                    if os.path.exists(round_file):
                        try:
                            current_round = int(open(round_file).read().strip())
                        except Exception:
                            pass
                    print(f"[c_actor {actor_id}] Reloaded weights, "
                          f"round={current_round}, t={temperature_for_round(current_round):.2f}",
                          flush=True)
            except Exception:
                pass

    if game_batch:
        sid = f"ca{actor_id:03d}_{shard_idx:06d}"
        save_shard(game_batch, pending_dir, sid)


# =====================================================================
# GPU Learner (PyTorch, same as selfplay.py but exports C weights)
# =====================================================================

def run_learner(checkpoint_path, weights_bin_path, shard_dir, ckpt_dir,
                batch_size, shards_per_train, eval_games, eval_interval,
                wandb_name=None):
    try:
        _run_learner(checkpoint_path, weights_bin_path, shard_dir, ckpt_dir,
                     batch_size, shards_per_train, eval_games, eval_interval,
                     wandb_name)
    except Exception:
        print("!!! [learner] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_learner(checkpoint_path, weights_bin_path, shard_dir, ckpt_dir,
                 batch_size, shards_per_train, eval_games, eval_interval,
                 wandb_name):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import torch.nn as nn
    from human_bot.model import HumanBotNet
    from human_bot.config import HumanBotTrainingConfig
    from human_bot.loss import (UncertaintyWeightedLoss, human_policy_loss,
                                value_loss, masked_entropy)
    from human_bot.train import build_cosine_scheduler
    from human_bot.export_nn import export as export_nn

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = HumanBotNet.load_checkpoint(checkpoint_path, device=device)
    net.train()
    print(f"[learner] Model: {net.num_parameters:,} params on {device}", flush=True)

    loss_combiner = UncertaintyWeightedLoss().to(device)
    cfg = HumanBotTrainingConfig(
        batch_size=batch_size, epochs=1,
        freeze_encoder_epochs=0, label_smoothing=0.05, entropy_weight=0.01,
    )

    from hexzero.game.interface import CatanGame
    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    edge_index = se._edge_index.to(device)

    # W&B
    wandb_run = None
    try:
        import wandb
        if "WANDB_API_KEY" in os.environ:
            run_name = wandb_name or f"c-sp-{time.strftime('%m%d-%H%M')}"
            wandb_run = wandb.init(project="human-bot-selfplay", name=run_name)
            print(f"[learner] W&B: {wandb_run.url}", flush=True)
    except Exception as e:
        print(f"[learner] W&B init failed: {e}", flush=True)

    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    round_file = os.path.join(ckpt_dir, ".round")
    round_num = 0
    total_examples = 0
    best_wr = 0.0
    t_start = time.time()

    # Export initial C weights
    ckpt_tmp = os.path.join(ckpt_dir, "_export_tmp.pt")
    net.save_checkpoint(ckpt_tmp, {"round": 0})
    export_nn(ckpt_tmp, weights_bin_path)
    with open(round_file, "w") as f:
        f.write("0")
    print(f"[learner] Initial C weights exported to {weights_bin_path}", flush=True)

    while True:
        shards = sorted(
            f for f in os.listdir(pending_dir)
            if f.endswith(".pt") and not f.endswith(".tmp")
        )
        if len(shards) < shards_per_train:
            time.sleep(5)
            continue

        n_pending = len(shards)
        group = shards[:shards_per_train]
        t_round_start = time.time()

        # Load shard group
        t_load_start = time.time()
        nfs, efs, ffs, masks, acts, players, rvs, sws = (
            [], [], [], [], [], [], [], [])
        for fn in group:
            path = os.path.join(pending_dir, fn)
            try:
                d = torch.load(path, weights_only=False, map_location="cpu")
            except Exception as e:
                print(f"[learner] Bad shard {fn}: {e}", flush=True)
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                continue

            p = d["player"].numpy()
            rv = d["reward_vec"].numpy()
            S = p.shape[0]

            # Use reward_vec directly as soft value target (normalized).
            # Winner gets 1.0-1.5, losers get VP/20 (0.0-0.5).
            # Normalize per row to a valid distribution.
            rv_safe = np.maximum(rv, 0.0)
            row_sums = rv_safe.sum(axis=1, keepdims=True)
            no_winner = (row_sums < 1e-8).squeeze()
            vt = np.where(row_sums > 1e-8, rv_safe / row_sums, 0.25)
            vt[no_winner] = 0.25
            # Rotate to current-player perspective
            shifts = (-p % 4).astype(np.int32)
            idx_arr = (np.arange(4)[None, :] + shifts[:, None]) % 4
            vt = np.take_along_axis(vt, idx_arr, axis=1)

            m = d["action_mask"]
            if m.shape[-1] < MASK_DIM:
                m = torch.cat([m, torch.zeros(S, MASK_DIM - m.shape[-1],
                               dtype=m.dtype)], dim=-1)

            nfs.append(d["node_features"])
            efs.append(d["edge_features"])
            ffs.append(d["flat_features"])
            masks.append(m)
            acts.append(d["action_idx"])
            players.append(d["player"])
            rvs.append(torch.from_numpy(vt))
            if "step_weight" in d:
                sws.append(d["step_weight"])
            else:
                is_w = (vt[:, 0] > 0.5).astype(np.float32)
                sws.append(torch.from_numpy(1.0 + 0.5 * is_w))

        if not nfs:
            for fn in group:
                try:
                    os.remove(os.path.join(pending_dir, fn))
                except FileNotFoundError:
                    pass
            continue

        t_load = time.time() - t_load_start

        t_train_start = time.time()
        all_nf = torch.cat(nfs).to(device, non_blocking=True)
        all_ef = torch.cat(efs).to(device, non_blocking=True)
        all_ff = torch.cat(ffs).to(device, non_blocking=True)
        all_mask = torch.cat(masks).to(device, non_blocking=True)
        all_act = torch.cat(acts).to(device, non_blocking=True)
        all_vt = torch.cat(rvs).to(device, non_blocking=True)
        all_sw = torch.cat(sws).to(device, non_blocking=True)

        n = all_nf.shape[0]
        total_examples += n

        # Train
        net.train()
        all_params = list(net.parameters()) + list(loss_combiner.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-4)
        n_steps = max(1, n // cfg.batch_size)
        scheduler = build_cosine_scheduler(optimizer, n_steps, min(50, n_steps))

        perm = torch.randperm(n, device=device)
        BS = cfg.batch_size
        avg = {"policy_loss": 0, "value_loss": 0, "policy_acc": 0,
               "value_acc": 0, "entropy": 0}
        n_batches = 0

        for bi in range(0, n, BS):
            idx = perm[bi:bi+BS]
            if len(idx) < 16:
                continue

            nf_b = all_nf[idx]
            ef_b = all_ef[idx]
            ff_b = all_ff[idx]
            mask_b = all_mask[idx]
            act_b = all_act[idx]
            vt_b = all_vt[idx]
            sw_b = all_sw[idx]

            out = net({
                "node_features": nf_b, "edge_index": edge_index,
                "edge_features": ef_b, "flat_features": ff_b,
                "action_mask": mask_b,
            })

            p_loss = human_policy_loss(
                out["policy_logits"], act_b, mask_b,
                label_smoothing=cfg.label_smoothing, winner_boost=sw_b)
            tp = ff_b[:, 114]
            v_loss = value_loss(out["value"], vt_b, turn_progress=tp)
            ent = masked_entropy(out["policy_logits"], mask_b)
            total_loss, _ = loss_combiner(p_loss, v_loss, ent, cfg.entropy_weight)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pred = out["policy_logits"].argmax(dim=-1)
                pacc = (pred == act_b).float().mean().item()
                vacc = (out["value"].argmax(dim=-1) == vt_b.argmax(dim=-1)).float().mean().item()
            avg["policy_loss"] += p_loss.item()
            avg["value_loss"] += v_loss.item()
            avg["policy_acc"] += pacc
            avg["value_acc"] += vacc
            avg["entropy"] += ent.item()
            n_batches += 1

        for k in avg:
            avg[k] /= max(n_batches, 1)

        t_train = time.time() - t_train_start

        # Cleanup GPU
        del all_nf, all_ef, all_ff, all_mask, all_act, all_vt, all_sw
        gc_mod.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Delete consumed shards (no need to keep them)
        for fn in group:
            src = os.path.join(pending_dir, fn)
            try:
                os.remove(src)
            except FileNotFoundError:
                pass

        round_num += 1

        # Save PyTorch checkpoint
        ckpt_path = os.path.join(ckpt_dir, "latest.pt")
        net.save_checkpoint(ckpt_path + ".tmp", {
            "round": round_num, "total_examples": total_examples,
            "best_wr": best_wr, **avg,
        })
        os.rename(ckpt_path + ".tmp", ckpt_path)

        # Export C weights for actors
        t_export_start = time.time()
        export_nn(ckpt_path, weights_bin_path)
        with open(round_file, "w") as f:
            f.write(str(round_num))
        t_export = time.time() - t_export_start

        t_round = time.time() - t_round_start
        elapsed = time.time() - t_start
        temp = temperature_for_round(round_num)
        print(f"[learner] Round {round_num}: {n:,} ex  "
              f"pacc={avg['policy_acc']:.3f}  vacc={avg['value_acc']:.3f}  "
              f"ploss={avg['policy_loss']:.3f}  vloss={avg['value_loss']:.3f}  "
              f"ent={avg['entropy']:.3f}  t={temp:.2f}  "
              f"total={total_examples:,}  ({elapsed:.0f}s)  "
              f"[load={t_load:.1f}s train={t_train:.1f}s export={t_export:.1f}s "
              f"round={t_round:.1f}s pending={n_pending}]", flush=True)

        if wandb_run:
            import wandb
            wandb.log({
                "train/policy_loss": avg["policy_loss"],
                "train/value_loss": avg["value_loss"],
                "train/policy_acc": avg["policy_acc"],
                "train/value_acc": avg["value_acc"],
                "train/entropy": avg["entropy"],
                "train/round": round_num,
                "train/total_examples": total_examples,
                "train/temperature": temp,
                "perf/round_sec": t_round,
                "perf/load_sec": t_load,
                "perf/train_sec": t_train,
                "perf/export_sec": t_export,
                "perf/pending_shards": n_pending,
                "perf/examples_per_sec": n / max(t_train, 0.01),
            })

        # Aggregate actor stats for W&B
        if wandb_run:
            stats_dir = os.path.join(ckpt_dir, ".actor_stats")
            if os.path.isdir(stats_dir):
                total_gps = 0.0
                n_actors_reporting = 0
                total_actor_games = 0
                for fn in os.listdir(stats_dir):
                    if not fn.endswith(".json"):
                        continue
                    try:
                        with open(os.path.join(stats_dir, fn)) as f:
                            s = json.load(f)
                        total_gps += s.get("gps", 0)
                        total_actor_games += s.get("games", 0)
                        n_actors_reporting += 1
                    except Exception:
                        continue
                if n_actors_reporting > 0:
                    wandb.log({
                        "actors/total_gps": total_gps,
                        "actors/num_reporting": n_actors_reporting,
                        "actors/total_games": total_actor_games,
                        "actors/avg_gps": total_gps / n_actors_reporting,
                    })

        # Periodic eval: 0-ply and 5-ply vs AB2, plus h2h vs past checkpoints
        if eval_games > 0 and round_num % eval_interval == 0:
            net.eval()
            try:
                from human_bot.eval_search import evaluate_search_vs_ab2
                from hexzero.encoder.action_encoder import ActionEncoder
                from hexzero.bindings.lib_loader import load_library
                lib = load_library()
                ae = ActionEncoder()

                # vs AB2 at 0-ply and 5-ply
                for depth in [0, 5]:
                    r = evaluate_search_vs_ab2(
                        net, se, ae, device, lib,
                        num_games=eval_games, search_depth=depth,
                        seed_offset=round_num * 100 + depth * 1000)
                    wr = r["win_rate"]
                    print(f"[learner] Eval {depth}-ply vs AB2: NN={r['hz_wins']} "
                          f"AB2={r['ab2_wins']} WR={wr:.1%}", flush=True)
                    if wandb_run:
                        wandb.log({
                            f"eval/{depth}ply_win_rate": wr,
                            f"eval/{depth}ply_nn_wins": r["hz_wins"],
                            f"eval/{depth}ply_ab2_wins": r["ab2_wins"],
                        })
                    if depth == 5 and wr > best_wr:
                        best_wr = wr
                        best_path = os.path.join(ckpt_dir, "best.pt")
                        net.save_checkpoint(best_path, {
                            "round": round_num, "5ply_wr": wr,
                            "total_examples": total_examples,
                        })

                # Head-to-head vs past 2 versions of itself
                prev_dir = os.path.join(ckpt_dir, "prev_checkpoints")
                os.makedirs(prev_dir, exist_ok=True)

                # Save current as a historical checkpoint
                hist_path = os.path.join(prev_dir, f"round_{round_num:04d}.pt")
                net.save_checkpoint(hist_path, {"round": round_num})

                # Find 2 most recent previous checkpoints
                prev_ckpts = sorted(
                    [f for f in os.listdir(prev_dir)
                     if f.startswith("round_") and f.endswith(".pt")
                     and f != os.path.basename(hist_path)],
                    reverse=True,
                )[:2]

                for prev_fn in prev_ckpts:
                    prev_path = os.path.join(prev_dir, prev_fn)
                    prev_round = prev_fn.replace("round_", "").replace(".pt", "")
                    try:
                        from human_bot.model import HumanBotNet as _HBN
                        prev_net = _HBN.load_checkpoint(prev_path, device=device)
                        prev_net.eval()

                        h2h_wins = {"current": 0, "prev": 0, "draw": 0}
                        for gi in range(eval_games):
                            from hexzero.game.interface import CatanGame
                            game = CatanGame(seed=80000 + round_num * 100 + gi)
                            game.reset()
                            cur_seats = {gi % 4, (gi + 2) % 4}
                            while not game.is_terminal() and game.turn_number < 1000:
                                le = game.get_legal_actions()
                                if not le: break
                                if len(le) == 1:
                                    game.step(0)
                                    continue
                                cp = game.current_player()
                                playing_net = net if cp in cur_seats else prev_net
                                nf_e = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
                                ef_e = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
                                ff_e = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
                                se.encode_into(game.get_state_view(), nf_e, ef_e, ff_e)
                                mk_e = ae.get_action_mask(le).numpy()
                                mk_full = np.zeros(MASK_DIM, dtype=np.float32)
                                mk_full[:len(mk_e)] = mk_e
                                with torch.no_grad():
                                    out = playing_net({
                                        "node_features": torch.from_numpy(nf_e[None]).to(device),
                                        "edge_index": edge_index,
                                        "edge_features": torch.from_numpy(ef_e[None]).to(device),
                                        "flat_features": torch.from_numpy(ff_e[None]).to(device),
                                        "action_mask": torch.from_numpy(mk_full[None]).to(device),
                                    })
                                logits = out["policy_logits"][0, :AD].cpu().numpy()
                                logits[mk_e < 0.5] = -1e9
                                best_enc = int(np.argmax(logits))
                                chosen = next((i for i, a in enumerate(le)
                                               if ae.encode(a) == best_enc), 0)
                                game.step(chosen)
                            w = game.winner()
                            if w is None:
                                h2h_wins["draw"] += 1
                            elif w in cur_seats:
                                h2h_wins["current"] += 1
                            else:
                                h2h_wins["prev"] += 1

                        cur_wr = h2h_wins["current"] / max(1, h2h_wins["current"] + h2h_wins["prev"])
                        print(f"[learner] H2H vs round {prev_round}: "
                              f"current={h2h_wins['current']} prev={h2h_wins['prev']} "
                              f"draw={h2h_wins['draw']} WR={cur_wr:.1%}", flush=True)
                        if wandb_run:
                            wandb.log({
                                f"h2h/vs_round_{prev_round}_wr": cur_wr,
                                f"h2h/vs_round_{prev_round}_wins": h2h_wins["current"],
                                f"h2h/vs_round_{prev_round}_losses": h2h_wins["prev"],
                            })
                        del prev_net
                    except Exception as e:
                        print(f"[learner] H2H vs {prev_fn} failed: {e}", flush=True)

                # Cleanup old historical checkpoints (keep last 5)
                all_hist = sorted(
                    [f for f in os.listdir(prev_dir)
                     if f.startswith("round_") and f.endswith(".pt")],
                    reverse=True,
                )
                for old_fn in all_hist[5:]:
                    try:
                        os.remove(os.path.join(prev_dir, old_fn))
                    except Exception:
                        pass

            except Exception as e:
                print(f"[learner] Eval failed: {e}", flush=True)
                traceback.print_exc()
            net.train()


# =====================================================================
# Main: launch CPU actors + GPU learner
# =====================================================================

def _run_actors_only(args, weights_bin):
    """Launch actors only (no learner). For scaling across multiple nodes."""
    ctx = mp.get_context("spawn")
    actors = []
    per_actor_games = 1_000_000
    for i in range(args.num_actors):
        aid = args.actor_id_offset + i
        seed_base = args.seed + aid * per_actor_games
        p = ctx.Process(
            target=run_actor,
            args=(aid, weights_bin, args.shard_dir, args.ckpt_dir,
                  args.search_depth, seed_base, args.max_pending,
                  args.explore_setup, args.dirichlet_alpha,
                  args.dirichlet_frac),
            daemon=True,
        )
        p.start()
        actors.append(p)
    print(f"[main] {len(actors)} actors started (actors-only mode, "
          f"ids {args.actor_id_offset}..{args.actor_id_offset + args.num_actors - 1})",
          flush=True)

    try:
        while True:
            time.sleep(30)
            alive = sum(1 for p in actors if p.is_alive())
            if alive == 0:
                print("[main] All actors died", flush=True)
                break
            if alive < len(actors):
                print(f"[main] {len(actors) - alive}/{len(actors)} actors died, "
                      f"{alive} still running", flush=True)
    except KeyboardInterrupt:
        print("[main] Interrupted", flush=True)

    for p in actors:
        if p.is_alive():
            p.terminate()


def main():
    parser = argparse.ArgumentParser(description="C-inference self-play")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Initial PyTorch checkpoint")
    parser.add_argument("--weights-bin", type=str, default=None,
                        help="C weights path (auto-derived if omitted)")
    parser.add_argument("--role", choices=["all", "learner", "actors"],
                        default="all",
                        help="all=learner+actors, learner=GPU only, actors=CPU only")
    parser.add_argument("--num-actors", type=int, default=40)
    parser.add_argument("--actor-id-offset", type=int, default=0,
                        help="Starting actor ID (use different offsets per node)")
    parser.add_argument("--search-depth", type=int, default=10)
    parser.add_argument("--shard-dir", type=str, default="data/c_selfplay")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/c_selfplay")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--shards-per-train", type=int, default=20)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=4)
    parser.add_argument("--max-pending", type=int, default=MAX_PENDING)
    parser.add_argument("--seed", type=int, default=500000)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--explore-setup", type=float, default=0.2,
                        help="Fraction of games with random setup exploration")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-frac", type=float, default=0.25)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(args.shard_dir, "pending"), exist_ok=True)

    weights_bin = args.weights_bin
    if weights_bin is None:
        weights_bin = os.path.join(args.ckpt_dir, "nn_weights_latest.bin")

    print(f"C-inference self-play (role={args.role})", flush=True)
    print(f"  Checkpoint:  {args.checkpoint}", flush=True)
    print(f"  C weights:   {weights_bin}", flush=True)
    print(f"  Shards:      {args.shard_dir}", flush=True)
    print(f"  Checkpoints: {args.ckpt_dir}", flush=True)

    # ── Actors-only mode ─────────────────────────────────────────
    if args.role == "actors":
        print(f"  Actors:      {args.num_actors} (CPU, depth={args.search_depth}, "
              f"offset={args.actor_id_offset})", flush=True)
        # Wait for C weights to exist (learner on another node creates them)
        print("[main] Waiting for C weights...", flush=True)
        for _ in range(600):
            if os.path.exists(weights_bin):
                break
            time.sleep(1)
        else:
            print(f"[main] ERROR: {weights_bin} not found after 600s", flush=True)
            return
        time.sleep(2)
        _run_actors_only(args, weights_bin)
        return

    # ── Learner-only mode ────────────────────────────────────────
    if args.role == "learner":
        print(f"  Learner:     GPU (batch={args.batch_size})", flush=True)
        run_learner(args.checkpoint, weights_bin, args.shard_dir, args.ckpt_dir,
                    args.batch_size, args.shards_per_train,
                    args.eval_games, args.eval_interval, args.wandb_name)
        return

    # ── All mode (learner + actors on same node) ─────────────────
    print(f"  Actors:      {args.num_actors} (CPU, depth={args.search_depth})", flush=True)
    print(f"  Learner:     GPU (batch={args.batch_size})", flush=True)
    print(f"  Exploration: setup={args.explore_setup}, "
          f"dirichlet(a={args.dirichlet_alpha}, f={args.dirichlet_frac})", flush=True)

    ctx = mp.get_context("spawn")

    learner = ctx.Process(
        target=run_learner,
        args=(args.checkpoint, weights_bin, args.shard_dir, args.ckpt_dir,
              args.batch_size, args.shards_per_train,
              args.eval_games, args.eval_interval, args.wandb_name),
        daemon=False,
    )
    learner.start()
    print(f"[main] Learner started (pid={learner.pid})", flush=True)

    print("[main] Waiting for initial C weights...", flush=True)
    for _ in range(120):
        if os.path.exists(weights_bin):
            break
        time.sleep(1)
    else:
        print("[main] ERROR: C weights not created after 120s", flush=True)
        learner.terminate()
        return

    time.sleep(2)
    print(f"[main] C weights ready: {weights_bin}", flush=True)

    actors = []
    per_actor_games = 1_000_000
    for i in range(args.num_actors):
        aid = args.actor_id_offset + i
        seed_base = args.seed + aid * per_actor_games
        p = ctx.Process(
            target=run_actor,
            args=(aid, weights_bin, args.shard_dir, args.ckpt_dir,
                  args.search_depth, seed_base, args.max_pending,
                  args.explore_setup, args.dirichlet_alpha,
                  args.dirichlet_frac),
            daemon=True,
        )
        p.start()
        actors.append(p)
    print(f"[main] {len(actors)} actors started", flush=True)

    try:
        while True:
            time.sleep(30)
            alive = sum(1 for p in actors if p.is_alive())
            if not learner.is_alive():
                print("[main] Learner died, shutting down", flush=True)
                break
            if alive == 0:
                print("[main] All actors died, shutting down", flush=True)
                break
            if alive < len(actors):
                dead = len(actors) - alive
                print(f"[main] {dead}/{len(actors)} actors died, "
                      f"{alive} still running", flush=True)
    except KeyboardInterrupt:
        print("[main] Interrupted", flush=True)

    for p in actors:
        if p.is_alive():
            p.terminate()
    if learner.is_alive():
        learner.terminate()
        learner.join(timeout=10)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
