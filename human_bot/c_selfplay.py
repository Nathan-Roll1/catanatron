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
    all_np = []
    for game in games_data:
        if len(game) == 4:
            steps, rv, sw, n_players = game
        else:
            steps, rv, sw = game
            n_players = 4  # legacy: assume 4-player
        for i, s in enumerate(steps):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_sw.append(sw[i])
            all_np.append(n_players)
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
        "num_players": torch.tensor(all_np, dtype=torch.int64),
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
              dirichlet_frac=0.25, player_counts=(2, 3, 4)):
    try:
        _run_actor(actor_id, weights_path, shard_dir, ckpt_dir,
                   search_depth, seed_base, max_pending,
                   explore_setup_frac, dirichlet_alpha, dirichlet_frac,
                   player_counts)
    except Exception:
        print(f"!!! [c_actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_actor(actor_id, weights_path, shard_dir, ckpt_dir,
               search_depth, seed_base, max_pending,
               explore_setup_frac, dirichlet_alpha, dirichlet_frac,
               player_counts):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    lib = load_library()
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    pc_arr = np.asarray(player_counts, dtype=np.int64)
    actor_rng = np.random.default_rng((seed_base + actor_id) ^ 0xC0DE)
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

    def nnt_search(game, le, depth, temperature=1.0, top_k=2):
        """Returns (target_action, played_action).

        target_action = search argmax on clean values (what we train on).
        played_action = temperature-sampled (for game-trajectory exploration).
        Uses AB2 base_value_fn as leaf evaluator (external oracle, not the NN
        value head) so the search provides signal strictly better than policy.
        """
        seat = game.current_player()
        candidates = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            candidates = c_topk(game, le, top_k)

        values = np.zeros(len(candidates), dtype=np.float64)
        for p, ci in enumerate(candidates):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal(): break
                c_argmax(gc)
            cg = gc._game
            bot_color = cg.state.colors[seat]
            values[p] = float(lib.base_value_fn(ctypes.byref(cg), bot_color))
            values[p] += apply_action_bonus(0.0, le[ci])

        K = len(candidates)
        argmax_p = int(np.argmax(values))
        target = fix_robber_steal(candidates[argmax_p], le)

        if K == 1 or temperature < 0.01:
            played_p = argmax_p
        else:
            v_max = values.max()
            v_min = values.min()
            spread = max(1.0, v_max - v_min)
            normalized = (values - v_max) / spread
            probs = np.exp(normalized / max(0.01, temperature))
            probs /= probs.sum()
            played_p = int(np.random.choice(K, p=probs))
        played = fix_robber_steal(candidates[played_p], le)

        return target, played

    def play_game(seed, temperature, num_players):
        cfg = GameConfig(num_players=num_players)
        game = CatanGame(seed=seed, config=cfg); game.reset()
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

            target_action = None
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
                    target_action = chosen
                else:
                    _, lo, mn = c_forward(game, le)
                    lo[mn[:AD] < 0.5] = -1e9
                    chosen = next((i for i, a in enumerate(le)
                                   if ae.encode(a) == int(np.argmax(lo[:AD]))), 0)
                    target_action = chosen
            elif search_depth > 1:
                target_action, chosen = nnt_search(game, le, search_depth, temperature)
            else:
                _, lo, mn = c_forward(game, le)
                lo[mn[:AD] < 0.5] = -1e9
                chosen = next((i for i, a in enumerate(le)
                               if ae.encode(a) == int(np.argmax(lo[:AD]))), 0)
                target_action = chosen

            recorded = target_action if target_action is not None else chosen
            try:
                enc_action = ae.encode(le[recorded])
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
            turns = game.turn_number
            speed_bonus = max(0.0, min(0.5, (300 - turns) / 300.0))
            reward_vec[winner] = 1.0 + speed_bonus

            # VP-graded losers: closer to winning = less negative signal.
            # Iterate only over actual seats; unused seats stay at 0.
            for seat in range(num_players):
                if seat == winner:
                    continue
                vp = game._game.state.player_state[seat][0]
                reward_vec[seat] = vp / 20.0
        sw = compute_step_weights(steps, reward_vec)
        return steps, reward_vec, sw, winner

    # ── Actor main loop ──────────────────────────────────────────
    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    stop_file = os.path.join(ckpt_dir, ".stop")
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
    games_by_pc = {int(pc): 0 for pc in pc_arr}
    t_start = time.time()

    print(f"[c_actor {actor_id}] Started, depth={search_depth}, "
          f"seed_base={seed_base}, player_counts={list(pc_arr)}", flush=True)

    while not os.path.exists(stop_file):
        seed = seed_base + total_games
        temp = temperature_for_round(current_round)
        n_players = int(actor_rng.choice(pc_arr))
        games_by_pc[n_players] = games_by_pc.get(n_players, 0) + 1
        steps, rv, sw, winner = play_game(seed, temp, n_players)

        game_batch.append((steps, rv, sw, n_players))
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
            pc_summary = " ".join(f"{k}p={v}" for k, v in sorted(games_by_pc.items()))
            print(f"[c_actor {actor_id}] {total_games} games, "
                  f"{shard_idx} shards, {gps:.2f} g/s, "
                  f"~{avg_s:.0f} steps/g, t={temp:.2f}, "
                  f"wins={wins.tolist()} | {pc_summary}", flush=True)

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

    if os.path.exists(stop_file):
        print(f"[c_actor {actor_id}] Stop file detected, exiting gracefully "
              f"({total_games} games)", flush=True)


# =====================================================================
# GPU Learner (PyTorch, same as selfplay.py but exports C weights)
# =====================================================================

# --- Source-mix sampling ----------------------------------------------------
#
# The learner reads from a single pending dir fed by multiple actor types
# (ab2_stream, exit_gpu_actors, exit_vs_ab2_actors). Filename prefixes are
# stable and source-tagged, so we can classify without opening each shard.

SOURCE_PREFIXES = {
    "ab2":         ("ab2_", "ab2-", "w"),            # ab2_stream, collect_ab2_games (legacy "w{worker}_")
    "exit_vs_ab2": ("exit_vs_ab2_",),
    "exit":        ("exit_a", "exit_vs_ab2_"),       # exit_a matches exit_a### but NOT exit_vs_ab2_; resolved below
}


def _classify_source(fn: str) -> str:
    """Return one of 'ab2' | 'exit' | 'exit_vs_ab2' | 'other'."""
    if fn.startswith("exit_vs_ab2_"):
        return "exit_vs_ab2"
    if fn.startswith("exit_a"):
        return "exit"
    if fn.startswith("ab2_") or fn.startswith("ab2-") or (fn.startswith("w") and "_" in fn):
        return "ab2"
    # Legacy c_selfplay shards were "ca{actor:03d}_{idx:06d}.pt"
    if fn.startswith("ca"):
        return "exit"
    return "other"


def _pick_shard_group(pending_dir, shards_per_train, source_mix=None):
    """Choose up to `shards_per_train` shards from pending/ honoring the source mix.

    source_mix: dict like {"exit_vs_ab2": 0.6, "exit": 0.25, "ab2": 0.15}.
    Missing sources fall back to whatever is available (never block progress).
    Returns the selected filenames plus a dict of counts per source.
    """
    all_shards = sorted(
        f for f in os.listdir(pending_dir)
        if f.endswith(".pt") and not f.endswith(".tmp")
    )
    if len(all_shards) < shards_per_train:
        return None, None, len(all_shards)

    by_src = {"ab2": [], "exit": [], "exit_vs_ab2": [], "other": []}
    for fn in all_shards:
        by_src[_classify_source(fn)].append(fn)

    # Uniform fallback: simple sorted prefix (preserves prior behavior)
    if not source_mix:
        chosen = all_shards[:shards_per_train]
    else:
        import random
        chosen = []
        # How many we want per source
        targets = {
            s: int(round(source_mix.get(s, 0.0) * shards_per_train))
            for s in ("ab2", "exit", "exit_vs_ab2")
        }
        # Rounding slack: ensure sum equals shards_per_train (drop from largest)
        diff = shards_per_train - sum(targets.values())
        if diff != 0:
            sorted_srcs = sorted(targets.keys(), key=lambda s: -source_mix.get(s, 0.0))
            # adjust the most-weighted first
            targets[sorted_srcs[0]] = max(0, targets[sorted_srcs[0]] + diff)
        # Draw from each source; if a source is short, overflow to others
        rng = random.Random()
        shortfall = 0
        for s, k in targets.items():
            pool = by_src[s]
            if len(pool) >= k:
                chosen.extend(rng.sample(pool, k))
            else:
                chosen.extend(pool)
                shortfall += k - len(pool)
        # Fill shortfall from any remaining shards
        if shortfall > 0:
            remaining = [f for f in all_shards if f not in set(chosen)]
            if len(remaining) >= shortfall:
                chosen.extend(rng.sample(remaining, shortfall))
            else:
                chosen.extend(remaining)

    # Count actual sources picked
    counts = {"ab2": 0, "exit": 0, "exit_vs_ab2": 0, "other": 0}
    for fn in chosen:
        counts[_classify_source(fn)] += 1
    return chosen, counts, len(all_shards)


def run_learner(checkpoint_path, weights_bin_path, shard_dir, ckpt_dir,
                batch_size, shards_per_train, eval_games, eval_interval,
                wandb_name=None, source_mix=None):
    try:
        _run_learner(checkpoint_path, weights_bin_path, shard_dir, ckpt_dir,
                     batch_size, shards_per_train, eval_games, eval_interval,
                     wandb_name, source_mix)
    except Exception:
        print("!!! [learner] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_learner(checkpoint_path, weights_bin_path, shard_dir, ckpt_dir,
                 batch_size, shards_per_train, eval_games, eval_interval,
                 wandb_name, source_mix=None):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import torch.nn as nn
    from human_bot.model import HumanBotNet
    from human_bot.config import HumanBotTrainingConfig
    from human_bot.loss import (UncertaintyWeightedLoss, human_policy_loss,
                                value_loss, masked_entropy)
    from human_bot.export_nn import export as export_nn

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = HumanBotNet.load_checkpoint(checkpoint_path, device=device)
    net.train()
    print(f"[learner] Model: {net.num_parameters:,} params on {device}", flush=True)

    loss_combiner = UncertaintyWeightedLoss().to(device)
    safe_bs = min(batch_size, 4096)
    cfg = HumanBotTrainingConfig(
        batch_size=safe_bs, epochs=1,
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
            # Resume-by-id so learner restarts continue the same W&B graph
            # instead of making a fresh run every cancel/relaunch cycle.
            wandb_id_path = os.path.join(ckpt_dir, ".wandb_id")
            existing_id = None
            if os.path.exists(wandb_id_path):
                try:
                    existing_id = open(wandb_id_path).read().strip() or None
                except Exception:
                    existing_id = None
            init_kwargs = dict(project="human-bot-selfplay", name=run_name)
            if existing_id:
                init_kwargs["id"] = existing_id
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)
            # Persist for next restart
            if not existing_id and wandb_run is not None:
                try:
                    with open(wandb_id_path, "w") as f:
                        f.write(wandb_run.id)
                except Exception:
                    pass
            print(f"[learner] W&B: {wandb_run.url} "
                  f"{'(resumed)' if existing_id else '(new)'}", flush=True)
    except Exception as e:
        print(f"[learner] W&B init failed: {e}", flush=True)

    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    round_file = os.path.join(ckpt_dir, ".round")
    round_num = 0
    total_examples = 0
    best_wr = 0.0
    t_start = time.time()

    # Resume round/example counters from the starting checkpoint's metadata
    # if it has them. Avoids resetting actor temperature (via .round) and
    # best_wr on every restart of the learner.
    try:
        ckpt_blob = torch.load(checkpoint_path, map_location="cpu",
                               weights_only=False)
        meta = ckpt_blob.get("metadata", {}) if isinstance(ckpt_blob, dict) else {}
        # Newer save_checkpoint writes a top-level "metadata" dict; older
        # code paths stored keys directly on the blob. Handle both.
        if not meta and isinstance(ckpt_blob, dict):
            meta = {k: ckpt_blob[k] for k in ("round", "total_examples",
                                               "best_wr", "ableaf15_wr")
                    if k in ckpt_blob}
        round_num = int(meta.get("round", 0) or 0)
        total_examples = int(meta.get("total_examples", 0) or 0)
        # Prefer ableaf15_wr (new best-metric) then best_wr for continuity
        if "ableaf15_wr" in meta and meta["ableaf15_wr"]:
            best_wr = float(meta["ableaf15_wr"])
        elif "best_wr" in meta and meta["best_wr"]:
            best_wr = float(meta["best_wr"])
        if round_num > 0:
            print(f"[learner] Resumed from round={round_num} "
                  f"total_examples={total_examples:,} best_wr={best_wr:.1%}",
                  flush=True)
    except Exception as e:
        print(f"[learner] Could not resume counters from checkpoint: {e}",
              flush=True)

    # Export initial C weights
    ckpt_tmp = os.path.join(ckpt_dir, "_export_tmp.pt")
    net.save_checkpoint(ckpt_tmp, {"round": round_num})
    export_nn(ckpt_tmp, weights_bin_path)
    # Seed the .round file from the resumed counter so actors keep their
    # temperature schedule consistent across restarts.
    with open(round_file, "w") as f:
        f.write(str(round_num))
    print(f"[learner] Initial C weights exported to {weights_bin_path}", flush=True)

    stop_file = os.path.join(ckpt_dir, ".stop")
    while not os.path.exists(stop_file):
        group, source_counts, n_pending = _pick_shard_group(
            pending_dir, shards_per_train, source_mix=source_mix)
        if group is None:
            time.sleep(5)
            continue
        if source_mix:
            # Loud warning when a target source produced 0 shards this
            # round. The sampler silently overflows from other sources
            # when one is empty; without this print, an offline actor
            # type can quietly disappear from the training mix.
            missing_sources = [
                s for s, w in source_mix.items()
                if w > 0 and (source_counts or {}).get(s, 0) == 0
            ]
            warn_suffix = (f"  ⚠ MISSING: {missing_sources}"
                           if missing_sources else "")
            print(f"[learner] shard mix this round: {source_counts} "
                  f"(target={source_mix}){warn_suffix}", flush=True)
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
            # Rotate absolute-seat targets → cp-relative slots matching encoder
            # (slot 0 = cp, slot i = seat (cp+i) % N); zero slots beyond N.
            from human_bot.dataset import rotate_value_targets_to_cp
            n_p_tensor = d.get("num_players")
            n_p_arr = n_p_tensor.numpy() if n_p_tensor is not None else None
            vt = rotate_value_targets_to_cp(vt, p, n_p_arr)

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
            # Prefer new policy_weight (action-type-only, no winner-boost).
            # Fall back to legacy step_weight (may include winner-boost from
            # older shards) and finally to a degenerate weight.
            if "policy_weight" in d:
                sws.append(d["policy_weight"])
            elif "step_weight" in d:
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
        peak_lr = 1e-5
        optimizer = torch.optim.AdamW(all_params, lr=peak_lr, weight_decay=1e-4)

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
                # Per-source shard counts consumed this round (for
                # verifying the intended mix is actually happening).
                "sources/ab2": (source_counts or {}).get("ab2", 0),
                "sources/exit": (source_counts or {}).get("exit", 0),
                "sources/exit_vs_ab2": (source_counts or {}).get("exit_vs_ab2", 0),
                "sources/other": (source_counts or {}).get("other", 0),
                "train/lr": float(peak_lr),
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

        # Periodic eval vs AB2 (reliable external metrics):
        #   1) 0-ply policy argmax vs proper AB2 (fast, pure-policy signal)
        #   2) ABt15 top-k=2 with AB-leaf search vs proper AB2
        #      (our strongest play config; does NOT depend on the NN
        #      value head so the metric is stable during training)
        # Best-checkpoint selection is driven by AB-leaf search WR so that
        # we track the combined policy+search quality we actually deploy.
        if eval_games > 0 and round_num % eval_interval == 0:
            net.eval()
            try:
                from human_bot.eval_search import evaluate_search_vs_ab2
                from hexzero.encoder.action_encoder import ActionEncoder
                from hexzero.bindings.lib_loader import load_library
                lib = load_library()
                ae = ActionEncoder()

                # (label, search_depth, ab_value_leaf, num_games, log_prefix)
                eval_configs = [
                    ("0ply",     0,  False, eval_games,             "0ply"),
                    ("ableaf15", 15, True,  max(1, eval_games // 2), "ableaf15"),
                ]
                for label, depth, ab_leaf, ng, prefix in eval_configs:
                    r = evaluate_search_vs_ab2(
                        net, se, ae, device, lib,
                        num_games=ng, search_depth=depth,
                        seed_offset=round_num * 100 + hash(label) % 100,
                        ab_value_leaf=ab_leaf)
                    wr = r["win_rate"]
                    print(f"[learner] Eval {label} vs AB2: NN={r['hz_wins']} "
                          f"AB2={r['ab2_wins']} WR={wr:.1%}", flush=True)
                    if wandb_run:
                        wandb.log({
                            f"eval/{prefix}_win_rate": wr,
                            f"eval/{prefix}_nn_wins": r["hz_wins"],
                            f"eval/{prefix}_ab2_wins": r["ab2_wins"],
                        })
                    if label == "ableaf15" and wr > best_wr:
                        best_wr = wr
                        best_path = os.path.join(ckpt_dir, "best.pt")
                        net.save_checkpoint(best_path, {
                            "round": round_num, "ableaf15_wr": wr,
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
                  args.dirichlet_frac, args.player_counts),
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
    parser.add_argument("--player-counts", type=str, default="2,3,4",
                        help="Comma-separated player counts to sample uniformly "
                             "per game (e.g. '2,3,4' or '4').")
    parser.add_argument("--source-mix", type=str, default="",
                        help="Source-aware shard sampling weights "
                             "(format: 'exit_vs_ab2:0.6,exit:0.25,ab2:0.15'). "
                             "Empty = plain sorted-FIFO like before.")
    args = parser.parse_args()
    try:
        args.player_counts = tuple(int(x) for x in args.player_counts.split(",")
                                    if x.strip())
    except ValueError:
        parser.error(f"--player-counts must be comma-separated ints, "
                     f"got: {args.player_counts}")
    if not args.player_counts:
        parser.error("--player-counts cannot be empty")
    for n in args.player_counts:
        if n not in (2, 3, 4):
            parser.error(f"--player-counts values must be 2, 3 or 4 (got {n})")

    source_mix = None
    if args.source_mix:
        source_mix = {}
        for pair in args.source_mix.split(","):
            if not pair.strip(): continue
            if ":" not in pair:
                parser.error(f"--source-mix entry missing ':'  got: {pair!r}")
            k, v = pair.split(":", 1)
            k = k.strip()
            if k not in ("ab2", "exit", "exit_vs_ab2"):
                parser.error(f"--source-mix: unknown source '{k}' "
                             "(allowed: ab2, exit, exit_vs_ab2)")
            try:
                source_mix[k] = float(v)
            except ValueError:
                parser.error(f"--source-mix weight must be float, got: {v!r}")
        total = sum(source_mix.values()) or 1.0
        source_mix = {k: v / total for k, v in source_mix.items()}

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
    print(f"  Player cnts: {list(args.player_counts)} (uniform per game)", flush=True)

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

    if source_mix:
        print(f"  Source mix:  {source_mix}", flush=True)

    # ── Learner-only mode ────────────────────────────────────────
    if args.role == "learner":
        print(f"  Learner:     GPU (batch={args.batch_size})", flush=True)
        run_learner(args.checkpoint, weights_bin, args.shard_dir, args.ckpt_dir,
                    args.batch_size, args.shards_per_train,
                    args.eval_games, args.eval_interval, args.wandb_name,
                    source_mix=source_mix)
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
              args.eval_games, args.eval_interval, args.wandb_name,
              source_mix),
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
                  args.dirichlet_frac, args.player_counts),
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
