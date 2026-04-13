#!/usr/bin/env python3
"""Self-play improvement via Expert Iteration (ExIt).

Architecture (4 JAG GPUs + 48 CPUs):
  GPU 0         — Learner: consumes trajectory shards, trains, evaluates
  GPUs 1,2,3    — Actors:  play 4-seat self-play with 1-ply value search,
                           write trajectory shards to shared directory

Data flow:
  Actors write .pt shards → shard_dir/pending/
  Learner moves shards  → shard_dir/consumed/  after training
  Learner writes checkpoints → ckpt_dir/latest.pt (atomic rename)
  Actors reload latest.pt every RELOAD_INTERVAL games

Usage:
  # On cluster (all-in-one, 4 GPUs):
  python3 -u human_bot/selfplay.py \\
      --checkpoint checkpoints/cluster_run/final.pt \\
      --role all --num-actor-gpus 3

  # Separate processes:
  python3 -u human_bot/selfplay.py --role learner --gpu-id 0 ...
  python3 -u human_bot/selfplay.py --role actor  --gpu-id 1 ...
"""

from __future__ import annotations

import argparse
import gc
import os
import signal
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn

AD = 337
MASK_DIM = 397
GAMES_PER_SHARD = 25
MAX_TURNS = 1000
MAX_STEPS_PER_GAME = 2000
MAX_PENDING_SHARDS = 200

# Action-type weight ranges for step weighting (index → modifier)
_ACT_MOD = np.ones(MASK_DIM, dtype=np.float32)
_ACT_MOD[0] = 0.2          # ROLL
_ACT_MOD[1] = 0.5          # END_TURN
_ACT_MOD[2:5] = 1.5        # BUY_DEV, KNIGHT, ROAD_BUILDING
_ACT_MOD[5:113] = 1.5      # settlements, cities
_ACT_MOD[113:185] = 1.5    # roads
_ACT_MOD[185:280] = 1.5    # robber
_ACT_MOD[280:285] = 0.3    # discard
_ACT_MOD[285:310] = 1.5    # YoP, monopoly
_ACT_MOD[310:397] = 1.3    # maritime, trades


# ── Helpers ──────────────────────────────────────────────────────

def compute_step_weights(steps, reward_vec):
    """Per-step policy weights: upweight winner's decisive moves, downweight noise."""
    winner = int(np.argmax(reward_vec)) if reward_vec.max() > 0 else -1
    S = len(steps)
    weights = np.ones(S, dtype=np.float32)
    for i, s in enumerate(steps):
        progress = i / max(S - 1, 1)
        if s["player"] == winner:
            base = 1.0 + progress          # 1.0 early → 2.0 late
        else:
            base = max(0.2, 0.6 - 0.4 * progress)  # 0.6 early → 0.2 late
        weights[i] = base * _ACT_MOD[min(s["action_idx"], MASK_DIM - 1)]
    return weights


def atomic_torch_save(data, path):
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.rename(tmp, path)


def _setup_game_and_encoders(device):
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    ae = ActionEncoder()
    g = CatanGame(seed=0)
    g.reset()
    se = g.make_state_encoder()
    edge_index = se._edge_index.to(device)
    return lib, ae, se, edge_index


def _load_net(checkpoint_path, device):
    from human_bot.model import HumanBotNet
    net = HumanBotNet.load_checkpoint(checkpoint_path, device=device)
    net.eval()
    return net


# ── 1-ply value search ──────────────────────────────────────────

def _policy_argmax(nf, ef, ff, mask, net, ae, edge_index, device, le):
    mask_full = np.zeros(MASK_DIM, dtype=np.float32)
    mask_full[:len(mask)] = mask
    batch = {
        "node_features": torch.from_numpy(nf[None]).to(device),
        "edge_index": edge_index,
        "edge_features": torch.from_numpy(ef[None]).to(device),
        "flat_features": torch.from_numpy(ff[None]).to(device),
        "action_mask": torch.from_numpy(mask_full[None]).to(device),
    }
    with torch.no_grad():
        out = net(batch)
    logits = out["policy_logits"][0, :AD].cpu().numpy()
    enc_idx = int(np.argmax(logits))
    for i, a in enumerate(le):
        if ae.encode(a) == enc_idx:
            return i
    return 0


def _value_search_1ply(game, le, net, se, ae, edge_index, device,
                       temperature=1.0):
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    our_seat = game.current_player()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    n_cand = len(le)

    nf_b = np.zeros((n_cand, N, NF), dtype=np.float32)
    ef_b = np.zeros((n_cand, E, EF), dtype=np.float32)
    ff_b = np.zeros((n_cand, FF), dtype=np.float32)
    mask_b = np.zeros((n_cand, MASK_DIM), dtype=np.float32)
    is_term = np.zeros(n_cand, dtype=bool)
    term_vals = np.zeros(n_cand, dtype=np.float32)
    child_pl = np.zeros(n_cand, dtype=np.int32)

    for i in range(n_cand):
        child = game.clone()
        child.step(i)
        if child.is_terminal():
            is_term[i] = True
            w = child.winner()
            if w == our_seat:
                term_vals[i] = 10.0
            elif w is not None:
                term_vals[i] = -10.0
        else:
            se.encode_into(child.get_state_view(), nf_b[i], ef_b[i], ff_b[i])
            child_pl[i] = child.current_player()
            child_mask = ae.get_action_mask(child.get_legal_actions()).numpy()
            mask_b[i, :len(child_mask)] = child_mask

    values = term_vals.copy()
    non_term = ~is_term
    n_eval = int(non_term.sum())

    if n_eval > 0:
        batch = {
            "node_features": torch.from_numpy(nf_b[non_term]).to(device),
            "edge_index": edge_index,
            "edge_features": torch.from_numpy(ef_b[non_term]).to(device),
            "flat_features": torch.from_numpy(ff_b[non_term]).to(device),
            "action_mask": torch.from_numpy(mask_b[non_term]).to(device),
        }
        with torch.no_grad():
            out = net(batch)
        v4 = out["value"].cpu().numpy()
        j = 0
        for i in range(n_cand):
            if non_term[i]:
                offset = (our_seat - child_pl[i]) % 4
                values[i] = float(v4[j, offset])
                j += 1

    for i in range(n_cand):
        values[i] = apply_action_bonus(values[i], le[i])

    top_k = min(5, n_cand)
    top_idx = np.argpartition(values, -top_k)[-top_k:]
    top_vals = values[top_idx]
    top_vals -= top_vals.max()
    t = max(temperature, 0.01)
    probs = np.exp(top_vals / t)
    probs /= probs.sum()
    best = int(top_idx[np.random.choice(top_k, p=probs)])

    best = fix_robber_steal(best, le)
    return best


# ── Multi-ply tapering search ────────────────────────────────────

TAPER_WIDTHS = [5, 3, 2, 2, 2, 2, 2, 2, 2, 2]

AT_MOVE_ROBBER = 1
AT_BUILD_ROAD = 3
AT_BUILD_SETTLEMENT = 4
AT_BUILD_CITY = 5
AT_BUY_DEV = 6
_IMPORTANT_TYPES = frozenset({AT_MOVE_ROBBER, AT_BUILD_ROAD, AT_BUILD_SETTLEMENT,
                               AT_BUILD_CITY, AT_BUY_DEV})


def _is_important_position(le):
    """True if any legal action is a build, buy dev, or robber move."""
    return any(a.type in _IMPORTANT_TYPES for a in le)


def _policy_top_k_indices(game, le, net, se, ae, edge_index, device, k):
    """Return up to k legal action indices ranked by policy logits."""
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FF, dtype=np.float32)
    se.encode_into(game.get_state_view(), nf, ef, ff)
    mask_np = ae.get_action_mask(le).numpy()
    mask_full = np.zeros(MASK_DIM, dtype=np.float32)
    mask_full[:len(mask_np)] = mask_np

    batch = {
        "node_features": torch.from_numpy(nf[None]).to(device),
        "edge_index": edge_index,
        "edge_features": torch.from_numpy(ef[None]).to(device),
        "flat_features": torch.from_numpy(ff[None]).to(device),
        "action_mask": torch.from_numpy(mask_full[None]).to(device),
    }
    with torch.no_grad():
        out = net(batch)
    logits = out["policy_logits"][0, :AD].cpu().numpy()

    enc_to_legal = {}
    for i, a in enumerate(le):
        try:
            enc_to_legal[ae.encode(a)] = i
        except ValueError:
            continue

    scored = [(logits[enc], li) for enc, li in enc_to_legal.items()]
    scored.sort(key=lambda x: -x[0])
    return [li for _, li in scored[:k]]


def _value_search_deep(game, le, net, se, ae, edge_index, device,
                        max_depth=5, temperature=1.0):
    """Level-by-level batched tapering search.

    Instead of 120+ individual NN forwards (recursive), this expands the tree
    breadth-first and batches all positions at each depth into ONE forward pass.
    Total forwards: ~6 (one policy per intermediate level + one value at leaves).
    """
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    our_seat = game.current_player()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    # ── Root: select top-K candidates via policy ──
    width = TAPER_WIDTHS[0]
    if len(le) > width:
        root_cands = _policy_top_k_indices(game, le, net, se, ae, edge_index,
                                           device, width)
        if not root_cands:
            root_cands = list(range(len(le)))
    else:
        root_cands = list(range(len(le)))

    # frontier: list of (game_clone, root_candidate_index)
    frontier = []
    for ci, ai in enumerate(root_cands):
        child = game.clone()
        child.step(ai)
        frontier.append((child, ci))

    # ── Expand level by level (depths 1 .. max_depth-1) ──
    for d in range(1, max_depth):
        if not frontier:
            break
        width = TAPER_WIDTHS[min(d, len(TAPER_WIDTHS) - 1)]

        expandable = []
        next_frontier = []

        for fg, rci in frontier:
            if fg.is_terminal() or fg.turn_number >= MAX_TURNS:
                next_frontier.append((fg, rci))
                continue
            fle = fg.get_legal_actions()
            if not fle:
                next_frontier.append((fg, rci))
                continue
            if len(fle) == 1:
                ch = fg.clone()
                ch.step(0)
                next_frontier.append((ch, rci))
                continue
            if len(fle) <= width:
                for ai in range(len(fle)):
                    ch = fg.clone()
                    ch.step(ai)
                    next_frontier.append((ch, rci))
                continue
            expandable.append((fg, fle, rci))

        if expandable:
            B = len(expandable)
            nf_b = np.zeros((B, N, NF), dtype=np.float32)
            ef_b = np.zeros((B, E, EF), dtype=np.float32)
            ff_b = np.zeros((B, FF), dtype=np.float32)
            mask_b = np.zeros((B, MASK_DIM), dtype=np.float32)

            for i, (g, fle, _) in enumerate(expandable):
                se.encode_into(g.get_state_view(), nf_b[i], ef_b[i], ff_b[i])
                m = ae.get_action_mask(fle).numpy()
                mask_b[i, :len(m)] = m

            batch = {
                "node_features": torch.from_numpy(nf_b).to(device),
                "edge_index": edge_index,
                "edge_features": torch.from_numpy(ef_b).to(device),
                "flat_features": torch.from_numpy(ff_b).to(device),
                "action_mask": torch.from_numpy(mask_b).to(device),
            }
            with torch.no_grad():
                out = net(batch)
            all_logits = out["policy_logits"][:, :AD].cpu().numpy()

            for i, (g, fle, rci) in enumerate(expandable):
                logits = all_logits[i]
                scored = []
                for j, a in enumerate(fle):
                    try:
                        scored.append((logits[ae.encode(a)], j))
                    except ValueError:
                        continue
                scored.sort(key=lambda x: -x[0])
                top = [li for _, li in scored[:width]]
                if not top:
                    top = list(range(min(width, len(fle))))
                for ai in top:
                    ch = g.clone()
                    ch.step(ai)
                    next_frontier.append((ch, rci))

        frontier = next_frontier

    # ── Leaf evaluation: one batched value forward ──
    leaf_games = [g for g, _ in frontier]
    leaf_rci = np.array([rci for _, rci in frontier], dtype=np.int32)
    B_leaf = len(leaf_games)
    leaf_vals = np.zeros(B_leaf, dtype=np.float32)
    non_term = []

    for i, g in enumerate(leaf_games):
        if g.is_terminal():
            w = g.winner()
            if w == our_seat:
                leaf_vals[i] = 10.0
            elif w is not None:
                leaf_vals[i] = -10.0
        else:
            non_term.append(i)

    if non_term:
        n_eval = len(non_term)
        nf_b = np.zeros((n_eval, N, NF), dtype=np.float32)
        ef_b = np.zeros((n_eval, E, EF), dtype=np.float32)
        ff_b = np.zeros((n_eval, FF), dtype=np.float32)
        mask_b = np.zeros((n_eval, MASK_DIM), dtype=np.float32)
        child_pl = np.zeros(n_eval, dtype=np.int32)

        for j, i in enumerate(non_term):
            g = leaf_games[i]
            se.encode_into(g.get_state_view(), nf_b[j], ef_b[j], ff_b[j])
            child_pl[j] = g.current_player()
            gle = g.get_legal_actions()
            m = ae.get_action_mask(gle).numpy()
            mask_b[j, :len(m)] = m

        batch = {
            "node_features": torch.from_numpy(nf_b).to(device),
            "edge_index": edge_index,
            "edge_features": torch.from_numpy(ef_b).to(device),
            "flat_features": torch.from_numpy(ff_b).to(device),
            "action_mask": torch.from_numpy(mask_b).to(device),
        }
        with torch.no_grad():
            out = net(batch)
        v4 = out["value"].cpu().numpy()
        for j, i in enumerate(non_term):
            offset = (our_seat - child_pl[j]) % 4
            leaf_vals[i] = float(v4[j, offset])

    # ── Max-pool leaf values back to root candidates ──
    root_vals = np.full(len(root_cands), -1e30, dtype=np.float32)
    for i in range(B_leaf):
        rci = leaf_rci[i]
        if leaf_vals[i] > root_vals[rci]:
            root_vals[rci] = leaf_vals[i]

    for ci, ai in enumerate(root_cands):
        root_vals[ci] = apply_action_bonus(root_vals[ci], le[ai])

    # ── Stochastic root selection ──
    top_k = min(5, len(root_cands))
    top_idx = np.argpartition(root_vals, -top_k)[-top_k:]
    top_v = root_vals[top_idx]
    top_v -= top_v.max()
    t = max(temperature, 0.01)
    probs = np.exp(top_v / t)
    probs /= probs.sum()
    chosen_ci = int(top_idx[np.random.choice(top_k, p=probs)])
    best = root_cands[chosen_ci]

    best = fix_robber_steal(best, le)
    return best


# ── Eval subprocess workers (module-level for pickling) ──────────

def _h2h_worker(q, new_path, old_path, dev, n_games, seed_off):
    """Head-to-head: new checkpoint vs old checkpoint at 0-ply."""
    import os as _os
    _os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        import torch as _torch
        _torch.cuda.set_device(int(dev.split(":")[-1]))
        from human_bot.model import HumanBotNet
        from hexzero.bindings.lib_loader import load_library
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder
        import numpy as np

        load_library()
        ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        se = g0.make_state_encoder()
        edge_index = se._edge_index.to(dev)

        net_new = HumanBotNet.load_checkpoint(new_path, device=dev)
        net_new.eval()
        net_old = HumanBotNet.load_checkpoint(old_path, device=dev)
        net_old.eval()

        AD = 337
        MASK_DIM = 397
        new_wins = old_wins = 0

        for gi in range(n_games):
            seed = 200000 + seed_off + gi
            game = CatanGame(seed=seed); game.reset()
            new_seats = {gi % 4, (gi + 2) % 4}
            old_seats = {(gi + 1) % 4, (gi + 3) % 4}

            N = se.num_nodes; NF = se.NODE_FEATURE_DIM
            E = se.num_edges; EF = se.EDGE_FEATURE_DIM
            FF = se.FLAT_FEATURE_DIM
            nf = np.zeros((N, NF), dtype=np.float32)
            ef = np.zeros((E, EF), dtype=np.float32)
            ff = np.zeros(FF, dtype=np.float32)

            while not game.is_terminal() and game.turn_number < MAX_TURNS:
                le = game.get_legal_actions()
                if not le: break
                if len(le) == 1:
                    game.step(0); continue
                cp = game.current_player()
                model = net_new if cp in new_seats else net_old

                se.encode_into(game.get_state_view(), nf, ef, ff)
                mask_np = ae.get_action_mask(le).numpy()
                mask_full = np.zeros(MASK_DIM, dtype=np.float32)
                mask_full[:len(mask_np)] = mask_np
                batch = {
                    "node_features": _torch.from_numpy(nf[None]).to(dev),
                    "edge_index": edge_index,
                    "edge_features": _torch.from_numpy(ef[None]).to(dev),
                    "flat_features": _torch.from_numpy(ff[None]).to(dev),
                    "action_mask": _torch.from_numpy(mask_full[None]).to(dev),
                }
                with _torch.no_grad():
                    out = model(batch)
                logits = out["policy_logits"][0, :AD].cpu().numpy()
                enc_idx = int(np.argmax(logits))
                chosen = next((i for i, a in enumerate(le)
                               if ae.encode(a) == enc_idx), 0)
                game.step(chosen)

            w = game.winner()
            if w is not None:
                if w in new_seats: new_wins += 1
                else: old_wins += 1

        q.put({"new_wins": new_wins, "old_wins": old_wins,
               "total": n_games})
    except Exception as e:
        q.put({"error": str(e)})


def _eval_worker(q, net_path, dev, d, n_games, seed_off):
    import os as _os
    _os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        import torch as _torch
        _torch.cuda.set_device(int(dev.split(":")[-1]))
        from human_bot.model import HumanBotNet
        from hexzero.bindings.lib_loader import load_library
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder
        from human_bot.eval_search import evaluate_search_vs_ab2
        _lib = load_library()
        _ae = ActionEncoder()
        _g = CatanGame(seed=0)
        _g.reset()
        _se = _g.make_state_encoder()
        _net = HumanBotNet.load_checkpoint(net_path, device=dev)
        _net.eval()
        r = evaluate_search_vs_ab2(
            _net, _se, _ae, dev, _lib,
            num_games=n_games, search_depth=d,
            seed_offset=seed_off,
        )
        q.put(r)
    except Exception as e:
        q.put({"error": str(e)})


# ── Play one game ────────────────────────────────────────────────

def play_game(seed, net, se, ae, edge_index, device, search_depth=1,
              deep_search_depth=0, temperature=1.0):
    from hexzero.game.interface import CatanGame

    game = CatanGame(seed=seed)
    game.reset()

    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FF, dtype=np.float32)

    steps = []
    while (not game.is_terminal()
           and game.turn_number < MAX_TURNS
           and len(steps) < MAX_STEPS_PER_GAME):
        le = game.get_legal_actions()
        if not le:
            break

        sv = game.get_state_view()
        se.encode_into(sv, nf, ef, ff)
        mask = ae.get_action_mask(le).numpy()

        if len(le) == 1:
            chosen = 0
        elif game.turn_number <= 7 or search_depth == 0:
            chosen = _policy_argmax(nf, ef, ff, mask, net, ae, edge_index, device, le)
        elif deep_search_depth > 1:
            chosen = _value_search_deep(game, le, net, se, ae, edge_index,
                                        device, max_depth=deep_search_depth,
                                        temperature=temperature)
        else:
            chosen = _value_search_1ply(game, le, net, se, ae, edge_index,
                                        device, temperature=temperature)

        try:
            enc_action = ae.encode(le[chosen])
        except ValueError:
            game.step(chosen)
            continue

        steps.append({
            "nf": nf.copy(),
            "ef": ef.copy(),
            "ff": ff.copy(),
            "mask": mask.copy(),
            "action_idx": enc_action,
            "player": game.current_player(),
        })
        game.step(chosen)

    winner = game.winner()
    reward_vec = np.zeros(4, dtype=np.float32)
    if winner is not None:
        reward_vec[winner] = 1.0

    step_weights = compute_step_weights(steps, reward_vec)
    return steps, reward_vec, step_weights, winner


# ── Shard I/O ────────────────────────────────────────────────────

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
    path = os.path.join(output_dir, f"{shard_id}.pt")
    atomic_torch_save(data, path)
    return len(all_nf)


# ── Actor ────────────────────────────────────────────────────────

def _temperature_for_round(round_num, temp_start=1.0, temp_end=0.2,
                           anneal_rounds=200):
    """Linear temperature decay: temp_start -> temp_end over anneal_rounds."""
    if round_num >= anneal_rounds:
        return temp_end
    return temp_start + (temp_end - temp_start) * (round_num / anneal_rounds)


def _run_actor_inner(gpu_id, actor_id, checkpoint_path, shard_dir, ckpt_dir,
                     search_depth, seed_base, reload_interval, max_pending,
                     deep_search_depth=0):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    print(f"[actor {actor_id}] Starting on {device}, depth={search_depth}", flush=True)

    from hexzero.bindings.lib_loader import load_library
    load_library()

    net = _load_net(checkpoint_path, device)
    _, ae, se, edge_index = _setup_game_and_encoders(device)

    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)

    game_batch = []
    shard_idx = 0
    total_games = 0
    total_steps = 0
    wins = np.zeros(4, dtype=np.int64)
    t_start = time.time()
    latest_ckpt_mtime = 0.0
    current_round = 0
    try:
        ckpt0 = torch.load(checkpoint_path, map_location="cpu",
                           weights_only=False)
        current_round = ckpt0.get("metadata", {}).get("round", 0)
        del ckpt0
    except Exception:
        pass
    _stop = False

    def _on_signal(sig, frame):
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGTERM, _on_signal)

    while not _stop:
        temp = _temperature_for_round(current_round)
        seed = seed_base + gpu_id * 10_000_000 + total_games
        steps, rv, sw, winner = play_game(seed, net, se, ae, edge_index, device,
                                         search_depth, deep_search_depth,
                                         temperature=temp)
        game_batch.append((steps, rv, sw))
        total_games += 1
        total_steps += len(steps)
        if winner is not None:
            wins[winner] += 1

        if len(game_batch) >= GAMES_PER_SHARD:
            sid = f"sp_a{actor_id:03d}_{shard_idx:06d}"
            save_shard(game_batch, pending_dir, sid)
            game_batch = []
            shard_idx += 1

            pending_count = sum(1 for f in os.listdir(pending_dir)
                                if f.endswith(".pt") and not f.endswith(".tmp"))
            while pending_count > max_pending and not _stop:
                time.sleep(2)
                pending_count = sum(1 for f in os.listdir(pending_dir)
                                    if f.endswith(".pt") and not f.endswith(".tmp"))

        if total_games % 10 == 0:
            elapsed = time.time() - t_start
            gps = total_games / elapsed if elapsed > 0 else 0
            avg_steps = total_steps / max(total_games, 1)
            print(f"[actor {actor_id}] {total_games} games, {shard_idx} shards, "
                  f"{gps:.1f} g/s, ~{avg_steps:.0f} steps/g, "
                  f"wins={wins.tolist()}", flush=True)

        if reload_interval > 0 and total_games % reload_interval == 0:
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            try:
                mt = os.path.getmtime(latest_path)
                if mt > latest_ckpt_mtime:
                    net = _load_net(latest_path, device)
                    latest_ckpt_mtime = mt
                    try:
                        ckpt = torch.load(latest_path, map_location="cpu",
                                          weights_only=False)
                        current_round = ckpt.get("metadata", {}).get("round", 0)
                    except Exception:
                        pass
                    temp = _temperature_for_round(current_round)
                    print(f"[actor {actor_id}] Reloaded (round={current_round}, "
                          f"temp={temp:.2f})", flush=True)
            except FileNotFoundError:
                pass

    if game_batch:
        sid = f"sp_a{actor_id:03d}_{shard_idx:06d}"
        save_shard(game_batch, pending_dir, sid)
        print(f"[actor {actor_id}] Flushed {len(game_batch)} games on shutdown",
              flush=True)


def run_actor(gpu_id, actor_id, checkpoint_path, shard_dir, ckpt_dir,
              search_depth=1, seed_base=0, reload_interval=100,
              max_pending=MAX_PENDING_SHARDS, deep_search_depth=0):
    import traceback
    try:
        _run_actor_inner(gpu_id, actor_id, checkpoint_path, shard_dir, ckpt_dir,
                         search_depth, seed_base, reload_interval, max_pending,
                         deep_search_depth)
    except Exception:
        print(f"\n!!! [actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()
        raise


# ── Learner ──────────────────────────────────────────────────────

def run_learner(gpu_id, shard_dir, ckpt_dir, initial_checkpoint,
                batch_size=8192, shards_per_train=20,
                eval_games=100, eval_interval=4, max_rounds=0,
                wandb_name=None):
    import traceback
    try:
        _run_learner_inner(gpu_id, shard_dir, ckpt_dir, initial_checkpoint,
                           batch_size, shards_per_train, eval_games,
                           eval_interval, max_rounds, wandb_name=wandb_name)
    except Exception:
        print(f"\n!!! [learner] CRASHED !!!", flush=True)
        traceback.print_exc()
        raise


def _run_learner_inner(gpu_id, shard_dir, ckpt_dir, initial_checkpoint,
                       batch_size, shards_per_train, eval_games,
                       eval_interval, max_rounds, wandb_name=None):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    print(f"[learner] Starting on {device}", flush=True)

    from hexzero.bindings.lib_loader import load_library
    load_library()

    from human_bot.model import HumanBotNet, SmallNetworkConfig
    from human_bot.config import HumanBotTrainingConfig
    from human_bot.loss import UncertaintyWeightedLoss, human_policy_loss, value_loss, masked_entropy
    from human_bot.train import DeviceDataset, build_cosine_scheduler
    from human_bot.eval_search import evaluate_search_vs_ab2
    from human_bot.dataset import HumanGameDataset

    net = _load_net(initial_checkpoint, device)
    net.train()

    loss_combiner = UncertaintyWeightedLoss().to(device)
    cfg = HumanBotTrainingConfig(
        batch_size=batch_size,
        epochs=1,
        freeze_encoder_epochs=0,
        label_smoothing=0.05,
        entropy_weight=0.01,
        gradient_clip=1.0,
    )

    lib, ae, se, edge_index = _setup_game_and_encoders(device)

    pending_dir = os.path.join(shard_dir, "pending")
    consumed_dir = os.path.join(shard_dir, "consumed")
    os.makedirs(pending_dir, exist_ok=True)
    os.makedirs(consumed_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_meta = {}
    try:
        ckpt_data = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        ckpt_meta = ckpt_data.get("metadata", {})
    except Exception:
        pass
    round_num = ckpt_meta.get("round", 0)
    total_examples = ckpt_meta.get("total_examples", 0)
    best_wr = ckpt_meta.get("best_wr", 0.0)
    if best_wr == 0.0:
        best_path = os.path.join(ckpt_dir, "best.pt")
        try:
            best_data = torch.load(best_path, map_location="cpu", weights_only=False)
            best_wr = best_data.get("metadata", {}).get("2ply_wr", 0.0)
        except Exception:
            pass
    t_start = time.time()
    print(f"[learner] Resuming from round={round_num}, examples={total_examples:,}, best_wr={best_wr:.1%}",
          flush=True)

    # W&B — resume from saved run ID if available
    wandb_run = None
    wandb_id_path = os.path.join(ckpt_dir, ".wandb_id")
    try:
        import wandb
        if "WANDB_API_KEY" not in os.environ:
            print("[learner] Warning: WANDB_API_KEY not set, W&B may fail",
                  flush=True)
        resume_id = None
        if os.path.exists(wandb_id_path):
            with open(wandb_id_path) as f:
                resume_id = f.read().strip()
        run_name = wandb_name or f"sp-{time.strftime('%m%d-%H%M')}"
        wandb_run = wandb.init(
            project="human-bot-selfplay",
            name=run_name,
            id=resume_id,
            resume="allow",
            config={"batch_size": batch_size, "shards_per_train": shards_per_train,
                    "eval_games": eval_games, "eval_interval": eval_interval},
        )
        with open(wandb_id_path, "w") as f:
            f.write(wandb_run.id)
        print(f"W&B: {wandb_run.url}", flush=True)
    except Exception as e:
        print(f"[learner] W&B init failed: {e}", flush=True)

    while max_rounds <= 0 or round_num < max_rounds:
        shards = sorted(
            f for f in os.listdir(pending_dir)
            if f.endswith(".pt") and not f.endswith(".tmp")
        )

        if len(shards) < shards_per_train:
            time.sleep(5)
            continue

        group = shards[:shards_per_train]
        print(f"\n[learner] Round {round_num}: loading {len(group)} shards...",
              flush=True)

        nfs, efs, ffs, masks, acts, vts, sws = [], [], [], [], [], [], []
        for fn in group:
            path = os.path.join(pending_dir, fn)
            try:
                d = torch.load(path, weights_only=False, map_location="cpu")
            except Exception as e:
                print(f"[learner] Skip corrupt shard {fn}: {e}", flush=True)
                os.rename(path, os.path.join(consumed_dir, fn))
                continue

            players = d["player"].numpy()
            rv = d["reward_vec"].numpy()
            S = players.shape[0]

            winners_idx = rv.argmax(axis=1)
            vt = np.zeros((S, 4), dtype=np.float32)
            vt[np.arange(S), winners_idx] = 1.0
            vt[rv.max(axis=1) < 1e-8] = 0.25
            shifts = (-players % 4).astype(np.int32)
            idx_arr = (np.arange(4)[None, :] + shifts[:, None]) % 4
            vt = np.take_along_axis(vt, idx_arr, axis=1)

            mask = d["action_mask"]
            if mask.shape[-1] < MASK_DIM:
                mask = torch.cat(
                    [mask, torch.zeros(S, MASK_DIM - mask.shape[-1], dtype=mask.dtype)],
                    dim=-1,
                )

            nfs.append(d["node_features"])
            efs.append(d["edge_features"])
            ffs.append(d["flat_features"])
            masks.append(mask)
            acts.append(d["action_idx"])
            vts.append(torch.from_numpy(vt))
            if "step_weight" in d:
                sws.append(d["step_weight"])
            else:
                is_w = torch.from_numpy((vt[:, 0] > 0.5).astype(np.float32))
                sws.append(1.0 + 0.5 * is_w)

        if not nfs:
            for fn in group:
                src = os.path.join(pending_dir, fn)
                if os.path.exists(src):
                    os.rename(src, os.path.join(consumed_dir, fn))
            continue

        all_nf = torch.cat(nfs).to(device, non_blocking=True)
        all_ef = torch.cat(efs).to(device, non_blocking=True)
        all_ff = torch.cat(ffs).to(device, non_blocking=True)
        all_mask = torch.cat(masks).to(device, non_blocking=True)
        all_act = torch.cat(acts).to(device, non_blocking=True)
        all_vt = torch.cat(vts).to(device, non_blocking=True)
        all_sw = torch.cat(sws).to(device, non_blocking=True)
        n = all_nf.shape[0]
        del nfs, efs, ffs, masks, acts, vts, sws
        gc.collect()

        all_params = list(net.parameters()) + list(loss_combiner.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=1e-4)
        n_steps = max(1, n // cfg.batch_size)
        scheduler = build_cosine_scheduler(optimizer, n_steps, min(50, n_steps))

        net.train()
        BS = cfg.batch_size
        perm = torch.randperm(n, device=device)
        sums = {k: 0.0 for k in ("policy_loss", "value_loss", "total_loss",
                                   "entropy", "policy_acc", "value_acc")}
        n_batches = 0

        for bi in range(0, n, BS):
            idx = perm[bi : bi + BS]
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
                label_smoothing=cfg.label_smoothing,
                winner_boost=sw_b,
            )
            turn_progress = ff_b[:, 114]
            v_loss = value_loss(out["value"], vt_b, turn_progress=turn_progress)
            ent = masked_entropy(out["policy_logits"], mask_b)
            total, _ = loss_combiner(p_loss, v_loss, ent, cfg.entropy_weight)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pacc = (out["policy_logits"].argmax(dim=-1) == act_b).float().mean().item()
                vacc = (out["value"].argmax(dim=-1) == vt_b.argmax(dim=-1)).float().mean().item()

            sums["policy_loss"] += p_loss.item()
            sums["value_loss"] += v_loss.item()
            sums["total_loss"] += total.item()
            sums["entropy"] += ent.item()
            sums["policy_acc"] += pacc
            sums["value_acc"] += vacc
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        total_examples += n
        round_num += 1

        del all_nf, all_ef, all_ff, all_mask, all_act, all_vt, all_sw, perm
        gc.collect()
        torch.cuda.empty_cache()

        elapsed = time.time() - t_start
        print(f"[learner] Round {round_num}: {n:,} ex  "
              f"pacc={avg['policy_acc']:.3f}  vacc={avg['value_acc']:.3f}  "
              f"ploss={avg['policy_loss']:.3f}  total={total_examples:,}  "
              f"({elapsed:.0f}s)", flush=True)

        if wandb_run:
            import wandb
            wandb.log({
                "train/policy_loss": avg["policy_loss"],
                "train/value_loss": avg["value_loss"],
                "train/policy_acc": avg["policy_acc"],
                "train/value_acc": avg["value_acc"],
                "train/entropy": avg["entropy"],
                "train/total_examples": total_examples,
                "train/round": round_num,
                "train/temperature": _temperature_for_round(round_num),
            })

        for fn in group:
            src = os.path.join(pending_dir, fn)
            if os.path.exists(src):
                os.rename(src, os.path.join(consumed_dir, fn))

        ckpt_path = os.path.join(ckpt_dir, "latest.pt")
        net.save_checkpoint(ckpt_path + ".tmp", {
            "round": round_num,
            "total_examples": total_examples,
            "best_wr": best_wr,
            **avg,
        })
        os.rename(ckpt_path + ".tmp", ckpt_path)

        if eval_interval > 0 and round_num % eval_interval == 0 and eval_games > 0:
            net.eval()
            print(f"[learner] Evaluating ({eval_games} games per depth)...",
                  flush=True)
            import multiprocessing as _mp
            eval_timeout = 180
            tmp_ckpt = os.path.join(ckpt_dir, "_eval_tmp.pt")
            net.save_checkpoint(tmp_ckpt)

            for depth in [0, 1, 2]:
                t0 = time.time()
                try:
                    _ctx = _mp.get_context("spawn")
                    result_q = _ctx.Queue()

                    p = _ctx.Process(
                        target=_eval_worker,
                        args=(result_q, tmp_ckpt, device, depth,
                              eval_games, round_num * 100 + depth),
                    )
                    p.start()
                    p.join(timeout=eval_timeout)
                    if p.is_alive():
                        print(f"  {depth}-ply: TIMEOUT after {eval_timeout}s",
                              flush=True)
                        p.kill()
                        p.join(timeout=5)
                        continue
                    if result_q.empty():
                        print(f"  {depth}-ply: eval process died "
                              f"(exit={p.exitcode})", flush=True)
                        continue
                    result = result_q.get_nowait()
                    if "error" in result:
                        print(f"  {depth}-ply: eval error: {result['error']}",
                              flush=True)
                        continue
                except Exception as e:
                    print(f"  {depth}-ply: eval failed: {e}", flush=True)
                    continue

                wr = result["win_rate"]
                dt = time.time() - t0
                print(f"  {depth}-ply: NN={result['hz_wins']}  "
                      f"AB2={result['ab2_wins']}  WR={wr:.1%}  "
                      f"rank={result['avg_rank']:.2f}  ({dt:.0f}s)",
                      flush=True)
                if wandb_run:
                    import wandb
                    wandb.log({
                        f"eval/{depth}ply_win_rate": wr,
                        f"eval/{depth}ply_avg_rank": result["avg_rank"],
                        "train/round": round_num,
                    })
                if depth == 2 and wr > best_wr:
                    best_wr = wr
                    best_path = os.path.join(ckpt_dir, "best.pt")
                    net.save_checkpoint(best_path, {
                        "round": round_num,
                        "total_examples": total_examples,
                        "2ply_wr": wr,
                    })
                    print(f"  ** New best: {wr:.1%} → {best_path}", flush=True)

            # ── Head-to-head vs previous checkpoint ──
            prev_ckpt = os.path.join(ckpt_dir, "_prev.pt")
            if os.path.exists(prev_ckpt):
                try:
                    _ctx2 = _mp.get_context("spawn")
                    h2h_q = _ctx2.Queue()
                    p2 = _ctx2.Process(
                        target=_h2h_worker,
                        args=(h2h_q, tmp_ckpt, prev_ckpt, device,
                              eval_games, round_num),
                    )
                    p2.start()
                    p2.join(timeout=eval_timeout)
                    if p2.is_alive():
                        print(f"  h2h: TIMEOUT", flush=True)
                        p2.kill(); p2.join(timeout=5)
                    elif h2h_q.empty():
                        print(f"  h2h: process died (exit={p2.exitcode})",
                              flush=True)
                    else:
                        h2h = h2h_q.get_nowait()
                        if "error" in h2h:
                            print(f"  h2h error: {h2h['error']}", flush=True)
                        else:
                            nw = h2h["new_wins"]; ow = h2h["old_wins"]
                            wr = nw / max(nw + ow, 1)
                            print(f"  h2h vs prev: new={nw} old={ow} "
                                  f"WR={wr:.0%}", flush=True)
                            if wandb_run:
                                import wandb
                                wandb.log({"eval/h2h_vs_prev": wr,
                                           "train/round": round_num})
                except Exception as e:
                    print(f"  h2h eval failed: {e}", flush=True)

            try:
                os.rename(tmp_ckpt, prev_ckpt)
            except Exception:
                try:
                    os.remove(tmp_ckpt)
                except FileNotFoundError:
                    pass
            net.train()


# ── Launcher ─────────────────────────────────────────────────────

def run_all(args):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    from hexzero.bindings.lib_loader import load_library
    load_library()

    shard_dir = args.shard_dir
    ckpt_dir = args.ckpt_dir
    os.makedirs(os.path.join(shard_dir, "pending"), exist_ok=True)
    os.makedirs(os.path.join(shard_dir, "consumed"), exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    procs = []

    learner = ctx.Process(
        target=run_learner,
        args=(0, shard_dir, ckpt_dir, args.checkpoint,
              args.batch_size, args.shards_per_train,
              args.eval_games, args.eval_interval,
              args.max_rounds),
        kwargs={"wandb_name": args.wandb_name},
        daemon=False,
    )
    learner.start()
    procs.append(learner)
    print(f"Started learner (pid={learner.pid}) on cuda:0", flush=True)

    total_actors = 0
    for g in range(1, args.num_actor_gpus + 1):
        for a in range(args.actors_per_gpu):
            aid = total_actors
            actor_seed = args.seed + aid * 100_000
            actor = ctx.Process(
                target=run_actor,
                args=(g, aid, args.checkpoint, shard_dir, ckpt_dir,
                      args.search_depth, actor_seed, args.reload_interval,
                      args.max_pending, args.deep_search_depth),
                daemon=True,
            )
            actor.start()
            procs.append(actor)
            total_actors += 1

    print(f"\nSelf-play running: 1 learner + {total_actors} actors "
          f"({args.actors_per_gpu}/gpu × {args.num_actor_gpus} gpus)",
          flush=True)

    reported_dead = set()
    try:
        while True:
            time.sleep(10)
            newly_dead = []
            for i, p in enumerate(procs):
                if not p.is_alive() and i not in reported_dead:
                    role = "learner" if i == 0 else f"actor {i-1}"
                    code = p.exitcode
                    newly_dead.append(role)
                    reported_dead.add(i)
                    print(f"!!! {role} (pid={p.pid}) died with exit code {code} !!!",
                          flush=True)
            if newly_dead:
                if "learner" in newly_dead:
                    print("!!! LEARNER DEAD — aborting all !!!", flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.terminate()
                    break
                alive = sum(1 for p in procs[1:] if p.is_alive())
                print(f"    {alive}/{total_actors} actors still alive", flush=True)
                if alive == 0:
                    print("!!! ALL ACTORS DEAD — aborting !!!", flush=True)
                    procs[0].terminate()
                    break
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
        for p in procs:
            p.terminate()

    for p in procs:
        p.join(timeout=10)


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-play improvement via Expert Iteration")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Initial model checkpoint")
    parser.add_argument("--role", choices=["actor", "learner", "all"],
                        default="all")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-actor-gpus", type=int, default=3)
    parser.add_argument("--actors-per-gpu", type=int, default=15,
                        help="Actor processes per GPU (share GPU via CUDA)")
    parser.add_argument("--shard-dir", type=str, default="data/selfplay")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints/selfplay")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--shards-per-train", type=int, default=20)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--deep-search-depth", type=int, default=0,
                        help="Ply depth for important moves (0 = disabled, 5 = recommended)")
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=4,
                        help="Evaluate every N training rounds")
    parser.add_argument("--reload-interval", type=int, default=100,
                        help="Actor reloads checkpoint every N games")
    parser.add_argument("--max-pending", type=int, default=200,
                        help="Actors pause when pending shard count exceeds this")
    parser.add_argument("--max-rounds", type=int, default=0,
                        help="Stop after N training rounds (0 = unlimited)")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name prefix (default: sp-MMDD-HHMM)")
    args = parser.parse_args()

    if args.role == "all":
        run_all(args)
    elif args.role == "actor":
        run_actor(args.gpu_id, args.gpu_id, args.checkpoint, args.shard_dir,
                  args.ckpt_dir, args.search_depth, args.seed,
                  args.reload_interval, args.max_pending,
                  args.deep_search_depth)
    elif args.role == "learner":
        run_learner(args.gpu_id, args.shard_dir, args.ckpt_dir,
                    args.checkpoint, args.batch_size, args.shards_per_train,
                    args.eval_games, args.eval_interval, args.max_rounds,
                    wandb_name=args.wandb_name)


if __name__ == "__main__":
    main()
