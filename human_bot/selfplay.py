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
    """Per-step policy weights: graded by final position, all positive."""
    winner = int(np.argmax(reward_vec)) if reward_vec.max() > 0 else -1
    S = len(steps)
    weights = np.ones(S, dtype=np.float32)
    speed_mult = 1.0 + max(0.0, min(0.5, (600 - S) / 600.0))

    rank_order = np.argsort(-reward_vec)
    seat_to_rank = {int(seat): rank for rank, seat in enumerate(rank_order)}
    rank_weights = {0: None, 1: 0.3, 2: 0.1, 3: 0.05}

    for i, s in enumerate(steps):
        progress = i / max(S - 1, 1)
        rank = seat_to_rank.get(s["player"], 3)
        if rank == 0:
            base = (1.0 + progress) * speed_mult
        else:
            base = rank_weights[rank]
        weights[i] = base * _ACT_MOD[min(s["action_idx"], MASK_DIM - 1)]
    return weights


def atomic_torch_save(data, path):
    tmp = path + ".tmp"
    for attempt in range(3):
        try:
            torch.save(data, tmp)
            os.rename(tmp, path)
            return
        except (FileNotFoundError, OSError) as e:
            if attempt < 2:
                time.sleep(1)
            else:
                raise


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

def _policy_argmax(nf, ef, ff, mask, net, ae, edge_index, device, le,
                   temperature=0.0):
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

    if temperature > 0.01:
        enc_to_legal = {}
        for i, a in enumerate(le):
            try:
                enc_to_legal[ae.encode(a)] = i
            except ValueError:
                continue
        if not enc_to_legal:
            return 0
        encs = list(enc_to_legal.keys())
        scores = np.array([logits[e] for e in encs])
        scores -= scores.max()
        probs = np.exp(scores / temperature)
        probs /= probs.sum()
        chosen_enc = encs[np.random.choice(len(encs), p=probs)]
        return enc_to_legal[chosen_enc]

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

TAPER_WIDTHS = [10, 4, 2, 2, 2, 2, 2, 2, 2, 2]

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


# ── Rollout search for policy improvement ────────────────────────

def _rollout_search(game, le, net, se, ae, edge_index, device, lib,
                    depth=60, top_k=2):
    """d60-style rollout search: try top_k moves, roll out with policy argmax,
    evaluate with AB value at leaf. Returns best action index."""
    import ctypes
    from hexzero.bindings.structs import Game as CGame
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    seat = game.current_player()
    cg = game._game
    color = cg.state.colors[cg.state.current_player_index]

    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FF, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)

    se.encode_into(game.get_state_view(), nf, ef, ff)
    mn = ae.get_action_mask(le).numpy()
    mk[:len(mn)] = mn
    batch = {
        "node_features": torch.from_numpy(nf[None]).to(device),
        "edge_index": edge_index,
        "edge_features": torch.from_numpy(ef[None]).to(device),
        "flat_features": torch.from_numpy(ff[None]).to(device),
        "action_mask": torch.from_numpy(mk[None]).to(device),
    }
    with torch.no_grad():
        logits = net(batch)["policy_logits"][0, :AD].cpu().numpy()

    scored = []
    for i, a in enumerate(le):
        try:
            scored.append((logits[ae.encode(a)], i))
        except ValueError:
            scored.append((-1e9, i))
    scored.sort(reverse=True)
    candidates = [idx for _, idx in scored[:top_k]]

    def _ab_leaf(g_obj):
        ch = CGame()
        lib.game_copy(ctypes.byref(ch), ctypes.byref(g_obj._game))
        return float(lib.base_value_fn(ctypes.byref(ch), color))

    def _policy_argmax_step(gc):
        le2 = gc.get_legal_actions()
        if not le2:
            return
        if len(le2) == 1:
            gc.step(0)
            return
        se.encode_into(gc.get_state_view(), nf, ef, ff)
        mn2 = ae.get_action_mask(le2).numpy()
        mk[:] = 0
        mk[:len(mn2)] = mn2
        b = {
            "node_features": torch.from_numpy(nf[None]).to(device),
            "edge_index": edge_index,
            "edge_features": torch.from_numpy(ef[None]).to(device),
            "flat_features": torch.from_numpy(ff[None]).to(device),
            "action_mask": torch.from_numpy(mk[None]).to(device),
        }
        with torch.no_grad():
            lo = net(b)["policy_logits"][0, :AD].cpu().numpy()
        lo[mn2[:AD] < 0.5] = -1e9
        ai = int(np.argmax(lo))
        gc.step(next((i for i, a in enumerate(le2) if ae.encode(a) == ai), 0))

    best_i, best_v = candidates[0], -1e30
    for ci in candidates:
        gc = game.clone()
        gc.step(ci)
        for _ in range(depth - 1):
            if gc.is_terminal():
                break
            _policy_argmax_step(gc)
        if gc.is_terminal():
            w = gc.winner()
            v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
        else:
            v = _ab_leaf(gc)
        v = apply_action_bonus(v, le[ci])
        if v > best_v:
            best_v = v
            best_i = ci
    return fix_robber_steal(best_i, le)


# ── Batched 0-ply play (multiple games in one actor) ─────────────

def play_games_batched(seeds, net, se, ae, edge_index, device,
                       temperature=1.0, random_board=False,
                       opp_net=None, opp_lib=None,
                       search_lib=None, search_fraction=0.0):
    """Play multiple games simultaneously with batched NN forward passes.

    All games use 0-ply policy (argmax or temperature sampling).
    If opp_net is provided, 2 seats per game use the training net and
    2 seats use the opponent net (asymmetric self-play).
    If opp_lib is provided (C library), opponent seats use AB2 2-ply.
    Returns a list of (steps, reward_vec, step_weights, winner) tuples,
    same format as play_game().
    """
    import ctypes
    from hexzero.game.interface import CatanGame

    B = len(seeds)
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    has_opponent = opp_net is not None or opp_lib is not None

    games = []
    train_seats = []
    for i, seed in enumerate(seeds):
        g = CatanGame(seed=seed, random_board=random_board)
        g.reset()
        games.append(g)
        if has_opponent:
            ts = {seed % 4, (seed + 2) % 4}
        else:
            ts = {0, 1, 2, 3}
        train_seats.append(ts)

    if opp_lib is not None:
        from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS
        ch = CGame()
        ca = (CAction * MAX_ACTIONS)()
        cn = ctypes.c_int(0)
        ch2 = CGame()
        ca2 = (CAction * MAX_ACTIONS)()
        cn2 = ctypes.c_int(0)

    all_steps = [[] for _ in range(B)]
    active = list(range(B))

    nf_buf = np.zeros((B, N, NF), dtype=np.float32)
    ef_buf = np.zeros((B, E, EF), dtype=np.float32)
    ff_buf = np.zeros((B, FF), dtype=np.float32)
    mk_buf = np.zeros((B, MASK_DIM), dtype=np.float32)

    def _ab2_step(gi, le):
        """AB2 2-ply greedy for opponent seats."""
        cg = games[gi]._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi_best, bv = 0, -1e30
        for i, act in enumerate(le):
            opp_lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            opp_lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if cn.value > 0 and opp_lib.game_winning_color(ctypes.byref(ch)) < 0:
                if cn.value > 1:
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    brj, brv = 0, -1e30
                    for j in range(cn.value):
                        opp_lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        opp_lib.game_execute(ctypes.byref(ch2), ca[j], ca2, ctypes.byref(cn2))
                        rv = opp_lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > brv:
                            brv = rv
                            brj = j
                    opp_lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    opp_lib.game_execute(ctypes.byref(ch2), ca[brj], ca2, ctypes.byref(cn2))
                    v = opp_lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    opp_lib.game_execute(ctypes.byref(ch), ca[0], ca, ctypes.byref(cn))
                    v = opp_lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = opp_lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv:
                bv = v
                bi_best = i
        return bi_best

    def _nn_choose(logits, le, temp):
        mask = ae.get_action_mask(le).numpy()
        if temp > 0.01:
            enc_to_legal = {}
            for i, a in enumerate(le):
                try:
                    enc_to_legal[ae.encode(a)] = i
                except ValueError:
                    continue
            if not enc_to_legal:
                return 0, mask
            encs = list(enc_to_legal.keys())
            scores = np.array([logits[e] for e in encs])
            scores -= scores.max()
            probs = np.exp(scores / temp)
            probs /= probs.sum()
            chosen_enc = encs[np.random.choice(len(encs), p=probs)]
            return enc_to_legal[chosen_enc], mask
        else:
            logits[mask < 0.5] = -1e9
            enc_idx = int(np.argmax(logits))
            return next((i for i, a in enumerate(le)
                         if ae.encode(a) == enc_idx), 0), mask

    while active:
        need_train = []
        need_opp = []
        for gi in list(active):
            game = games[gi]
            if game.is_terminal() or game.turn_number >= MAX_TURNS or len(all_steps[gi]) >= MAX_STEPS_PER_GAME:
                active.remove(gi)
                continue
            le = game.get_legal_actions()
            if not le:
                active.remove(gi)
                continue
            if len(le) == 1:
                game.step(0)
                continue
            cp = game.current_player()
            if cp in train_seats[gi]:
                need_train.append((gi, le))
            else:
                need_opp.append((gi, le))

        # Handle opponent seats first (AB2 or opp_net)
        if need_opp:
            if opp_lib is not None:
                for gi, le in need_opp:
                    chosen = _ab2_step(gi, le)
                    games[gi].step(chosen)
            elif opp_net is not None:
                opp_batch_size = len(need_opp)
                for bi, (gi, le) in enumerate(need_opp):
                    se.encode_into(games[gi].get_state_view(), nf_buf[bi], ef_buf[bi], ff_buf[bi])
                    mn = ae.get_action_mask(le).numpy()
                    mk_buf[bi, :] = 0
                    mk_buf[bi, :len(mn)] = mn
                with torch.no_grad():
                    opp_out = opp_net({
                        "node_features": torch.from_numpy(nf_buf[:opp_batch_size]).to(device),
                        "edge_index": edge_index,
                        "edge_features": torch.from_numpy(ef_buf[:opp_batch_size]).to(device),
                        "flat_features": torch.from_numpy(ff_buf[:opp_batch_size]).to(device),
                        "action_mask": torch.from_numpy(mk_buf[:opp_batch_size]).to(device),
                    })
                opp_logits = opp_out["policy_logits"][:, :AD].cpu().numpy()
                for bi, (gi, le) in enumerate(need_opp):
                    chosen, _ = _nn_choose(opp_logits[bi], le, 0.0)
                    games[gi].step(chosen)

        if not need_train:
            if not active:
                break
            continue

        # Batch encode training seats
        batch_size = len(need_train)
        for bi, (gi, le) in enumerate(need_train):
            se.encode_into(games[gi].get_state_view(), nf_buf[bi], ef_buf[bi], ff_buf[bi])
            mn = ae.get_action_mask(le).numpy()
            mk_buf[bi, :] = 0
            mk_buf[bi, :len(mn)] = mn

        with torch.no_grad():
            out = net({
                "node_features": torch.from_numpy(nf_buf[:batch_size]).to(device),
                "edge_index": edge_index,
                "edge_features": torch.from_numpy(ef_buf[:batch_size]).to(device),
                "flat_features": torch.from_numpy(ff_buf[:batch_size]).to(device),
                "action_mask": torch.from_numpy(mk_buf[:batch_size]).to(device),
            })
        all_logits = out["policy_logits"][:, :AD].cpu().numpy()
        all_values = out["value"].cpu().numpy()

        for bi, (gi, le) in enumerate(need_train):
            logits = all_logits[bi]
            chosen, mask = _nn_choose(logits, le, temperature)

            if (search_lib is not None and search_fraction > 0
                    and len(le) > 2 and games[gi].turn_number > 7):
                chosen = _rollout_search(
                    games[gi], le, net, se, ae, edge_index,
                    device, search_lib, depth=60, top_k=3)

            try:
                enc_action = ae.encode(le[chosen])
            except ValueError:
                games[gi].step(chosen)
                continue

            value_pred = all_values[bi].copy()

            lo_for_lp = all_logits[bi].copy()
            lo_for_lp[mk_buf[bi, :AD] < 0.5] = -1e9
            mx = lo_for_lp.max()
            lse = mx + np.log(np.sum(np.exp(lo_for_lp - mx)) + 1e-38)
            actor_log_prob = float(lo_for_lp[enc_action] - lse)

            all_steps[gi].append({
                "nf": nf_buf[bi].copy(),
                "ef": ef_buf[bi].copy(),
                "ff": ff_buf[bi].copy(),
                "mask": mask.copy(),
                "action_idx": enc_action,
                "player": games[gi].current_player(),
                "log_prob": actor_log_prob,
                "value_pred": value_pred,
            })
            games[gi].step(chosen)

    results = []
    for gi in range(B):
        game = games[gi]
        steps = all_steps[gi]
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
        sw = compute_step_weights(steps, reward_vec)
        results.append((steps, reward_vec, sw, winner, final_vp))
    return results


# ── Play one game ────────────────────────────────────────────────

def play_game(seed, net, se, ae, edge_index, device, search_depth=1,
              deep_search_depth=0, temperature=1.0, random_board=False):
    from hexzero.game.interface import CatanGame

    game = CatanGame(seed=seed, random_board=random_board)
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
            chosen = _policy_argmax(nf, ef, ff, mask, net, ae, edge_index, device, le,
                                    temperature=temperature)
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
        turns = game.turn_number
        speed_bonus = max(0.0, min(0.5, (300 - turns) / 300.0))
        reward_vec[winner] = 1.0 + speed_bonus
        for seat in range(4):
            if seat != winner:
                vp = game._game.state.player_state[seat][0]
                reward_vec[seat] = vp / 20.0

    step_weights = compute_step_weights(steps, reward_vec)
    return steps, reward_vec, step_weights, winner


def play_game_vs_ab2(seed, net, se, ae, edge_index, device, lib,
                     search_depth=1, deep_search_depth=0, temperature=1.0,
                     ab_depth=2, random_board=False):
    """Play a game with 2 NN seats vs 2 AB2 seats. Record all decisions."""
    import ctypes
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS

    game = CatanGame(seed=seed, random_board=random_board)
    game.reset()

    nn_seats = {seed % 4, (seed + 2) % 4}

    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FF, dtype=np.float32)

    ch = CGame()
    ca = (CAction * MAX_ACTIONS)()
    cn = ctypes.c_int(0)
    ch2 = CGame()
    ca2 = (CAction * MAX_ACTIONS)()
    cn2 = ctypes.c_int(0)

    def ab2_choose(le):
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if ab_depth >= 2 and cn.value > 0 \
                    and lib.game_winning_color(ctypes.byref(ch)) < 0:
                if cn.value > 1:
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    best_resp, best_rv = 0, -1e30
                    for j in range(cn.value):
                        lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        lib.game_execute(ctypes.byref(ch2), ca[j], ca2, ctypes.byref(cn2))
                        rv = lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > best_rv:
                            best_rv = rv
                            best_resp = j
                    lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    lib.game_execute(ctypes.byref(ch2), ca[best_resp], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    lib.game_execute(ctypes.byref(ch), ca[0], ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv:
                bv = v
                bi = i
        return bi

    steps = []
    while (not game.is_terminal()
           and game.turn_number < MAX_TURNS
           and len(steps) < MAX_STEPS_PER_GAME):
        le = game.get_legal_actions()
        if not le:
            break

        cp = game.current_player()

        if len(le) == 1:
            game.step(0)
            continue

        sv = game.get_state_view()
        se.encode_into(sv, nf, ef, ff)
        mask = ae.get_action_mask(le).numpy()

        if cp not in nn_seats:
            chosen = ab2_choose(le)
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
            "nf": nf.copy(), "ef": ef.copy(), "ff": ff.copy(),
            "mask": mask.copy(),
            "action_idx": enc_action, "player": cp,
        })
        game.step(chosen)

    winner = game.winner()
    reward_vec = np.zeros(4, dtype=np.float32)
    if winner is not None:
        reward_vec[winner] = 1.0
        turns = game.turn_number
        speed_bonus = max(0.0, min(0.5, (300 - turns) / 300.0))
        reward_vec[winner] = 1.0 + speed_bonus
        for seat in range(4):
            if seat != winner:
                vp = game._game.state.player_state[seat][0]
                reward_vec[seat] = vp / 20.0

    step_weights = compute_step_weights(steps, reward_vec)
    return steps, reward_vec, step_weights, winner


# ── Shard I/O ────────────────────────────────────────────────────

def save_shard(games_data, output_dir, shard_id):
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player, all_reward, all_sw = [], [], [], [], []
    all_log_prob, all_value_pred, all_final_vp = [], [], []

    for game_data in games_data:
        if len(game_data) == 4:
            steps, rv, sw, final_vp = game_data
        else:
            steps, rv, sw = game_data
            final_vp = np.zeros(4, dtype=np.float32)
        for i, s in enumerate(steps):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_sw.append(sw[i])
            all_log_prob.append(s.get("log_prob", 0.0))
            all_value_pred.append(s.get("value_pred", np.zeros(4, dtype=np.float32)))
            all_final_vp.append(final_vp)

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
        "log_prob_old": torch.tensor(all_log_prob, dtype=torch.float32),
        "value_pred_old": torch.from_numpy(np.stack(all_value_pred)),
        "final_vp": torch.from_numpy(np.stack(all_final_vp)),
    }
    path = os.path.join(output_dir, f"{shard_id}.pt")
    atomic_torch_save(data, path)
    return len(all_nf)


# ── Actor ────────────────────────────────────────────────────────

def _temperature_for_round(round_num, temp_start=0.7, temp_end=0.3,
                           anneal_start=3300, anneal_rounds=2000):
    """Linear temperature decay: temp_start -> temp_end over anneal_rounds."""
    progress = (round_num - anneal_start) / anneal_rounds
    if progress <= 0:
        return temp_start
    if progress >= 1:
        return temp_end
    return temp_start + (temp_end - temp_start) * progress


def _run_actor_inner(gpu_id, actor_id, checkpoint_path, shard_dir, ckpt_dir,
                     search_depth, seed_base, reload_interval, max_pending,
                     deep_search_depth=0, mix_ab2=False, asymmetric=False):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    mode = "asymmetric" if asymmetric else ("vs-AB2" if mix_ab2 else "self-play")
    print(f"[actor {actor_id}] Starting on {device}, depth={search_depth}, {mode}", flush=True)

    from hexzero.bindings.lib_loader import load_library
    lib = load_library()

    net = _load_net(checkpoint_path, device)
    _, ae, se, edge_index = _setup_game_and_encoders(device)

    opp_net = None
    opp_lib = None

    pool_dir = os.path.join(ckpt_dir, "pool")
    os.makedirs(pool_dir, exist_ok=True)
    opp_pool_paths = sorted(
        [os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
         if f.endswith(".pt")]) if os.path.isdir(pool_dir) else []
    opp_pool_wins = {}

    def _load_random_opponent():
        nonlocal opp_net
        if not opp_pool_paths:
            return
        weights = np.array([
            (1.0 - opp_pool_wins.get(p, 0.5)) ** 2
            for p in opp_pool_paths])
        if weights.sum() < 1e-8:
            weights = np.ones(len(opp_pool_paths))
        weights /= weights.sum()
        chosen = np.random.choice(opp_pool_paths, p=weights)
        try:
            opp_net = _load_net(chosen, device)
            for p in opp_net.parameters():
                p.requires_grad = False
        except Exception:
            pass

    if opp_pool_paths:
        _load_random_opponent()
        print(f"[actor {actor_id}] Pool has {len(opp_pool_paths)} opponents", flush=True)

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

    BATCH_GAMES = 8 if (search_depth == 0 and deep_search_depth <= 1 and not mix_ab2) else 1

    while not _stop:
        temp = _temperature_for_round(current_round)

        if BATCH_GAMES > 1:
            batch_size_this = 1 if asymmetric else BATCH_GAMES
            seeds = [seed_base + gpu_id * 10_000_000 + total_games + i
                     for i in range(batch_size_this)]
            results = play_games_batched(seeds, net, se, ae, edge_index, device,
                                        temperature=temp, random_board=True,
                                        opp_net=opp_net, opp_lib=opp_lib,
                                        search_lib=lib if asymmetric else None,
                                        search_fraction=1.0 if asymmetric else 0.0)
            for steps, rv, sw, winner, fvp in results:
                game_batch.append((steps, rv, sw, fvp))
                total_games += 1
                total_steps += len(steps)
                if winner is not None:
                    wins[winner] += 1
        else:
            seed = seed_base + gpu_id * 10_000_000 + total_games
            if mix_ab2:
                steps, rv, sw, winner = play_game_vs_ab2(
                    seed, net, se, ae, edge_index, device, lib,
                    search_depth, deep_search_depth, temperature=temp, ab_depth=2,
                    random_board=True)
            else:
                steps, rv, sw, winner = play_game(
                    seed, net, se, ae, edge_index, device,
                    search_depth, deep_search_depth, temperature=temp,
                    random_board=True)
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
                    import shutil
                    pool_path = os.path.join(pool_dir, f"pool_r{current_round}.pt")
                    if not os.path.exists(pool_path):
                        try:
                            shutil.copy2(latest_path, pool_path)
                            opp_pool_paths.clear()
                            opp_pool_paths.extend(sorted(
                                os.path.join(pool_dir, f)
                                for f in os.listdir(pool_dir) if f.endswith(".pt")))
                            if len(opp_pool_paths) > 20:
                                oldest = opp_pool_paths.pop(0)
                                try:
                                    os.remove(oldest)
                                except OSError:
                                    pass
                        except OSError:
                            pass

                    net = _load_net(latest_path, device)
                    latest_ckpt_mtime = mt
                    try:
                        ckpt = torch.load(latest_path, map_location="cpu",
                                          weights_only=False)
                        current_round = ckpt.get("metadata", {}).get("round", 0)
                    except Exception:
                        pass

                    if opp_pool_paths:
                        _load_random_opponent()

                    temp = _temperature_for_round(current_round)
                    print(f"[actor {actor_id}] Reloaded (round={current_round}, "
                          f"temp={temp:.2f}, pool={len(opp_pool_paths)})", flush=True)
            except FileNotFoundError:
                pass

    if game_batch:
        sid = f"sp_a{actor_id:03d}_{shard_idx:06d}"
        save_shard(game_batch, pending_dir, sid)
        print(f"[actor {actor_id}] Flushed {len(game_batch)} games on shutdown",
              flush=True)


def run_actor(gpu_id, actor_id, checkpoint_path, shard_dir, ckpt_dir,
              search_depth=1, seed_base=0, reload_interval=100,
              max_pending=MAX_PENDING_SHARDS, deep_search_depth=0,
              mix_ab2=False, asymmetric=False):
    import traceback
    try:
        _run_actor_inner(gpu_id, actor_id, checkpoint_path, shard_dir, ckpt_dir,
                         search_depth, seed_base, reload_interval, max_pending,
                         deep_search_depth, mix_ab2=mix_ab2,
                         asymmetric=asymmetric)
    except Exception:
        print(f"\n!!! [actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()
        raise


# ── Learner ──────────────────────────────────────────────────────

def run_learner(gpu_id, shard_dir, ckpt_dir, initial_checkpoint,
                batch_size=8192, shards_per_train=20,
                eval_games=100, eval_interval=4, max_rounds=0,
                wandb_name=None, freeze_policy_rounds=0, force_ppo=False):
    import traceback
    try:
        _run_learner_inner(gpu_id, shard_dir, ckpt_dir, initial_checkpoint,
                           batch_size, shards_per_train, eval_games,
                           eval_interval, max_rounds, wandb_name=wandb_name,
                           freeze_policy_rounds=freeze_policy_rounds,
                           force_ppo=force_ppo)
    except Exception:
        print(f"\n!!! [learner] CRASHED !!!", flush=True)
        traceback.print_exc()
        raise


def _run_learner_inner(gpu_id, shard_dir, ckpt_dir, initial_checkpoint,
                       batch_size, shards_per_train, eval_games,
                       eval_interval, max_rounds, wandb_name=None,
                       freeze_policy_rounds=0, force_ppo=False):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    print(f"[learner] Starting on {device}", flush=True)

    from hexzero.bindings.lib_loader import load_library
    load_library()

    from human_bot.model import HumanBotNet, SmallNetworkConfig
    from human_bot.config import HumanBotTrainingConfig
    import torch.nn.functional as F
    from human_bot.loss import (UncertaintyWeightedLoss, human_policy_loss,
                                value_loss, masked_entropy,
                                ppo_policy_loss, value_loss_mse,
                                awr_policy_loss)
    from human_bot.train import DeviceDataset, build_cosine_scheduler
    from human_bot.eval_search import evaluate_search_vs_ab2
    from human_bot.dataset import HumanGameDataset

    net = _load_net(initial_checkpoint, device)
    net.train()

    from human_bot.loss import FixedWeightLoss
    loss_combiner = FixedWeightLoss(
        policy_weight=1.0, value_weight=1.0).to(device)
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
    start_round = round_num
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
        lp_olds, vp_olds, returns_list, fvp_list = [], [], [], []
        use_ppo = False
        for fn in group:
            path = os.path.join(pending_dir, fn)
            try:
                d = torch.load(path, weights_only=False, map_location="cpu")
                _ = d["player"], d["reward_vec"], d["node_features"]
            except Exception as e:
                print(f"[learner] Skip bad shard {fn}: {e}", flush=True)
                try:
                    os.rename(path, os.path.join(consumed_dir, fn))
                except Exception:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                continue

            players = d["player"].numpy()
            rv = d["reward_vec"].numpy()
            S = players.shape[0]

            rv_safe = np.maximum(rv, 0.0)
            sharp = np.power(rv_safe + 1e-8, 3.0)
            row_sums = sharp.sum(axis=1, keepdims=True)
            no_winner = (row_sums < 1e-8).squeeze()
            vt = np.where(row_sums > 1e-8, sharp / row_sums, 0.25)
            if vt.ndim == 1:
                vt = vt.reshape(1, -1)
            vt[no_winner] = 0.25
            # Rotate absolute-seat → cp-relative (slot 0 = cp), handling
            # variable num_players and using the CORRECT sign (+cp) so the
            # rotation matches state_encoder's (cp+i) % N convention.
            from human_bot.dataset import rotate_value_targets_to_cp
            n_p_tensor = d.get("num_players")
            n_p_arr = n_p_tensor.numpy() if n_p_tensor is not None else None
            vt = rotate_value_targets_to_cp(vt, players, n_p_arr)
            rv_rotated = rotate_value_targets_to_cp(rv, players, n_p_arr)
            ret = rv_rotated[:, 0].astype(np.float32)

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
            returns_list.append(torch.from_numpy(ret))
            if "step_weight" in d:
                sws.append(d["step_weight"])
            else:
                is_w = torch.from_numpy((vt[:, 0] > 0.5).astype(np.float32))
                sws.append(1.0 + 0.5 * is_w)

            if force_ppo and "log_prob_old" in d and "value_pred_old" in d:
                use_ppo = True
                lp_olds.append(d["log_prob_old"])
                vp_raw = d["value_pred_old"].numpy()
                vp_rotated = rotate_value_targets_to_cp(vp_raw, players, n_p_arr)
                vp_exp = np.exp(vp_rotated - vp_rotated.max(axis=1, keepdims=True))
                vp_probs = vp_exp / vp_exp.sum(axis=1, keepdims=True)
                v_old_for_seat = vp_probs[:, 0].astype(np.float32)
                vp_olds.append(torch.from_numpy(v_old_for_seat))
            else:
                lp_olds.append(torch.zeros(S))
                vp_olds.append(torch.zeros(S))

            if "final_vp" in d:
                fvp = d["final_vp"].numpy()
                fvp_rotated = rotate_value_targets_to_cp(fvp, players, n_p_arr)
                fvp_list.append(torch.from_numpy(fvp_rotated.astype(np.float32) / 10.0))
            else:
                fvp_list.append(torch.zeros(S, 4))

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
        all_returns = torch.cat(returns_list).to(device, non_blocking=True)
        all_lp_old = torch.cat(lp_olds).to(device, non_blocking=True)
        all_vp_old = torch.cat(vp_olds).to(device, non_blocking=True)
        all_fvp = torch.cat(fvp_list).to(device, non_blocking=True)

        if use_ppo:
            all_advantages = all_returns - all_vp_old
            adv_std = all_advantages.std()
            if adv_std > 1e-6:
                all_advantages = (all_advantages - all_advantages.mean()) / adv_std

        n = all_nf.shape[0]
        del nfs, efs, ffs, masks, acts, vts, sws, lp_olds, vp_olds, returns_list, fvp_list
        gc.collect()

        # Freeze everything except value head for initial rounds
        freeze_active = freeze_policy_rounds > 0 and round_num < (start_round + freeze_policy_rounds)
        if freeze_active:
            for name, p in net.named_parameters():
                p.requires_grad = "value_head" in name
            if round_num == start_round:
                print(f"[learner] Freezing policy for {freeze_policy_rounds} rounds, training value head only", flush=True)
        else:
            for p in net.parameters():
                p.requires_grad = True

        trainable = [p for p in net.parameters() if p.requires_grad]
        all_params = trainable + list(loss_combiner.parameters())
        rounds_into_run = round_num - start_round
        lr = max(1e-5, 3e-4 * (0.1 ** (rounds_into_run / 3000)))
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

        net.train()
        BS = cfg.batch_size
        PPO_EPOCHS = 1
        sums = {k: 0.0 for k in ("policy_loss", "value_loss", "total_loss",
                                   "entropy", "policy_acc", "value_acc")}
        n_batches = 0

        for ppo_epoch in range(PPO_EPOCHS):
            perm = torch.randperm(n, device=device)
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

                fvp_b = all_fvp[idx]
                if "vp_pred" in out and fvp_b.abs().sum() > 0:
                    vp_aux = F.mse_loss(out["vp_pred"], fvp_b)
                    v_loss = v_loss + 0.1 * vp_aux

                ent = masked_entropy(out["policy_logits"], mask_b)
                total, _ = loss_combiner(p_loss, v_loss, ent, 0.0)

                optimizer.zero_grad(set_to_none=True)
                total.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.gradient_clip)
                optimizer.step()

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

        try:
            del all_nf, all_ef, all_ff, all_mask, all_act, all_vt, all_sw
            del all_returns, all_lp_old, all_vp_old, all_fvp
            if use_ppo:
                del all_advantages
        except NameError:
            pass
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
                "train/lr": lr,
                "train/ppo": 1.0 if use_ppo else 0.0,
            })

        for fn in group:
            try:
                os.remove(os.path.join(pending_dir, fn))
            except FileNotFoundError:
                pass

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

            for depth in [0]:
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
                if depth == 0 and wr > best_wr:
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

    n_gpus = torch.cuda.device_count()
    num_actor_gpus = min(args.num_actor_gpus, n_gpus)
    print(f"Detected {n_gpus} GPUs, using {num_actor_gpus} for actors", flush=True)

    learner = ctx.Process(
        target=run_learner,
        args=(0, shard_dir, ckpt_dir, args.checkpoint,
              args.batch_size, args.shards_per_train,
              args.eval_games, args.eval_interval,
              args.max_rounds),
        kwargs={"wandb_name": args.wandb_name,
                "freeze_policy_rounds": args.freeze_policy_rounds},
        daemon=False,
    )
    learner.start()
    procs.append(learner)
    print(f"Started learner (pid={learner.pid}) on cuda:0", flush=True)

    total_actors = 0
    for g in range(num_actor_gpus):
        for a in range(args.actors_per_gpu):
            aid = total_actors
            actor_seed = args.seed + aid * 100_000
            actor = ctx.Process(
                target=run_actor,
                args=(g, aid, args.checkpoint, shard_dir, ckpt_dir,
                      args.search_depth, actor_seed, args.reload_interval,
                      args.max_pending, args.deep_search_depth,
                      args.mix_ab2, args.asymmetric),
                daemon=True,
            )
            actor.start()
            procs.append(actor)
            total_actors += 1

    print(f"\nSelf-play running: 1 learner + {total_actors} actors "
          f"({args.actors_per_gpu}/gpu × {num_actor_gpus} gpus)",
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
    parser.add_argument("--mix-ab2", action="store_true",
                        help="2 NN seats vs 2 AB2 (2-ply) seats per game")
    parser.add_argument("--freeze-policy-rounds", type=int, default=0,
                        help="Freeze encoder+trunk+policy for N rounds, train only value head")
    parser.add_argument("--asymmetric", action="store_true",
                        help="Asymmetric self-play: 2 NN seats vs 2 frozen-opponent seats")
    parser.add_argument("--use-ppo", action="store_true",
                        help="Use PPO loss instead of BC (default: BC with winner weighting)")
    args = parser.parse_args()

    if args.role == "all":
        run_all(args)
    elif args.role == "actor":
        run_actor(args.gpu_id, args.gpu_id, args.checkpoint, args.shard_dir,
                  args.ckpt_dir, args.search_depth, args.seed,
                  args.reload_interval, args.max_pending,
                  args.deep_search_depth, asymmetric=args.asymmetric)
    elif args.role == "learner":
        run_learner(args.gpu_id, args.shard_dir, args.ckpt_dir,
                    args.checkpoint, args.batch_size, args.shards_per_train,
                    args.eval_games, args.eval_interval, args.max_rounds,
                    wandb_name=args.wandb_name,
                    freeze_policy_rounds=args.freeze_policy_rounds,
                    force_ppo=args.use_ppo)


if __name__ == "__main__":
    main()
