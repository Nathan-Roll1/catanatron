"""Head-to-head: two NN checkpoints play each other, 2 seats each, 2-ply.

Usage:
    python3 -m human_bot.head_to_head \
      --model-a checkpoints/human_bot_pipeline/stage2_human_epoch1.pt \
      --model-b checkpoints/human_bot_pipeline/stage3_human_epoch2.pt \
      --num-games 20
"""

from __future__ import annotations

import argparse
import ctypes
import time

import numpy as np
import torch
import torch.nn.functional as F

from human_bot.model import HumanBotNet
from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal


def head_to_head(
    net_a, net_b, state_enc, action_enc, device, lib,
    num_games=20, seed_offset=0, depth_a=2, depth_b=2,
):
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM
    edge_index_dev = state_enc._edge_index.to(device)

    net_a.eval()
    net_b.eval()

    games = [CatanGame(seed=90000 + seed_offset * 1000 + i) for i in range(num_games)]
    for g in games:
        g.reset()

    a_seats = [{i % 4, (i + 2) % 4} for i in range(num_games)]
    b_seats = [{(i + 1) % 4, (i + 3) % 4} for i in range(num_games)]

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    _mar_received: dict[tuple[int, int], set[int]] = {}
    active = list(range(num_games))

    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        any_moved = False
        for idx in active:
            g = games[idx]
            if g.is_terminal() or g.turn_number >= 1000:
                continue
            cp = g.current_player()
            le = g.get_legal_actions()
            if not le:
                continue

            if cp in a_seats[idx]:
                net = net_a
                opp_seats = b_seats[idx]
                depth = depth_a
            else:
                net = net_b
                opp_seats = a_seats[idx]
                depth = depth_b

            if len(le) == 1:
                act = le[0]
                if act.type == 11:
                    _mar_received.setdefault((idx, cp), set()).add(act.value[4])
                else:
                    _mar_received.pop((idx, cp), None)
                g.step(0)
                any_moved = True
                continue

            received = _mar_received.get((idx, cp), set())
            if received:
                filtered = [a for a in le if a.type != 11 or a.value[0] not in received]
                if filtered:
                    le_use = filtered
                    idx_map = [le.index(a) for a in le_use]
                else:
                    le_use = le
                    idx_map = list(range(len(le)))
            else:
                le_use = le
                idx_map = list(range(len(le)))

            chosen = _value_pick(
                g, le_use, cp, opp_seats, net, state_enc, action_enc,
                device, edge_index_dev, lib, N, E, NF, EF, FF,
                ch, ca, cn, depth=depth,
            )
            chosen = idx_map[chosen]

            act = le[chosen]
            if act.type == 11:
                _mar_received.setdefault((idx, cp), set()).add(act.value[4])
            else:
                _mar_received.pop((idx, cp), None)

            g.step(chosen)
            any_moved = True

        if not any_moved:
            break

    a_wins = b_wins = draws = 0
    for idx in range(num_games):
        w = games[idx].winner()
        if w is None:
            draws += 1
        elif w in a_seats[idx]:
            a_wins += 1
        else:
            b_wins += 1

    return {"a_wins": a_wins, "b_wins": b_wins, "draws": draws}


def _value_pick(g, le, our_seat, opp_seats, net, state_enc, action_enc,
                device, edge_index_dev, lib, N, E, NF, EF, FF,
                ch, ca, cn, depth=2, top_k=5):
    """N-ply search with policy-guided pruning at depth 3."""
    AD = 337

    # At depth >= 3, restrict to top-k by policy
    candidates = list(range(len(le)))
    if len(le) > top_k and depth >= 3:
        candidates = _policy_top_k(
            g, le, net, action_enc, state_enc,
            device, edge_index_dev, AD, N, E, NF, EF, FF, top_k,
        )

    B = len(candidates)
    nf_buf = np.zeros((B, N, NF), dtype=np.float32)
    ef_buf = np.zeros((B, E, EF), dtype=np.float32)
    ff_buf = np.zeros((B, FF), dtype=np.float32)
    terminal = np.zeros(B, dtype=np.float32)
    terminal_val = np.zeros(B, dtype=np.float32)
    child_current = np.zeros(B, dtype=np.int32)
    nt = 0

    for bi, ai in enumerate(candidates):
        gc = g.clone()
        gc.step(ai)

        # Ply 2: opponent response
        if depth >= 2 and not gc.is_terminal():
            cp2 = gc.current_player()
            if cp2 in opp_seats:
                _greedy_respond(gc, lib, ch, ca, cn)

        # Ply 3: our follow-up (policy argmax)
        if depth >= 3 and not gc.is_terminal():
            cp3 = gc.current_player()
            if cp3 not in opp_seats:
                _nn_argmax_respond(gc, net, action_enc, state_enc,
                                   device, edge_index_dev, AD, N, E, NF, EF, FF)

        if gc.is_terminal():
            terminal[bi] = 1.0
            w = gc.winner()
            if w is not None and w == our_seat:
                terminal_val[bi] = 10.0
            elif w is not None:
                terminal_val[bi] = -10.0
        else:
            sv = gc.get_state_view()
            state_enc.encode_into(sv, nf_buf[nt], ef_buf[nt], ff_buf[nt])
            child_current[bi] = gc.current_player()
            nt += 1

    values = np.zeros((B, 4), dtype=np.float32)
    if nt > 0:
        with torch.no_grad():
            batch = {
                "node_features": torch.from_numpy(nf_buf[:nt].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:nt].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:nt].copy()).to(device),
            }
            out = net(batch)
            rv = out["value"].cpu().numpy()
        vi = 0
        for bi in range(B):
            if terminal[bi] == 0:
                values[bi] = rv[vi]
                vi += 1

    best_bi, best_val = 0, -1e30
    for bi in range(B):
        if terminal[bi] > 0:
            v = terminal_val[bi]
        else:
            offset = (our_seat - child_current[bi]) % 4
            v = float(values[bi, offset])
        v = apply_action_bonus(v, le[candidates[bi]])
        if v > best_val:
            best_val = v
            best_bi = bi

    chosen = candidates[best_bi]
    chosen = fix_robber_steal(chosen, le)
    return chosen


def _policy_top_k(g, le, net, action_enc, state_enc,
                  device, edge_index_dev, AD, N, E, NF, EF, FF, k):
    nf = np.zeros((1, N, NF), dtype=np.float32)
    ef = np.zeros((1, E, EF), dtype=np.float32)
    ff = np.zeros((1, FF), dtype=np.float32)
    state_enc.encode_into(g.get_state_view(), nf[0], ef[0], ff[0])
    mask_np = action_enc.get_action_mask(le).numpy()

    with torch.no_grad():
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        pad = torch.zeros(1, 397 - AD, device=device)
        mask_397 = torch.cat([mask_t, pad], dim=1)
        batch = {
            "node_features": torch.from_numpy(nf.copy()).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(ef.copy()).to(device),
            "flat_features": torch.from_numpy(ff.copy()).to(device),
            "action_mask": mask_397,
        }
        out = net(batch)
        logits = out["policy_logits"][0, :AD].cpu().numpy()

    action_to_le = {}
    for i, act in enumerate(le):
        action_to_le[action_enc.encode(act)] = i
    scored = sorted([(logits[enc], le_idx) for enc, le_idx in action_to_le.items()], reverse=True)
    return [le_idx for _, le_idx in scored[:k]]


def _nn_argmax_respond(gc, net, action_enc, state_enc,
                       device, edge_index_dev, AD, N, E, NF, EF, FF):
    if gc.is_terminal():
        return
    le = gc.get_legal_actions()
    if not le:
        return
    if len(le) == 1:
        gc.step(0)
        return

    nf = np.zeros((1, N, NF), dtype=np.float32)
    ef = np.zeros((1, E, EF), dtype=np.float32)
    ff = np.zeros((1, FF), dtype=np.float32)
    state_enc.encode_into(gc.get_state_view(), nf[0], ef[0], ff[0])
    mask_np = action_enc.get_action_mask(le).numpy()

    with torch.no_grad():
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        pad = torch.zeros(1, 397 - AD, device=device)
        mask_397 = torch.cat([mask_t, pad], dim=1)
        batch = {
            "node_features": torch.from_numpy(nf.copy()).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(ef.copy()).to(device),
            "flat_features": torch.from_numpy(ff.copy()).to(device),
            "action_mask": mask_397,
        }
        out = net(batch)
        logits = out["policy_logits"][0, :AD]
        logits = logits.masked_fill(mask_t[0] == 0, -1e9)
        best_aidx = int(logits.argmax().item())

    chosen = next((i for i, a in enumerate(le) if action_enc.encode(a) == best_aidx), 0)
    gc.step(chosen)


def _greedy_respond(gc, lib, ch, ca, cn):
    le = gc.get_legal_actions()
    if not le:
        return
    if len(le) == 1:
        gc.step(0)
        return
    cg = gc._game
    bc = cg.state.colors[cg.state.current_player_index]
    bi, bv = 0, -1e30
    for i, act in enumerate(le):
        lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
        lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
        v = lib.base_value_fn(ctypes.byref(ch), bc)
        if v > bv:
            bv = v
            bi = i
    gc.step(bi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--depth-a", type=int, default=2)
    parser.add_argument("--depth-b", type=int, default=2)
    parser.add_argument("--num-games", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    action_enc = ActionEncoder()
    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()

    net_a = HumanBotNet.load_checkpoint(args.model_a, device=device)
    net_b = HumanBotNet.load_checkpoint(args.model_b, device=device)
    print(f"Model A: {args.model_a} ({args.depth_a}-ply)")
    print(f"Model B: {args.model_b} ({args.depth_b}-ply)")
    print(f"{args.num_games} games, 2 seats each\n")

    t0 = time.perf_counter()
    result = head_to_head(
        net_a, net_b, state_enc, action_enc, device, lib,
        num_games=args.num_games, seed_offset=args.seed_offset,
        depth_a=args.depth_a, depth_b=args.depth_b,
    )
    sec = time.perf_counter() - t0

    print(f"Model A wins: {result['a_wins']}")
    print(f"Model B wins: {result['b_wins']}")
    print(f"Draws:        {result['draws']}")
    total = result["a_wins"] + result["b_wins"]
    if total > 0:
        print(f"\nA win rate: {result['a_wins']/total:.1%}")
        print(f"B win rate: {result['b_wins']/total:.1%}")
    print(f"Time: {sec:.1f}s")


if __name__ == "__main__":
    main()
