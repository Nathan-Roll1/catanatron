"""Trace 1-ply value-search games in detail for analysis."""

from __future__ import annotations

import ctypes
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from human_bot.evaluate import ACTION_TYPE_RANGES
from human_bot.model import HumanBotNet

RESOURCES = ["Lumber", "Ore", "Wool", "Grain", "Brick"]
ACTION_TYPES = {
    0: "ROLL", 1: "MOVE_ROBBER", 2: "DISCARD", 3: "BUILD_ROAD",
    4: "BUILD_SETTLEMENT", 5: "BUILD_CITY", 6: "BUY_DEV_CARD",
    7: "PLAY_KNIGHT", 8: "YEAR_OF_PLENTY", 9: "MONOPOLY",
    10: "ROAD_BUILDING", 11: "MARITIME_TRADE", 12: "OFFER_TRADE",
    13: "ACCEPT_TRADE", 14: "REJECT_TRADE", 15: "CONFIRM_TRADE",
    16: "CANCEL_TRADE", 17: "END_TURN",
}


def action_type_label(idx):
    for name, (lo, hi) in ACTION_TYPE_RANGES.items():
        if lo <= idx < hi:
            return name
    return "unknown"


def describe_action(act):
    t, v = act.type, act.value
    name = ACTION_TYPES.get(t, f"type_{t}")
    if t == 4: return f"{name}(node={v[0]})"
    if t == 5: return f"{name}(node={v[0]})"
    if t == 3: return f"{name}(edge={v[0]}-{v[1]})"
    if t == 1:
        steal = f",steal=P{v[3]}" if v[3] >= 0 else ",no_steal"
        return f"{name}(tile=({v[0]},{v[1]},{v[2]}){steal})"
    if t == 11:
        g = RESOURCES[v[0]] if 0 <= v[0] < 5 else str(v[0])
        r = RESOURCES[v[4]] if 0 <= v[4] < 5 else str(v[4])
        return f"{name}(give={g},get={r})"
    if t == 9: return f"{name}({RESOURCES[v[0]] if 0<=v[0]<5 else v[0]})"
    if t == 2: return f"{name}(res={RESOURCES[v[0]] if 0<=v[0]<5 else v[0]})"
    return name


def trace_1ply_games(net, state_enc, action_enc, device, lib,
                     num_games=50, seed_offset=999):
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    N, E = state_enc.num_nodes, state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM
    edge_index_dev = state_enc._edge_index.to(device)
    net.eval()

    games = [CatanGame(seed=80000 + seed_offset * 1000 + i) for i in range(num_games)]
    for g in games:
        g.reset()

    hz_seats = [{i % 4, (i + 2) % 4} for i in range(num_games)]
    ab2_seats = [{(i + 1) % 4, (i + 3) % 4} for i in range(num_games)]
    _mar_received: dict[tuple[int, int], set[int]] = {}

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    logs = {i: [] for i in range(num_games)}
    move_count = [0] * num_games

    active = list(range(num_games))
    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # AB2 seats
        progress = True
        while progress:
            progress = False
            for idx in active:
                g = games[idx]
                if g.is_terminal() or g.turn_number >= 1000:
                    continue
                cp = g.current_player()
                if cp not in ab2_seats[idx]:
                    continue
                le = g.get_legal_actions()
                if not le:
                    continue
                cg = g._game
                bc = cg.state.colors[cg.state.current_player_index]
                bi, bv = 0, -1e30
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
                    if v > bv:
                        bv = v
                        bi = i
                chosen_act = le[bi]
                move_count[idx] += 1
                logs[idx].append({
                    "move": move_count[idx], "turn": g.turn_number,
                    "player": f"P{cp}", "seat_type": "AB2",
                    "action": describe_action(chosen_act),
                    "num_legal": len(le),
                })
                g.step(bi)
                progress = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # NN 1-ply seats
        for idx in active:
            g = games[idx]
            if g.is_terminal() or g.turn_number >= 1000:
                continue
            cp = g.current_player()
            if cp not in hz_seats[idx]:
                continue
            le = g.get_legal_actions()
            if not le:
                continue

            our_seat = cp

            if len(le) == 1:
                act = le[0]
                if act.type == 11:
                    _mar_received.setdefault((idx, cp), set()).add(act.value[4])
                else:
                    _mar_received.pop((idx, cp), None)
                move_count[idx] += 1
                logs[idx].append({
                    "move": move_count[idx], "turn": g.turn_number,
                    "player": f"P{cp}", "seat_type": "NN",
                    "action": describe_action(le[0]), "forced": True,
                    "num_legal": 1,
                })
                g.step(0)
                continue

            # Filter circular maritime trades
            received = _mar_received.get((idx, cp), set())
            if received:
                filtered = [a for a in le if a.type != 11 or a.value[0] not in received]
                if filtered:
                    le = filtered

            # 1-ply: evaluate each candidate
            B = len(le)
            nf_buf = np.zeros((B, N, NF), dtype=np.float32)
            ef_buf = np.zeros((B, E, EF), dtype=np.float32)
            ff_buf = np.zeros((B, FF), dtype=np.float32)
            child_current = np.zeros(B, dtype=np.int32)
            child_terminal = np.zeros(B, dtype=bool)
            child_winner = np.full(B, -1, dtype=np.int32)
            nt = 0

            for ai in range(B):
                gc = g.clone()
                gc.step(ai)
                if gc.is_terminal():
                    child_terminal[ai] = True
                    w = gc.winner()
                    child_winner[ai] = w if w is not None else -1
                else:
                    sv = gc.get_state_view()
                    state_enc.encode_into(sv, nf_buf[nt], ef_buf[nt], ff_buf[nt])
                    child_current[ai] = gc.current_player()
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
                for ai in range(B):
                    if not child_terminal[ai]:
                        values[ai] = rv[vi]
                        vi += 1

            scored = []
            for ai in range(B):
                if child_terminal[ai]:
                    if child_winner[ai] == our_seat:
                        v = 10.0
                    elif child_winner[ai] >= 0:
                        v = -10.0
                    else:
                        v = 0.0
                else:
                    offset = (our_seat - child_current[ai]) % 4
                    v = float(values[ai, offset])
                scored.append((v, ai))

            scored.sort(reverse=True)
            best_val, best_ai = scored[0]

            # Log top-5 by value
            top5 = []
            for v, ai in scored[:5]:
                top5.append({
                    "action": describe_action(le[ai]),
                    "value": round(v, 4),
                    "value_vec": [round(float(x), 4) for x in values[ai]] if not child_terminal[ai] else None,
                })

            chosen_act = le[best_ai]
            if chosen_act.type == 11:
                _mar_received.setdefault((idx, cp), set()).add(chosen_act.value[4])
            else:
                _mar_received.pop((idx, cp), None)

            move_count[idx] += 1
            logs[idx].append({
                "move": move_count[idx], "turn": g.turn_number,
                "player": f"P{cp}", "seat_type": "NN",
                "action": describe_action(chosen_act),
                "chosen_value": round(best_val, 4),
                "num_legal": len(le),
                "top5_by_value": top5,
            })
            g.step(best_ai)

    results = []
    for idx in range(num_games):
        w = games[idx].winner()
        results.append({
            "game_idx": idx,
            "seed": 80000 + seed_offset * 1000 + idx,
            "hz_seats": sorted(hz_seats[idx]),
            "ab2_seats": sorted(ab2_seats[idx]),
            "winner_seat": w,
            "winner_type": "NN" if w is not None and w in hz_seats[idx]
                           else ("AB2" if w is not None else None),
            "total_moves": move_count[idx],
            "final_turn": games[idx].turn_number,
            "moves": logs[idx],
        })
    return results


def main():
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    action_enc = ActionEncoder()
    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()

    ckpt = "checkpoints/human_bot_experiment/latest.pt"
    net = HumanBotNet.load_checkpoint(ckpt, device=device)
    np.random.seed(42)

    print(f"Tracing 50 1-ply games ...")
    t0 = time.perf_counter()
    results = trace_1ply_games(net, state_enc, action_enc, device, lib,
                               num_games=50, seed_offset=999)
    print(f"Done in {time.perf_counter()-t0:.1f}s")

    out_path = "checkpoints/human_bot_experiment/trace_1ply.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    wins = [r for r in results if r["winner_type"] == "NN"]
    losses = [r for r in results if r["winner_type"] == "AB2"]
    draws = [r for r in results if r["winner_type"] is None]
    print(f"\nNN wins: {len(wins)}  AB2 wins: {len(losses)}  Draws: {len(draws)}")
    for r in wins:
        print(f"  WIN:  game {r['game_idx']} turn={r['final_turn']} moves={r['total_moves']}")
    for r in draws:
        print(f"  DRAW: game {r['game_idx']} turn={r['final_turn']}")


if __name__ == "__main__":
    main()
