"""Trace 0-ply games in excruciating detail for analysis."""

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


def action_type_label(idx: int) -> str:
    for name, (lo, hi) in ACTION_TYPE_RANGES.items():
        if lo <= idx < hi:
            return name
    return "unknown"


def describe_action(act, action_enc) -> str:
    t = act.type
    v = act.value
    name = ACTION_TYPES.get(t, f"type_{t}")
    if t == 4:  # BUILD_SETTLEMENT
        return f"{name}(node={v[0]})"
    if t == 5:  # BUILD_CITY
        return f"{name}(node={v[0]})"
    if t == 3:  # BUILD_ROAD
        return f"{name}(edge={v[0]}-{v[1]})"
    if t == 1:  # MOVE_ROBBER
        steal = f",steal=P{v[3]}" if v[3] >= 0 else ",no_steal"
        return f"{name}(tile=({v[0]},{v[1]},{v[2]}){steal})"
    if t == 11:  # MARITIME_TRADE
        return f"{name}(give={RESOURCES[v[0]] if 0<=v[0]<5 else v[0]},get={RESOURCES[v[4]] if 0<=v[4]<5 else v[4]})"
    if t == 8:  # YEAR_OF_PLENTY
        r1 = RESOURCES[v[0]] if 0 <= v[0] < 5 else str(v[0])
        r2 = RESOURCES[v[1]] if 0 <= v[1] < 5 else "none"
        return f"{name}({r1},{r2})"
    if t == 9:  # MONOPOLY
        return f"{name}({RESOURCES[v[0]] if 0<=v[0]<5 else v[0]})"
    if t == 2:  # DISCARD
        return f"{name}(resource={RESOURCES[v[0]] if 0<=v[0]<5 else v[0]})"
    return name


def get_vp_info(g) -> list[int]:
    """Extract VP for each player from the C game state."""
    state = g._game.state
    vps = []
    for seat in range(4):
        vp = state.player_state[seat][0]  # first field is VP
        vps.append(int(vp))
    return vps


def get_resource_counts(g) -> list[list[int]]:
    """Extract hand resources for each player."""
    state = g._game.state
    hands = []
    for seat in range(4):
        ps = state.player_state[seat]
        hand = [int(ps[i]) for i in range(1, 6)]
        hands.append(hand)
    return hands


def trace_games(
    net, state_enc, action_enc, device, lib,
    num_games=50, temperature=0.1, seed_offset=0,
):
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    AD = 337
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

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    logs = {i: [] for i in range(num_games)}
    move_count = [0] * num_games

    active = list(range(num_games))
    iteration = 0
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
                vals = []
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
                    vals.append(v)
                    if v > bv:
                        bv = v
                        bi = i

                chosen_act = le[bi]
                move_count[idx] += 1
                logs[idx].append({
                    "move": move_count[idx],
                    "turn": g.turn_number,
                    "player": f"P{cp}",
                    "seat_type": "AB2",
                    "action": describe_action(chosen_act, action_enc),
                    "action_idx": action_enc.encode(chosen_act),
                    "num_legal": len(le),
                    "value": round(bv, 4),
                })
                g.step(bi)
                progress = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # NN seats
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
                lo = out["policy_logits"][:, :AD] / temperature
                lo = lo.masked_fill(mask_t == 0, -1e9)
                pr = F.softmax(lo, dim=-1).cpu().numpy()[0]
                value_out = out["value"].cpu().numpy()[0]

            if pr.sum() < 1e-6:
                pr = mask_np / max(mask_np.sum(), 1e-8)
            pr = pr / pr.sum()

            aidx = int(np.random.choice(AD, p=pr))
            chosen_le_idx = next((i for i, a in enumerate(le) if action_enc.encode(a) == aidx), 0)
            chosen_act = le[chosen_le_idx]

            # Top-5 actions by probability
            top_indices = np.argsort(pr)[::-1][:5]
            top5 = []
            for ti in top_indices:
                if pr[ti] < 0.001:
                    break
                matching = [a for a in le if action_enc.encode(a) == ti]
                desc = describe_action(matching[0], action_enc) if matching else f"idx={ti}"
                top5.append({"action": desc, "prob": round(float(pr[ti]), 4),
                             "type": action_type_label(ti)})

            move_count[idx] += 1
            entry = {
                "move": move_count[idx],
                "turn": g.turn_number,
                "player": f"P{cp}",
                "seat_type": "NN",
                "action": describe_action(chosen_act, action_enc),
                "action_idx": aidx,
                "action_type": action_type_label(aidx),
                "chosen_prob": round(float(pr[aidx]), 4),
                "num_legal": len(le),
                "top5": top5,
                "value_head": [round(float(v), 4) for v in value_out],
            }

            if len(le) == 1:
                entry["forced"] = True

            logs[idx].append(entry)
            g.step(chosen_le_idx)

        iteration += 1

    # Collect results
    results = []
    for idx in range(num_games):
        w = games[idx].winner()
        winner_seat = None
        winner_type = None
        if w is not None:
            winner_seat = w
            winner_type = "NN" if w in hz_seats[idx] else "AB2"

        results.append({
            "game_idx": idx,
            "seed": 80000 + seed_offset * 1000 + idx,
            "hz_seats": sorted(hz_seats[idx]),
            "ab2_seats": sorted(ab2_seats[idx]),
            "winner_seat": winner_seat,
            "winner_type": winner_type,
            "total_moves": move_count[idx],
            "final_turn": games[idx].turn_number,
            "moves": logs[idx],
        })

    return results


def main():
    device = "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    action_enc = ActionEncoder()
    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()

    ckpt = "checkpoints/human_bot_experiment/epoch1.pt"
    net = HumanBotNet.load_checkpoint(ckpt, device=device)
    print(f"Loaded {ckpt} on {device}")

    np.random.seed(42)

    print("Tracing 50 games (0-ply, temperature=0.1)...")
    t0 = time.perf_counter()
    results = trace_games(net, state_enc, action_enc, device, lib,
                          num_games=50, temperature=0.1, seed_offset=0)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    # Save full trace
    out_path = "checkpoints/human_bot_experiment/game_traces.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full traces to {out_path}")

    # Summary
    wins = [r for r in results if r["winner_type"] == "NN"]
    losses = [r for r in results if r["winner_type"] == "AB2"]
    draws = [r for r in results if r["winner_type"] is None]
    print(f"\nNN wins: {len(wins)}  AB2 wins: {len(losses)}  Draws: {len(draws)}")
    for r in wins:
        print(f"  WIN:  game {r['game_idx']} (seed {r['seed']}) "
              f"winner=P{r['winner_seat']} turn={r['final_turn']} moves={r['total_moves']}")
    for r in draws:
        print(f"  DRAW: game {r['game_idx']} (seed {r['seed']}) "
              f"turn={r['final_turn']} moves={r['total_moves']}")


if __name__ == "__main__":
    main()
