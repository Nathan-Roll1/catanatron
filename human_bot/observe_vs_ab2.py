#!/usr/bin/env python3
"""Play NN vs AB2 games with detailed observation."""

from __future__ import annotations
import argparse
import ctypes
import numpy as np
import torch

RESOURCE_NAMES = ["Lumber", "Brick", "Sheep", "Wheat", "Ore"]
ACTION_TYPE_NAMES = {
    0: "ROLL", 1: "ROBBER", 2: "DISCARD", 3: "ROAD",
    4: "SETT", 5: "CITY", 6: "BUY_DEV",
    7: "KNIGHT", 8: "YOP", 9: "MONOPOLY",
    10: "ROAD_BUILD", 11: "TRADE", 17: "END",
}
AD = 337


def describe_action(act):
    atype = ACTION_TYPE_NAMES.get(act.type, f"T{act.type}")
    v = [act.value[i] for i in range(5)]
    if act.type == 0: return "ROLL"
    if act.type == 17: return "END"
    if act.type == 1:
        victim = f"|P{v[3]}" if v[3] >= 0 else ""
        return f"ROB({v[0]},{v[1]},{v[2]}{victim})"
    if act.type == 2:
        return f"DISC({RESOURCE_NAMES[v[0]][0:2]})"
    if act.type == 3: return f"ROAD({v[0]}-{v[1]})"
    if act.type == 4: return f"SETT(n{v[0]})"
    if act.type == 5: return f"CITY(n{v[0]})"
    if act.type == 6: return "DEV"
    if act.type == 11:
        return f"TR({RESOURCE_NAMES[v[0]][0:2]}->{RESOURCE_NAMES[v[4]][0:2]})"
    return f"{atype}({v})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/selfplay_v2/latest.pt")
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=80000)
    args = parser.parse_args()

    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS
    from human_bot.model import HumanBotNet

    lib = load_library()
    net = HumanBotNet.load_checkpoint(args.checkpoint, device=args.device)
    net.eval()
    ae = ActionEncoder()

    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    edge_index = se._edge_index.to(args.device)
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    child = CGame()
    child_acts = (CAction * MAX_ACTIONS)()
    child_cnt = ctypes.c_int(0)

    hz_total, ab2_total = 0, 0

    for gi in range(args.num_games):
        seed = args.seed + gi
        game = CatanGame(seed=seed)
        game.reset()

        hz_seats = {gi % 4, (gi + 2) % 4}
        ab2_seats = {(gi + 1) % 4, (gi + 3) % 4}

        nf = np.zeros((N, NF), dtype=np.float32)
        ef = np.zeros((E, EF), dtype=np.float32)
        ff = np.zeros(FF, dtype=np.float32)

        print(f"\n{'='*70}")
        print(f"  Game {gi+1}  seed={seed}  NN={sorted(hz_seats)}  AB2={sorted(ab2_seats)}")
        print(f"{'='*70}")

        last_turn = -1
        step = 0

        while not game.is_terminal() and game.turn_number < 500:
            le = game.get_legal_actions()
            if not le:
                break

            cp = game.current_player()
            turn = game.turn_number
            is_nn = cp in hz_seats

            if turn != last_turn and (turn <= 8 or turn % 20 == 0):
                last_turn = turn
                se.encode_into(game.get_state_view(), nf, ef, ff)
                vps = []
                for p in range(4):
                    vp = int(round(ff[p * 24] * 14))
                    tag = "NN" if p in hz_seats else "AB"
                    vps.append(f"P{p}({tag}):{vp}")
                print(f"\n--- Turn {turn} --- {' '.join(vps)} ---")

            if len(le) == 1:
                chosen = 0
            elif is_nn:
                se.encode_into(game.get_state_view(), nf, ef, ff)
                mask_np = ae.get_action_mask(le).numpy()
                mask_full = np.zeros(397, dtype=np.float32)
                mask_full[:len(mask_np)] = mask_np
                batch = {
                    "node_features": torch.from_numpy(nf[None]).to(args.device),
                    "edge_index": edge_index,
                    "edge_features": torch.from_numpy(ef[None]).to(args.device),
                    "flat_features": torch.from_numpy(ff[None]).to(args.device),
                    "action_mask": torch.from_numpy(mask_full[None]).to(args.device),
                }
                with torch.no_grad():
                    out = net(batch)
                logits = out["policy_logits"][0, :AD].cpu().numpy()
                enc_idx = int(np.argmax(logits))
                chosen = 0
                for i, a in enumerate(le):
                    try:
                        if ae.encode(a) == enc_idx:
                            chosen = i
                            break
                    except ValueError:
                        continue
            else:
                cg = game._game
                bc = cg.state.colors[cg.state.current_player_index]
                best_i, best_v = 0, -1e30
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(child), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(child), act,
                                     child_acts, ctypes.byref(child_cnt))
                    v = lib.base_value_fn(ctypes.byref(child), bc)
                    if v > best_v:
                        best_v = v
                        best_i = i
                chosen = best_i

            act = le[chosen]
            if act.type not in (0, 17, 2):
                tag = "NN" if is_nn else "AB"
                desc = describe_action(act)
                print(f"  T{turn:>3d} P{cp}({tag}) -> {desc}")

            game.step(chosen)
            step += 1

        winner = game.winner()
        print(f"\n  Result: ", end="")
        if winner is not None:
            tag = "NN" if winner in hz_seats else "AB2"
            print(f"P{winner} ({tag}) wins at turn {game.turn_number} ({step} steps)")
            if winner in hz_seats:
                hz_total += 1
            else:
                ab2_total += 1
        else:
            print(f"No winner after {game.turn_number} turns")

    print(f"\n{'='*70}")
    print(f"  TOTAL: NN={hz_total}  AB2={ab2_total}  "
          f"WR={hz_total/max(hz_total+ab2_total,1):.0%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
