#!/usr/bin/env python3
"""Play and observe full games with a trained model. Prints every action in detail."""

from __future__ import annotations
import argparse
import numpy as np
import torch

RESOURCE_NAMES = ["Lumber", "Brick", "Sheep", "Wheat", "Ore"]
ACTION_TYPE_NAMES = {
    0: "ROLL", 1: "MOVE_ROBBER", 2: "DISCARD", 3: "BUILD_ROAD",
    4: "BUILD_SETTLEMENT", 5: "BUILD_CITY", 6: "BUY_DEV",
    7: "PLAY_KNIGHT", 8: "PLAY_YOP", 9: "PLAY_MONOPOLY",
    10: "PLAY_ROAD_BUILDING", 11: "MARITIME_TRADE", 12: "OFFER_TRADE",
    13: "ACCEPT_TRADE", 14: "REJECT_TRADE", 15: "CONFIRM_TRADE",
    16: "CANCEL_TRADE", 17: "END_TURN",
}
AD = 337


def describe_action(act, ff=None):
    atype = ACTION_TYPE_NAMES.get(act.type, f"TYPE_{act.type}")
    vals = [act.value[i] for i in range(5)]

    if act.type == 0:
        return "ROLL"
    elif act.type == 17:
        return "END_TURN"
    elif act.type == 1:
        victim = f"|P{vals[3]}" if vals[3] >= 0 else ""
        return f"ROBBER({vals[0]},{vals[1]},{vals[2]}{victim})"
    elif act.type == 2:
        res = RESOURCE_NAMES[vals[0]] if 0 <= vals[0] < 5 else f"r{vals[0]}"
        return f"DISCARD({res})"
    elif act.type == 3:
        return f"ROAD({vals[0]}-{vals[1]})"
    elif act.type == 4:
        return f"SETTLEMENT(n{vals[0]})"
    elif act.type == 5:
        return f"CITY(n{vals[0]})"
    elif act.type == 6:
        return "BUY_DEV"
    elif act.type in (7, 8, 9, 10):
        return atype
    elif act.type == 11:
        give = RESOURCE_NAMES[vals[0]] if 0 <= vals[0] < 5 else f"r{vals[0]}"
        get = RESOURCE_NAMES[vals[4]] if 0 <= vals[4] < 5 else f"r{vals[4]}"
        return f"TRADE({give}->{get})"
    else:
        return f"{atype}({vals})"


def resources_str(ff, player_offset=0):
    base = player_offset * 24
    res = []
    for i in range(5):
        count = int(round(ff[base + 1 + i] * 19))
        if count > 0:
            res.append(f"{count}{RESOURCE_NAMES[i][0]}")
    return " ".join(res) if res else "empty"


def play_and_observe(checkpoint_path, seed, device="cpu"):
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from human_bot.model import HumanBotNet

    load_library()
    net = HumanBotNet.load_checkpoint(checkpoint_path, device=device)
    net.eval()
    ae = ActionEncoder()
    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    edge_index = se._edge_index.to(device)

    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FF, dtype=np.float32)

    print(f"\n{'='*70}")
    print(f"  Game seed={seed}  |  Device={device}")
    print(f"{'='*70}")

    step_count = 0
    last_turn = -1
    player_vp = [0, 0, 0, 0]
    player_stats = [{
        "settlements": 0, "cities": 0, "roads": 0, "dev": 0,
        "knights": 0, "trades": 0, "steals": 0,
    } for _ in range(4)]

    while not game.is_terminal() and game.turn_number < 750:
        le = game.get_legal_actions()
        if not le:
            break

        cp = game.current_player()
        turn = game.turn_number

        sv = game.get_state_view()
        se.encode_into(sv, nf, ef, ff)
        mask_np = ae.get_action_mask(le).numpy()
        mask_full = np.zeros(397, dtype=np.float32)
        mask_full[:len(mask_np)] = mask_np

        if turn != last_turn:
            last_turn = turn
            vps = []
            for p in range(4):
                vp = int(round(ff[p * 24] * 14))
                player_vp[p] = vp
                vps.append(f"P{p}:{vp}")
            if turn % 10 == 0 or turn <= 8:
                print(f"\n--- Turn {turn} --- VP: {' '.join(vps)} ---")

        if len(le) == 1:
            chosen = 0
        else:
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
            chosen = 0
            for i, a in enumerate(le):
                try:
                    if ae.encode(a) == enc_idx:
                        chosen = i
                        break
                except ValueError:
                    continue

            value = out["value"][0].cpu().numpy()
            win_probs = np.exp(value) / np.exp(value).sum()

        act = le[chosen]
        desc = describe_action(act, ff)
        res = resources_str(ff, 0)

        if act.type in (0, 17, 2):
            pass
        else:
            neg_resources = False
            for i in range(5):
                count = int(round(ff[1 + i] * 19))
                if count < 0:
                    neg_resources = True
            neg_flag = " *** NEG RESOURCES ***" if neg_resources else ""
            print(f"  T{turn:>3d} P{cp} [{res}] -> {desc}{neg_flag}")

        if act.type == 4:
            player_stats[cp]["settlements"] += 1
        elif act.type == 5:
            player_stats[cp]["cities"] += 1
        elif act.type == 3:
            player_stats[cp]["roads"] += 1
        elif act.type == 6:
            player_stats[cp]["dev"] += 1
        elif act.type == 7:
            player_stats[cp]["knights"] += 1
        elif act.type == 11:
            player_stats[cp]["trades"] += 1
        elif act.type == 1 and act.value[3] >= 0:
            player_stats[cp]["steals"] += 1

        game.step(chosen)
        step_count += 1

    winner = game.winner()
    print(f"\n{'='*70}")
    if winner is not None:
        print(f"  WINNER: Player {winner} at turn {game.turn_number} "
              f"({step_count} steps)")
    else:
        print(f"  NO WINNER after {game.turn_number} turns ({step_count} steps)")

    print(f"\n  Player Stats:")
    for p in range(4):
        s = player_stats[p]
        print(f"    P{p}: VP={player_vp[p]}  S={s['settlements']} C={s['cities']} "
              f"R={s['roads']} Dev={s['dev']} K={s['knights']} "
              f"Tr={s['trades']} Steal={s['steals']}")

    for i in range(5):
        bank = int(round(ff[96 + i] * 19))
        if bank < 0:
            print(f"\n  *** BANK HAS NEGATIVE {RESOURCE_NAMES[i]}: {bank} ***")

    print(f"{'='*70}\n")
    return winner


def main():
    parser = argparse.ArgumentParser(description="Observe model playing full games")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/selfplay_v2/latest.pt")
    parser.add_argument("--num-games", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    wins = [0, 0, 0, 0]
    for i in range(args.num_games):
        w = play_and_observe(args.checkpoint, args.seed + i, args.device)
        if w is not None:
            wins[w] += 1

    print(f"Results: {args.num_games} games")
    print(f"  Wins: {wins}")
    print(f"  Win rates: {[f'{w/args.num_games:.0%}' for w in wins]}")


if __name__ == "__main__":
    main()
