#!/usr/bin/env python3
"""Play 0-ply NN vs 5-ply NN (same model, different search depths)."""

from __future__ import annotations
import argparse
import numpy as np
import torch

AD = 337
MASK_DIM = 397


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/selfplay_v2/latest.pt")
    parser.add_argument("--num-games", type=int, default=10)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=50000)
    args = parser.parse_args()

    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from human_bot.model import HumanBotNet
    from human_bot.selfplay import _value_search_deep, _policy_argmax, _is_important_position

    load_library()
    net = HumanBotNet.load_checkpoint(args.checkpoint, device=args.device)
    net.eval()
    ae = ActionEncoder()

    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    edge_index = se._edge_index.to(args.device)
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    zero_wins, five_wins, draws = 0, 0, 0

    for gi in range(args.num_games):
        seed = args.seed + gi
        game = CatanGame(seed=seed)
        game.reset()

        zero_seats = {gi % 4, (gi + 2) % 4}
        five_seats = {(gi + 1) % 4, (gi + 3) % 4}

        nf = np.zeros((N, NF), dtype=np.float32)
        ef = np.zeros((E, EF), dtype=np.float32)
        ff = np.zeros(FF, dtype=np.float32)

        step = 0
        while not game.is_terminal() and game.turn_number < 500:
            le = game.get_legal_actions()
            if not le:
                break

            cp = game.current_player()
            is_5ply = cp in five_seats

            if len(le) == 1:
                chosen = 0
            elif game.turn_number <= 7:
                se.encode_into(game.get_state_view(), nf, ef, ff)
                mask = ae.get_action_mask(le).numpy()
                chosen = _policy_argmax(nf, ef, ff, mask, net, ae,
                                        edge_index, args.device, le)
            elif is_5ply and _is_important_position(le):
                chosen = _value_search_deep(game, le, net, se, ae,
                                            edge_index, args.device,
                                            max_depth=5)
            elif is_5ply:
                from human_bot.selfplay import _value_search_1ply
                chosen = _value_search_1ply(game, le, net, se, ae,
                                            edge_index, args.device)
            else:
                se.encode_into(game.get_state_view(), nf, ef, ff)
                mask = ae.get_action_mask(le).numpy()
                mask_full = np.zeros(MASK_DIM, dtype=np.float32)
                mask_full[:len(mask)] = mask
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

            game.step(chosen)
            step += 1

        winner = game.winner()
        if winner is None:
            result = "DRAW"
            draws += 1
        elif winner in five_seats:
            result = "5-PLY"
            five_wins += 1
        else:
            result = "0-PLY"
            zero_wins += 1

        se.encode_into(game.get_state_view(), nf, ef, ff)
        vps = []
        for p in range(4):
            vp = int(round(ff[p * 24] * 14))
            tag = "5p" if p in five_seats else "0p"
            vps.append(f"P{p}({tag}):{vp}")

        print(f"Game {gi+1:>2d}  seed={seed}  T={game.turn_number:>3d}  "
              f"steps={step:>4d}  winner={result:<5s}  {' '.join(vps)}")

    total = zero_wins + five_wins + draws
    print(f"\nResults ({args.num_games} games):")
    print(f"  0-ply wins: {zero_wins}  ({zero_wins/max(total,1):.0%})")
    print(f"  5-ply wins: {five_wins}  ({five_wins/max(total,1):.0%})")
    print(f"  Draws:      {draws}")


if __name__ == "__main__":
    main()
