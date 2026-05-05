#!/usr/bin/env python3
"""Watch NN vs AB2 games with detailed move-by-move output.

Usage:
    python -m human_bot.observe --checkpoint checkpoints/sp_latest.pt --num-games 3
"""
from __future__ import annotations

import argparse
import ctypes
import numpy as np
import torch
import torch.nn.functional as F

from human_bot.model import HumanBotNet
from hexzero.game.interface import CatanGame
from hexzero.encoder.action_encoder import ActionEncoder
from hexzero.bindings.lib_loader import load_library
from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS

AD = 337
MASK_DIM = 397
TYPE_NAMES = {
    0: "ROLL", 1: "END_TURN", 2: "DISCARD", 3: "ROAD", 4: "SETTLE",
    5: "CITY", 6: "BUY_DEV", 7: "KNIGHT", 8: "ROAD_BUILD",
    9: "YEAR_PLENTY", 10: "MONOPOLY", 11: "TRADE", 13: "ROBBER",
}
INTERESTING_TYPES = {3, 4, 5, 6, 7, 8, 9, 10, 11}


def observe_game(seed, net, se, ae, edge_index, device, lib, verbose=True):
    game = CatanGame(seed=seed)
    game.reset()

    nn_seats = {seed % 4, (seed + 2) % 4}
    ab2_seats = {(seed + 1) % 4, (seed + 3) % 4}

    N, E = se.num_nodes, se.num_edges
    NF = se.NODE_FEATURE_DIM
    EF = se.EDGE_FEATURE_DIM
    FF = se.FLAT_FEATURE_DIM
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
            if cn.value > 0 and lib.game_winning_color(ctypes.byref(ch)) < 0:
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

    stats = {
        "nn_trades": 0, "ab2_trades": 0,
        "nn_builds": 0, "ab2_builds": 0,
        "nn_trade_log": [], "end_turn_skips": [],
    }
    step = 0
    last_turn = -1

    while not game.is_terminal() and game.turn_number < 500 and step < 2000:
        le = game.get_legal_actions()
        if not le:
            break

        cp = game.current_player()
        turn = game.turn_number

        if len(le) == 1:
            game.step(0)
            step += 1
            continue

        sv = game.get_state_view()
        se.encode_into(sv, nf, ef, ff)
        mask = ae.get_action_mask(le).numpy()
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
        values = F.softmax(out["value"][0], dim=-1).cpu().numpy()

        if cp in ab2_seats:
            chosen = ab2_choose(le)
            act = le[chosen]
            if act.type == 11:
                stats["ab2_trades"] += 1
            if act.type in (3, 4, 5, 6):
                stats["ab2_builds"] += 1
        else:
            enc_idx = int(np.argmax(logits))
            chosen = 0
            for j, a in enumerate(le):
                try:
                    if ae.encode(a) == enc_idx:
                        chosen = j
                        break
                except ValueError:
                    continue
            act = le[chosen]
            tname = TYPE_NAMES.get(act.type, f"T{act.type}")

            if act.type == 11:
                stats["nn_trades"] += 1
                stats["nn_trade_log"].append(
                    (turn, cp, list(act.value[:5]))
                )
            if act.type in (3, 4, 5, 6):
                stats["nn_builds"] += 1

            if verbose:
                if turn != last_turn:
                    if turn >= 8:
                        vps = [game._game.state.player_state[s][0]
                               for s in range(4)]
                        vp_str = "  ".join(f"P{s}:{vps[s]}" for s in range(4))
                        print(f"\n  -- T{turn} -- {vp_str}  "
                              f"val={values}")
                    last_turn = turn

                if act.type in INTERESTING_TYPES or turn <= 7:
                    print(f"  T{turn:3d} P{cp}(NN) [{len(le):3d}] "
                          f"{tname} {list(act.value[:3])}")
                elif act.type == 1 and len(le) > 3:
                    avail = {TYPE_NAMES.get(a.type, f"T{a.type}")
                             for a in le if a.type not in (0, 1)}
                    if avail:
                        print(f"  T{turn:3d} P{cp}(NN) [{len(le):3d}] "
                              f"END_TURN (skipped: {avail})")
                        stats["end_turn_skips"].append(
                            (turn, cp, avail)
                        )

        game.step(chosen)
        step += 1

    winner = game.winner()
    vps = [game._game.state.player_state[s][0] for s in range(4)]
    wt = ("NN" if winner in nn_seats
          else "AB2" if winner in ab2_seats
          else "NONE")
    return {
        "seed": seed, "winner": winner, "winner_team": wt,
        "turns": game.turn_number, "steps": step, "vps": vps,
        "nn_seats": nn_seats, "ab2_seats": ab2_seats, **stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-games", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=60000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    lib = load_library()
    ae = ActionEncoder()
    device = "cpu"
    net = HumanBotNet.load_checkpoint(args.checkpoint, device=device)
    net.eval()
    g = CatanGame(seed=0)
    g.reset()
    se = g.make_state_encoder()
    edge_index = se._edge_index.to(device)

    results = []
    for i in range(args.num_games):
        seed = args.seed_base + i * 37
        print(f"\n{'=' * 60}")
        print(f"GAME {i + 1} (seed={seed})")
        print(f"{'=' * 60}")

        r = observe_game(seed, net, se, ae, edge_index, device, lib,
                         verbose=not args.quiet)

        print(f"\n  === P{r['winner']} wins ({r['winner_team']}), "
              f"turns={r['turns']}, steps={r['steps']}")
        print(f"  === VPs: {r['vps']}  "
              f"NN={r['nn_seats']} AB2={r['ab2_seats']}")
        print(f"  === NN trades: {r['nn_trades']}, "
              f"AB2 trades: {r['ab2_trades']}")
        if r["nn_trade_log"]:
            print(f"  === NN trade log:")
            for t, p, v in r["nn_trade_log"][:15]:
                print(f"      T{t} P{p}: {v}")
        if r["end_turn_skips"]:
            print(f"  === NN chose END_TURN over options "
                  f"({len(r['end_turn_skips'])}x):")
            for t, p, avail in r["end_turn_skips"][:10]:
                print(f"      T{t} P{p}: skipped {avail}")
        results.append(r)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    nn_wins = sum(1 for r in results if r["winner_team"] == "NN")
    ab2_wins = sum(1 for r in results if r["winner_team"] == "AB2")
    avg_turns = np.mean([r["turns"] for r in results])
    avg_nn_trades = np.mean([r["nn_trades"] for r in results])
    print(f"  NN wins: {nn_wins}/{len(results)}  "
          f"AB2 wins: {ab2_wins}/{len(results)}")
    print(f"  Avg turns: {avg_turns:.0f}  "
          f"Avg NN trades/game: {avg_nn_trades:.1f}")


if __name__ == "__main__":
    main()
