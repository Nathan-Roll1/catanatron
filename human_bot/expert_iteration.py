#!/usr/bin/env python3
"""Expert Iteration: collect self-play data at N-ply search, retrain policy+value.

1. Play games using N-ply search (the "expert")
2. Record (state, search_chosen_action, game_outcome) at every decision
3. Train the "apprentice" network to predict search actions and outcomes

Usage:
    python -m human_bot.expert_iteration \
        --checkpoint checkpoints/human_bot_pipeline/final_all.pt \
        --num-games 1000 --search-depth 5 --epochs 3
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import math
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from human_bot.model import HumanBotNet, SmallNetworkConfig
from human_bot.loss import UncertaintyWeightedLoss, human_policy_loss, value_loss, masked_entropy
from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal


def collect_self_play(
    nn_lib, model_ptr, se, ae, lib, device,
    num_games: int, search_depth: int, top_k: int = 5,
    seed_base: int = 100000,
) -> dict[str, torch.Tensor]:
    """Play games with N-ply search, record every decision point."""
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    N, E = se.num_nodes, se.num_edges
    NF = se.NODE_FEATURE_DIM
    EF = se.EDGE_FEATURE_DIM
    FFD = se.FLAT_FEATURE_DIM
    AD = 337
    FP = ctypes.POINTER(ctypes.c_float)

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    nf_buf = np.zeros((N, NF), dtype=np.float32)
    ef_buf = np.zeros((E, EF), dtype=np.float32)
    ff_buf = np.zeros(FFD, dtype=np.float32)
    mask_buf = np.zeros(397, dtype=np.float32)
    val_buf = np.zeros(4, dtype=np.float32)
    nfp = nf_buf.ctypes.data_as(FP)
    efp = ef_buf.ctypes.data_as(FP)
    ffp = ff_buf.ctypes.data_as(FP)
    mkp = mask_buf.ctypes.data_as(FP)
    vlp = val_buf.ctypes.data_as(FP)

    def c_value(game):
        se.encode_into(game.get_state_view(), nf_buf, ef_buf, ff_buf)
        le = game.get_legal_actions()
        mask_buf[:] = 0
        mn = ae.get_action_mask(le).numpy()
        mask_buf[:len(mn)] = mn
        nn_lib.nn_value_only(model_ptr, nfp, efp, ffp, mkp, vlp)
        return val_buf.copy()

    def c_policy_top_k(game, le, k):
        se.encode_into(game.get_state_view(), nf_buf, ef_buf, ff_buf)
        mask_buf[:] = 0
        mn = ae.get_action_mask(le).numpy()
        mask_buf[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(model_ptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        logits = out[4:4+AD]
        a2i = {ae.encode(a): i for i, a in enumerate(le)}
        scored = sorted([(logits[enc], li) for enc, li in a2i.items()], reverse=True)
        return [li for _, li in scored[:k]]

    def c_policy_argmax(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1:
            gc.step(0); return
        se.encode_into(gc.get_state_view(), nf_buf, ef_buf, ff_buf)
        mask_buf[:] = 0
        mn = ae.get_action_mask(le).numpy()
        mask_buf[:len(mn)] = mn
        out = np.zeros(4 + 397, dtype=np.float32)
        nn_lib.nn_forward(model_ptr, nfp, efp, ffp, mkp,
                          out.ctypes.data_as(ctypes.c_void_p))
        logits = out[4:4+AD]
        logits[mn < 0.5] = -1e9
        ai = int(np.argmax(logits))
        chosen = next((i for i, a in enumerate(le) if ae.encode(a) == ai), 0)
        gc.step(chosen)

    def ab2_respond(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1:
            gc.step(0); return
        cg = gc._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv:
                bv = v; bi = i
        gc.step(bi)

    def depth_search(game, le, my_seats, opp_seats, depth):
        our_seat = game.current_player()
        candidates = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            candidates = c_policy_top_k(game, le, top_k)
        best_pos, best_val = 0, -1e30
        for pos, ci in enumerate(candidates):
            gc = game.clone()
            gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal():
                    break
                cp = gc.current_player()
                if cp in opp_seats:
                    ab2_respond(gc)
                elif cp in my_seats:
                    c_policy_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                if w is not None and w == our_seat:
                    v = 10.0
                elif w is not None:
                    v = -10.0
                else:
                    v = 0.0
            else:
                vals = c_value(gc)
                offset = (our_seat - gc.current_player()) % 4
                v = float(vals[offset])
            v = apply_action_bonus(v, le[ci])
            if v > best_val:
                best_val = v; best_pos = pos
        chosen = candidates[best_pos]
        chosen = fix_robber_steal(chosen, le)
        return chosen

    # Collection buffers
    all_nf, all_ef, all_ff, all_mask = [], [], [], []
    all_action, all_player, all_reward = [], [], []

    game_outcomes = []
    t0 = time.perf_counter()

    for gi in range(num_games):
        game = CatanGame(seed=seed_base + gi)
        game.reset()
        nn_seats = {gi % 4, (gi + 2) % 4}
        ab2_seats = {(gi + 1) % 4, (gi + 3) % 4}
        mar_received = {}

        game_nf, game_ef, game_ff, game_mask = [], [], [], []
        game_action, game_player = [], []

        while not game.is_terminal() and game.turn_number < 1000:
            le = game.get_legal_actions()
            if not le:
                break
            cp = game.current_player()

            if cp in ab2_seats:
                if len(le) == 1:
                    game.step(0)
                else:
                    ab2_respond(game)
                mar_received.pop(cp, None)
                continue

            if len(le) == 1:
                act = le[0]
                if act.type == 11:
                    mar_received.setdefault(cp, set()).add(act.value[4])
                else:
                    mar_received.pop(cp, None)
                game.step(0)
                continue

            # Multi-choice decision: record state and search result
            received = mar_received.get(cp, set())
            if received:
                filtered = [a for a in le if a.type != 11 or a.value[0] not in received]
                if not filtered:
                    filtered = le
                le_use = filtered
                idx_map = [le.index(a) for a in le_use]
            else:
                le_use = le
                idx_map = list(range(len(le)))

            # Encode state BEFORE search (this is the training input)
            nf_rec = np.zeros((N, NF), dtype=np.float32)
            ef_rec = np.zeros((E, EF), dtype=np.float32)
            ff_rec = np.zeros(FFD, dtype=np.float32)
            se.encode_into(game.get_state_view(), nf_rec, ef_rec, ff_rec)
            mn = ae.get_action_mask(le_use).numpy()
            mask_rec = np.zeros(397, dtype=np.float32)
            mask_rec[:len(mn)] = mn

            # Search
            chosen_in_use = depth_search(game, le_use, nn_seats, ab2_seats, search_depth)
            chosen = idx_map[chosen_in_use]
            act = le[chosen]
            action_idx = ae.encode(act)

            game_nf.append(nf_rec)
            game_ef.append(ef_rec)
            game_ff.append(ff_rec)
            game_mask.append(mask_rec)
            game_action.append(action_idx)
            game_player.append(cp)

            if act.type == 11:
                mar_received.setdefault(cp, set()).add(act.value[4])
            else:
                mar_received.pop(cp, None)
            game.step(chosen)

        # Game over: compute rewards
        winner = game.winner()
        n_decisions = len(game_action)
        if n_decisions == 0:
            continue

        # Build reward vectors: one-hot winner, rotated to current player's perspective
        for di in range(n_decisions):
            p = game_player[di]
            rv = np.zeros(4, dtype=np.float32)
            if winner is not None:
                rv[winner] = 1.0
            else:
                rv[:] = 0.25
            shift = (-p) % 4
            rv = np.roll(rv, shift)
            all_reward.append(rv)

        all_nf.extend(game_nf)
        all_ef.extend(game_ef)
        all_ff.extend(game_ff)
        all_mask.extend(game_mask)
        all_action.extend(game_action)
        all_player.extend(game_player)

        win_tag = "NN" if winner in nn_seats else "AB2" if winner in ab2_seats else "?"
        game_outcomes.append(winner in nn_seats if winner is not None else False)

        if (gi + 1) % 50 == 0 or gi + 1 == num_games:
            elapsed = time.perf_counter() - t0
            nn_wr = sum(game_outcomes) / len(game_outcomes)
            print(f"  [{gi+1}/{num_games}] {len(all_action):,} decisions  "
                  f"NN WR={nn_wr:.1%}  ({elapsed:.0f}s, {elapsed/(gi+1):.1f}s/game)")

    # Stack into tensors
    S = len(all_action)
    result = {
        "node_features": torch.from_numpy(np.stack(all_nf)),
        "edge_features": torch.from_numpy(np.stack(all_ef)),
        "flat_features": torch.from_numpy(np.stack(all_ff)),
        "action_mask": torch.from_numpy(np.stack(all_mask)),
        "action_idx": torch.tensor(all_action, dtype=torch.long),
        "value_target": torch.from_numpy(np.stack(all_reward)),
        "player": torch.tensor(all_player, dtype=torch.long),
    }
    nn_wr = sum(game_outcomes) / max(len(game_outcomes), 1)
    print(f"\n  Collection done: {S:,} decisions from {num_games} games  "
          f"(NN WR={nn_wr:.1%})")
    return result


def train_exit(
    net: nn.Module, data: dict[str, torch.Tensor],
    edge_index: torch.Tensor, device: str,
    epochs: int = 3, lr: float = 3e-3, batch_size: int = 4096,
) -> list[dict]:
    """Train on ExIt data (search-selected actions + outcomes)."""
    from human_bot.dataset import HumanGameDataset
    from human_bot.train import DeviceDataset, build_cosine_scheduler

    ds = HumanGameDataset(
        data["node_features"], data["edge_features"], data["flat_features"],
        data["action_mask"], data["action_idx"], data["value_target"],
    )
    dd = DeviceDataset(ds, device)
    S = len(ds)
    print(f"\n  Training on {S:,} ExIt examples, {epochs} epochs, "
          f"lr={lr}, bs={batch_size}")

    loss_combiner = UncertaintyWeightedLoss().to(device)
    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, S // batch_size)
    total_steps = steps_per_epoch * epochs
    scheduler = build_cosine_scheduler(optimizer, total_steps, min(100, total_steps // 5))

    history = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(S, device=device)
        sums = dict.fromkeys(
            ["policy_loss", "value_loss", "entropy", "policy_acc", "value_acc"], 0.0
        )
        n_batches = 0

        for i in range(0, S, batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 16:
                continue
            nf, ef, ff, mask, action_idx, vt = dd.get_batch(idx)

            out = net({
                "node_features": nf, "edge_index": edge_index,
                "edge_features": ef, "flat_features": ff, "action_mask": mask,
            })

            p_loss = human_policy_loss(
                out["policy_logits"], action_idx, mask,
                label_smoothing=0.02,
            )
            turn_progress = ff[:, 114]
            v_loss = value_loss(out["value"], vt, turn_progress=turn_progress)
            ent = masked_entropy(out["policy_logits"], mask)
            total, _ = loss_combiner(p_loss, v_loss, ent, 0.01)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pacc = (out["policy_logits"].argmax(-1) == action_idx).float().mean().item()
                vacc = (out["value"].argmax(-1) == vt.argmax(-1)).float().mean().item()

            sums["policy_loss"] += p_loss.item()
            sums["value_loss"] += v_loss.item()
            sums["entropy"] += ent.item()
            sums["policy_acc"] += pacc
            sums["value_acc"] += vacc
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        history.append(avg)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {ep+1}/{epochs}: "
              f"ploss={avg['policy_loss']:.3f} pacc={avg['policy_acc']:.3f}  "
              f"vloss={avg['value_loss']:.3f} vacc={avg['value_acc']:.3f}  "
              f"ent={avg['entropy']:.3f}  lr={cur_lr:.1e}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Expert Iteration")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=1000)
    parser.add_argument("--search-depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--output", type=str, default="checkpoints/exit_round1.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = args.device
    t_start = time.perf_counter()

    # Load C NN for fast self-play inference
    nn_lib = ctypes.CDLL(os.path.join("csrc", "libnn.dylib"))
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ]
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
    ]
    model_buf = (ctypes.c_char * (8 * 1024 * 1024))()
    model_ptr = ctypes.cast(model_buf, ctypes.c_void_p)
    assert nn_lib.nn_load(model_ptr, b"csrc/nn_weights.bin") == 0
    print("C NN loaded for self-play inference")

    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library

    ae = ActionEncoder()
    lib = load_library()
    g = CatanGame(seed=0); g.reset()
    se = g.make_state_encoder()
    edge_index = se._edge_index.to(device)

    # ── Phase 1: Collect self-play data ──
    print(f"\n{'='*60}")
    print(f"  Phase 1: Self-play collection ({args.num_games} games, "
          f"{args.search_depth}-ply, top-{args.top_k})")
    print(f"{'='*60}")

    data = collect_self_play(
        nn_lib, model_ptr, se, ae, lib, device,
        num_games=args.num_games,
        search_depth=args.search_depth,
        top_k=args.top_k,
    )

    # ── Phase 2: Train on ExIt data ──
    print(f"\n{'='*60}")
    print(f"  Phase 2: ExIt training ({args.epochs} epochs, lr={args.lr})")
    print(f"{'='*60}")

    net = HumanBotNet.load_checkpoint(args.checkpoint, device=device)
    print(f"  Loaded base model: {net.num_parameters:,} params on {device}")

    history = train_exit(
        net, data, edge_index, device,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    net.save_checkpoint(args.output, {
        "method": "expert_iteration",
        "base_checkpoint": args.checkpoint,
        "num_games": args.num_games,
        "search_depth": args.search_depth,
        "epochs": args.epochs,
        "lr": args.lr,
        "history": history,
    })
    print(f"\n  Saved ExIt model: {args.output}")

    # ── Phase 3: Evaluate ──
    if args.eval_games > 0:
        print(f"\n{'='*60}")
        print(f"  Phase 3: Evaluation ({args.eval_games} games per depth)")
        print(f"{'='*60}")

        net.eval()
        from human_bot.eval_search import evaluate_search_vs_ab2

        for depth in [0, 1, 2]:
            t0 = time.perf_counter()
            result = evaluate_search_vs_ab2(
                net, se, ae, device, lib,
                num_games=args.eval_games,
                search_depth=depth,
                seed_offset=depth * 100 + 7777,
            )
            sec = time.perf_counter() - t0
            print(f"  {depth}-ply: NN={result['hz_wins']}  "
                  f"AB2={result['ab2_wins']}  "
                  f"WR={result['win_rate']:.1%}  ({sec:.0f}s)")

    total = time.perf_counter() - t_start
    print(f"\nTotal time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
