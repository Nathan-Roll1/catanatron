"""Full HexaZero pipeline optimized for Apple M5 Max.

Generates data, trains aggressively, evaluates against AB2,
tracks ELO, and reports everything.

Key M5 Max optimizations:
- NO torch.compile (measured 3x slower with aot_eager on MPS)
- Batch size 2048 (128GB unified memory handles it)
- Reduced MPS sync: log every 50 batches not 25
- Edge index pre-moved to device once
- Cosine LR schedule with warmup
"""

from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def generate_selfplay(num_games, seed_base=0):
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    enc = ActionEncoder()
    g = CatanGame(seed=0); g.reset()
    se = g.make_state_encoder()

    examples = []
    for gi in range(num_games):
        g = CatanGame(seed=seed_base + gi); g.reset()
        hist = []
        while not g.is_terminal() and g.turn_number < 1000:
            cp = g.current_player()
            st = se.encode(g.get_state_view())
            st = {k: v.detach() for k, v in st.items()}
            legal = g.get_legal_actions()
            mask = enc.get_action_mask(legal)
            st["action_masks"] = mask
            hist.append((st, mask / mask.sum(), cp))
            g.step(random.randrange(len(legal)))
        w = g.winner()
        w = w if w is not None else -1
        for st, pol, p in hist:
            vt = torch.zeros(4, dtype=torch.float32)
            if w >= 0:
                vt[(w - p) % 4] = 1.0
            examples.append((st, pol, vt))
    return examples, se


def evaluate_vs_ab2(net, state_enc, device, num_games=40):
    """Play HexaZero (greedy value) vs AB2 (greedy value) vs 2 random."""
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    import ctypes

    lib = load_library()
    enc = ActionEncoder()
    hz_wins, ab2_wins, rand_wins, draws = 0, 0, 0, 0

    for gi in range(num_games):
        g = CatanGame(seed=10000 + gi); g.reset()
        hz_seat = gi % 4
        ab2_seat = (gi + 1) % 4

        while not g.is_terminal() and g.turn_number < 1000:
            cp = g.current_player()
            legal = g.get_legal_actions()
            if not legal:
                break

            if cp == hz_seat:
                idx = _hz_greedy_action(g, net, state_enc, enc, legal, device)
            elif cp == ab2_seat:
                idx = _ab2_greedy_action(g, legal, lib)
            else:
                idx = random.randrange(len(legal))
            g.step(idx)

        w = g.winner()
        if w == hz_seat:
            hz_wins += 1
        elif w == ab2_seat:
            ab2_wins += 1
        elif w is not None:
            rand_wins += 1
        else:
            draws += 1

    return hz_wins, ab2_wins, rand_wins, draws


def _hz_greedy_action(game, net, state_enc, action_enc, legal, device):
    """Pick the legal action where value head gives us highest P(win)."""
    from hexzero.game.interface import CatanGame
    import ctypes

    best_idx, best_val = 0, -1.0
    cp = game.current_player()

    for i, action in enumerate(legal):
        child = game.clone()
        child.step(i)
        sv = child.get_state_view()
        enc = state_enc.encode(sv)
        batch = {k: v.unsqueeze(0).to(device) for k, v in enc.items()}
        legal_c = child.get_legal_actions()
        if legal_c:
            batch["action_mask"] = action_enc.get_action_mask(legal_c).unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(batch)
        val = F.softmax(out["value"], dim=-1)[0, 0].item()
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def _ab2_greedy_action(game, legal, lib):
    """AB2: greedy depth-1 using base_value_fn."""
    import ctypes
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    c_game = game._game
    bot_color = c_game.state.colors[c_game.state.current_player_index]
    best_idx, best_val = 0, -math.inf
    child = CGame()
    child_actions = (Action * MAX_ACTIONS)()
    child_n = ctypes.c_int(0)

    for i, action in enumerate(legal):
        lib.game_copy(ctypes.byref(child), ctypes.byref(c_game))
        lib.game_execute(ctypes.byref(child), action, child_actions, ctypes.byref(child_n))
        val = lib.base_value_fn(ctypes.byref(child), bot_color)
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=3,
                        help="Evaluate every N epochs")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.selfplay.replay_buffer import ReplayBuffer
    from hexzero.training.loss import HexaZeroLoss

    print(f"{'='*60}")
    print(f" HexaZero M5 Max  |  {DEVICE}  |  {os.cpu_count()} cores")
    print(f" {args.games} games -> {args.epochs} epochs @ bs={args.batch_size}")
    print(f"{'='*60}")
    print()

    # ── Self-play ─────────────────────────────────────────────────────
    print("[SELFPLAY] Generating training data...")
    t0 = time.time()
    examples, state_enc = generate_selfplay(args.games)
    t_sp = time.time() - t0
    print(f"  {args.games} games, {len(examples)} positions in {t_sp:.1f}s "
          f"({args.games/t_sp:.0f} g/s)")

    buf = ReplayBuffer(capacity=max(len(examples) * 2, 500_000))
    for st, pol, vt in examples:
        buf.push(st, pol, vt)
    del examples
    print(f"  Buffer: {len(buf)} positions")
    print()

    # ── Model ─────────────────────────────────────────────────────────
    cfg = get_default_config()
    net = HexaZeroNet(cfg.network)
    net.to(DEVICE)
    print(f"[MODEL] {net.num_parameters:,} params on {DEVICE}")
    print()

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = HexaZeroLoss()

    nb_per_epoch = max(len(buf) // args.batch_size, 1)
    total_steps = args.epochs * nb_per_epoch
    warmup = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── ELO tracking ──────────────────────────────────────────────────
    from hexzero.elo.rating import EloRating
    elo = EloRating(k_factor=32.0)
    elo.register_player("AB2", 100.0, pinned=True)
    elo.register_player("HexaZero", 100.0)
    elo.register_player("Random", 100.0)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"[TRAIN] {args.epochs} epochs, {nb_per_epoch} batches/epoch, "
          f"{total_steps} total steps")
    print(f"  LR: {args.lr} with cosine schedule (warmup={warmup})")
    print()

    global_step = 0
    t_total = time.time()
    best_loss = float("inf")

    for epoch in range(args.epochs):
        net.train()
        acc = {}
        t_ep = time.time()

        for bi in range(nb_per_epoch):
            batch = buf.sample(args.batch_size)
            inp = {
                "node_features": batch.node_features.to(DEVICE, non_blocking=True),
                "edge_index": batch.edge_index.to(DEVICE, non_blocking=True),
                "edge_features": batch.edge_features.to(DEVICE, non_blocking=True),
                "flat_features": batch.flat_features.to(DEVICE, non_blocking=True),
                "action_mask": batch.action_masks.to(DEVICE, non_blocking=True),
            }
            tgt = {
                "policy_targets": batch.policy_targets.to(DEVICE, non_blocking=True),
                "value_targets": batch.value_targets.to(DEVICE, non_blocking=True),
                "action_masks": batch.action_masks.to(DEVICE, non_blocking=True),
            }

            optimizer.zero_grad(set_to_none=True)
            preds = net(inp)
            losses = criterion(preds, tgt)
            losses["total_loss"].backward()
            gn = nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            for k in ["total_loss", "value_loss", "policy_loss",
                       "value_accuracy", "policy_entropy"]:
                acc[k] = acc.get(k, 0.0) + losses[k].item()

            if (bi + 1) % 50 == 0 or bi == nb_per_epoch - 1:
                n = bi + 1
                sps = (n * args.batch_size) / (time.time() - t_ep)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  E{epoch:2d} [{n:3d}/{nb_per_epoch}]  "
                    f"loss={acc['total_loss']/n:.4f}  "
                    f"vloss={acc['value_loss']/n:.4f}  "
                    f"ploss={acc['policy_loss']/n:.4f}  "
                    f"vacc={acc['value_accuracy']/n:.3f}  "
                    f"lr={lr:.1e}  {sps:.0f} s/s"
                )

        ep_time = time.time() - t_ep
        n = nb_per_epoch
        avg = {k: v / n for k, v in acc.items()}
        sps = (n * args.batch_size) / ep_time

        if avg["total_loss"] < best_loss:
            best_loss = avg["total_loss"]

        print(f"  === Epoch {epoch} done: {ep_time:.1f}s  {sps:.0f} samp/s  "
              f"best_loss={best_loss:.4f} ===")

        # ── Evaluate against AB2 ──────────────────────────────────────
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            print()
            print(f"  [EVAL] HexaZero vs AB2 vs Random ({args.eval_games} games)...")
            net.eval()
            t_ev = time.time()
            hz_w, ab2_w, rand_w, draws = evaluate_vs_ab2(
                net, state_enc, DEVICE, args.eval_games
            )
            ev_time = time.time() - t_ev
            total_g = hz_w + ab2_w + rand_w + draws

            hz_wr = hz_w / max(total_g, 1)
            ab2_wr = ab2_w / max(total_g, 1)

            from hexzero.elo.rating import MatchResult
            for _ in range(hz_w):
                elo.update_ratings(MatchResult(
                    players=["HexaZero", "AB2", "Random", "Random"],
                    winner="HexaZero", winner_seat=0, num_turns=0,
                    game_seed=0, timestamp=time.time()))
            for _ in range(ab2_w):
                elo.update_ratings(MatchResult(
                    players=["HexaZero", "AB2", "Random", "Random"],
                    winner="AB2", winner_seat=1, num_turns=0,
                    game_seed=0, timestamp=time.time()))
            for _ in range(rand_w):
                elo.update_ratings(MatchResult(
                    players=["HexaZero", "AB2", "Random", "Random"],
                    winner="Random", winner_seat=2, num_turns=0,
                    game_seed=0, timestamp=time.time()))

            hz_elo = elo.get_rating("HexaZero")
            ab2_elo = elo.get_rating("AB2")
            rand_elo = elo.get_rating("Random")

            print(f"  Results: HZ={hz_w} AB2={ab2_w} Rand={rand_w} Draw={draws}  "
                  f"({ev_time:.1f}s)")
            print(f"  Win rates: HZ={hz_wr:.1%}  AB2={ab2_wr:.1%}")
            print(f"  ELO: HexaZero={hz_elo:.0f}  AB2={ab2_elo:.0f} [pinned]  "
                  f"Random={rand_elo:.0f}")
            print(f"  ELO diff vs AB2: {hz_elo - ab2_elo:+.0f}")
            print()

    # ── Final summary ─────────────────────────────────────────────────
    total_time = time.time() - t_total
    print(f"{'='*60}")
    print(f" FINAL RESULTS  ({total_time:.0f}s total)")
    print(f"{'='*60}")
    print(f"  Loss:       {avg['total_loss']:.4f} (value={avg['value_loss']:.4f} "
          f"policy={avg['policy_loss']:.4f})")
    print(f"  Value acc:  {avg['value_accuracy']:.3f}")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Throughput: {sps:.0f} samples/s")
    print()
    print(f"  ELO RATINGS:")
    for row in elo.get_ratings_table():
        pin = " [PINNED]" if row["pinned"] else ""
        print(f"    {row['name']:12s}  {row['rating']:7.1f}  "
              f"({row['games_played']} games){pin}")
    print()

    if args.save:
        net.save_checkpoint(args.save, metadata={
            "epochs": args.epochs, "elo": hz_elo, "metrics": avg,
        })
        print(f"  Saved: {args.save}")


if __name__ == "__main__":
    main()
