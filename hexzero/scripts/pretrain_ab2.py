#!/usr/bin/env python3
"""Supervised pretraining: learn to imitate AB2's play.

Generates games where all 4 players use AB2 (greedy C heuristic),
records every (state, action, outcome), then trains the network
to predict AB2's moves (policy) and game outcomes (value).

This gives the network a strong baseline of Catan domain knowledge
before switching to self-play R-NaD for refinement.

Usage:
    python -m hexzero.scripts.pretrain_ab2 \
        --games 10000 --epochs 10 --device cuda \
        --checkpoint-dir hexzero/checkpoints \
        --wandb-key <key>
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import random
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def main():
    parser = argparse.ArgumentParser(description="Pretrain on AB2 games")
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-dir", type=str, default="hexzero/checkpoints")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="hexazero")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    device = detect_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()

    g = CatanGame(seed=0); g.reset()
    state_enc = g.make_state_encoder()

    N = state_enc.num_nodes
    E = state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM
    AD = 337

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    print("=" * 60, flush=True)
    print(" AB2 Supervised Pretraining", flush=True)
    print(f" Device: {device} ({gpu_name})", flush=True)
    print(f" Games: {args.games}, Epochs: {args.epochs}", flush=True)
    print("=" * 60, flush=True)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"pretrain-ab2-{os.uname().nodename}",
                config=vars(args),
                tags=["pretrain", "ab2", device],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] Failed: {e}", flush=True)

    # ── Generate AB2 games ────────────────────────────────────────────
    print(f"\n[data] Generating {args.games} AB2-vs-AB2 games...", flush=True)
    t0 = time.time()

    all_nf = []
    all_ef = []
    all_ff = []
    all_masks = []
    all_actions = []  # one-hot over 337
    all_values = []   # game outcome per player

    nf_buf = np.zeros((N, NF), dtype=np.float32)
    ef_buf = np.zeros((E, EF), dtype=np.float32)
    ff_buf = np.zeros(FF, dtype=np.float32)

    for gi in range(args.games):
        game = CatanGame(seed=gi)
        game.reset()
        history = []

        while not game.is_terminal() and game.turn_number < 750:
            cp = game.current_player()
            le = game.get_legal_actions()
            if not le:
                break

            # Encode state
            state_enc.encode_into(game.get_state_view(), nf_buf, ef_buf, ff_buf)
            mask = action_enc.get_action_mask(le).numpy()

            # AB2 picks action: greedy 1-ply with base_value_fn
            cg = game._game
            bc = cg.state.colors[cg.state.current_player_index]
            best_i, best_v = 0, -math.inf
            ch = CGame()
            ca = (Action * MAX_ACTIONS)()
            cn = ctypes.c_int(0)
            for i, act in enumerate(le):
                lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                v = lib.base_value_fn(ctypes.byref(ch), bc)
                if v > best_v:
                    best_v = v
                    best_i = i

            # Record: state, AB2's action as one-hot, player
            action_onehot = np.zeros(AD, dtype=np.float32)
            action_onehot[action_enc.encode(le[best_i])] = 1.0

            history.append((
                nf_buf.copy(), ef_buf.copy(), ff_buf.copy(),
                mask.copy(), action_onehot, cp
            ))

            game.step(best_i)

        # Assign value targets from outcome
        winner = game.winner()
        for nf, ef, ff, mask, action_oh, player in history:
            if winner is None:
                val = 0.25  # draw
            elif winner == player:
                val = 1.0
            else:
                val = 0.0
            all_nf.append(nf)
            all_ef.append(ef)
            all_ff.append(ff)
            all_masks.append(mask)
            all_actions.append(action_oh)
            all_values.append(val)

        if (gi + 1) % max(args.games // 10, 1) == 0:
            elapsed = time.time() - t0
            print(f"  {gi+1}/{args.games} games | {len(all_nf)} positions | "
                  f"{(gi+1)/elapsed:.0f} g/s", flush=True)

    t_data = time.time() - t0
    n_pos = len(all_nf)
    print(f"  Done: {n_pos} positions from {args.games} games in {t_data:.0f}s",
          flush=True)

    # Convert to numpy arrays
    all_nf = np.stack(all_nf)
    all_ef = np.stack(all_ef)
    all_ff = np.stack(all_ff)
    all_masks = np.stack(all_masks)
    all_actions = np.stack(all_actions)
    all_values = np.array(all_values, dtype=np.float32)

    # ── Build model ───────────────────────────────────────────────────
    net = HexaZeroNet(cfg.network).to(device)
    print(f"\n[model] {net.num_parameters:,} params on {device}", flush=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    edge_index_dev = state_enc._edge_index.to(device)

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n[train] {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}",
          flush=True)

    indices = np.arange(n_pos)
    bs = args.batch_size

    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        net.train()
        epoch_ploss = 0.0
        epoch_vloss = 0.0
        epoch_pacc = 0.0
        epoch_vacc = 0.0
        n_batches = 0
        t_ep = time.time()

        for start in range(0, n_pos, bs):
            batch_idx = indices[start:start + bs]
            if len(batch_idx) < 8:
                continue

            b_nf = torch.from_numpy(all_nf[batch_idx]).to(device)
            b_ef = torch.from_numpy(all_ef[batch_idx]).to(device)
            b_ff = torch.from_numpy(all_ff[batch_idx]).to(device)
            b_mask = torch.from_numpy(all_masks[batch_idx]).to(device)
            b_act = torch.from_numpy(all_actions[batch_idx]).to(device)
            b_val = torch.from_numpy(all_values[batch_idx]).to(device)

            batch_input = {
                "node_features": b_nf,
                "edge_index": edge_index_dev,
                "edge_features": b_ef,
                "flat_features": b_ff,
                "action_mask": b_mask,
            }

            optimizer.zero_grad(set_to_none=True)
            out = net(batch_input)

            # Policy loss: cross-entropy with AB2's action
            masked_logits = out["policy_logits"]
            log_probs = F.log_softmax(masked_logits.masked_fill(b_mask == 0, -1e9), dim=-1)
            policy_loss = -(b_act * log_probs).sum(dim=-1).mean()
            policy_loss = torch.nan_to_num(policy_loss, nan=0.0)

            # Value loss: cross-entropy with 4-dim target
            # b_val is scalar (current player outcome); build 4-dim target
            vt = torch.zeros(out["value"].shape, device=device)
            vt[:, 0] = b_val
            vt[:, 1:] = (1.0 - b_val.unsqueeze(-1)) / 3.0
            vt_sum = vt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            vt_dist = vt / vt_sum
            value_log_probs = F.log_softmax(out["value"], dim=-1)
            value_loss = -(vt_dist * value_log_probs).sum(dim=-1).mean()

            total_loss = policy_loss + value_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred_act = out["policy_probs"].argmax(dim=-1)
                true_act = b_act.argmax(dim=-1)
                pacc = (pred_act == true_act).float().mean().item()

                pred_win = F.softmax(out["value"], dim=-1)[:, 0]
                vacc = ((pred_win > 0.5) == (b_val > 0.5)).float().mean().item()

            epoch_ploss += policy_loss.item()
            epoch_vloss += value_loss.item()
            epoch_pacc += pacc
            epoch_vacc += vacc
            n_batches += 1

        ep_time = time.time() - t_ep
        avg_pl = epoch_ploss / n_batches
        avg_vl = epoch_vloss / n_batches
        avg_pa = epoch_pacc / n_batches
        avg_va = epoch_vacc / n_batches

        print(f"  E{epoch:2d}: ploss={avg_pl:.4f} vloss={avg_vl:.4f} "
              f"policy_acc={avg_pa:.3f} value_acc={avg_va:.3f} ({ep_time:.0f}s)",
              flush=True)

        if wandb_run:
            import wandb
            wandb.log({
                "pretrain/policy_loss": avg_pl,
                "pretrain/value_loss": avg_vl,
                "pretrain/policy_accuracy": avg_pa,
                "pretrain/value_accuracy": avg_va,
                "pretrain/epoch": epoch,
            })

        # Save checkpoint each epoch
        ckpt_path = os.path.join(args.checkpoint_dir, f"pretrain_e{epoch:02d}.pt")
        net.save_checkpoint(ckpt_path, metadata={
            "epoch": epoch, "policy_acc": avg_pa, "value_acc": avg_va,
        })

    # Save final
    latest = os.path.join(args.checkpoint_dir, "latest.pt")
    net.save_checkpoint(latest, metadata={
        "pretrained": True, "games": args.games, "epochs": args.epochs,
        "policy_accuracy": avg_pa, "value_accuracy": avg_va,
    })
    print(f"\nSaved: {latest}", flush=True)
    print(f"Policy accuracy: {avg_pa:.3f} (chance=~1/{AD}={1/AD:.4f})", flush=True)
    print(f"Value accuracy: {avg_va:.3f}", flush=True)

    # ── Quick eval vs AB2 ─────────────────────────────────────────────
    print(f"\n[eval] 24 games vs AB2...", flush=True)
    from hexzero.elo.rating import EloRating, MatchResult
    net.eval()
    hz_w = ab2_w = rand_w = 0
    for gi in range(24):
        game = CatanGame(seed=50000 + gi)
        game.reset()
        hz_s, ab2_s = gi % 4, (gi + 1) % 4
        while not game.is_terminal() and game.turn_number < 1000:
            cp = game.current_player()
            le = game.get_legal_actions()
            if not le:
                break
            if cp == hz_s:
                bi, bv = 0, -1e9
                for i in range(len(le)):
                    c = game.clone(); c.step(i)
                    if c.is_terminal():
                        v = 10.0 if c.winner() == hz_s else -10.0
                    else:
                        enc_nf = np.zeros((N, NF), dtype=np.float32)
                        enc_ef = np.zeros((E, EF), dtype=np.float32)
                        enc_ff = np.zeros(FF, dtype=np.float32)
                        state_enc.encode_into(c.get_state_view(), enc_nf, enc_ef, enc_ff)
                        cl = c.get_legal_actions()
                        bb = {
                            "node_features": torch.from_numpy(enc_nf).unsqueeze(0).to(device),
                            "edge_index": edge_index_dev,
                            "edge_features": torch.from_numpy(enc_ef).unsqueeze(0).to(device),
                            "flat_features": torch.from_numpy(enc_ff).unsqueeze(0).to(device),
                            "action_mask": action_enc.get_action_mask(cl).unsqueeze(0).to(device),
                        }
                        with torch.no_grad():
                            v = F.softmax(net(bb)["value"], dim=-1)[0, 0].item()
                    if v > bv:
                        bv = v; bi = i
                game.step(bi)
            elif cp == ab2_s:
                cg = game._game
                bc = cg.state.colors[cg.state.current_player_index]
                bi, bv = 0, -math.inf
                ch = CGame(); ca = (Action * MAX_ACTIONS)(); cn = ctypes.c_int(0)
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
                    if v > bv:
                        bv = v; bi = i
                game.step(bi)
            else:
                game.step(random.randrange(len(le)))
        w = game.winner()
        if w == hz_s: hz_w += 1
        elif w == ab2_s: ab2_w += 1
        elif w is not None: rand_w += 1

    elo = EloRating(k_factor=32.0)
    elo.register_player("AB2", 1000.0, pinned=True)
    elo.register_player("HexaZero", 1000.0)
    for _ in range(hz_w):
        elo.update_ratings(MatchResult(["HexaZero","AB2","Random","Random"],"HexaZero",0,0,0,time.time()))
    for _ in range(ab2_w):
        elo.update_ratings(MatchResult(["HexaZero","AB2","Random","Random"],"AB2",1,0,0,time.time()))

    print(f"  HZ={hz_w} AB2={ab2_w} Rand={rand_w} | "
          f"ELO={elo.get_rating('HexaZero'):.0f} (AB2=1000)", flush=True)

    if wandb_run:
        import wandb
        wandb.log({
            "eval/hz_wins": hz_w,
            "eval/ab2_wins": ab2_w,
            "eval/hz_win_rate": hz_w / max(hz_w + ab2_w + rand_w, 1),
            "eval/hz_elo": elo.get_rating("HexaZero"),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
