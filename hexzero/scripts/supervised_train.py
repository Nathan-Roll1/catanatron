#!/usr/bin/env python3
"""Supervised behavioral cloning from pre-collected AB2 game data.

Trains HexaZeroNet to imitate all four AB2 players from recorded games.
Policy target is one-hot on AB2's actual action (pure behavioral cloning).
Value target is the game's 4-player win distribution rotated to current player.

Data format: each .pt file contains a list of games, each game is a list of
step dicts with keys {nf, ef, ff, mask, action_idx, player, reward_vec}.

Usage:
    python -m hexzero.scripts.supervised_train \
        --data-dir data/ab2_games \
        --checkpoint-dir checkpoints \
        --epochs 10 --batch-size 2048 --lr 0.001 \
        --eval-games 25 \
        --wandb-key KEY
"""

from __future__ import annotations

import argparse
import ctypes
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


# ======================================================================
# Data loading
# ======================================================================

class TensorDataset:
    """Memory-efficient dataset backed by concatenated tensors."""

    def __init__(self, nf, ef, ff, mask, action_idx, value_target):
        self.nf = nf          # (S, N, 18) float32
        self.ef = ef          # (S, E, 5) float32
        self.ff = ff          # (S, 115) float32
        self.mask = mask      # (S, 337) float32
        self.action_idx = action_idx  # (S,) int64
        self.value_target = value_target  # (S, 4) float32
        self.n = nf.shape[0]

    def __len__(self):
        return self.n

    def get_batch(self, indices):
        return {
            "nf": self.nf[indices],
            "ef": self.ef[indices],
            "ff": self.ff[indices],
            "mask": self.mask[indices],
            "action_idx": self.action_idx[indices],
            "value_target": self.value_target[indices],
        }


def load_and_process_data(data_dir: str) -> TensorDataset:
    """Load batched .pt files and return a single TensorDataset."""
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".pt") and f != "metadata.pt"
    )
    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    all_nf, all_ef, all_ff, all_mask = [], [], [], []
    all_act, all_vt = [], []

    for fi, fname in enumerate(files):
        path = os.path.join(data_dir, fname)
        try:
            data = torch.load(path, weights_only=False, map_location="cpu")
        except Exception as e:
            print(f"  WARN: skipping {fname}: {e}", flush=True)
            continue

        players = data["player"].numpy()
        reward_vecs = data["reward_vec"].numpy()

        # Build rotated value targets for all steps at once
        S = players.shape[0]
        vt = np.zeros((S, 4), dtype=np.float32)
        for i in range(S):
            rv = reward_vecs[i]
            rot = np.roll(rv, -int(players[i]))
            rsum = rot.sum()
            vt[i] = rot / rsum if rsum > 1e-8 else 0.25

        all_nf.append(data["node_features"])
        all_ef.append(data["edge_features"])
        all_ff.append(data["flat_features"])
        all_mask.append(data["action_mask"])
        all_act.append(data["action_idx"])
        all_vt.append(torch.from_numpy(vt))

        if (fi + 1) % 20 == 0 or fi + 1 == len(files):
            total = sum(t.shape[0] for t in all_act)
            print(f"  {fi + 1}/{len(files)} files  →  "
                  f"{total:,} examples", flush=True)

    return TensorDataset(
        nf=torch.cat(all_nf),
        ef=torch.cat(all_ef),
        ff=torch.cat(all_ff),
        mask=torch.cat(all_mask),
        action_idx=torch.cat(all_act),
        value_target=torch.cat(all_vt),
    )


# ======================================================================
# Evaluation (policy-only, NO 1-ply lookahead)
# ======================================================================

def evaluate_policy(net, state_enc, action_enc, device, lib,
                    num_games, epoch, temperature):
    """Play concurrent games: 2 HZ (policy, no lookahead) vs 2 AB2.

    All games run simultaneously with batched GPU inference for HZ moves.
    Returns (hz_wins, ab2_wins).
    """
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF, EF, FF = (state_enc.NODE_FEATURE_DIM,
                  state_enc.EDGE_FEATURE_DIM,
                  state_enc.FLAT_FEATURE_DIM)
    nf_buf = np.zeros((num_games + 1, N, NF), dtype=np.float32)
    ef_buf = np.zeros((num_games + 1, E, EF), dtype=np.float32)
    ff_buf = np.zeros((num_games + 1, FF), dtype=np.float32)
    mask_buf = np.zeros((num_games + 1, AD), dtype=np.float32)

    edge_index_dev = state_enc._edge_index.to(device)
    net.eval()

    games = [CatanGame(seed=90000 + epoch * 1000 + i)
             for i in range(num_games)]
    for g in games:
        g.reset()

    hz_seats = [None] * num_games
    ab2_seats = [None] * num_games
    for i in range(num_games):
        hz_seats[i] = {i % 4, (i + 2) % 4}
        ab2_seats[i] = {(i + 1) % 4, (i + 3) % 4}

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    active = list(range(num_games))
    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # AB2 turns (sequential C engine)
        prog = True
        while prog:
            prog = False
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
                        bv = v; bi = i
                g.step(bi)
                prog = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # HZ turns (batched GPU)
        B = 0
        imap = []
        for idx in active:
            g = games[idx]
            if g.current_player() not in hz_seats[idx]:
                continue
            le = g.get_legal_actions()
            if not le:
                continue
            state_enc.encode_into(
                g.get_state_view(), nf_buf[B], ef_buf[B], ff_buf[B])
            mask_buf[B] = action_enc.get_action_mask(le).numpy()
            imap.append((idx, le))
            B += 1

        if B == 0:
            continue

        with torch.no_grad():
            batch = {
                "node_features": torch.from_numpy(nf_buf[:B].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B].copy()).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B].copy()).to(device),
            }
            out = net(batch)
            lo = out["policy_logits"] / temperature
            lo = lo.masked_fill(batch["action_mask"] == 0, -1e9)
            pr = F.softmax(lo, dim=-1).cpu().numpy()

        for b, (idx, le) in enumerate(imap):
            p = pr[b]
            if p.sum() < 1e-6:
                p = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
            p = p / p.sum()
            aidx = int(np.random.choice(AD, p=p))
            chosen = next((i for i, a in enumerate(le)
                           if action_enc.encode(a) == aidx), 0)
            games[idx].step(chosen)

    hz_wins = ab2_wins = 0
    for idx in range(num_games):
        w = games[idx].winner()
        if w is not None:
            if w in hz_seats[idx]:
                hz_wins += 1
            elif w in ab2_seats[idx]:
                ab2_wins += 1

    return hz_wins, ab2_wins


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Supervised behavioral cloning from AB2 game data")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-temperature", type=float, default=0.1)
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

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()

    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    # ── Load data ────────────────────────────────────────────────────
    print(f"Loading data from {args.data_dir} ...", flush=True)
    dataset = load_and_process_data(args.data_dir)
    n_examples = len(dataset)
    if n_examples == 0:
        raise RuntimeError("No training examples found")
    print(f"Total: {n_examples:,} training examples\n", flush=True)

    # ── Model ────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        net = HexaZeroNet.load_checkpoint(args.resume, device=device)
        print(f"Resumed from {args.resume}", flush=True)
    else:
        net = HexaZeroNet(cfg.network).to(device)
        print("Random initialization", flush=True)
    print(f"{net.num_parameters:,} parameters", flush=True)

    raw_net = net

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=1e-4)

    print("=" * 60, flush=True)
    print(" Supervised Behavioral Cloning (AB2)", flush=True)
    print(f" Device       : {device} ({gpu_name})", flush=True)
    print(f" Examples     : {n_examples:,}", flush=True)
    print(f" Epochs       : {args.epochs}", flush=True)
    print(f" Batch size   : {args.batch_size}", flush=True)
    print(f" LR           : {args.lr}", flush=True)
    print(f" Entropy wt   : {args.entropy_weight}", flush=True)
    print(f" Eval games   : {args.eval_games} (temp={args.eval_temperature})", flush=True)
    print("=" * 60, flush=True)

    # ── W&B ──────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"supervised-bc-{time.strftime('%m%d-%H%M')}",
                config=vars(args),
                tags=["supervised", "behavioral-cloning", device],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed: {e}", flush=True)

    # ── Training loop ────────────────────────────────────────────────
    best_hz_wr = -1.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        perm = torch.randperm(n_examples)
        net.train()

        sums = {
            "policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0,
            "entropy": 0.0, "policy_acc": 0.0, "value_acc": 0.0,
        }
        n_batches = 0

        for i in range(0, n_examples, args.batch_size):
            idx = perm[i : i + args.batch_size]
            B = len(idx)
            if B < 8:
                continue

            batch_data = dataset.get_batch(idx)
            nf = batch_data["nf"].to(device)
            ef = batch_data["ef"].to(device)
            ff = batch_data["ff"].to(device)
            mask = batch_data["mask"].to(device)
            action_idx = batch_data["action_idx"].to(device)
            vt = batch_data["value_target"].to(device)

            batch_input = {
                "node_features": nf,
                "edge_index": edge_index_dev,
                "edge_features": ef,
                "flat_features": ff,
                "action_mask": mask,
            }

            optimizer.zero_grad(set_to_none=True)
            out = net(batch_input)

            # ── Policy: cross-entropy with one-hot target ─────────
            logits = out["policy_logits"]
            policy_loss = F.cross_entropy(logits, action_idx)
            policy_loss = torch.nan_to_num(policy_loss, nan=0.0)

            # ── Value: cross-entropy with 4-dim distribution ──────
            value_lp = F.log_softmax(out["value"], dim=-1)
            value_loss = -(vt.detach() * value_lp).sum(dim=-1).mean()
            value_loss = torch.nan_to_num(value_loss, nan=0.0)

            # ── Entropy bonus ─────────────────────────────────────
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs * mask).sum(dim=-1)
            entropy = torch.nan_to_num(entropy, nan=0.0).mean()

            total_loss = policy_loss + value_loss - args.entropy_weight * entropy

            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                pacc = (pred == action_idx).float().mean().item()

                vp_w = F.softmax(out["value"], dim=-1).argmax(dim=-1)
                vt_w = vt.argmax(dim=-1)
                vacc = (vp_w == vt_w).float().mean().item()

            sums["policy_loss"] += policy_loss.item()
            sums["value_loss"] += value_loss.item()
            sums["total_loss"] += total_loss.item()
            sums["entropy"] += entropy.item()
            sums["policy_acc"] += pacc
            sums["value_acc"] += vacc
            n_batches += 1

        # ── Epoch averages ───────────────────────────────────────
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        epoch_sec = time.time() - t_epoch

        # ── Evaluate ─────────────────────────────────────────────
        print(f"[epoch {epoch}] evaluating vs AB2 "
              f"({args.eval_games} games) ...", flush=True)
        hz_w, ab2_w = evaluate_policy(
            raw_net, state_enc, action_enc, device, lib,
            args.eval_games, epoch, args.eval_temperature)
        total_eval = hz_w + ab2_w
        hz_wr = hz_w / max(total_eval, 1)

        print(
            f"[epoch {epoch}/{args.epochs}] "
            f"ploss={avg['policy_loss']:.4f}  pacc={avg['policy_acc']:.3f}  "
            f"vloss={avg['value_loss']:.4f}  vacc={avg['value_acc']:.3f}  "
            f"ent={avg['entropy']:.3f}  "
            f"| EVAL HZ={hz_w} AB2={ab2_w} WR={hz_wr:.1%}  "
            f"| {epoch_sec:.0f}s",
            flush=True,
        )

        # ── Save checkpoint ──────────────────────────────────────
        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        raw_net.save_checkpoint(ckpt_path, metadata={
            "epoch": epoch,
            "hz_win_rate": hz_wr,
            "policy_loss": avg["policy_loss"],
            "policy_acc": avg["policy_acc"],
            "value_loss": avg["value_loss"],
        })

        if hz_wr > best_hz_wr:
            best_hz_wr = hz_wr
            best_path = os.path.join(args.checkpoint_dir, "best.pt")
            raw_net.save_checkpoint(best_path, metadata={
                "epoch": epoch,
                "hz_win_rate": hz_wr,
            })
            print(f"  ★ New best HZ WR: {hz_wr:.1%}  → {best_path}", flush=True)

        # ── W&B ──────────────────────────────────────────────────
        if wandb_run:
            import wandb
            wandb.log({
                "train/policy_loss": avg["policy_loss"],
                "train/value_loss": avg["value_loss"],
                "train/total_loss": avg["total_loss"],
                "train/policy_entropy": avg["entropy"],
                "train/policy_accuracy": avg["policy_acc"],
                "train/value_accuracy": avg["value_acc"],
                "train/learning_rate": args.lr,
                "eval/hz_wins": hz_w,
                "eval/ab2_wins": ab2_w,
                "eval/hz_win_rate": hz_wr,
                "eval/best_hz_win_rate": best_hz_wr,
                "epoch": epoch,
            })

    elapsed = time.time() - t_start
    print(f"\nDone. {args.epochs} epochs in {elapsed:.0f}s.  "
          f"Best HZ WR: {best_hz_wr:.1%}", flush=True)
    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
