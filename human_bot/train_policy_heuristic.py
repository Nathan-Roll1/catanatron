#!/usr/bin/env python3
"""Train a cheap flat-feature policy heuristic from dense Super M2 data.

The model is intentionally simple:

    score[action] = bias[action] + flat_features @ W[:, action]

It uses only the 115 flat state features plus the legal action mask. This is
far cheaper than the GNN policy and can be evaluated directly inside C search.
"""
from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F


AD = 337
FD = 115
TYPE_RANGES = [
    ("ROLL", 0, 1),
    ("END", 1, 2),
    ("BUY_DEV", 2, 3),
    ("KNIGHT", 3, 4),
    ("ROAD_BUILDING", 4, 5),
    ("SETTLEMENT", 5, 59),
    ("CITY", 59, 113),
    ("ROAD", 113, 185),
    ("ROBBER", 185, 280),
    ("DISCARD", 280, 285),
    ("YOP", 285, 305),
    ("MONOPOLY", 305, 310),
    ("MARITIME", 310, 330),
    ("TRADE_MISC", 330, 337),
]


def action_type_ids(action_idx: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(action_idx, -1)
    for tid, (_, lo, hi) in enumerate(TYPE_RANGES):
        out[(action_idx >= lo) & (action_idx < hi)] = tid
    return out


def masked_ce(logits, target_idx, mask, sample_weight=None):
    masked = logits.masked_fill(~mask.bool(), -1e9)
    per = F.cross_entropy(masked, target_idx, reduction="none")
    if sample_weight is not None:
        per = per * sample_weight
    return per.mean()


def masked_soft_ce(logits, target_probs, mask, sample_weight=None):
    masked = logits.masked_fill(~mask.bool(), -1e9)
    logp = F.log_softmax(masked, dim=-1)
    per = -(target_probs * logp).sum(dim=-1)
    if sample_weight is not None:
        per = per * sample_weight
    return per.mean()


@torch.no_grad()
def metrics(linear, x, mask, y):
    logits = linear(x).masked_fill(~mask.bool(), -1e9)
    pred = logits.argmax(dim=-1)
    out = {"top1": (pred == y).float().mean().item()}
    for k in (3, 5, 8, 12):
        kk = min(k, logits.shape[-1])
        topk = logits.topk(kk, dim=-1).indices
        out[f"top{k}"] = (topk == y[:, None]).any(dim=-1).float().mean().item()
    return out


def write_binary(path, linear):
    W = linear.weight.detach().cpu().float().numpy()  # (AD, FD)
    b = linear.bias.detach().cpu().float().numpy()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"HPOL")
        f.write(struct.pack("<III", 1, FD, AD))
        f.write(b.astype("<f4").tobytes())
        f.write(W.astype("<f4").tobytes())
    os.rename(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="csrc/data_super_m2/super_m2_4way_dense100_seed300000.pt")
    p.add_argument("--out", default="csrc/policy_heuristic_dense100.bin")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--target", choices=["hard", "soft", "mixed"], default="hard")
    p.add_argument("--hard-weight", type=float, default=0.8)
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    d = torch.load(args.shard, map_location="cpu", weights_only=False)
    x = d["flat_features"].float()
    mask = d["action_mask"][:, :AD].bool()
    y = d["action_idx"].long()
    sw = d.get("step_weight", torch.ones_like(y, dtype=torch.float32)).float()
    policy_target = d.get("policy_target")
    if policy_target is not None:
        policy_target = policy_target[:, :AD].float()

    keep = (y >= 0) & (y < AD)
    if args.search_only and "signal_kind" in d:
        keep &= d["signal_kind"].long() == 0
    x, mask, y, sw = x[keep], mask[keep], y[keep], sw[keep]
    if policy_target is not None:
        policy_target = policy_target[keep]

    n = y.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(0.1 * n))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]

    print(f"Loaded {n:,} examples from {args.shard}")
    print(f"  train={trn_idx.numel():,} val={val_idx.numel():,} target={args.target} search_only={args.search_only}")
    print("Chosen action type distribution:")
    tids = action_type_ids(y)
    for tid, (name, _, _) in enumerate(TYPE_RANGES):
        cnt = int((tids == tid).sum())
        if cnt:
            print(f"  {name:13s} {cnt:5d} ({100*cnt/n:5.1f}%)")

    linear = torch.nn.Linear(FD, AD)
    with torch.no_grad():
        counts = torch.bincount(y[trn_idx], minlength=AD).float() + 1.0
        linear.bias.copy_(counts.log() - counts.sum().log())
        linear.weight.zero_()

    opt = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bs = min(args.batch_size, trn_idx.numel())
    best = -1.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        linear.train()
        order = trn_idx[torch.randperm(trn_idx.numel(), generator=g)]
        total_loss = 0.0
        seen = 0
        for start in range(0, order.numel(), bs):
            idx = order[start:start + bs]
            logits = linear(x[idx])
            if args.target == "hard" or policy_target is None:
                loss = masked_ce(logits, y[idx], mask[idx], sw[idx])
            else:
                pt = policy_target[idx]
                if args.target == "mixed":
                    hard = torch.zeros_like(pt).scatter_(1, y[idx, None], 1.0)
                    pt = args.hard_weight * hard + (1.0 - args.hard_weight) * pt
                loss = masked_soft_ce(logits, pt, mask[idx], sw[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.numel()
            seen += idx.numel()

        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            linear.eval()
            tr = metrics(linear, x[trn_idx], mask[trn_idx], y[trn_idx])
            va = metrics(linear, x[val_idx], mask[val_idx], y[val_idx])
            print(f"ep {ep:4d} loss={total_loss/max(seen,1):.4f} "
                  f"train top1={tr['top1']:.3f} top12={tr['top12']:.3f} "
                  f"val top1={va['top1']:.3f} top5={va['top5']:.3f} top12={va['top12']:.3f}",
                  flush=True)
            if va["top12"] > best:
                best = va["top12"]
                best_state = {k: v.detach().clone() for k, v in linear.state_dict().items()}

    if best_state is not None:
        linear.load_state_dict(best_state)
    final = metrics(linear, x[val_idx], mask[val_idx], y[val_idx])
    print("Best val:", " ".join(f"{k}={v:.3f}" for k, v in final.items()))
    write_binary(args.out, linear)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
