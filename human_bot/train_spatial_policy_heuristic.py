#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from human_bot.spatial_policy_heuristic import (
    AD, SpatialPolicyHeuristic, SpatialPolicyHeuristicMLP,
)


def masked_ce(logits, target_idx, sample_weight=None):
    per = F.cross_entropy(logits, target_idx, reduction="none")
    if sample_weight is not None:
        per = per * sample_weight
    return per.mean()


def masked_soft_ce(logits, target_probs, sample_weight=None):
    logp = F.log_softmax(logits, dim=-1)
    per = -(target_probs * logp).sum(dim=-1)
    if sample_weight is not None:
        per = per * sample_weight
    return per.mean()


@torch.no_grad()
def metrics(model, nf, ef, ff, mask, y, batch_size=4096):
    model.eval()
    total = 0
    hits = {1: 0, 3: 0, 5: 0, 8: 0, 12: 0}
    for start in range(0, y.shape[0], batch_size):
        sl = slice(start, start + batch_size)
        logits = model(nf[sl], ef[sl], ff[sl], mask[sl])
        total += logits.shape[0]
        for k in hits:
            pred = logits.topk(min(k, AD), dim=-1).indices
            hits[k] += int((pred == y[sl, None]).any(dim=-1).sum())
    return {f"top{k}": hits[k] / max(total, 1) for k in hits}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", default="csrc/data_super_m2/super_m2_4way_dense100_seed300000.pt")
    p.add_argument("--out", default="checkpoints/spatial_heuristic_dense100.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--target", choices=["hard", "soft", "mixed"], default="mixed")
    p.add_argument("--hard-weight", type=float, default=0.8)
    p.add_argument("--label-key", type=str, default="action_idx",
                   help="Shard tensor to imitate, e.g. action_idx or m2_action_idx.")
    p.add_argument("--search-only", action="store_true")
    p.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    p.add_argument("--model", choices=["linear", "mlp"], default="linear")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    d = torch.load(args.shard, map_location="cpu", weights_only=False)
    nf = d["node_features"].float()
    ef = d["edge_features"].float()
    ff = d["flat_features"].float()
    mask = d["action_mask"][:, :AD].bool()
    y = d[args.label_key].long()
    sw = d.get("step_weight", torch.ones_like(y, dtype=torch.float32)).float()
    pt = d.get("policy_target")
    if pt is not None:
        pt = pt[:, :AD].float()

    keep = (y >= 0) & (y < AD)
    if args.search_only and "signal_kind" in d:
        keep &= d["signal_kind"].long() == 0
    nf, ef, ff, mask, y, sw = nf[keep], ef[keep], ff[keep], mask[keep], y[keep], sw[keep]
    if pt is not None:
        pt = pt[keep]

    from hexzero.game.interface import CatanGame
    g0 = CatanGame(seed=0)
    g0.reset()
    tile_nodes = g0.make_state_encoder()._ltiles.copy()

    n = y.shape[0]
    rng = np.random.RandomState(args.seed)
    perm = torch.from_numpy(rng.permutation(n)).long()
    n_val = max(1, int(0.1 * n))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]

    nf = nf.to(device); ef = ef.to(device); ff = ff.to(device)
    mask = mask.to(device); y = y.to(device); sw = sw.to(device)
    if pt is not None:
        pt = pt.to(device)
    trn_idx = trn_idx.to(device); val_idx = val_idx.to(device)

    if args.model == "mlp":
        model = SpatialPolicyHeuristicMLP(torch.from_numpy(tile_nodes), hidden=args.hidden).to(device)
    else:
        model = SpatialPolicyHeuristic(torch.from_numpy(tile_nodes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    g = torch.Generator(device=device).manual_seed(args.seed)

    print(f"device={device} examples={n:,} train={trn_idx.numel():,} val={val_idx.numel():,} "
          f"target={args.target} label_key={args.label_key} model={args.model} search_only={args.search_only}", flush=True)

    best = -1.0
    best_state = None
    bs = min(args.batch_size, trn_idx.numel())
    for ep in range(1, args.epochs + 1):
        model.train()
        order = trn_idx[torch.randperm(trn_idx.numel(), generator=g, device=device)]
        total_loss = 0.0
        seen = 0
        for start in range(0, order.numel(), bs):
            idx = order[start:start + bs]
            logits = model(nf[idx], ef[idx], ff[idx], mask[idx])
            if args.target == "hard" or pt is None:
                loss = masked_ce(logits, y[idx], sw[idx])
            else:
                target = pt[idx]
                if args.target == "mixed":
                    hard = torch.zeros_like(target).scatter_(1, y[idx, None], 1.0)
                    target = args.hard_weight * hard + (1.0 - args.hard_weight) * target
                loss = masked_soft_ce(logits, target, sw[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * idx.numel()
            seen += idx.numel()

        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            tr = metrics(model, nf[trn_idx], ef[trn_idx], ff[trn_idx], mask[trn_idx], y[trn_idx])
            va = metrics(model, nf[val_idx], ef[val_idx], ff[val_idx], mask[val_idx], y[val_idx])
            print(f"ep {ep:4d} loss={total_loss/max(seen,1):.4f} "
                  f"train top1={tr['top1']:.3f} top12={tr['top12']:.3f} "
                  f"val top1={va['top1']:.3f} top5={va['top5']:.3f} top12={va['top12']:.3f}",
                  flush=True)
            if va["top12"] > best:
                best = va["top12"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "tile_nodes": torch.from_numpy(tile_nodes).long(),
        "meta": vars(args),
    }, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
