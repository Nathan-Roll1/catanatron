"""Conservative no-search fine-tuning for exported M2.

This keeps deployment as one plain M2 forward pass.  It trains only selected
policy parameters against human action labels while a frozen copy of the seed
model supplies a KL anchor, so useful changes can be tested without turning the
argmax policy into an unrelated imitation model.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from human_bot.dataset import HumanGameDataset, load_tensor_shards
from human_bot.export_nn import export as export_nn
from human_bot.loss import _build_action_weights
from human_bot.model import HumanBotNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_PREFIXES = (
    "policy_head.type_fc.3.",
    "policy_head.discard_yop_mono_fc.2.",
    "policy_head.maritime_fc.2.",
    "policy_head.trade_fc.2.",
    "policy_head.settlement_scorer.2.",
    "policy_head.city_scorer.2.",
    "policy_head.road_scorer.2.",
    "policy_head.robber_scorer.2.",
)


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def filter_dataset(ds: HumanGameDataset, winner_only: bool,
                   strategic_only: bool) -> HumanGameDataset:
    keep = torch.ones(ds.n, dtype=torch.bool)
    if winner_only:
        keep &= ds.value_target[:, 0] > 0.5
    if strategic_only:
        keep &= ds.action_idx >= 2
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    return HumanGameDataset(
        ds.nf[idx], ds.ef[idx], ds.ff[idx], ds.mask[idx],
        ds.action_idx[idx], ds.value_target[idx],
    )


def set_train_scope(net: HumanBotNet, scope: str) -> int:
    for param in net.parameters():
        param.requires_grad = False
    for name, param in net.named_parameters():
        if scope == "policy_final":
            param.requires_grad = name.startswith(FINAL_PREFIXES)
        elif scope == "policy_head":
            param.requires_grad = name.startswith("policy_head.")
        elif scope == "trunk_policy":
            param.requires_grad = (
                name.startswith("trunk.") or name.startswith("policy_head.")
            )
        else:
            raise ValueError(f"unknown train scope: {scope}")
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    fill = -6e4 if logits.dtype == torch.float16 else -1e9
    return F.log_softmax(logits.masked_fill(~mask.bool(), fill), dim=-1)


def policy_ce(logits: torch.Tensor, action_idx: torch.Tensor,
              mask: torch.Tensor, label_smoothing: float,
              action_weights: torch.Tensor | None) -> torch.Tensor:
    logp = masked_log_probs(logits, mask)
    legal = mask.float()
    n_legal = legal.sum(dim=-1, keepdim=True).clamp(min=1.0)
    one_hot = torch.zeros_like(logits).scatter_(1, action_idx[:, None], 1.0)
    target = (1.0 - label_smoothing) * one_hot + label_smoothing * legal / n_legal
    per = -(target * logp).sum(dim=-1)
    if action_weights is not None:
        per = per * action_weights[action_idx]
    return per.mean()


def policy_kl_to_base(new_logits: torch.Tensor, base_logits: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    new_logp = masked_log_probs(new_logits, mask)
    base_logp = masked_log_probs(base_logits, mask)
    base_p = base_logp.exp()
    return (base_p * (base_logp - new_logp)).sum(dim=-1).mean()


def batch_forward(net: HumanBotNet, nf: torch.Tensor, ef: torch.Tensor,
                  ff: torch.Tensor, mask: torch.Tensor,
                  edge_index: torch.Tensor) -> torch.Tensor:
    return net({
        "node_features": nf,
        "edge_index": edge_index,
        "edge_features": ef,
        "flat_features": ff,
        "action_mask": mask,
    })["policy_logits"]


@torch.no_grad()
def filter_indices_by_base_topk(base: HumanBotNet, data: dict[str, torch.Tensor],
                                idx: torch.Tensor, edge_index: torch.Tensor,
                                batch_size: int, k: int) -> torch.Tensor:
    if k <= 0:
        return idx
    base.eval()
    kept = []
    for start in range(0, idx.numel(), batch_size):
        b = idx[start:start + batch_size]
        nf, ef, ff, mask, act = (data["nf"][b], data["ef"][b], data["ff"][b],
                                 data["mask"][b], data["act"][b])
        logits = batch_forward(base, nf, ef, ff, mask, edge_index)
        masked = logits.masked_fill(~mask.bool(), -1e9)
        kk = min(k, masked.shape[-1])
        topk = torch.topk(masked, k=kk, dim=-1).indices
        keep = (topk == act[:, None]).any(dim=-1)
        if keep.any():
            kept.append(b[keep])
    if not kept:
        return idx[:0]
    return torch.cat(kept)


@torch.no_grad()
def evaluate(net: HumanBotNet, base: HumanBotNet, data: dict[str, torch.Tensor],
             idx: torch.Tensor, edge_index: torch.Tensor, batch_size: int,
             label_smoothing: float, action_weights: torch.Tensor | None) -> dict:
    net.eval()
    base.eval()
    ce_sum = 0.0
    kl_sum = 0.0
    top1 = 0
    n = 0
    for start in range(0, idx.numel(), batch_size):
        b = idx[start:start + batch_size]
        nf, ef, ff, mask, act = (data["nf"][b], data["ef"][b], data["ff"][b],
                                 data["mask"][b], data["act"][b])
        logits = batch_forward(net, nf, ef, ff, mask, edge_index)
        base_logits = batch_forward(base, nf, ef, ff, mask, edge_index)
        bs = b.numel()
        ce_sum += float(policy_ce(logits, act, mask, label_smoothing,
                                  action_weights)) * bs
        kl_sum += float(policy_kl_to_base(logits, base_logits, mask)) * bs
        pred = logits.masked_fill(~mask.bool(), -1e9).argmax(dim=-1)
        top1 += int((pred == act).sum().item())
        n += bs
    net.train()
    return {
        "val_ce": ce_sum / max(1, n),
        "val_kl": kl_sum / max(1, n),
        "val_top1": top1 / max(1, n),
        "val_n": n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", default="checkpoints/sp_latest2.pt")
    p.add_argument("--data-dir", default="data/human_v2_fixed")
    p.add_argument("--out-pt", default="autoresearch-results/m2_no_search/latest.pt")
    p.add_argument("--out-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument("--max-examples", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--kl-weight", type=float, default=5.0)
    p.add_argument("--scope", choices=("policy_final", "policy_head", "trunk_policy"),
                   default="policy_final")
    p.add_argument("--winner-only", action="store_true")
    p.add_argument("--strategic-only", action="store_true")
    p.add_argument("--action-weights", action="store_true")
    p.add_argument("--base-topk-filter", type=int, default=0,
                   help="Train/evaluate only examples where the human action "
                        "is already in frozen base M2's legal top-k.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = detect_device(args.device)

    print(f"Loading data from {args.data_dir} max_examples={args.max_examples}", flush=True)
    ds = load_tensor_shards(args.data_dir, max_examples=args.max_examples)
    ds = filter_dataset(ds, args.winner_only, args.strategic_only)
    if ds.n < 100:
        raise RuntimeError(f"too few examples after filtering: {ds.n}")

    rng = np.random.RandomState(args.seed)
    perm = torch.from_numpy(rng.permutation(ds.n)).long()
    n_val = max(1, int(ds.n * args.val_fraction))
    val_idx_cpu = perm[:n_val]
    train_idx_cpu = perm[n_val:]

    print(f"Filtered examples: {ds.n:,} train={train_idx_cpu.numel():,} "
          f"val={val_idx_cpu.numel():,}", flush=True)
    data = {
        "nf": ds.nf.to(device),
        "ef": ds.ef.to(device),
        "ff": ds.ff.to(device),
        "mask": ds.mask.to(device),
        "act": ds.action_idx.to(device),
    }
    train_idx = train_idx_cpu.to(device)
    val_idx = val_idx_cpu.to(device)

    net = HumanBotNet.load_checkpoint(args.seed_checkpoint, device=device)
    base = HumanBotNet.load_checkpoint(args.seed_checkpoint, device=device)
    base.eval()
    for param in base.parameters():
        param.requires_grad = False
    n_trainable = set_train_scope(net, args.scope)
    net.train()

    from hexzero.game.interface import CatanGame
    g = CatanGame(seed=0)
    g.reset()
    edge_index = g.make_state_encoder()._edge_index.to(device)

    if args.base_topk_filter > 0:
        before_train = train_idx.numel()
        before_val = val_idx.numel()
        train_idx = filter_indices_by_base_topk(
            base, data, train_idx, edge_index, args.batch_size,
            args.base_topk_filter)
        val_idx = filter_indices_by_base_topk(
            base, data, val_idx, edge_index, args.batch_size,
            args.base_topk_filter)
        if train_idx.numel() < 100 or val_idx.numel() < 10:
            raise RuntimeError(
                f"base-topk filter kept too few examples: "
                f"train={train_idx.numel()} val={val_idx.numel()}"
            )
        print(
            f"base_topk_filter={args.base_topk_filter}: "
            f"train {train_idx.numel():,}/{before_train:,} "
            f"val {val_idx.numel():,}/{before_val:,}",
            flush=True,
        )

    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    action_weights = _build_action_weights(397, device) if args.action_weights else None

    print(f"device={device} scope={args.scope} trainable={n_trainable:,} "
          f"lr={args.lr} kl={args.kl_weight}", flush=True)
    print(f"winner_only={args.winner_only} strategic_only={args.strategic_only} "
          f"action_weights={args.action_weights} "
          f"base_topk_filter={args.base_topk_filter}", flush=True)

    best = None
    best_metrics = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = train_idx[torch.randperm(train_idx.numel(), device=device)]
        sums = {"ce": 0.0, "kl": 0.0, "loss": 0.0, "top1": 0}
        seen = 0
        for start in range(0, order.numel(), args.batch_size):
            b = order[start:start + args.batch_size]
            nf, ef, ff, mask, act = (data["nf"][b], data["ef"][b], data["ff"][b],
                                     data["mask"][b], data["act"][b])
            logits = batch_forward(net, nf, ef, ff, mask, edge_index)
            with torch.no_grad():
                base_logits = batch_forward(base, nf, ef, ff, mask, edge_index)
            ce = policy_ce(logits, act, mask, args.label_smoothing, action_weights)
            kl = policy_kl_to_base(logits, base_logits, mask)
            loss = ce + args.kl_weight * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            bs = b.numel()
            pred = logits.masked_fill(~mask.bool(), -1e9).argmax(dim=-1)
            sums["ce"] += float(ce.detach()) * bs
            sums["kl"] += float(kl.detach()) * bs
            sums["loss"] += float(loss.detach()) * bs
            sums["top1"] += int((pred == act).sum().item())
            seen += bs

        if device == "mps":
            torch.mps.synchronize()
        metrics = evaluate(net, base, data, val_idx, edge_index, args.batch_size,
                           args.label_smoothing, action_weights)
        train_ce = sums["ce"] / max(1, seen)
        train_kl = sums["kl"] / max(1, seen)
        train_top1 = sums["top1"] / max(1, seen)
        print(
            f"[{epoch}/{args.epochs}] train_ce={train_ce:.4f} "
            f"train_kl={train_kl:.4f} train_top1={train_top1:.3f} "
            f"val_ce={metrics['val_ce']:.4f} val_kl={metrics['val_kl']:.4f} "
            f"val_top1={metrics['val_top1']:.3f}",
            flush=True,
        )
        if best_metrics is None or metrics["val_ce"] < best_metrics["val_ce"]:
            best_metrics = metrics
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best is not None:
        net.load_state_dict(best)
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(out_pt), {
        "finetune_m2_no_search": True,
        "scope": args.scope,
        "lr": args.lr,
        "kl_weight": args.kl_weight,
        "winner_only": args.winner_only,
        "strategic_only": args.strategic_only,
        "best_metrics": best_metrics or {},
    })
    export_nn(str(out_pt), args.out_bin)
    print({
        "out_pt": str(out_pt),
        "out_bin": args.out_bin,
        "elapsed_sec": time.time() - t0,
        "best_metrics": best_metrics,
    }, flush=True)


if __name__ == "__main__":
    main()
