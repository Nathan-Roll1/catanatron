"""train_m3_local: offline trainer that fine-tunes M2 -> M3 on the
300-game dense super_m2 dataset.

Differences from `super_learner`:
  - One-shot offline training over a fixed dataset (no shard polling, no
    deletion). Iterates the entire dataset N epochs.
  - Train/val split (90/10 by example, fixed RNG).
  - Saves the best-val checkpoint as M3 + exports to .bin.
  - Same loss stack as super_learner: dense soft-policy CE +
    rotated-outcome value CE + search-value MSE + entropy bonus.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_DIM = 397


def load_all_shards(shard_dir, glob_pat):
    """Concatenate all shards into one in-memory dataset."""
    import glob
    files = sorted(glob.glob(os.path.join(shard_dir, glob_pat)))
    if not files:
        raise FileNotFoundError(f"No shards matching {shard_dir}/{glob_pat}")
    print(f"Loading {len(files)} shards from {shard_dir} ...", flush=True)

    keys = ["node_features", "edge_features", "flat_features", "action_mask",
            "action_idx", "player", "reward_vec", "step_weight",
            "policy_target", "signal_kind", "search_value"]
    bufs = {k: [] for k in keys}

    for fp in files:
        d = torch.load(fp, weights_only=False, map_location="cpu")
        n = d["action_idx"].shape[0]
        for k in keys:
            if k in d:
                bufs[k].append(d[k])
            elif k == "policy_target":
                bufs[k].append(torch.zeros(n, MASK_DIM))
            elif k == "signal_kind":
                bufs[k].append(torch.zeros(n, dtype=torch.int64))
            elif k == "search_value":
                bufs[k].append(torch.zeros(n))
            elif k == "step_weight":
                bufs[k].append(torch.ones(n))
            else:
                raise KeyError(f"shard {fp} missing required field: {k}")
        print(f"  {os.path.basename(fp)}: {n:>5d} examples", flush=True)

    out = {k: torch.cat(v) for k, v in bufs.items()}

    # Pad action_mask if it was 337 instead of 397
    if out["action_mask"].shape[-1] < MASK_DIM:
        pad = torch.zeros(out["action_mask"].shape[0],
                          MASK_DIM - out["action_mask"].shape[-1])
        out["action_mask"] = torch.cat([out["action_mask"], pad], dim=-1)

    print(f"TOTAL: {out['action_idx'].shape[0]:,} examples", flush=True)
    return out


def value_targets_from_reward(reward_vec, players, num_players_default=4):
    """Rotate winner one-hot so slot 0 = current player. Defaults to
    num_players=4 (all our 4-way data).
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from human_bot.dataset import rotate_value_targets_to_cp
    rv = reward_vec.numpy()
    pl = players.numpy()
    S = rv.shape[0]
    winners = rv.argmax(axis=1)
    has_winner = rv.max(axis=1) > 0.5
    vt = np.zeros((S, 4), dtype=np.float32)
    vt[np.arange(S), winners] = 1.0
    vt[~has_winner] = 0.25
    np_arr = np.full(S, num_players_default, dtype=np.int64)
    return torch.from_numpy(rotate_value_targets_to_cp(vt, pl, np_arr))


def super_policy_loss(logits, target, mask, sample_weight=None):
    """Soft cross-entropy: target is a (S, 397) probability distribution."""
    fill = -6e4 if logits.dtype == torch.float16 else -1e9
    masked = logits.masked_fill(~mask.bool(), fill)
    log_p = F.log_softmax(masked, dim=-1)
    per = -(target * log_p).sum(dim=-1)
    if sample_weight is not None:
        return (per * sample_weight).mean()
    return per.mean()


def hard_policy_loss(logits, action_idx, mask, label_smoothing=0.05,
                      sample_weight=None):
    """One-hot CE on action_idx (= super_m2's chosen action) with optional
    label smoothing over the legal-action set."""
    fill = -6e4 if logits.dtype == torch.float16 else -1e9
    masked = logits.masked_fill(~mask.bool(), fill)
    log_p = F.log_softmax(masked, dim=-1)
    n_legal = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    one_hot = torch.zeros_like(logits).scatter_(1, action_idx.unsqueeze(1), 1.0)
    smooth = (1.0 - label_smoothing) * one_hot + label_smoothing * (mask / n_legal)
    per = -(smooth * log_p).sum(dim=-1)
    if sample_weight is not None:
        return (per * sample_weight).mean()
    return per.mean()


def sharpen_policy_target(target, temperature=0.05):
    """Take a soft policy_target and re-sharpen it via temperature.
    Re-normalizes so each row still sums to 1.

    target: (S, 397) probabilities (must sum to ~1)
    Returns sharper distribution with same support.
    """
    # Convert probs to log-probs, divide by temperature, softmax
    eps = 1e-12
    log_p = torch.log(target.clamp(min=eps))
    sharpened = F.softmax(log_p / temperature, dim=-1)
    # Zero where original was zero (preserve support)
    sharpened = sharpened * (target > 0).float()
    sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True).clamp(min=eps)
    return sharpened


def mix_target(soft, hard_idx, mask, hard_weight=0.7):
    """Linear mix: mixed = hard_weight * one_hot(hard_idx) + (1-hw) * soft.

    hard_weight=1.0 → pure one-hot (hard label)
    hard_weight=0.0 → pure soft target
    hard_weight=0.7 → strongly favor super_m2's pick but keep some soft mass
    """
    one_hot = torch.zeros_like(soft).scatter_(1, hard_idx.unsqueeze(1), 1.0)
    return hard_weight * one_hot + (1.0 - hard_weight) * soft


def search_value_loss(value_logits, search_value, signal_kind):
    """MSE between value-head's predicted V(s) and search V(s)."""
    probs = F.softmax(value_logits, dim=-1)
    pred_v = 2.0 * probs[:, 0] - 1.0
    valid = (signal_kind != 1) & torch.isfinite(search_value)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)
    diff = (pred_v - search_value).pow(2) * valid_f
    return diff.sum() / n_valid, valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", type=str,
                   default="checkpoints/sp_v7_latest.pt")
    p.add_argument("--shard-dir", type=str,
                   default="csrc/data_super_m2")
    p.add_argument("--shard-glob", type=str,
                   default="super_m2_4way_300g_chunk*.pt")
    p.add_argument("--out-pt", type=str, default="checkpoints/m3.pt")
    p.add_argument("--out-bin", type=str, default="csrc/nn_weights_m3.bin")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--policy-mode", type=str, default="soft",
                   choices=["soft", "hard", "sharp", "mixed"],
                   help="soft: use policy_target as-is (flat, good for "
                        "search prior; weak at 0-ply argmax). "
                        "hard: one-hot CE on super_m2's chosen action "
                        "(decisive, strong at 0-ply). "
                        "sharp: temperature-sharpen policy_target via "
                        "--policy-temperature. "
                        "mixed: blend hard + soft via --hard-weight.")
    p.add_argument("--policy-temperature", type=float, default=0.03,
                   help="Temperature for 'sharp' mode (lower = peakier).")
    p.add_argument("--hard-weight", type=float, default=0.7,
                   help="Weight on hard label in 'mixed' mode "
                        "(0=pure soft, 1=pure hard).")
    p.add_argument("--search-only", action="store_true",
                   help="Drop lowH and terminal examples; train only on "
                        "search-derived decisions (highest-quality signal).")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--search-value-weight", type=float, default=0.5)
    p.add_argument("--freeze-bn", action="store_true",
                   help="Keep BatchNorm layers in eval mode during finetune. "
                        "Useful when starting from C-imported fused BN weights.")
    p.add_argument("--freeze-value-head", action="store_true",
                   help="Freeze value/vp heads and train policy-only. Value "
                        "metrics are still reported, but value losses do not "
                        "contribute gradients to the shared trunk.")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "mps", "cuda"])
    args = p.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from human_bot.model import HumanBotNet
    from human_bot.loss import (
        UncertaintyWeightedLoss, value_loss as outcome_value_loss,
        masked_entropy,
    )
    from human_bot.export_nn import export as export_nn

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"device = {device}")

    # Load M2 checkpoint
    print(f"Loading seed: {args.seed_checkpoint}")
    net = HumanBotNet.load_checkpoint(args.seed_checkpoint, device=device)
    net.train()
    print(f"  params: {sum(p.numel() for p in net.parameters()):,}")

    # Load dataset
    data = load_all_shards(args.shard_dir, args.shard_glob)
    N = data["action_idx"].shape[0]

    # Optionally filter to search-only examples
    if args.search_only:
        sk_np = data["signal_kind"].numpy()
        keep_idx = np.where(sk_np == 0)[0]
        print(f"\nFilter to search-only: {len(keep_idx):,} / {N:,} "
              f"({100*len(keep_idx)/N:.1f}%)")
        for k in list(data.keys()):
            data[k] = data[k][keep_idx]
        N = data["action_idx"].shape[0]

    # Optionally pre-shape the policy target according to --policy-mode
    if args.policy_mode == "soft":
        # Use policy_target as-is (the original behavior)
        pt_pre = data["policy_target"]
    elif args.policy_mode == "sharp":
        pt_pre = sharpen_policy_target(data["policy_target"],
                                       args.policy_temperature)
        print(f"  Sharpened policy_target with T={args.policy_temperature} "
              f"(mean max-prob: {pt_pre.max(dim=1).values.mean():.3f})")
    elif args.policy_mode == "mixed":
        pt_pre = mix_target(data["policy_target"], data["action_idx"],
                            data["action_mask"], args.hard_weight)
        print(f"  Mixed policy_target with hard_weight={args.hard_weight} "
              f"(mean max-prob: {pt_pre.max(dim=1).values.mean():.3f})")
    else:  # hard
        pt_pre = None  # handled separately in training loop

    # Train / val split (deterministic)
    rng = np.random.RandomState(args.seed)
    perm = torch.from_numpy(rng.permutation(N)).long()
    n_val = int(N * args.val_fraction)
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]
    print(f"\nTrain: {trn_idx.shape[0]:,}  Val: {val_idx.shape[0]:,}")

    # Move static data to device
    nf = data["node_features"].to(device)
    ef = data["edge_features"].to(device)
    ff = data["flat_features"].to(device)
    mask = data["action_mask"].to(device)
    act = data["action_idx"].to(device)
    pt = pt_pre.to(device) if pt_pre is not None else None
    sv = data["search_value"].to(device)
    sk = data["signal_kind"].to(device)
    sw = data["step_weight"].to(device)
    vt = value_targets_from_reward(data["reward_vec"], data["player"]).to(device)

    # Setup edge_index
    from hexzero.game.interface import CatanGame
    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    edge_index = se._edge_index.to(device)

    # Loss + optimizer
    loss_combiner = UncertaintyWeightedLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(loss_combiner.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)

    print(f"\nHyperparams: lr={args.lr}, batch={args.batch_size}, "
          f"epochs={args.epochs}")
    print(f"  policy_mode={args.policy_mode}", end="")
    if args.policy_mode == "sharp":
        print(f"  T={args.policy_temperature}")
    elif args.policy_mode == "mixed":
        print(f"  hard_weight={args.hard_weight}")
    elif args.policy_mode == "hard":
        print(f"  label_smoothing={args.label_smoothing}")
    else:
        print()
    print(f"  sv_weight={args.search_value_weight}")
    print(f"  freeze_bn={args.freeze_bn} freeze_value_head={args.freeze_value_head}")
    print(f"  device={device}")
    print()

    if args.freeze_value_head:
        for module in (net.value_head, net.vp_head):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def run_epoch(idx_set, train: bool):
        if train:
            net.train()
            if args.freeze_bn:
                for module in net.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.eval()
            if args.freeze_value_head:
                net.value_head.eval()
                net.vp_head.eval()
        else:
            net.eval()
        n_seen = 0
        n_seen_sv = 0
        accum = {
            "policy_loss": 0.0, "value_loss": 0.0, "search_v_loss": 0.0,
            "policy_acc": 0.0, "value_acc": 0.0, "entropy": 0.0,
            "search_v_mae": 0.0, "total_loss": 0.0,
        }
        # shuffle for train
        if train:
            perm_e = idx_set[torch.randperm(idx_set.shape[0])]
        else:
            perm_e = idx_set
        bs = args.batch_size
        n_steps = (perm_e.shape[0] + bs - 1) // bs
        for s in range(n_steps):
            idx = perm_e[s * bs : (s + 1) * bs]
            B = idx.shape[0]
            batch = {
                "node_features": nf[idx],
                "edge_features": ef[idx],
                "edge_index": edge_index,
                "flat_features": ff[idx],
                "action_mask": mask[idx],
            }
            with torch.set_grad_enabled(train):
                out = net(batch)
                p_logits = out["policy_logits"]
                v_logits = out["value"]

                if args.policy_mode == "hard":
                    p_loss = hard_policy_loss(
                        p_logits, act[idx], mask[idx],
                        label_smoothing=args.label_smoothing,
                        sample_weight=sw[idx])
                else:
                    p_loss = super_policy_loss(
                        p_logits, pt[idx], mask[idx],
                        sample_weight=sw[idx])
                tp = ff[idx][:, 114] if ff.shape[-1] > 114 else None
                v_loss = outcome_value_loss(v_logits, vt[idx], turn_progress=tp)
                ent = masked_entropy(p_logits, mask[idx])

                sv_l, valid = search_value_loss(v_logits, sv[idx], sk[idx])
                if args.freeze_value_head:
                    total = p_loss - args.entropy_weight * ent
                else:
                    total, _ = loss_combiner(p_loss, v_loss, ent,
                                              args.entropy_weight)
                    if args.search_value_weight > 0:
                        total = total + args.search_value_weight * sv_l

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()

                with torch.no_grad():
                    p_pred = p_logits.argmax(dim=-1)
                    p_acc = (p_pred == act[idx]).float().mean().item()
                    v_pred = v_logits.argmax(dim=-1)
                    v_tgt = vt[idx].argmax(dim=-1)
                    v_acc = (v_pred == v_tgt).float().mean().item()
                    if valid.any():
                        probs0 = F.softmax(v_logits, dim=-1)[:, 0]
                        pv = 2 * probs0 - 1
                        mae = (pv[valid] - sv[idx][valid]).abs().mean().item()
                        accum["search_v_mae"] += mae * valid.sum().item()
                        n_seen_sv += valid.sum().item()

            accum["policy_loss"] += p_loss.item() * B
            accum["value_loss"] += v_loss.item() * B
            accum["search_v_loss"] += sv_l.item() * B
            accum["entropy"] += ent.item() * B
            accum["total_loss"] += total.item() * B
            accum["policy_acc"] += p_acc * B
            accum["value_acc"] += v_acc * B
            n_seen += B

        out = {}
        for k in accum:
            if k == "search_v_mae":
                out[k] = accum[k] / max(n_seen_sv, 1)
            else:
                out[k] = accum[k] / max(n_seen, 1)
        return out

    best_val_total = float("inf")
    best_epoch = -1
    for ep in range(args.epochs):
        t0 = time.time()
        trn = run_epoch(trn_idx, train=True)
        t_train = time.time() - t0

        t0 = time.time()
        val = run_epoch(val_idx, train=False)
        t_val = time.time() - t0

        print(f"Epoch {ep+1}/{args.epochs} ({t_train:.0f}s train, {t_val:.0f}s val)")
        print(f"  TRAIN: p_loss={trn['policy_loss']:.4f} "
              f"v_loss={trn['value_loss']:.4f} "
              f"sv_loss={trn['search_v_loss']:.4f} "
              f"sv_mae={trn['search_v_mae']:.3f} "
              f"p_acc={trn['policy_acc']:.3f} v_acc={trn['value_acc']:.3f} "
              f"ent={trn['entropy']:.3f}")
        print(f"  VAL:   p_loss={val['policy_loss']:.4f} "
              f"v_loss={val['value_loss']:.4f} "
              f"sv_loss={val['search_v_loss']:.4f} "
              f"sv_mae={val['search_v_mae']:.3f} "
              f"p_acc={val['policy_acc']:.3f} v_acc={val['value_acc']:.3f} "
              f"ent={val['entropy']:.3f}")

        if val["total_loss"] < best_val_total:
            best_val_total = val["total_loss"]
            best_epoch = ep + 1
            ckpt_dir = os.path.dirname(args.out_pt) or "."
            os.makedirs(ckpt_dir, exist_ok=True)
            net.save_checkpoint(args.out_pt + ".tmp", {
                "round": ep + 1,
                "total_examples": (ep + 1) * trn_idx.shape[0],
                "val_total_loss": val["total_loss"],
                "val_policy_acc": val["policy_acc"],
                "val_value_acc": val["value_acc"],
                "val_search_v_mae": val["search_v_mae"],
                "source": "train_m3_local",
                "seed_checkpoint": args.seed_checkpoint,
            })
            os.rename(args.out_pt + ".tmp", args.out_pt)
            print(f"  ✓ saved best to {args.out_pt} "
                  f"(val_total={val['total_loss']:.4f}, epoch {ep+1})")
        print()

    # Export final best to C binary
    print(f"\n=== Exporting best ({best_epoch}) to {args.out_bin} ===")
    bin_dir = os.path.dirname(args.out_bin) or "."
    os.makedirs(bin_dir, exist_ok=True)
    export_nn(args.out_pt, args.out_bin)
    print(f"\n✓ DONE")
    print(f"  PyTorch checkpoint: {args.out_pt}")
    print(f"  C binary:           {args.out_bin}")
    print(f"  Best epoch:         {best_epoch}/{args.epochs}")
    print(f"  Best val total:     {best_val_total:.4f}")


if __name__ == "__main__":
    main()
