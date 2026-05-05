"""super_learner: trains M2 policy/value head on super_m2 expert actions.

Watches `shard_dir/pending/` for new .pt shards from `super_actor.py`,
trains for one round per `shards_per_train`, exports updated weights to
`weights_bin_path`, and runs evaluation every `eval_interval` rounds.

The training loss is the standard c_selfplay formulation:
    masked + label-smoothed cross-entropy (policy) on super_m2's `action_idx`
    + cross-entropy (value) on per-seat winner distribution
    + entropy bonus, all combined via UncertaintyWeightedLoss.

Eval (every `eval_interval` rounds, blocking on the learner):
    - 1v3: current NN-0ply vs 3x AB2 (depth=ab_depth)         -> winrate, avg rank
    - 1v3: current NN-0ply vs 3x frozen baseline NN-0ply      -> winrate, avg rank

W&B metrics:
    train/* : loss components, lr, round, total_examples
    perf/*  : timings, pending shards
    eval/vs_ab2_winrate, eval/vs_ab2_avg_rank
    eval/vs_0ply_winrate, eval/vs_0ply_avg_rank
    eval/round, eval/elapsed_sec

Usage (called from inside nlprun via deploy_super_exit.sh):
    python -m human_bot.super_learner \
        --checkpoint   checkpoints/super_exit/init.pt \
        --weights-bin  checkpoints/super_exit/nn_weights_latest.bin \
        --baseline-bin checkpoints/super_exit/nn_weights_baseline.bin \
        --shard-dir    data/super_exit \
        --ckpt-dir     checkpoints/super_exit \
        --shards-per-train 4 \
        --batch-size 4096 \
        --eval-interval 10 \
        --eval-games 20 \
        --eval-workers 8 \
        --wandb-project human-bot-super-exit \
        --wandb-name super-exit-$(date +%m%d-%H%M)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# -------- shard ingestion (matches c_selfplay shard layout) ----------
MASK_DIM = 397


def _list_pending(pending_dir):
    if not os.path.isdir(pending_dir):
        return []
    return sorted(
        f for f in os.listdir(pending_dir)
        if f.endswith(".pt") and not f.endswith(".tmp")
    )


def _load_shards(pending_dir, names, num_players_default=4):
    """Load + concatenate a group of shards. Returns dict-of-tensors.

    Optional dense fields (`policy_target`, `signal_kind`) are loaded if
    present in ALL shards; otherwise we fall back to None and the trainer
    uses one-hot from `action_idx`.
    """
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player = [], [], []
    all_reward, all_sw, all_np = [], [], []
    all_pt, all_kind, all_sv = [], [], []
    pt_present_all = True
    sv_present_all = True

    for name in names:
        p = os.path.join(pending_dir, name)
        d = torch.load(p, weights_only=False, map_location="cpu")
        nf = d["node_features"]
        ef = d["edge_features"]
        ff = d["flat_features"]
        mask = d["action_mask"]
        if mask.shape[-1] < MASK_DIM:
            pad = torch.zeros(mask.shape[0], MASK_DIM - mask.shape[-1],
                              dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=-1)
        act = d["action_idx"].to(torch.int64)
        player = d.get("player",
                       torch.zeros(act.shape[0], dtype=torch.int64))
        reward = d.get("reward_vec",
                       torch.zeros(act.shape[0], 4, dtype=torch.float32))
        sw = d.get("step_weight",
                   torch.ones(act.shape[0], dtype=torch.float32))
        np_t = d.get("num_players",
                     torch.full((act.shape[0],), num_players_default,
                                dtype=torch.int64))
        all_nf.append(nf); all_ef.append(ef); all_ff.append(ff)
        all_mask.append(mask); all_act.append(act); all_player.append(player)
        all_reward.append(reward); all_sw.append(sw); all_np.append(np_t)

        if "policy_target" in d:
            pt = d["policy_target"]
            if pt.shape[-1] < MASK_DIM:
                pad = torch.zeros(pt.shape[0], MASK_DIM - pt.shape[-1],
                                  dtype=pt.dtype)
                pt = torch.cat([pt, pad], dim=-1)
            all_pt.append(pt)
            all_kind.append(d.get("signal_kind",
                                  torch.zeros(act.shape[0], dtype=torch.int64)))
        else:
            pt_present_all = False

        if "search_value" in d:
            all_sv.append(d["search_value"].to(torch.float32))
        else:
            sv_present_all = False

    out = {
        "nf": torch.cat(all_nf),
        "ef": torch.cat(all_ef),
        "ff": torch.cat(all_ff),
        "mask": torch.cat(all_mask),
        "action_idx": torch.cat(all_act),
        "player": torch.cat(all_player),
        "reward_vec": torch.cat(all_reward),
        "step_weight": torch.cat(all_sw),
        "num_players": torch.cat(all_np),
    }
    if pt_present_all and all_pt:
        out["policy_target"] = torch.cat(all_pt)
        out["signal_kind"] = torch.cat(all_kind)
    if sv_present_all and all_sv:
        out["search_value"] = torch.cat(all_sv)
    return out


def super_policy_loss(logits, policy_target, mask, sample_weight=None):
    """Soft cross-entropy: target is a probability distribution over the 397
    action space (e.g., softmax of search values).

    Equivalent to KL divergence up to a target-only entropy term:
        H(target) is constant w.r.t. logits, so minimizing -sum(target *
        log_softmax(logits)) is the same as minimizing KL(target || policy).
    """
    fill = -6e4 if logits.dtype == torch.float16 else -1e9
    masked_logits = logits.masked_fill(~mask.bool(), fill)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    per_example = -(policy_target * log_probs).sum(dim=-1)
    if sample_weight is not None:
        return (per_example * sample_weight).mean()
    return per_example.mean()


def search_value_loss(value_logits, search_value, signal_kind):
    """MSE between predicted state value and search-derived V(s).

    The value head produces (B, 4) logits over which seat will win. We
    interpret slot 0 (= acting player after rotate_value_targets_to_cp)
    as P(acting_player wins). To compare against `search_value` ∈ [-1, 1]
    (the deep_search V(s) from acting player's perspective), we transform:
        predicted_v = 2 * softmax(logits)[:, 0] - 1     ∈ [-1, 1]

    Loss = MSE(predicted_v, search_value), masked to exclude rows where
    signal_kind == 1 ('lowH', no search was run so search_value is a
    placeholder). Also excludes rows with NaN search_value.

    Returns a tuple (loss_scalar, n_valid) where n_valid is the number of
    rows that contributed (useful for metric reporting).
    """
    probs = torch.nn.functional.softmax(value_logits, dim=-1)
    pred_v = 2.0 * probs[:, 0] - 1.0   # (B,) ∈ [-1, 1]

    # Mask: exclude lowH (signal_kind == 1) and any NaN/inf in search_value
    valid = (signal_kind != 1) & torch.isfinite(search_value)
    valid_f = valid.float()
    n_valid = valid_f.sum().clamp(min=1.0)

    diff = (pred_v - search_value).pow(2)
    diff = diff * valid_f
    return diff.sum() / n_valid, int(valid.sum().item())


def _value_targets_from_reward(reward_vec, players, num_players):
    """Rotate winner one-hot so slot 0 = current player."""
    from human_bot.dataset import rotate_value_targets_to_cp
    rv = reward_vec.numpy()
    pl = players.numpy()
    np_arr = num_players.numpy()
    winners = rv.argmax(axis=1)
    has_winner = rv.max(axis=1) > 0.5
    S = rv.shape[0]
    vt = np.zeros((S, 4), dtype=np.float32)
    vt[np.arange(S), winners] = 1.0
    vt[~has_winner] = 0.25
    vt = rotate_value_targets_to_cp(vt, pl, np_arr)
    return torch.from_numpy(vt)


# -------- main learner ----------

def run(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from human_bot.model import HumanBotNet
    from human_bot.loss import (
        UncertaintyWeightedLoss, human_policy_loss, value_loss,
        masked_entropy,
    )
    from human_bot.export_nn import export as export_nn

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[learner] device = {device}", flush=True)
    if device == "cuda:0":
        print(f"[learner] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    shard_dir = os.path.abspath(args.shard_dir)
    weights_bin = os.path.abspath(args.weights_bin)
    baseline_bin = os.path.abspath(args.baseline_bin)
    pending_dir = os.path.join(shard_dir, "pending")
    stop_file = os.path.join(ckpt_dir, ".stop")
    round_file = os.path.join(ckpt_dir, ".round")
    wandb_id_path = os.path.join(ckpt_dir, ".wandb_id")
    history_dir = os.path.join(ckpt_dir, "prev_checkpoints")
    eval_dir = os.path.join(ckpt_dir, "eval")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(pending_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.dirname(weights_bin) or ".", exist_ok=True)

    # Resume vs fresh
    if os.path.exists(stop_file):
        print(f"[learner] removing pre-existing {stop_file}", flush=True)
        os.remove(stop_file)

    seed_ckpt = args.checkpoint
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        ckpt_to_load = latest_ckpt
        print(f"[learner] resuming from {latest_ckpt}", flush=True)
    else:
        if not os.path.exists(seed_ckpt):
            raise FileNotFoundError(
                f"--checkpoint not found: {seed_ckpt}\n"
                "Provide an initial M2 .pt as the seed.")
        ckpt_to_load = seed_ckpt
        print(f"[learner] seeding from {seed_ckpt}", flush=True)

    net = HumanBotNet.load_checkpoint(ckpt_to_load, device=device)
    net.train()

    round_num = 0
    total_examples = 0
    if os.path.exists(round_file):
        try:
            round_num = int(open(round_file).read().strip())
        except Exception:
            round_num = 0

    # Always (re)export baseline_bin from the SEED checkpoint at startup.
    # This is the frozen "0-ply baseline" used for the vs_0ply eval.
    # If baseline_bin already exists we leave it as-is (frozen forever).
    if not os.path.exists(baseline_bin):
        print(f"[learner] exporting baseline weights {baseline_bin} "
              f"from {seed_ckpt}", flush=True)
        export_nn(seed_ckpt, baseline_bin)

    # Export current weights for actors to consume.
    print(f"[learner] exporting current weights {weights_bin}", flush=True)
    tmp_pt = os.path.join(ckpt_dir, "_export_tmp.pt")
    net.save_checkpoint(tmp_pt, {"round": round_num})
    export_nn(tmp_pt, weights_bin)
    with open(round_file, "w") as f:
        f.write(str(round_num))

    # -- W&B init (resume-by-id) --
    wandb_run = None
    try:
        import wandb
        if "WANDB_API_KEY" in os.environ:
            run_name = args.wandb_name or f"super-exit-{time.strftime('%m%d-%H%M')}"
            existing_id = None
            if os.path.exists(wandb_id_path):
                try:
                    existing_id = open(wandb_id_path).read().strip() or None
                except Exception:
                    existing_id = None
            init_kwargs = {
                "project": args.wandb_project,
                "name": run_name,
                "config": {
                    "shards_per_train": args.shards_per_train,
                    "batch_size": args.batch_size,
                    "eval_interval": args.eval_interval,
                    "eval_games": args.eval_games,
                    "label_smoothing": args.label_smoothing,
                    "entropy_weight": args.entropy_weight,
                    "search_value_weight": args.search_value_weight,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "device": device,
                    "ab_depth": args.ab_depth,
                },
            }
            if existing_id:
                init_kwargs["id"] = existing_id
                init_kwargs["resume"] = "allow"
                print(f"[learner] resuming W&B run {existing_id}", flush=True)
            wandb_run = wandb.init(**init_kwargs)
            if not existing_id and wandb_run is not None:
                with open(wandb_id_path, "w") as f:
                    f.write(wandb_run.id)
                print(f"[learner] new W&B run id {wandb_run.id} -> {wandb_id_path}",
                      flush=True)
        else:
            print(f"[learner] WANDB_API_KEY not set; W&B disabled", flush=True)
    except Exception as e:
        print(f"[learner] W&B init failed: {e}; continuing without W&B",
              flush=True)
        wandb_run = None

    # -- Loss + optimizer --
    loss_combiner = UncertaintyWeightedLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(net.parameters()) + list(loss_combiner.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    grad_clip = 1.0

    # Cache the static edge_index on the GPU (needed by HumanBotNet.forward)
    from hexzero.game.interface import CatanGame
    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    edge_index = se._edge_index.to(device)

    print(f"[learner] starting main loop "
          f"(shards_per_train={args.shards_per_train}, "
          f"batch_size={args.batch_size}, "
          f"eval_interval={args.eval_interval})", flush=True)

    last_eval_round = -1
    while not os.path.exists(stop_file):
        # 1) Wait for enough shards
        names = _list_pending(pending_dir)
        if len(names) < args.shards_per_train:
            time.sleep(5)
            continue
        chosen = names[:args.shards_per_train]
        n_pending = len(names)

        # 2) Load shards
        t_load_start = time.time()
        try:
            data = _load_shards(pending_dir, chosen)
        except Exception as e:
            print(f"[learner] shard load error: {e}; skipping batch",
                  flush=True)
            traceback.print_exc()
            for nm in chosen:
                p = os.path.join(pending_dir, nm)
                try:
                    os.remove(p)
                except OSError:
                    pass
            continue
        t_load = time.time() - t_load_start

        # Delete consumed shards (so actors can produce more under backpressure)
        for nm in chosen:
            try:
                os.remove(os.path.join(pending_dir, nm))
            except OSError:
                pass

        # 3) Build training tensors
        N_examples = data["nf"].shape[0]
        if N_examples == 0:
            print(f"[learner] all shards empty; skipping", flush=True)
            continue
        total_examples += N_examples

        nf_d = data["nf"].to(device)
        ef_d = data["ef"].to(device)
        ff_d = data["ff"].to(device)
        mask_d = data["mask"].to(device)
        act_d = data["action_idx"].to(device)
        sw_d = data["step_weight"].to(device)
        vt_t = _value_targets_from_reward(
            data["reward_vec"], data["player"], data["num_players"])
        vt_d = vt_t.to(device)

        has_dense = "policy_target" in data
        if has_dense:
            pt_d = data["policy_target"].to(device)
            kind_d = data["signal_kind"].to(device)
        else:
            pt_d = None
            kind_d = None
        has_sv = "search_value" in data
        sv_d = data["search_value"].to(device) if has_sv else None

        # 4) Single-pass shuffle-and-train (one "epoch" = one pass)
        t_train_start = time.time()
        perm = torch.randperm(N_examples, device=device)
        bs = min(args.batch_size, 4096)
        n_steps = (N_examples + bs - 1) // bs
        accum = {"policy_loss": 0.0, "value_loss": 0.0,
                 "search_v_loss": 0.0,
                 "policy_acc": 0.0, "value_acc": 0.0,
                 "entropy": 0.0, "total_loss": 0.0,
                 "search_v_mae": 0.0}
        n_seen = 0
        n_seen_sv = 0
        for s in range(n_steps):
            idx = perm[s * bs : (s + 1) * bs]
            B = idx.shape[0]
            batch = {
                "node_features": nf_d[idx],
                "edge_features": ef_d[idx],
                "edge_index": edge_index,
                "flat_features": ff_d[idx],
                "action_mask": mask_d[idx],
            }
            out = net(batch)

            policy_logits = out["policy_logits"]
            value_logits = out["value"]

            if has_dense:
                # Dense soft-target loss: search-derived distribution
                p_loss = super_policy_loss(
                    policy_logits, pt_d[idx], mask_d[idx],
                    sample_weight=sw_d[idx])
            else:
                # Fallback: hard-label CE on argmax action
                p_loss = human_policy_loss(
                    policy_logits, act_d[idx], mask_d[idx],
                    label_smoothing=args.label_smoothing,
                    winner_boost=sw_d[idx])

            tp = ff_d[idx][:, 114] if ff_d.shape[-1] > 114 else None
            v_loss = value_loss(value_logits, vt_d[idx], turn_progress=tp)
            ent = masked_entropy(policy_logits, mask_d[idx])
            total, _ = loss_combiner(p_loss, v_loss, ent, args.entropy_weight)

            # Auxiliary search-value regression: dense per-state V(s) target.
            # Trains the value head to predict deep_search's exact value
            # estimate, in addition to the (sparse) game-outcome target.
            if has_sv and args.search_value_weight > 0:
                sv_l, n_valid = search_value_loss(
                    value_logits, sv_d[idx],
                    kind_d[idx] if kind_d is not None else
                    torch.zeros(B, dtype=torch.long, device=device))
                total = total + args.search_value_weight * sv_l
                with torch.no_grad():
                    probs0 = torch.nn.functional.softmax(value_logits, dim=-1)[:, 0]
                    pred_v = 2.0 * probs0 - 1.0
                    valid_mask = (
                        (kind_d[idx] != 1) if kind_d is not None
                        else torch.ones_like(pred_v, dtype=torch.bool)
                    ) & torch.isfinite(sv_d[idx])
                    if valid_mask.any():
                        mae = (pred_v[valid_mask] - sv_d[idx][valid_mask]).abs().mean().item()
                    else:
                        mae = 0.0
                accum["search_v_loss"] += sv_l.item() * B
                accum["search_v_mae"] += mae * max(n_valid, 1)
                n_seen_sv += max(n_valid, 1)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                p_pred = policy_logits.argmax(dim=-1)
                p_acc = (p_pred == act_d[idx]).float().mean().item()
                v_pred = value_logits.argmax(dim=-1)
                v_tgt = vt_d[idx].argmax(dim=-1)
                v_acc = (v_pred == v_tgt).float().mean().item()

            accum["policy_loss"] += p_loss.item() * B
            accum["value_loss"] += v_loss.item() * B
            accum["entropy"] += ent.item() * B
            accum["total_loss"] += total.item() * B
            accum["policy_acc"] += p_acc * B
            accum["value_acc"] += v_acc * B
            n_seen += B

        avg = {k: v / max(n_seen, 1) for k, v in accum.items()
               if k != "search_v_mae"}
        avg["search_v_mae"] = accum["search_v_mae"] / max(n_seen_sv, 1)
        t_train = time.time() - t_train_start

        round_num += 1

        # 5) Save checkpoint + export weights
        t_export_start = time.time()
        ckpt_path = os.path.join(ckpt_dir, "latest.pt")
        net.save_checkpoint(ckpt_path + ".tmp", {
            "round": round_num,
            "total_examples": total_examples,
            **avg,
        })
        os.rename(ckpt_path + ".tmp", ckpt_path)
        export_nn(ckpt_path, weights_bin)
        with open(round_file, "w") as f:
            f.write(str(round_num))
        t_export = time.time() - t_export_start

        # Snapshot every 50 rounds (small disk footprint, useful for rollback)
        if round_num % 50 == 0:
            snap = os.path.join(history_dir, f"round_{round_num:05d}.pt")
            net.save_checkpoint(snap, {"round": round_num})

        elapsed = t_load + t_train + t_export
        signal_tag = "DENSE" if has_dense else "sparse"
        sv_tag = ""
        if has_sv and args.search_value_weight > 0:
            sv_tag = (f" sv_loss={avg['search_v_loss']:.4f} "
                      f"sv_mae={avg['search_v_mae']:.3f}")
        print(f"[learner] round {round_num} ({signal_tag}) | "
              f"shards={len(chosen)} ex={N_examples} "
              f"p_loss={avg['policy_loss']:.4f} "
              f"v_loss={avg['value_loss']:.4f}"
              f"{sv_tag} "
              f"p_acc={avg['policy_acc']:.3f} v_acc={avg['value_acc']:.3f} "
              f"ent={avg['entropy']:.3f} | "
              f"load={t_load:.1f}s train={t_train:.1f}s "
              f"export={t_export:.1f}s pending={n_pending}",
              flush=True)

        # 6) W&B log
        if wandb_run is not None:
            try:
                import wandb
                log_dict = {
                    "train/policy_loss": avg["policy_loss"],
                    "train/value_loss": avg["value_loss"],
                    "train/total_loss": avg["total_loss"],
                    "train/policy_acc": avg["policy_acc"],
                    "train/value_acc": avg["value_acc"],
                    "train/entropy": avg["entropy"],
                    "train/round": round_num,
                    "train/total_examples": total_examples,
                    "train/lr": float(args.lr),
                    "train/examples_this_round": N_examples,
                    "train/dense": int(has_dense),
                    "train/has_search_value": int(has_sv),
                    "perf/load_sec": t_load,
                    "perf/train_sec": t_train,
                    "perf/export_sec": t_export,
                    "perf/round_sec": elapsed,
                    "perf/pending_shards": n_pending,
                    "perf/examples_per_sec": N_examples / max(t_train, 0.01),
                }
                if has_sv and args.search_value_weight > 0:
                    log_dict["train/search_v_loss"] = avg["search_v_loss"]
                    log_dict["train/search_v_mae"] = avg["search_v_mae"]
                    log_dict["train/search_v_weight"] = float(args.search_value_weight)
                wandb.log(log_dict, step=round_num)
            except Exception as e:
                print(f"[learner] W&B log failed: {e}", flush=True)

        # 7) Eval every `eval_interval` rounds
        if (round_num % args.eval_interval == 0
                and round_num != last_eval_round):
            last_eval_round = round_num
            print(f"[learner] running eval at round {round_num}",
                  flush=True)
            eval_t0 = time.time()
            try:
                from human_bot.super_eval import eval_1v3
                ab2_res = eval_1v3(
                    our_weights_bin=weights_bin,
                    opp_kind="ab2",
                    n_games=args.eval_games,
                    ab_depth=args.ab_depth,
                    num_workers=args.eval_workers,
                    seed_base=900000 + round_num * 1000)
                nn_res = eval_1v3(
                    our_weights_bin=weights_bin,
                    opp_kind="nn",
                    opp_weights_bin=baseline_bin,
                    n_games=args.eval_games,
                    num_workers=args.eval_workers,
                    seed_base=950000 + round_num * 1000)
                eval_elapsed = time.time() - eval_t0

                print(f"[learner] eval round {round_num}:"
                      f"  vs_AB2  WR={100*ab2_res['winrate']:.1f}% "
                      f"rank={ab2_res['avg_rank']:.2f}  "
                      f"vs_0ply WR={100*nn_res['winrate']:.1f}% "
                      f"rank={nn_res['avg_rank']:.2f}  "
                      f"({eval_elapsed:.0f}s)", flush=True)

                # Persist eval to disk
                import json
                with open(os.path.join(eval_dir,
                                       f"round_{round_num:05d}.json"),
                          "w") as f:
                    json.dump({
                        "round": round_num,
                        "vs_ab2": ab2_res,
                        "vs_0ply": nn_res,
                    }, f, indent=2, default=str)

                if wandb_run is not None:
                    import wandb
                    wandb.log({
                        "eval/vs_ab2_winrate": ab2_res["winrate"],
                        "eval/vs_ab2_avg_rank": ab2_res["avg_rank"],
                        "eval/vs_ab2_avg_vp": ab2_res["vps_avg"],
                        "eval/vs_0ply_winrate": nn_res["winrate"],
                        "eval/vs_0ply_avg_rank": nn_res["avg_rank"],
                        "eval/vs_0ply_avg_vp": nn_res["vps_avg"],
                        "eval/round": round_num,
                        "eval/elapsed_sec": eval_elapsed,
                    }, step=round_num)
            except Exception as e:
                print(f"[learner] eval failed: {e}", flush=True)
                traceback.print_exc()

    print(f"[learner] stop file detected; exiting at round {round_num}",
          flush=True)
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Initial seed .pt (e.g. M2 weights). Used only on "
                        "first launch; subsequent restarts resume from "
                        "ckpt_dir/latest.pt.")
    p.add_argument("--weights-bin", type=str, required=True,
                   help="Path where the latest C-format weights are written "
                        "for actors to consume.")
    p.add_argument("--baseline-bin", type=str, required=True,
                   help="Path to frozen baseline .bin (exported from "
                        "--checkpoint at first launch). Used as the "
                        "opponent in vs_0ply eval.")
    p.add_argument("--shard-dir", type=str, required=True)
    p.add_argument("--ckpt-dir", type=str, required=True)
    p.add_argument("--shards-per-train", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--search-value-weight", type=float, default=0.5,
                   help="Weight on dense search-value MSE loss "
                        "(0 to disable). Auxiliary to outcome-based "
                        "value_loss. Both signals teach the value head.")
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--eval-workers", type=int, default=8)
    p.add_argument("--ab-depth", type=int, default=2)
    p.add_argument("--wandb-project", type=str,
                   default="human-bot-super-exit")
    p.add_argument("--wandb-name", type=str, default=None)
    args = p.parse_args()

    try:
        run(args)
    except Exception:
        print("!!! [learner] CRASHED !!!", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
