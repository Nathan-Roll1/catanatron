"""Eggroll-style evolution over existing final policy bias tensors.

This is more expressive than the 12-d type-bias search but still keeps runtime
unchanged: it only edits existing exported weights.

Evolved tensors:
  - policy_head.type_fc.3.bias                 (12)
  - policy_head.discard_yop_mono_fc.2.bias     (30)
  - policy_head.maritime_fc.2.bias             (20)
  - policy_head.trade_fc.2.bias                (67)
  - policy_head.robber_scorer.2.bias           (5)

Spatial settlement/city/road global final biases are intentionally excluded
because they cancel inside per-type log_softmax.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from human_bot.evolve_m2_type_bias import (
    PROJECT_ROOT,
    ROAD_PENALTY_X3,
    export,
    run_many_eval,
    run_single_eval,
)
from human_bot.model import HumanBotNet


BIAS_KEYS = [
    ("policy_head.type_fc.3.bias", 12),
    ("policy_head.discard_yop_mono_fc.2.bias", 30),
    ("policy_head.maritime_fc.2.bias", 20),
    ("policy_head.trade_fc.2.bias", 67),
    ("policy_head.robber_scorer.2.bias", 5),
]


def split_vec(vec: np.ndarray):
    offset = 0
    for key, n in BIAS_KEYS:
        yield key, vec[offset:offset + n]
        offset += n


def initial_vec(name: str) -> np.ndarray:
    total = sum(n for _key, n in BIAS_KEYS)
    out = np.zeros(total, dtype=np.float32)
    if name == "road_penalty_x3":
        out[:12] = ROAD_PENALTY_X3
    elif name != "zero":
        vals = np.array(json.loads(name), dtype=np.float32)
        if vals.shape != (total,):
            raise ValueError(f"expected JSON list of {total} floats")
        out[:] = vals
    return out


def clamp(vec: np.ndarray, max_abs: float) -> np.ndarray:
    out = np.clip(vec, -max_abs, max_abs).astype(np.float32)
    out[0] = 0.0  # ROLL bias is not useful and can only add noise.
    return out


def export_policy_bias(base_state, seed_checkpoint: Path, vec: np.ndarray,
                       ckpt_path: Path, bin_path: Path, metadata: dict,
                       quiet: bool = True):
    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    state = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    for key, sub in split_vec(vec):
        state[key] = state[key] + torch.from_numpy(sub).to(state[key].dtype)
    net.load_state_dict(state, strict=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(ckpt_path), metadata)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            export(str(ckpt_path), str(bin_path))
    else:
        export(str(ckpt_path), str(bin_path))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", default="checkpoints/sp_latest2.pt")
    p.add_argument("--opponent-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument("--out-pt", default="autoresearch-results/m2_policy_bias_es.pt")
    p.add_argument("--out-bin", default="/tmp/catan_m2_policy_bias_es.bin")
    p.add_argument("--work-dir", default="autoresearch-results/policy_bias_es")
    p.add_argument("--initial", default="road_penalty_x3")
    p.add_argument("--pairs", type=int, default=24)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.8)
    p.add_argument("--max-abs", type=float, default=3.0)
    p.add_argument("--seed-base", type=int, default=3440000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    seed_checkpoint = (PROJECT_ROOT / args.seed_checkpoint).resolve()
    opponent_bin = (PROJECT_ROOT / args.opponent_bin).resolve()
    out_pt = (PROJECT_ROOT / args.out_pt).resolve() if not Path(args.out_pt).is_absolute() else Path(args.out_pt)
    out_bin = (PROJECT_ROOT / args.out_bin).resolve() if not Path(args.out_bin).is_absolute() else Path(args.out_bin)
    work_dir = (PROJECT_ROOT / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    base_net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in base_net.state_dict().items()}
    mean = clamp(initial_vec(args.initial), args.max_abs)
    dims = mean.shape[0]
    directions = rng.normal(size=(args.pairs, dims)).astype(np.float32)
    directions[:, 0] = 0.0

    vecs = [mean]
    tags = ["mean"]
    for i, eps in enumerate(directions):
        vecs.extend([
            clamp(mean + args.sigma * eps, args.max_abs),
            clamp(mean - args.sigma * eps, args.max_abs),
        ])
        tags.extend([f"p{i}_plus", f"p{i}_minus"])

    bins = []
    t0 = time.time()
    for i, (tag, vec) in enumerate(zip(tags, vecs)):
        ckpt = work_dir / f"{i:02d}_{tag}.pt"
        bin_path = work_dir / f"{i:02d}_{tag}.bin"
        export_policy_bias(
            base_state,
            seed_checkpoint,
            vec,
            ckpt,
            bin_path,
            {"policy_bias_es": True, "tag": tag, "vec": vec.tolist()},
        )
        bins.append(bin_path)

    eval_res = run_many_eval(bins, opponent_bin, args.games, args.workers,
                             args.seed_base)
    scores = np.array([r["a_winrate"] for r in eval_res["results"]], dtype=np.float32)
    best_idx = int(np.argmax(scores))
    control = run_single_eval(opponent_bin, opponent_bin, args.games,
                              args.workers, args.seed_base)

    export_policy_bias(
        base_state,
        seed_checkpoint,
        vecs[best_idx],
        out_pt,
        out_bin,
        {
            "policy_bias_es": True,
            "selected": tags[best_idx],
            "score": float(scores[best_idx]),
            "wins": int(eval_res["results"][best_idx]["a_wins"]),
            "games": args.games,
            "vec": vecs[best_idx].tolist(),
        },
        quiet=False,
    )

    result = {
        "ok": True,
        "elapsed_sec": time.time() - t0,
        "dims": dims,
        "population": len(vecs),
        "best_tag": tags[best_idx],
        "best_score": float(scores[best_idx]),
        "best_wins": int(eval_res["results"][best_idx]["a_wins"]),
        "games": args.games,
        "control_a_winrate": float(control["a_winrate"]),
        "control_a_wins": int(control["a_wins"]),
        "out_pt": str(out_pt),
        "out_bin": str(out_bin),
        "best_vec": vecs[best_idx].tolist(),
        "scores": [
            {"tag": tag, **row}
            for tag, row in zip(tags, eval_res["results"])
        ],
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
