"""Compact no-search head calibration evolution for M2.

This evolves only low-dimensional calibration knobs that are exported as
ordinary weights:

  - 12 action-type biases
  - type head temperature
  - discard/yop/mono, maritime, trade head temperatures
  - settlement/city/road/robber spatial head temperatures

Unlike pure type-bias tweaks, temperature scaling can change state-dependent
argmax decisions by sharpening or flattening learned logits while preserving
one plain C ``nn_forward`` at runtime.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from human_bot.evolve_m2_spatial_policy import run_many_eval_ffa
from human_bot.evolve_m2_type_bias import (
    PROJECT_ROOT,
    ROAD_PENALTY_X3,
    export,
    run_single_eval,
)
from human_bot.model import HumanBotNet


TEMP_KEYS = [
    ("type", ("policy_head.type_fc.3.weight", "policy_head.type_fc.3.bias")),
    ("discard_yop_mono", ("policy_head.discard_yop_mono_fc.2.weight", "policy_head.discard_yop_mono_fc.2.bias")),
    ("maritime", ("policy_head.maritime_fc.2.weight", "policy_head.maritime_fc.2.bias")),
    ("trade", ("policy_head.trade_fc.2.weight", "policy_head.trade_fc.2.bias")),
    ("settlement", ("policy_head.settlement_scorer.2.weight", "policy_head.settlement_scorer.2.bias")),
    ("city", ("policy_head.city_scorer.2.weight", "policy_head.city_scorer.2.bias")),
    ("road", ("policy_head.road_scorer.2.weight", "policy_head.road_scorer.2.bias")),
    ("robber", ("policy_head.robber_scorer.2.weight", "policy_head.robber_scorer.2.bias")),
]

DIM = 12 + len(TEMP_KEYS)


def initial_vec(name: str) -> np.ndarray:
    out = np.zeros(DIM, dtype=np.float32)
    if name == "road_penalty_x3":
        out[:12] = ROAD_PENALTY_X3
    elif name == "zero":
        pass
    else:
        vals = np.array(json.loads(name), dtype=np.float32)
        if vals.shape != (12,):
            raise ValueError("initial must be zero, road_penalty_x3, or JSON 12-list")
        out[:12] = vals
    return out


def scale_vec(type_bias_scale: float, log_temp_scale: float) -> np.ndarray:
    out = np.empty(DIM, dtype=np.float32)
    out[:12] = type_bias_scale
    out[12:] = log_temp_scale
    out[0] = 0.0
    return out


def clamp_vec(vec: np.ndarray, type_max_abs: float, log_temp_max_abs: float) -> np.ndarray:
    out = vec.astype(np.float32).copy()
    out[:12] = np.clip(out[:12], -type_max_abs, type_max_abs)
    out[12:] = np.clip(out[12:], -log_temp_max_abs, log_temp_max_abs)
    out[0] = 0.0
    return out


def export_calibrated(
    base_state: dict[str, torch.Tensor],
    seed_checkpoint: Path,
    vec: np.ndarray,
    ckpt_path: Path,
    bin_path: Path,
    metadata: dict,
    quiet: bool = True,
) -> None:
    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    state = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    for offset, (_name, keys) in enumerate(TEMP_KEYS):
        mult = float(math.exp(float(vec[12 + offset])))
        for key in keys:
            state[key] = state[key] * mult
    key = "policy_head.type_fc.3.bias"
    state[key] = state[key] + torch.from_numpy(vec[:12]).to(state[key].dtype)
    net.load_state_dict(state, strict=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(ckpt_path), metadata)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            export(str(ckpt_path), str(bin_path))
    else:
        export(str(ckpt_path), str(bin_path))


def dedupe_top(records: list[dict], limit: int) -> list[dict]:
    out = []
    seen = set()
    for rec in sorted(records, key=lambda r: r["score"], reverse=True):
        key = tuple(np.round(rec["vec"], 5).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", default="checkpoints/sp_latest2.pt")
    p.add_argument("--opponent-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument("--out-pt", default="autoresearch-results/m2_head_calibration.pt")
    p.add_argument("--out-bin", default="/tmp/catan_m2_head_calibration.bin")
    p.add_argument("--work-dir", default="autoresearch-results/head_calibration_es")
    p.add_argument("--initial", default="road_penalty_x3")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--promote-top", type=int, default=8)
    p.add_argument("--promote-games", type=int, default=128)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.35)
    p.add_argument("--type-bias-scale", type=float, default=0.5)
    p.add_argument("--log-temp-scale", type=float, default=0.25)
    p.add_argument("--type-max-abs", type=float, default=3.0)
    p.add_argument("--temp-max", type=float, default=2.5)
    p.add_argument("--seed-base", type=int, default=4100000)
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
    base_vec = initial_vec(args.initial)
    per_dim_scale = scale_vec(args.type_bias_scale, args.log_temp_scale)
    log_temp_max_abs = float(math.log(args.temp_max))
    mean_z = np.zeros(DIM, dtype=np.float32)
    records: list[dict] = []
    history: list[dict] = []
    t0 = time.time()

    for gen in range(args.generations):
        gen_dir = work_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        dirs = rng.normal(size=(args.pairs, DIM)).astype(np.float32)
        dirs[:, 0] = 0.0
        zs = [mean_z.copy()]
        tags = ["mean"]
        pair_index = []
        for i, eps in enumerate(dirs):
            pair_index.append((len(zs), len(zs) + 1))
            zs.extend([mean_z + args.sigma * eps, mean_z - args.sigma * eps])
            tags.extend([f"p{i}_plus", f"p{i}_minus"])

        bins = []
        vecs = []
        for i, (tag, z) in enumerate(zip(tags, zs)):
            vec = clamp_vec(base_vec + per_dim_scale * z,
                            args.type_max_abs, log_temp_max_abs)
            vecs.append(vec)
            ckpt = gen_dir / f"{i:02d}_{tag}.pt"
            bin_path = gen_dir / f"{i:02d}_{tag}.bin"
            export_calibrated(base_state, seed_checkpoint, vec, ckpt, bin_path,
                              {"head_calibration_es": True, "tag": tag, "vec": vec.tolist()})
            bins.append(bin_path)

        seed_base = args.seed_base + gen * 10000
        eval_res = run_many_eval_ffa(bins, opponent_bin, args.games,
                                     args.workers, seed_base)
        scores = np.array([r["a_winrate"] for r in eval_res["results"]], dtype=np.float32)
        update = np.zeros(DIM, dtype=np.float32)
        for i, (plus_idx, minus_idx) in enumerate(pair_index):
            update += (scores[plus_idx] - scores[minus_idx]) * dirs[i]
        update /= max(1e-6, 2.0 * args.pairs * args.sigma)
        mean_z = mean_z + args.lr * update
        mean_vec = clamp_vec(base_vec + per_dim_scale * mean_z,
                             args.type_max_abs, log_temp_max_abs)
        mean_z = (mean_vec - base_vec) / np.maximum(per_dim_scale, 1e-6)
        mean_z[0] = 0.0
        best_idx = int(np.argmax(scores))

        for tag, row, vec in zip(tags, eval_res["results"], vecs):
            records.append({
                "generation": gen,
                "tag": tag,
                "score": float(row["a_winrate"]),
                "wins": int(row["a_wins"]),
                "games": int(row["games"]),
                "vec": vec,
            })
        h = {
            "generation": gen,
            "seed_base": seed_base,
            "mean_score": float(scores[0]),
            "best_score": float(scores[best_idx]),
            "best_tag": tags[best_idx],
            "best_wins": int(eval_res["results"][best_idx]["a_wins"]),
            "control_a_winrate": 0.25,
            "control_a_wins": args.games,
        }
        history.append(h)
        print(json.dumps(h, sort_keys=True), flush=True)

    promote = dedupe_top(records, args.promote_top)
    promote_dir = work_dir / "promote"
    promote_bins = []
    for i, rec in enumerate(promote):
        ckpt = promote_dir / f"{i:02d}_g{rec['generation']}_{rec['tag']}.pt"
        bin_path = promote_dir / f"{i:02d}_g{rec['generation']}_{rec['tag']}.bin"
        export_calibrated(base_state, seed_checkpoint, rec["vec"], ckpt, bin_path,
                          {
                              "head_calibration_es": True,
                              "source_generation": rec["generation"],
                              "source_tag": rec["tag"],
                              "proxy_score": rec["score"],
                              "vec": rec["vec"].tolist(),
                          })
        promote_bins.append(bin_path)

    promote_seed_base = args.seed_base + args.generations * 10000 + 5000
    promote_res = run_many_eval_ffa(promote_bins, opponent_bin,
                                    args.promote_games, args.workers,
                                    promote_seed_base)
    promote_scores = np.array([r["a_winrate"] for r in promote_res["results"]], dtype=np.float32)
    best_idx = int(np.argmax(promote_scores))
    best = promote[best_idx]
    export_calibrated(
        base_state,
        seed_checkpoint,
        best["vec"],
        out_pt,
        out_bin,
        {
            "head_calibration_es": True,
            "selected": f"g{best['generation']}_{best['tag']}",
            "proxy_score": best["score"],
            "promote_score": float(promote_scores[best_idx]),
            "promote_wins": int(promote_res["results"][best_idx]["a_wins"]),
            "promote_games": int(promote_res["results"][best_idx]["games"]),
            "vec": best["vec"].tolist(),
        },
        quiet=False,
    )
    result = {
        "ok": True,
        "elapsed_sec": time.time() - t0,
        "dims": DIM,
        "generations": args.generations,
        "population_per_generation": 1 + 2 * args.pairs,
        "best_tag": f"g{best['generation']}_{best['tag']}",
        "best_proxy_score": best["score"],
        "best_promote_score": float(promote_scores[best_idx]),
        "best_promote_wins": int(promote_res["results"][best_idx]["a_wins"]),
        "promote_games": int(promote_res["results"][best_idx]["games"]),
        "out_pt": str(out_pt),
        "out_bin": str(out_bin),
        "history": history,
        "promote_scores": [
            {
                "source_tag": f"g{rec['generation']}_{rec['tag']}",
                "proxy_score": rec["score"],
                **row,
            }
            for rec, row in zip(promote, promote_res["results"])
        ],
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
