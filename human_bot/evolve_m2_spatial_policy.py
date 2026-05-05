"""Eggroll-style evolution over compact spatial policy calibration weights.

This keeps runtime identical to base M2 inference: candidates are ordinary
HumanBotNet checkpoints exported to the same C binary format.  The evolved
parameters are intentionally near the policy surface:

  - policy_head.type_fc.3.bias            (12)
  - policy_head.settlement_scorer.2.weight (48)
  - policy_head.city_scorer.2.weight       (48)
  - policy_head.road_scorer.2.weight       (48)

The scorer vectors affect which settlement/city/road action is preferred
within a chosen action type, so this is a better target for road-quality
failures than leaf tweaks or more global action bias.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
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


PARAM_SPECS = [
    ("policy_head.type_fc.3.bias", (12,), "type"),
    ("policy_head.settlement_scorer.2.weight", (1, 48), "scorer"),
    ("policy_head.city_scorer.2.weight", (1, 48), "scorer"),
    ("policy_head.road_scorer.2.weight", (1, 48), "scorer"),
]


def _dim() -> int:
    return sum(int(np.prod(shape)) for _key, shape, _group in PARAM_SPECS)


def _base_delta(initial_type_bias: str) -> np.ndarray:
    out = np.zeros(_dim(), dtype=np.float32)
    if initial_type_bias == "road_penalty_x3":
        out[:12] = ROAD_PENALTY_X3
    elif initial_type_bias != "zero":
        vals = np.array(json.loads(initial_type_bias), dtype=np.float32)
        if vals.shape != (12,):
            raise ValueError("initial_type_bias must be zero, road_penalty_x3, or a JSON 12-list")
        out[:12] = vals
    return out


def _scale_vec(type_scale: float, scorer_scale: float) -> np.ndarray:
    vals: list[float] = []
    for _key, shape, group in PARAM_SPECS:
        n = int(np.prod(shape))
        vals.extend([type_scale if group == "type" else scorer_scale] * n)
    return np.array(vals, dtype=np.float32)


def _max_abs_vec(type_max_abs: float, scorer_max_abs: float) -> np.ndarray:
    vals: list[float] = []
    for _key, shape, group in PARAM_SPECS:
        n = int(np.prod(shape))
        vals.extend([type_max_abs if group == "type" else scorer_max_abs] * n)
    out = np.array(vals, dtype=np.float32)
    out[0] = 0.0
    return out


def _clamp_actual(vec: np.ndarray, max_abs: np.ndarray) -> np.ndarray:
    out = np.clip(vec, -max_abs, max_abs).astype(np.float32)
    out[0] = 0.0
    return out


def _iter_slices(vec: np.ndarray):
    offset = 0
    for key, shape, _group in PARAM_SPECS:
        n = int(np.prod(shape))
        yield key, shape, vec[offset:offset + n].reshape(shape)
        offset += n


def export_spatial_delta(
    base_state: dict[str, torch.Tensor],
    seed_checkpoint: Path,
    actual_delta: np.ndarray,
    ckpt_path: Path,
    bin_path: Path,
    metadata: dict,
    quiet: bool = True,
) -> None:
    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    state = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    for key, _shape, arr in _iter_slices(actual_delta):
        state[key] = state[key] + torch.from_numpy(arr).to(state[key].dtype)
    net.load_state_dict(state, strict=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(ckpt_path), metadata)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            export(str(ckpt_path), str(bin_path))
    else:
        export(str(ckpt_path), str(bin_path))


def _dedupe_top(records: list[dict], limit: int) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[float, ...]] = set()
    for rec in sorted(records, key=lambda r: r["proxy_score"], reverse=True):
        key = tuple(np.round(rec["actual_delta"], 5).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def run_many_eval_paired(candidate_bins: list[Path], opponent_bin: Path,
                         games: int, workers: int, seed_base: int) -> dict:
    cmd = [sys.executable, "-m", "human_bot.eval_2v2_many_nn_paired_fast"]
    for path in candidate_bins:
        cmd.extend(["--a-weight", str(path)])
    cmd.extend([
        "--b-weights", str(opponent_bin),
        "--games", str(games),
        "--workers", str(workers),
        "--seed-base", str(seed_base),
        "--json",
    ])
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_many_eval_ffa(candidate_bins: list[Path], opponent_bin: Path,
                      games: int, workers: int, seed_base: int) -> dict:
    cmd = [sys.executable, "-m", "human_bot.eval_1v3_many_nn_fast"]
    for path in candidate_bins:
        cmd.extend(["--a-weight", str(path)])
    cmd.extend([
        "--b-weights", str(opponent_bin),
        "--games", str(games),
        "--workers", str(workers),
        "--seed-base", str(seed_base),
        "--json",
    ])
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout[-4000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def control_for_mode(mode: str, games: int, opponent_bin: Path,
                     workers: int, seed_base: int) -> dict:
    if mode == "ffa":
        return {"a_winrate": 0.25, "a_wins": games, "games": 4 * games}
    return run_single_eval(opponent_bin, opponent_bin, games, workers, seed_base)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", default="checkpoints/sp_latest2.pt")
    p.add_argument("--opponent-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument("--out-pt", default="autoresearch-results/m2_spatial_policy_es.pt")
    p.add_argument("--out-bin", default="/tmp/catan_m2_spatial_policy_es.bin")
    p.add_argument("--work-dir", default="autoresearch-results/spatial_policy_es")
    p.add_argument("--initial-type-bias", default="road_penalty_x3")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--pairs", type=int, default=20)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--promote-top", type=int, default=6)
    p.add_argument("--promote-games", type=int, default=96)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--paired-proxy", action="store_true")
    p.add_argument("--ffa-proxy", action="store_true")
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=0.35)
    p.add_argument("--type-scale", type=float, default=0.35)
    p.add_argument("--scorer-scale", type=float, default=0.10)
    p.add_argument("--type-max-abs", type=float, default=3.0)
    p.add_argument("--scorer-max-abs", type=float, default=0.65)
    p.add_argument("--seed-base", type=int, default=3500000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    seed_checkpoint = (PROJECT_ROOT / args.seed_checkpoint).resolve()
    opponent_bin = (PROJECT_ROOT / args.opponent_bin).resolve()
    out_pt = (PROJECT_ROOT / args.out_pt).resolve() if not Path(args.out_pt).is_absolute() else Path(args.out_pt)
    out_bin = (PROJECT_ROOT / args.out_bin).resolve() if not Path(args.out_bin).is_absolute() else Path(args.out_bin)
    work_dir = (PROJECT_ROOT / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.ffa_proxy:
        eval_many = run_many_eval_ffa
        eval_mode = "ffa"
    elif args.paired_proxy:
        eval_many = run_many_eval_paired
        eval_mode = "paired2v2"
    else:
        eval_many = run_many_eval
        eval_mode = "2v2"

    base_net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in base_net.state_dict().items()}
    base_delta = _base_delta(args.initial_type_bias)
    scale = _scale_vec(args.type_scale, args.scorer_scale)
    max_abs = _max_abs_vec(args.type_max_abs, args.scorer_max_abs)
    mean_z = np.zeros(_dim(), dtype=np.float32)
    all_records: list[dict] = []
    history: list[dict] = []
    t0 = time.time()

    for gen in range(args.generations):
        gen_dir = work_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        directions = rng.normal(size=(args.pairs, _dim())).astype(np.float32)
        directions[:, 0] = 0.0

        zs = [mean_z.copy()]
        tags = ["mean"]
        pair_index = []
        for i, eps in enumerate(directions):
            pair_index.append((len(zs), len(zs) + 1))
            zs.extend([mean_z + args.sigma * eps, mean_z - args.sigma * eps])
            tags.extend([f"p{i}_plus", f"p{i}_minus"])

        bins = []
        actuals = []
        for idx, (tag, z) in enumerate(zip(tags, zs)):
            actual = _clamp_actual(base_delta + scale * z, max_abs)
            actuals.append(actual)
            ckpt = gen_dir / f"{idx:02d}_{tag}.pt"
            bin_path = gen_dir / f"{idx:02d}_{tag}.bin"
            export_spatial_delta(
                base_state,
                seed_checkpoint,
                actual,
                ckpt,
                bin_path,
                {
                    "spatial_policy_es": True,
                    "stage": "proxy",
                    "generation": gen,
                    "tag": tag,
                    "actual_delta": actual.tolist(),
                },
            )
            bins.append(bin_path)

        seed_base = args.seed_base + gen * 10000
        eval_res = eval_many(bins, opponent_bin, args.games, args.workers, seed_base)
        scores = np.array([r["a_winrate"] for r in eval_res["results"]], dtype=np.float32)

        update = np.zeros(_dim(), dtype=np.float32)
        for i, (plus_idx, minus_idx) in enumerate(pair_index):
            update += (scores[plus_idx] - scores[minus_idx]) * directions[i]
        update /= max(1e-6, 2.0 * args.pairs * args.sigma)
        mean_z = mean_z + args.lr * update
        mean_actual = _clamp_actual(base_delta + scale * mean_z, max_abs)
        mean_z = (mean_actual - base_delta) / np.maximum(scale, 1e-6)
        mean_z[0] = 0.0

        best_idx = int(np.argmax(scores))
        control = control_for_mode(eval_mode, args.games, opponent_bin,
                                   args.workers, seed_base)
        for tag, row, actual, bin_path in zip(tags, eval_res["results"], actuals, bins):
            all_records.append({
                "generation": gen,
                "tag": tag,
                "proxy_score": float(row["a_winrate"]),
                "proxy_wins": int(row["a_wins"]),
                "proxy_games": int(row["games"]),
                "actual_delta": actual,
                "bin": str(bin_path),
            })
        rec = {
            "generation": gen,
            "seed_base": seed_base,
            "mean_score": float(scores[0]),
            "best_score": float(scores[best_idx]),
            "best_tag": tags[best_idx],
            "best_wins": int(eval_res["results"][best_idx]["a_wins"]),
            "control_a_winrate": float(control["a_winrate"]),
            "control_a_wins": int(control["a_wins"]),
        }
        history.append(rec)
        print(json.dumps(rec, sort_keys=True), flush=True)

    promote = _dedupe_top(all_records, args.promote_top)
    promote_dir = work_dir / "promote"
    promote_bins = []
    for idx, rec in enumerate(promote):
        ckpt = promote_dir / f"{idx:02d}_g{rec['generation']}_{rec['tag']}.pt"
        bin_path = promote_dir / f"{idx:02d}_g{rec['generation']}_{rec['tag']}.bin"
        export_spatial_delta(
            base_state,
            seed_checkpoint,
            rec["actual_delta"],
            ckpt,
            bin_path,
            {
                "spatial_policy_es": True,
                "stage": "promote",
                "source_generation": rec["generation"],
                "source_tag": rec["tag"],
                "proxy_score": rec["proxy_score"],
                "actual_delta": rec["actual_delta"].tolist(),
            },
        )
        promote_bins.append(bin_path)

    promote_seed_base = args.seed_base + args.generations * 10000 + 5000
    promote_res = eval_many(promote_bins, opponent_bin, args.promote_games,
                            args.workers, promote_seed_base)
    promote_scores = np.array([r["a_winrate"] for r in promote_res["results"]], dtype=np.float32)
    best_promote_idx = int(np.argmax(promote_scores))
    best_rec = promote[best_promote_idx]
    control = control_for_mode(eval_mode, args.promote_games, opponent_bin,
                               args.workers, promote_seed_base)

    export_spatial_delta(
        base_state,
        seed_checkpoint,
        best_rec["actual_delta"],
        out_pt,
        out_bin,
        {
            "spatial_policy_es": True,
            "selected": f"g{best_rec['generation']}_{best_rec['tag']}",
            "proxy_score": best_rec["proxy_score"],
            "promote_score": float(promote_scores[best_promote_idx]),
            "promote_wins": int(promote_res["results"][best_promote_idx]["a_wins"]),
            "promote_games": args.promote_games,
            "actual_delta": best_rec["actual_delta"].tolist(),
        },
        quiet=False,
    )

    result = {
        "ok": True,
        "elapsed_sec": time.time() - t0,
        "dims": _dim(),
        "population_per_generation": 1 + 2 * args.pairs,
        "generations": args.generations,
        "paired_proxy": bool(args.paired_proxy),
        "ffa_proxy": bool(args.ffa_proxy),
        "eval_mode": eval_mode,
        "initial_type_bias": args.initial_type_bias,
        "promote_top": len(promote),
        "best_tag": f"g{best_rec['generation']}_{best_rec['tag']}",
        "best_proxy_score": best_rec["proxy_score"],
        "best_promote_score": float(promote_scores[best_promote_idx]),
        "best_promote_wins": int(promote_res["results"][best_promote_idx]["a_wins"]),
        "promote_games": args.promote_games,
        "control_a_winrate": float(control["a_winrate"]),
        "control_a_wins": int(control["a_wins"]),
        "out_pt": str(out_pt),
        "out_bin": str(out_bin),
        "history": history,
        "promote_scores": [
            {
                "source_tag": f"g{rec['generation']}_{rec['tag']}",
                "proxy_score": rec["proxy_score"],
                **row,
            }
            for rec, row in zip(promote, promote_res["results"])
        ],
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
