"""Paired beam/coordinate evolution for M2 action-type calibration.

This is deliberately lower-dimensional than the broad ES attempts.  The only
retained no-search gain so far came from action-type bias calibration, so this
script searches that surface with paired candidate-as-A/candidate-as-B fitness
instead of noisy single-side proxies.
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

from human_bot.evolve_m2_spatial_policy import run_many_eval_paired
from human_bot.evolve_m2_type_bias import (
    PROJECT_ROOT,
    ROAD_PENALTY_X3,
    TYPE_NAMES,
    export,
    export_bias,
    parse_bias,
    run_single_eval,
)
from human_bot.model import HumanBotNet


TUNED_TYPE_INDEXES = [1, 2, 5, 6, 7, 10, 11]


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


def control_for_metric(metric: str, games: int, opponent_bin: Path,
                       workers: int, seed_base: int) -> dict:
    if metric == "ffa":
        return {
            "a_winrate": 0.25,
            "a_wins": games,
            "games": 4 * games,
        }
    return run_single_eval(opponent_bin, opponent_bin, games, workers, seed_base)


def clamp_bias(bias: np.ndarray, max_abs: float) -> np.ndarray:
    out = np.clip(bias, -max_abs, max_abs).astype(np.float32)
    out[0] = 0.0
    return out


def unique_biases(candidates: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for tag, bias in candidates:
        key = tuple(np.round(bias, 5).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append((tag, bias))
    return out


def export_many(base_state, seed_checkpoint: Path, records, out_dir: Path):
    bins = []
    for idx, rec in enumerate(records):
        tag = rec["tag"]
        bias = rec["bias"]
        ckpt = out_dir / f"{idx:02d}_{tag}.pt"
        bin_path = out_dir / f"{idx:02d}_{tag}.bin"
        export_bias(
            base_state,
            seed_checkpoint,
            bias,
            ckpt,
            bin_path,
            {
                "type_bias_beam": True,
                "tag": tag,
                "bias": bias.tolist(),
            },
        )
        bins.append(bin_path)
    return bins


def candidate_neighborhood(center: np.ndarray, step: float, max_abs: float) -> list[tuple[str, np.ndarray]]:
    candidates = [("center", center.copy())]
    for idx in TUNED_TYPE_INDEXES:
        for sign, name in ((1.0, "up"), (-1.0, "down")):
            b = center.copy()
            b[idx] += sign * step
            candidates.append((f"{TYPE_NAMES[idx]}_{name}_{step:.3f}", clamp_bias(b, max_abs)))
    # A couple of coupled moves that reflect the retained hand-shaped prior.
    coupled = [
        ("more_tall_less_road", {2: step, 5: 0.5 * step, 6: 0.5 * step, 7: -step}),
        ("more_settle_less_dev", {5: step, 2: -0.5 * step, 7: -0.5 * step}),
        ("less_end_more_build", {1: -step, 2: 0.5 * step, 5: 0.5 * step, 6: 0.5 * step}),
        ("less_trade_more_port", {11: -step, 10: 0.5 * step}),
    ]
    for tag, delta in coupled:
        b = center.copy()
        for idx, val in delta.items():
            b[idx] += val
        candidates.append((f"{tag}_{step:.3f}", clamp_bias(b, max_abs)))
    return unique_biases(candidates)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", default="checkpoints/sp_latest2.pt")
    p.add_argument("--opponent-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument("--out-pt", default="autoresearch-results/m2_type_bias_beam.pt")
    p.add_argument("--out-bin", default="/tmp/catan_m2_type_bias_beam.bin")
    p.add_argument("--work-dir", default="autoresearch-results/type_bias_beam")
    p.add_argument("--initial-bias", default="road_penalty_x3")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--promote-top", type=int, default=8)
    p.add_argument("--promote-games", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--step", type=float, default=0.20)
    p.add_argument("--decay", type=float, default=0.6)
    p.add_argument("--max-abs", type=float, default=3.0)
    p.add_argument("--seed-base", type=int, default=3700000)
    p.add_argument("--metric", choices=("2v2", "ffa"), default="2v2")
    args = p.parse_args()

    seed_checkpoint = (PROJECT_ROOT / args.seed_checkpoint).resolve()
    opponent_bin = (PROJECT_ROOT / args.opponent_bin).resolve()
    out_pt = (PROJECT_ROOT / args.out_pt).resolve() if not Path(args.out_pt).is_absolute() else Path(args.out_pt)
    out_bin = (PROJECT_ROOT / args.out_bin).resolve() if not Path(args.out_bin).is_absolute() else Path(args.out_bin)
    work_dir = (PROJECT_ROOT / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    eval_many = run_many_eval_ffa if args.metric == "ffa" else run_many_eval_paired

    base_net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in base_net.state_dict().items()}
    center = clamp_bias(parse_bias(args.initial_bias), args.max_abs)
    step = args.step
    all_records: list[dict] = []
    history: list[dict] = []
    t0 = time.time()

    for round_idx in range(args.rounds):
        round_dir = work_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidates = candidate_neighborhood(center, step, args.max_abs)
        records = [
            {"tag": tag.replace(".", "p").replace("-", "m"), "bias": bias}
            for tag, bias in candidates
        ]
        bins = export_many(base_state, seed_checkpoint, records, round_dir)
        seed_base = args.seed_base + round_idx * 10000
        eval_res = eval_many(bins, opponent_bin, args.games, args.workers, seed_base)
        scores = np.array([r["a_winrate"] for r in eval_res["results"]], dtype=np.float32)
        best_idx = int(np.argmax(scores))
        center_idx = next(i for i, rec in enumerate(records) if rec["tag"] == "center")
        if scores[best_idx] >= scores[center_idx]:
            center = records[best_idx]["bias"].copy()
        step *= args.decay

        control = control_for_metric(args.metric, args.games, opponent_bin,
                                     args.workers, seed_base)
        for rec, row in zip(records, eval_res["results"]):
            all_records.append({
                "round": round_idx,
                "tag": rec["tag"],
                "bias": rec["bias"],
                "score": float(row["a_winrate"]),
                "wins": int(row["a_wins"]),
                "games": int(row["games"]),
            })
        h = {
            "round": round_idx,
            "seed_base": seed_base,
            "step": step / args.decay,
            "center_score": float(scores[center_idx]),
            "best_score": float(scores[best_idx]),
            "best_tag": records[best_idx]["tag"],
            "best_wins": int(eval_res["results"][best_idx]["a_wins"]),
            "games": int(eval_res["results"][best_idx]["games"]),
            "control_a_winrate": float(control["a_winrate"]),
            "control_a_wins": int(control["a_wins"]),
            "center_bias": center.tolist(),
        }
        history.append(h)
        print(json.dumps(h, sort_keys=True), flush=True)

    promote_records = []
    seen: set[tuple[float, ...]] = set()
    for rec in sorted(all_records, key=lambda r: r["score"], reverse=True):
        key = tuple(np.round(rec["bias"], 5).tolist())
        if key in seen:
            continue
        seen.add(key)
        promote_records.append(rec)
        if len(promote_records) >= args.promote_top:
            break

    promote_dir = work_dir / "promote"
    promote_export_records = [
        {
            "tag": f"r{rec['round']}_{rec['tag']}",
            "bias": rec["bias"],
        }
        for rec in promote_records
    ]
    promote_bins = export_many(base_state, seed_checkpoint,
                               promote_export_records, promote_dir)
    promote_seed_base = args.seed_base + args.rounds * 10000 + 5000
    promote_res = eval_many(promote_bins, opponent_bin, args.promote_games,
                            args.workers, promote_seed_base)
    promote_scores = np.array([r["a_winrate"] for r in promote_res["results"]], dtype=np.float32)
    best_idx = int(np.argmax(promote_scores))
    best = promote_records[best_idx]
    control = control_for_metric(args.metric, args.promote_games, opponent_bin,
                                 args.workers, promote_seed_base)

    export_bias(
        base_state,
        seed_checkpoint,
        best["bias"],
        out_pt,
        out_bin,
        {
            "type_bias_beam": True,
            "selected": f"r{best['round']}_{best['tag']}",
            "proxy_score": best["score"],
            "promote_score": float(promote_scores[best_idx]),
            "promote_wins": int(promote_res["results"][best_idx]["a_wins"]),
            "promote_games": int(promote_res["results"][best_idx]["games"]),
            "bias": best["bias"].tolist(),
        },
        quiet=False,
    )

    result = {
        "ok": True,
        "elapsed_sec": time.time() - t0,
        "rounds": args.rounds,
        "metric": args.metric,
        "candidates_per_round": len(candidate_neighborhood(center, step, args.max_abs)),
        "promote_top": len(promote_records),
        "best_tag": f"r{best['round']}_{best['tag']}",
        "best_proxy_score": best["score"],
        "best_promote_score": float(promote_scores[best_idx]),
        "best_promote_wins": int(promote_res["results"][best_idx]["a_wins"]),
        "promote_games": int(promote_res["results"][best_idx]["games"]),
        "control_a_winrate": float(control["a_winrate"]),
        "control_a_wins": int(control["a_wins"]),
        "out_pt": str(out_pt),
        "out_bin": str(out_bin),
        "best_bias": best["bias"].tolist(),
        "history": history,
        "promote_scores": [
            {
                "source_tag": f"r{rec['round']}_{rec['tag']}",
                "proxy_score": rec["score"],
                **row,
            }
            for rec, row in zip(promote_records, promote_res["results"])
        ],
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
