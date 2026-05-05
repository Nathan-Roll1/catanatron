"""Eggroll-style evolution over M2 action-type policy biases.

The search space is deliberately tiny: a 12-d vector added to
``policy_head.type_fc.3.bias``.  This preserves the product architecture and
runtime cost: exported candidates still use one plain M2 forward pass.

The loop uses antithetic Gaussian pairs plus batched no-search 2v2 fitness.
It is noisy by design, so keep/promote decisions should still use larger
fresh-seed paired gates outside this script.
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

from human_bot.export_nn import export
from human_bot.model import HumanBotNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TYPE_NAMES = [
    "ROLL", "END_TURN", "BUY_DEV", "PLAY_KNIGHT", "ROAD_BUILDING",
    "BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MOVE_ROBBER",
    "DISCARD_YOP_MONO", "MARITIME", "TRADE",
]
ROAD_PENALTY_X3 = np.array(
    [0.0, -0.15, 0.45, 0.0, 0.0, 0.45, 0.30, -1.20, 0.0, 0.0, 0.15, -0.15],
    dtype=np.float32,
)


def parse_bias(text: str) -> np.ndarray:
    if text == "road_penalty_x3":
        return ROAD_PENALTY_X3.copy()
    if text == "zero":
        return np.zeros(12, dtype=np.float32)
    vals = json.loads(text)
    arr = np.array(vals, dtype=np.float32)
    if arr.shape != (12,):
        raise ValueError("bias must be a JSON list of 12 numbers")
    return arr


def export_bias(base_state: dict[str, torch.Tensor], seed_checkpoint: Path,
                bias: np.ndarray, ckpt_path: Path, bin_path: Path,
                metadata: dict, quiet: bool = True) -> None:
    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    state = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    key = "policy_head.type_fc.3.bias"
    state[key] = state[key] + torch.from_numpy(bias).to(state[key].dtype)
    net.load_state_dict(state, strict=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(ckpt_path), metadata)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            export(str(ckpt_path), str(bin_path))
    else:
        export(str(ckpt_path), str(bin_path))


def run_many_eval(candidate_bins: list[Path], opponent_bin: Path, games: int,
                  workers: int, seed_base: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "human_bot.eval_2v2_many_nn_fast",
    ]
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


def run_single_eval(a_bin: Path, b_bin: Path, games: int, workers: int,
                    seed_base: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "human_bot.eval_2v2_nn_fast",
        "--a-weights", str(a_bin),
        "--b-weights", str(b_bin),
        "--games", str(games),
        "--workers", str(workers),
        "--seed-base", str(seed_base),
        "--json",
    ]
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


def clamp_bias(bias: np.ndarray, max_abs: float) -> np.ndarray:
    out = np.clip(bias, -max_abs, max_abs).astype(np.float32)
    out[0] = 0.0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", default="checkpoints/sp_latest2.pt")
    p.add_argument("--opponent-bin", default="csrc/nn_weights_m2.bin")
    p.add_argument("--out-pt", default="autoresearch-results/m2_type_bias_es.pt")
    p.add_argument("--out-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument("--work-dir", default="autoresearch-results/type_bias_es")
    p.add_argument("--initial-bias", default="road_penalty_x3")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.35)
    p.add_argument("--lr", type=float, default=0.8)
    p.add_argument("--max-abs", type=float, default=2.5)
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed-base", type=int, default=3300000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    seed_checkpoint = (PROJECT_ROOT / args.seed_checkpoint).resolve()
    opponent_bin = (PROJECT_ROOT / args.opponent_bin).resolve()
    out_pt = (PROJECT_ROOT / args.out_pt).resolve()
    out_bin = (PROJECT_ROOT / args.out_bin).resolve()
    work_dir = (PROJECT_ROOT / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    base_net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in base_net.state_dict().items()}
    mean = clamp_bias(parse_bias(args.initial_bias), args.max_abs)

    best = {
        "score": -1.0,
        "bias": mean.copy(),
        "tag": "initial",
        "wins": 0,
        "games": args.games,
    }
    history = []
    t0 = time.time()

    for gen in range(args.generations):
        gen_dir = work_dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        directions = rng.normal(size=(args.pairs, 12)).astype(np.float32)
        directions[:, 0] = 0.0

        biases = [mean.copy()]
        tags = ["mean"]
        pair_index = []
        for i, eps in enumerate(directions):
            plus = clamp_bias(mean + args.sigma * eps, args.max_abs)
            minus = clamp_bias(mean - args.sigma * eps, args.max_abs)
            pair_index.append((len(biases), len(biases) + 1))
            biases.extend([plus, minus])
            tags.extend([f"p{i}_plus", f"p{i}_minus"])

        bins = []
        for ci, (tag, bias) in enumerate(zip(tags, biases)):
            ckpt = gen_dir / f"{ci:02d}_{tag}.pt"
            bin_path = gen_dir / f"{ci:02d}_{tag}.bin"
            export_bias(
                base_state,
                seed_checkpoint,
                bias,
                ckpt,
                bin_path,
                {
                    "type_bias_es": True,
                    "generation": gen,
                    "tag": tag,
                    "bias": bias.tolist(),
                },
            )
            bins.append(bin_path)

        seed_base = args.seed_base + gen * 10000
        eval_res = run_many_eval(bins, opponent_bin, args.games, args.workers, seed_base)
        scores = np.array([r["a_winrate"] for r in eval_res["results"]], dtype=np.float32)

        update = np.zeros(12, dtype=np.float32)
        for i, (plus_idx, minus_idx) in enumerate(pair_index):
            update += (scores[plus_idx] - scores[minus_idx]) * directions[i]
        update = update / max(1e-6, 2.0 * args.pairs * args.sigma)
        mean = clamp_bias(mean + args.lr * update, args.max_abs)

        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > best["score"]:
            row = eval_res["results"][best_idx]
            best = {
                "score": float(scores[best_idx]),
                "bias": biases[best_idx].copy(),
                "tag": f"gen{gen}_{tags[best_idx]}",
                "wins": int(row["a_wins"]),
                "games": int(row["games"]),
            }

        control = run_single_eval(
            opponent_bin,
            opponent_bin,
            args.games,
            args.workers,
            seed_base,
        )
        rec = {
            "generation": gen,
            "seed_base": seed_base,
            "mean_score": float(scores[0]),
            "best_score": float(scores[best_idx]),
            "best_tag": tags[best_idx],
            "best_wins": int(eval_res["results"][best_idx]["a_wins"]),
            "control_a_winrate": float(control["a_winrate"]),
            "control_a_wins": int(control["a_wins"]),
            "mean_bias": mean.tolist(),
            "gen_best_bias": biases[best_idx].tolist(),
            "scores": [
                {"tag": tag, **row}
                for tag, row in zip(tags, eval_res["results"])
            ],
        }
        history.append(rec)
        print(json.dumps(rec, sort_keys=True), flush=True)

    export_bias(
        base_state,
        seed_checkpoint,
        best["bias"],
        out_pt,
        out_bin,
        {
            "type_bias_es": True,
            "selected": best["tag"],
            "score": best["score"],
            "wins": best["wins"],
            "games": best["games"],
            "bias": best["bias"].tolist(),
        },
        quiet=False,
    )

    result = {
        "ok": True,
        "elapsed_sec": time.time() - t0,
        "best_score": best["score"],
        "best_wins": best["wins"],
        "best_games": best["games"],
        "best_tag": best["tag"],
        "best_bias": best["bias"].tolist(),
        "out_pt": str(out_pt),
        "out_bin": str(out_bin),
        "history": history,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
