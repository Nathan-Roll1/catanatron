"""Benchmark worker counts for many-candidate M2 game evaluation.

This must be run as a module/file, not through ``python - <<'PY'``.  macOS uses
spawn multiprocessing, and spawn workers need a real importable main file.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from human_bot.eval_1v3_many_nn_fast import run as run_ffa
from human_bot.eval_2v2_many_nn_paired_fast import run as run_paired2v2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True, help="candidate weight glob")
    p.add_argument("--opponent-bin", default="csrc/nn_weights_m2.bin")
    p.add_argument("--mode", choices=("paired2v2", "ffa"), default="paired2v2")
    p.add_argument("--games", type=int, default=8)
    p.add_argument("--workers", default="1,2,4,8,12,18")
    p.add_argument("--seed-base", type=int, default=7_100_000)
    args = p.parse_args()

    bins = [
        str(Path(path).resolve())
        for path in sorted(glob.glob(args.glob))
        if not path.endswith("_test.bin")
    ]
    if not bins:
        raise RuntimeError(f"no candidates matched {args.glob!r}")
    worker_counts = [int(x) for x in args.workers.split(",") if x.strip()]
    runner = run_paired2v2 if args.mode == "paired2v2" else run_ffa

    for idx, workers in enumerate(worker_counts):
        t0 = time.time()
        res = runner(
            bins,
            str(Path(args.opponent_bin).resolve()),
            args.games,
            workers,
            args.seed_base + idx * 10_000,
        )
        games = int(res["candidates"] * res["games_per_candidate"])
        elapsed = float(res["elapsed_sec"])
        print(json.dumps({
            "mode": args.mode,
            "workers": workers,
            "candidates": int(res["candidates"]),
            "games_per_candidate": int(res["games_per_candidate"]),
            "total_games": games,
            "elapsed_sec": elapsed,
            "wall_sec": time.time() - t0,
            "games_per_sec": games / max(elapsed, 1e-9),
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
