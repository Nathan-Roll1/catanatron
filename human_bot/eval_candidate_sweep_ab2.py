#!/usr/bin/env python3
"""Sweep many 0-ply candidate binaries against a fixed AB2-pinned zoo."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_bot.search_distill_m2 import (  # noqa: E402
    CSRC,
    DEFAULT_WORK,
    EGGROLL_DIR,
    LeagueEntry,
    evaluate_roster,
)


def _fixed_roster(candidate_bin: Path) -> list[LeagueEntry]:
    return [
        LeagueEntry("ab2", "ab2"),
        LeagueEntry("m2", "nn", str((CSRC / "nn_weights_m2.bin").resolve())),
        LeagueEntry("eg0143", "nn", str((EGGROLL_DIR / "kept_iter_0143.bin").resolve())),
        LeagueEntry("eg0150", "nn", str((EGGROLL_DIR / "kept_iter_0150.bin").resolve())),
        LeagueEntry("sd0029", "nn", str((DEFAULT_WORK / "kept_iter_0029.bin").resolve())),
        LeagueEntry("sd0034", "nn", str((DEFAULT_WORK / "kept_iter_0034.bin").resolve())),
        LeagueEntry("sd0054", "nn", str((DEFAULT_WORK / "kept_iter_0054.bin").resolve())),
        LeagueEntry("candidate", "nn", str(candidate_bin.resolve())),
    ]


def _best_other(eval_res: dict[str, Any]) -> dict[str, Any]:
    return max((row for row in eval_res["rows"] if row["label"] != "candidate"), key=lambda row: float(row["elo"]))


def _candidate_row(eval_res: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in eval_res["rows"] if row["label"] == "candidate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--games-per-combo", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=973000000)
    parser.add_argument("--ab-depth", type=int, default=2)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    summaries = []
    for idx, item in enumerate(manifest):
        name = item["name"]
        candidate_bin = Path(item["bin"])
        print(f"\n=== candidate sweep {idx + 1}/{len(manifest)}: {name} ===", flush=True)
        roster = _fixed_roster(candidate_bin)
        eval_res = evaluate_roster(
            roster,
            games_per_combo=args.games_per_combo,
            workers=args.workers,
            seed_base=args.seed_base + idx * 100_000,
            ab_depth=args.ab_depth,
            hs_depth=0,
            hs_time_ms=0.0,
        )
        cand = _candidate_row(eval_res)
        best = _best_other(eval_res)
        delta = float(cand["elo"]) - float(best["elo"])
        print(
            f"{name}: candidate={float(cand['elo']):.1f} "
            f"best_other={best['label']} {float(best['elo']):.1f} "
            f"delta={delta:+.1f} wr={100*float(cand['wr']):.1f}%",
            flush=True,
        )
        summaries.append({
            "name": name,
            "candidate_bin": str(candidate_bin),
            "candidate_pt": item.get("pt"),
            "candidate_elo": float(cand["elo"]),
            "candidate_wr": float(cand["wr"]),
            "best_other_label": best["label"],
            "best_other_elo": float(best["elo"]),
            "delta_elo": delta,
            "games": eval_res["games"],
            "elapsed_sec": eval_res["elapsed_sec"],
            "roster": [asdict(e) for e in roster],
            "rows": eval_res["rows"],
        })

    summaries.sort(key=lambda row: row["delta_elo"], reverse=True)
    print("\nSweep ranking:")
    print(f"{'rank':>4} {'name':<32} {'cand':>8} {'best':<10} {'best elo':>8} {'delta':>8} {'wr':>7}")
    for rank, row in enumerate(summaries, 1):
        print(
            f"{rank:>4} {row['name']:<32} {row['candidate_elo']:>8.1f} "
            f"{row['best_other_label']:<10} {row['best_other_elo']:>8.1f} "
            f"{row['delta_elo']:>+8.1f} {100*row['candidate_wr']:>6.1f}%"
        )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
