#!/usr/bin/env python3
"""Evaluate a fixed mixed policy zoo with AB2 pinned at Elo 1000.

This is deliberately a runtime-policy benchmark: NN entries are 0-ply argmax
policies, and the only searched player is the explicit AB2 anchor.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_bot.search_distill_m2 import (
    CSRC,
    DEFAULT_WORK,
    EGGROLL_DIR,
    LeagueEntry,
    _fit_bt_elo,
    _print_elo_table,
    evaluate_roster,
)


def _default_models() -> list[LeagueEntry]:
    candidates = [
        LeagueEntry("ab2", "ab2"),
        LeagueEntry("m2", "nn", str(CSRC / "nn_weights_m2.bin")),
        LeagueEntry("eg0143", "nn", str(EGGROLL_DIR / "kept_iter_0143.bin")),
        LeagueEntry("eg0150", "nn", str(EGGROLL_DIR / "kept_iter_0150.bin")),
        LeagueEntry("sd0012", "nn", str(DEFAULT_WORK / "kept_iter_0012.bin")),
        LeagueEntry("sd0029", "nn", str(DEFAULT_WORK / "kept_iter_0029.bin")),
        LeagueEntry("sd0034", "nn", str(DEFAULT_WORK / "kept_iter_0034.bin")),
    ]
    out: list[LeagueEntry] = []
    for entry in candidates:
        if entry.kind == "ab2" or (entry.bin_path and Path(entry.bin_path).exists()):
            out.append(entry)
    return out


def _parse_entry(raw: str) -> LeagueEntry:
    if raw == "ab2" or raw == "ab2=ab2":
        return LeagueEntry("ab2", "ab2")
    if "=" not in raw:
        raise ValueError(f"Expected label=/path/to/model.bin, got {raw!r}")
    label, path = raw.split("=", 1)
    return LeagueEntry(label.strip(), "nn", str(Path(path).expanduser().resolve()))


def _with_bootstrap_ci(
    eval_res: dict[str, Any],
    labels: list[str],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if n_boot <= 0:
        return eval_res
    raw = eval_res["raw"]
    if not raw:
        return eval_res
    rng = np.random.default_rng(seed)
    elo_samples: dict[str, list[float]] = {label: [] for label in labels}
    wr_samples: dict[str, list[float]] = {label: [] for label in labels}
    raw_arr = np.asarray(raw, dtype=object)
    for _ in range(n_boot):
        sample = raw_arr[rng.integers(0, len(raw_arr), size=len(raw_arr))].tolist()
        elo = _fit_bt_elo(sample, labels, pinned_label="ab2" if "ab2" in labels else labels[0])
        wins = {label: 0 for label in labels}
        played = {label: 0 for label in labels}
        for row in sample:
            for label in row["players"]:
                played[label] += 1
            if row["winner"] is not None:
                wins[row["winner"]] += 1
        for label in labels:
            elo_samples[label].append(1000.0 if label == "ab2" else float(elo[label]))
            wr_samples[label].append(wins[label] / max(1, played[label]))

    by_label = {row["label"]: row for row in eval_res["rows"]}
    for label in labels:
        row = by_label[label]
        lo, hi = np.percentile(elo_samples[label], [2.5, 97.5])
        wr_lo, wr_hi = np.percentile(wr_samples[label], [2.5, 97.5])
        row["elo_ci95"] = [float(lo), float(hi)]
        row["wr_ci95"] = [float(wr_lo), float(wr_hi)]
    return eval_res


def _print_table_with_ci(eval_res: dict[str, Any]) -> None:
    print("\nFixed 0-ply policy zoo Elo (AB2 pinned at 1000):")
    print(
        f"{'rank':>4}  {'model':<12} {'kind':<4} {'elo':>8} {'elo 95% ci':>19} "
        f"{'games':>6} {'wins':>5} {'wr':>7} {'wr 95% ci':>17}"
    )
    for i, row in enumerate(eval_res["rows"], 1):
        elo_ci = row.get("elo_ci95")
        wr_ci = row.get("wr_ci95")
        elo_ci_s = "" if not elo_ci else f"[{elo_ci[0]:.1f}, {elo_ci[1]:.1f}]"
        wr_ci_s = "" if not wr_ci else f"[{100*wr_ci[0]:.1f}, {100*wr_ci[1]:.1f}]"
        print(
            f"{i:>4}  {row['label']:<12} {row['kind']:<4} "
            f"{row['elo']:>8.1f} {elo_ci_s:>19} "
            f"{row['games']:>6} {row['wins']:>5} {100*row['wr']:>6.1f}% {wr_ci_s:>17}"
        )
    print(f"eval games={eval_res['games']} elapsed={eval_res['elapsed_sec']:.1f}s\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], help="label=weights.bin, or ab2")
    parser.add_argument("--games-per-combo", type=int, default=24)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=970000000)
    parser.add_argument("--ab-depth", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=1234)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    roster = [_parse_entry(raw) for raw in args.model] if args.model else _default_models()
    labels = [entry.label for entry in roster]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate labels in roster: {labels}")
    if "ab2" not in labels:
        raise ValueError("Roster must include ab2 so the Elo anchor is fixed at 1000")
    missing = [entry.bin_path for entry in roster if entry.kind == "nn" and not Path(entry.bin_path or "").exists()]
    if missing:
        raise FileNotFoundError(f"Missing NN weights: {missing}")

    print("Evaluating roster:")
    for entry in roster:
        print(f"  {entry.label:<12} {entry.kind:<3} {entry.bin_path or ''}")

    eval_res = evaluate_roster(
        roster,
        games_per_combo=args.games_per_combo,
        workers=args.workers,
        seed_base=args.seed_base,
        ab_depth=args.ab_depth,
        hs_depth=0,
        hs_time_ms=0.0,
    )
    eval_res = _with_bootstrap_ci(
        eval_res,
        labels,
        n_boot=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    if args.bootstrap > 0:
        _print_table_with_ci(eval_res)
    else:
        _print_elo_table(eval_res)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "roster": [asdict(entry) for entry in roster],
                "games_per_combo": args.games_per_combo,
                "seed_base": args.seed_base,
                "ab_depth": args.ab_depth,
                "bootstrap": args.bootstrap,
                "rows": eval_res["rows"],
                "games": eval_res["games"],
                "elapsed_sec": eval_res["elapsed_sec"],
            }, f, indent=2)


if __name__ == "__main__":
    main()
