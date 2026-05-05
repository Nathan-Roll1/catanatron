#!/usr/bin/env python3
"""Foreground autoresearch loop for pure 0-ply M2 policy improvement.

Runtime benchmark rule: all NN policies are single-forward 0-ply argmax.  AB2
is the only searched player and is pinned at Elo 1000.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
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
    build_training_league,
    evaluate_roster,
    generate_winner_data,
    train_bc_candidate,
)


@dataclass(frozen=True)
class TrainConfig:
    name: str
    train_scope: str
    lr: float
    epochs: int
    batch_size: int
    max_examples: int = 0
    label_smoothing: float = 0.02
    action_weights: bool = True
    teacher_balanced: bool = False
    teacher_cap_ratio: float = 1.0
    strategic_only: bool = False
    opening_turns: int = 0
    top_teachers: int = 0
    search_only: bool = False
    teacher_allowlist: str = ""
    kl_alpha: float = 1.0
    disagreement_boost: float = 0.0


def _schedule(name: str) -> list[TrainConfig]:
    """Conservative BC probes from broad to surgical.

    The model is already strong, so most candidates use KL anchoring and tiny
    learning rates.  The schedule cycles; kept candidates become the next
    incumbent and continue from the following probe.
    """
    if name == "trunk":
        return [
            TrainConfig("elite_trunk_lr2e7_kl10", "policy_trunk", 2e-7, 1, 4096, max_examples=160000, kl_alpha=10.0),
            TrainConfig("elite_trunk_lr1e7_kl12", "policy_trunk", 1e-7, 1, 4096, max_examples=160000, kl_alpha=12.0),
            TrainConfig("elite_trunk_lr3e7_kl12", "policy_trunk", 3e-7, 1, 4096, max_examples=180000, kl_alpha=12.0),
            TrainConfig("elite_head_after_trunk_lr5e7_kl8", "policy_head", 5e-7, 1, 8192, kl_alpha=8.0),
            TrainConfig("elite_type_lr2e6_kl6", "policy_type", 2e-6, 2, 8192, kl_alpha=6.0),
        ]
    if name == "topnn":
        allow = "incumbent,eg0150,sd0054,sd0034,sd0029,eg0143"
        return [
            TrainConfig("topnn_trunk_lr2e7_kl10", "policy_trunk", 2e-7, 1, 4096, max_examples=160000, kl_alpha=10.0, teacher_allowlist=allow),
            TrainConfig("topnn_trunk_lr1e7_kl12", "policy_trunk", 1e-7, 1, 4096, max_examples=160000, kl_alpha=12.0, teacher_allowlist=allow),
            TrainConfig("topnn_head_lr5e7_kl8", "policy_head", 5e-7, 1, 8192, kl_alpha=8.0, teacher_allowlist=allow),
            TrainConfig("topnn_type_lr2e6_kl6", "policy_type", 2e-6, 2, 8192, kl_alpha=6.0, teacher_allowlist=allow),
        ]
    return [
        TrainConfig("winner_head_lr1e6_kl3", "policy_head", 1e-6, 1, 8192, kl_alpha=3.0),
        TrainConfig("winner_head_lr7e7_kl5", "policy_head", 7e-7, 1, 8192, kl_alpha=5.0),
        TrainConfig("winner_head_lr1e6_kl5_smooth01", "policy_head", 1e-6, 1, 8192, label_smoothing=0.01, kl_alpha=5.0),
        TrainConfig("winner_type_lr4e6_kl3", "policy_type", 4e-6, 2, 8192, kl_alpha=3.0),
        TrainConfig("winner_open_spatial_lr4e6_kl1", "opening_spatial", 4e-6, 2, 8192, opening_turns=80, kl_alpha=1.0),
        TrainConfig("winner_trunk_lr3e7_kl8", "policy_trunk", 3e-7, 1, 4096, max_examples=120000, kl_alpha=8.0),
    ]


def _elite_training_league(incumbent_bin: Path, incumbent_pt: Path) -> list[LeagueEntry]:
    """Six-slot stronger teacher pool for winner-as-teacher data."""
    anchors = [
        ("eg0150", EGGROLL_DIR / "kept_iter_0150.bin", EGGROLL_DIR / "kept_iter_0150.pt"),
        ("sd0054", DEFAULT_WORK / "kept_iter_0054.bin", DEFAULT_WORK / "kept_iter_0054.pt"),
        ("sd0034", DEFAULT_WORK / "kept_iter_0034.bin", DEFAULT_WORK / "kept_iter_0034.pt"),
        ("sd0029", DEFAULT_WORK / "kept_iter_0029.bin", DEFAULT_WORK / "kept_iter_0029.pt"),
        ("eg0143", EGGROLL_DIR / "kept_iter_0143.bin", EGGROLL_DIR / "kept_iter_0143.pt"),
    ]
    incumbent_resolved = incumbent_bin.resolve()
    league = [
        LeagueEntry("m2", "nn", str((CSRC / "nn_weights_m2.bin").resolve())),
        LeagueEntry("ab2", "ab2"),
    ]
    for label, bin_path, pt_path in anchors:
        if len(league) >= 5:
            break
        if not bin_path.exists() or bin_path.resolve() == incumbent_resolved:
            continue
        league.append(LeagueEntry(label, "nn", str(bin_path.resolve()), str(pt_path.resolve()) if pt_path.exists() else None))
    league.append(LeagueEntry("incumbent", "nn", str(incumbent_bin.resolve()), str(incumbent_pt.resolve())))
    return league


def _nn_entry(label: str, path: Path) -> LeagueEntry | None:
    return LeagueEntry(label, "nn", str(path.resolve())) if path.exists() else None


def _fixed_eval_roster(
    incumbent_label: str,
    incumbent_bin: Path,
    candidate_bin: Path | None = None,
    prior_incumbents: list[tuple[str, Path]] | None = None,
) -> list[LeagueEntry]:
    entries: list[LeagueEntry] = [
        LeagueEntry("ab2", "ab2"),
        LeagueEntry("m2", "nn", str((CSRC / "nn_weights_m2.bin").resolve())),
    ]
    anchors = [
        ("eg0143", EGGROLL_DIR / "kept_iter_0143.bin"),
        ("eg0150", EGGROLL_DIR / "kept_iter_0150.bin"),
        ("sd0034", DEFAULT_WORK / "kept_iter_0034.bin"),
        ("sd0029", DEFAULT_WORK / "kept_iter_0029.bin"),
        ("sd0012", DEFAULT_WORK / "kept_iter_0012.bin"),
    ]
    if prior_incumbents:
        anchors.extend(prior_incumbents[-3:])

    seen = {e.label for e in entries}
    for label, path in anchors:
        if label == incumbent_label or label in seen:
            continue
        entry = _nn_entry(label, path)
        if entry is not None:
            entries.append(entry)
            seen.add(label)
        # Keep evaluation compact: AB2, M2, three fixed zoo anchors, incumbent,
        # and candidate.  This matches the stable 7-policy re-anchor tables and
        # halves quick-gate cost versus an 8-policy roster.
        if len(entries) >= 5:
            break

    entries.append(LeagueEntry("incumbent", "nn", str(incumbent_bin.resolve())))
    if candidate_bin is not None:
        entries.append(LeagueEntry("candidate", "nn", str(candidate_bin.resolve())))
    return entries


def _row_by_label(eval_res: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in eval_res["rows"]}


def _best_non_candidate(eval_res: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in eval_res["rows"] if row["label"] != "candidate"]
    return max(rows, key=lambda row: float(row["elo"]))


def _print_eval(eval_res: dict[str, Any], title: str) -> None:
    print(f"\n{title} (AB2 pinned at 1000):")
    print(f"{'rank':>4}  {'model':<12} {'kind':<4} {'elo':>8} {'games':>6} {'wins':>5} {'wr':>7}")
    for rank, row in enumerate(eval_res["rows"], 1):
        print(
            f"{rank:>4}  {row['label']:<12} {row['kind']:<4} "
            f"{row['elo']:>8.1f} {row['games']:>6} {row['wins']:>5} {100*row['wr']:>6.1f}%"
        )
    print(f"eval games={eval_res['games']} elapsed={eval_res['elapsed_sec']:.1f}s", flush=True)


def _json_safe_eval(eval_res: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in eval_res.items() if key != "raw"}


def run_loop(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "zero_ply_autoresearch_log.jsonl"

    incumbent_bin = Path(args.incumbent_bin).resolve()
    incumbent_pt = Path(args.incumbent_pt).resolve()
    incumbent_label = args.incumbent_label
    prior_incumbents: list[tuple[str, Path]] = []
    configs = _schedule(args.schedule)

    for iteration in range(args.start_iteration, args.start_iteration + args.max_iterations):
        config = configs[(iteration - args.start_iteration) % len(configs)]
        iter_dir = work_dir / f"iter_{iteration:04d}"
        shard_dir = iter_dir / "shards"
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n=== 0-ply BC autoresearch iter {iteration:04d}: {config.name} ===\n"
            f"incumbent={incumbent_label} bin={incumbent_bin}\n"
            f"target_elo={args.target_elo:.1f}",
            flush=True,
        )

        if args.training_league_mode == "elite":
            training_league = _elite_training_league(incumbent_bin, incumbent_pt)
        else:
            training_league = build_training_league(iteration, incumbent_bin, incumbent_pt, include_hs=False)
        print("Training league:")
        for entry in training_league:
            print(f"  {entry.label:<12} {entry.kind:<3} {entry.bin_path or ''}")

        data_summary = generate_winner_data(
            training_league,
            shard_dir,
            games=args.data_games,
            workers=args.workers,
            seed_base=args.seed_base + iteration * 1_000_000,
            games_per_shard=args.games_per_shard,
            ab_depth=args.ab_depth,
            hs_depth=0,
            hs_time_ms=0.0,
        )
        print(
            f"Data: {data_summary['games']} games, {data_summary['steps']} winner steps, "
            f"{data_summary['elapsed_sec']:.1f}s, wins={data_summary['wins']}",
            flush=True,
        )

        candidate_pt = iter_dir / "candidate.pt"
        candidate_bin = iter_dir / "candidate.bin"
        train_summary = train_bc_candidate(
            incumbent_pt,
            shard_dir,
            candidate_pt,
            candidate_bin,
            device=args.device,
            lr=config.lr,
            epochs=config.epochs,
            batch_size=config.batch_size,
            max_examples=config.max_examples,
            train_scope=config.train_scope,
            label_smoothing=config.label_smoothing,
            weight_format=args.weight_format,
            use_action_weights=config.action_weights,
            teacher_balanced=config.teacher_balanced,
            teacher_cap_ratio=config.teacher_cap_ratio,
            strategic_only=config.strategic_only,
            opening_turns=config.opening_turns,
            top_teachers=config.top_teachers,
            search_only=config.search_only,
            teacher_allowlist=config.teacher_allowlist,
            kl_alpha=config.kl_alpha,
            disagreement_boost=config.disagreement_boost,
        )

        eval_roster = _fixed_eval_roster(
            incumbent_label,
            incumbent_bin,
            candidate_bin=candidate_bin,
            prior_incumbents=prior_incumbents,
        )
        quick_eval = evaluate_roster(
            eval_roster,
            games_per_combo=args.quick_games_per_combo,
            workers=args.workers,
            seed_base=args.eval_seed_base + iteration * 1_000_000,
            ab_depth=args.ab_depth,
            hs_depth=0,
            hs_time_ms=0.0,
        )
        _print_eval(quick_eval, "Quick fixed 0-ply zoo")
        quick_rows = _row_by_label(quick_eval)
        quick_best_other = _best_non_candidate(quick_eval)
        quick_delta = float(quick_rows["candidate"]["elo"]) - float(quick_best_other["elo"])
        keep = quick_delta > args.quick_min_delta
        confirm_eval = None

        if keep:
            print(
                f"Quick gate passed by {quick_delta:+.1f} Elo over "
                f"{quick_best_other['label']}; "
                f"confirming with gpc={args.confirm_games_per_combo}",
                flush=True,
            )
            confirm_eval = evaluate_roster(
                eval_roster,
                games_per_combo=args.confirm_games_per_combo,
                workers=args.workers,
                seed_base=args.eval_seed_base + iteration * 1_000_000 + 500_000,
                ab_depth=args.ab_depth,
                hs_depth=0,
                hs_time_ms=0.0,
            )
            _print_eval(confirm_eval, "Confirm fixed 0-ply zoo")
            confirm_rows = _row_by_label(confirm_eval)
            confirm_best_other = _best_non_candidate(confirm_eval)
            delta = float(confirm_rows["candidate"]["elo"]) - float(confirm_best_other["elo"])
            keep = delta > 0.0
            active_eval = confirm_eval
        else:
            delta = quick_delta
            active_eval = quick_eval

        rows = _row_by_label(active_eval)
        candidate_elo = float(rows["candidate"]["elo"])
        incumbent_elo = float(rows["incumbent"]["elo"])
        best_other = _best_non_candidate(active_eval)
        summary = {
            "iteration": iteration,
            "config": asdict(config),
            "keep": keep,
            "candidate_elo": candidate_elo,
            "incumbent_elo": incumbent_elo,
            "best_other_label": best_other["label"],
            "best_other_elo": float(best_other["elo"]),
            "delta_elo": candidate_elo - float(best_other["elo"]),
            "target_elo": args.target_elo,
            "incumbent_label": incumbent_label,
            "incumbent_bin": str(incumbent_bin),
            "data": data_summary,
            "train": train_summary,
            "training_league": [asdict(e) for e in training_league],
            "eval_roster": [asdict(e) for e in eval_roster],
            "quick_eval": _json_safe_eval(quick_eval),
            "confirm_eval": _json_safe_eval(confirm_eval) if confirm_eval else None,
        }

        if keep:
            kept_pt = work_dir / f"kept_iter_{iteration:04d}.pt"
            kept_bin = work_dir / f"kept_iter_{iteration:04d}.bin"
            shutil.copy2(candidate_pt, kept_pt)
            shutil.copy2(candidate_bin, kept_bin)
            prior_incumbents.append((incumbent_label, incumbent_bin))
            incumbent_label = f"sd{iteration:04d}"
            incumbent_bin = kept_bin.resolve()
            incumbent_pt = kept_pt.resolve()
            summary["kept_pt"] = str(kept_pt)
            summary["kept_bin"] = str(kept_bin)
            print(
                f"KEEP iter {iteration:04d}: candidate {candidate_elo:.1f} "
                f"> best other {best_other['label']} {float(best_other['elo']):.1f}; "
                f"new incumbent={incumbent_label}",
                flush=True,
            )
        else:
            print(
                f"DISCARD iter {iteration:04d}: candidate {candidate_elo:.1f} "
                f"<= best other {best_other['label']} {float(best_other['elo']):.1f} "
                f"({candidate_elo - float(best_other['elo']):+.1f})",
                flush=True,
            )

        with open(iter_dir / "summary_0ply.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.time(),
                "iteration": iteration,
                "config": config.name,
                "keep": keep,
                "candidate_elo": candidate_elo,
                "incumbent_elo": incumbent_elo,
                "best_other_label": best_other["label"],
                "best_other_elo": float(best_other["elo"]),
                "delta_elo": candidate_elo - float(best_other["elo"]),
                "active_incumbent": incumbent_label,
            }) + "\n")

        if keep and candidate_elo >= args.target_elo:
            print(f"Target reached: {candidate_elo:.1f} >= {args.target_elo:.1f}", flush=True)
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK))
    parser.add_argument("--incumbent-label", default="eg0150")
    parser.add_argument("--incumbent-bin", default=str(EGGROLL_DIR / "kept_iter_0150.bin"))
    parser.add_argument("--incumbent-pt", default=str(EGGROLL_DIR / "kept_iter_0150.pt"))
    parser.add_argument("--start-iteration", type=int, default=40)
    parser.add_argument("--max-iterations", type=int, default=9999)
    parser.add_argument("--target-elo", type=float, default=1300.0)
    parser.add_argument("--data-games", type=int, default=1024)
    parser.add_argument("--games-per-shard", type=int, default=32)
    parser.add_argument("--quick-games-per-combo", type=int, default=16)
    parser.add_argument("--quick-min-delta", type=float, default=0.0)
    parser.add_argument("--confirm-games-per-combo", type=int, default=64)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=981000000)
    parser.add_argument("--eval-seed-base", type=int, default=982000000)
    parser.add_argument("--ab-depth", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weight-format", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--schedule", choices=["default", "trunk", "topnn"], default="default")
    parser.add_argument("--training-league-mode", choices=["rotating", "elite"], default="rotating")
    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
