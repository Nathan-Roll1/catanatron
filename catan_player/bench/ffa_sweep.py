#!/usr/bin/env python3
"""Deterministic FFA sweep harness for no-NN H-S+ tuning."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
BENCH_DIR = SCRIPT_PATH.parent
PLAYER_DIR = BENCH_DIR.parent
DEFAULT_BINARY = PLAYER_DIR / "catan_player"
DEFAULT_BUILD_SCRIPT = PLAYER_DIR / "build.sh"
DEFAULT_LOG_DIR = BENCH_DIR / "logs"

FFA_RE = re.compile(
    r"^FFA: (?P<agent>\S+)=(?P<wins>\d+) "
    r"field_(?P<field>\S+)=(?P<losses>\d+) WR=(?P<wr>[0-9.]+)%"
)
TOTAL_RE = re.compile(
    r"^(?P<games>\d+) games in (?P<seconds>[0-9.]+)s "
    r"\((?P<gps>[0-9.]+) games/s\), avg turns (?P<turns>[0-9.]+)"
)


@dataclass(frozen=True)
class Variant:
    name: str
    args: tuple[str, ...]


@dataclass
class Result:
    variant: Variant
    field: str
    seed: int
    games: int
    returncode: int
    wall_seconds: float
    wins: int = 0
    losses: int = 0
    wr: float = 0.0
    player_seconds: float = 0.0
    avg_turns: float = 0.0
    output: str = ""


VARIANTS: dict[str, Variant] = {
    "default": Variant("default", ()),
    "old-default": Variant(
        "old-default",
        ("--plus-leaf-mode", "5", "--plus-policy-profile", "1"),
    ),
    "leaf5-policy2": Variant(
        "leaf5-policy2",
        ("--plus-leaf-mode", "5", "--plus-policy-profile", "2"),
    ),
    "leaf7-policy1": Variant(
        "leaf7-policy1",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "1"),
    ),
    "leaf7-policy3": Variant(
        "leaf7-policy3",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "3"),
    ),
    "leaf7-policy4": Variant(
        "leaf7-policy4",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "4"),
    ),
    "leaf7-policy5": Variant(
        "leaf7-policy5",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "5"),
    ),
    "leaf7-policy6": Variant(
        "leaf7-policy6",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "6"),
    ),
    "leaf7-policy7": Variant(
        "leaf7-policy7",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "7"),
    ),
    "leaf7-policy8": Variant(
        "leaf7-policy8",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "8"),
    ),
    "leaf7": Variant("leaf7", ("--plus-leaf-mode", "7")),
    "leaf8": Variant("leaf8", ("--plus-leaf-mode", "8")),
    "leaf12": Variant("leaf12", ("--plus-leaf-mode", "12")),
    "leaf13": Variant("leaf13", ("--plus-leaf-mode", "13")),
    "leaf14": Variant("leaf14", ("--plus-leaf-mode", "14")),
    "leaf16": Variant("leaf16", ("--plus-leaf-mode", "16")),
    "policy2": Variant("policy2", ("--plus-policy-profile", "2")),
    "leaf8-policy2": Variant(
        "leaf8-policy2",
        ("--plus-leaf-mode", "8", "--plus-policy-profile", "2"),
    ),
    "leaf7-policy2": Variant(
        "leaf7-policy2",
        ("--plus-leaf-mode", "7", "--plus-policy-profile", "2"),
    ),
    "root-ensemble1": Variant("root-ensemble1", ("--plus-root-ensemble", "1")),
    "root-ensemble2": Variant("root-ensemble2", ("--plus-root-ensemble", "2")),
    "root-scoreblend3": Variant("root-scoreblend3", ("--plus-root-ensemble", "3")),
    "root-scoreblend4": Variant("root-scoreblend4", ("--plus-root-ensemble", "4")),
    "tt": Variant("tt", ("--plus-tt-bits", "18")),
    "pvs": Variant("pvs", ("--plus-pvs", "1")),
    "ttid": Variant("ttid", ("--plus-tt-bits", "18", "--plus-id", "1")),
    "tt-pvs-noid": Variant(
        "tt-pvs-noid",
        ("--plus-tt-bits", "18", "--plus-pvs", "1"),
    ),
    "tt-pvs": Variant(
        "tt-pvs",
        ("--plus-tt-bits", "18", "--plus-pvs", "1", "--plus-id", "1"),
    ),
    "tt-pvs-lmr": Variant(
        "tt-pvs-lmr",
        (
            "--plus-tt-bits", "18",
            "--plus-pvs", "1",
            "--plus-lmr", "1",
            "--plus-id", "1",
        ),
    ),
    "leaf8-ttid": Variant(
        "leaf8-ttid",
        ("--plus-leaf-mode", "8", "--plus-tt-bits", "18", "--plus-id", "1"),
    ),
    "k7": Variant("k7", ("--plus-k", "7,4,2,2,2,2")),
    "k7-pvs": Variant("k7-pvs", ("--plus-k", "7,4,2,2,2,2", "--plus-pvs", "1")),
    "leaf8-k7": Variant("leaf8-k7", ("--plus-leaf-mode", "8", "--plus-k", "7,4,2,2,2,2")),
    "leaf8-k7-pvs": Variant(
        "leaf8-k7-pvs",
        ("--plus-leaf-mode", "8", "--plus-k", "7,4,2,2,2,2", "--plus-pvs", "1"),
    ),
    "k6-5": Variant("k6-5", ("--plus-k", "6,5,2,2,2,2")),
    "k6-5-pvs": Variant("k6-5-pvs", ("--plus-k", "6,5,2,2,2,2", "--plus-pvs", "1")),
    "depth6-k6-4-3": Variant("depth6-k6-4-3", ("--plus-k", "6,4,3,2,2,2")),
    "depth7-tight": Variant(
        "depth7-tight",
        ("--plus-depth", "7", "--plus-k", "6,4,2,2,1,1,1"),
    ),
    "det-maxn": Variant("det-maxn", ("--plus-opp-model", "det-maxn")),
    "det-kf-maxn": Variant("det-kf-maxn", ("--plus-opp-model", "det-kf-maxn")),
    "det-kf-ab2": Variant("det-kf-ab2", ("--plus-opp-model", "det-kf-ab2")),
    "nested-hs2": Variant("nested-hs2", ("--plus-opp-model", "nested-hs2")),
    "nested-hs3": Variant("nested-hs3", ("--plus-opp-model", "nested-hs3")),
    "nested-hs4": Variant("nested-hs4", ("--plus-opp-model", "nested-hs4")),
    "opp-hs": Variant("opp-hs", ("--plus-opp-model", "hs")),
    "opp-hs-leaf": Variant("opp-hs-leaf", ("--plus-opp-model", "hs-leaf")),
    "rescue": Variant("rescue", ("--plus-rescue", "1")),
    "leaf-extend": Variant("leaf-extend", ("--plus-leaf-extend", "1")),
    "root-rollout": Variant("root-rollout", ("--plus-root-rollout", "1")),
}


def quote_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_command(command: list[str], cwd: Path) -> tuple[int, float, str]:
    start = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, time.monotonic() - start, proc.stdout


def parse_result(
    variant: Variant,
    field: str,
    seed: int,
    games: int,
    returncode: int,
    wall: float,
    output: str,
) -> Result:
    result = Result(variant, field, seed, games, returncode, wall, output=output)
    for line in output.splitlines():
        m = FFA_RE.match(line)
        if m:
            result.wins = int(m.group("wins"))
            result.losses = int(m.group("losses"))
            result.wr = float(m.group("wr"))
            continue
        m = TOTAL_RE.match(line)
        if m:
            result.player_seconds = float(m.group("seconds"))
            result.avg_turns = float(m.group("turns"))
    return result


def run_one(args: argparse.Namespace, variant: Variant, field: str, seed: int) -> Result:
    binary = Path(args.binary).resolve()
    command = [
        str(binary),
        "--ffa",
        "--agent",
        args.agent,
        "--opponent",
        field,
        "--games",
        str(args.games),
        "--seed",
        str(seed),
        *variant.args,
    ]
    if args.verbose:
        print(f"$ {quote_cmd(command)}", flush=True)
    if args.dry_run:
        return Result(variant, field, seed, args.games, 0, 0.0)
    rc, wall, out = run_command(command, PLAYER_DIR)
    return parse_result(variant, field, seed, args.games, rc, wall, out)


def aggregate(results: list[Result]) -> list[tuple[str, str, int, int, float, float]]:
    buckets: dict[tuple[str, str], list[Result]] = {}
    for result in results:
        buckets.setdefault((result.variant.name, result.field), []).append(result)

    rows: list[tuple[str, str, int, int, float, float]] = []
    for (variant, field), items in buckets.items():
        wins = sum(item.wins for item in items)
        losses = sum(item.losses for item in items)
        decided = wins + losses
        wr = 100.0 * wins / decided if decided else 0.0
        seconds = sum(item.player_seconds or item.wall_seconds for item in items)
        rows.append((variant, field, wins, losses, wr, seconds))
    rows.sort(key=lambda row: (row[1], -row[4], row[5], row[0]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic FFA sweeps for H-S+ variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", default="h-s+")
    parser.add_argument("--fields", default="h-s,ab2")
    parser.add_argument("--variants", default="default,leaf8-policy2,leaf8-ttid,k7,k6-5,depth6-k6-4-3,depth7-tight")
    parser.add_argument("--seeds", default="981000,982000")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--build-script", default=str(DEFAULT_BUILD_SCRIPT))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.games < 1:
        parser.error("--games must be >= 1")
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    selected_variants: list[Variant] = []
    for name in parse_csv(args.variants):
        if name not in VARIANTS:
            parser.error(f"unknown variant {name!r}; known: {', '.join(sorted(VARIANTS))}")
        selected_variants.append(VARIANTS[name])
    fields = parse_csv(args.fields)
    seeds = [int(seed, 0) for seed in parse_csv(args.seeds)]

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"ffa_sweep_{stamp}.log"

    with log_path.open("w", encoding="utf-8") as log:
        def emit(text: str) -> None:
            print(text, end="", flush=True)
            log.write(text)
            log.flush()

        emit(f"ffa_sweep log: {log_path}\n")
        if not args.skip_build:
            build_cmd = [str(Path(args.build_script).resolve())]
            emit(f"$ {quote_cmd(build_cmd)}\n")
            if args.dry_run:
                emit("[dry-run] skipped build\n")
            else:
                rc, wall, out = run_command(build_cmd, PLAYER_DIR)
                emit(out)
                emit(f"build exit={rc} wall={wall:.2f}s\n")
                if rc != 0:
                    return rc or 1

        jobs: list[tuple[Variant, str, int]] = [
            (variant, field, seed)
            for variant in selected_variants
            for field in fields
            for seed in seeds
        ]

        results: list[Result] = []
        if args.jobs == 1:
            for variant, field, seed in jobs:
                result = run_one(args, variant, field, seed)
                results.append(result)
                emit(
                    f"{variant.name:18} field={field:4} seed={seed} "
                    f"exit={result.returncode} {result.wins}-{result.losses} "
                    f"WR={result.wr:5.1f}% time={result.player_seconds or result.wall_seconds:.2f}s\n"
                )
                if result.returncode != 0:
                    emit(result.output)
                    return result.returncode or 1
        else:
            with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                future_map = {
                    pool.submit(run_one, args, variant, field, seed): (variant, field, seed)
                    for variant, field, seed in jobs
                }
                for future in futures.as_completed(future_map):
                    result = future.result()
                    results.append(result)
                    emit(
                        f"{result.variant.name:18} field={result.field:4} seed={result.seed} "
                        f"exit={result.returncode} {result.wins}-{result.losses} "
                        f"WR={result.wr:5.1f}% time={result.player_seconds or result.wall_seconds:.2f}s\n"
                    )
                    if result.returncode != 0:
                        emit(result.output)
                        return result.returncode or 1

        emit("\nsummary\n")
        emit(f"{'variant':18} {'field':5} {'wins':>4} {'loss':>4} {'WR':>7} {'seconds':>9}\n")
        for variant, field, wins, losses, wr, seconds in aggregate(results):
            emit(f"{variant:18} {field:5} {wins:4d} {losses:4d} {wr:6.1f}% {seconds:8.2f}s\n")
        emit(f"\nLog written to: {log_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
