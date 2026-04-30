#!/usr/bin/env python3
"""Local deterministic H-S benchmark/regression harness.

Runs the standalone catan_player build, then a small deterministic battery:
H-S/H-S+/AB2 self-play smoke seeds, H-S+ vs H-S 2v2, and H-S+ vs AB2 2v2.
All command output plus parsed winner counts and wall times are written to a
timestamped log under catan_player/bench/logs by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO


SCRIPT_PATH = Path(__file__).resolve()
BENCH_DIR = SCRIPT_PATH.parent
PLAYER_DIR = BENCH_DIR.parent
DEFAULT_BINARY = PLAYER_DIR / "catan_player"
DEFAULT_BUILD_SCRIPT = PLAYER_DIR / "build.sh"
DEFAULT_LOG_DIR = BENCH_DIR / "logs"

TOTAL_RE = re.compile(
    r"^(?P<games>\d+) games in (?P<seconds>[0-9.]+)s "
    r"\((?P<gps>[0-9.]+) games/s\), avg turns (?P<turns>[0-9.]+)"
)
SEAT_WINS_RE = re.compile(
    r"^Seat wins: P0=(?P<p0>\d+) P1=(?P<p1>\d+) "
    r"P2=(?P<p2>\d+) P3=(?P<p3>\d+)"
)
H2H_RE = re.compile(
    r"^H2H: (?P<a>\S+)=(?P<a_wins>\d+) "
    r"(?P<b>\S+)=(?P<b_wins>\d+) WR=(?P<wr>[0-9.]+)%"
)
GAME_RE = re.compile(
    r"^\[(?P<idx>\d+)/(?P<total>\d+)\] seed=(?P<seed>\d+) "
    r"winner=P(?P<winner>-?\d+) actualVP=\[(?P<vp>[0-9 ]+)\] "
    r"turns=(?P<turns>\d+) decisions=(?P<decisions>\d+)"
)


@dataclass
class CommandSpec:
    label: str
    command: list[str]
    cwd: Path
    kind: str = "run"
    agent: str | None = None
    opponent: str | None = None
    seed: int | None = None


@dataclass
class CommandResult:
    spec: CommandSpec
    returncode: int
    wall_seconds: float
    output: str
    parsed: dict[str, object] = field(default_factory=dict)


class Tee:
    def __init__(self, log_file: TextIO) -> None:
        self.log_file = log_file

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.log_file.write(text)
        self.log_file.flush()


def parse_csv(value: str, label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return items


def parse_seed_csv(value: str) -> list[int]:
    seeds: list[int] = []
    for item in parse_csv(value, "--smoke-seeds"):
        try:
            seeds.append(int(item, 0))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed: {item}") from exc
    return seeds


def positive_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def quote_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_player_output(output: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    game_rows: list[dict[str, object]] = []

    for line in output.splitlines():
        match = TOTAL_RE.match(line)
        if match:
            parsed["player_games"] = int(match.group("games"))
            parsed["player_seconds"] = float(match.group("seconds"))
            parsed["player_games_per_second"] = float(match.group("gps"))
            parsed["avg_turns"] = float(match.group("turns"))
            continue

        match = SEAT_WINS_RE.match(line)
        if match:
            parsed["seat_wins"] = {
                "P0": int(match.group("p0")),
                "P1": int(match.group("p1")),
                "P2": int(match.group("p2")),
                "P3": int(match.group("p3")),
            }
            continue

        match = H2H_RE.match(line)
        if match:
            parsed["h2h"] = {
                match.group("a"): int(match.group("a_wins")),
                match.group("b"): int(match.group("b_wins")),
                "wr_percent": float(match.group("wr")),
            }
            continue

        match = GAME_RE.match(line)
        if match:
            vp = tuple(int(x) for x in match.group("vp").split())
            game_rows.append(
                {
                    "seed": int(match.group("seed")),
                    "winner": int(match.group("winner")),
                    "vp": vp,
                    "turns": int(match.group("turns")),
                    "decisions": int(match.group("decisions")),
                }
            )

    if game_rows:
        parsed["games"] = game_rows
    return parsed


def summarize_counts(parsed: dict[str, object]) -> str:
    h2h = parsed.get("h2h")
    if isinstance(h2h, dict):
        parts = [
            f"{key}={value}"
            for key, value in h2h.items()
            if key != "wr_percent"
        ]
        wr = h2h.get("wr_percent")
        if isinstance(wr, float):
            parts.append(f"WR={wr:.1f}%")
        return " ".join(parts)

    seat_wins = parsed.get("seat_wins")
    if isinstance(seat_wins, dict):
        return " ".join(f"{key}={value}" for key, value in seat_wins.items())

    games = parsed.get("games")
    if isinstance(games, list) and games:
        winners = [f"seed={row['seed']} winner=P{row['winner']}" for row in games]
        return "; ".join(winners)

    return "-"


def run_command(spec: CommandSpec, tee: Tee, dry_run: bool) -> CommandResult:
    tee.write(f"\n=== {spec.label} ===\n")
    tee.write(f"$ {quote_cmd(spec.command)}\n")

    if dry_run:
        tee.write("[dry-run] skipped\n")
        return CommandResult(spec, 0, 0.0, "", {})

    start = time.monotonic()
    output_parts: list[str] = []
    try:
        process = subprocess.Popen(
            spec.command,
            cwd=str(spec.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        wall = time.monotonic() - start
        message = f"[error] failed to start command: {exc}\n"
        tee.write(message)
        tee.write(f"--- result: exit=127 wall={wall:.2f}s counts=-\n")
        return CommandResult(spec, 127, wall, message, {})
    assert process.stdout is not None
    try:
        for line in process.stdout:
            output_parts.append(line)
            tee.write(line)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise

    wall = time.monotonic() - start
    output = "".join(output_parts)
    parsed = parse_player_output(output)
    tee.write(
        f"--- result: exit={returncode} wall={wall:.2f}s "
        f"counts={summarize_counts(parsed)}\n"
    )
    return CommandResult(spec, returncode, wall, output, parsed)


def player_passthrough_args(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    passthrough: list[str] = []

    for name in (
        "plus_workers",
        "plus_depth",
        "plus_k",
        "plus_time_ms",
        "plus_leaf_mode",
        "plus_opp_model",
        "plus_opp_depth",
        "plus_cache_bits",
    ):
        value = getattr(args, name)
        if value is not None:
            passthrough.extend([f"--{name.replace('_', '-')}", str(value)])

    for item in args.player_arg:
        passthrough.append(item)

    if unknown and unknown[0] == "--":
        unknown = unknown[1:]
    passthrough.extend(unknown)
    return passthrough


def build_specs(args: argparse.Namespace, extra_player_args: list[str]) -> list[CommandSpec]:
    binary = Path(args.binary).resolve()
    build_script = Path(args.build_script).resolve()
    smoke_agents = parse_csv(args.smoke_agents, "--smoke-agents")
    smoke_seeds = parse_seed_csv(args.smoke_seeds)

    specs: list[CommandSpec] = []
    if not args.skip_build:
        specs.append(CommandSpec("build", [str(build_script)], PLAYER_DIR, kind="build"))

    if not args.no_smoke:
        for agent in smoke_agents:
            for seed in smoke_seeds:
                specs.append(
                    CommandSpec(
                        f"smoke:{agent}:seed{seed}",
                        [
                            str(binary),
                            "--agent",
                            agent,
                            "--games",
                            str(args.smoke_games),
                            "--seed",
                            str(seed),
                            *extra_player_args,
                        ],
                        PLAYER_DIR,
                        kind="smoke",
                        agent=agent,
                        seed=seed,
                    )
                )

    if not args.no_h2h:
        specs.append(
            CommandSpec(
                f"h2h:{args.agent}_vs_{args.baseline_agent}",
                [
                    str(binary),
                    "--h2h",
                    "--agent",
                    args.agent,
                    "--opponent",
                    args.baseline_agent,
                    "--games",
                    str(args.games),
                    "--seed",
                    str(args.seed),
                    *extra_player_args,
                ],
                PLAYER_DIR,
                kind="h2h",
                agent=args.agent,
                opponent=args.baseline_agent,
                seed=args.seed,
            )
        )
        specs.append(
            CommandSpec(
                f"h2h:{args.agent}_vs_{args.ab_agent}",
                [
                    str(binary),
                    "--h2h",
                    "--agent",
                    args.agent,
                    "--opponent",
                    args.ab_agent,
                    "--games",
                    str(args.games),
                    "--seed",
                    str(args.seed),
                    *extra_player_args,
                ],
                PLAYER_DIR,
                kind="h2h",
                agent=args.agent,
                opponent=args.ab_agent,
                seed=args.seed,
            )
        )

    return specs


def smoke_signature(result: CommandResult) -> tuple[int, tuple[int, ...], int] | None:
    games = result.parsed.get("games")
    if not isinstance(games, list) or len(games) != 1:
        return None
    row = games[0]
    if not isinstance(row, dict):
        return None
    vp = row.get("vp")
    if not isinstance(vp, tuple):
        return None
    return int(row["winner"]), tuple(int(x) for x in vp), int(row["turns"])


def deterministic_smoke_notes(results: list[CommandResult]) -> list[str]:
    by_seed_agent: dict[tuple[int, str], CommandResult] = {}
    for result in results:
        if result.spec.kind == "smoke" and result.spec.seed is not None and result.spec.agent:
            by_seed_agent[(result.spec.seed, result.spec.agent.lower())] = result

    notes: list[str] = []
    seeds = sorted({seed for seed, _agent in by_seed_agent})
    for seed in seeds:
        hs = by_seed_agent.get((seed, "h-s")) or by_seed_agent.get((seed, "hs"))
        hsp = (
            by_seed_agent.get((seed, "h-s+"))
            or by_seed_agent.get((seed, "hs+"))
            or by_seed_agent.get((seed, "hsp"))
        )
        if hs is None or hsp is None:
            continue
        hs_sig = smoke_signature(hs)
        hsp_sig = smoke_signature(hsp)
        if hs_sig is None or hsp_sig is None:
            notes.append(f"seed {seed}: skipped H-S/H-S+ signature check")
        elif hs_sig == hsp_sig:
            winner, vp, turns = hs_sig
            notes.append(f"seed {seed}: H-S and H-S+ match winner=P{winner} vp={vp} turns={turns}")
        else:
            notes.append(f"seed {seed}: mismatch H-S={hs_sig} H-S+={hsp_sig}")
    return notes


def write_summary(results: list[CommandResult], notes: list[str], tee: Tee) -> None:
    tee.write("\n=== summary ===\n")
    tee.write(f"{'label':34} {'exit':>4} {'wall':>9} counts\n")
    tee.write(f"{'-' * 34} {'-' * 4:>4} {'-' * 9:>9} {'-' * 20}\n")
    for result in results:
        tee.write(
            f"{result.spec.label[:34]:34} "
            f"{result.returncode:>4} "
            f"{result.wall_seconds:>8.2f}s "
            f"{summarize_counts(result.parsed)}\n"
        )
    if notes:
        tee.write("\nDeterministic smoke check:\n")
        for note in notes:
            tee.write(f"  {note}\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local deterministic catan_player H-S regression benchmarks.",
        epilog="Unknown arguments after -- are appended to every catan_player run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=positive_int, default=2, help="2v2 games per H2H matchup")
    parser.add_argument("--smoke-games", type=positive_int, default=1, help="self-play games per smoke seed")
    parser.add_argument("--seed", type=int, default=810000, help="seed base for H2H matchups")
    parser.add_argument("--smoke-seeds", default="810000,810001", help="comma-separated self-play smoke seeds")
    parser.add_argument("--smoke-agents", default="h-s,h-s+,ab2", help="comma-separated self-play agents")
    parser.add_argument("--agent", default="h-s+", help="primary H2H agent")
    parser.add_argument("--baseline-agent", default="h-s", help="baseline for the H-S+ vs H-S matchup")
    parser.add_argument("--ab-agent", default="ab2", help="baseline for the H-S+ vs AB2 matchup")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY), help="path to catan_player binary")
    parser.add_argument("--build-script", default=str(DEFAULT_BUILD_SCRIPT), help="path to build.sh")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="directory for timestamped logs")
    parser.add_argument("--skip-build", action="store_true", help="do not run build.sh first")
    parser.add_argument("--no-smoke", action="store_true", help="skip self-play smoke runs")
    parser.add_argument("--no-h2h", action="store_true", help="skip H2H matchups")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed command")
    parser.add_argument("--dry-run", action="store_true", help="write the planned commands without running them")
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="exit nonzero if H-S and H-S+ smoke signatures differ",
    )
    parser.add_argument(
        "--player-arg",
        action="append",
        default=[],
        help="append one raw argument to every catan_player run; repeat for values",
    )

    parser.add_argument("--plus-workers", type=positive_int, default=None, help="pass through to catan_player")
    parser.add_argument("--plus-depth", type=positive_int, default=None, help="pass through to catan_player")
    parser.add_argument("--plus-k", default=None, help="pass through to catan_player")
    parser.add_argument("--plus-time-ms", type=positive_int, default=None, help="pass through to catan_player")
    parser.add_argument("--plus-leaf-mode", type=nonnegative_int, default=None, help="pass through to catan_player")
    parser.add_argument("--plus-opp-model", default=None, help="pass through to catan_player")
    parser.add_argument("--plus-opp-depth", type=positive_int, default=None, help="pass through to catan_player")
    parser.add_argument("--plus-cache-bits", type=positive_int, default=None, help="pass through to catan_player")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args, unknown = parser.parse_known_args(argv)
    extra_player_args = player_passthrough_args(args, unknown)

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"hs_regression_{stamp}.log"

    results: list[CommandResult] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = Tee(log_file)
        tee.write(f"hs_regression log: {log_path}\n")
        tee.write(f"cwd: {Path.cwd()}\n")
        if extra_player_args:
            tee.write(f"extra catan_player args: {quote_cmd(extra_player_args)}\n")

        specs = build_specs(args, extra_player_args)
        if not specs:
            tee.write("No commands selected.\n")
            return 2

        for spec in specs:
            result = run_command(spec, tee, args.dry_run)
            results.append(result)
            if result.returncode != 0 and (spec.kind == "build" or not args.keep_going):
                tee.write(f"Stopping after failed command: {spec.label}\n")
                break

        notes = deterministic_smoke_notes(results)
        write_summary(results, notes, tee)
        tee.write(f"\nLog written to: {log_path}\n")

    failed = [result for result in results if result.returncode != 0]
    if failed:
        return failed[0].returncode or 1

    if args.strict_determinism and any("mismatch" in note for note in notes):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
