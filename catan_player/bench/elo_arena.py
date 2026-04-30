#!/usr/bin/env python3
"""Mixed-seat Elo arena for deterministic Catan variant benchmarking."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as dt
import math
import random
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
BENCH_DIR = SCRIPT_PATH.parent
PLAYER_DIR = BENCH_DIR.parent
DEFAULT_BINARY = PLAYER_DIR / "catan_player"
DEFAULT_BUILD_SCRIPT = PLAYER_DIR / "build.sh"
DEFAULT_LOG_DIR = BENCH_DIR / "logs"

CORE_VARIANTS = (
    "default",
    "old-default",
    "leaf5-policy2",
    "leaf7-policy1",
    "leaf7-policy3",
    "leaf7-policy4",
    "leaf7-policy5",
    "leaf7-policy6",
    "leaf7-policy7",
    "leaf7-policy8",
    "leaf8",
    "leaf12",
    "leaf14",
    "leaf16",
    "pvs",
    "k7",
    "k7-pvs",
    "leaf8-k7",
    "leaf8-k7-pvs",
    "k6-5",
    "k6-5-pvs",
    "depth7-tight",
    "root-ensemble1",
    "root-scoreblend3",
    "ab2",
    "h-s",
)

SLOW_VARIANTS = (
    "leaf13",
    "det-maxn",
    "det-kf-maxn",
    "det-kf-ab2",
    "nested-hs2",
    "nested-hs3",
    "nested-hs4",
    "opp-hs",
    "opp-hs-leaf",
    "tt",
    "ttid",
    "tt-pvs-noid",
    "tt-pvs",
    "tt-pvs-lmr",
    "leaf8-ttid",
    "rescue",
    "leaf-extend",
    "root-rollout",
    "root-ensemble2",
    "root-scoreblend4",
)

ALL_VARIANTS = tuple(dict.fromkeys((*CORE_VARIANTS, *SLOW_VARIANTS)))

ARENA_RE = re.compile(
    r"^ARENA: seed=(?P<seed>\d+) winner=P(?P<winner>-?\d+) "
    r"winner_variant=(?P<winner_variant>\S+) "
    r"variants=\[(?P<variants>[^\]]+)\] "
    r"actualVP=\[(?P<vp>[0-9 ]+)\] "
    r"turns=(?P<turns>\d+) decisions=(?P<decisions>\d+)"
)


@dataclass(frozen=True)
class Job:
    index: int
    seed: int
    variants: tuple[str, str, str, str]


@dataclass
class GameResult:
    job: Job
    returncode: int
    wall_seconds: float
    winner: int = -1
    vp: tuple[int, int, int, int] = (0, 0, 0, 0)
    turns: int = 0
    decisions: int = 0
    output: str = ""


@dataclass
class Standing:
    elo: float = 1500.0
    games: int = 0
    wins: int = 0
    vp_sum: int = 0
    rank_sum: float = 0.0
    seconds: float = 0.0


def quote_cmd(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_variants(value: str) -> list[str]:
    if value == "core":
        return list(CORE_VARIANTS)
    if value == "all":
        return list(ALL_VARIANTS)
    if value == "slow":
        return list(SLOW_VARIANTS)
    out = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in out if item not in ALL_VARIANTS]
    if unknown:
        known = ", ".join(ALL_VARIANTS)
        raise ValueError(f"unknown variants {unknown}; known: {known}")
    return out


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


def make_schedule(variants: list[str], games: int, seed: int) -> list[Job]:
    if len(variants) < 4:
        raise ValueError("Elo arena needs at least four variants")
    rng = random.Random(seed)
    jobs: list[Job] = []
    for i in range(games):
        seats = tuple(rng.sample(variants, 4))
        game_seed = rng.randrange(1, 2**31 - 1)
        jobs.append(Job(i, game_seed, seats))
    return jobs


def parse_result(job: Job, returncode: int, wall: float, output: str) -> GameResult:
    result = GameResult(job, returncode, wall, output=output)
    for line in output.splitlines():
        m = ARENA_RE.match(line)
        if not m:
            continue
        variants = tuple(item.strip() for item in m.group("variants").split(","))
        if variants != job.variants:
            continue
        result.winner = int(m.group("winner"))
        vp_vals = tuple(int(item) for item in m.group("vp").split())
        if len(vp_vals) == 4:
            result.vp = vp_vals  # type: ignore[assignment]
        result.turns = int(m.group("turns"))
        result.decisions = int(m.group("decisions"))
    return result


def run_one(args: argparse.Namespace, job: Job) -> GameResult:
    binary = Path(args.binary).resolve()
    command = [
        str(binary),
        "--arena",
        ",".join(job.variants),
        "--games",
        "1",
        "--seed",
        str(job.seed),
    ]
    if args.verbose:
        print(f"$ {quote_cmd(command)}", flush=True)
    if args.dry_run:
        return GameResult(job, 0, 0.0)
    rc, wall, out = run_command(command, PLAYER_DIR)
    return parse_result(job, rc, wall, out)


def game_ranks(result: GameResult) -> list[float]:
    scores = []
    for seat, vp in enumerate(result.vp):
        winner_bonus = 100 if seat == result.winner else 0
        scores.append((-(winner_bonus + vp), seat))
    scores.sort()

    ranks = [0.0, 0.0, 0.0, 0.0]
    pos = 0
    while pos < len(scores):
        end = pos + 1
        while end < len(scores) and scores[end][0] == scores[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for _, seat in scores[pos:end]:
            ranks[seat] = avg_rank
        pos = end
    return ranks


def update_elo(
    standings: dict[str, Standing],
    result: GameResult,
    k_factor: float,
) -> None:
    ranks = game_ranks(result)
    deltas = {name: 0.0 for name in result.job.variants}

    for i in range(4):
        vi = result.job.variants[i]
        si = standings[vi]
        si.games += 1
        si.vp_sum += result.vp[i]
        si.rank_sum += ranks[i]
        si.seconds += result.wall_seconds / 4.0
        if i == result.winner:
            si.wins += 1

    for i in range(4):
        for j in range(i + 1, 4):
            vi = result.job.variants[i]
            vj = result.job.variants[j]
            ri = standings[vi].elo
            rj = standings[vj].elo
            expected_i = 1.0 / (1.0 + 10.0 ** ((rj - ri) / 400.0))
            if ranks[i] < ranks[j]:
                score_i = 1.0
            elif ranks[i] > ranks[j]:
                score_i = 0.0
            else:
                score_i = 0.5
            delta = k_factor * (score_i - expected_i)
            deltas[vi] += delta
            deltas[vj] -= delta

    for name, delta in deltas.items():
        standings[name].elo += delta


def leaderboard_rows(standings: dict[str, Standing]) -> list[tuple[str, Standing]]:
    return sorted(standings.items(), key=lambda item: (-item[1].elo, item[0]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mixed FFA Elo arena games across Catan variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--variants", default="core")
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--k-factor", type=float, default=8.0)
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

    try:
        variants = parse_variants(args.variants)
    except ValueError as exc:
        parser.error(str(exc))

    jobs = make_schedule(variants, args.games, args.seed)
    standings = {variant: Standing() for variant in variants}

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"elo_arena_{stamp}.log"

    with log_path.open("w", encoding="utf-8") as log:
        def emit(text: str) -> None:
            print(text, end="", flush=True)
            log.write(text)
            log.flush()

        emit(f"elo_arena log: {log_path}\n")
        emit(f"variants: {','.join(variants)}\n")
        emit(f"games={args.games} seed={args.seed} jobs={args.jobs}\n")

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

        completed = 0
        failures = 0
        start = time.monotonic()
        if args.jobs == 1:
            for job in jobs:
                result = run_one(args, job)
                if result.returncode != 0 or result.winner < 0:
                    failures += 1
                    emit(f"FAIL game={job.index} seed={job.seed} rc={result.returncode}\n")
                    emit(result.output)
                    continue
                update_elo(standings, result, args.k_factor)
                completed += 1
                if completed % max(1, args.games // 20) == 0 or completed == args.games:
                    leader = leaderboard_rows(standings)[0]
                    emit(
                        f"progress {completed}/{args.games} "
                        f"leader={leader[0]} elo={leader[1].elo:.1f}\n"
                    )
        else:
            with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                future_map = {pool.submit(run_one, args, job): job for job in jobs}
                for future in futures.as_completed(future_map):
                    job = future_map[future]
                    result = future.result()
                    if result.returncode != 0 or result.winner < 0:
                        failures += 1
                        emit(f"FAIL game={job.index} seed={job.seed} rc={result.returncode}\n")
                        emit(result.output)
                        continue
                    update_elo(standings, result, args.k_factor)
                    completed += 1
                    if completed % max(1, args.games // 20) == 0 or completed == args.games:
                        leader = leaderboard_rows(standings)[0]
                        emit(
                            f"progress {completed}/{args.games} "
                            f"leader={leader[0]} elo={leader[1].elo:.1f}\n"
                        )

        elapsed = time.monotonic() - start
        emit("\nleaderboard\n")
        emit(f"{'rank':>4} {'variant':18} {'elo':>8} {'games':>6} {'wins':>5} {'win%':>7} {'avgVP':>7} {'avgRank':>8}\n")
        for rank, (name, st) in enumerate(leaderboard_rows(standings), start=1):
            win_pct = 100.0 * st.wins / st.games if st.games else 0.0
            avg_vp = st.vp_sum / st.games if st.games else 0.0
            avg_rank = st.rank_sum / st.games if st.games else 0.0
            emit(
                f"{rank:4d} {name:18} {st.elo:8.1f} {st.games:6d} "
                f"{st.wins:5d} {win_pct:6.1f}% {avg_vp:7.2f} {avg_rank:8.2f}\n"
            )
        emit(f"\ncompleted={completed} failures={failures} wall={elapsed:.2f}s\n")
        emit(f"Log written to: {log_path}\n")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
