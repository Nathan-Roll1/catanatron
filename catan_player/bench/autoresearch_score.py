#!/usr/bin/env python3
"""Scalar benchmark wrapper for codex-autoresearch.

Runs a small fixed-seed mixed arena and prints one higher-is-better score as
the final line so the autoresearch helper can track progress mechanically.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PLAYER_DIR = SCRIPT_PATH.parent.parent
BUILD_SCRIPT = PLAYER_DIR / "build.sh"
ARENA = PLAYER_DIR / "bench" / "elo_arena.py"

ROW_RE = re.compile(
    r"^\s*\d+\s+(?P<variant>\S+)\s+(?P<elo>[0-9.]+)\s+"
    r"(?P<games>\d+)\s+(?P<wins>\d+)"
)
WALL_RE = re.compile(r"^completed=\d+ failures=(?P<failures>\d+) wall=(?P<wall>[0-9.]+)s")


def run(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=str(PLAYER_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the small deterministic Catan strength/speed benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260439)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--penalty", type=float, default=0.5)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if not args.skip_build:
        rc, out = run([str(BUILD_SCRIPT)])
        sys.stdout.write(out)
        if rc != 0:
            return rc or 1

    command = [
        str(ARENA),
        "--skip-build",
        "--variants",
        "default,leaf14,k7,pvs,h-s,ab2",
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--jobs",
        str(args.jobs),
    ]
    rc, out = run(command)
    sys.stdout.write(out)
    if rc != 0:
        return rc or 1

    default_elo = None
    wall = None
    failures = None
    for line in out.splitlines():
        m = ROW_RE.match(line)
        if m and m.group("variant") == "default":
            default_elo = float(m.group("elo"))
            continue
        m = WALL_RE.match(line)
        if m:
            failures = int(m.group("failures"))
            wall = float(m.group("wall"))

    if default_elo is None or wall is None or failures is None:
        print("ERROR: could not parse default Elo/wall from benchmark output", file=sys.stderr)
        return 2
    if failures != 0:
        print(f"ERROR: benchmark failures={failures}", file=sys.stderr)
        return 2

    score = default_elo - args.penalty * wall
    print(f"{score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
