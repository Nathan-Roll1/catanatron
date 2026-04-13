"""Merge self-play game files into the shared replay buffer.

Scans an output directory for .pt game files, loads each, and pushes
the examples into the central replay buffer with file locking.

Usage:
    python -m hexzero.scripts.merge_buffer \
        --source /scr-ssd/hexazero_sp_12345 \
        --target /nlp/scr/user/hexazero/replay_buffer/buffer.pt
"""

from __future__ import annotations

import argparse
import fcntl
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("merge_buffer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge game data into replay buffer")
    parser.add_argument("--source", type=str, required=True,
                        help="Directory with .pt game files from self-play")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to shared replay buffer file")
    parser.add_argument("--capacity", type=int, default=1_000_000,
                        help="Buffer capacity if creating new")
    args = parser.parse_args()

    from hexzero.selfplay.replay_buffer import ReplayBuffer

    source = Path(args.source)
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)

    game_files = sorted(source.glob("game_*.pt"))
    if not game_files:
        log.info("No game files found in %s", source)
        return

    lock_path = target.with_suffix(".lock")

    with open(lock_path, "w") as lock_file:
        log.info("Acquiring lock on %s ...", lock_path)
        fcntl.flock(lock_file, fcntl.LOCK_EX)

        try:
            if target.exists():
                buf = ReplayBuffer.load(str(target))
                log.info("Loaded existing buffer: %d positions", len(buf))
            else:
                buf = ReplayBuffer(capacity=args.capacity)
                log.info("Created new buffer (capacity=%d)", args.capacity)

            total = 0
            for gf in game_files:
                examples = torch.load(gf, weights_only=False, map_location="cpu")
                buf.push_game(examples)
                total += len(examples)

            buf.save(str(target))
            log.info("Merged %d files (%d examples) -> buffer now has %d positions",
                     len(game_files), total, len(buf))

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
