"""Compile the C engine into a shared library for FFI use."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

_BINDINGS_DIR = Path(__file__).resolve().parent
_CSRC_DIR = _BINDINGS_DIR.parent.parent / "csrc"
_LIB_DIR = _BINDINGS_DIR / "lib"

CORE_SOURCES = [
    "rng.c",
    "map.c",
    "board.c",
    "state.c",
    "actions.c",
    "apply_action.c",
    "game.c",
    "value.c",
    "search.c",
]

CFLAGS = ["-O3", "-march=native", "-flto", "-fPIC"]


def _lib_name() -> str:
    return "libcatan.dylib" if platform.system() == "Darwin" else "libcatan.so"


def _shared_flag() -> str:
    return "-dynamiclib" if platform.system() == "Darwin" else "-shared"


def _find_compiler() -> str:
    for cc in ("cc", "gcc", "clang"):
        try:
            subprocess.run(
                [cc, "--version"], capture_output=True, check=True
            )
            return cc
        except FileNotFoundError:
            continue
    raise RuntimeError("No C compiler found — install gcc or clang.")


def build(*, csrc_dir: Path | None = None, output_dir: Path | None = None) -> Path:
    """Compile core C sources into a shared library.

    Returns the path to the built library.
    """
    csrc = Path(csrc_dir) if csrc_dir else _CSRC_DIR
    lib_dir = Path(output_dir) if output_dir else _LIB_DIR
    lib_dir.mkdir(parents=True, exist_ok=True)

    sources: list[str] = []
    for name in CORE_SOURCES:
        src = csrc / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing source file: {src}")
        sources.append(str(src))

    output = lib_dir / _lib_name()
    cc = _find_compiler()

    cmd = [
        cc,
        _shared_flag(),
        *CFLAGS,
        "-I", str(csrc),
        *sources,
        "-o", str(output),
        "-lm",
    ]

    print(f"[build_lib] Compiling {_lib_name()} with {cc} ...")
    print(f"[build_lib] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Compilation failed (exit {result.returncode}). "
            f"See stderr above for details."
        )
    if result.stderr:
        # Warnings are useful but not fatal
        print(result.stderr, file=sys.stderr)

    size_kb = output.stat().st_size / 1024
    print(f"[build_lib] Built {output} ({size_kb:.0f} KB)")
    return output


if __name__ == "__main__":
    build()
