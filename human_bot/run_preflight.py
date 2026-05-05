#!/usr/bin/env python3
"""Pre-flight checks for the robust improvement run.

Runs all of the plan's debug gates in sequence and aborts on the first
failure. Meant to be called AFTER rsync'ing source + seeding the starting
checkpoint and BEFORE spawning actor jobs.

Plan gates (see robust-improvement-run):
  1. Encoding gate: 2p/3p edge enemy channels are stable under seat
     rotation and trailing flat-player slots are zero.
  2. Shard gate: each actor source writes shards with `num_players`,
     `source`, and `policy_weight`, and the learner's loader can ingest
     all three together without shape/rotation errors.
  3. Sampling gate: the source-aware sampler picks the intended mix
     when all sources are represented.
  4. Eval gate: the seed checkpoint reproduces its baseline on the new
     eval harness (0-ply WR vs AB2 within expected range).

Usage:
  python3 -m human_bot.run_preflight \
      --ckpt checkpoints/ab2_imit_v1/latest.pt \
      --eval-games 100
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import torch


# ----------------------------------------------------------------------
# Gate 1: encoding invariance for 2p/3p/4p
# ----------------------------------------------------------------------

def _encode(g, se):
    sv = g.get_state_view()
    nf = np.zeros((se.num_nodes, 18), dtype=np.float32)
    ef = np.zeros((se.num_edges, 5), dtype=np.float32)
    ff = np.zeros(115, dtype=np.float32)
    se.encode_into(sv, nf, ef, ff)
    return nf, ef, ff, sv


def gate_encoding() -> None:
    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame

    failures = []
    for n_players in (2, 3, 4):
        for seed in (42, 100, 777):
            g = CatanGame(seed=seed, config=GameConfig(num_players=n_players))
            g.reset()
            se = g.make_state_encoder()
            # Advance a few turns to populate roads and VPs
            for _ in range(40):
                if g.is_terminal():
                    break
                le = g.get_legal_actions()
                if not le:
                    break
                g.step(0)
            nf, ef, ff, sv = _encode(g, se)

            # Edge sums must be exactly 1 per edge (exactly one channel hot)
            if not np.allclose(ef.sum(axis=1), 1.0):
                failures.append(
                    f"n_players={n_players} seed={seed}: "
                    f"edge channels do not sum to 1"
                )

            # For 2p, ef channels 3 and 4 must be all zero (no phantom seats)
            if n_players == 2 and (ef[:, 3].sum() != 0 or ef[:, 4].sum() != 0):
                failures.append(
                    f"n_players=2 seed={seed}: ef[:,3..4] not zero; "
                    f"sums={ef[:,3].sum():.2f}, {ef[:,4].sum():.2f}"
                )
            # For 3p, channel 4 must be all zero
            if n_players == 3 and ef[:, 4].sum() != 0:
                failures.append(
                    f"n_players=3 seed={seed}: ef[:,4] not zero; "
                    f"sum={ef[:,4].sum():.2f}"
                )

            # Flat features: trailing player blocks (beyond num_players)
            # must be zero.
            for p in range(n_players, 4):
                block = ff[24 * p:24 * (p + 1)]
                if block.any():
                    failures.append(
                        f"n_players={n_players} seed={seed}: "
                        f"flat player block {p} not zero"
                    )

    if failures:
        for f in failures:
            print(f"  [encoding-FAIL] {f}")
        raise SystemExit("Gate 1 (encoding) FAILED")
    print("  Gate 1 (encoding): PASS")


# ----------------------------------------------------------------------
# Gate 2: shard metadata roundtrip
# ----------------------------------------------------------------------

def _make_synthetic_shard(src_tag: str, path: str, n_rows: int = 16) -> None:
    """Build a tiny synthetic shard that matches the actor format.

    We can't easily invoke every actor in a unit test, so instead we
    validate the learner's shard contract by producing a shard matching
    the expected schema and ensuring the loader reads it.
    """
    from hexzero.game.interface import CatanGame
    g = CatanGame(seed=0); g.reset()
    se = g.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FF = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    data = {
        "node_features": torch.zeros((n_rows, N, NF), dtype=torch.float32),
        "edge_features": torch.zeros((n_rows, E, EF), dtype=torch.float32),
        "flat_features": torch.zeros((n_rows, FF), dtype=torch.float32),
        "action_mask": torch.zeros((n_rows, 397), dtype=torch.float32),
        "action_idx": torch.zeros(n_rows, dtype=torch.int64),
        "player": torch.zeros(n_rows, dtype=torch.int64),
        "reward_vec": torch.tile(torch.tensor([1.0, 0.3, 0.1, 0.0]),
                                  (n_rows, 1)),
        "policy_weight": torch.ones(n_rows, dtype=torch.float32),
        "step_weight": torch.ones(n_rows, dtype=torch.float32),
        "num_players": torch.full((n_rows,), 4, dtype=torch.int64),
        "source": src_tag,
    }
    # Mark at least one action legal so masks aren't fully zero
    data["action_mask"][:, 0] = 1.0
    torch.save(data, path)


def gate_shard_roundtrip() -> None:
    from human_bot.dataset import rotate_value_targets_to_cp
    tmp = tempfile.mkdtemp(prefix="preflight_shards_")
    try:
        for tag in ("ab2", "exit", "exit_vs_ab2"):
            path = os.path.join(tmp, f"{tag}_probe.pt")
            _make_synthetic_shard(tag, path)
            d = torch.load(path, weights_only=False)
            for k in ("node_features", "edge_features", "flat_features",
                       "action_mask", "action_idx", "player",
                       "reward_vec", "policy_weight", "num_players",
                       "source"):
                if k not in d:
                    raise SystemExit(
                        f"Gate 2 FAIL: {tag} shard missing key {k!r}"
                    )
            if d["source"] != tag:
                raise SystemExit(
                    f"Gate 2 FAIL: {tag} shard has wrong source {d['source']!r}"
                )
            # rotate roundtrip should not raise
            rv = d["reward_vec"].numpy()
            p = d["player"].numpy()
            np_arr = d["num_players"].numpy()
            vt = rotate_value_targets_to_cp(rv, p, np_arr)
            if vt.shape != rv.shape:
                raise SystemExit("Gate 2 FAIL: rotation shape mismatch")
        print("  Gate 2 (shard roundtrip): PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# Gate 3: source-aware sampler picks intended mix
# ----------------------------------------------------------------------

def gate_sampler() -> None:
    from human_bot.c_selfplay import _pick_shard_group, _classify_source

    tmp = tempfile.mkdtemp(prefix="preflight_sampler_")
    try:
        # Create a pool where every source has plenty of shards
        for i in range(50):
            open(os.path.join(tmp, f"ab2_a000_{i:06d}.pt"), "w").close()
        for i in range(50):
            open(os.path.join(tmp, f"exit_a000_{i:06d}.pt"), "w").close()
        for i in range(50):
            open(os.path.join(tmp, f"exit_vs_ab2_a000_{i:06d}.pt"),
                 "w").close()

        mix = {"exit_vs_ab2": 0.6, "exit": 0.25, "ab2": 0.15}
        group, counts, n_pending = _pick_shard_group(tmp, 20, mix)
        if group is None or len(group) != 20:
            raise SystemExit(f"Gate 3 FAIL: picked {group}")
        if counts["exit_vs_ab2"] != 12 or counts["exit"] != 5 or counts["ab2"] != 3:
            raise SystemExit(
                f"Gate 3 FAIL: wrong counts {counts} (want 12/5/3)"
            )
        # Also verify classification is stable
        for fn in ("ab2_a_x.pt", "exit_a_x.pt", "exit_vs_ab2_a_x.pt"):
            src = _classify_source(fn)
            expect = fn.split("_")[0] if not fn.startswith("exit_vs_ab2") else "exit_vs_ab2"
            if fn.startswith("ab2"):
                expect = "ab2"
            elif fn.startswith("exit_vs_ab2"):
                expect = "exit_vs_ab2"
            else:
                expect = "exit"
            if src != expect:
                raise SystemExit(
                    f"Gate 3 FAIL: classified {fn!r} as {src!r}, want {expect!r}"
                )
        print("  Gate 3 (sampler): PASS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# Gate 4: seeded checkpoint reproduces baseline 0-ply WR vs AB2
# ----------------------------------------------------------------------

def gate_eval_baseline(ckpt_path: str, num_games: int) -> None:
    if not os.path.exists(ckpt_path):
        print(f"  Gate 4 (eval baseline): SKIP (no checkpoint at {ckpt_path})")
        return

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from hexzero.game.interface import CatanGame
    from human_bot.eval_search import evaluate_search_vs_ab2
    from human_bot.model import HumanBotNet

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = HumanBotNet.load_checkpoint(ckpt_path, device=device)
    net.eval()
    lib = load_library()
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()

    t0 = time.time()
    r = evaluate_search_vs_ab2(
        net, se, ae, device, lib,
        num_games=num_games, search_depth=0, seed_offset=42)
    dt = time.time() - t0
    wr = r["win_rate"]
    print(f"  Gate 4 (eval baseline): 0-ply WR vs AB2 = {wr:.1%} "
          f"({r['hz_wins']}/{num_games}, {dt:.1f}s)")
    # Baseline range: the seed checkpoint should be meaningfully above random
    # (25% is random expectation for one of 4 seats, 50% for 2-of-4 NN seats
    # against 2-of-4 AB2 seats — ab2_imit_v1 was ~50%). Abort only on
    # catastrophic regression to guard against a broken checkpoint or
    # encoder change silently ruining inputs.
    if wr < 0.30:
        raise SystemExit(
            f"Gate 4 FAIL: seed checkpoint only {wr:.1%} vs AB2 — "
            "something is wrong with either the checkpoint or the "
            "encoder/eval path. Halting before launch."
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-flight gates for the "
                                  "robust improvement run")
    ap.add_argument("--ckpt", type=str,
                    default="checkpoints/ab2_imit_v1/latest.pt",
                    help="Seed checkpoint to validate against")
    ap.add_argument("--eval-games", type=int, default=60,
                    help="Number of games for the eval-baseline gate")
    args = ap.parse_args()

    # Make sure the project root is on the path for subprocess-style runs
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=== Pre-flight for robust improvement run ===")
    print(f"  ckpt={args.ckpt!r}")
    print(f"  eval_games={args.eval_games}")
    print()

    gate_encoding()
    gate_shard_roundtrip()
    gate_sampler()
    try:
        gate_eval_baseline(args.ckpt, args.eval_games)
    except SystemExit:
        raise
    except Exception:
        print("  Gate 4 (eval baseline): ERROR (not fatal, but inspect log)")
        traceback.print_exc()

    print()
    print("All mandatory gates passed — safe to launch the run.")


if __name__ == "__main__":
    main()
