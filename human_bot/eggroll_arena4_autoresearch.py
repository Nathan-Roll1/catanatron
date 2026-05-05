#!/usr/bin/env python3
"""Fixed-arena EGGROLL autoresearch for the pure 0-ply M2 policy.

The objective is deliberately narrow and measurable:

    candidate vs incumbent vs M2 vs AB2

Every promotion is decided inside that exact 4-player arena.  AB2 is pinned at
Elo 1000, and the candidate must out-win M2, AB2, and the incumbent in the
same symmetric gate games before it can replace the incumbent.
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_bot import search_distill_m2 as sdm
from human_bot.eggroll_m2_prod import (
    _apply_noise,
    _export_state,
    _noise_tree,
    _parameter_keys,
    _update_state,
)
from human_bot.model import HumanBotNet


CSRC = ROOT / "csrc"
DEFAULT_WORK = ROOT / "autoresearch-results" / "eggroll_arena4"
DEFAULT_INCUMBENT_PT = ROOT / "autoresearch-results" / "search_distill_m2" / "kept_iter_0054.pt"
DEFAULT_INCUMBENT_BIN = ROOT / "autoresearch-results" / "search_distill_m2" / "kept_iter_0054.bin"
FALLBACK_INCUMBENT_PT = CSRC / "nn_weights_candidate.pt"
FALLBACK_INCUMBENT_BIN = CSRC / "nn_weights_candidate.bin"
MASK_DIM = sdm.MASK_DIM


@dataclass(frozen=True)
class EggrollSetup:
    name: str
    scope: str
    sigma: float
    lr: float
    rank: int = 1
    relative: bool = True


@dataclass
class CandidateSpec:
    label: str
    pt_path: Path
    bin_path: Path
    kind: str
    pair_idx: int | None = None
    sign: int | None = None
    noise_seed: int | None = None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _resolve_incumbent(pt: Path, bin_path: Path) -> tuple[Path, Path]:
    if pt.exists() and bin_path.exists():
        return pt.resolve(), bin_path.resolve()
    if FALLBACK_INCUMBENT_PT.exists() and FALLBACK_INCUMBENT_BIN.exists():
        return FALLBACK_INCUMBENT_PT.resolve(), FALLBACK_INCUMBENT_BIN.resolve()
    raise FileNotFoundError(
        f"No incumbent found. Tried {(pt, bin_path)} and "
        f"{(FALLBACK_INCUMBENT_PT, FALLBACK_INCUMBENT_BIN)}"
    )


def _schedule() -> list[EggrollSetup]:
    return [
        EggrollSetup("spatial_final_wide", "policy_head_spatial_final", sigma=0.070, lr=0.18),
        EggrollSetup("policy_final_wide", "policy_head_final", sigma=0.060, lr=0.16),
        EggrollSetup("policy_head_surgical", "policy_head", sigma=0.030, lr=0.08),
        EggrollSetup("trunk_policy_micro", "trunk_policy", sigma=0.010, lr=0.030),
        EggrollSetup("spatial_final_cool", "policy_head_spatial_final", sigma=0.040, lr=0.10),
        EggrollSetup("policy_final_cool", "policy_head_final", sigma=0.035, lr=0.09),
    ]


def _mask397(mask_raw: np.ndarray) -> np.ndarray:
    mask = np.zeros(MASK_DIM, dtype=np.float32)
    n = min(MASK_DIM, int(mask_raw.shape[0]))
    mask[:n] = mask_raw[:n]
    return mask


def _play_arena_game(seed: int, seat_labels: list[str]) -> dict[str, Any]:
    CatanGame = sdm._G["CatanGame"]
    ae = sdm._G["ae"]
    entries = sdm._G["entries"]

    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
    out = np.zeros(4 + MASK_DIM, dtype=np.float32)

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        cp = game.current_player()
        label = seat_labels[cp]
        entry = entries[label]

        if len(legal) == 1:
            chosen = 0
        elif entry.kind == "ab2":
            chosen = sdm._ab2_choose(game, legal)
        else:
            se.encode_into(game.get_state_view(), nf, ef, ff)
            mask = _mask397(ae.get_action_mask(legal).numpy())
            chosen = sdm._nn_choose(label, legal, se, nf, ef, ff, mask, out)
        game.step(chosen)

    winner = game.winner()
    return {
        "winner": seat_labels[winner] if winner is not None else None,
        "players": list(seat_labels),
        "turns": int(game.turn_number),
    }


def _arena_seats(candidate_label: str, seed_idx: int, rotation: int) -> list[str]:
    seats: list[str | None] = [None, None, None, None]
    seats[rotation] = candidate_label
    anchors = ["incumbent", "m2", "ab2"]
    shift = (seed_idx + rotation) % len(anchors)
    ordered = anchors[shift:] + anchors[:shift]
    remaining = [idx for idx in range(4) if idx != rotation]
    for seat, label in zip(remaining, ordered):
        seats[seat] = label
    return [str(label) for label in seats]


def _arena_job(job: tuple[str, int, int, int]) -> dict[str, Any]:
    candidate_label, seed_base, game_idx, rotation = job
    seed_idx = game_idx // 4
    seed = seed_base + seed_idx
    players = _arena_seats(candidate_label, seed_idx, rotation)
    row = _play_arena_game(seed, players)
    row.update({"candidate": candidate_label, "seed": seed, "game_idx": game_idx, "rotation": rotation})
    return row


def _bootstrap_elo(
    results: list[dict[str, Any]],
    labels: list[str],
    bootstraps: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if bootstraps <= 0 or not results:
        return {}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {label: [] for label in labels}
    n = len(results)
    for _ in range(bootstraps):
        sample = [results[int(i)] for i in rng.integers(0, n, size=n)]
        try:
            elos = sdm._fit_bt_elo(sample, labels, pinned_label="ab2")
        except Exception:
            continue
        for label in labels:
            values[label].append(float(elos[label]))
    ci: dict[str, tuple[float, float]] = {}
    for label, vals in values.items():
        if vals:
            ci[label] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return ci


def _summarize_candidate(
    candidate_label: str,
    rows: list[dict[str, Any]],
    bootstraps: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    labels = [candidate_label, "incumbent", "m2", "ab2"]
    elos = sdm._fit_bt_elo(rows, labels, pinned_label="ab2")
    wins = collections.Counter(row["winner"] for row in rows if row["winner"] is not None)
    played = len(rows)
    winrates = {label: float(wins.get(label, 0) / max(1, played)) for label in labels}
    seat_wins: dict[str, list[int]] = {label: [0, 0, 0, 0] for label in labels}
    seat_counts: dict[str, list[int]] = {label: [0, 0, 0, 0] for label in labels}
    total_turns = 0
    for row in rows:
        total_turns += int(row.get("turns", 0))
        for seat, label in enumerate(row["players"]):
            if label in seat_counts:
                seat_counts[label][seat] += 1
        winner = row["winner"]
        if winner in seat_wins:
            seat = row["players"].index(winner)
            seat_wins[winner][seat] += 1
    ci = _bootstrap_elo(rows, labels, bootstraps, bootstrap_seed)
    return {
        "candidate": candidate_label,
        "games": played,
        "wins": {label: int(wins.get(label, 0)) for label in labels},
        "winrates": winrates,
        "elos": {label: float(elos[label]) for label in labels},
        "elo_ci": {label: [lo, hi] for label, (lo, hi) in ci.items()},
        "seat_wins": seat_wins,
        "seat_counts": seat_counts,
        "avg_turns": total_turns / max(1, played),
    }


def evaluate_arena_candidates(
    candidates: list[CandidateSpec],
    incumbent_bin: Path,
    games_per_candidate: int,
    workers: int,
    seed_base: int,
    ab_depth: int,
    batch_size: int,
    bootstraps: int = 0,
) -> list[dict[str, Any]]:
    if games_per_candidate < 4:
        raise ValueError("games_per_candidate must be at least 4")
    anchor_entries = [
        sdm.LeagueEntry("incumbent", "nn", str(incumbent_bin.resolve()), None),
        sdm.LeagueEntry("m2", "nn", str((CSRC / "nn_weights_m2.bin").resolve()), None),
        sdm.LeagueEntry("ab2", "ab2"),
    ]
    summaries: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        entries = list(anchor_entries)
        for cand in batch:
            entries.append(sdm.LeagueEntry(cand.label, "nn", str(cand.bin_path.resolve()), str(cand.pt_path.resolve())))
        roster_json = json.dumps([asdict(e) for e in entries])
        jobs: list[tuple[str, int, int, int]] = []
        for cand in batch:
            for game_idx in range(games_per_candidate):
                jobs.append((cand.label, seed_base, game_idx, game_idx % 4))
        with ctx.Pool(
            processes=max(1, int(workers)),
            initializer=sdm._init_worker,
            initargs=(roster_json, int(ab_depth), 0, 0.0),
        ) as pool:
            rows = list(pool.imap_unordered(_arena_job, jobs, chunksize=1))
        by_candidate: dict[str, list[dict[str, Any]]] = {cand.label: [] for cand in batch}
        for row in rows:
            by_candidate[row["candidate"]].append(row)
        for cand in batch:
            summaries.append(
                _summarize_candidate(
                    cand.label,
                    by_candidate[cand.label],
                    bootstraps=bootstraps,
                    bootstrap_seed=seed_base + batch_start,
                )
            )
    summaries.sort(key=lambda row: row["elos"][row["candidate"]], reverse=True)
    return summaries


def _export_candidate(
    incumbent_pt: Path,
    state: dict[str, torch.Tensor],
    pt_path: Path,
    bin_path: Path,
    metadata: dict[str, Any],
    weight_format: str,
) -> None:
    _export_state(incumbent_pt, state, pt_path, bin_path, metadata, weight_format)


def _make_candidates(
    incumbent_pt: Path,
    out_dir: Path,
    iteration: int,
    setup: EggrollSetup,
    pairs: int,
    sigma_scale: float,
    weight_format: str,
    seed: int,
) -> tuple[list[CandidateSpec], list[dict[str, torch.Tensor]], dict[str, torch.Tensor], list[str], float, float]:
    net = HumanBotNet.load_checkpoint(str(incumbent_pt), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    keys = _parameter_keys(net, setup.scope)
    if not keys:
        raise RuntimeError(f"No parameters selected for scope {setup.scope}")

    sigma = setup.sigma * sigma_scale
    lr = setup.lr * sigma_scale
    cand_dir = out_dir / f"iter_{iteration:04d}" / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[CandidateSpec] = []
    noises: list[dict[str, torch.Tensor]] = []
    for pair_idx in range(pairs):
        noise_seed = seed + pair_idx * 104729
        noise = _noise_tree(base_state, keys, setup.rank, noise_seed, setup.relative)
        noises.append(noise)
        for sign in (1, -1):
            label = f"cand_{pair_idx:04d}_{'p' if sign > 0 else 'm'}"
            pt_path = cand_dir / f"{label}.pt"
            bin_path = cand_dir / f"{label}.bin"
            state = _apply_noise(base_state, noise, sigma, float(sign))
            _export_candidate(
                incumbent_pt,
                state,
                pt_path,
                bin_path,
                {
                    "source": "eggroll_arena4",
                    "iteration": iteration,
                    "setup": asdict(setup),
                    "sigma": sigma,
                    "lr": lr,
                    "pair_idx": pair_idx,
                    "sign": sign,
                    "noise_seed": noise_seed,
                },
                weight_format,
            )
            candidates.append(CandidateSpec(label, pt_path, bin_path, "perturb", pair_idx, sign, noise_seed))
    return candidates, noises, base_state, keys, sigma, lr


def _pair_diffs(proxy_rows: list[dict[str, Any]], pairs: int) -> list[float]:
    by_label = {row["candidate"]: row for row in proxy_rows}
    diffs: list[float] = []
    for pair_idx in range(pairs):
        plus = by_label.get(f"cand_{pair_idx:04d}_p")
        minus = by_label.get(f"cand_{pair_idx:04d}_m")
        if plus is None or minus is None:
            diffs.append(0.0)
            continue
        p_elo = plus["elos"][plus["candidate"]]
        m_elo = minus["elos"][minus["candidate"]]
        diffs.append(float((p_elo - m_elo) / 400.0))
    return diffs


def _export_es_update(
    incumbent_pt: Path,
    out_dir: Path,
    iteration: int,
    setup: EggrollSetup,
    base_state: dict[str, torch.Tensor],
    noises: list[dict[str, torch.Tensor]],
    pair_diffs: list[float],
    keys: list[str],
    sigma: float,
    lr: float,
    weight_format: str,
) -> CandidateSpec:
    state = _update_state(
        base_state,
        noises,
        pair_diffs,
        keys,
        lr=lr,
        sigma=sigma,
        update_clip_rms=0.025,
    )
    label = "es_update"
    pt_path = out_dir / f"iter_{iteration:04d}" / f"{label}.pt"
    bin_path = out_dir / f"iter_{iteration:04d}" / f"{label}.bin"
    _export_candidate(
        incumbent_pt,
        state,
        pt_path,
        bin_path,
        {
            "source": "eggroll_arena4_es_update",
            "iteration": iteration,
            "setup": asdict(setup),
            "sigma": sigma,
            "lr": lr,
            "pair_diffs": pair_diffs[:128],
        },
        weight_format,
    )
    return CandidateSpec(label, pt_path, bin_path, "es_update")


def _promoted(gate: dict[str, Any], min_elo_delta: float) -> tuple[bool, str]:
    cand = gate["candidate"]
    wins = gate["wins"]
    elos = gate["elos"]
    if wins[cand] <= wins["incumbent"]:
        return False, f"candidate wins {wins[cand]} <= incumbent wins {wins['incumbent']}"
    if wins[cand] <= wins["m2"]:
        return False, f"candidate wins {wins[cand]} <= m2 wins {wins['m2']}"
    if wins[cand] <= wins["ab2"]:
        return False, f"candidate wins {wins[cand]} <= ab2 wins {wins['ab2']}"
    for anchor in ("incumbent", "m2", "ab2"):
        if elos[cand] <= elos[anchor] + min_elo_delta:
            return (
                False,
                f"candidate Elo {elos[cand]:.1f} <= {anchor} Elo "
                f"{elos[anchor]:.1f} + {min_elo_delta:.1f}",
            )
    return True, "positive WR and Elo lead vs all anchors"


def _short_row(row: dict[str, Any]) -> str:
    cand = row["candidate"]
    ci = row.get("elo_ci", {}).get(cand)
    ci_text = "" if not ci else f" [{ci[0]:.1f}, {ci[1]:.1f}]"
    wr = row["winrates"]
    wins = row["wins"]
    return (
        f"{cand}: Elo {row['elos'][cand]:.1f}{ci_text}; "
        f"wins {wins[cand]}/{row['games']} "
        f"(inc {wins['incumbent']}, m2 {wins['m2']}, ab2 {wins['ab2']}); "
        f"WR cand {wr[cand]:.3f}, inc {wr['incumbent']:.3f}, "
        f"m2 {wr['m2']:.3f}, ab2 {wr['ab2']:.3f}"
    )


def _copy_keep(best: CandidateSpec, work_dir: Path, iteration: int) -> tuple[Path, Path]:
    kept_pt = work_dir / f"kept_iter_{iteration:04d}.pt"
    kept_bin = work_dir / f"kept_iter_{iteration:04d}.bin"
    shutil.copy2(best.pt_path, kept_pt)
    shutil.copy2(best.bin_path, kept_bin)
    return kept_pt.resolve(), kept_bin.resolve()


def _maybe_update_live_candidate(pt_path: Path, bin_path: Path) -> None:
    if (CSRC / "nn_weights_candidate.pt").exists():
        shutil.copy2(pt_path, CSRC / "nn_weights_candidate.pt")
    # The .bin is ignored but useful for local C/browser parity.
    shutil.copy2(bin_path, CSRC / "nn_weights_candidate.bin")


def run_loop(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "arena4_log.jsonl"
    incumbent_pt, incumbent_bin = _resolve_incumbent(Path(args.incumbent_pt), Path(args.incumbent_bin))
    setups = _schedule()
    consecutive_discards = 0
    sigma_scale = float(args.sigma_scale)

    print(f"[arena4] incumbent_pt={incumbent_pt}", flush=True)
    print(f"[arena4] incumbent_bin={incumbent_bin}", flush=True)
    print("[arena4] Elo anchor: AB2 fixed at 1000.0", flush=True)

    for iteration in range(int(args.start_iteration), int(args.start_iteration) + int(args.max_iterations)):
        setup = setups[iteration % len(setups)]
        if consecutive_discards >= args.decay_after_discards:
            sigma_scale = max(args.min_sigma_scale, sigma_scale * args.decay_factor)
            consecutive_discards = 0
            print(f"[arena4] decay event: sigma/lr scale -> {sigma_scale:.4f}", flush=True)

        t0 = time.time()
        seed = int(args.seed_base) + iteration * 1_000_000
        print(
            f"\n[arena4] iter={iteration:04d} setup={setup.name} "
            f"scope={setup.scope} pairs={args.pairs} scale={sigma_scale:.4f}",
            flush=True,
        )
        candidates, noises, base_state, keys, sigma, lr = _make_candidates(
            incumbent_pt,
            work_dir,
            iteration,
            setup,
            int(args.pairs),
            sigma_scale,
            args.weight_format,
            seed,
        )
        proxy = evaluate_arena_candidates(
            candidates,
            incumbent_bin,
            games_per_candidate=int(args.proxy_games),
            workers=int(args.workers),
            seed_base=seed + 101_000,
            ab_depth=int(args.ab_depth),
            batch_size=int(args.candidate_batch_size),
        )
        top_proxy = proxy[: max(1, int(args.top_k))]
        print("[arena4] proxy top:", flush=True)
        for row in top_proxy[: min(5, len(top_proxy))]:
            print("  " + _short_row(row), flush=True)

        pair_diffs = _pair_diffs(proxy, int(args.pairs))
        es_candidate = _export_es_update(
            incumbent_pt,
            work_dir,
            iteration,
            setup,
            base_state,
            noises,
            pair_diffs,
            keys,
            sigma,
            lr,
            args.weight_format,
        )
        by_label = {cand.label: cand for cand in candidates}
        rerank_candidates = [by_label[row["candidate"]] for row in top_proxy if row["candidate"] in by_label]
        rerank_candidates.append(es_candidate)
        rerank = evaluate_arena_candidates(
            rerank_candidates,
            incumbent_bin,
            games_per_candidate=int(args.rerank_games),
            workers=int(args.workers),
            seed_base=seed + 202_000,
            ab_depth=int(args.ab_depth),
            batch_size=min(int(args.candidate_batch_size), len(rerank_candidates)),
        )
        print("[arena4] rerank top:", flush=True)
        for row in rerank[: min(5, len(rerank))]:
            print("  " + _short_row(row), flush=True)

        best_row = rerank[0]
        best_label = best_row["candidate"]
        best_spec = next(c for c in rerank_candidates if c.label == best_label)
        gate = evaluate_arena_candidates(
            [best_spec],
            incumbent_bin,
            games_per_candidate=int(args.gate_games),
            workers=int(args.workers),
            seed_base=seed + 303_000,
            ab_depth=int(args.ab_depth),
            batch_size=1,
            bootstraps=int(args.gate_bootstraps),
        )[0]
        accept, reason = _promoted(gate, float(args.min_elo_delta))
        print(f"[arena4] gate: {_short_row(gate)}", flush=True)
        print(f"[arena4] decision={'KEEP' if accept else 'discard'} reason={reason}", flush=True)

        kept_pt = None
        kept_bin = None
        if accept:
            kept_pt, kept_bin = _copy_keep(best_spec, work_dir, iteration)
            incumbent_pt, incumbent_bin = kept_pt, kept_bin
            if args.update_live_candidate:
                _maybe_update_live_candidate(kept_pt, kept_bin)
            consecutive_discards = 0
            sigma_scale = min(1.0, sigma_scale / max(1e-9, float(args.decay_factor)))
        else:
            consecutive_discards += 1

        row = {
            "event": "iteration",
            "iteration": iteration,
            "setup": asdict(setup),
            "sigma": sigma,
            "lr": lr,
            "sigma_scale": sigma_scale,
            "pairs": int(args.pairs),
            "proxy_games": int(args.proxy_games),
            "rerank_games": int(args.rerank_games),
            "gate_games": int(args.gate_games),
            "proxy_top": top_proxy[:10],
            "rerank": rerank,
            "gate": gate,
            "accepted": accept,
            "reason": reason,
            "kept_pt": kept_pt,
            "kept_bin": kept_bin,
            "elapsed_sec": time.time() - t0,
        }
        _write_jsonl(log_path, row)

        if args.cleanup_candidates:
            iter_dir = work_dir / f"iter_{iteration:04d}"
            if iter_dir.exists():
                shutil.rmtree(iter_dir)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK))
    parser.add_argument("--incumbent-pt", default=str(DEFAULT_INCUMBENT_PT))
    parser.add_argument("--incumbent-bin", default=str(DEFAULT_INCUMBENT_BIN))
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--proxy-games", type=int, default=48, help="Symmetric arena games per candidate.")
    parser.add_argument("--rerank-games", type=int, default=96, help="Symmetric arena games for top candidates.")
    parser.add_argument("--gate-games", type=int, default=100, help="Promotion gate games for the best candidate.")
    parser.add_argument("--gate-bootstraps", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    parser.add_argument("--candidate-batch-size", type=int, default=24)
    parser.add_argument("--ab-depth", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=1_431_000_000)
    parser.add_argument("--weight-format", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--decay-after-discards", type=int, default=3)
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--min-sigma-scale", type=float, default=0.125)
    parser.add_argument("--min-elo-delta", type=float, default=0.0)
    parser.add_argument("--update-live-candidate", action="store_true")
    parser.add_argument("--cleanup-candidates", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_loop(args)


if __name__ == "__main__":
    main()
