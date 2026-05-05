"""Production-oriented Eggroll-style evolution for pure M2 0-ply.

This keeps deployment as one ordinary M2 forward pass.  Training is black-box:
sample low-rank antithetic perturbations of selected policy tensors, export each
candidate to the C inference binary format, evaluate all candidates in one
many-model game pool, then update the current mean in the direction of better
fitness.

The script is intentionally local-machine friendly:
  - defaults to all CPU cores for game evaluation
  - evaluates full populations in a single worker pool
  - uses paired seat swaps for the 2v2 proxy
  - can blend 2v2 and 1v3 FFA fitness
  - writes JSONL logs for every generation and promotion gate
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from human_bot.eval_1v3_many_nn_fast import run as run_many_ffa
from human_bot.eval_2v2_many_nn_paired_fast import run as run_many_paired2v2
from human_bot.export_nn import export as export_checkpoint
from human_bot.model import HumanBotNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_CKPT = PROJECT_ROOT / "checkpoints" / "sp_latest2.pt"
DEFAULT_BASE_BIN = PROJECT_ROOT / "csrc" / "nn_weights_m2.bin"
DEFAULT_WORK_DIR = PROJECT_ROOT / "autoresearch-results" / "eggroll_m2_prod"
DEFAULT_OUT_BIN = DEFAULT_WORK_DIR / "latest.bin"
DEFAULT_OUT_PT = DEFAULT_WORK_DIR / "latest.pt"

FINAL_PREFIXES = (
    "policy_head.type_fc.3.",
    "policy_head.discard_yop_mono_fc.2.",
    "policy_head.maritime_fc.2.",
    "policy_head.trade_fc.2.",
    "policy_head.settlement_scorer.2.",
    "policy_head.city_scorer.2.",
    "policy_head.road_scorer.2.",
    "policy_head.robber_scorer.2.",
)

SPATIAL_FINAL_PREFIXES = (
    "policy_head.type_fc.3.",
    "policy_head.settlement_scorer.2.",
    "policy_head.city_scorer.2.",
    "policy_head.road_scorer.2.",
    "policy_head.robber_scorer.2.",
)

TYPE_FINAL_PREFIXES = ("policy_head.type_fc.3.",)

BUILD_FINAL_PREFIXES = (
    "policy_head.settlement_scorer.2.",
    "policy_head.city_scorer.2.",
    "policy_head.road_scorer.2.",
)

RESOURCE_ACTION_FINAL_PREFIXES = (
    "policy_head.discard_yop_mono_fc.2.",
    "policy_head.maritime_fc.2.",
    "policy_head.trade_fc.2.",
)


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _parameter_keys(net: HumanBotNet, scope: str) -> list[str]:
    names = [name for name, param in net.named_parameters() if torch.is_floating_point(param)]
    if scope == "policy_head_final":
        return [name for name in names if name.startswith(FINAL_PREFIXES)]
    if scope == "policy_head_spatial_final":
        return [name for name in names if name.startswith(SPATIAL_FINAL_PREFIXES)]
    if scope == "policy_type_final":
        return [name for name in names if name.startswith(TYPE_FINAL_PREFIXES)]
    if scope == "policy_build_final":
        return [name for name in names if name.startswith(BUILD_FINAL_PREFIXES)]
    if scope == "policy_resource_action_final":
        return [name for name in names if name.startswith(RESOURCE_ACTION_FINAL_PREFIXES)]
    if scope == "policy_road_final":
        return [name for name in names if name.startswith("policy_head.road_scorer.2.")]
    if scope == "policy_head":
        return [name for name in names if name.startswith("policy_head.")]
    if scope == "trunk":
        return [name for name in names if name.startswith("trunk.")]
    if scope == "trunk_policy":
        return [
            name for name in names
            if name.startswith("trunk.") or name.startswith("policy_head.")
        ]
    raise ValueError(f"unknown scope: {scope}")


def _low_rank_noise(t: torch.Tensor, rank: int, gen: torch.Generator,
                    relative: bool) -> torch.Tensor:
    tcpu = t.detach().cpu()
    if tcpu.ndim == 2 and min(tcpu.shape) > 1:
        out_dim, in_dim = tcpu.shape
        r = max(1, min(rank, out_dim, in_dim))
        a = torch.randn((out_dim, r), generator=gen, dtype=tcpu.dtype)
        b = torch.randn((r, in_dim), generator=gen, dtype=tcpu.dtype)
        noise = (a @ b) / math.sqrt(float(r * in_dim))
    else:
        noise = torch.randn(tcpu.shape, generator=gen, dtype=tcpu.dtype)
        noise = noise / math.sqrt(float(max(1, tcpu.numel())))
    if relative:
        rms = float(torch.sqrt(torch.mean(tcpu.float() * tcpu.float())).item())
        noise = noise * max(rms, 1e-3)
    return noise


def _noise_tree(state: dict[str, torch.Tensor], keys: Iterable[str], rank: int,
                seed: int, relative: bool) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return {
        key: _low_rank_noise(state[key], rank, gen, relative)
        for key in keys
    }


def _apply_noise(state: dict[str, torch.Tensor], noise: dict[str, torch.Tensor],
                 sigma: float, sign: float) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in state.items()}
    for key, eps in noise.items():
        out[key] = out[key] + float(sign * sigma) * eps.to(out[key].dtype)
    return out


def _update_state(mean_state: dict[str, torch.Tensor],
                  noises: list[dict[str, torch.Tensor]],
                  pair_diffs: list[float],
                  keys: list[str],
                  lr: float,
                  sigma: float,
                  update_clip_rms: float) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in mean_state.items()}
    if not noises:
        return out

    denom = max(1e-12, 2.0 * len(noises) * float(sigma))
    for key in keys:
        acc = torch.zeros_like(out[key])
        for noise, diff in zip(noises, pair_diffs):
            acc += float(diff) * noise[key].to(acc.dtype)
        step = float(lr) * acc / denom
        if update_clip_rms > 0:
            step_rms = float(torch.sqrt(torch.mean(step.float() * step.float())).item())
            base_rms = float(torch.sqrt(torch.mean(out[key].float() * out[key].float())).item())
            max_step_rms = float(update_clip_rms) * max(base_rms, 1e-3)
            if step_rms > max_step_rms and step_rms > 0:
                step = step * (max_step_rms / step_rms)
        out[key] = out[key] + step.to(out[key].dtype)
    return out


def _save_state(seed_checkpoint: Path, state: dict[str, torch.Tensor],
                checkpoint_path: Path, metadata: dict) -> None:
    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    net.load_state_dict(state, strict=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_checkpoint(str(checkpoint_path), metadata)


def _export_checkpoint(checkpoint_path: Path, bin_path: Path,
                       weight_format: str, quiet: bool = True) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            export_checkpoint(
                str(checkpoint_path),
                str(bin_path),
                weight_format=weight_format,
                write_test_vectors=False,
            )
    else:
        export_checkpoint(
            str(checkpoint_path),
            str(bin_path),
            weight_format=weight_format,
            write_test_vectors=False,
        )


def _export_state(seed_checkpoint: Path, state: dict[str, torch.Tensor],
                  checkpoint_path: Path, bin_path: Path, metadata: dict,
                  weight_format: str) -> None:
    _save_state(seed_checkpoint, state, checkpoint_path, metadata)
    _export_checkpoint(checkpoint_path, bin_path, weight_format=weight_format)


def _eval_population_batch(paths: list[Path], opponent_bin: Path, mode: str,
                           games: int, workers: int, seed_base: int,
                           mix_2v2_weight: float) -> tuple[list[dict], dict]:
    abs_paths = [str(p.resolve()) for p in paths]
    opp = str(opponent_bin.resolve())
    meta: dict = {"mode": mode, "games": games, "workers": workers, "seed_base": seed_base}

    if mode == "paired2v2":
        res = run_many_paired2v2(abs_paths, opp, games, workers, seed_base)
        rows = []
        for row in res["results"]:
            score = float(row["a_winrate"])
            rows.append({**row, "score": score, "paired2v2_winrate": score})
        meta["paired2v2"] = {k: v for k, v in res.items() if k != "results"}
        return rows, meta

    if mode == "ffa":
        res = run_many_ffa(abs_paths, opp, games, workers, seed_base)
        rows = []
        for row in res["results"]:
            wr = float(row["a_winrate"])
            score = 0.5 + 2.0 * (wr - 0.25)
            rows.append({**row, "score": score, "ffa_winrate": wr})
        meta["ffa"] = {k: v for k, v in res.items() if k != "results"}
        return rows, meta

    if mode != "mix":
        raise ValueError(f"unknown eval mode: {mode}")

    w2 = float(mix_2v2_weight)
    wf = 1.0 - w2
    paired = run_many_paired2v2(abs_paths, opp, games, workers, seed_base)
    ffa = run_many_ffa(abs_paths, opp, games, workers, seed_base + 500000)
    rows = []
    for p_row, f_row in zip(paired["results"], ffa["results"]):
        p_wr = float(p_row["a_winrate"])
        f_wr = float(f_row["a_winrate"])
        f_norm = 0.5 + 2.0 * (f_wr - 0.25)
        score = w2 * p_wr + wf * f_norm
        rows.append({
            **p_row,
            "score": score,
            "paired2v2_winrate": p_wr,
            "ffa_winrate": f_wr,
            "ffa_wins": int(f_row["a_wins"]),
            "ffa_games": int(f_row["games"]),
            "ffa_no_winner": int(f_row["no_winner"]),
        })
    meta["paired2v2"] = {k: v for k, v in paired.items() if k != "results"}
    meta["ffa"] = {k: v for k, v in ffa.items() if k != "results"}
    meta["mix_2v2_weight"] = w2
    return rows, meta


def _eval_population(paths: list[Path], opponent_bin: Path, mode: str,
                     games: int, workers: int, seed_base: int,
                     mix_2v2_weight: float,
                     candidate_batch_size: int = 0) -> tuple[list[dict], dict]:
    """Evaluate candidates, sharding model loads for wide populations.

    The many-candidate evaluator loads every candidate into every worker. That
    is fast for small populations but memory-heavy for wide Eggroll sweeps. A
    batch size keeps each worker's resident model set bounded while preserving
    identical seeds for every candidate.
    """
    if candidate_batch_size <= 0 or len(paths) <= candidate_batch_size:
        return _eval_population_batch(
            paths,
            opponent_bin,
            mode,
            games,
            workers,
            seed_base,
            mix_2v2_weight,
        )

    all_rows: list[dict] = []
    batches: list[dict] = []
    elapsed_total = 0.0
    for start in range(0, len(paths), candidate_batch_size):
        chunk = paths[start:start + candidate_batch_size]
        rows, meta = _eval_population_batch(
            chunk,
            opponent_bin,
            mode,
            games,
            workers,
            seed_base,
            mix_2v2_weight,
        )
        elapsed = 0.0
        if mode == "paired2v2":
            elapsed = float(meta.get("paired2v2", {}).get("elapsed_sec", 0.0))
        elif mode == "ffa":
            elapsed = float(meta.get("ffa", {}).get("elapsed_sec", 0.0))
        elif mode == "mix":
            elapsed = (
                float(meta.get("paired2v2", {}).get("elapsed_sec", 0.0))
                + float(meta.get("ffa", {}).get("elapsed_sec", 0.0))
            )
        elapsed_total += elapsed
        for row in rows:
            row = dict(row)
            row["candidate"] = int(row["candidate"]) + start
            all_rows.append(row)
        batches.append({
            "start": start,
            "count": len(chunk),
            "elapsed_sec": elapsed,
            "meta": meta,
        })

    meta = {
        "mode": mode,
        "games": games,
        "workers": workers,
        "seed_base": seed_base,
        "candidate_batch_size": candidate_batch_size,
        "candidates": len(paths),
        "batches": batches,
        "elapsed_sec": elapsed_total,
    }
    if mode == "paired2v2":
        meta["paired2v2"] = {
            "candidates": len(paths),
            "games_per_candidate": 2 * games,
            "games_per_side": games,
            "elapsed_sec": elapsed_total,
        }
    elif mode == "ffa":
        meta["ffa"] = {
            "candidates": len(paths),
            "elapsed_sec": elapsed_total,
        }
    elif mode == "mix":
        meta["mix_2v2_weight"] = mix_2v2_weight
    return all_rows, meta


def _score_baseline(mode: str, mix_2v2_weight: float) -> float:
    if mode == "ffa":
        return 0.5
    if mode == "mix":
        return float(mix_2v2_weight) * 0.5 + (1.0 - float(mix_2v2_weight)) * 0.5
    return 0.5


def _candidate_order_key(rec: dict) -> int:
    return int(rec.get("candidate_index", rec.get("candidate", 0)))


def _spread_bucket(records: list[dict], limit: int) -> list[dict]:
    """Return an even deterministic slice across a tied proxy-score bucket."""
    bucket = sorted(records, key=_candidate_order_key)
    if limit <= 0:
        return []
    if len(bucket) <= limit:
        return bucket
    if limit == 1:
        return [bucket[0]]

    selected: list[dict] = []
    used: set[int] = set()

    def add_nearest(idx: int) -> None:
        if len(selected) >= limit:
            return
        idx = max(0, min(len(bucket) - 1, idx))
        if idx not in used:
            used.add(idx)
            selected.append(bucket[idx])
            return
        for radius in range(1, len(bucket)):
            for alt in (idx - radius, idx + radius):
                if 0 <= alt < len(bucket) and alt not in used:
                    used.add(alt)
                    selected.append(bucket[alt])
                    return

    for slot in range(limit):
        idx = round(slot * (len(bucket) - 1) / (limit - 1))
        add_nearest(idx)
    return selected


def _dedupe_records(records: list[dict], limit: int) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for rec in records:
        path = str(rec["bin"])
        if path in seen:
            continue
        seen.add(path)
        unique.append(rec)

    buckets: dict[float, list[dict]] = {}
    for rec in unique:
        buckets.setdefault(round(float(rec["score"]), 9), []).append(rec)

    out: list[dict] = []
    for score in sorted(buckets, reverse=True):
        remaining = limit - len(out)
        if remaining <= 0:
            break
        out.extend(_spread_bucket(buckets[score], remaining))
    return out[:limit]


def _copy_top_candidates(records: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict] = []
    for idx, rec in enumerate(records):
        source = f"g{rec['generation']}_{rec['tag']}"
        safe_source = source.replace("/", "_")
        dst_pt = out_dir / f"{idx:03d}_{safe_source}.pt"
        dst_bin = out_dir / f"{idx:03d}_{safe_source}.bin"
        shutil.copy2(rec["pt"], dst_pt)
        shutil.copy2(rec["bin"], dst_bin)
        copied.append({
            "rank": idx,
            "source_tag": source,
            "proxy_score": float(rec["score"]),
            "candidate_index": int(rec["candidate_index"]),
            "pt": str(dst_pt),
            "bin": str(dst_bin),
        })
    return copied


def _test_vector_path(bin_path: Path) -> Path:
    return bin_path.with_name(f"{bin_path.stem}_test.bin")


def _cleanup_generation_artifacts(gen_records: list[dict], keep_records: list[dict]) -> dict:
    keep_paths: set[Path] = set()
    for rec in keep_records:
        keep_paths.add(Path(str(rec["pt"])).resolve())
        keep_paths.add(Path(str(rec["bin"])).resolve())

    removed = {"pt": 0, "bin": 0, "test_bin": 0}
    for rec in gen_records:
        pt = Path(str(rec["pt"])).resolve()
        bin_path = Path(str(rec["bin"])).resolve()
        test_path = _test_vector_path(bin_path)
        if test_path.exists():
            test_path.unlink()
            removed["test_bin"] += 1
        if pt not in keep_paths and pt.exists():
            pt.unlink()
            removed["pt"] += 1
        if bin_path not in keep_paths and bin_path.exists():
            bin_path.unlink()
            removed["bin"] += 1
    return removed


def _resolve_workers(requested: int) -> int:
    cpu_count = max(1, os.cpu_count() or 1)
    if requested > 0:
        return requested
    if requested < 0:
        return cpu_count
    return min(cpu_count, 12)


def run(args: argparse.Namespace) -> dict:
    t0 = time.time()
    workers = _resolve_workers(args.workers)
    seed_checkpoint = Path(args.seed_checkpoint).resolve()
    opponent_bin = Path(args.opponent_bin).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_pt = Path(args.out_pt).resolve()
    out_bin = Path(args.out_bin).resolve()
    log_path = work_dir / "eggroll_m2_prod.jsonl"

    if args.clean_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    seed_net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    mean_state = {k: v.detach().cpu().clone() for k, v in seed_net.state_dict().items()}
    keys = _parameter_keys(seed_net, args.scope)
    if not keys:
        raise RuntimeError(f"no parameters selected for scope={args.scope}")

    config = {
        "event": "start",
        "seed_checkpoint": str(seed_checkpoint),
        "opponent_bin": str(opponent_bin),
        "scope": args.scope,
        "selected_params": len(keys),
        "selected_param_count": int(sum(mean_state[k].numel() for k in keys)),
        "generations": args.generations,
        "pairs": args.pairs,
        "rank": args.rank,
        "sigma": args.sigma,
        "lr": args.lr,
        "mode": args.mode,
        "games": args.games,
        "promote_games": args.promote_games,
        "candidate_batch_size": args.candidate_batch_size,
        "workers": workers,
        "weight_format": args.weight_format,
        "seed": args.seed,
    }
    _write_jsonl(log_path, config)

    all_records: list[dict] = []
    history: list[dict] = []
    baseline_score = _score_baseline(args.mode, args.mix_2v2_weight)

    for gen_idx in range(args.generations):
        gen_dir = work_dir / f"gen_{gen_idx:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        noises = []
        tags = ["mean"]
        candidate_bins = []
        candidate_pts = []
        candidate_noise_refs: list[tuple[int | None, float]] = [(None, 0.0)]

        mean_pt = gen_dir / "00_mean.pt"
        mean_bin = gen_dir / "00_mean.bin"
        _export_state(
            seed_checkpoint,
            mean_state,
            mean_pt,
            mean_bin,
            {"eggroll_m2_prod": True, "generation": gen_idx, "tag": "mean"},
            args.weight_format,
        )
        candidate_pts.append(mean_pt)
        candidate_bins.append(mean_bin)

        for pair_idx in range(args.pairs):
            noise_seed = int(args.seed + 1_000_003 * gen_idx + 9176 * pair_idx)
            noise = _noise_tree(mean_state, keys, args.rank, noise_seed, args.relative_noise)
            noises.append(noise)
            for sign, sign_name in ((1.0, "plus"), (-1.0, "minus")):
                tag = f"p{pair_idx}_{sign_name}"
                idx = len(tags)
                tags.append(tag)
                state = _apply_noise(mean_state, noise, args.sigma, sign)
                pt = gen_dir / f"{idx:02d}_{tag}.pt"
                bin_path = gen_dir / f"{idx:02d}_{tag}.bin"
                _export_state(
                    seed_checkpoint,
                    state,
                    pt,
                    bin_path,
                    {
                        "eggroll_m2_prod": True,
                        "generation": gen_idx,
                        "tag": tag,
                        "pair": pair_idx,
                        "sign": sign_name,
                        "scope": args.scope,
                        "rank": args.rank,
                        "sigma": args.sigma,
                    },
                    args.weight_format,
                )
                candidate_pts.append(pt)
                candidate_bins.append(bin_path)
                candidate_noise_refs.append((pair_idx, sign))

        seed_base = args.seed_base + gen_idx * 100_000
        rows, meta = _eval_population(
            candidate_bins,
            opponent_bin,
            args.mode,
            args.games,
            workers,
            seed_base,
            args.mix_2v2_weight,
            args.candidate_batch_size,
        )
        scores = np.array([float(row["score"]) for row in rows], dtype=np.float32)

        pair_diffs = []
        for pair_idx in range(args.pairs):
            plus_idx = 1 + 2 * pair_idx
            minus_idx = plus_idx + 1
            pair_diffs.append(float(scores[plus_idx] - scores[minus_idx]))

        mean_state = _update_state(
            mean_state,
            noises,
            pair_diffs,
            keys,
            args.lr,
            args.sigma,
            args.update_clip_rms,
        )

        best_idx = int(np.argmax(scores))
        gen_records = []
        for idx, (tag, row, bin_path, pt_path, noise_ref) in enumerate(
            zip(tags, rows, candidate_bins, candidate_pts, candidate_noise_refs)
        ):
            rec = {
                "generation": gen_idx,
                "tag": tag,
                "candidate_index": idx,
                "score": float(row["score"]),
                "bin": str(bin_path),
                "pt": str(pt_path),
                "noise_ref": noise_ref,
                **{k: v for k, v in row.items() if k not in ("score", "a_weights")},
            }
            gen_records.append(rec)

        persisted_records = _dedupe_records(gen_records, max(1, args.promote_top))
        cleanup = _cleanup_generation_artifacts(gen_records, persisted_records)
        all_records.extend(persisted_records)

        hist = {
            "event": "generation",
            "generation": gen_idx,
            "seed_base": seed_base,
            "mean_score": float(scores[0]),
            "best_score": float(scores[best_idx]),
            "best_tag": tags[best_idx],
            "best_index": best_idx,
            "baseline_score": baseline_score,
            "elapsed_sec": time.time() - t0,
            "eval_meta": meta,
            "persisted_candidates": len(persisted_records),
            "artifact_cleanup": cleanup,
        }
        history.append(hist)
        _write_jsonl(log_path, hist)
        for rec in gen_records:
            _write_jsonl(log_path, {"event": "candidate", **rec})
        print(json.dumps(hist, sort_keys=True, default=_json_default), flush=True)

    promote_records = _dedupe_records(all_records, args.promote_top)
    top_candidates = _copy_top_candidates(promote_records, work_dir / "top_candidates")
    promote_seed_base = args.seed_base + args.generations * 100_000 + 50_000
    if args.promote_games <= 0:
        best_promote_idx = 0
        best_record = promote_records[0]
        best_row = {
            "candidate": int(best_record["candidate_index"]),
            "score": float(best_record["score"]),
            "proxy_only": True,
            "source_tag": f"g{best_record['generation']}_{best_record['tag']}",
        }
        promote_meta = {
            "skipped": True,
            "reason": "league_gate_replaces_pairwise_promote_gate",
            "promote_top": args.promote_top,
            "seed_base": promote_seed_base,
        }
        passed_gate = True
    else:
        promote_bins = [Path(rec["bin"]) for rec in promote_records]
        promote_rows, promote_meta = _eval_population(
            promote_bins,
            opponent_bin,
            args.mode,
            args.promote_games,
            workers,
            promote_seed_base,
            args.mix_2v2_weight,
            args.candidate_batch_size,
        )
        promote_scores = np.array([float(row["score"]) for row in promote_rows], dtype=np.float32)
        best_promote_idx = int(np.argmax(promote_scores))
        best_record = promote_records[best_promote_idx]
        best_row = promote_rows[best_promote_idx]
        passed_gate = float(best_row["score"]) >= baseline_score + float(args.min_promote_delta)

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    if passed_gate or not args.require_improvement:
        shutil.copy2(best_record["pt"], out_pt)
        shutil.copy2(best_record["bin"], out_bin)

    promote_log = {
        "event": "promote",
        "seed_base": promote_seed_base,
        "baseline_score": baseline_score,
        "min_promote_delta": args.min_promote_delta,
        "passed_gate": passed_gate,
        "best_source_tag": f"g{best_record['generation']}_{best_record['tag']}",
        "best_proxy_score": best_record["score"],
        "best_promote_score": float(best_row["score"]),
        "best_promote_row": best_row,
        "promote_meta": promote_meta,
        "top_candidates": top_candidates,
        "out_pt": str(out_pt) if passed_gate or not args.require_improvement else "",
        "out_bin": str(out_bin) if passed_gate or not args.require_improvement else "",
        "elapsed_sec": time.time() - t0,
    }
    _write_jsonl(log_path, promote_log)
    print(json.dumps(promote_log, sort_keys=True, default=_json_default), flush=True)

    return {
        "ok": True,
        "passed_gate": passed_gate,
        "best_promote_score": float(best_row["score"]),
        "baseline_score": baseline_score,
        "best_source_tag": promote_log["best_source_tag"],
        "out_pt": promote_log["out_pt"],
        "out_bin": promote_log["out_bin"],
        "top_candidates": top_candidates,
        "log_path": str(log_path),
        "history": history,
        "elapsed_sec": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", default=str(DEFAULT_SEED_CKPT))
    parser.add_argument("--opponent-bin", default=str(DEFAULT_BASE_BIN))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--out-pt", default=str(DEFAULT_OUT_PT))
    parser.add_argument("--out-bin", default=str(DEFAULT_OUT_BIN))
    parser.add_argument(
        "--scope",
        choices=(
            "policy_head_final",
            "policy_head_spatial_final",
            "policy_type_final",
            "policy_build_final",
            "policy_resource_action_final",
            "policy_road_final",
            "policy_head",
            "trunk",
            "trunk_policy",
        ),
        default="policy_head_final",
    )
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--relative-noise", action="store_true")
    parser.add_argument("--update-clip-rms", type=float, default=0.10)
    parser.add_argument("--mode", choices=("paired2v2", "ffa", "mix"), default="paired2v2")
    parser.add_argument("--mix-2v2-weight", type=float, default=0.5)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--promote-top", type=int, default=8)
    parser.add_argument("--promote-games", type=int, default=128)
    parser.add_argument("--min-promote-delta", type=float, default=0.0)
    parser.add_argument("--candidate-batch-size", type=int, default=0,
                        help="Shard candidate model loads for wide populations; 0 disables sharding")
    parser.add_argument("--no-require-improvement", dest="require_improvement", action="store_false")
    parser.set_defaults(require_improvement=True)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 means auto, capped at 12 on this machine; -1 means all visible CPU cores")
    parser.add_argument("--seed-base", type=int, default=6_500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-format", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--clean-work-dir", action="store_true")
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
