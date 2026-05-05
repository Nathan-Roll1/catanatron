"""Foreground production hillclimb for pure M2 Eggroll evolution.

This is the autoresearch runner around ``eggroll_m2_prod``.  Each outer
iteration treats the current incumbent as the seed checkpoint and the policy-zoo
leader:

1. Run one production Eggroll generation with a specific hyperparameter setup.
2. Rerank promising candidates in a mixed 4-player policy-zoo table.
3. Evaluate the selected candidate against original M2 as a diagnostic anchor.
4. Keep only if the selected candidate beats the fixed incumbent by a relative
   same-table PSRO league win-rate margin.

It writes an audit JSONL and records every completed iteration through the
codex-autoresearch helper.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from human_bot.eggroll_m2_prod import run as run_prod
from human_bot.eval_4way_league_many_nn_fast import (
    run as run_league4way,
    run_no_incumbent as run_league4way_no_incumbent,
)
from human_bot.eval_2v2_many_nn_paired_fast import run as run_paired2v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER = Path("/Users/nathanroll/.codex/skills/codex-autoresearch/scripts/autoresearch_record_iteration.py")


@dataclass(frozen=True)
class Setup:
    name: str
    scope: str
    pairs: int
    proxy_games: int
    lr: float
    sigma: float
    rank: int = 1
    relative_noise: bool = False
    generations: int = 1
    fitness_opponent: str = "previous"
    candidate_batch_size: int = 0
    promote_top: int = 8


SETUPS = [
    # Phase 9: spend production cycles on the surfaces that have kept finding
    # signal under the same-table PSRO gate.  The flat policy_head probes are
    # intentionally gone from this rotation.
    Setup("psro_spatial_p512_g16_lr012_s006", "policy_head_spatial_final", 512, 16, 0.12, 0.06, candidate_batch_size=64, promote_top=48),
    Setup("psro_spatial_p768_g8_lr025_s016", "policy_head_spatial_final", 768, 8, 0.25, 0.16, candidate_batch_size=48, promote_top=48),
    Setup("psro_final_p384_g16_lr025_s014", "policy_head_final", 384, 16, 0.25, 0.14, candidate_batch_size=64, promote_top=48),
    Setup("psro_final_p256_g24_lr018_s010", "policy_head_final", 256, 24, 0.18, 0.10, candidate_batch_size=64, promote_top=48),
]


DEEP_SETUPS = [
    # Unlocked only after repeated cool-downs indicate the final policy heads
    # are no longer yielding clean improvements.  These mutate trunk tensors
    # with much smaller relative perturbations so the frozen policy head can
    # test new representations without immediately destroying legality priors.
    Setup("psro_trunk_p256_g8_lr035_s018", "trunk", 256, 8, 0.035, 0.018, rank=1, relative_noise=True, candidate_batch_size=32, promote_top=48),
    Setup("psro_trunk_p384_g6_lr025_s012", "trunk", 384, 6, 0.025, 0.012, rank=1, relative_noise=True, candidate_batch_size=32, promote_top=48),
    Setup("psro_trunk_policy_p256_g6_lr020_s010", "trunk_policy", 256, 6, 0.020, 0.010, rank=1, relative_noise=True, candidate_batch_size=32, promote_top=48),
]


@dataclass
class PlateauState:
    decay_level: int = 0
    consecutive_gauntlet_discards: int = 0
    deep_unlocked: bool = False
    deep_unlocked_level: int = 0
    last_decay_iteration: int = 0


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {
            (key if isinstance(key, str) else str(key)): _json_sanitize(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _json_dumps(row: dict) -> str:
    return json.dumps(_json_sanitize(row), sort_keys=True, default=_json_default)


def _write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_json_dumps(row) + "\n")


def _next_iteration_start(log_path: Path) -> int:
    max_iter = 0
    if log_path.exists():
        try:
            with log_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("event") == "iteration":
                        max_iter = max(max_iter, int(row.get("iteration", 0)))
        except OSError:
            pass
    try:
        for kept in log_path.parent.glob("kept_iter_*.bin"):
            stem = kept.stem
            try:
                max_iter = max(max_iter, int(stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    except OSError:
        pass
    return max_iter


def _wilson_lower(wins: int, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    phat = wins / games
    denom = 1.0 + z * z / games
    center = phat + z * z / (2.0 * games)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * games)) / games)
    return (center - spread) / denom


def _load_state_metric(default: float) -> float:
    path = PROJECT_ROOT / "autoresearch-results" / "state.json"
    try:
        with path.open(encoding="utf-8") as f:
            state = json.load(f)
        return float(state["state"].get("current_metric", default))
    except Exception:
        return default


def _record(status: str, metric: float, description: str, labels: list[str]) -> None:
    cmd = [
        "python3",
        str(HELPER),
        "--status",
        status,
        "--metric",
        f"{metric:.12f}",
        "--commit",
        _git_head(),
        "--guard",
        "-",
        "--description",
        description,
    ]
    for label in labels:
        cmd.extend(["--label", label])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(PROJECT_ROOT),
        text=True,
    ).strip()


def _copy_pair(src_pt: Path, src_bin: Path, dst_pt: Path, dst_bin: Path) -> None:
    dst_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pt, dst_pt)
    shutil.copy2(src_bin, dst_bin)


def _file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _elo_from_score(score: float, opponent_elo: float = 1500.0) -> float:
    clipped = min(0.999, max(0.001, float(score)))
    return float(opponent_elo) + 400.0 * math.log10(clipped / (1.0 - clipped))


def _load_league_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
    except Exception:
        return []
    return []


def _save_league_manifest(path: Path, members: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(members, f, sort_keys=True, indent=2)


_RATING_SOURCE_PRIORITY = {
    "keep": 100,
    "diversity-keeper": 95,
    "recovered-log-keep": 90,
    "seed-current": 70,
    "seed-original": 70,
    "recovered-keep": 10,
}


def _add_league_member(members: list[dict], src_bin: Path, league_dir: Path,
                       label: str, rating: float, source: str) -> list[dict]:
    src_bin = src_bin.resolve()
    if not src_bin.exists():
        return members
    league_dir.mkdir(parents=True, exist_ok=True)
    digest = _file_hash(src_bin)
    for member in members:
        if member.get("sha256") == digest:
            old_source = str(member.get("source", ""))
            old_priority = _RATING_SOURCE_PRIORITY.get(old_source, 0)
            new_priority = _RATING_SOURCE_PRIORITY.get(source, 0)
            if new_priority > old_priority or (
                new_priority == old_priority
                and float(rating) > float(member.get("rating", 0.0))
            ):
                member["label"] = label
                member["rating"] = float(rating)
                member["source"] = source
            return members
    dst = league_dir / f"{label}_{digest[:12]}.bin"
    shutil.copy2(src_bin, dst)
    members.append({
        "label": label,
        "path": str(dst.resolve()),
        "sha256": digest,
        "rating": float(rating),
        "source": source,
        "added_at": time.time(),
    })
    return members


def _recover_log_keeps(members: list[dict], root: Path, league_dir: Path) -> list[dict]:
    log_path = root / "hillclimb.jsonl"
    if not log_path.exists():
        return members
    try:
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") != "iteration" or row.get("decision") != "keep":
                    continue
                prod = row.get("prod") if isinstance(row.get("prod"), dict) else {}
                out_bin = prod.get("out_bin")
                if not out_bin:
                    continue
                wr = float(row.get("original_eval", {}).get("winrate", row.get("current_metric", 0.5)))
                rating = float(row.get("candidate_elo", _elo_from_score(wr, 1500.0)))
                iteration = int(row.get("iteration", 0))
                members = _add_league_member(
                    members,
                    Path(str(out_bin)),
                    league_dir,
                    f"json_keep_{iteration:04d}_{int(wr * 10000)}",
                    rating,
                    "recovered-log-keep",
                )
    except OSError:
        return members
    return members


def _seed_league(root: Path, original_bin: Path, live_bin: Path,
                 current_rating: float, league_size: int) -> list[dict]:
    league_dir = root / "league"
    manifest_path = league_dir / "manifest.json"
    members = _load_league_manifest(manifest_path)
    members = [m for m in members if Path(str(m.get("path", ""))).exists()]

    members = _add_league_member(
        members,
        original_bin,
        league_dir,
        "original_m2",
        1500.0,
        "seed-original",
    )
    members = _add_league_member(
        members,
        live_bin,
        league_dir,
        "current_start",
        current_rating,
        "seed-current",
    )

    # Recover any historical keep artifacts that survived earlier relaunches.
    for path in sorted(root.glob("kept_iter_*.bin")):
        members = _add_league_member(
            members,
            path,
            league_dir,
            path.stem,
            current_rating,
            "recovered-keep",
        )
    members = _recover_log_keeps(members, root, league_dir)

    # Keep original/current anchors plus the strongest-rated distinct keepers.
    original_hash = _file_hash(original_bin)
    current_hash = _file_hash(live_bin)
    original_anchors: list[dict] = []
    current_anchors: list[dict] = []
    rest: list[dict] = []
    for member in members:
        digest = str(member.get("sha256", ""))
        if digest == original_hash:
            original_anchors.append(member)
        elif digest == current_hash:
            current_anchors.append(member)
        else:
            rest.append(member)
    original_anchors.sort(key=lambda m: float(m.get("added_at", 0.0)), reverse=True)
    current_anchors.sort(key=lambda m: float(m.get("added_at", 0.0)), reverse=True)
    anchors = original_anchors[:1] + current_anchors[:1]
    rest.extend(original_anchors[1:])
    rest.extend(current_anchors[1:])
    rest.sort(
        key=lambda m: (
            float(m.get("rating", 1500.0)),
            float(m.get("added_at", 0.0)),
        ),
        reverse=True,
    )
    if league_size > 0:
        members = anchors + rest[:max(0, league_size - len(anchors))]
    else:
        members = anchors + rest
    _save_league_manifest(manifest_path, members)
    return members


def _iter_from_text(text: str) -> int | None:
    import re

    match = re.search(r"(?:iter_|keep_?|json_keep_)(\d{2,5})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{2,5})", text)
    return int(match.group(1)) if match else None


def _member_vector_key(member_row: dict) -> str:
    path = str(member_row.get("path", "")).strip()
    if path:
        return str(Path(path).resolve())
    return str(member_row.get("label", "")).strip()


def _collect_member_response_vectors(log_path: Path, max_iterations: int) -> dict[str, dict[str, float]]:
    rows: list[dict] = []
    if not log_path.exists():
        return {}
    try:
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "iteration" and isinstance(row.get("league_rerank"), dict):
                    rows.append(row)
    except OSError:
        return {}

    vectors: dict[str, dict[str, float]] = {}
    for row in rows[-max(1, int(max_iterations)):]:
        iteration = int(row.get("iteration", 0))
        rerank = row.get("league_rerank") if isinstance(row.get("league_rerank"), dict) else {}
        evals = rerank.get("evaluations") if isinstance(rerank.get("evaluations"), list) else []
        for ev_idx, ev in enumerate(evals):
            if not isinstance(ev, dict):
                continue
            entry = ev.get("candidate_entry") if isinstance(ev.get("candidate_entry"), dict) else {}
            feature = f"{iteration}:{entry.get('rank', ev_idx)}:{entry.get('source_tag', ev_idx)}"
            for member_row in ev.get("members", []):
                if not isinstance(member_row, dict):
                    continue
                key = _member_vector_key(member_row)
                if not key:
                    continue
                score = member_row.get("score", member_row.get("winrate"))
                if isinstance(score, (int, float)):
                    vectors.setdefault(key, {})[feature] = float(score)
    return vectors


def _pearson_common(a: dict[str, float], b: dict[str, float],
                    min_common: int) -> tuple[float | None, int]:
    common = sorted(set(a).intersection(b))
    if len(common) < int(min_common):
        return None, len(common)
    xs = [a[k] for k in common]
    ys = [b[k] for k in common]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) * (x - mx) for x in xs)
    vy = sum((y - my) * (y - my) for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None, len(common)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy), len(common)


def _historical_keep_rating(log_path: Path) -> dict[int, float]:
    ratings: dict[int, float] = {}
    if not log_path.exists():
        return ratings
    try:
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") == "iteration" and row.get("decision") == "keep":
                    iteration = int(row.get("iteration", 0))
                    ratings[iteration] = float(row.get("candidate_elo", 1500.0))
    except OSError:
        pass
    return ratings


def _choose_historical_replacement(root: Path, members: list[dict], log_path: Path,
                                   target_iters: list[int]) -> tuple[Path | None, int | None, float]:
    present_hashes = {str(m.get("sha256", "")) for m in members}
    ratings = _historical_keep_rating(log_path)
    candidates: list[tuple[tuple[float, int], Path, int, float]] = []
    for path in root.glob("kept_iter_*.bin"):
        iteration = _iter_from_text(path.stem)
        if iteration is None:
            continue
        try:
            digest = _file_hash(path)
        except OSError:
            continue
        if digest in present_hashes:
            continue
        if target_iters:
            distance = min(abs(iteration - target) for target in target_iters)
        else:
            distance = abs(iteration)
        rating = ratings.get(iteration, 1500.0)
        candidates.append(((distance, -iteration), path, iteration, rating))
    if not candidates:
        return None, None, 1500.0
    _, path, iteration, rating = min(candidates, key=lambda item: item[0])
    return path, iteration, rating


def _curate_league_diversity(members: list[dict], root: Path, log_path: Path,
                             original_bin: Path, live_bin: Path,
                             table_incumbent_bin: Path, args: argparse.Namespace,
                             iteration: int) -> tuple[list[dict], dict | None]:
    if args.league_diversity_interval <= 0:
        return members, None
    if iteration <= 0 or iteration % args.league_diversity_interval != 0:
        return members, None
    vectors = _collect_member_response_vectors(log_path, args.league_diversity_history)
    if len(vectors) < 2:
        return members, None

    protected_hashes = {
        _file_hash(original_bin),
        _file_hash(live_bin),
        _file_hash(table_incumbent_bin),
    }
    by_path = {
        str(Path(str(member.get("path", ""))).resolve()): member
        for member in members
        if member.get("path")
    }
    best_pair = None
    for i, left_key in enumerate(by_path):
        left = by_path[left_key]
        if left.get("sha256") in protected_hashes:
            continue
        for right_key in list(by_path)[i + 1:]:
            right = by_path[right_key]
            if right.get("sha256") in protected_hashes:
                continue
            corr, common = _pearson_common(
                vectors.get(left_key, {}),
                vectors.get(right_key, {}),
                args.league_diversity_min_common,
            )
            if corr is None or corr < args.league_diversity_corr:
                continue
            if best_pair is None or corr > best_pair[0]:
                best_pair = (corr, common, left, right)

    if best_pair is None:
        event = {
            "event": "league_curation",
            "iteration": iteration,
            "action": "none",
            "reason": "no_redundant_pair",
            "threshold": args.league_diversity_corr,
        }
        _write_jsonl(log_path, event)
        print(_json_dumps(event), flush=True)
        return members, event

    corr, common, left, right = best_pair
    left_rating = float(left.get("rating", 1500.0))
    right_rating = float(right.get("rating", 1500.0))
    if abs(left_rating - right_rating) < 10.0:
        evict = left if float(left.get("added_at", 0.0)) > float(right.get("added_at", 0.0)) else right
        keep = right if evict is left else left
    else:
        evict = left if left_rating < right_rating else right
        keep = right if evict is left else left

    curated = [m for m in members if m.get("sha256") != evict.get("sha256")]
    replacement_path, replacement_iter, replacement_rating = _choose_historical_replacement(
        root,
        curated,
        log_path,
        args.league_diversity_replacement_iters,
    )
    replacement = None
    if replacement_path is not None and replacement_iter is not None:
        curated = _add_league_member(
            curated,
            replacement_path,
            root / "league",
            f"diverse_iter_{replacement_iter:04d}",
            replacement_rating,
            "diversity-keeper",
        )
        replacement = {
            "iteration": replacement_iter,
            "path": str(replacement_path.resolve()),
            "rating": replacement_rating,
        }
    _save_league_manifest(root / "league" / "manifest.json", curated)
    event = {
        "event": "league_curation",
        "iteration": iteration,
        "action": "evict_redundant",
        "correlation": corr,
        "common_observations": common,
        "evicted": {
            "label": evict.get("label"),
            "rating": evict.get("rating"),
            "source": evict.get("source"),
        },
        "kept_from_pair": {
            "label": keep.get("label"),
            "rating": keep.get("rating"),
            "source": keep.get("source"),
        },
        "replacement": replacement,
        "threshold": args.league_diversity_corr,
    }
    _write_jsonl(log_path, event)
    print(_json_dumps(event), flush=True)
    return curated, event


def _eval_original(candidate_bin: Path, original_bin: Path, games_per_side: int,
                   workers: int, seed_base: int) -> dict:
    res = run_paired2v2(
        [str(candidate_bin.resolve())],
        str(original_bin.resolve()),
        games_per_side,
        workers,
        seed_base,
    )
    row = res["results"][0]
    wins = int(row["a_wins"])
    games = int(row["games"])
    return {
        "wins": wins,
        "games": games,
        "winrate": wins / max(1, games),
        "wilson_lower": _wilson_lower(wins, games),
        "elapsed_sec": float(res["elapsed_sec"]),
        "row": row,
    }


def _eval_pair(candidate_bin: Path, opponent_bin: Path, games_per_side: int,
               workers: int, seed_base: int) -> dict:
    return _eval_original(candidate_bin, opponent_bin, games_per_side, workers, seed_base)


def _member_rating_for_bin(members: list[dict], bin_path: Path, default: float) -> float:
    try:
        digest = _file_hash(bin_path)
    except OSError:
        return default
    for member in members:
        if member.get("sha256") == digest:
            return float(member.get("rating", default))
    return default


def _league_keeper_members(members: list[dict], original_bin: Path,
                           incumbent_bin: Path,
                           extra_exclude_bins: list[Path] | None = None) -> list[dict]:
    excluded = set()
    for path in (original_bin, incumbent_bin, *(extra_exclude_bins or [])):
        try:
            excluded.add(_file_hash(path))
        except OSError:
            pass
    keepers = [
        member for member in members
        if member.get("sha256") not in excluded
        and Path(str(member.get("path", ""))).exists()
    ]
    return keepers or [m for m in members if Path(str(m.get("path", ""))).exists()]


def _eval_league_pairwise(candidate_bin: Path, members: list[dict], games_per_side: int,
                          workers: int, seed_base: int) -> dict:
    rows: list[dict] = []
    wins = 0
    games = 0
    weighted_elo = 0.0
    weight_count = 0
    t0 = time.time()
    for idx, member in enumerate(members):
        opponent = Path(str(member["path"]))
        res = _eval_pair(
            candidate_bin,
            opponent,
            games_per_side,
            workers,
            seed_base + idx * 100_000,
        )
        wr = float(res["winrate"])
        opp_elo = float(member.get("rating", 1500.0))
        provisional_elo = _elo_from_score(wr, opp_elo)
        wins += int(res["wins"])
        games += int(res["games"])
        weighted_elo += provisional_elo
        weight_count += 1
        rows.append({
            "label": member.get("label", f"member_{idx}"),
            "path": str(opponent),
            "opponent_rating": opp_elo,
            "wins": int(res["wins"]),
            "games": int(res["games"]),
            "winrate": wr,
            "candidate_elo_vs_member": provisional_elo,
            "wilson_lower": float(res["wilson_lower"]),
        })
    avg_wr = wins / max(1, games)
    return {
        "members": rows,
        "wins": wins,
        "games": games,
        "avg_winrate": avg_wr,
        "candidate_elo": weighted_elo / max(1, weight_count),
        "elapsed_sec": time.time() - t0,
        "mode": "pairwise2v2",
        "avg_score": avg_wr,
    }


def _eval_league_mixed4way(candidate_bin: Path, members: list[dict],
                           incumbent_bin: Path, original_bin: Path,
                           games_per_member: int, workers: int,
                           seed_base: int,
                           extra_exclude_bins: list[Path] | None = None) -> dict:
    league = _eval_league_candidates_mixed4way(
        [{"bin": str(candidate_bin), "pt": "", "source_tag": "candidate", "proxy_score": 0.0}],
        members,
        incumbent_bin,
        original_bin,
        games_per_member,
        workers,
        seed_base,
        extra_exclude_bins,
    )
    return league["evaluations"][0]


def _combine_same_table_evals(first: dict, second: dict) -> dict:
    games = int(first.get("games", 0)) + int(second.get("games", 0))
    if games <= 0:
        return first
    wins = int(first.get("wins", 0)) + int(second.get("wins", 0))
    incumbent_wins = int(first.get("incumbent_wins", 0)) + int(second.get("incumbent_wins", 0))
    combined = dict(first)
    combined["primary_eval"] = first
    combined["confirmation_eval"] = second
    combined["games"] = games
    combined["wins"] = wins
    combined["incumbent_wins"] = incumbent_wins
    combined["avg_winrate"] = wins / games
    combined["incumbent_winrate"] = incumbent_wins / games
    combined["candidate_vs_incumbent_winrate_delta"] = combined["avg_winrate"] - combined["incumbent_winrate"]
    combined["avg_score"] = (
        (float(first.get("avg_score", 0.0)) * int(first.get("games", 0)))
        + (float(second.get("avg_score", 0.0)) * int(second.get("games", 0)))
    ) / games
    combined["candidate_elo"] = (
        (float(first.get("candidate_elo", 1500.0)) * int(first.get("games", 0)))
        + (float(second.get("candidate_elo", 1500.0)) * int(second.get("games", 0)))
    ) / games
    combined["elapsed_sec"] = float(first.get("elapsed_sec", 0.0)) + float(second.get("elapsed_sec", 0.0))
    combined["mode"] = "mixed4way-confirmed"
    return combined


def _combine_many_same_table_evals(evals: list[dict]) -> dict:
    if not evals:
        return {}
    combined = dict(evals[0])
    for ev in evals[1:]:
        combined = _combine_same_table_evals(combined, ev)
    combined["stage_evals"] = evals
    combined["mode"] = "mixed4way-staged"
    return combined


def _eval_sort_key(item: dict) -> tuple[float, float, float, float]:
    entry = item.get("candidate_entry", {})
    return (
        float(item.get("candidate_vs_incumbent_winrate_delta", 0.0)),
        float(item.get("avg_score", item.get("avg_winrate", 0.0))),
        float(item.get("candidate_elo", 1500.0)),
        float(entry.get("proxy_score", 0.0)),
    )


def _parse_int_csv(value: str, default: list[int]) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(max(0, int(part)))
    return out or list(default)


def _parse_float_csv(value: str, default: list[float]) -> list[float]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out or list(default)


def _plateau_min_scaled_lr(decay_level: int, decay_factor: float) -> float:
    scale = float(decay_factor) ** max(0, int(decay_level))
    return min(setup.lr * scale for setup in SETUPS)


def _setup_scaled(setup: Setup, decay_level: int, decay_factor: float,
                  name_suffix: str = "") -> Setup:
    scale = float(decay_factor) ** max(0, int(decay_level))
    suffix = name_suffix or (f"_d{decay_level}" if decay_level > 0 else "")
    return replace(
        setup,
        name=f"{setup.name}{suffix}",
        lr=max(1e-6, setup.lr * scale),
        sigma=max(1e-6, setup.sigma * scale),
    )


def _active_setups(plateau_state: PlateauState, decay_factor: float,
                   unfreeze_min_lr: float, enabled: bool) -> list[Setup]:
    if not enabled:
        return list(SETUPS)
    deep_ready = (
        plateau_state.deep_unlocked
        or _plateau_min_scaled_lr(plateau_state.decay_level, decay_factor) < float(unfreeze_min_lr)
    )
    if deep_ready:
        unlock_level = plateau_state.deep_unlocked_level or plateau_state.decay_level
        deep_decay = max(0, plateau_state.decay_level - unlock_level)
        return [
            _setup_scaled(setup, deep_decay, decay_factor, f"_deepd{deep_decay}" if deep_decay > 0 else "")
            for setup in DEEP_SETUPS
        ]
    return [_setup_scaled(setup, plateau_state.decay_level, decay_factor) for setup in SETUPS]


_GAUNTLET_DISCARDS = {
    "discard_same_table_league",
    "discard_league_gate",
    "discard_league_elo",
    "discard_original_confirm",
}


def _is_gauntlet_discard(row: dict) -> bool:
    if row.get("event") != "iteration" or row.get("decision") not in _GAUNTLET_DISCARDS:
        return False
    setup = row.get("setup") if isinstance(row.get("setup"), dict) else {}
    return str(setup.get("scope", "")) in {
        "policy_head_final",
        "policy_head_spatial_final",
        "policy_head",
        "trunk",
        "trunk_policy",
    }


def _load_plateau_state(log_path: Path, decay_factor: float,
                        unfreeze_min_lr: float) -> PlateauState:
    state = PlateauState()
    if not log_path.exists():
        return state
    try:
        with log_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = row.get("event")
                if event == "plateau_decay":
                    state.decay_level = max(state.decay_level, int(row.get("decay_level", 0)))
                    state.consecutive_gauntlet_discards = 0
                    state.last_decay_iteration = int(row.get("iteration", state.last_decay_iteration))
                    if row.get("deep_unlocked"):
                        state.deep_unlocked = True
                        state.deep_unlocked_level = int(row.get("deep_unlocked_level", state.decay_level))
                elif event == "iteration":
                    if row.get("decision") == "keep":
                        state.consecutive_gauntlet_discards = 0
                    elif _is_gauntlet_discard(row):
                        state.consecutive_gauntlet_discards += 1
    except OSError:
        return state
    if _plateau_min_scaled_lr(state.decay_level, decay_factor) < float(unfreeze_min_lr):
        state.deep_unlocked = True
        if state.deep_unlocked_level <= 0:
            state.deep_unlocked_level = state.decay_level
    return state


def _update_plateau_state(row: dict, state: PlateauState, args: argparse.Namespace,
                          log_path: Path) -> PlateauState:
    if not args.plateau_schedule:
        return state
    if row.get("decision") == "keep":
        state.consecutive_gauntlet_discards = 0
        return state
    if not _is_gauntlet_discard(row):
        return state

    state.consecutive_gauntlet_discards += 1
    if state.consecutive_gauntlet_discards < args.plateau_decay_discards:
        return state

    state.decay_level += 1
    state.consecutive_gauntlet_discards = 0
    state.last_decay_iteration = int(row.get("iteration", 0))
    min_lr = _plateau_min_scaled_lr(state.decay_level, args.plateau_decay_factor)
    if min_lr < args.plateau_unfreeze_min_lr and not state.deep_unlocked:
        state.deep_unlocked = True
        state.deep_unlocked_level = state.decay_level

    event = {
        "event": "plateau_decay",
        "iteration": int(row.get("iteration", 0)),
        "trigger": f"{args.plateau_decay_discards}_consecutive_gauntlet_discards",
        "decay_level": state.decay_level,
        "decay_factor": args.plateau_decay_factor,
        "lr_sigma_scale": args.plateau_decay_factor ** state.decay_level,
        "min_scaled_head_lr": min_lr,
        "deep_unlocked": state.deep_unlocked,
        "deep_unlocked_level": state.deep_unlocked_level,
        "next_schedule": [
            setup.__dict__
            for setup in _active_setups(
                state,
                args.plateau_decay_factor,
                args.plateau_unfreeze_min_lr,
                args.plateau_schedule,
            )
        ],
    }
    _write_jsonl(log_path, event)
    print(_json_dumps(event), flush=True)
    return state


def _select_adaptive_top_entries(entries: list[dict], max_top: int, min_top: int,
                                 enabled: bool, weak_best_score: float = 0.625,
                                 weak_top: int = 16, mid_best_score: float = 0.67,
                                 mid_top: int = 32) -> tuple[list[dict], dict]:
    max_top = max(1, int(max_top))
    min_top = max(1, min(int(min_top), max_top))
    weak_top = max(min_top, min(int(weak_top), max_top))
    mid_top = max(weak_top, min(int(mid_top), max_top))
    capped = list(entries[:max_top])
    if not enabled or len(capped) <= min_top:
        return capped, {
            "enabled": bool(enabled),
            "requested_top": max_top,
            "selected_top": len(capped),
            "reason": "disabled_or_small",
        }

    scores = [float(entry.get("proxy_score", 0.0)) for entry in capped]
    positive_count = sum(1 for score in scores if score > 0.500000001)
    best = scores[0] if scores else 0.0
    score_cap = max_top
    reason = "positive_proxy_plus_floor"
    if best < float(weak_best_score):
        score_cap = weak_top
        reason = "weak_proxy_cap"
    elif best < float(mid_best_score):
        score_cap = mid_top
        reason = "mid_proxy_cap"
    selected_top = min(score_cap, max(min_top, positive_count))
    best_tie_count = sum(1 for score in scores if abs(score - best) <= 1e-12)
    selected = capped[:selected_top]
    return selected, {
        "enabled": True,
        "requested_top": max_top,
        "selected_top": selected_top,
        "min_top": min_top,
        "weak_best_score": float(weak_best_score),
        "weak_top": weak_top,
        "mid_best_score": float(mid_best_score),
        "mid_top": mid_top,
        "score_cap": score_cap,
        "positive_proxy_count": positive_count,
        "best_proxy_score": best,
        "best_tie_count": best_tie_count,
        "last_selected_proxy_score": scores[selected_top - 1] if selected else None,
        "next_proxy_score": scores[selected_top] if selected_top < len(scores) else None,
        "reason": reason,
    }


def _eval_league_candidates_mixed4way(candidate_entries: list[dict], members: list[dict],
                                      incumbent_bin: Path, original_bin: Path,
                                      games_per_member: int, workers: int,
                                      seed_base: int,
                                      extra_exclude_bins: list[Path] | None = None) -> dict:
    eval_members = _league_keeper_members(members, original_bin, incumbent_bin, extra_exclude_bins)
    candidate_bins = [Path(str(entry["bin"])).resolve() for entry in candidate_entries]
    res = run_league4way(
        [str(path) for path in candidate_bins],
        str(incumbent_bin.resolve()),
        str(original_bin.resolve()),
        [str(Path(str(m["path"])).resolve()) for m in eval_members],
        games_per_member,
        workers,
        seed_base,
    )
    incumbent_rating = _member_rating_for_bin(members, incumbent_bin, 1500.0)
    original_rating = _member_rating_for_bin(members, original_bin, 1500.0)
    evaluations: list[dict] = []
    for entry, row in zip(candidate_entries, res["results"]):
        league_rows = []
        candidate_elos = []
        for member, member_row in zip(eval_members, row.get("per_member", [])):
            member_rating = float(member.get("rating", 1500.0))
            field_rating = (incumbent_rating + original_rating + member_rating) / 3.0
            member_score = float(member_row["score"])
            candidate_elo = _elo_from_score(member_score, field_rating)
            candidate_elos.append(candidate_elo)
            league_rows.append({
                "label": member.get("label", f"member_{member_row.get('member', 0)}"),
                "path": str(member.get("path", "")),
                "opponent_rating": member_rating,
                "field_rating": field_rating,
                "wins": int(member_row["candidate_wins"]),
                "incumbent_wins": int(member_row.get("incumbent_wins", 0)),
                "games": int(member_row["games"]),
                "winrate": float(member_row["candidate_winrate"]),
                "incumbent_winrate": float(member_row.get("incumbent_winrate", 0.0)),
                "candidate_vs_incumbent_winrate_delta": float(
                    member_row.get("candidate_vs_incumbent_winrate_delta", 0.0)
                ),
                "score": member_score,
                "candidate_elo_vs_member": candidate_elo,
                "candidate_seat_wins": member_row.get("candidate_seat_wins", []),
                "winner_labels": member_row.get("winner_labels", {}),
            })
        avg_wr = float(row["candidate_winrate"])
        incumbent_wr = float(row.get("incumbent_winrate", 0.0))
        same_table_delta = float(row.get("candidate_vs_incumbent_winrate_delta", avg_wr - incumbent_wr))
        evaluations.append({
            "candidate_entry": entry,
            "members": league_rows,
            "wins": int(row["candidate_wins"]),
            "incumbent_wins": int(row.get("incumbent_wins", 0)),
            "games": int(row["games"]),
            "avg_winrate": avg_wr,
            "incumbent_winrate": incumbent_wr,
            "candidate_vs_incumbent_winrate_delta": same_table_delta,
            "avg_score": float(row["score"]),
            "candidate_elo": sum(candidate_elos) / max(1, len(candidate_elos)),
            "elapsed_sec": float(res["elapsed_sec"]),
            "mode": "mixed4way",
            "raw_candidate": {k: v for k, v in row.items() if k != "per_member"},
        })
    evaluations.sort(key=_eval_sort_key, reverse=True)
    return {
        "evaluations": evaluations,
        "elapsed_sec": float(res["elapsed_sec"]),
        "mode": "mixed4way",
        "raw": {k: v for k, v in res.items() if k != "results"},
    }


def _eval_league_candidates_mixed4way_staged(
    candidate_entries: list[dict],
    members: list[dict],
    incumbent_bin: Path,
    original_bin: Path,
    stage_games: list[int],
    stage_keeps: list[int],
    workers: int,
    seed_base: int,
    extra_exclude_bins: list[Path] | None = None,
) -> dict:
    active = list(candidate_entries)
    by_key: dict[str, list[dict]] = {
        str(Path(str(entry["bin"])).resolve()): []
        for entry in active
    }
    stages = []
    eliminated: list[dict] = []
    total_elapsed = 0.0

    for stage_idx, games in enumerate(stage_games):
        games = int(games)
        if games <= 0 or not active:
            continue
        stage = _eval_league_candidates_mixed4way(
            active,
            members,
            incumbent_bin,
            original_bin,
            games,
            workers,
            seed_base + stage_idx * 100_000,
            extra_exclude_bins,
        )
        total_elapsed += float(stage.get("elapsed_sec", 0.0))
        for ev in stage["evaluations"]:
            key = str(Path(str(ev["candidate_entry"]["bin"])).resolve())
            by_key.setdefault(key, []).append(ev)

        combined_active = []
        for entry in active:
            key = str(Path(str(entry["bin"])).resolve())
            combined = _combine_many_same_table_evals(by_key.get(key, []))
            if combined:
                combined_active.append(combined)
        combined_active.sort(key=_eval_sort_key, reverse=True)

        keep_n = len(combined_active)
        if stage_idx < len(stage_keeps):
            keep_n = min(keep_n, max(1, int(stage_keeps[stage_idx])))
        keep_keys = {
            str(Path(str(ev["candidate_entry"]["bin"])).resolve())
            for ev in combined_active[:keep_n]
        }
        if stage_idx < len(stage_games) - 1:
            eliminated.extend(combined_active[keep_n:])
            active = [
                entry for entry in active
                if str(Path(str(entry["bin"])).resolve()) in keep_keys
            ]

        stages.append({
            "stage": stage_idx + 1,
            "games_per_member": games,
            "input_candidates": int(stage["raw"].get("candidates", len(active))),
            "kept_candidates": keep_n if stage_idx < len(stage_games) - 1 else len(combined_active),
            "elapsed_sec": float(stage.get("elapsed_sec", 0.0)),
            "top": [
                {
                    "rank": int(ev.get("candidate_entry", {}).get("rank", -1)),
                    "source_tag": ev.get("candidate_entry", {}).get("source_tag", ""),
                    "proxy_score": float(ev.get("candidate_entry", {}).get("proxy_score", 0.0)),
                    "delta": float(ev.get("candidate_vs_incumbent_winrate_delta", 0.0)),
                    "avg_score": float(ev.get("avg_score", 0.0)),
                    "candidate_elo": float(ev.get("candidate_elo", 0.0)),
                    "games": int(ev.get("games", 0)),
                }
                for ev in combined_active[:min(8, len(combined_active))]
            ],
        })

    final_evals = []
    for entry in active:
        key = str(Path(str(entry["bin"])).resolve())
        combined = _combine_many_same_table_evals(by_key.get(key, []))
        if combined:
            final_evals.append(combined)
    final_evals.sort(key=_eval_sort_key, reverse=True)
    eliminated.sort(key=_eval_sort_key, reverse=True)
    return {
        "evaluations": final_evals + eliminated,
        "finalist_evaluations": final_evals,
        "eliminated_evaluations": eliminated,
        "elapsed_sec": total_elapsed,
        "mode": "mixed4way-staged",
        "stages": stages,
        "stage_games": stage_games,
        "stage_keeps": stage_keeps,
        "raw": {
            "candidates": len(candidate_entries),
            "finalists": len(final_evals),
            "stages": len(stages),
        },
    }


def _eval_league_candidates_mixed4way_no_incumbent(
    candidate_entries: list[dict],
    members: list[dict],
    incumbent_bin: Path,
    original_bin: Path,
    games_per_member: int,
    workers: int,
    seed_base: int,
    extra_exclude_bins: list[Path] | None = None,
) -> dict:
    eval_members = _league_keeper_members(members, original_bin, incumbent_bin, extra_exclude_bins)
    candidate_bins = [Path(str(entry["bin"])).resolve() for entry in candidate_entries]
    res = run_league4way_no_incumbent(
        [str(path) for path in candidate_bins],
        str(original_bin.resolve()),
        [str(Path(str(m["path"])).resolve()) for m in eval_members],
        games_per_member,
        workers,
        seed_base,
    )
    original_rating = _member_rating_for_bin(members, original_bin, 1500.0)
    evaluations: list[dict] = []
    for entry, row in zip(candidate_entries, res["results"]):
        league_rows = []
        candidate_elos = []
        for pair_row in row.get("per_pair", []):
            a_idx = int(pair_row.get("member_a", 0))
            b_idx = int(pair_row.get("member_b", 0))
            member_a = eval_members[a_idx] if 0 <= a_idx < len(eval_members) else {}
            member_b = eval_members[b_idx] if 0 <= b_idx < len(eval_members) else {}
            rating_a = float(member_a.get("rating", 1500.0))
            rating_b = float(member_b.get("rating", 1500.0))
            field_rating = (original_rating + rating_a + rating_b) / 3.0
            pair_score = float(pair_row["score"])
            candidate_elo = _elo_from_score(pair_score, field_rating)
            candidate_elos.append(candidate_elo)
            league_rows.append({
                "label": f"{member_a.get('label', f'member_{a_idx}')}+{member_b.get('label', f'member_{b_idx}')}",
                "paths": [str(member_a.get("path", "")), str(member_b.get("path", ""))],
                "opponent_ratings": [rating_a, rating_b],
                "field_rating": field_rating,
                "wins": int(pair_row["candidate_wins"]),
                "games": int(pair_row["games"]),
                "winrate": float(pair_row["candidate_winrate"]),
                "score": pair_score,
                "candidate_elo_vs_pair": candidate_elo,
                "candidate_seat_wins": pair_row.get("candidate_seat_wins", []),
                "winner_labels": pair_row.get("winner_labels", {}),
            })
        evaluations.append({
            "candidate_entry": entry,
            "members": league_rows,
            "wins": int(row["candidate_wins"]),
            "games": int(row["games"]),
            "avg_winrate": float(row["candidate_winrate"]),
            "avg_score": float(row["score"]),
            "candidate_elo": sum(candidate_elos) / max(1, len(candidate_elos)),
            "elapsed_sec": float(res["elapsed_sec"]),
            "mode": "mixed4way-no-incumbent",
            "raw_candidate": {k: v for k, v in row.items() if k != "per_pair"},
        })
    evaluations.sort(key=lambda item: (float(item["avg_score"]), float(item["candidate_elo"])), reverse=True)
    return {
        "evaluations": evaluations,
        "elapsed_sec": float(res["elapsed_sec"]),
        "mode": "mixed4way-no-incumbent",
        "raw": {k: v for k, v in res.items() if k != "results"},
    }


def _eval_league(candidate_bin: Path, members: list[dict], incumbent_bin: Path,
                 original_bin: Path, games_per_member: int, workers: int,
                 seed_base: int, league_mode: str,
                 extra_exclude_bins: list[Path] | None = None) -> dict:
    if league_mode == "pairwise2v2":
        return _eval_league_pairwise(candidate_bin, members, games_per_member, workers, seed_base)
    if league_mode == "mixed4way":
        return _eval_league_mixed4way(
            candidate_bin,
            members,
            incumbent_bin,
            original_bin,
            games_per_member,
            workers,
            seed_base,
            extra_exclude_bins,
        )
    if league_mode in ("mixed4way-no-incumbent", "mixed4way_no_incumbent"):
        league = _eval_league_candidates_mixed4way_no_incumbent(
            [{"bin": str(candidate_bin), "pt": "", "source_tag": "candidate", "proxy_score": 0.0}],
            members,
            incumbent_bin,
            original_bin,
            games_per_member,
            workers,
            seed_base,
            extra_exclude_bins,
        )
        return league["evaluations"][0]
    raise ValueError(f"unknown league_mode={league_mode}")


def _prod_args(args: argparse.Namespace, setup: Setup, iteration: int,
               incumbent_pt: Path, fitness_opponent_bin: Path, trial_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seed_checkpoint=str(incumbent_pt),
        opponent_bin=str(fitness_opponent_bin),
        work_dir=str(trial_dir),
        out_pt=str(trial_dir / "promoted.pt"),
        out_bin=str(trial_dir / "promoted.bin"),
        scope=setup.scope,
        rank=setup.rank,
        generations=setup.generations,
        pairs=setup.pairs,
        sigma=setup.sigma,
        lr=setup.lr,
        relative_noise=setup.relative_noise,
        update_clip_rms=args.update_clip_rms,
        mode=args.mode,
        mix_2v2_weight=args.mix_2v2_weight,
        games=setup.proxy_games,
        promote_top=setup.promote_top or args.promote_top,
        promote_games=0 if args.skip_previous_gate else args.previous_games_per_side,
        min_promote_delta=0.0 if args.skip_previous_gate else args.previous_min_delta,
        require_improvement=True,
        workers=args.workers,
        seed_base=args.seed_base + iteration * 1_000_000,
        seed=args.seed + iteration * 10_007,
        weight_format="fp32",
        candidate_batch_size=setup.candidate_batch_size,
        clean_work_dir=True,
    )


def _prune_population_dirs(trial_dir: Path) -> None:
    for path in trial_dir.glob("gen_*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default="autoresearch-results/eggroll_m2_hillclimb")
    p.add_argument("--incumbent-pt", default="csrc/nn_weights_candidate.pt")
    p.add_argument("--incumbent-bin", default="csrc/nn_weights_candidate.bin")
    p.add_argument(
        "--table-incumbent-bin",
        default="",
        help="Fixed PSRO table incumbent. Defaults to --incumbent-bin/live incumbent.",
    )
    p.add_argument(
        "--exclude-seed-incumbent-from-league",
        action="store_true",
        help="When --table-incumbent-bin differs from the seed incumbent, keep the seed incumbent out of the variable league slot.",
    )
    p.add_argument("--original-bin", default="csrc/nn_weights_m2.bin")
    p.add_argument("--current-metric", type=float, default=None)
    p.add_argument(
        "--primary-metric",
        choices=("league_elo", "original_wr", "same_table_league_delta"),
        default="league_elo",
        help="Metric used for promotion. same_table_league_delta compares candidate vs incumbent in the exact same mixed tables.",
    )
    p.add_argument("--min-league-elo-delta", type=float, default=0.0)
    p.add_argument(
        "--min-same-table-league-delta",
        type=float,
        default=0.007,
        help="For same_table_league_delta, keep if candidate league WR beats same-table incumbent WR by more than this.",
    )
    p.add_argument(
        "--same-table-confirm-margin",
        type=float,
        default=0.002,
        help="Replay near-miss same-table candidates within this margin below the keep gate and combine the batches.",
    )
    p.add_argument(
        "--same-table-confirm-games-per-side",
        type=int,
        default=0,
        help="Near-miss confirmation games per league member; 0 reuses --league-games-per-side.",
    )
    p.add_argument("--min-original-delta", type=float, default=0.003)
    p.add_argument("--previous-min-delta", type=float, default=0.01)
    p.add_argument("--skip-previous-gate", action="store_true",
                   help="Use proxy selection only; PSRO league + original confirmation do the real gating")
    p.add_argument("--previous-games-per-side", type=int, default=200,
                   help="paired evaluator games per side; 200 means 400 total")
    p.add_argument("--original-games-per-side", type=int, default=800,
                   help="paired evaluator games per side; 800 means 1600 total")
    p.add_argument("--stop-wilson-lower", type=float, default=0.5)
    p.add_argument("--stop-winrate", type=float, default=0.53)
    p.add_argument("--promote-top", type=int, default=8)
    p.add_argument("--league-size", type=int, default=10)
    p.add_argument("--league-games-per-side", type=int, default=80,
                   help="league seeds/games unit; mixed4way runs all four candidate seats per seed")
    p.add_argument(
        "--league-mode",
        choices=("mixed4way", "mixed4way-no-incumbent", "pairwise2v2"),
        default="mixed4way",
    )
    p.add_argument(
        "--league-min-winrate",
        type=float,
        default=0.0,
        help="Optional normalized-score floor. 0 disables the old 51%% cutoff.",
    )
    p.add_argument("--league-rerank-top", type=int, default=48,
                   help="When skipping the pairwise gate, PSRO-rerank this many top proxy candidates")
    p.add_argument("--league-adaptive-min-top", type=int, default=16,
                   help="Minimum PSRO candidates after adaptive proxy funneling")
    p.add_argument("--league-adaptive-weak-best-score", type=float, default=0.625,
                   help="Cap PSRO candidates when the best proxy score is below this value")
    p.add_argument("--league-adaptive-weak-top", type=int, default=16,
                   help="Maximum PSRO candidates for weak proxy rounds")
    p.add_argument("--league-adaptive-mid-best-score", type=float, default=0.67,
                   help="Cap PSRO candidates when the best proxy score is below this medium value")
    p.add_argument("--league-adaptive-mid-top", type=int, default=32,
                   help="Maximum PSRO candidates for medium proxy rounds")
    p.add_argument("--no-league-adaptive-rerank", dest="league_adaptive_rerank",
                   action="store_false",
                   help="Disable adaptive proxy funneling before PSRO")
    p.set_defaults(league_adaptive_rerank=True)
    p.add_argument("--no-league-staged-rerank", dest="league_staged_rerank",
                   action="store_false",
                   help="Disable successive-halving PSRO rerank")
    p.set_defaults(league_staged_rerank=True)
    p.add_argument("--league-stage-games", default="20,20,40",
                   help="Comma-separated mixed4way seeds/member for successive-halving stages")
    p.add_argument("--league-stage-keeps", default="12,3",
                   help="Comma-separated survivor counts after each non-final stage")
    p.add_argument("--no-plateau-schedule", dest="plateau_schedule",
                   action="store_false",
                   help="Disable automated LR/sigma decay and deep-scope schedule switching")
    p.set_defaults(plateau_schedule=True)
    p.add_argument("--plateau-decay-discards", type=int, default=3,
                   help="Trigger LR/sigma decay after this many consecutive gauntlet discards")
    p.add_argument("--plateau-decay-factor", type=float, default=0.5,
                   help="Multiplier applied to LR and sigma for each plateau decay event")
    p.add_argument("--plateau-unfreeze-min-lr", type=float, default=0.02,
                   help="Switch from final heads to deep trunk scopes once scaled head LR falls below this")
    p.add_argument("--league-diversity-interval", type=int, default=10,
                   help="Run response-correlation league curation every N completed gauntlet iterations; 0 disables")
    p.add_argument("--league-diversity-corr", type=float, default=0.90,
                   help="Evict one non-anchor league member when response correlation exceeds this")
    p.add_argument("--league-diversity-min-common", type=int, default=24,
                   help="Minimum shared candidate-response observations before correlation curation")
    p.add_argument("--league-diversity-history", type=int, default=20,
                   help="Recent league-rerank iterations used for member response vectors")
    p.add_argument("--league-diversity-replacement-iters", default="45,68,80",
                   help="Preferred historical keeper eras for diversity replacement")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--mode", choices=("paired2v2", "ffa", "mix"), default="paired2v2")
    p.add_argument("--mix-2v2-weight", type=float, default=0.5)
    p.add_argument("--update-clip-rms", type=float, default=0.10)
    p.add_argument("--seed-base", type=int, default=7_200_000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-iterations", type=int, default=0,
                   help="0 means unlimited")
    args = p.parse_args()
    args.league_stage_games_list = _parse_int_csv(args.league_stage_games, [20, 20, 40])
    if args.league_staged_rerank:
        stage_total = sum(args.league_stage_games_list)
        if stage_total < args.league_games_per_side:
            args.league_stage_games_list.append(args.league_games_per_side - stage_total)
        args.league_stage_games_list = [games for games in args.league_stage_games_list if games > 0]
    args.league_stage_keeps_list = _parse_int_csv(args.league_stage_keeps, [12, 3])
    args.league_diversity_replacement_iters = _parse_int_csv(
        args.league_diversity_replacement_iters,
        [45, 68, 80],
    )

    root = (PROJECT_ROOT / args.work_dir).resolve() if not Path(args.work_dir).is_absolute() else Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "hillclimb.jsonl"
    plateau_state = _load_plateau_state(
        log_path,
        args.plateau_decay_factor,
        args.plateau_unfreeze_min_lr,
    )
    incumbent_pt = Path(args.incumbent_pt).resolve()
    incumbent_bin = Path(args.incumbent_bin).resolve()
    original_bin = Path(args.original_bin).resolve()
    live_pt = root / "incumbent.pt"
    live_bin = root / "incumbent.bin"
    _copy_pair(incumbent_pt, incumbent_bin, live_pt, live_bin)
    table_incumbent_bin = (
        Path(args.table_incumbent_bin).resolve()
        if str(args.table_incumbent_bin).strip()
        else live_bin
    )
    if not table_incumbent_bin.exists():
        raise FileNotFoundError(f"--table-incumbent-bin does not exist: {table_incumbent_bin}")

    if args.primary_metric == "league_elo":
        state_metric = _load_state_metric(1500.0)
        default_metric = state_metric if state_metric > 10.0 else 1500.0
        current_metric = float(args.current_metric if args.current_metric is not None else default_metric)
        current_rating = current_metric
        helper_metric_floor = current_metric
    elif args.primary_metric == "same_table_league_delta":
        state_metric = _load_state_metric(1500.0)
        helper_metric_floor = state_metric if state_metric > 10.0 else 1500.0
        current_metric = float(args.current_metric if args.current_metric is not None else 0.0)
        current_rating = helper_metric_floor
    else:
        current_arg_metric = float(args.current_metric if args.current_metric is not None else 0.5064174107142857)
        current_metric = max(current_arg_metric, _load_state_metric(current_arg_metric))
        current_rating = _elo_from_score(current_metric, 1500.0)
        helper_metric_floor = current_metric
    league_members = _seed_league(root, original_bin, live_bin, current_rating, args.league_size)
    if args.primary_metric == "same_table_league_delta":
        live_hash = _file_hash(live_bin)
        for member in league_members:
            if member.get("sha256") == live_hash:
                try:
                    current_rating = float(member.get("rating", current_rating))
                except (TypeError, ValueError):
                    pass
                break
    table_incumbent_hash = _file_hash(table_incumbent_bin)
    if not any(m.get("sha256") == table_incumbent_hash for m in league_members):
        league_members = _add_league_member(
            league_members,
            table_incumbent_bin,
            root / "league",
            "table_incumbent",
            1500.0,
            "seed-table-incumbent",
        )
        _save_league_manifest(root / "league" / "manifest.json", league_members)
    extra_league_excludes: list[Path] = []
    if args.exclude_seed_incumbent_from_league and _file_hash(table_incumbent_bin) != _file_hash(live_bin):
        extra_league_excludes.append(live_bin)
    league_games_per_member = (4 if args.league_mode.startswith("mixed4way") else 2) * args.league_games_per_side
    start = time.time()
    _write_jsonl(log_path, {
        "event": "start",
        "current_metric": current_metric,
        "primary_metric": args.primary_metric,
        "current_rating": current_rating,
        "min_league_elo_delta": args.min_league_elo_delta,
        "min_same_table_league_delta": args.min_same_table_league_delta,
        "min_original_delta": args.min_original_delta,
        "previous_gate_total_games": 2 * args.previous_games_per_side,
        "skip_previous_gate": args.skip_previous_gate,
        "original_confirm_total_games": 2 * args.original_games_per_side,
        "league_members": league_members,
        "league_games_per_member": league_games_per_member,
        "league_mode": args.league_mode,
        "league_min_winrate": args.league_min_winrate,
        "league_adaptive_rerank": args.league_adaptive_rerank,
        "league_adaptive_min_top": args.league_adaptive_min_top,
        "league_adaptive_weak_best_score": args.league_adaptive_weak_best_score,
        "league_adaptive_weak_top": args.league_adaptive_weak_top,
        "league_adaptive_mid_best_score": args.league_adaptive_mid_best_score,
        "league_adaptive_mid_top": args.league_adaptive_mid_top,
        "league_staged_rerank": args.league_staged_rerank,
        "league_stage_games": args.league_stage_games_list,
        "league_stage_keeps": args.league_stage_keeps_list,
        "plateau_schedule": args.plateau_schedule,
        "plateau_state": plateau_state.__dict__,
        "plateau_decay_discards": args.plateau_decay_discards,
        "plateau_decay_factor": args.plateau_decay_factor,
        "plateau_unfreeze_min_lr": args.plateau_unfreeze_min_lr,
        "league_diversity_interval": args.league_diversity_interval,
        "league_diversity_corr": args.league_diversity_corr,
        "league_diversity_min_common": args.league_diversity_min_common,
        "league_diversity_history": args.league_diversity_history,
        "league_diversity_replacement_iters": args.league_diversity_replacement_iters,
        "table_incumbent_bin": str(table_incumbent_bin),
        "extra_league_excludes": [str(path) for path in extra_league_excludes],
        "setups": [s.__dict__ for s in SETUPS],
        "deep_setups": [s.__dict__ for s in DEEP_SETUPS],
        "active_setups": [
            s.__dict__ for s in _active_setups(
                plateau_state,
                args.plateau_decay_factor,
                args.plateau_unfreeze_min_lr,
                args.plateau_schedule,
            )
        ],
    })
    print(json.dumps({
        "event": "start",
        "current_metric": current_metric,
        "primary_metric": args.primary_metric,
        "current_rating": current_rating,
        "min_same_table_league_delta": args.min_same_table_league_delta,
        "previous_gate_total_games": 2 * args.previous_games_per_side,
        "original_confirm_total_games": 2 * args.original_games_per_side,
        "league_members": len(league_members),
        "league_mode": args.league_mode,
        "league_adaptive_rerank": args.league_adaptive_rerank,
        "league_adaptive_weak_best_score": args.league_adaptive_weak_best_score,
        "league_adaptive_weak_top": args.league_adaptive_weak_top,
        "league_adaptive_mid_best_score": args.league_adaptive_mid_best_score,
        "league_adaptive_mid_top": args.league_adaptive_mid_top,
        "league_staged_rerank": args.league_staged_rerank,
        "league_stage_games": args.league_stage_games_list,
        "league_stage_keeps": args.league_stage_keeps_list,
        "plateau_schedule": args.plateau_schedule,
        "plateau_state": plateau_state.__dict__,
        "league_diversity_interval": args.league_diversity_interval,
        "league_diversity_corr": args.league_diversity_corr,
        "table_incumbent_bin": str(table_incumbent_bin),
        "extra_league_excludes": len(extra_league_excludes),
        "skip_previous_gate": args.skip_previous_gate,
    }, sort_keys=True), flush=True)

    iteration = _next_iteration_start(log_path)
    while args.max_iterations <= 0 or iteration < args.max_iterations:
        active_setups = _active_setups(
            plateau_state,
            args.plateau_decay_factor,
            args.plateau_unfreeze_min_lr,
            args.plateau_schedule,
        )
        setup = active_setups[iteration % len(active_setups)]
        trial_dir = root / f"iter_{iteration + 1:04d}_{setup.name}"
        if setup.fitness_opponent not in ("previous", "original"):
            raise ValueError(f"unknown fitness_opponent={setup.fitness_opponent}")
        fitness_opponent_bin = original_bin if setup.fitness_opponent == "original" else live_bin
        t0 = time.time()
        prod_result = run_prod(_prod_args(args, setup, iteration + 1, live_pt, fitness_opponent_bin, trial_dir))
        _prune_population_dirs(trial_dir)
        promoted = bool(prod_result.get("passed_gate")) and bool(prod_result.get("out_bin"))
        row = {
            "event": "iteration",
            "iteration": iteration + 1,
            "setup": setup.__dict__,
            "prod": prod_result,
            "fitness_opponent": setup.fitness_opponent,
            "skip_previous_gate": args.skip_previous_gate,
            "promoted_by_fitness_gate": promoted,
            "promoted_by_previous_gate": (None if args.skip_previous_gate else promoted and setup.fitness_opponent == "previous"),
            "plateau_state": plateau_state.__dict__,
            "active_schedule_size": len(active_setups),
            "elapsed_sec": time.time() - t0,
        }

        if not promoted:
            metric = current_metric
            gate_name = "proxy selector" if args.skip_previous_gate else "400-game previous-incumbent gate"
            desc = (
                f"EGGROLL setup {setup.name} failed {gate_name}; "
                f"best promote score={prod_result.get('best_promote_score')}."
            )
            _record(
                "discard",
                metric,
                desc,
                ["base-m2-only", "no-search", "eggroll-prod", "proxy-selector" if args.skip_previous_gate else "previous-gate"],
            )
            row["decision"] = "discard_previous_gate"
            row["current_metric"] = current_metric
            _write_jsonl(log_path, row)
            print(_json_dumps(row), flush=True)
            iteration += 1
            continue

        promoted_pt = Path(prod_result["out_pt"]).resolve()
        promoted_bin = Path(prod_result["out_bin"]).resolve()
        if setup.fitness_opponent == "original" and not args.skip_previous_gate:
            previous_eval = _eval_pair(
                promoted_bin,
                live_bin,
                args.previous_games_per_side,
                args.workers,
                args.seed_base + (iteration + 1) * 1_000_000 + 250_000,
            )
            row["previous_eval"] = previous_eval
            previous_wr = float(previous_eval["winrate"])
            previous_gate = previous_wr >= 0.5 + args.previous_min_delta
            row["promoted_by_previous_gate"] = previous_gate
            if not previous_gate:
                desc = (
                    f"EGGROLL setup {setup.name} passed original-fitness gate but failed "
                    f"400-game previous-incumbent gate: {previous_wr:.4f} "
                    f"({previous_eval['wins']}/{previous_eval['games']})."
                )
                _record(
                    "discard",
                    current_metric,
                    desc,
                    ["base-m2-only", "no-search", "eggroll-prod", "previous-gate"],
                )
                row["decision"] = "discard_previous_gate"
                row["current_metric"] = current_metric
                _write_jsonl(log_path, row)
                print(_json_dumps(row), flush=True)
                iteration += 1
                continue

        league_rerank = None
        if args.skip_previous_gate and args.league_mode.startswith("mixed4way"):
            top_entries = [
                entry for entry in prod_result.get("top_candidates", [])
                if Path(str(entry.get("bin", ""))).exists()
                and Path(str(entry.get("pt", ""))).exists()
            ][:max(1, args.league_rerank_top)]
            top_entries, adaptive_meta = _select_adaptive_top_entries(
                top_entries,
                args.league_rerank_top,
                args.league_adaptive_min_top,
                args.league_adaptive_rerank,
                args.league_adaptive_weak_best_score,
                args.league_adaptive_weak_top,
                args.league_adaptive_mid_best_score,
                args.league_adaptive_mid_top,
            )
            if not top_entries:
                top_entries = [{
                    "bin": str(promoted_bin),
                    "pt": str(promoted_pt),
                    "source_tag": prod_result.get("best_source_tag", "promoted"),
                    "proxy_score": prod_result.get("best_promote_score", 0.0),
                }]
                adaptive_meta["selected_top"] = 1
                adaptive_meta["reason"] = "fallback_promoted"
            row["league_adaptive_funnel"] = adaptive_meta
            if args.league_mode == "mixed4way-no-incumbent":
                league_rerank = _eval_league_candidates_mixed4way_no_incumbent(
                    top_entries,
                    league_members,
                    table_incumbent_bin,
                    original_bin,
                    args.league_games_per_side,
                    args.workers,
                    args.seed_base + (iteration + 1) * 1_000_000 + 375_000,
                    extra_league_excludes,
                )
            else:
                if args.league_staged_rerank:
                    league_rerank = _eval_league_candidates_mixed4way_staged(
                        top_entries,
                        league_members,
                        table_incumbent_bin,
                        original_bin,
                        args.league_stage_games_list,
                        args.league_stage_keeps_list,
                        args.workers,
                        args.seed_base + (iteration + 1) * 1_000_000 + 375_000,
                        extra_league_excludes,
                    )
                else:
                    league_rerank = _eval_league_candidates_mixed4way(
                        top_entries,
                        league_members,
                        table_incumbent_bin,
                        original_bin,
                        args.league_games_per_side,
                        args.workers,
                        args.seed_base + (iteration + 1) * 1_000_000 + 375_000,
                        extra_league_excludes,
                    )
            league_eval = league_rerank["evaluations"][0]
            selected_entry = league_eval["candidate_entry"]
            selected_pt = Path(str(selected_entry["pt"])).resolve()
            selected_bin = Path(str(selected_entry["bin"])).resolve()
            promoted_pt = trial_dir / "league_selected.pt"
            promoted_bin = trial_dir / "league_selected.bin"
            _copy_pair(selected_pt, selected_bin, promoted_pt, promoted_bin)
            prod_result["out_pt"] = str(promoted_pt)
            prod_result["out_bin"] = str(promoted_bin)
            row["league_rerank"] = league_rerank
            row["league_selected_candidate"] = selected_entry
        else:
            league_eval = _eval_league(
                promoted_bin,
                league_members,
                table_incumbent_bin,
                original_bin,
                args.league_games_per_side,
                args.workers,
                args.seed_base + (iteration + 1) * 1_000_000 + 375_000,
                args.league_mode,
                extra_league_excludes,
            )
        row["league_eval"] = league_eval
        row["candidate_elo"] = league_eval["candidate_elo"]
        row["league_avg_winrate"] = league_eval["avg_winrate"]
        row["league_incumbent_winrate"] = league_eval.get("incumbent_winrate", 0.0)
        row["league_same_table_delta"] = league_eval.get("candidate_vs_incumbent_winrate_delta", 0.0)
        row["league_avg_score"] = league_eval.get("avg_score", league_eval["avg_winrate"])
        league_score = float(row["league_avg_score"])
        candidate_elo = float(league_eval["candidate_elo"])
        same_table_delta = float(row["league_same_table_delta"])
        if (
            args.primary_metric == "same_table_league_delta"
            and args.league_mode == "mixed4way"
            and args.same_table_confirm_margin > 0.0
            and same_table_delta < args.min_same_table_league_delta
            and same_table_delta >= args.min_same_table_league_delta - args.same_table_confirm_margin
        ):
            confirm_games = args.same_table_confirm_games_per_side or args.league_games_per_side
            confirm_eval = _eval_league_mixed4way(
                promoted_bin,
                league_members,
                table_incumbent_bin,
                original_bin,
                confirm_games,
                args.workers,
                args.seed_base + (iteration + 1) * 1_000_000 + 425_000,
                extra_league_excludes,
            )
            row["near_miss_confirmation"] = {
                "trigger_delta": same_table_delta,
                "gate": args.min_same_table_league_delta,
                "margin": args.same_table_confirm_margin,
                "games_per_side": confirm_games,
                "confirmation_eval": confirm_eval,
            }
            league_eval = _combine_same_table_evals(league_eval, confirm_eval)
            row["league_eval"] = league_eval
            row["candidate_elo"] = league_eval["candidate_elo"]
            row["league_avg_winrate"] = league_eval["avg_winrate"]
            row["league_incumbent_winrate"] = league_eval.get("incumbent_winrate", 0.0)
            row["league_same_table_delta"] = league_eval.get("candidate_vs_incumbent_winrate_delta", 0.0)
            row["league_avg_score"] = league_eval.get("avg_score", league_eval["avg_winrate"])
            league_score = float(row["league_avg_score"])
            candidate_elo = float(league_eval["candidate_elo"])
            same_table_delta = float(row["league_same_table_delta"])
        if args.league_min_winrate > 0.0 and league_score < args.league_min_winrate:
            desc = (
                f"EGGROLL setup {setup.name} passed incumbent gate but failed keeper-league gate: "
                f"{league_score:.4f} normalized score, {league_eval['avg_winrate']:.4f} raw WR, "
                f"Elo {candidate_elo:.1f}."
            )
            _record(
                "discard",
                candidate_elo if args.primary_metric == "league_elo" else current_metric,
                desc,
                ["base-m2-only", "no-search", "eggroll-prod", "league-gate"],
            )
            row["decision"] = "discard_league_gate"
            row["current_metric"] = current_metric
            _write_jsonl(log_path, row)
            print(_json_dumps(row), flush=True)
            iteration += 1
            continue

        original_eval = _eval_original(
            promoted_bin,
            original_bin,
            args.original_games_per_side,
            args.workers,
            args.seed_base + (iteration + 1) * 1_000_000 + 500_000,
        )
        row["original_eval"] = original_eval
        original_wr = float(original_eval["winrate"])
        if args.primary_metric == "league_elo":
            candidate_metric = candidate_elo
            min_delta = args.min_league_elo_delta
            improved = candidate_metric >= current_metric + min_delta
        elif args.primary_metric == "same_table_league_delta":
            candidate_metric = same_table_delta
            min_delta = args.min_same_table_league_delta
            improved = candidate_metric > min_delta
        else:
            candidate_metric = original_wr
            min_delta = args.min_original_delta
            improved = candidate_metric >= current_metric + min_delta
        row["candidate_metric"] = candidate_metric
        row["metric_delta"] = candidate_metric - (0.0 if args.primary_metric == "same_table_league_delta" else current_metric)
        row["min_required_delta"] = min_delta
        row["primary_metric"] = args.primary_metric
        significant = (
            original_eval["wilson_lower"] > args.stop_wilson_lower
            or original_wr >= args.stop_winrate
        )

        if improved:
            current_metric = candidate_metric
            if args.primary_metric == "league_elo":
                current_rating = candidate_elo
            elif args.primary_metric == "same_table_league_delta":
                current_rating = candidate_elo
            else:
                current_rating = _elo_from_score(original_wr, 1500.0)
            _copy_pair(promoted_pt, promoted_bin, live_pt, live_bin)
            _copy_pair(promoted_pt, promoted_bin,
                       root / f"kept_iter_{iteration + 1:04d}.pt",
                       root / f"kept_iter_{iteration + 1:04d}.bin")
            metric_tag = (
                f"{int(round(candidate_elo))}elo"
                if args.primary_metric == "league_elo"
                else f"{int(round(candidate_metric * 10000))}reld"
                if args.primary_metric == "same_table_league_delta"
                else str(int(original_wr * 10000))
            )
            league_members = _add_league_member(
                league_members,
                promoted_bin,
                root / "league",
                f"keep_{iteration + 1:04d}_{metric_tag}",
                current_rating,
                "keep",
            )
            if args.league_size > 0:
                league_members = _seed_league(root, original_bin, promoted_bin, current_rating, args.league_size)
            else:
                _save_league_manifest(root / "league" / "manifest.json", league_members)
            # Keep the public candidate files synchronized with the retained hillclimb incumbent.
            _copy_pair(promoted_pt, promoted_bin,
                       PROJECT_ROOT / "csrc" / "nn_weights_candidate.pt",
                       PROJECT_ROOT / "csrc" / "nn_weights_candidate.bin")
            status = "keep"
            if args.primary_metric == "league_elo":
                desc = (
                    f"EGGROLL setup {setup.name} improved policy-zoo Elo to {candidate_elo:.1f} "
                    f"(delta {row['metric_delta']:+.1f}); league score {league_score:.4f}, "
                    f"raw WR {league_eval['avg_winrate']:.4f}; original-M2 anchor "
                    f"{original_wr:.4f} ({original_eval['wins']}/{original_eval['games']}, "
                    f"Wilson lower {original_eval['wilson_lower']:.4f})."
                )
            elif args.primary_metric == "same_table_league_delta":
                desc = (
                    f"EGGROLL setup {setup.name} beat same-table incumbent in the PSRO league: "
                    f"candidate WR {league_eval['avg_winrate']:.4f} vs incumbent WR "
                    f"{league_eval.get('incumbent_winrate', 0.0):.4f} "
                    f"(delta {same_table_delta:+.4f}, gate > {args.min_same_table_league_delta:.4f}); "
                    f"diagnostic Elo {candidate_elo:.1f}; original-M2 anchor {original_wr:.4f} "
                    f"({original_eval['wins']}/{original_eval['games']}, Wilson lower "
                    f"{original_eval['wilson_lower']:.4f})."
                )
            else:
                desc = (
                    f"EGGROLL setup {setup.name} passed PSRO league gate "
                    f"and improved original-M2 confirmation to {original_wr:.4f} "
                    f"({original_eval['wins']}/{original_eval['games']}, "
                    f"Wilson lower {original_eval['wilson_lower']:.4f}); "
                    f"league score {league_score:.4f}, Elo {candidate_elo:.1f}."
                )
            labels = ["base-m2-only", "no-search", "eggroll-prod", "kept-candidate"]
            row["decision"] = "keep"
        else:
            status = "discard"
            if args.primary_metric == "league_elo":
                desc = (
                    f"EGGROLL setup {setup.name} did not improve policy-zoo Elo: "
                    f"{candidate_elo:.1f} vs retained {current_metric:.1f} "
                    f"(delta {row['metric_delta']:+.1f}); original-M2 anchor "
                    f"{original_wr:.4f} ({original_eval['wins']}/{original_eval['games']})."
                )
                labels = ["base-m2-only", "no-search", "eggroll-prod", "league-elo"]
                row["decision"] = "discard_league_elo"
            elif args.primary_metric == "same_table_league_delta":
                desc = (
                    f"EGGROLL setup {setup.name} failed same-table PSRO gate: "
                    f"candidate WR {league_eval['avg_winrate']:.4f} vs incumbent WR "
                    f"{league_eval.get('incumbent_winrate', 0.0):.4f} "
                    f"(delta {same_table_delta:+.4f}, need > {args.min_same_table_league_delta:.4f}); "
                    f"diagnostic Elo {candidate_elo:.1f}; original-M2 anchor "
                    f"{original_wr:.4f} ({original_eval['wins']}/{original_eval['games']})."
                )
                labels = ["base-m2-only", "no-search", "eggroll-prod", "same-table-league"]
                row["decision"] = "discard_same_table_league"
            else:
                desc = (
                    f"EGGROLL setup {setup.name} passed PSRO league gate but failed original-M2 "
                    f"confirmation: {original_wr:.4f} ({original_eval['wins']}/{original_eval['games']}), "
                    f"retained metric remains {current_metric:.4f}."
                )
                labels = ["base-m2-only", "no-search", "eggroll-prod", "original-confirm"]
                row["decision"] = "discard_original_confirm"

        record_metric = candidate_metric
        if status == "keep" and args.primary_metric == "same_table_league_delta":
            # The autoresearch helper's retained metric is monotone, while the
            # real promotion rule is relative to the incumbent inside this
            # batch.  Store a monotone audit scalar so helper bookkeeping does
            # not reintroduce the old absolute Elo threshold.
            helper_metric_floor = max(helper_metric_floor, _load_state_metric(helper_metric_floor))
            record_metric = helper_metric_floor + max(candidate_metric, 1e-12)
            row["helper_metric"] = record_metric
            row["helper_metric_floor"] = helper_metric_floor
        _record(status, record_metric, desc, labels)
        if status == "keep" and args.primary_metric == "same_table_league_delta":
            helper_metric_floor = max(helper_metric_floor, record_metric)
        row["current_metric"] = current_metric
        row["current_rating"] = current_rating
        row["significant_vs_original"] = significant and improved
        row["elapsed_total_sec"] = time.time() - start
        _write_jsonl(log_path, row)
        print(_json_dumps(row), flush=True)
        plateau_state = _update_plateau_state(row, plateau_state, args, log_path)
        league_members, _ = _curate_league_diversity(
            league_members,
            root,
            log_path,
            original_bin,
            live_bin,
            table_incumbent_bin,
            args,
            iteration + 1,
        )

        iteration += 1
        if significant and improved and args.primary_metric == "original_wr":
            print(json.dumps({
                "event": "stop",
                "reason": "significant_vs_original",
                "current_metric": current_metric,
                "iteration": iteration,
                "log_path": str(log_path),
            }, sort_keys=True), flush=True)
            return

        if iteration % 10 == 0:
            print(json.dumps({
                "event": "protocol_fingerprint_check",
                "iteration": iteration,
                "baseline_before_init": True,
                "logged_before_next_experiment": True,
                "helper_owned_state": True,
                "primary_metric": args.primary_metric,
                "plateau_state": plateau_state.__dict__,
                "active_schedule": [
                    s.name for s in _active_setups(
                        plateau_state,
                        args.plateau_decay_factor,
                        args.plateau_unfreeze_min_lr,
                        args.plateau_schedule,
                    )
                ],
                "stop_condition": (
                    "manual interrupt"
                    if args.primary_metric in ("league_elo", "same_table_league_delta")
                    else "significant original-M2 edge or manual interrupt"
                ),
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
