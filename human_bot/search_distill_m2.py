#!/usr/bin/env python3
"""Winner-trajectory search distillation for the M2 Catan policy.

This is intentionally separate from the EGGROLL loop.  Each iteration:

1. Build a compact 6-slot league: M2, AB2, incumbent, and 3 rotating keepers.
2. Generate mixed 4-player games from that league.
3. Save only the move trajectory of the player that actually won each game.
4. Behavioral-clone the incumbent toward those winner moves.
5. Evaluate candidate vs incumbent in a compact league table with AB2 pinned
   at Elo 1000.  Keep only if the candidate's Elo beats the incumbent's.

AB2 is not a privileged teacher.  It is just one policy in the league; its
trajectory is cloned only in games it wins.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
EGGROLL_DIR = ROOT / "autoresearch-results" / "eggroll_m2_hillclimb"
DEFAULT_WORK = ROOT / "autoresearch-results" / "search_distill_m2"
AD = 337
MASK_DIM = 397
MODEL_BYTES = 16 * 1024 * 1024
FP = ctypes.POINTER(ctypes.c_float)

_G: dict[str, Any] = {}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class LeagueEntry:
    label: str
    kind: str  # "nn" or "ab2"
    bin_path: str | None = None
    pt_path: str | None = None


def _abs(path: str | Path | None) -> str | None:
    return None if path is None else str(Path(path).expanduser().resolve())


def _default_incumbent_bin() -> Path:
    # Iteration 143 was the high-water keeper in the prior EGGROLL stocktake.
    return EGGROLL_DIR / "kept_iter_0143.bin"


def _default_incumbent_pt() -> Path:
    return EGGROLL_DIR / "kept_iter_0143.pt"


def _kept_iter_paths() -> list[tuple[int, str, Path, Path]]:
    out: list[tuple[int, str, Path, Path]] = []
    sources = [
        ("eg", 0, EGGROLL_DIR),
        ("sd", 10000, DEFAULT_WORK),
    ]
    for prefix, offset, directory in sources:
        if not directory.exists():
            continue
        for bin_path in sorted(directory.glob("kept_iter_*.bin")):
            stem = bin_path.stem
            try:
                idx = int(stem.split("_")[-1])
            except ValueError:
                continue
            pt_path = bin_path.with_suffix(".pt")
            if pt_path.exists():
                out.append((offset + idx, f"{prefix}{idx:04d}", bin_path, pt_path))
    return sorted(out, key=lambda row: row[0])


def build_training_league(
    iteration: int,
    incumbent_bin: Path,
    incumbent_pt: Path,
    include_hs: bool = False,
) -> list[LeagueEntry]:
    """Return exactly six slots: M2, AB2, three rotating keepers, incumbent."""
    incumbent_bin = incumbent_bin.resolve()
    kept = [
        (sort_idx, label, b, p) for sort_idx, label, b, p in _kept_iter_paths()
        if b.resolve() != incumbent_bin
    ]
    if not kept:
        raise FileNotFoundError(f"No historical keepers found in {EGGROLL_DIR}")

    # Rotate through the full keeper pool so old styles keep reappearing, but
    # always include one prior search-distilled keeper when available.  This
    # prevents the new loop from forgetting its own recent successful styles.
    picked: list[tuple[int, str, Path, Path]] = []
    sd_pool = [row for row in kept if row[1].startswith("sd")]
    if sd_pool:
        picked.append(sd_pool[iteration % len(sd_pool)])
    start = (iteration * 3) % len(kept)
    for i in range(len(kept)):
        row = kept[(start + i) % len(kept)]
        if row not in picked:
            picked.append(row)
        if len(picked) >= 3:
            break
    while len(picked) < 3:
        picked.append(picked[-1])

    league = [
        LeagueEntry("m2", "nn", _abs(CSRC / "nn_weights_m2.bin"), _abs(CSRC / "nn_weights_candidate.pt")),
        LeagueEntry("ab2", "ab2"),
    ]
    keeper_slots = 3
    if include_hs:
        league.append(LeagueEntry("hs", "hs"))
        keeper_slots = 2
    for _sort_idx, label, bin_path, pt_path in picked[:keeper_slots]:
        league.append(LeagueEntry(label, "nn", _abs(bin_path), _abs(pt_path)))
    league.append(LeagueEntry("incumbent", "nn", _abs(incumbent_bin), _abs(incumbent_pt)))
    return league


def build_eval_roster(
    training_league: list[LeagueEntry],
    candidate_bin: Path | None = None,
    candidate_pt: Path | None = None,
) -> list[LeagueEntry]:
    """Keep the evaluated roster capped at 6 slots.

    If a candidate is present, it replaces the third rotating keeper so the
    table still includes M2, AB2, two historical keepers, incumbent, candidate.
    """
    if candidate_bin is None:
        return list(training_league[:6])
    by_label = {e.label: e for e in training_league}
    roster: list[LeagueEntry] = []
    for label in ("m2", "ab2", "hs"):
        if label in by_label:
            roster.append(by_label[label])
    keeper_slots = max(0, 5 - len(roster) - 1)  # reserve incumbent before candidate
    for e in training_league:
        if e.label in {"m2", "ab2", "hs", "incumbent"}:
            continue
        if keeper_slots <= 0:
            break
        roster.append(e)
        keeper_slots -= 1
    if "incumbent" in by_label:
        roster.append(by_label["incumbent"])
    roster.append(LeagueEntry("candidate", "nn", _abs(candidate_bin), _abs(candidate_pt)))
    return roster


def _resolve_libnn_path() -> str:
    import platform

    hostname = platform.node().split(".")[0]
    candidates = [
        CSRC / f"libnn_{hostname}.so",
        CSRC / "libnn.so",
        CSRC / "libnn.dylib",
        ROOT / "catan_player" / "libcatan_nn.so",
        ROOT / "catan_player" / "libcatan_nn.dylib",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"No libnn found, tried: {[str(p) for p in candidates]}")


def _load_model(lib: ctypes.CDLL, weights: str) -> tuple[Any, ctypes.c_void_p]:
    buf = (ctypes.c_char * MODEL_BYTES)()
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    rc = lib.nn_load(ptr, weights.encode())
    if rc != 0:
        raise RuntimeError(f"nn_load failed for {weights}: {rc}")
    return buf, ptr


def _init_worker(roster_json: str, ab_depth: int, hs_depth: int, hs_time_ms: float) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Action, MAX_ACTIONS, SearchCtx, ValueFn
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    nn_lib = ctypes.CDLL(_resolve_libnn_path())
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]

    entries = [LeagueEntry(**x) for x in json.loads(roster_json)]
    model_bufs: dict[str, Any] = {}
    model_ptrs: dict[str, ctypes.c_void_p] = {}
    for entry in entries:
        if entry.kind == "nn":
            assert entry.bin_path is not None
            buf, ptr = _load_model(nn_lib, entry.bin_path)
            model_bufs[entry.label] = buf
            model_ptrs[entry.label] = ptr

    hs_bots: dict[str, Any] = {}
    if any(entry.kind == "hs" for entry in entries):
        from human_bot.superbot_v3_c2 import SuperBotV3C2
        for entry in entries:
            if entry.kind == "hs":
                hs_bots[entry.label] = SuperBotV3C2(
                    str(CSRC / "nn_weights_m2.bin"),
                    our_depth=int(hs_depth),
                    top_k_schedule=(6, 4, 2, 2, 2, 2),
                    entropy_fast_thresh=0.15,
                    leaf_cache_bits=18,
                    time_budget_ms=float(hs_time_ms),
                    opponent_ab_depth=int(ab_depth),
                    leaf_mode=4,
                    algo_policy=True,
                    opponent_model="ab2",
                )

    lib = load_library()
    ab_eval = ValueFn(lib.base_value_fn)

    _G.clear()
    _G.update({
        "entries": {entry.label: entry for entry in entries},
        "labels": [entry.label for entry in entries],
        "nn_lib": nn_lib,
        "model_bufs": model_bufs,
        "model_ptrs": model_ptrs,
        "hs_bots": hs_bots,
        "ae": ActionEncoder(),
        "CatanGame": CatanGame,
        "lib": lib,
        "Action": Action,
        "MAX_ACTIONS": MAX_ACTIONS,
        "SearchCtx": SearchCtx,
        "ValueFn": ValueFn,
        "ab_eval": ab_eval,
        "ab_depth": int(ab_depth),
    })


def _nn_choose(
    label: str,
    legal: list[Any],
    se: Any,
    nf: np.ndarray,
    ef: np.ndarray,
    ff: np.ndarray,
    mask: np.ndarray,
    out: np.ndarray,
) -> int:
    ae = _G["ae"]
    nn_lib = _G["nn_lib"]
    ptr = _G["model_ptrs"][label]
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mask.ctypes.data_as(FP)
    outp = out.ctypes.data_as(ctypes.c_void_p)
    nn_lib.nn_forward(ptr, nfp, efp, ffp, mkp, outp)
    logits = out[4:4 + MASK_DIM].copy()
    logits[mask < 0.5] = -1e9
    policy_idx = int(np.argmax(logits))
    for i, action in enumerate(legal):
        try:
            if ae.encode(action) == policy_idx:
                return i
        except ValueError:
            continue
    return 0


def _ab2_choose(game: Any, legal: list[Any]) -> int:
    lib = _G["lib"]
    Action = _G["Action"]
    MAX_ACTIONS = _G["MAX_ACTIONS"]
    SearchCtx = _G["SearchCtx"]
    ab_buf = (Action * MAX_ACTIONS)()
    for i, action in enumerate(legal):
        ab_buf[i] = action
    cg = game._game
    cur_color = cg.state.colors[cg.state.current_player_index]
    ctx = SearchCtx()
    res = lib.alphabeta_search(
        ctypes.byref(ctx), ctypes.byref(cg), ab_buf,
        ctypes.c_int(len(legal)), ctypes.c_int(_G["ab_depth"]),
        ctypes.c_double(-1e30), ctypes.c_double(1e30),
        ctypes.c_int(cur_color), _G["ab_eval"],
    )
    chosen_bytes = ctypes.string_at(ctypes.byref(res.action), ctypes.sizeof(res.action))
    for i, action in enumerate(legal):
        if ctypes.string_at(ctypes.byref(action), ctypes.sizeof(action)) == chosen_bytes:
            return i
    return 0


def _hs_choose(label: str, game: Any, legal: list[Any]) -> int:
    bot = _G["hs_bots"][label]
    try:
        chosen = int(bot.pick(game))
    except Exception:
        return 0
    if 0 <= chosen < len(legal):
        return chosen
    return 0


def _mask397(mask_raw: np.ndarray) -> np.ndarray:
    mask = np.zeros(MASK_DIM, dtype=np.float32)
    n = min(MASK_DIM, int(mask_raw.shape[0]))
    mask[:n] = mask_raw[:n]
    return mask


def _play_logged_game(seed: int, seat_labels: list[str]) -> dict[str, Any]:
    CatanGame = _G["CatanGame"]
    ae = _G["ae"]
    entries = _G["entries"]

    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
    out = np.zeros(4 + MASK_DIM, dtype=np.float32)
    by_seat: list[list[dict[str, Any]]] = [[], [], [], []]

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        cp = game.current_player()
        label = seat_labels[cp]

        se.encode_into(game.get_state_view(), nf, ef, ff)
        mask = _mask397(ae.get_action_mask(legal).numpy())

        if len(legal) == 1:
            chosen = 0
        elif entries[label].kind == "ab2":
            chosen = _ab2_choose(game, legal)
        elif entries[label].kind == "hs":
            chosen = _hs_choose(label, game, legal)
        else:
            chosen = _nn_choose(label, legal, se, nf, ef, ff, mask, out)

        try:
            action_idx = int(ae.encode(legal[chosen]))
        except ValueError:
            game.step(chosen)
            continue
        if 0 <= action_idx < MASK_DIM:
            mask[action_idx] = 1.0
            by_seat[cp].append({
                "nf": nf.copy(),
                "ef": ef.copy(),
                "ff": ff.copy(),
                "mask": mask.copy(),
                "action_idx": action_idx,
                "player": cp,
                "teacher": label,
                "turn": int(game.turn_number),
            })
        game.step(chosen)

    winner = game.winner()
    winner_label = seat_labels[winner] if winner is not None else None
    return {
        "winner": winner,
        "winner_label": winner_label,
        "seat_labels": seat_labels,
        "turns": game.turn_number,
        "steps": by_seat[winner] if winner is not None else [],
    }


def _save_steps_shard(steps: list[dict[str, Any]], path: Path) -> int:
    if not steps:
        return 0
    reward_vec = np.zeros(4, dtype=np.float32)
    winner = int(steps[0]["player"])
    reward_vec[winner] = 1.0
    data = {
        "node_features": torch.from_numpy(np.stack([s["nf"] for s in steps]).astype(np.float32)),
        "edge_features": torch.from_numpy(np.stack([s["ef"] for s in steps]).astype(np.float32)),
        "flat_features": torch.from_numpy(np.stack([s["ff"] for s in steps]).astype(np.float32)),
        "action_mask": torch.from_numpy(np.stack([s["mask"] for s in steps]).astype(np.float32)),
        "action_idx": torch.tensor([s["action_idx"] for s in steps], dtype=torch.int64),
        "player": torch.tensor([s["player"] for s in steps], dtype=torch.int64),
        "turn": torch.tensor([s.get("turn", -1) for s in steps], dtype=torch.int64),
        "reward_vec": torch.from_numpy(np.repeat(reward_vec[None, :], len(steps), axis=0)),
        "num_players": torch.full((len(steps),), 4, dtype=torch.int64),
        "teacher_label": [s["teacher"] for s in steps],
    }
    tmp = str(path) + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)
    return len(steps)


def _combo_for_game(labels: list[str], game_idx: int, seed: int) -> list[str]:
    combos = list(itertools.combinations(labels, 4))
    combo = list(combos[(game_idx + seed) % len(combos)])
    rng = random.Random((seed << 8) ^ (game_idx * 0x9E3779B1))
    rng.shuffle(combo)
    return combo


def _generate_chunk(job: tuple[int, int, int, int, str]) -> dict[str, Any]:
    chunk_idx, start_game, n_games, seed_base, out_dir_raw = job
    labels = _G["labels"]
    out_dir = Path(out_dir_raw)
    all_steps: list[dict[str, Any]] = []
    wins: dict[str, int] = {label: 0 for label in labels}
    games_done = 0
    total_turns = 0
    for offset in range(n_games):
        game_idx = start_game + offset
        seed = seed_base + game_idx
        seat_labels = _combo_for_game(labels, game_idx, seed)
        row = _play_logged_game(seed, seat_labels)
        games_done += 1
        total_turns += int(row["turns"])
        if row["winner_label"] is not None:
            wins[row["winner_label"]] = wins.get(row["winner_label"], 0) + 1
            all_steps.extend(row["steps"])
    shard_path = out_dir / f"winner_shard_{chunk_idx:05d}.pt"
    n_steps = _save_steps_shard(all_steps, shard_path)
    return {
        "chunk": chunk_idx,
        "games": games_done,
        "steps": n_steps,
        "wins": wins,
        "avg_turns": total_turns / max(1, games_done),
        "shard": str(shard_path) if n_steps else None,
    }


def generate_winner_data(
    roster: list[LeagueEntry],
    out_dir: Path,
    games: int,
    workers: int,
    seed_base: int,
    games_per_shard: int,
    ab_depth: int,
    hs_depth: int,
    hs_time_ms: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("winner_shard_*.pt"):
        old.unlink()
    roster_json = json.dumps([asdict(e) for e in roster])
    jobs = []
    for start in range(0, games, games_per_shard):
        jobs.append((len(jobs), start, min(games_per_shard, games - start), seed_base, str(out_dir)))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(roster_json, ab_depth, hs_depth, hs_time_ms),
    ) as pool:
        for row in pool.imap_unordered(_generate_chunk, jobs):
            rows.append(row)
            print(
                f"  data chunk {row['chunk']:03d}: {row['games']}g "
                f"{row['steps']} winner steps avg_turns={row['avg_turns']:.1f}",
                flush=True,
            )

    wins: dict[str, int] = {entry.label: 0 for entry in roster}
    for row in rows:
        for label, count in row["wins"].items():
            wins[label] = wins.get(label, 0) + int(count)
    total_steps = sum(int(row["steps"]) for row in rows)
    return {
        "games": games,
        "steps": total_steps,
        "wins": wins,
        "elapsed_sec": time.time() - t0,
        "shards": sorted(str(p) for p in out_dir.glob("winner_shard_*.pt")),
    }


def _detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_distill_dataset(
    shard_dir: Path,
    max_examples: int = 0,
    teacher_balanced: bool = False,
    teacher_cap_ratio: float = 1.0,
    strategic_only: bool = False,
    opening_turns: int = 0,
    top_teachers: int = 0,
    search_only: bool = False,
    teacher_allowlist: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    files = sorted(shard_dir.glob("winner_shard_*.pt"))
    if not files:
        raise FileNotFoundError(f"No winner shards in {shard_dir}")
    chunks: dict[str, list[torch.Tensor]] = {
        "node_features": [],
        "edge_features": [],
        "flat_features": [],
        "action_mask": [],
        "action_idx": [],
        "turn": [],
        "signal_kind": [],
    }
    teacher_labels: list[str] = []
    total = 0
    for path in files:
        data = torch.load(path, weights_only=False, map_location="cpu")
        n = int(data["action_idx"].shape[0])
        take = n
        if max_examples > 0 and total + n > max_examples:
            take = max(0, max_examples - total)
        if take <= 0:
            break
        for key in chunks:
            if key == "turn" and key not in data:
                t = torch.full((take,), -1, dtype=torch.int64)
            elif key == "signal_kind" and key not in data:
                t = torch.zeros((take,), dtype=torch.int64)
            else:
                t = data[key][:take]
            if key == "action_mask" and t.shape[-1] < MASK_DIM:
                pad = torch.zeros(t.shape[0], MASK_DIM - t.shape[-1], dtype=t.dtype)
                t = torch.cat([t, pad], dim=-1)
            chunks[key].append(t)
        labels = data.get("teacher_label")
        if labels is None:
            teacher_labels.extend(["unknown"] * take)
        else:
            teacher_labels.extend([str(x) for x in labels[:take]])
        total += take
        if max_examples > 0 and total >= max_examples:
            break
    out = {key: torch.cat(vals, dim=0) for key, vals in chunks.items()}
    if search_only:
        search = out["signal_kind"] == 0
        if search.any():
            out = {key: val[search] for key, val in out.items()}
            teacher_labels = [
                label for label, keep in zip(teacher_labels, search.cpu().numpy().tolist())
                if keep
            ]
    if opening_turns > 0:
        opening = (out["turn"] >= 0) & (out["turn"] <= opening_turns)
        if opening.any():
            out = {key: val[opening] for key, val in out.items()}
            teacher_labels = [
                label for label, keep in zip(teacher_labels, opening.cpu().numpy().tolist())
                if keep
            ]
    if strategic_only:
        strategic = out["action_idx"] >= 2
        if strategic.any():
            out = {key: val[strategic] for key, val in out.items()}
            teacher_labels = [
                label for label, keep in zip(teacher_labels, strategic.cpu().numpy().tolist())
                if keep
            ]
    if top_teachers > 0 and teacher_labels:
        counts = collections.Counter(teacher_labels)
        keep_labels = {
            label for label, _count in counts.most_common(max(1, top_teachers))
        }
        teacher_keep = [label in keep_labels for label in teacher_labels]
        if any(teacher_keep):
            idx = torch.tensor(teacher_keep, dtype=torch.bool)
            out = {key: val[idx] for key, val in out.items()}
            teacher_labels = [
                label for label, keep in zip(teacher_labels, teacher_keep)
                if keep
            ]
    if teacher_allowlist and teacher_labels:
        teacher_keep = [label in teacher_allowlist for label in teacher_labels]
        if any(teacher_keep):
            idx = torch.tensor(teacher_keep, dtype=torch.bool)
            out = {key: val[idx] for key, val in out.items()}
            teacher_labels = [
                label for label, keep in zip(teacher_labels, teacher_keep)
                if keep
            ]
    if teacher_balanced and teacher_labels:
        by_label: dict[str, list[int]] = {}
        for i, label in enumerate(teacher_labels):
            by_label.setdefault(label, []).append(i)
        if len(by_label) > 1:
            min_count = min(len(v) for v in by_label.values())
            cap_count = max(min_count, int(math.ceil(min_count * max(1.0, teacher_cap_ratio))))
            rng = np.random.default_rng(12345)
            picked: list[int] = []
            for label in sorted(by_label):
                rows = np.asarray(by_label[label], dtype=np.int64)
                if rows.shape[0] > cap_count:
                    rows = rng.choice(rows, size=cap_count, replace=False)
                picked.extend(int(x) for x in rows)
            rng.shuffle(picked)
            idx = torch.tensor(picked, dtype=torch.long)
            out = {key: val[idx] for key, val in out.items()}
    out["action_mask"][torch.arange(out["action_idx"].shape[0]), out["action_idx"]] = 1.0
    out.pop("turn", None)
    out.pop("signal_kind", None)
    return out


def train_bc_candidate(
    incumbent_pt: Path,
    shard_dir: Path,
    out_pt: Path,
    out_bin: Path,
    *,
    device: str,
    lr: float,
    epochs: int,
    batch_size: int,
    max_examples: int,
    train_scope: str,
    label_smoothing: float,
    weight_format: str,
    use_action_weights: bool,
    teacher_balanced: bool,
    teacher_cap_ratio: float,
    strategic_only: bool,
    opening_turns: int,
    top_teachers: int,
    search_only: bool,
    teacher_allowlist: str,
    kl_alpha: float,
    disagreement_boost: float,
) -> dict[str, Any]:
    from hexzero.game.interface import CatanGame
    from human_bot.export_nn import export as export_nn
    from human_bot.loss import _build_action_weights, human_policy_loss
    from human_bot.model import HumanBotNet

    device = _detect_device(device)
    data = _load_distill_dataset(
        shard_dir,
        max_examples=max_examples,
        teacher_balanced=teacher_balanced,
        teacher_cap_ratio=teacher_cap_ratio,
        strategic_only=strategic_only,
        opening_turns=opening_turns,
        top_teachers=top_teachers,
        search_only=search_only,
        teacher_allowlist=(
            {x.strip() for x in teacher_allowlist.split(",") if x.strip()}
            if teacher_allowlist else None
        ),
    )
    n = int(data["action_idx"].shape[0])
    if n < 32:
        raise RuntimeError(f"Too few distillation examples: {n}")

    g0 = CatanGame(seed=0)
    g0.reset()
    edge_index = g0.make_state_encoder()._edge_index.to(device)

    net = HumanBotNet.load_checkpoint(str(incumbent_pt), device=device)
    anchor_net = None
    if kl_alpha > 0.0:
        anchor_net = HumanBotNet.load_checkpoint(str(incumbent_pt), device=device)
        anchor_net.eval()
        for param in anchor_net.parameters():
            param.requires_grad = False
    for param in net.parameters():
        param.requires_grad = False
    if train_scope == "policy_type":
        prefixes = ("policy_head.type_fc.",)
    elif train_scope == "opening_spatial":
        prefixes = ("policy_head.settlement_scorer.", "policy_head.road_scorer.")
    elif train_scope == "policy_head":
        prefixes = ("policy_head.",)
    elif train_scope == "policy_trunk":
        prefixes = ("trunk.", "policy_head.")
    elif train_scope == "policy_all":
        prefixes = ("board_encoder.", "trunk.", "policy_head.")
    else:
        raise ValueError(f"Unknown train scope: {train_scope}")
    for name, param in net.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = True

    net.train()
    if train_scope in {"policy_head", "policy_trunk", "opening_spatial"}:
        net.board_encoder.eval()
    if train_scope in {"policy_type", "policy_head", "opening_spatial"}:
        net.trunk.eval()
    net.value_head.eval()
    net.vp_head.eval()

    dev = {
        "node_features": data["node_features"].to(device),
        "edge_features": data["edge_features"].to(device),
        "flat_features": data["flat_features"].to(device),
        "action_mask": data["action_mask"].to(device),
        "action_idx": data["action_idx"].to(device),
    }
    del data

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    action_weights = _build_action_weights(MASK_DIM, device) if use_action_weights else None
    losses: list[float] = []
    bc_losses: list[float] = []
    kl_losses: list[float] = []
    accs: list[float] = []
    t0 = time.time()

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_losses: list[float] = []
        epoch_bc_losses: list[float] = []
        epoch_kl_losses: list[float] = []
        epoch_accs: list[float] = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.shape[0] < 8:
                continue
            batch = {
                "node_features": dev["node_features"][idx],
                "edge_index": edge_index,
                "edge_features": dev["edge_features"][idx],
                "flat_features": dev["flat_features"][idx],
                "action_mask": dev["action_mask"][idx],
            }
            action_idx = dev["action_idx"][idx]
            out = net(batch)
            example_weights = None
            if disagreement_boost > 0.0:
                with torch.no_grad():
                    pred = out["policy_logits"].argmax(dim=-1)
                    wrong = (pred != action_idx).float()
                    example_weights = 1.0 + disagreement_boost * wrong
                    example_weights = example_weights / example_weights.mean().clamp(min=1e-6)
            bc_loss = human_policy_loss(
                out["raw_policy_logits"],
                action_idx,
                batch["action_mask"],
                label_smoothing=label_smoothing,
                action_weights=action_weights,
                example_weights=example_weights,
            )
            kl_loss = torch.zeros((), dtype=bc_loss.dtype, device=device)
            if anchor_net is not None:
                with torch.no_grad():
                    anchor_out = anchor_net(batch)
                    anchor_logp = F.log_softmax(anchor_out["policy_logits"], dim=-1)
                    anchor_prob = anchor_logp.exp()
                new_logp = F.log_softmax(out["policy_logits"], dim=-1)
                kl_per = anchor_prob * (anchor_logp - new_logp)
                kl_loss = (kl_per * batch["action_mask"]).sum(dim=-1).mean()
            loss = bc_loss + kl_alpha * kl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            with torch.no_grad():
                pred = out["policy_logits"].argmax(dim=-1)
                epoch_accs.append(float((pred == action_idx).float().mean().item()))
            epoch_losses.append(float(loss.item()))
            epoch_bc_losses.append(float(bc_loss.item()))
            epoch_kl_losses.append(float(kl_loss.item()))
        losses.append(float(np.mean(epoch_losses)))
        bc_losses.append(float(np.mean(epoch_bc_losses)))
        kl_losses.append(float(np.mean(epoch_kl_losses)))
        accs.append(float(np.mean(epoch_accs)))
        print(
            f"  bc epoch {epoch + 1}/{epochs}: loss={losses[-1]:.4f} "
            f"bc={bc_losses[-1]:.4f} kl={kl_losses[-1]:.5f} "
            f"pacc={accs[-1]:.3f}",
            flush=True,
        )

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    net.eval()
    net.save_checkpoint(str(out_pt), {
        "stage": "winner_search_distill",
        "incumbent": str(incumbent_pt),
        "examples": n,
        "lr": lr,
        "epochs": epochs,
        "train_scope": train_scope,
        "use_action_weights": use_action_weights,
        "teacher_balanced": teacher_balanced,
        "teacher_cap_ratio": teacher_cap_ratio,
        "strategic_only": strategic_only,
        "opening_turns": opening_turns,
        "top_teachers": top_teachers,
        "search_only": search_only,
        "teacher_allowlist": teacher_allowlist,
        "kl_alpha": kl_alpha,
        "disagreement_boost": disagreement_boost,
    })
    export_nn(str(out_pt), str(out_bin), weight_format=weight_format, write_test_vectors=False)
    return {
        "examples": n,
        "loss": losses[-1],
        "bc_loss": bc_losses[-1],
        "kl_loss": kl_losses[-1],
        "policy_acc": accs[-1],
        "elapsed_sec": time.time() - t0,
        "out_pt": str(out_pt),
        "out_bin": str(out_bin),
    }


def _eval_job(job: tuple[int, int, int, int, list[str]]) -> dict[str, Any]:
    combo_idx, rep_idx, rot, seed, combo = job
    labels = list(combo)
    # Rotate seats deterministically so each label occupies each seat.
    labels = labels[rot:] + labels[:rot]
    row = _play_logged_game(seed, labels)
    return {
        "combo": combo_idx,
        "rep": rep_idx,
        "rot": rot,
        "seed": seed,
        "players": labels,
        "winner": row["winner_label"],
        "turns": row["turns"],
    }


def _fit_bt_elo(results: list[dict[str, Any]], labels: list[str], pinned_label: str = "ab2") -> dict[str, float]:
    pair_wins: dict[tuple[str, str], list[int]] = {}
    for row in results:
        winner = row["winner"]
        if winner is None:
            continue
        for loser in row["players"]:
            if loser == winner:
                continue
            a, b = sorted((winner, loser))
            vals = pair_wins.setdefault((a, b), [0, 0])
            vals[0 if winner == a else 1] += 1

    free = [label for label in labels if label != pinned_label]
    idx = {label: i for i, label in enumerate(free)}

    def unpack(params: np.ndarray) -> dict[str, float]:
        scores = {pinned_label: 0.0}
        for label in free:
            scores[label] = float(params[idx[label]])
        return scores

    def loss_fn(params: np.ndarray) -> float:
        scores = unpack(params)
        loss = 0.0
        for (a, b), (wa, wb) in pair_wins.items():
            sa, sb = scores[a], scores[b]
            den = np.logaddexp(sa, sb)
            loss -= wa * (sa - den)
            loss -= wb * (sb - den)
        loss += 0.001 * float(np.sum(params ** 2))
        return loss

    try:
        from scipy.optimize import minimize

        res = minimize(loss_fn, np.zeros(len(free), dtype=np.float64), method="L-BFGS-B")
        params = res.x
    except Exception:
        # Small dependency-free fallback: a few steps of finite-difference-free
        # Bradley-Terry gradient ascent anchored at the AB2 score.
        params = np.zeros(len(free), dtype=np.float64)
        lr = 0.02
        for _ in range(500):
            scores = unpack(params)
            grad = np.zeros_like(params)
            for (a, b), (wa, wb) in pair_wins.items():
                sa, sb = scores[a], scores[b]
                ea = math.exp(sa) / (math.exp(sa) + math.exp(sb))
                total = wa + wb
                ga = wa - total * ea
                if a != pinned_label:
                    grad[idx[a]] += ga
                if b != pinned_label:
                    grad[idx[b]] -= ga
            params += lr * (grad - 0.002 * params)
            lr *= 0.995

    scale = 400.0 / math.log(10.0)
    scores = unpack(params)
    return {label: 1000.0 + scores[label] * scale for label in labels}


def evaluate_roster(
    roster: list[LeagueEntry],
    games_per_combo: int,
    workers: int,
    seed_base: int,
    ab_depth: int,
    hs_depth: int,
    hs_time_ms: float,
) -> dict[str, Any]:
    labels = [e.label for e in roster]
    roster_json = json.dumps([asdict(e) for e in roster])
    combos = list(itertools.combinations(labels, 4))
    jobs = []
    for combo_idx, combo in enumerate(combos):
        for rep in range(games_per_combo):
            seed = seed_base + combo_idx * 10000 + rep
            for rot in range(4):
                jobs.append((combo_idx, rep, rot, seed, list(combo)))
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(roster_json, ab_depth, hs_depth, hs_time_ms),
    ) as pool:
        results = list(pool.imap_unordered(_eval_job, jobs))

    wins = {label: 0 for label in labels}
    played = {label: 0 for label in labels}
    for row in results:
        for label in row["players"]:
            played[label] += 1
        if row["winner"] is not None:
            wins[row["winner"]] += 1
    elo = _fit_bt_elo(results, labels, pinned_label="ab2" if "ab2" in labels else labels[0])
    rows = []
    for label in labels:
        rows.append({
            "label": label,
            "kind": next(e.kind for e in roster if e.label == label),
            "elo": 1000.0 if label == "ab2" else float(elo[label]),
            "games": played[label],
            "wins": wins[label],
            "wr": wins[label] / max(1, played[label]),
        })
    rows.sort(key=lambda r: -float(r["elo"]))
    return {
        "games": len(results),
        "games_per_combo": games_per_combo,
        "elapsed_sec": time.time() - t0,
        "rows": rows,
        "raw": results,
    }


def _print_elo_table(eval_res: dict[str, Any]) -> None:
    print("\nElo / WR table (AB2 pinned at 1000):")
    print(f"{'rank':>4}  {'model':<12} {'kind':<4} {'elo':>8} {'games':>6} {'wins':>5} {'wr':>7}")
    for i, row in enumerate(eval_res["rows"], 1):
        print(
            f"{i:>4}  {row['label']:<12} {row['kind']:<4} "
            f"{row['elo']:>8.1f} {row['games']:>6} {row['wins']:>5} {100*row['wr']:>6.1f}%"
        )
    print(f"eval games={eval_res['games']} elapsed={eval_res['elapsed_sec']:.1f}s\n", flush=True)


def run_iteration(args: argparse.Namespace) -> dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    incumbent_bin = Path(args.incumbent_bin).resolve()
    incumbent_pt = Path(args.incumbent_pt).resolve()
    training_league = build_training_league(
        args.iteration, incumbent_bin, incumbent_pt, include_hs=args.include_hs)
    iter_dir = work_dir / f"iter_{args.iteration:04d}"
    shard_dir = iter_dir / "shards"
    iter_dir.mkdir(parents=True, exist_ok=True)

    print("Training league:")
    for entry in training_league:
        print(f"  {entry.label:<12} {entry.kind:<3} {entry.bin_path or ''}")

    data_summary = generate_winner_data(
        training_league,
        shard_dir,
        games=args.data_games,
        workers=args.workers,
        seed_base=args.seed_base + args.iteration * 1_000_000,
        games_per_shard=args.games_per_shard,
        ab_depth=args.ab_depth,
        hs_depth=args.hs_depth,
        hs_time_ms=args.hs_time_ms,
    )
    print(
        f"Data: {data_summary['games']} games, {data_summary['steps']} winner steps, "
        f"{data_summary['elapsed_sec']:.1f}s, wins={data_summary['wins']}",
        flush=True,
    )

    candidate_pt = iter_dir / "candidate.pt"
    candidate_bin = iter_dir / "candidate.bin"
    train_summary = train_bc_candidate(
        incumbent_pt,
        shard_dir,
        candidate_pt,
        candidate_bin,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
        train_scope=args.train_scope,
        label_smoothing=args.label_smoothing,
        weight_format=args.weight_format,
        use_action_weights=args.action_weights,
        teacher_balanced=args.teacher_balanced,
        teacher_cap_ratio=args.teacher_cap_ratio,
        strategic_only=args.strategic_only,
        opening_turns=args.opening_turns,
        top_teachers=args.top_teachers,
        search_only=args.search_only,
        teacher_allowlist=args.teacher_allowlist,
        kl_alpha=args.kl_alpha,
        disagreement_boost=args.disagreement_boost,
    )

    eval_roster = build_eval_roster(training_league, candidate_bin, candidate_pt)
    quick_eval_res = evaluate_roster(
        eval_roster,
        games_per_combo=args.eval_games_per_combo,
        workers=args.workers,
        seed_base=args.eval_seed_base + args.iteration * 1_000_000,
        ab_depth=args.ab_depth,
        hs_depth=args.hs_depth,
        hs_time_ms=args.hs_time_ms,
    )
    _print_elo_table(quick_eval_res)

    eval_res = quick_eval_res
    by_label = {row["label"]: row for row in eval_res["rows"]}
    cand_elo = float(by_label["candidate"]["elo"])
    inc_elo = float(by_label["incumbent"]["elo"])
    keep = cand_elo > inc_elo
    confirm_res = None
    if keep and args.confirm_games_per_combo > args.eval_games_per_combo:
        print(
            f"Quick gate passed by {cand_elo - inc_elo:+.1f} Elo; "
            f"running confirmation gpc={args.confirm_games_per_combo}",
            flush=True,
        )
        confirm_res = evaluate_roster(
            eval_roster,
            games_per_combo=args.confirm_games_per_combo,
            workers=args.workers,
            seed_base=args.eval_seed_base + args.iteration * 1_000_000 + 500_000,
            ab_depth=args.ab_depth,
            hs_depth=args.hs_depth,
            hs_time_ms=args.hs_time_ms,
        )
        _print_elo_table(confirm_res)
        eval_res = confirm_res
        by_label = {row["label"]: row for row in eval_res["rows"]}
        cand_elo = float(by_label["candidate"]["elo"])
        inc_elo = float(by_label["incumbent"]["elo"])
        keep = cand_elo > inc_elo
    summary = {
        "iteration": args.iteration,
        "keep": keep,
        "candidate_elo": cand_elo,
        "incumbent_elo": inc_elo,
        "candidate_wr": by_label["candidate"]["wr"],
        "incumbent_wr": by_label["incumbent"]["wr"],
        "data": data_summary,
        "train": train_summary,
        "quick_eval": {k: v for k, v in quick_eval_res.items() if k != "raw"},
        "confirmation_eval": ({k: v for k, v in confirm_res.items() if k != "raw"} if confirm_res else None),
        "eval": {k: v for k, v in eval_res.items() if k != "raw"},
        "training_league": [asdict(e) for e in training_league],
        "eval_roster": [asdict(e) for e in eval_roster],
    }
    with open(iter_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if keep:
        kept_pt = work_dir / f"kept_iter_{args.iteration:04d}.pt"
        kept_bin = work_dir / f"kept_iter_{args.iteration:04d}.bin"
        shutil.copy2(candidate_pt, kept_pt)
        shutil.copy2(candidate_bin, kept_bin)
        summary["kept_pt"] = str(kept_pt)
        summary["kept_bin"] = str(kept_bin)
        print(
            f"KEEP: candidate Elo {cand_elo:.1f} > incumbent {inc_elo:.1f}; "
            f"saved {kept_bin}",
            flush=True,
        )
    else:
        print(
            f"DISCARD: candidate Elo {cand_elo:.1f} <= incumbent {inc_elo:.1f}",
            flush=True,
        )
    return summary


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    league = build_training_league(
        args.iteration, Path(args.incumbent_bin), Path(args.incumbent_pt),
        include_hs=args.include_hs)
    eval_res = evaluate_roster(
        league,
        games_per_combo=args.eval_games_per_combo,
        workers=args.workers,
        seed_base=args.eval_seed_base + args.iteration * 1_000_000,
        ab_depth=args.ab_depth,
        hs_depth=args.hs_depth,
        hs_time_ms=args.hs_time_ms,
    )
    _print_elo_table(eval_res)
    return eval_res


def run_candidate_eval(args: argparse.Namespace) -> dict[str, Any]:
    if not args.eval_candidate_bin:
        raise ValueError("--eval-candidate-bin is required for --eval-only")
    league = build_training_league(
        args.iteration, Path(args.incumbent_bin), Path(args.incumbent_pt),
        include_hs=args.include_hs)
    roster = build_eval_roster(
        league,
        Path(args.eval_candidate_bin),
        Path(args.eval_candidate_pt) if args.eval_candidate_pt else None,
    )
    eval_res = evaluate_roster(
        roster,
        games_per_combo=args.eval_games_per_combo,
        workers=args.workers,
        seed_base=args.eval_seed_base + args.iteration * 1_000_000,
        ab_depth=args.ab_depth,
        hs_depth=args.hs_depth,
        hs_time_ms=args.hs_time_ms,
    )
    _print_elo_table(eval_res)
    if args.eval_output:
        out_path = Path(args.eval_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({k: v for k, v in eval_res.items() if k != "raw"}, f, indent=2)
    return eval_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Winner-trajectory M2 search distillation loop")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK))
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--incumbent-bin", default=str(_default_incumbent_bin()))
    parser.add_argument("--incumbent-pt", default=str(_default_incumbent_pt()))
    parser.add_argument("--data-games", type=int, default=512)
    parser.add_argument("--games-per-shard", type=int, default=32)
    parser.add_argument("--eval-games-per-combo", type=int, default=8)
    parser.add_argument("--confirm-games-per-combo", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=880000000)
    parser.add_argument("--eval-seed-base", type=int, default=890000000)
    parser.add_argument("--ab-depth", type=int, default=2)
    parser.add_argument("--include-hs", action="store_true")
    parser.add_argument("--hs-depth", type=int, default=4)
    parser.add_argument("--hs-time-ms", type=float, default=50.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--train-scope", choices=["policy_type", "opening_spatial", "policy_head", "policy_trunk", "policy_all"], default="policy_head")
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--action-weights", action="store_true")
    parser.add_argument("--teacher-balanced", action="store_true")
    parser.add_argument("--teacher-cap-ratio", type=float, default=1.0)
    parser.add_argument("--strategic-only", action="store_true")
    parser.add_argument("--opening-turns", type=int, default=0)
    parser.add_argument("--top-teachers", type=int, default=0)
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--teacher-allowlist", default="")
    parser.add_argument("--kl-alpha", type=float, default=0.0)
    parser.add_argument("--disagreement-boost", type=float, default=0.0)
    parser.add_argument("--weight-format", choices=["fp32", "fp16", "int8"], default="fp16")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-candidate-bin", default="")
    parser.add_argument("--eval-candidate-pt", default="")
    parser.add_argument("--eval-output", default="")
    args = parser.parse_args()

    if args.eval_only:
        run_candidate_eval(args)
    elif args.baseline_only:
        run_baseline(args)
    else:
        run_iteration(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
