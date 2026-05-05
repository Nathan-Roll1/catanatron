"""Base-M2-only Eggroll-style low-rank ES experiment.

This is deliberately small and mechanical:
  - no search teacher
  - no super-M2 labels
  - candidate fitness comes from 0-ply NN-vs-NN games against frozen M2

The first target is policy-head-only perturbation because it avoids changing
the expensive encoder/trunk path while still changing decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_CKPT = PROJECT_ROOT / "checkpoints" / "sp_latest2.pt"
DEFAULT_BASE_BIN = PROJECT_ROOT / "csrc" / "nn_weights_m2.bin"
DEFAULT_OUT_BIN = PROJECT_ROOT / "csrc" / "nn_weights_candidate.bin"


def _selected_keys(state: dict[str, torch.Tensor], scope: str) -> list[str]:
    keys = []
    for k, v in state.items():
        if not torch.is_floating_point(v):
            continue
        if scope == "policy_head" and not k.startswith("policy_head."):
            continue
        if scope == "policy_head_final" and not (
            k.startswith("policy_head.type_fc.3.")
            or k.startswith("policy_head.discard_yop_mono_fc.2.")
            or k.startswith("policy_head.maritime_fc.2.")
            or k.startswith("policy_head.trade_fc.2.")
            or k.startswith("policy_head.settlement_scorer.2.")
            or k.startswith("policy_head.city_scorer.2.")
            or k.startswith("policy_head.road_scorer.2.")
            or k.startswith("policy_head.robber_scorer.2.")
        ):
            continue
        if scope == "value_head" and not k.startswith("value_head."):
            continue
        keys.append(k)
    return keys


def _low_rank_noise(t: torch.Tensor, rank: int, gen: torch.Generator) -> torch.Tensor:
    if t.ndim == 2 and min(t.shape) > 1:
        out_dim, in_dim = t.shape
        r = max(1, min(rank, out_dim, in_dim))
        a = torch.randn((out_dim, r), generator=gen, dtype=t.dtype)
        b = torch.randn((r, in_dim), generator=gen, dtype=t.dtype)
        # Keep perturbation RMS roughly independent of rank/input width.
        return (a @ b) / ((r * in_dim) ** 0.5)
    return torch.randn(t.shape, generator=gen, dtype=t.dtype) / max(1.0, t.numel() ** 0.5)


def _noise_tree(state: dict[str, torch.Tensor], keys: list[str], rank: int,
                seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return {k: _low_rank_noise(state[k].cpu(), rank, gen) for k in keys}


def _apply_noise(base: dict[str, torch.Tensor],
                 noise: dict[str, torch.Tensor],
                 sigma: float,
                 sign: float) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in base.items()}
    for k, n in noise.items():
        out[k] = out[k] + float(sign * sigma) * n.to(out[k].dtype)
    return out


def _save_checkpoint_from_state(seed_checkpoint: Path,
                                state: dict[str, torch.Tensor],
                                path: Path,
                                metadata: dict) -> None:
    from human_bot.model import HumanBotNet

    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    net.load_state_dict(state, strict=False)
    net.save_checkpoint(str(path), metadata)


def _export(checkpoint: Path, output_bin: Path) -> None:
    from human_bot.export_nn import export

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    export(str(checkpoint), str(output_bin), weight_format="fp32")


def _eval_2v2(candidate_bin: Path, base_bin: Path, games: int, workers: int,
              seed_base: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "human_bot.eval_2v2_nn_fast",
        "--a-weights",
        str(candidate_bin),
        "--b-weights",
        str(base_bin),
        "--games",
        str(games),
        "--workers",
        str(workers),
        "--seed-base",
        str(seed_base),
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.time() - t0
    out = proc.stdout
    m = re.search(r"A wins:\s+(\d+)/(\d+)\s+\(([0-9.]+)%\)", out)
    if proc.returncode != 0 or not m:
        return {
            "ok": False,
            "returncode": proc.returncode,
            "winrate": 0.0,
            "wins": 0,
            "games": games,
            "elapsed_sec": elapsed,
            "output": out[-4000:],
        }
    wins = int(m.group(1))
    total = int(m.group(2))
    return {
        "ok": True,
        "returncode": proc.returncode,
        "winrate": wins / max(1, total),
        "wins": wins,
        "games": total,
        "elapsed_sec": elapsed,
        "output": out[-4000:],
    }


def _policy_drift(candidate_bin: Path, base_bin: Path, seed_base: int,
                  seeds: int, steps: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "human_bot.compare_policy_bins",
        "--a-weights",
        str(candidate_bin),
        "--b-weights",
        str(base_bin),
        "--seeds",
        str(seeds),
        "--steps",
        str(steps),
        "--seed-base",
        str(seed_base),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"states": 0, "disagreement": 0.0, "illegal_rate": 1.0, "output": proc.stdout[-1000:]}


def _passes_drift_gate(drift: dict, args: argparse.Namespace) -> bool:
    disagreement = float(drift.get("disagreement", 0.0))
    illegal_rate = float(drift.get("illegal_rate", 1.0))
    states = int(drift.get("states", 0))
    if states <= 0:
        return False
    return (
        disagreement >= args.min_disagreement
        and disagreement <= args.max_disagreement
        and illegal_rate <= args.max_illegal_rate
    )


def _eval_or_skip(candidate_bin: Path, base_bin: Path, drift: dict,
                  args: argparse.Namespace, seed_base: int) -> dict:
    if args.skip_game_on_drift and not _passes_drift_gate(drift, args):
        return {
            "ok": True,
            "returncode": 0,
            "winrate": 0.0,
            "wins": 0,
            "games": 0,
            "elapsed_sec": 0.0,
            "skipped": True,
            "skip_reason": "drift_gate",
        }
    ev = _eval_2v2(candidate_bin, base_bin, args.games, args.workers, seed_base)
    ev["skipped"] = False
    return ev


def _state_update(base: dict[str, torch.Tensor],
                  noises: list[dict[str, torch.Tensor]],
                  fitness_diffs: list[float],
                  keys: list[str],
                  lr: float,
                  sigma: float) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in base.items()}
    if not noises:
        return out
    denom = max(1, len(noises)) * max(float(sigma), 1e-12) * 2.0
    for k in keys:
        acc = torch.zeros_like(out[k])
        for noise, diff in zip(noises, fitness_diffs):
            acc += float(diff) * noise[k].to(acc.dtype)
        out[k] = out[k] + float(lr) * acc / denom
    return out


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    seed_checkpoint = Path(args.seed_checkpoint).resolve()
    base_bin = Path(args.base_bin).resolve()
    out_bin = Path(args.out_bin).resolve()
    tmp_root = Path(args.tmp_dir).resolve() if args.tmp_dir else Path(tempfile.mkdtemp(prefix="eggroll-m2-"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    from human_bot.model import HumanBotNet

    net = HumanBotNet.load_checkpoint(str(seed_checkpoint), device="cpu")
    base_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    keys = _selected_keys(base_state, args.scope)
    if not keys:
        raise RuntimeError(f"No floating parameters selected for scope={args.scope}")

    best = {"winrate": -1.0, "state": None, "tag": "none", "eval": None}
    pair_noises: list[dict[str, torch.Tensor]] = []
    pair_diffs: list[float] = []
    evals = []
    start = time.time()

    for pair_idx in range(args.pairs):
        noise = _noise_tree(base_state, keys, args.rank, args.seed + 1009 * pair_idx)
        pair = {}
        for sign, name in ((1.0, "plus"), (-1.0, "minus")):
            state = _apply_noise(base_state, noise, args.sigma, sign)
            ckpt = tmp_root / f"candidate_pair{pair_idx}_{name}.pt"
            bin_path = tmp_root / f"candidate_pair{pair_idx}_{name}.bin"
            _save_checkpoint_from_state(
                seed_checkpoint,
                state,
                ckpt,
                {
                    "eggroll": True,
                    "scope": args.scope,
                    "rank": args.rank,
                    "sigma": args.sigma,
                    "pair": pair_idx,
                    "sign": name,
                },
            )
            _export(ckpt, bin_path)
            drift = _policy_drift(
                bin_path,
                base_bin,
                args.seed_base + pair_idx * 100 + (0 if sign > 0 else 50),
                args.drift_seeds,
                args.drift_steps,
            )
            ev = _eval_or_skip(bin_path, base_bin, drift, args,
                               args.seed_base + pair_idx * 100 + (0 if sign > 0 else 50))
            ev["candidate"] = f"pair{pair_idx}_{name}"
            ev["policy_disagreement"] = drift.get("disagreement", 0.0)
            ev["policy_illegal_rate"] = drift.get("illegal_rate", 1.0)
            ev["policy_drift_states"] = drift.get("states", 0)
            evals.append({k: v for k, v in ev.items() if k != "output"})
            pair[name] = ev["winrate"] if ev["ok"] else 0.0
            if ev["ok"] and not ev.get("skipped") and ev["winrate"] > best["winrate"]:
                best = {"winrate": ev["winrate"], "state": state, "tag": f"pair{pair_idx}_{name}", "eval": ev}

        pair_noises.append(noise)
        pair_diffs.append(pair.get("plus", 0.0) - pair.get("minus", 0.0))

    es_state = _state_update(base_state, pair_noises, pair_diffs, keys, args.lr, args.sigma)
    es_ckpt = tmp_root / "candidate_es_update.pt"
    es_bin = tmp_root / "candidate_es_update.bin"
    _save_checkpoint_from_state(
        seed_checkpoint,
        es_state,
        es_ckpt,
        {
            "eggroll": True,
            "scope": args.scope,
            "rank": args.rank,
            "sigma": args.sigma,
            "lr": args.lr,
            "pairs": args.pairs,
            "selection": "es_update",
        },
    )
    _export(es_ckpt, es_bin)
    es_drift = _policy_drift(es_bin, base_bin, args.seed_base + 999000,
                             args.drift_seeds, args.drift_steps)
    es_eval = _eval_or_skip(es_bin, base_bin, es_drift, args, args.seed_base + 9000)
    es_eval["candidate"] = "es_update"
    es_eval["policy_disagreement"] = es_drift.get("disagreement", 0.0)
    es_eval["policy_illegal_rate"] = es_drift.get("illegal_rate", 1.0)
    es_eval["policy_drift_states"] = es_drift.get("states", 0)
    evals.append({k: v for k, v in es_eval.items() if k != "output"})
    if es_eval["ok"] and not es_eval.get("skipped") and es_eval["winrate"] > best["winrate"]:
        best = {"winrate": es_eval["winrate"], "state": es_state, "tag": "es_update", "eval": es_eval}

    selected_state = es_state if args.selection == "es" and not es_eval.get("skipped") else best["state"]
    if selected_state is None:
        selected_state = base_state
    selected_tag = best["tag"] if best["state"] is not None else "base"
    out_ckpt = out_bin.with_suffix(".pt")
    _save_checkpoint_from_state(
        seed_checkpoint,
        selected_state,
        out_ckpt,
        {
            "eggroll": True,
            "scope": args.scope,
            "rank": args.rank,
            "sigma": args.sigma,
            "lr": args.lr,
            "pairs": args.pairs,
            "selection": args.selection,
            "selected": selected_tag,
        },
    )
    _export(out_ckpt, out_bin)

    if args.clean_tmp and not args.tmp_dir:
        shutil.rmtree(tmp_root, ignore_errors=True)

    return {
        "ok": True,
        "score": float(best["winrate"] if best["state"] is not None else 0.0),
        "selected": selected_tag,
        "selection_mode": args.selection,
        "pairs": args.pairs,
        "population": args.pairs * 2,
        "rank": args.rank,
        "sigma": args.sigma,
        "lr": args.lr,
        "scope": args.scope,
        "games_per_candidate": args.games,
        "workers": args.workers,
        "elapsed_sec": time.time() - start,
        "out_bin": str(out_bin),
        "evals": evals,
        "drift_gate": {
            "skip_game_on_drift": args.skip_game_on_drift,
            "min_disagreement": args.min_disagreement,
            "max_disagreement": args.max_disagreement,
            "max_illegal_rate": args.max_illegal_rate,
            "drift_seeds": args.drift_seeds,
            "drift_steps": args.drift_steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-checkpoint", default=str(DEFAULT_SEED_CKPT))
    parser.add_argument("--base-bin", default=str(DEFAULT_BASE_BIN))
    parser.add_argument("--out-bin", default=str(DEFAULT_OUT_BIN))
    parser.add_argument("--scope", choices=("policy_head", "policy_head_final", "value_head"),
                        default="policy_head_final")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drift-seeds", type=int, default=4)
    parser.add_argument("--drift-steps", type=int, default=30)
    parser.add_argument("--min-disagreement", type=float, default=0.0)
    parser.add_argument("--max-disagreement", type=float, default=1.0)
    parser.add_argument("--max-illegal-rate", type=float, default=0.0)
    parser.add_argument("--skip-game-on-drift", action="store_true")
    parser.add_argument("--seed-base", type=int, default=5000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=("best", "es"), default="best")
    parser.add_argument("--tmp-dir", default="")
    parser.add_argument("--clean-tmp", action="store_true")
    args = parser.parse_args()

    result = run(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
