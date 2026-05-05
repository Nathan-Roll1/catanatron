#!/usr/bin/env python3
"""Create zero-runtime checkpoint soups from strong 0-ply policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from human_bot.export_nn import export as export_nn  # noqa: E402


DEFAULT_MODELS = {
    "sd0054": ROOT / "autoresearch-results/search_distill_m2/kept_iter_0054.pt",
    "eg0150": ROOT / "autoresearch-results/eggroll_m2_hillclimb/kept_iter_0150.pt",
    "sd0034": ROOT / "autoresearch-results/search_distill_m2/kept_iter_0034.pt",
    "sd0029": ROOT / "autoresearch-results/search_distill_m2/kept_iter_0029.pt",
    "eg0143": ROOT / "autoresearch-results/eggroll_m2_hillclimb/kept_iter_0143.pt",
}


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _soup(base: dict[str, Any], parts: list[tuple[str, dict[str, Any], float]], metadata: dict[str, Any]) -> dict[str, Any]:
    state = base["model_state_dict"]
    out_state: dict[str, torch.Tensor] = {}
    total = sum(weight for _label, _ckpt, weight in parts)
    for key, value in state.items():
        if torch.is_floating_point(value):
            acc = torch.zeros_like(value, dtype=torch.float32)
            for _label, ckpt, weight in parts:
                acc += ckpt["model_state_dict"][key].detach().cpu().float() * (weight / total)
            out_state[key] = acc.to(dtype=value.dtype)
        else:
            out_state[key] = value.detach().cpu().clone()
    return {
        "config": base["config"],
        "model_state_dict": out_state,
        "metadata": metadata,
        "model_type": base.get("model_type", "HumanBotNet"),
    }


def _candidate_specs() -> list[tuple[str, list[tuple[str, float]]]]:
    specs: list[tuple[str, list[tuple[str, float]]]] = []
    for other in ["eg0150", "sd0034", "sd0029", "eg0143"]:
        for alpha in [0.25, 0.50, 0.75]:
            pct = int(alpha * 100)
            specs.append((f"soup_sd0054_{other}_a{pct:02d}", [("sd0054", 1.0 - alpha), (other, alpha)]))
    specs.extend([
        ("soup_top3_uniform", [("sd0054", 1.0), ("eg0150", 1.0), ("sd0034", 1.0)]),
        ("soup_top4_uniform", [("sd0054", 1.0), ("eg0150", 1.0), ("sd0034", 1.0), ("sd0029", 1.0)]),
        ("soup_top5_uniform", [("sd0054", 1.0), ("eg0150", 1.0), ("sd0034", 1.0), ("sd0029", 1.0), ("eg0143", 1.0)]),
        ("soup_weighted_top4", [("sd0054", 4.0), ("eg0150", 3.0), ("sd0034", 2.0), ("sd0029", 1.0)]),
    ])
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ROOT / "autoresearch-results/search_distill_m2/soups_0064"))
    parser.add_argument("--weight-format", choices=["fp32", "fp16", "int8"], default="fp16")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = {label: _load(path) for label, path in DEFAULT_MODELS.items() if path.exists()}
    missing = sorted(set(DEFAULT_MODELS) - set(ckpts))
    if missing:
        raise FileNotFoundError(f"Missing checkpoints for soup generation: {missing}")
    base = ckpts["sd0054"]

    manifest = []
    for name, weights in _candidate_specs():
        parts = [(label, ckpts[label], weight) for label, weight in weights]
        metadata = {
            "stage": "zero_ply_policy_soup",
            "name": name,
            "parts": [{"label": label, "weight": weight} for label, _ckpt, weight in parts],
        }
        ckpt = _soup(base, parts, metadata)
        pt_path = out_dir / f"{name}.pt"
        bin_path = out_dir / f"{name}.bin"
        torch.save(ckpt, pt_path)
        export_nn(str(pt_path), str(bin_path), weight_format=args.weight_format, write_test_vectors=False)
        manifest.append({"name": name, "pt": str(pt_path), "bin": str(bin_path), "parts": metadata["parts"]})
        print(f"wrote {name}: {bin_path}", flush=True)

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
