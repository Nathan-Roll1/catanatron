"""Evaluation metrics for the human-bot model.

Reports per-action-type accuracy, top-k accuracy, value calibration,
and optional live play evaluation against the AB2 baseline.
"""

from __future__ import annotations

import ctypes
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


# Action-type names and the index ranges they occupy in the 337-dim space.
ACTION_TYPE_RANGES: dict[str, tuple[int, int]] = {
    "singleton":   (0, 5),
    "settlement":  (5, 59),
    "city":        (59, 113),
    "road":        (113, 185),
    "robber":      (185, 280),
    "discard":     (280, 285),
    "yop":         (285, 305),
    "monopoly":    (305, 310),
    "maritime":    (310, 330),
    "trade_resp":  (330, 337),
    "trade_1for1": (337, 357),
    "trade_1for2": (357, 377),
    "trade_2for1": (377, 397),
}


def action_type_label(action_idx: int) -> str:
    """Return the human-readable action-type name for an action index."""
    for name, (lo, hi) in ACTION_TYPE_RANGES.items():
        if lo <= action_idx < hi:
            return name
    return "unknown"


# ======================================================================
# Offline metrics (on held-out tensor data)
# ======================================================================

@torch.inference_mode()
def compute_metrics(
    net: torch.nn.Module,
    dataset,
    edge_index: torch.Tensor,
    device: str,
    batch_size: int = 4096,
    top_k: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Compute comprehensive policy + value metrics on a dataset.

    Returns a flat dict suitable for logging to W&B / stdout.
    Uses inference_mode + autocast for maximum MPS throughput.
    """
    net.eval()
    n = len(dataset)
    if n == 0:
        return {}

    topk_correct = {k: 0 for k in top_k}
    per_type_correct: dict[str, int] = defaultdict(int)
    per_type_total: dict[str, int] = defaultdict(int)
    value_correct = 0
    brier_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = torch.arange(start, end)
        batch_data = dataset.get_batch(idx)
        B = end - start

        nf = batch_data["nf"].to(device, non_blocking=True)
        ef = batch_data["ef"].to(device, non_blocking=True)
        ff = batch_data["ff"].to(device, non_blocking=True)
        mask = batch_data["mask"].to(device, non_blocking=True)
        action_idx = batch_data["action_idx"].to(device, non_blocking=True)
        vt = batch_data["value_target"].to(device, non_blocking=True)

        out = net({
            "node_features": nf,
            "edge_index": edge_index,
            "edge_features": ef,
            "flat_features": ff,
            "action_mask": mask,
        })

        logits = out["policy_logits"]
        value_logits = out["value"]

        ploss = F.cross_entropy(logits, action_idx, reduction="sum")
        policy_loss_sum += ploss.item()

        vt_clamped = vt.clamp(min=0.0)
        vt_sum = vt_clamped.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vt_normed = vt_clamped / vt_sum
        vlp = F.log_softmax(value_logits, dim=-1)
        vloss = -(vt_normed * vlp).sum(dim=-1).sum()
        value_loss_sum += vloss.item()

        for k in top_k:
            _, topk_pred = logits.topk(k, dim=-1)
            topk_correct[k] += (topk_pred == action_idx.unsqueeze(1)).any(dim=1).sum().item()

        pred_winner = value_logits.argmax(dim=-1)
        true_winner = vt.argmax(dim=-1)
        value_correct += (pred_winner == true_winner).sum().item()

        value_probs = F.softmax(value_logits, dim=-1)
        brier_sum += ((value_probs - vt_normed) ** 2).sum(dim=-1).sum().item()

        action_idx_cpu = action_idx.cpu().numpy()
        pred_cpu = logits.argmax(dim=-1).cpu().numpy()
        for i in range(B):
            label = action_type_label(int(action_idx_cpu[i]))
            per_type_total[label] += 1
            if pred_cpu[i] == action_idx_cpu[i]:
                per_type_correct[label] += 1

        count += B

    metrics: dict[str, float] = {}
    for k in top_k:
        metrics[f"top{k}_acc"] = topk_correct[k] / max(count, 1)

    metrics["policy_loss"] = policy_loss_sum / max(count, 1)
    metrics["value_loss"] = value_loss_sum / max(count, 1)
    metrics["value_winner_acc"] = value_correct / max(count, 1)
    metrics["brier_score"] = brier_sum / max(count, 1)

    for label in sorted(per_type_total):
        total = per_type_total[label]
        correct = per_type_correct[label]
        metrics[f"acc/{label}"] = correct / max(total, 1)
        metrics[f"count/{label}"] = total

    return metrics


# ======================================================================
# Live play evaluation (HZ-human-bot vs AB2)
# ======================================================================

def evaluate_vs_ab2(
    net: torch.nn.Module,
    state_enc,
    action_enc,
    device: str,
    lib,
    num_games: int = 25,
    temperature: float = 0.1,
    seed_offset: int = 0,
) -> dict[str, int | float]:
    """Play games: 2 human-bot seats vs 2 AB2 seats, batched inference.

    Returns ``{hz_wins, ab2_wins, draws, win_rate}``.
    """
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM

    nf_buf = np.zeros((num_games + 1, N, NF), dtype=np.float32)
    ef_buf = np.zeros((num_games + 1, E, EF), dtype=np.float32)
    ff_buf = np.zeros((num_games + 1, FF), dtype=np.float32)
    mask_buf = np.zeros((num_games + 1, AD), dtype=np.float32)

    edge_index_dev = state_enc._edge_index.to(device)
    net.eval()

    games = [CatanGame(seed=80000 + seed_offset * 1000 + i) for i in range(num_games)]
    for g in games:
        g.reset()

    hz_seats = [{i % 4, (i + 2) % 4} for i in range(num_games)]
    ab2_seats = [{(i + 1) % 4, (i + 3) % 4} for i in range(num_games)]

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    active = list(range(num_games))
    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        progress = True
        while progress:
            progress = False
            for idx in active:
                g = games[idx]
                if g.is_terminal() or g.turn_number >= 1000:
                    continue
                cp = g.current_player()
                if cp not in ab2_seats[idx]:
                    continue
                le = g.get_legal_actions()
                if not le:
                    continue
                cg = g._game
                bc = cg.state.colors[cg.state.current_player_index]
                bi, bv = 0, -1e30
                for i, act in enumerate(le):
                    lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                    lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
                    if v > bv:
                        bv = v
                        bi = i
                g.step(bi)
                progress = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        B = 0
        imap: list[tuple[int, list]] = []
        for idx in active:
            g = games[idx]
            if g.current_player() not in hz_seats[idx]:
                continue
            le = g.get_legal_actions()
            if not le:
                continue
            state_enc.encode_into(g.get_state_view(), nf_buf[B], ef_buf[B], ff_buf[B])
            mask_buf[B] = action_enc.get_action_mask(le).numpy()
            imap.append((idx, le))
            B += 1

        if B == 0:
            continue

        with torch.no_grad():
            batch = {
                "node_features": torch.from_numpy(nf_buf[:B].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B].copy()).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B].copy()).to(device),
            }
            out = net(batch)
            lo = out["policy_logits"] / temperature
            lo = lo.masked_fill(batch["action_mask"] == 0, -1e9)
            pr = F.softmax(lo, dim=-1).cpu().numpy()

        for b, (idx, le) in enumerate(imap):
            p = pr[b]
            if p.sum() < 1e-6:
                p = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
            p = p / p.sum()
            aidx = int(np.random.choice(AD, p=p))
            chosen = next((i for i, a in enumerate(le) if action_enc.encode(a) == aidx), 0)
            games[idx].step(chosen)

    hz_wins = ab2_wins = 0
    for idx in range(num_games):
        w = games[idx].winner()
        if w is not None:
            if w in hz_seats[idx]:
                hz_wins += 1
            elif w in ab2_seats[idx]:
                ab2_wins += 1

    total = hz_wins + ab2_wins
    return {
        "hz_wins": hz_wins,
        "ab2_wins": ab2_wins,
        "draws": num_games - total,
        "win_rate": hz_wins / max(total, 1),
    }
