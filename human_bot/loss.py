"""Loss functions for human-bot training.

Two heads, one shared trunk:
  - Policy: label-smoothed cross-entropy over the human's chosen action.
  - Value:  cross-entropy over the 4-player win distribution.

Optionally combined via learned uncertainty weighting (Kendall et al., 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Per-example loss weights by action index range.
# Downweight trivial actions (ROLL), upweight strategic ones (build/trade/robber).
_ACTION_WEIGHT_RANGES: list[tuple[int, int, float]] = [
    (0, 1, 0.2),     # ROLL
    (1, 2, 1.0),     # END_TURN
    (2, 3, 3.0),     # BUY_DEV
    (3, 4, 3.0),     # PLAY_KNIGHT
    (4, 5, 3.0),     # ROAD_BUILDING
    (5, 59, 3.0),    # settlement
    (59, 113, 3.0),  # city
    (113, 185, 3.0), # road
    (185, 280, 3.0), # robber
    (280, 285, 1.5), # discard
    (285, 305, 3.0), # yop
    (305, 310, 3.0), # monopoly
    (310, 330, 2.0), # maritime
    (330, 397, 2.0), # trade responses + offers
]


def _build_action_weights(size: int = 397, device: str = "cpu") -> torch.Tensor:
    w = torch.ones(size)
    for lo, hi, weight in _ACTION_WEIGHT_RANGES:
        w[lo:min(hi, size)] = weight
    return w.to(device)


def human_policy_loss(
    logits: torch.Tensor,
    action_idx: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.05,
    action_weights: torch.Tensor | None = None,
    winner_boost: torch.Tensor | None = None,
) -> torch.Tensor:
    """Label-smoothed cross-entropy over legal actions.

    Smoothing mass is distributed *only* among legal actions so the model
    is never rewarded for predicting illegal moves.

    If ``winner_boost`` is provided (1-d tensor, same length as batch),
    examples where the acting player won the game get upweighted.
    Typical usage: pass ``1.0 + 0.5 * (vt[:, 0] > 0.5).float()`` to
    give winner actions 1.5x weight.
    """
    fill_val = -6e4 if logits.dtype == torch.float16 else -1e9
    masked_logits = logits.masked_fill(~mask.bool(), fill_val)

    n_legal = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    one_hot = torch.zeros_like(logits).scatter_(1, action_idx.unsqueeze(1), 1.0)
    smooth = (1.0 - label_smoothing) * one_hot + label_smoothing * (mask / n_legal)

    log_probs = F.log_softmax(masked_logits, dim=-1)
    per_example = -(smooth * log_probs).sum(dim=-1)

    if winner_boost is not None:
        per_example = per_example * winner_boost

    if action_weights is not None:
        w = action_weights[action_idx]
        return (per_example * w).mean()
    return per_example.mean()


def value_loss(
    pred_logits: torch.Tensor,
    target_dist: torch.Tensor,
    turn_progress: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy loss for win prediction with turn-based weighting.

    Targets are binary winner labels (one-hot [1,0,0,0] for current
    player wins, rotated by seat).  Predictions are raw logits passed
    through log_softmax.

    Late-game positions (where outcome is more determined) are weighted
    higher via ``turn_progress`` (num_turns / 1000, from flat_features[:, 114]).
    Weight = 0.2 + 0.8 * (turn / max_turn), so early positions still
    contribute but late positions dominate.
    """
    target = target_dist.detach().clamp(min=0.0)
    tsum = target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    target = target / tsum
    log_pred = F.log_softmax(pred_logits, dim=-1)
    per_example = -(target * log_pred).sum(dim=-1)

    if turn_progress is not None:
        tp = turn_progress.detach().clamp(min=0.0)
        max_tp = tp.max().clamp(min=1e-4)
        weight = 0.2 + 0.8 * (tp / max_tp)
        return (per_example * weight).mean()
    return per_example.mean()


def masked_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Entropy of the masked policy distribution (higher = more exploration)."""
    fill_val = -6e4 if logits.dtype == torch.float16 else -1e9
    masked_logits = logits.masked_fill(~mask.bool(), fill_val)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    probs = log_probs.exp()
    ent = -(probs * log_probs * mask).sum(dim=-1)
    return torch.nan_to_num(ent, nan=0.0).mean()


class UncertaintyWeightedLoss(nn.Module):
    """Homoscedastic uncertainty weighting for multi-task learning.

    Learns log-variance parameters that automatically balance the policy
    and value losses during training (Kendall et al., CVPR 2018).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log_var_policy = nn.Parameter(torch.zeros(1))
        self.log_var_value = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy: torch.Tensor,
        entropy_weight: float = 0.01,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        w_p = torch.exp(-self.log_var_policy)
        w_v = torch.exp(-self.log_var_value)
        total = (
            w_p * policy_loss + 0.5 * self.log_var_policy
            + w_v * value_loss + 0.5 * self.log_var_value
            - entropy_weight * entropy
        )
        diagnostics = {
            "w_policy": w_p.item(),
            "w_value": w_v.item(),
        }
        return total.squeeze(), diagnostics


class FixedWeightLoss(nn.Module):
    """Simple fixed-weight combination of policy + value losses."""

    def __init__(self, policy_weight: float = 1.0, value_weight: float = 0.5) -> None:
        super().__init__()
        self.pw = policy_weight
        self.vw = value_weight

    def forward(
        self,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
        entropy: torch.Tensor,
        entropy_weight: float = 0.01,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = self.pw * policy_loss + self.vw * value_loss - entropy_weight * entropy
        diagnostics = {"w_policy": self.pw, "w_value": self.vw}
        return total, diagnostics
