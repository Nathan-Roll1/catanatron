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


def human_policy_loss(
    logits: torch.Tensor,
    action_idx: torch.Tensor,
    mask: torch.Tensor,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Label-smoothed cross-entropy over legal actions.

    Smoothing mass is distributed *only* among legal actions so the model
    is never rewarded for predicting illegal moves.
    """
    fill_val = -6e4 if logits.dtype == torch.float16 else -1e9
    masked_logits = logits.masked_fill(~mask.bool(), fill_val)

    n_legal = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    one_hot = torch.zeros_like(logits).scatter_(1, action_idx.unsqueeze(1), 1.0)
    smooth = (1.0 - label_smoothing) * one_hot + label_smoothing * (mask / n_legal)

    log_probs = F.log_softmax(masked_logits, dim=-1)
    return -(smooth * log_probs).sum(dim=-1).mean()


def value_loss(
    pred_logits: torch.Tensor,
    target_dist: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between predicted win logits and target distribution."""
    vt = target_dist.detach().clamp(min=0.0)
    vt_sum = vt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    vt_normed = vt / vt_sum
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return -(vt_normed * log_probs).sum(dim=-1).mean()


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
