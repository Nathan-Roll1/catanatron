"""Combined AlphaZero loss function for HexaZero.

L = alpha * value_loss + beta * policy_loss

Value loss:  Cross-entropy between predicted win logits and outcome distribution.
Policy loss: Cross-entropy between MCTS visit distribution pi and network
             log-policy P, masked to legal actions only.
L2 regularisation is handled by the optimizer's weight_decay parameter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HexaZeroLoss(nn.Module):
    """Combined value + policy loss with diagnostic metrics."""

    def __init__(
        self,
        value_weight: float = 1.0,
        policy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.value_weight = value_weight
        self.policy_weight = policy_weight

    def forward(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute combined loss and auxiliary metrics.

        Args:
            predictions: network outputs containing
                ``policy_logits`` (batch, 337) and ``value`` (batch, 4).
            targets: training targets containing
                ``policy_targets`` (batch, 337) — MCTS visit distribution,
                ``value_targets`` (batch, 4) — one-hot game outcome, and
                ``action_masks`` (batch, 337) — legal-action mask.

        Returns:
            Dict with scalar tensors: ``total_loss``, ``value_loss``,
            ``policy_loss``, ``value_accuracy``, ``policy_entropy``.
        """
        policy_logits: Tensor = predictions["policy_logits"]
        value_pred: Tensor = predictions["value"]

        policy_targets: Tensor = targets["policy_targets"]
        value_targets: Tensor = targets["value_targets"]
        action_masks: Tensor = targets["action_masks"]

        value_loss = self._value_loss(value_pred, value_targets)
        policy_loss = self._policy_loss(policy_logits, policy_targets, action_masks)

        total_loss = (
            self.value_weight * value_loss + self.policy_weight * policy_loss
        )

        with torch.no_grad():
            value_accuracy = self._value_accuracy(value_pred, value_targets)
            policy_entropy = self._policy_entropy(policy_logits, action_masks)

        return {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "value_accuracy": value_accuracy,
            "policy_entropy": policy_entropy,
        }

    @staticmethod
    def _value_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy between predicted win logits and target distribution."""
        vt = target.detach().clamp(min=0.0)
        vt_sum = vt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vt_dist = vt / vt_sum
        log_probs = F.log_softmax(pred, dim=-1)
        return -(vt_dist * log_probs).sum(dim=-1).mean()

    @staticmethod
    def _policy_loss(
        logits: Tensor,
        target_probs: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Masked cross-entropy: -pi^T log(P).

        Illegal actions are pushed to a large negative value before
        log-softmax.  We use nan_to_num to handle 0 * (-inf) = NaN
        on MPS/CPU backends.
        """
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        per_sample = -(target_probs * log_probs).sum(dim=-1)
        per_sample = torch.nan_to_num(per_sample, nan=0.0)
        return per_sample.mean()

    @staticmethod
    def _value_accuracy(pred: Tensor, target: Tensor) -> Tensor:
        """Fraction of samples where predicted winner matches actual winner."""
        return (pred.argmax(dim=-1) == target.argmax(dim=-1)).float().mean()

    @staticmethod
    def _policy_entropy(logits: Tensor, mask: Tensor) -> Tensor:
        """Entropy of the masked predicted policy (in nats)."""
        masked_logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0)
        return entropy.mean()
