"""NERD loss with cross-entropy value head.

Policy: NERD gradient over full action space with 1-ply Q-values.
Value: Cross-entropy over 4-player win distribution.
Entropy bonus to prevent policy collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RNaDLoss(nn.Module):
    """NERD policy gradient + cross-entropy value + entropy bonus."""

    def __init__(
        self,
        eta: float = 0.5,
        clip_bound: float = 100.0,
        value_weight: float = 1.0,
        entropy_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.eta = eta
        self.clip_bound = clip_bound
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight

    def forward(
        self,
        policy_logits: Tensor,
        anchor_log_probs: Tensor,
        action_mask: Tensor,
        q_all: Tensor,
        value_targets: Tensor,
        value_logits: Tensor,
    ) -> dict[str, Tensor]:
        """Compute loss.

        Args:
            policy_logits:  (B, A) raw logits from current network.
            anchor_log_probs: (B, A) log pi_bar from frozen anchor.
            action_mask:    (B, A) 1.0 for legal, 0.0 for illegal.
            q_all:          (B, A) Q(s,a) from 1-ply lookahead.
            value_targets:  (B, 4) win distribution target (sums to ~1).
            value_logits:   (B, 4) raw logits from value head.
        """
        B, A = policy_logits.shape

        # ── Policy ────────────────────────────────────────────────────
        masked_logits = policy_logits.masked_fill(action_mask == 0, -1e9)
        clamped = masked_logits.clamp(-3.0, 3.0).masked_fill(action_mask == 0, -1e9)
        log_pi = F.log_softmax(clamped, dim=-1)
        pi = log_pi.exp()

        # NERD with R-NaD anchor
        n_legal = action_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        uniform_lp = torch.log(action_mask / n_legal + 1e-10)
        smoothed_anchor = 0.9 * anchor_log_probs.detach() + 0.1 * uniform_lp
        q_transformed = q_all.detach() + self.eta * (smoothed_anchor - log_pi)
        q_transformed = q_transformed * action_mask

        v_transformed = (pi * q_transformed).sum(dim=-1, keepdim=True)
        advantages = (q_transformed - v_transformed) * action_mask

        clipped_adv = advantages.detach().clamp(-self.clip_bound, self.clip_bound)
        nerd_loss = -(pi * log_pi * clipped_adv).sum(dim=-1).mean()
        nerd_loss = torch.nan_to_num(nerd_loss, nan=0.0)

        # ── Value: cross-entropy over 4-player win distribution ───────
        # Normalize targets to a valid distribution
        vt = value_targets.detach().clamp(min=0.0)
        vt_sum = vt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vt_dist = vt / vt_sum

        value_log_probs = F.log_softmax(value_logits, dim=-1)
        value_loss = -(vt_dist * value_log_probs).sum(dim=-1).mean()
        value_loss = torch.nan_to_num(value_loss, nan=0.0)

        # ── Entropy bonus ─────────────────────────────────────────────
        entropy = -(pi * log_pi * action_mask).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0).mean()

        total_loss = nerd_loss + self.value_weight * value_loss - self.entropy_weight * entropy

        # ── Metrics ───────────────────────────────────────────────────
        with torch.no_grad():
            mean_adv = (advantages * action_mask).abs().mean()
            value_pred_winner = F.softmax(value_logits, dim=-1).argmax(dim=-1)
            value_true_winner = vt_dist.argmax(dim=-1)
            value_accuracy = (value_pred_winner == value_true_winner).float().mean()

        return {
            "total_loss": total_loss,
            "nerd_loss": nerd_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "mean_advantage": mean_adv,
            "value_accuracy": value_accuracy,
        }
