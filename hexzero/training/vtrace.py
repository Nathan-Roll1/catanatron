"""V-trace off-policy correction (Espeholt et al., 2018).

Computes corrected value targets for policy gradient methods when the
behavior policy (used during data collection) differs slightly from the
current policy (used during training).  For near-on-policy R-NaD this
reduces to standard TD(lambda).
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_vtrace(
    rewards: Tensor,
    values: Tensor,
    log_pi: Tensor,
    log_mu: Tensor,
    gamma: float = 1.0,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Compute V-trace targets and one-step Q-values.

    Args:
        rewards:  (T,) transformed rewards r_tilde at each step.
        values:   (T+1,) predicted values V(s_t).  values[T] is the
                  bootstrap value at the terminal/truncated state
                  (0 for true terminal).
        log_pi:   (T,) log pi_theta(a_t | s_t) under current policy.
        log_mu:   (T,) log mu(a_t | s_t) under behavior policy.
        gamma:    Discount factor (1.0 for undiscounted episodic games).
        rho_bar:  Clipping for importance weight rho.
        c_bar:    Clipping for trace-cutting coefficient c.

    Returns:
        vtrace_targets: (T,) corrected value targets for each step.
        vtrace_q:       (T,) one-step Q-values: r_t + gamma * V(s_{t+1}).
    """
    T = rewards.shape[0]
    device = rewards.device

    log_rho = (log_pi - log_mu).clamp(max=20.0)
    rho = log_rho.exp().clamp(max=rho_bar)
    c = log_rho.exp().clamp(max=c_bar)

    deltas = rho * (rewards + gamma * values[1:] - values[:-1])

    vtrace_targets = torch.zeros(T, device=device)
    acc = torch.tensor(0.0, device=device)
    for t in range(T - 1, -1, -1):
        acc = deltas[t] + gamma * c[t] * acc
        vtrace_targets[t] = values[t] + acc

    vtrace_q = rewards + gamma * values[1:]

    vtrace_targets = vtrace_targets.clamp(-1.0, 2.0)
    vtrace_q = vtrace_q.clamp(-1.0, 2.0)

    return vtrace_targets, vtrace_q


def transform_rewards(
    rewards: Tensor,
    anchor_log_probs: Tensor,
    current_log_probs: Tensor,
    eta: float,
) -> Tensor:
    """R-NaD reward transformation.

    r_tilde_t = r_t + eta * [log pi_bar(a_t|s_t) - log pi_theta(a_t|s_t)]

    This converts the KL-regularized game into a standard RL problem.
    """
    raw = rewards + eta * (anchor_log_probs - current_log_probs)
    return raw.clamp(-2.0, 2.0)
