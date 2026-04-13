# R-NaD Adaptation Plan for HexaZero

**Adapting DeepNash's Regularized Nash Dynamics to 4-Player Settlers of Catan**

---

## Table of Contents

1. [Technical Deep Dive on R-NaD](#part-1-technical-deep-dive-on-r-nad)
2. [Adaptation to 4-Player Catan](#part-2-adaptation-to-4-player-catan)
3. [Implementation Plan for HexaZero](#part-3-implementation-plan-for-hexazero)
4. [Concrete Speedup Analysis](#part-4-concrete-speedup-analysis)
5. [Hybrid R-NaD + MARL Trading Module](#part-5-hybrid-r-nad--marl-trading-module)

---

## Part 1: Technical Deep Dive on R-NaD

### 1.1 Background: Why Standard RL Fails in Games

Standard policy gradient methods (REINFORCE, PPO, A2C) optimize a single agent's expected
return against a fixed or slowly-adapting opponent. In competitive multi-agent settings this
creates a non-stationary optimization landscape: as player 1 improves, the optimal strategy for
player 2 shifts, which shifts the optimal strategy for player 1, and so on. The result is policy
cycling rather than convergence. This is well-known in game theory as the failure of
gradient descent in bilinear games.

Fictitious play and its neural variants (NFSP, Deep CFR) address this by maintaining an average
policy over training history. R-NaD takes a different approach rooted in dynamical systems
theory.

### 1.2 Replicator Dynamics and Their Failure

The **replicator dynamics** describe how strategy frequencies evolve in an evolutionary game:

```
dπ_i(a|s)/dt = π_i(a|s) · [q_i(s,a) - v_i(s)]
```

where `q_i(s,a)` is the action-value for player `i` and `v_i(s) = Σ_a π_i(a|s) · q_i(s,a)`.
In matrix form, strategies that perform above average grow; those below average shrink. This
is equivalent to the multiplicative weights update and to the natural policy gradient in the
tabular case.

The problem: in two-player zero-sum games, replicator dynamics **cycle** around the Nash
equilibrium. The time-average converges, but the instantaneous policy does not. In function
approximation settings (neural networks), time-averaging is impractical because you'd need to
maintain and average over a growing history of network parameters.

### 1.3 Regularized Nash Dynamics (R-NaD)

R-NaD (Perolat et al., 2022) fixes the cycling problem by adding a **regularization anchor**.
The algorithm has two nested loops:

**Outer loop:** Maintain an anchor policy `π̄` (a frozen copy of the current network).
Periodically reset: `π̄ ← π_θ`.

**Inner loop:** Solve a **regularized game** where each player's payoff is penalized by a
KL divergence from the anchor:

```
ũ_i(π) = u_i(π) - η · D_KL(π_i ∥ π̄_i)
```

where:
- `u_i(π)` is player `i`'s expected payoff under joint policy `π`
- `η > 0` is the regularization strength
- `D_KL(π_i ∥ π̄_i) = Σ_{s,a} d_π(s) · π_i(a|s) · log[π_i(a|s) / π̄_i(a|s)]`

The KL penalty makes each player's payoff **strongly concave** in their own policy (for fixed
opponent policies). This guarantees the regularized game has a **unique** Nash equilibrium,
and the regularized replicator dynamics converge to it (rather than cycling).

### 1.4 The Reward Transformation Trick

Rather than modifying the loss function directly, R-NaD applies an equivalent **reward
transformation** that converts the regularized game into a standard RL problem:

```
r̃_t = r_t + η · [log π̄(a_t | s_t) - log π_θ(a_t | s_t)]
```

Expanding:
- `r_t`: environment reward at timestep `t`
- `η · log π̄(a_t | s_t)`: bonus for actions the anchor would have taken (pulls toward anchor)
- `-η · log π_θ(a_t | s_t)`: entropy bonus (encourages exploration)

This transformation is mathematically equivalent to the KL-penalized payoff:

```
E_π[Σ_t r̃_t] = E_π[Σ_t r_t] + η · H(π) - η · D_KL(π ∥ π̄)
             = u_i(π) - η · D_KL(π_i ∥ π̄_i) + const
             = ũ_i(π) + const
```

The advantage of the reward transformation: **any standard RL algorithm** (V-trace, PPO,
IMPALA) can be used on the inner loop. You just feed it the transformed rewards instead of
the raw rewards.

### 1.5 NERD: Neural Replicator Dynamics

DeepNash uses a specific policy update called **NERD** (Neural Replicator Dynamics) for the
inner loop. NERD is the continuous-time replicator dynamics discretized for neural networks
with a crucial **advantage clipping** step.

The NERD policy gradient for player `i`:

```
∇_θ L_NERD = -E_{s ~ d_π} [ Σ_a π_θ(a|s) · ∇_θ log π_θ(a|s) · clip(Ã_i(s,a), -C, C) ]
```

where:
- `Ã_i(s,a) = q̃_i(s,a) - ṽ_i(s)` is the **transformed advantage**
- `q̃_i(s,a) = q_i(s,a) + η · log π̄_i(a|s) - η · log π_θ_i(a|s)` (transformed Q-value)
- `ṽ_i(s) = Σ_a π_θ_i(a|s) · q̃_i(s,a)` (transformed value)
- `clip(x, -C, C) = max(-C, min(C, x))` with `C` a hyperparameter (DeepNash uses `C = 10000`)

Why clipping matters: without it, the multiplicative update `π(a) ∝ π(a) · exp(η · Â(a))`
can produce infinite gradients when advantages are large, destabilizing training.

Note the structural difference from standard policy gradient:
- **Standard PG:** `∇L = -E_{a~π}[∇log π(a) · Â(a)]` — samples actions from π
- **NERD:** `∇L = -E_s[Σ_a π(a) · ∇log π(a) · clip(Ã(a))]` — sums over **all** actions

NERD sums over the full action space (weighted by π), which is feasible when the action space
is discrete and manageable (337 actions in our case). This full-support gradient is what makes
it correspond to the replicator dynamics rather than a stochastic approximation.

### 1.6 V-trace for Off-Policy Correction

DeepNash generates trajectories from a behavior policy `μ` (the current network during data
collection) and trains on them with the latest parameters `θ`. Since `μ ≠ π_θ` after gradient
updates, off-policy correction is needed.

R-NaD uses **V-trace** (Espeholt et al., 2018):

```
v_s = V(s) + Σ_{t=s}^{s+n-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) · δ_t
```

where:
- `δ_t = ρ_t · (r̃_t + γ · V(s_{t+1}) - V(s_t))` is the corrected TD error
- `ρ_t = min(ρ̄, π(a_t|s_t)/μ(a_t|s_t))` is the clipped importance ratio
- `c_t = min(c̄, π(a_t|s_t)/μ(a_t|s_t))` is the trace-cutting coefficient
- Typical: `ρ̄ = 1, c̄ = 1` (on-policy V-trace ≈ TD(λ))

The V-trace target gives the transformed Q-value for NERD:
```
q̃(s,a) = r̃(s,a) + γ · v_{s'}   (one-step transformed Q-value)
```

### 1.7 Complete R-NaD Algorithm

```
Algorithm: R-NaD (from Perolat et al., 2022)
─────────────────────────────────────────────
Initialize: θ (network params), π̄ ← π_θ (anchor)
Hyperparams: η (regularization), C (clip), K (outer steps), T (inner steps)

for k = 1 to K:                                    # Outer loop
    π̄ ← stop_gradient(π_θ)                         # Reset anchor

    for t = 1 to T:                                 # Inner loop
        Collect trajectories τ using π_θ (self-play)
        For each (s, a, r, s') in τ:
            r̃ = r + η · [log π̄(a|s) - log π_θ(a|s)]   # Transform rewards
        Compute V-trace targets v_s from r̃
        Compute transformed advantages:
            q̃(s,a) = r̃ + γ · v_{s'}
            Ã(s,a) = q̃(s,a) - Σ_a' π_θ(a'|s) · q̃(s,a')
        NERD gradient:
            ∇L = -E_s[Σ_a π_θ(a|s) · ∇log π_θ(a|s) · clip(Ã(s,a), -C, C)]
        Update: θ ← θ - α · ∇L

return π_θ
```

### 1.8 Why R-NaD Converges in 2-Player Zero-Sum

The convergence argument has three parts:

1. **Inner loop convergence**: The regularized game `ũ` is strongly concave in each player's
   strategy (the KL term ensures this). Therefore the regularized replicator dynamics (NERD)
   converge to the unique Nash equilibrium `π^*_η` of the regularized game, rather than
   cycling.

2. **Anchor sequence convergence**: Each time the anchor is reset (`π̄ ← π_θ`), the
   regularized game changes. But the sequence of regularized Nash equilibria
   `{π^*_{η,k}}_{k=1}^K` converges to the Nash equilibrium of the **original** (unregularized)
   game. This follows from the proximal point interpretation: each outer step is a proximal
   operator applied to the original payoff, and iterated proximal point converges in monotone
   games (which includes two-player zero-sum games).

3. **Last-iterate convergence**: Unlike fictitious play which only guarantees time-average
   convergence, R-NaD provides **last-iterate** convergence. The current policy `π_θ` itself
   converges to Nash, not just an average over training history. This is critical for neural
   network implementations where maintaining a running average of policies across network
   parameters is impractical.

**Key assumption**: Two-player zero-sum. The monotone game structure is what makes the
proximal point iteration converge. This is the main challenge for adapting to 4-player Catan.

---

## Part 2: Adaptation to 4-Player Catan

### 2.1 The Fundamental Challenge

Nash equilibrium in n-player general-sum games has three properties that make it harder than
the 2-player zero-sum case:

1. **PPAD-completeness**: Finding a Nash equilibrium in n-player games (n ≥ 3) is
   PPAD-complete (Chen & Deng, 2006; Daskalakis et al., 2009). No polynomial-time algorithm
   is known.

2. **Non-uniqueness**: Multiple Nash equilibria can exist with different payoff profiles. There
   is no canonical "the" Nash equilibrium to converge to.

3. **Non-interchangeability**: In 2-player zero-sum, all Nash equilibria yield the same payoff
   (minimax theorem). In general-sum games, players may prefer different equilibria, creating
   coordination problems.

4. **Payoff structure**: Catan is not zero-sum. Trading creates mutual value. Building a road
   doesn't directly reduce opponents' resources (unlike capturing pieces in Stratego). The
   game has cooperative sub-problems embedded in a competitive framework.

R-NaD's convergence proof relies on the monotone operator structure of 2-player zero-sum
games. In 4-player Catan, this structure is absent. We cannot simply run R-NaD unmodified and
expect convergence to any equilibrium concept.

### 2.2 Solution Concepts for 4-Player Games

| Concept | Tractability | Quality | Applicability |
|---------|-------------|---------|--------------|
| Nash Equilibrium | PPAD-complete | Optimal (no unilateral deviation) | Theoretically ideal, computationally infeasible |
| Coarse Correlated Eq. (CCE) | Polynomial (regret minimization) | Weaker than Nash, but still a stable profile | Good fit for independent learners |
| Correlated Equilibrium (CE) | Polynomial (linear program) | Between CCE and Nash | Requires a mediator/correlating device |
| α-Rank | Polynomial (Markov chain) | Evolutionary stability ranking | Good for evaluating, not for training |
| Mean-Field NE | Depends on structure | Approximate (treats opponents as aggregate) | Useful simplification for symmetric games |

**Recommendation: Target CCE via independent R-NaD learners.**

### 2.3 Coarse Correlated Equilibrium via Independent R-NaD

A **Coarse Correlated Equilibrium** (CCE) is a distribution over joint action profiles such
that no player can improve their expected payoff by committing to a deviation **before**
seeing the recommended action. Formally:

```
For all players i, for all alternative strategies π'_i:
E_{a ~ σ}[u_i(a)] ≥ E_{a ~ σ}[u_i(π'_i, a_{-i})]
```

CCE has a crucial computational advantage: if each player independently minimizes their
**external regret**, the empirical distribution of play converges to the set of CCEs
(Hart & Mas-Colell, 2000). This holds for **any** number of players and **any** payoff
structure.

**The adaptation**: Run R-NaD independently for each player seat. Each player treats the
other three as part of the environment. The R-NaD anchor/regularization still provides
stability for each individual learner. The joint policy converges to a CCE of the 4-player
game.

```
Independent R-NaD for 4-Player Catan:
──────────────────────────────────────
Single shared network π_θ (all 4 seats use the same weights).
Anchor π̄ is also shared.

for k = 1 to K:
    π̄ ← stop_gradient(π_θ)                  # Single shared anchor

    for t = 1 to T:
        Play full 4-player games using π_θ for all seats
        For each player p in {0,1,2,3}:
            Collect that player's trajectory τ_p
            Transform rewards: r̃_p = r_p + η·[log π̄(a|s) - log π_θ(a|s)]
            Compute V-trace targets for player p
            Accumulate NERD gradient for player p's transitions
        Average gradients across all 4 players' transitions
        Update θ
```

Why a single shared network works: Catan has symmetric roles (all players have the same
action space, the same win condition). The state encoder already rotates the representation
so the current player is always at index 0. A single network trained on all seats naturally
learns a symmetric strategy. This is equivalent to DeepNash's single network for both
Stratego players.

### 2.4 Why CCE Is Sufficient for Catan

Three practical arguments for targeting CCE rather than Nash:

1. **Human play is CCE, not Nash.** Humans don't compute Nash equilibria; they adapt to
   opponents' tendencies. A strong CCE strategy beats human-level play. No Catan AI needs to
   be Nash-optimal to be superhuman.

2. **CCE handles cooperation implicitly.** In a CCE, correlated strategies can emerge through
   shared training. If trading benefits both parties on average, the CCE can include
   "trade-cooperative" joint strategies. This doesn't require explicit cooperative reasoning.

3. **CCE is robust to opponent modeling.** A CCE strategy guarantees low regret against the
   training distribution. Against unknown opponents, it provides a reasonable baseline.

### 2.5 The Trading Problem

Trading in Catan is fundamentally cooperative: both parties must agree, and both should
benefit. Pure Nash strategies in zero-sum games never cooperate. But Catan isn't zero-sum
and CCE allows correlated cooperative behavior.

**Current state in HexaZero:**
- The C engine has `AT_OFFER_TRADE`, `AT_ACCEPT_TRADE` action types in the enum
- `apply_action.c` does NOT implement domestic trade resolution (only maritime trade works)
- `ActionEncoder.encode()` raises `ValueError` for `AT_OFFER_TRADE`
- `PolicyHeadB` exists but has no training signal

**Proposed trade mechanism (see Part 5 for full design):**

The core insight: treat trade proposals as a **two-phase action** within the existing turn
structure:

Phase 1 (Proposer's turn): The network outputs a trade proposal from a structured sub-head.
Phase 2 (Responders' turns): Each other player's network outputs accept/reject.

The training signal is simple: **did the trade contribute to the proposer winning?** and
**did accepting contribute to the responder winning?** This naturally filters for
mutually-beneficial trades: a player only proposes trades that help them win, and only accepts
trades that help them win. Trades that help both parties get reinforced for both.

### 2.6 Kingmaking Defense

**Kingmaking**: when a player who cannot win chooses actions that determine which of the
remaining contenders wins. This is a pervasive problem in multiplayer games and has no
clean game-theoretic solution.

**Practical defenses for HexaZero:**

1. **Value-weighted exploration.** A player with 2 VP has a 15-20% win probability. Their
   policy should still maximize their own P(win), not spite any particular opponent. The
   R-NaD value head already provides this signal.

2. **Reward shaping for finishing position.** Instead of binary win/loss, use a graded reward:
   ```
   r_i = 1.0 if player i wins
   r_i = 0.3 if player i finishes 2nd
   r_i = 0.1 if player i finishes 3rd
   r_i = 0.0 if player i finishes 4th
   ```
   This incentivizes competitive play even when winning is unlikely. A player at 2 VP still
   tries to reach 5-6 VP for a better finishing position rather than kingmaking.

3. **Self-play averaging.** Since all seats use the same network, kingmaking strategies that
   benefit one seat at the expense of another are not reinforced on average. The network
   learns to play well from every seat, including losing positions.

4. **Population diversity (optional).** Train a population of 4-8 agents with different seeds.
   During self-play, sample 4 agents from the population for each game. This prevents the
   network from learning brittle collusive strategies and ensures robustness to different
   opponent styles.

### 2.7 Single Network vs. Separate Networks

| Approach | Pros | Cons |
|----------|------|------|
| **Single shared network** | 4x more training data per parameter; symmetric by construction; simpler infrastructure | Cannot model role-specific strategies (but Catan roles are symmetric) |
| **Separate networks** | Can specialize by seat position; more capacity | 4x less data per network; prone to overfitting to each other; harder to train |
| **Population of shared networks** | Diversity; robust to opponent distribution shifts | Higher infra complexity; need meta-selection |

**Recommendation: Single shared network** (as currently implemented). Catan's seat symmetry
makes specialization unnecessary. The state encoder's rotation already handles seat-relative
encoding. Use population training only if kingmaking becomes a problem in practice.

---

## Part 3: Implementation Plan for HexaZero

### 3.1 What We Keep

| Component | File(s) | Status |
|-----------|---------|--------|
| C game engine | `csrc/` | Keep entirely. ~34K moves/sec throughput is the backbone. |
| ctypes bindings | `hexzero/bindings/` | Keep. Need minor extension for trade action support. |
| GNN board encoder | `hexzero/model/gnn.py` | Keep. 6-layer MPNN with 128-dim messages, 256-dim output. |
| ResNet trunk | `hexzero/model/trunk.py` | Keep. 20 blocks, 256 channels. |
| State encoder | `hexzero/encoder/state_encoder.py` | Keep. Player-rotated encoding with 54 nodes × 18 features + 115 flat features. |
| Action encoder | `hexzero/encoder/action_encoder.py` | Keep + extend for trade proposals. |
| PolicyHeadA | `hexzero/model/heads.py` | Keep. 337-dim action logits. |
| PolicyHeadB | `hexzero/model/heads.py` | Keep + add training signal. |
| ValueHead | `hexzero/model/heads.py` | Keep. 4-player softmax output. |
| W&B logging | `hexzero/scripts/train_loop.py` | Keep. |
| ELO evaluation | `hexzero/elo/` | Keep. |
| Distributed launch | `hexzero/scripts/launch_distributed.sh` | Modify (different worker script). |

### 3.2 What We Replace

| Component | Current | R-NaD Replacement |
|-----------|---------|-------------------|
| **Self-play** | MCTS (50 sims/move, 10K forward passes/game) | Direct policy play (1 forward pass/move, 200 passes/game) |
| **Training targets** | MCTS visit distribution π, game outcome Z | Transformed rewards r̃, V-trace value targets |
| **Loss function** | `HexaZeroLoss`: MSE(V,Z) + CE(π,P) | `RNaDLoss`: NERD policy gradient + V-trace value loss |
| **Training loop** | Ingest .pt files → replay buffer → supervised | On-policy self-play → reward transform → NERD + V-trace |
| **Data pipeline** | Workers write game files; trainer reads | Workers send trajectories; trainer processes with transforms |

### 3.3 Network Architecture Changes

The network architecture is almost entirely preserved. Three changes:

**Change 1: Value head output — from softmax to raw logits.**

The current `ValueHead` applies softmax, outputting win probabilities. For R-NaD, the value
head should output raw value estimates (expected transformed return), not probabilities.

```python
# Current (heads.py, ValueHead.forward):
return F.softmax(self.fc_out(self.act(self.bn(self.fc1(x)))), dim=-1)

# R-NaD version:
return self.fc_out(self.act(self.bn(self.fc1(x))))
```

The output shape remains (B, 4) — one value per seat (rotated so index 0 = current player).
But interpretation changes: V(s)[0] estimates the current player's expected transformed return
from state s, not a win probability.

**Change 2: Add anchor policy storage.**

The R-NaD anchor `π̄` is a frozen copy of the policy network. We need a mechanism to store
and periodically update it.

```python
class HexaZeroNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing init ...
        self.anchor_log_probs = None  # set by R-NaD outer loop

    def compute_anchor_log_probs(self, batch):
        """Compute log π̄(a|s) for all actions, using the frozen anchor."""
        with torch.no_grad():
            raw_logits = self._anchor_policy_head(self._anchor_trunk(
                self._anchor_board_encoder(batch)))
            mask = batch.get("action_mask")
            if mask is not None:
                raw_logits = raw_logits.masked_fill(~mask.bool(), float("-inf"))
            return F.log_softmax(raw_logits, dim=-1)
```

Implementation detail: rather than duplicating the full network, store the anchor as a
`state_dict` snapshot and load it into a second network instance (or use `torch.no_grad()`
with parameter swapping).

**Change 3: Trade proposal head (see Part 5).**

New structured output head for domestic trade proposals. This is additive — doesn't modify
existing heads.

### 3.4 New Loss Function: R-NaD Loss

Replace `HexaZeroLoss` with `RNaDLoss`:

```python
class RNaDLoss(nn.Module):
    """R-NaD loss: NERD policy gradient + value regression.

    L = L_nerd + β · L_value

    L_nerd: Neural Replicator Dynamics gradient (Section 1.5)
    L_value: MSE between V-trace targets and predicted values
    """

    def __init__(
        self,
        eta: float = 0.2,
        clip_bound: float = 10_000.0,
        value_weight: float = 0.5,
    ):
        super().__init__()
        self.eta = eta            # regularization strength
        self.clip_bound = clip_bound  # NERD advantage clip
        self.value_weight = value_weight

    def forward(
        self,
        policy_logits: torch.Tensor,     # (B, 337) raw logits from current net
        anchor_log_probs: torch.Tensor,  # (B, 337) log π̄(a|s) from frozen anchor
        action_mask: torch.Tensor,       # (B, 337) boolean legal-action mask
        vtrace_values: torch.Tensor,     # (B,) V-trace return targets
        pred_values: torch.Tensor,       # (B,) predicted values V(s)
        vtrace_q: torch.Tensor,          # (B, 337) Q-values from V-trace
    ) -> dict[str, torch.Tensor]:

        # --- Current policy log-probs ---
        masked_logits = policy_logits.masked_fill(~action_mask.bool(), -1e9)
        log_pi = F.log_softmax(masked_logits, dim=-1)
        pi = log_pi.exp()

        # --- Transformed Q-values and advantages ---
        # q̃(s,a) = q(s,a) + η·log π̄(a|s) - η·log π_θ(a|s)
        q_transformed = vtrace_q + self.eta * (anchor_log_probs - log_pi)
        v_transformed = (pi * q_transformed).sum(dim=-1, keepdim=True)
        advantages = q_transformed - v_transformed  # (B, 337)

        # --- NERD gradient (as a loss) ---
        # L_nerd = -Σ_a π(a|s) · log π(a|s) · clip(Ã(a), -C, C)
        clipped_adv = advantages.clamp(-self.clip_bound, self.clip_bound)
        nerd_loss = -(pi * log_pi * clipped_adv.detach()).sum(dim=-1).mean()

        # --- Value loss ---
        value_loss = F.mse_loss(pred_values, vtrace_values.detach())

        total_loss = nerd_loss + self.value_weight * value_loss

        return {
            "total_loss": total_loss,
            "nerd_loss": nerd_loss,
            "value_loss": value_loss,
            "mean_advantage": advantages.mean().detach(),
            "policy_entropy": -(pi * log_pi).sum(dim=-1).mean().detach(),
        }
```

### 3.5 V-trace Implementation

```python
def compute_vtrace(
    rewards: torch.Tensor,        # (T,) transformed rewards
    values: torch.Tensor,         # (T+1,) predicted values V(s_t)
    log_pi: torch.Tensor,         # (T,) log π_θ(a_t|s_t)
    log_mu: torch.Tensor,         # (T,) log μ(a_t|s_t), behavior policy
    gamma: float = 1.0,           # discount (1.0 for episodic undiscounted)
    rho_bar: float = 1.0,         # importance weight clip
    c_bar: float = 1.0,           # trace coefficient clip
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace targets and Q-values.

    Returns:
        vtrace_values: (T,) V-trace corrected value targets
        vtrace_q: (T,) one-step Q-value estimates
    """
    T = rewards.shape[0]
    log_rho = log_pi - log_mu
    rho = torch.exp(log_rho).clamp(max=rho_bar)
    c = torch.exp(log_rho).clamp(max=c_bar)

    # TD errors: δ_t = ρ_t · (r_t + γ·V(s_{t+1}) - V(s_t))
    deltas = rho * (rewards + gamma * values[1:] - values[:-1])

    # V-trace: v_s = V(s) + Σ_{t=s}^{T-1} γ^{t-s} (Π c_i) δ_t
    vtrace_values = torch.zeros(T, device=rewards.device)
    last = torch.tensor(0.0, device=rewards.device)
    for t in reversed(range(T)):
        last = deltas[t] + gamma * c[t] * last
        vtrace_values[t] = values[t] + last

    vtrace_q = rewards + gamma * values[1:]

    return vtrace_values, vtrace_q
```

### 3.6 Reward Transformation

```python
def transform_rewards(
    rewards: torch.Tensor,          # (T,) raw environment rewards
    anchor_log_probs: torch.Tensor, # (T,) log π̄(a_t|s_t) for actions taken
    current_log_probs: torch.Tensor,# (T,) log π_θ(a_t|s_t) for actions taken
    eta: float = 0.2,
) -> torch.Tensor:
    """Apply R-NaD reward transformation.

    r̃_t = r_t + η · [log π̄(a_t|s_t) - log π_θ(a_t|s_t)]
    """
    return rewards + eta * (anchor_log_probs - current_log_probs)
```

### 3.7 New Self-Play Worker (No MCTS)

The self-play worker becomes dramatically simpler. No MCTS tree, no simulations, no
determinizations. Just run the C engine and query the network for each move.

```python
def rnad_selfplay_worker(net, anchor_net, state_enc, action_enc, device,
                         n_concurrent, seed_base):
    """Play games at C engine speed with direct policy sampling.

    No MCTS. Each move: encode state → forward pass → sample action.
    Collect full trajectories for R-NaD training.
    """
    games = [CatanGame(seed=seed_base + i) for i in range(n_concurrent)]
    for g in games:
        g.reset()
    trajectories = [[] for _ in range(n_concurrent)]  # per-game trajectory storage
    active = list(range(n_concurrent))

    while active:
        # --- Batch encode all active game states ---
        batch_states = []
        batch_masks = []
        batch_indices = []

        for idx in active:
            g = games[idx]
            if g.is_terminal() or g.turn_number >= 1000:
                continue
            batch_indices.append(idx)
            st = state_enc.encode(g.get_state_view())
            le = g.get_legal_actions()
            mask = action_enc.get_action_mask(le)
            batch_states.append(st)
            batch_masks.append(mask)

        if not batch_indices:
            break

        # --- Batched forward pass (single GPU call for all active games) ---
        batch = collate_states(batch_states, batch_masks, device)
        with torch.no_grad():
            out = net(batch)
            anchor_out = anchor_net(batch)

        policy_probs = out["policy_probs"]        # (B, 337)
        values = out["value"][:, 0]               # (B,) current player's value
        log_pi = torch.log(policy_probs + 1e-8)   # (B, 337)
        anchor_log_pi = F.log_softmax(
            anchor_out["policy_logits"].masked_fill(
                ~batch["action_mask"].bool(), -1e9), dim=-1)

        # --- Sample actions and step environments ---
        still_active = []
        for b, idx in enumerate(batch_indices):
            g = games[idx]
            le = g.get_legal_actions()

            # Sample from policy (with temperature for exploration)
            probs = policy_probs[b].cpu()
            action_idx = torch.multinomial(probs, 1).item()

            # Record transition
            trajectories[idx].append({
                "state": batch_states[b],
                "action_idx": action_idx,
                "player": g.current_player(),
                "log_pi": log_pi[b, action_idx].item(),
                "anchor_log_pi": anchor_log_pi[b, action_idx].item(),
                "value": values[b].item(),
                "full_log_pi": log_pi[b].cpu(),        # needed for NERD
                "full_anchor_log_pi": anchor_log_pi[b].cpu(),
            })

            # Step the C engine
            chosen = next((i for i, a in enumerate(le)
                          if action_enc.encode(a) == action_idx), 0)
            g.step(chosen)

            if not g.is_terminal() and g.turn_number < 1000:
                still_active.append(idx)

        active = still_active

    # --- Compute terminal rewards ---
    for idx in range(n_concurrent):
        g = games[idx]
        winner = g.winner()
        for step in trajectories[idx]:
            p = step["player"]
            if winner is None:
                step["reward"] = 0.0
            elif winner == p:
                step["reward"] = 1.0
            else:
                step["reward"] = 0.0  # or graded: 0.3/0.1/0.0 by finish position

    return trajectories
```

### 3.8 New Training Loop

```python
def rnad_training_loop(config):
    """Main R-NaD training loop.

    Outer loop: reset anchor every ANCHOR_INTERVAL trajectories.
    Inner loop: self-play → reward transform → V-trace → NERD update.
    """
    net = HexaZeroNet(config.network).to(config.device)
    anchor_net = HexaZeroNet(config.network).to(config.device)
    anchor_net.load_state_dict(net.state_dict())
    anchor_net.eval()

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.training.learning_rate, weight_decay=1e-4)
    loss_fn = RNaDLoss(eta=0.2, clip_bound=10_000.0, value_weight=0.5)

    ANCHOR_INTERVAL = 200    # reset anchor every 200 training steps
    BATCH_CONCURRENT = 64    # games running simultaneously per worker
    ETA = 0.2                # regularization strength

    global_step = 0

    for outer_step in range(config.rnad.num_outer_steps):
        # --- Reset anchor ---
        anchor_net.load_state_dict(net.state_dict())
        anchor_net.eval()

        for inner_step in range(ANCHOR_INTERVAL):
            # --- Self-play: generate trajectories ---
            net.eval()
            trajectories = rnad_selfplay_worker(
                net, anchor_net, state_enc, action_enc,
                config.device, BATCH_CONCURRENT,
                seed_base=global_step * 1000)
            net.train()

            # --- Process each trajectory ---
            all_nerd_data = []
            for traj in trajectories:
                if len(traj) < 2:
                    continue

                # Separate by player (each player has their own value stream)
                for player_id in range(4):
                    player_steps = [s for s in traj if s["player"] == player_id]
                    if len(player_steps) < 2:
                        continue

                    rewards = torch.tensor([s["reward"] for s in player_steps])
                    values = torch.tensor([s["value"] for s in player_steps]
                                          + [0.0])  # terminal
                    log_pi = torch.tensor([s["log_pi"] for s in player_steps])
                    # On-policy: log_mu = log_pi (no off-policy correction needed
                    # since we just generated these trajectories)
                    log_mu = log_pi.clone()

                    anchor_lp = torch.tensor(
                        [s["anchor_log_pi"] for s in player_steps])
                    current_lp = log_pi

                    # Reward transformation
                    r_tilde = transform_rewards(rewards, anchor_lp, current_lp, ETA)

                    # V-trace
                    vt_values, vt_q = compute_vtrace(
                        r_tilde, values, log_pi, log_mu, gamma=1.0)

                    for t, step in enumerate(player_steps):
                        all_nerd_data.append({
                            "state": step["state"],
                            "full_log_pi": step["full_log_pi"],
                            "full_anchor_log_pi": step["full_anchor_log_pi"],
                            "vtrace_value": vt_values[t],
                            "vtrace_q_scalar": vt_q[t],
                            "pred_value": values[t],
                        })

            # --- NERD gradient update ---
            random.shuffle(all_nerd_data)
            for mini_batch in chunk(all_nerd_data, config.training.batch_size):
                batch = collate_nerd_batch(mini_batch, config.device)
                optimizer.zero_grad(set_to_none=True)

                out = net(batch["input"])
                losses = loss_fn(
                    policy_logits=out["policy_logits"],
                    anchor_log_probs=batch["anchor_log_probs"],
                    action_mask=batch["action_mask"],
                    vtrace_values=batch["vtrace_values"],
                    pred_values=out["value"][:, 0],
                    vtrace_q=batch["vtrace_q"],
                )

                losses["total_loss"].backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

            global_step += 1

            # Periodic: save checkpoint, evaluate, log metrics
            if global_step % 50 == 0:
                net.save_checkpoint(f"checkpoints/rnad_step_{global_step}.pt")
```

### 3.9 Summary of Code Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `hexzero/model/heads.py` | Modify | `ValueHead.forward`: remove softmax, return raw logits |
| `hexzero/model/network.py` | Modify | Add anchor computation methods; new `forward` return key for anchor |
| `hexzero/training/loss.py` | New file or rewrite | `RNaDLoss` replacing `HexaZeroLoss` |
| `hexzero/training/vtrace.py` | New file | V-trace computation |
| `hexzero/training/reward_transform.py` | New file | Reward transformation utility |
| `hexzero/scripts/rnad_selfplay.py` | New file | MCTS-free self-play worker |
| `hexzero/scripts/rnad_train_loop.py` | New file | R-NaD outer/inner training loop |
| `hexzero/config.py` | Modify | Add `RNaDConfig` dataclass |
| `hexzero/encoder/action_encoder.py` | Modify | Support `AT_OFFER_TRADE` encoding |
| `csrc/apply_action.c` | Modify | Implement domestic trade action handling |

Files **not** modified: `gnn.py`, `trunk.py`, `state_encoder.py`, `replay_buffer.py` (replaced
by trajectory buffer), all test files (need new tests).

---

## Part 4: Concrete Speedup Analysis

### 4.1 Current System Performance

**Self-play bottleneck analysis (current MCTS approach):**

```
Per move:
  MCTS simulations:           50
  Forward passes per sim:     1  (leaf evaluation)
  Total forward passes/move:  50
  C engine ops per sim:       ~3 (select, expand, backup)

Per game:
  Average game length:        ~200 moves
  Forward passes per game:    200 × 50 = 10,000
  C engine operations:        200 × 50 × 3 = 30,000

Per worker (8 concurrent games, batch_size=8):
  GPU inference throughput:   ~3,000 inf/s   (TITAN RTX, batch=8)
  Effective per-game inf/s:   3,000 / 8 = 375 inf/s per game
  Time per game:              10,000 / 375 ≈ 26.7 sec
  Games per second:           8 / 26.7 ≈ 0.3 g/s

Cluster (4 workers):
  Total throughput:           4 × 0.3 = 1.2 games/sec
  Positions per second:       1.2 × 200 = 240 pos/sec
  Daily throughput:           ~103,000 games/day
```

### 4.2 R-NaD System Performance (Projected)

**No MCTS means 50x fewer forward passes per game:**

```
Per move:
  Forward passes:             1  (direct policy query)
  C engine ops:               1  (apply action)

Per game:
  Average game length:        ~200 moves
  Forward passes per game:    200
  C engine operations:        200

Speedup factor per game:      10,000 / 200 = 50x fewer forward passes
```

**GPU throughput scaling with larger batches:**

Since we're no longer bottlenecked by MCTS tree expansion, we can run many more concurrent
games and batch their forward passes together:

```
Projected GPU inference throughput (TITAN RTX, 3.8M param model):
  Batch   1:    ~500 inf/s     (baseline from selfplay_loop.py comment)
  Batch   8:  ~3,000 inf/s     (6x, measured)
  Batch  32:  ~8,000 inf/s     (extrapolated, ~16x)
  Batch  64: ~12,000 inf/s     (extrapolated, ~24x)
  Batch 128: ~16,000 inf/s     (extrapolated, ~32x; nearing memory limit)
  Batch 256: ~18,000 inf/s     (diminishing returns, VRAM dependent)
```

**Per-worker throughput with 64 concurrent games:**

```
  GPU inference:              ~12,000 inf/s (batch=64)
  Per-game advance rate:      12,000 / 64 = 187 steps/sec
  Time per game:              200 / 187 ≈ 1.07 sec
  Games per second:           64 / 1.07 ≈ 60 g/s

  C engine check:
    Required:  64 × 187 = 12,000 actions/sec
    Available: 34,000 actions/sec per CPU core
    Utilization: 35% → NOT the bottleneck

  State encoding check:
    Encoding overhead: ~0.05ms per state
    Required: 12,000 × 0.05ms = 0.6 sec/sec → NOT the bottleneck
```

**Realistic estimate (accounting for overhead):**

```
  Python overhead, GIL, data transfer: ~40% efficiency loss
  Realistic per-worker throughput:     60 × 0.6 ≈ 36 g/s
  Conservative estimate:               ~25 g/s per worker
```

### 4.3 Cluster-Wide Projection

```
                          Current (MCTS)    R-NaD (projected)    Speedup
                          ──────────────    ─────────────────    ───────
Per worker (g/s):              0.3               25               83x
Cluster 4 workers (g/s):      1.2              100               83x
Positions per second:          240           20,000               83x
Games per hour:              4,320          360,000               83x
Games per day:             103,000        8,640,000               83x
```

### 4.4 Training Throughput Impact

With R-NaD, the training loop is tighter (on-policy, no replay buffer aging). Estimated
training step throughput:

```
  Positions per training step:    2048 (batch size)
  Position generation rate:       20,000 pos/sec (4 workers)
  Time to fill one batch:        2048 / 20,000 = 0.1 sec
  Forward + backward pass:       ~0.05 sec (TITAN RTX, batch=2048)
  Training steps per second:     ~6-7 steps/sec

  Compare current:
  Position generation rate:       240 pos/sec
  Time to fill one batch:        2048 / 240 = 8.5 sec
  Training is starved for data → workers are the bottleneck
```

**Key insight**: With R-NaD, the bottleneck shifts from self-play to training GPU throughput.
The trainer can barely keep up with 4 workers generating 20K positions/sec. Solutions:
- Increase batch size to 4096 or 8192
- Use gradient accumulation across multiple self-play batches
- Add a second trainer GPU for data-parallel training

### 4.5 Total Compute Comparison

To generate 1 million training positions:

```
Current (MCTS):     1,000,000 / 240 = 4,167 sec ≈ 69 minutes
R-NaD (projected):  1,000,000 / 20,000 = 50 sec ≈ 1 minute
```

To train for 1 million positions at batch_size=2048:

```
Training steps:     1,000,000 / 2048 ≈ 488 steps
Training time:      488 / 7 ≈ 70 sec ≈ 1.2 minutes
```

An entire training run that currently takes days could complete in hours.

---

## Part 5: Hybrid R-NaD + MARL Trading Module

### 5.1 Design Philosophy

The core tension: R-NaD learns competitive strategies (maximize own payoff against opponents),
but trading requires cooperation (mutual benefit). A pure R-NaD agent would learn to never
propose trades (since any trade helps an opponent), which is suboptimal in Catan where
4:1 bank trades are far worse than 1:1 player trades.

The hybrid design separates these concerns:

- **R-NaD component** handles all competitive actions: building, development cards, robber
  placement, road placement, maritime trade, end turn.
- **Trading component** handles domestic trade: propose, accept/reject. Trained with a
  cooperative MARL signal that rewards mutually-beneficial trades.

The two components share the same trunk (GNN + ResNet) but have separate heads and separate
loss terms. During a turn, the network first decides whether to trade (trading head), then
plays competitively (R-NaD head).

### 5.2 Network Architecture with Trading Heads

```
                    ┌─────────────────────────────────┐
                    │         Input Encoding           │
                    │  GNN(54 nodes) + Flat(115 dims)  │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │       ResNet Trunk (20 blk)      │
                    │         256-dim output           │
                    └───┬───────┬────────┬────────┬───┘
                        │       │        │        │
                   ┌────▼──┐ ┌──▼───┐ ┌──▼───┐ ┌──▼────┐
                   │PolicyA│ │Trade │ │Trade │ │ Value │
                   │ Head  │ │Propose│ │Accept│ │ Head  │
                   │(337)  │ │ Head │ │ Head │ │  (4)  │
                   └───────┘ └──────┘ └──────┘ └───────┘
                   R-NaD      MARL     MARL     Shared
```

### 5.3 Trade Proposal Head

The trade proposal head outputs a structured trade action. Rather than enumerating all
possible trades (combinatorial explosion), we use a **factored** output:

```python
class TradeProposalHead(nn.Module):
    """Propose a domestic trade or pass.

    Outputs:
        propose_prob: P(propose a trade this turn) — scalar in [0,1]
        give_logits: (5,) logits over resource types to give
        want_logits: (5,) logits over resource types to want
        give_amount: (4,) logits over amounts {1, 2, 3, 4}
        want_amount: (4,) logits over amounts {1, 2, 3, 4}
    """

    def __init__(self, trunk_channels: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(trunk_channels, 128),
            nn.Mish(),
        )
        self.propose_gate = nn.Sequential(
            nn.Linear(128, 1),
        )
        self.give_type = nn.Linear(128, 5)
        self.want_type = nn.Linear(128, 5)
        self.give_amount = nn.Linear(128, 4)
        self.want_amount = nn.Linear(128, 4)

    def forward(self, trunk_out: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(trunk_out)
        return {
            "propose_prob": torch.sigmoid(self.propose_gate(h)),
            "give_type_logits": self.give_type(h),
            "want_type_logits": self.want_type(h),
            "give_amount_logits": self.give_amount(h),
            "want_amount_logits": self.want_amount(h),
        }
```

**Action masking for trades**: The proposer can only give resources they actually have. The
`give_type_logits` should be masked by the proposer's resource counts. Similarly,
`give_amount` should be capped by the available amount of the selected resource type.

### 5.4 Trade Acceptance Head (Enhanced PolicyHeadB)

The existing `PolicyHeadB` already outputs P(accept). We enhance it to condition on the
proposed trade:

```python
class TradeAcceptHead(nn.Module):
    """Accept or reject a proposed trade, conditioned on the offer.

    Input: trunk output (256) concatenated with trade encoding (14).
    Trade encoding: [give_type_onehot(5), give_amount_onehot(4),
                     want_type_onehot(5)]
    (want_amount is implicit from give_amount for 1:1 trades,
     or encoded separately for asymmetric trades)
    """

    def __init__(self, trunk_channels: int = 256, trade_dim: int = 14):
        super().__init__()
        self.fc1 = nn.Linear(trunk_channels + trade_dim, 64)
        self.act = nn.Mish()
        self.fc_out = nn.Linear(64, 1)

    def forward(self, trunk_out: torch.Tensor,
                trade_encoding: torch.Tensor) -> torch.Tensor:
        h = torch.cat([trunk_out, trade_encoding], dim=-1)
        return torch.sigmoid(self.fc_out(self.act(self.fc1(h))))
```

### 5.5 Turn Flow with Trading

```
Start of Turn
    │
    ├─ Roll dice (or play knight) ← PolicyHeadA (R-NaD)
    │
    ├─ Resolve robber if 7 ← PolicyHeadA (R-NaD)
    │
    ├─ TRADE PHASE:
    │   ├─ TradeProposalHead: should I propose? → P(propose)
    │   │   ├─ If yes: sample (give_type, give_amount, want_type, want_amount)
    │   │   │   ├─ For each opponent p:
    │   │   │   │   TradeAcceptHead(p): accept? → P(accept)
    │   │   │   │   If any accept → execute trade (pick highest-priority acceptor)
    │   │   │   └─ If none accept → no trade
    │   │   └─ If no: skip to build phase
    │   └─ Optional: allow multiple trade proposals per turn (cap at 3)
    │
    ├─ BUILD PHASE:
    │   ├─ PolicyHeadA: build road/settlement/city/dev card, maritime trade, or end
    │   └─ (loop until END_TURN action selected)
    │
    └─ End Turn
```

### 5.6 Trading Loss Function

The trading heads use a **policy gradient** loss with a reward signal derived from the game
outcome, but filtered through a **mutual benefit** lens:

```python
class TradingLoss(nn.Module):
    """Loss for trade proposal and acceptance heads.

    For the proposer:
      L_propose = -log P(propose) · [R_proposer · I(trade_executed)]
                  -log(1 - P(propose)) · I(no_trade_proposed)
      where R_proposer = game_outcome_for_proposer (did proposing help them win?)

    For the acceptor:
      L_accept = -log P(accept) · R_acceptor · I(accepted)
                 -log(1 - P(accept)) · (-R_acceptor) · I(rejected)
      where R_acceptor = game_outcome_for_acceptor (did accepting help them win?)

    The key: trades that help the proposer win AND the acceptor place well
    get reinforced for both parties. Exploitative trades (proposer wins,
    acceptor loses badly) get reinforced for proposer but penalized for
    acceptor, causing acceptors to learn to reject them.
    """

    def __init__(self, propose_weight: float = 1.0, accept_weight: float = 1.0):
        super().__init__()
        self.propose_weight = propose_weight
        self.accept_weight = accept_weight

    def forward(
        self,
        propose_prob: torch.Tensor,       # (B,) P(propose)
        accept_prob: torch.Tensor,         # (B,) P(accept), for responding player
        proposed: torch.Tensor,            # (B,) bool: was a trade proposed?
        executed: torch.Tensor,            # (B,) bool: was a trade executed?
        proposer_reward: torch.Tensor,     # (B,) proposer's game outcome
        acceptor_reward: torch.Tensor,     # (B,) acceptor's game outcome
    ) -> dict[str, torch.Tensor]:

        # Proposal loss (REINFORCE on propose decision)
        propose_loss = -(
            proposed * torch.log(propose_prob + 1e-8) * proposer_reward * executed
            + (1 - proposed) * torch.log(1 - propose_prob + 1e-8)
        ).mean()

        # Acceptance loss (REINFORCE on accept decision)
        accept_loss = -(
            executed * torch.log(accept_prob + 1e-8) * acceptor_reward
            + (1 - executed) * torch.log(1 - accept_prob + 1e-8) * (-acceptor_reward)
        ).mean()

        total = self.propose_weight * propose_loss + self.accept_weight * accept_loss

        with torch.no_grad():
            propose_rate = proposed.float().mean()
            accept_rate = executed.float().mean()

        return {
            "trade_loss": total,
            "propose_loss": propose_loss,
            "accept_loss": accept_loss,
            "propose_rate": propose_rate,
            "accept_rate": accept_rate,
        }
```

### 5.7 Training Regime: Alternating Updates

The hybrid system alternates between competitive (R-NaD) and cooperative (trading) updates:

```python
def hybrid_training_step(net, anchor_net, optimizer, batch, config):
    """One training step combining R-NaD and trading losses.

    The total loss is:
        L = L_nerd + β_v · L_value + β_t · L_trade

    L_nerd and L_value use transformed rewards (R-NaD formulation).
    L_trade uses raw game outcomes (cooperative signal).

    These are NOT alternated in separate phases. They are computed on the
    same batch and summed. The gradients flow through the shared trunk,
    allowing the trunk to learn representations useful for both competitive
    play and trade evaluation.
    """
    optimizer.zero_grad(set_to_none=True)

    out = net(batch["input"])

    # --- R-NaD competitive loss ---
    rnad_losses = rnad_loss_fn(
        policy_logits=out["policy_logits"],
        anchor_log_probs=batch["anchor_log_probs"],
        action_mask=batch["action_mask"],
        vtrace_values=batch["vtrace_values"],
        pred_values=out["value"][:, 0],
        vtrace_q=batch["vtrace_q"],
    )

    # --- Trading cooperative loss ---
    trade_losses = trade_loss_fn(
        propose_prob=out["trade_propose"]["propose_prob"].squeeze(-1),
        accept_prob=out["trade_accept"].squeeze(-1),
        proposed=batch["trade_proposed"],
        executed=batch["trade_executed"],
        proposer_reward=batch["proposer_reward"],
        acceptor_reward=batch["acceptor_reward"],
    )

    # --- Combined ---
    total = (rnad_losses["total_loss"]
             + config.trade_loss_weight * trade_losses["trade_loss"])

    total.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()

    return {**rnad_losses, **trade_losses, "combined_loss": total}
```

### 5.8 Trade Curriculum

Trading is hard to learn from scratch because:
1. Early in training, the policy is random, so trades are random and produce no useful signal.
2. The acceptance head needs to see proposed trades to learn, but the proposal head needs
   accepted trades to learn. Chicken-and-egg problem.

**Phased curriculum:**

```
Phase 1 (steps 0 - 50K): No trading.
    Train R-NaD only on competitive actions.
    The network learns basic board play, resource management, building priorities.
    trade_loss_weight = 0.0

Phase 2 (steps 50K - 100K): Forced random trades.
    Inject random trade proposals with p=0.3 per turn.
    Train acceptance head on whether accepting random trades correlates with winning.
    trade_loss_weight = 0.1

Phase 3 (steps 100K - 200K): Learned proposals.
    Enable proposal head. Start with high exploration (temperature=2.0 on trade heads).
    Gradually reduce temperature to 1.0.
    trade_loss_weight = 0.5

Phase 4 (steps 200K+): Full hybrid.
    Both heads active, temperature=1.0.
    trade_loss_weight = 1.0
```

### 5.9 C Engine Modifications for Trade Support

The C engine needs modifications to support domestic trade execution:

```c
/* In apply_action.c, add to the switch statement: */

case AT_OFFER_TRADE: {
    /* action.value layout: [give_type, give_amount, want_type, want_amount, 0] */
    int give_type = action.value[0];
    int give_amount = action.value[1];
    int want_type = action.value[2];
    int want_amount = action.value[3];

    state->is_resolving_trade = 1;
    state->current_trade[0] = give_type;
    state->current_trade[1] = give_amount;
    state->current_trade[2] = want_type;
    state->current_trade[3] = want_amount;
    state->current_trade[4] = action.color;  /* proposer */
    state->prompt = PROMPT_DECIDE_TRADE;
    break;
}

case AT_ACCEPT_TRADE: {
    int proposer = state->current_trade[4];
    int acceptor = action.color;
    int give_type = state->current_trade[0];
    int give_amount = state->current_trade[1];
    int want_type = state->current_trade[2];
    int want_amount = state->current_trade[3];

    /* Transfer resources */
    state->player_state[proposer][RESOURCE_OFFSET + give_type] -= give_amount;
    state->player_state[acceptor][RESOURCE_OFFSET + give_type] += give_amount;
    state->player_state[acceptor][RESOURCE_OFFSET + want_type] -= want_amount;
    state->player_state[proposer][RESOURCE_OFFSET + want_type] += want_amount;

    state->is_resolving_trade = 0;
    state->prompt = PROMPT_BUILD_OR_TRADE;
    break;
}
```

### 5.10 Full Hybrid Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     HexaZero Hybrid System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─── Self-Play Worker (no MCTS) ────────────────────────────┐  │
│  │  C Engine (34K moves/sec)                                 │  │
│  │    ↓ state                                                │  │
│  │  StateEncoder → GNN(6 layers) + Flat → ResNet(20 blk)     │  │
│  │    ↓ trunk output                                         │  │
│  │  ├→ PolicyHeadA(337) ──── R-NaD competitive actions       │  │
│  │  ├→ TradeProposalHead ─── propose/pass + trade params     │  │
│  │  ├→ TradeAcceptHead ───── accept/reject (conditioned)     │  │
│  │  └→ ValueHead(4) ──────── per-seat value estimates        │  │
│  │    ↓ action                                               │  │
│  │  C Engine applies action, advance game                    │  │
│  │    ↓ trajectory                                           │  │
│  │  Store: (state, action, log_π, log_π̄, reward, player)    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓ trajectories                         │
│  ┌─── Trainer ───────────────────────────────────────────────┐  │
│  │  Reward Transform: r̃ = r + η·(log π̄ - log π)            │  │
│  │  V-trace: compute value targets and Q-values              │  │
│  │  NERD loss: ∇L = -Σ π(a)·∇log π(a)·clip(Ã(a))           │  │
│  │  + Value loss: MSE(V_pred, V_vtrace)                      │  │
│  │  + Trade loss: REINFORCE on propose/accept decisions      │  │
│  │  AdamW update, grad clip 1.0                              │  │
│  │  Every 200 steps: reset anchor π̄ ← π_θ                   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Hyperparameters:                                               │
│    η = 0.2 (regularization), C = 10000 (clip)                  │
│    γ = 1.0 (undiscounted, episodic)                             │
│    lr = 1e-3 (cosine schedule), batch = 2048                    │
│    64 concurrent games per worker, 4 workers                    │
│    Anchor reset every 200 training steps                        │
│                                                                 │
│  Expected throughput:                                           │
│    ~100 games/sec cluster-wide (83x over MCTS baseline)         │
│    ~20,000 positions/sec                                        │
│    Training not data-starved (unlike current MCTS setup)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Hyperparameter Recommendations

| Hyperparameter | DeepNash Value | HexaZero Recommendation | Rationale |
|---------------|----------------|------------------------|-----------|
| η (regularization) | 0.2 | 0.2 | Start with DeepNash default; tune based on policy entropy |
| C (NERD clip) | 10,000 | 10,000 | Large clip ≈ no clipping in practice; reduce if training unstable |
| Anchor reset interval | 200 steps | 200 steps | Paper default; tune based on convergence speed |
| γ (discount) | 1.0 | 1.0 | Episodic game, no discounting needed |
| ρ̄ (V-trace IS clip) | 1.0 | 1.0 | On-policy data, IS ratios ≈ 1.0 |
| c̄ (V-trace trace clip) | 1.0 | 1.0 | Standard |
| Concurrent games | - | 64 per worker | Balance GPU batch efficiency vs memory |
| Learning rate | - | 1e-3 with cosine | Match current setup |
| Batch size | - | 2048-4096 | Larger batches for stability |
| Value loss weight | - | 0.5 | Lower than policy (NERD) weight |
| Trade loss weight | - | 0.0 → 1.0 (curriculum) | Phase in over 200K steps |

## Appendix B: Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| R-NaD doesn't converge in 4-player setting | Medium | High | Fall back to independent PPO; CCE still holds with any no-regret learner |
| Policy quality worse than MCTS | Medium | Medium | MCTS can be added back as test-time search on top of R-NaD policy (best of both) |
| Trading module doesn't learn useful trades | High | Low | Maritime trades still work; disable domestic trading if it hurts |
| Anchor reset frequency wrong | Medium | Low | Grid search over {100, 200, 500, 1000}; monitor policy entropy |
| V-trace off-policy correction insufficient | Low | Medium | Data is nearly on-policy (single GPU step between collection and training) |
| C engine trade bugs | Low | Medium | Unit test trade action handling thoroughly before training |
| GPU memory insufficient for 64 concurrent | Low | Low | Reduce to 32; TITAN RTX has 24GB, model is only 3.8M params |

## Appendix C: Implementation Phasing

**Phase 1 (Week 1-2): Core R-NaD without trading.**
- Implement `RNaDLoss`, `compute_vtrace`, `transform_rewards`
- Implement MCTS-free self-play worker
- Modify `ValueHead` to output raw values
- Run baseline experiment: R-NaD vs AlphaZero on win rate against AB2
- Target: confirm R-NaD trains stably and produces competitive play

**Phase 2 (Week 3): Throughput optimization.**
- Optimize batched inference for 64+ concurrent games
- Profile GPU utilization, identify remaining bottlenecks
- Implement efficient trajectory storage and collation
- Target: achieve >50 games/sec per worker

**Phase 3 (Week 4-5): Trading module.**
- Implement C engine domestic trade support
- Add `TradeProposalHead` and `TradeAcceptHead`
- Implement `TradingLoss` and trade curriculum
- Extend `ActionEncoder` for trade actions
- Target: network learns to propose and accept beneficial trades

**Phase 4 (Week 6): Evaluation and tuning.**
- Full evaluation suite: vs AB2, vs random, self-play ELO
- Hyperparameter sweep on η, anchor interval, trade loss weight
- Compare: R-NaD only, R-NaD + trading, MCTS baseline
- Target: R-NaD + trading beats MCTS baseline on win rate and throughput

## References

- Perolat, J., et al. (2022). "Mastering the Game of Stratego with Model-Free Multiagent
  Reinforcement Learning." *Science*, 378(6623), 990-996. (DeepNash / R-NaD)
- Espeholt, L., et al. (2018). "IMPALA: Scalable Distributed Deep-RL with Importance
  Weighted Actor-Learner Architectures." *ICML 2018*. (V-trace)
- Hart, S. & Mas-Colell, A. (2000). "A Simple Adaptive Procedure Leading to Correlated
  Equilibrium." *Econometrica*, 68(5), 1127-1150. (CCE via no-regret)
- Chen, X. & Deng, X. (2006). "Settling the Complexity of Two-Player Nash Equilibrium."
  *FOCS 2006*. (PPAD-completeness)
- Silver, D., et al. (2018). "A General Reinforcement Learning Algorithm that Masters Chess,
  Shogi, and Go through Self-Play." *Science*, 362(6419), 1140-1144. (AlphaZero)
