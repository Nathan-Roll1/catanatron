"""MCTS search engine with IS-MCTS, PUCT, Dirichlet noise, and temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from hexzero.mcts.node import MCTSNode

if TYPE_CHECKING:
    from hexzero.config import MCTSConfig
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.encoder.state_encoder import StateEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.model.network import HexaZeroNet

ACTION_SPACE_SIZE: int = 337


@dataclass
class MCTSResult:
    """Outcome of an MCTS search."""

    action_probs: np.ndarray  # (ACTION_SPACE_SIZE,) visit-count distribution
    best_action: int  # action index with the most aggregate visits
    root_value: np.ndarray  # (4,) value estimate for the searched position
    visit_counts: dict[int, int]  # action_index -> aggregate visit count
    num_simulations: int  # total simulations executed


class MCTSSearch:
    """Full MCTS search with neural-network evaluation.

    Implements:
      1. PUCT-based tree policy
      2. Dirichlet noise at the root for exploration
      3. Temperature-controlled action selection
      4. Information Set MCTS via determinization
      5. Virtual loss bookkeeping for future parallelism
      6. Optional tree reuse across consecutive searches
    """

    def __init__(
        self,
        network: HexaZeroNet,
        encoder: StateEncoder,
        action_encoder: ActionEncoder,
        config: MCTSConfig,
        device: str = "cuda",
    ) -> None:
        self.network = network
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.config = config
        self.device = device
        self._cached_root: MCTSNode | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self, game: CatanGame, num_simulations: int | None = None
    ) -> MCTSResult:
        """Run IS-MCTS from the current game position.

        Spawns *num_determinizations* independent trees (each with a
        sampled view of hidden information), runs simulations on each,
        and aggregates visit counts to produce a single action distribution.
        """
        num_sims = num_simulations or self.config.num_simulations
        num_det = self.config.num_determinizations
        sims_per_det = max(num_sims // num_det, 1)

        agg_visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.float64)
        agg_value = np.zeros(4, dtype=np.float64)
        trees_used = 0

        for _ in range(num_det):
            det_game = self._determinize_state(game)
            root = self._make_root(det_game)

            if root.is_terminal or not root.is_expanded:
                if root.is_terminal and root.terminal_value is not None:
                    agg_value += root.terminal_value
                    trees_used += 1
                continue

            self._add_dirichlet_noise(root)

            for _ in range(sims_per_det):
                self._run_simulation(root, det_game)

            for aidx, child in root.children.items():
                agg_visits[aidx] += child.visit_count
            agg_value += root.total_value / max(root.visit_count, 1)
            trees_used += 1

            if self.config.max_tree_reuse and num_det == 1:
                self._cached_root = root

        if trees_used > 0:
            agg_value /= trees_used

        total_visits = agg_visits.sum()
        if total_visits > 0:
            action_probs = (agg_visits / total_visits).astype(np.float32)
        else:
            action_probs = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

        visit_counts = {
            int(i): int(v) for i, v in enumerate(agg_visits) if v > 0
        }
        best_action = int(np.argmax(agg_visits)) if total_visits > 0 else 0

        return MCTSResult(
            action_probs=action_probs,
            best_action=best_action,
            root_value=agg_value.astype(np.float32),
            visit_counts=visit_counts,
            num_simulations=num_sims,
        )

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float,
        turn_number: int,
    ) -> int:
        """Pick an action from the visit-count distribution.

        Early game (turn < *temperature_threshold*): sample from
        ``probs^{1/tau}`` (exploratory).
        Late game: near-deterministic argmax.
        """
        if turn_number >= self.config.temperature_threshold:
            temperature = self.config.temperature_final

        if temperature < 0.05:
            return int(np.argmax(action_probs))

        nonzero = action_probs > 0
        if not np.any(nonzero):
            return int(np.argmax(action_probs))

        log_p = np.full_like(action_probs, -np.inf, dtype=np.float64)
        log_p[nonzero] = np.log(action_probs[nonzero]) / temperature
        log_p -= log_p[nonzero].max()
        scaled = np.zeros_like(action_probs, dtype=np.float64)
        scaled[nonzero] = np.exp(log_p[nonzero])
        total = scaled.sum()
        if total <= 0:
            return int(np.argmax(action_probs))
        scaled /= total

        return int(np.random.choice(len(action_probs), p=scaled))

    def advance_tree(self, action_index: int) -> None:
        """Advance the cached tree after a real move is played.

        Promotes the subtree under *action_index* to root, discarding
        the rest of the old tree.  Only meaningful when tree reuse is
        enabled and ``num_determinizations == 1``.
        """
        if self._cached_root is None:
            return
        if action_index in self._cached_root.children:
            self._cached_root = self._cached_root.children[action_index]
            self._cached_root.parent = None
        else:
            self._cached_root = None

    # ------------------------------------------------------------------
    # Simulation internals
    # ------------------------------------------------------------------

    def _make_root(self, game: CatanGame) -> MCTSNode:
        """Return a root node, reusing a cached subtree when possible."""
        if (
            self.config.max_tree_reuse
            and self.config.num_determinizations == 1
            and self._cached_root is not None
        ):
            root = self._cached_root
            self._cached_root = None
            return root

        root = MCTSNode(player=game.current_player())

        if game.is_terminal():
            root.is_terminal = True
            root.terminal_value = self._terminal_value(game)
            return root

        priors, value = self._evaluate_leaf(game)
        if priors:
            root.expand(priors, game.current_player())
            # Seed root with one backup so the first simulation has a
            # non-zero parent visit count for PUCT exploration.
            root.backup(value)

        return root

    def _run_simulation(self, root: MCTSNode, game: CatanGame) -> None:
        """Execute one select -> expand -> evaluate -> backup pass."""
        node = root
        scratch = game.clone()
        path: list[MCTSNode] = [node]

        # --- Selection: walk down the tree by PUCT ---
        while node.is_expanded and not node.is_terminal and not scratch.is_terminal():
            player = scratch.current_player()
            node.player = player
            action_idx, node = node.select_child(self.config.c_puct, player)
            self._apply_action_by_index(scratch, action_idx)
            path.append(node)

        # --- Virtual loss (discourages concurrent threads from this path) ---
        vl = self.config.virtual_loss
        for n in path:
            n.apply_virtual_loss(vl)

        # --- Evaluate ---
        value: np.ndarray
        if node.is_terminal:
            assert node.terminal_value is not None
            value = node.terminal_value
        elif scratch.is_terminal():
            value = self._terminal_value(scratch)
            node.is_terminal = True
            node.terminal_value = value
        else:
            priors, value = self._evaluate_leaf(scratch)
            if priors:
                node.expand(priors, scratch.current_player())
            else:
                value = self._terminal_value(scratch)
                node.is_terminal = True
                node.terminal_value = value

        # --- Revert virtual loss, then real backup ---
        for n in path:
            n.revert_virtual_loss(vl)
        node.backup(value)

    def _evaluate_leaf(
        self, game: CatanGame
    ) -> tuple[dict[int, float], np.ndarray]:
        """Query the neural network for policy priors and a value estimate.

        Returns:
            action_priors: ``{action_index: prior_probability}`` for legal moves.
            value: Shape ``(4,)`` win-probability vector.
        """
        # TODO: batch multiple positions into a single forward pass for
        # higher GPU throughput during parallel self-play.
        state_view = game.get_state_view()
        encoded = self.encoder.encode(state_view)
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.network.predict(batch)

        policy = output["policy_probs"][0]   # (337,) numpy
        value_raw: np.ndarray = output["value"][0]  # (4,) raw logits
        # Softmax to convert logits to probabilities for MCTS backup
        value_exp = np.exp(value_raw - value_raw.max())
        value: np.ndarray = value_exp / value_exp.sum()

        legal_actions = game.get_legal_actions()
        if not legal_actions:
            return {}, value

        mask_t = self.action_encoder.get_action_mask(legal_actions)
        mask = mask_t.numpy()
        masked = policy * mask
        total = float(masked.sum())

        if total > 0:
            masked = masked / total
        else:
            masked = mask / float(mask.sum())

        action_priors: dict[int, float] = {}
        for action in legal_actions:
            idx = self.action_encoder.encode(action)
            prob = float(masked[idx])
            if prob > 0:
                action_priors[idx] = prob

        if not action_priors:
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                action_priors[self.action_encoder.encode(action)] = uniform

        return action_priors, value

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Mix Dirichlet noise into the root's children priors.

        ``P'(s,a) = (1 - eps) * P_nn(s,a) + eps * Dir(alpha)``
        """
        if not root.children:
            return

        actions = list(root.children.keys())
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(actions)
        )
        eps = self.config.dirichlet_epsilon
        for i, aidx in enumerate(actions):
            child = root.children[aidx]
            child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])

    def _determinize_state(self, game: CatanGame) -> CatanGame:
        """Sample a determinization of hidden information.

        A proper implementation resamples opponent hands and the dev-card
        deck order from the distribution consistent with public observations.
        Requires game-level support for hidden-state resampling.
        """
        # TODO: implement proper IS sampling once the game interface
        # exposes hidden-state resampling (opponent hands, deck order).
        return game.clone()

    def _apply_action_by_index(self, game: CatanGame, action_space_idx: int) -> None:
        """Apply an action_space_index to the game by finding it in legal actions."""
        if game.is_terminal():
            return
        legal = game.get_legal_actions()
        if not legal:
            return
        for i, action in enumerate(legal):
            if self.action_encoder.encode(action) == action_space_idx:
                game.step(i)
                return
        game.step(0)

    @staticmethod
    def _terminal_value(game: CatanGame) -> np.ndarray:
        """Build a value vector for a finished game."""
        value = np.zeros(4, dtype=np.float32)
        winner = game.winner()
        if winner is not None:
            value[winner] = 1.0
        else:
            value[:] = 0.25
        return value
