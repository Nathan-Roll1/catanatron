"""MCTS tree node with PUCT selection and multi-player value backup."""

from __future__ import annotations

import math

import numpy as np


class MCTSNode:
    """A node in the MCTS search tree.

    Each node corresponds to a game state reached by taking a specific action
    from its parent.  Stores visit statistics for PUCT-based selection and
    multi-player value backup.

    Attributes:
        parent: Parent node, or ``None`` for the root.
        children: Map from action index to child node.
        visit_count: Number of times this node has been visited (*N*).
        total_value: Sum of backed-up values, shape ``(4,)``, one per player.
        prior: Prior probability *P(s, a)* from the neural network.
        action_index: The action that led to this node (``-1`` for root).
        player: The player whose turn it is at this node.
        is_expanded: Whether children have been created.
        is_terminal: Whether the game is over at this node.
        terminal_value: Final outcome if terminal, shape ``(4,)`` or ``None``.
    """

    __slots__ = (
        "parent",
        "children",
        "visit_count",
        "total_value",
        "prior",
        "action_index",
        "player",
        "is_expanded",
        "is_terminal",
        "terminal_value",
    )

    def __init__(
        self,
        prior: float = 0.0,
        action_index: int = -1,
        player: int = 0,
        parent: MCTSNode | None = None,
    ) -> None:
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: np.ndarray = np.zeros(4, dtype=np.float32)
        self.prior = prior
        self.action_index = action_index
        self.player = player
        self.is_expanded: bool = False
        self.is_terminal: bool = False
        self.terminal_value: np.ndarray | None = None

    # -----------------------------------------------------------------
    # PUCT selection
    # -----------------------------------------------------------------

    def q_value(self, player: int) -> float:
        """Average value *Q(s, a)* from the perspective of *player*."""
        if self.visit_count == 0:
            return 0.0
        return float(self.total_value[player]) / self.visit_count

    def ucb_score(
        self, c_puct: float, parent_visits: int, player: int
    ) -> float:
        """PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1+N)."""
        q = self.q_value(player)
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

    def select_child(
        self, c_puct: float, player: int
    ) -> tuple[int, MCTSNode]:
        """Return ``(action_index, child)`` with the highest PUCT score."""
        best_score = -math.inf
        best_action = -1
        best_child: MCTSNode | None = None

        for action_idx, child in self.children.items():
            score = child.ucb_score(c_puct, self.visit_count, player)
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child

        assert best_child is not None, "select_child called on node with no children"
        return best_action, best_child

    # -----------------------------------------------------------------
    # Expansion
    # -----------------------------------------------------------------

    def expand(self, action_priors: dict[int, float], player: int) -> None:
        """Create child nodes for every legal action with its NN prior.

        Args:
            action_priors: ``{action_index: prior_probability}``.
            player: The player whose turn it is at *this* node.
        """
        self.player = player
        for action_idx, prob in action_priors.items():
            if action_idx not in self.children:
                self.children[action_idx] = MCTSNode(
                    prior=prob, action_index=action_idx, parent=self
                )
        self.is_expanded = True

    # -----------------------------------------------------------------
    # Backup
    # -----------------------------------------------------------------

    def backup(self, value: np.ndarray) -> None:
        """Propagate *value* from this node up to the root.

        Args:
            value: Win-probability vector, shape ``(4,)``, one per player.
        """
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent

    # -----------------------------------------------------------------
    # Virtual loss (for future parallel MCTS)
    # -----------------------------------------------------------------

    def apply_virtual_loss(self, virtual_loss: float) -> None:
        """Inflate visit count by *virtual_loss* phantom visits.

        Drives Q toward zero, discouraging concurrent threads from
        selecting this node while a simulation is in flight.
        """
        self.visit_count += int(virtual_loss)

    def revert_virtual_loss(self, virtual_loss: float) -> None:
        """Remove phantom visits added by :meth:`apply_virtual_loss`."""
        self.visit_count -= int(virtual_loss)

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        q = self.total_value / max(self.visit_count, 1)
        return (
            f"MCTSNode(action={self.action_index}, player={self.player}, "
            f"visits={self.visit_count}, prior={self.prior:.4f}, "
            f"q=[{', '.join(f'{v:.3f}' for v in q)}], "
            f"children={len(self.children)})"
        )
