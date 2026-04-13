from __future__ import annotations

import threading
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

ACTION_SPACE_SIZE = 337
NUM_PLAYERS = 4


@dataclass
class TrainingExample:
    """Single training example from self-play."""

    state_tensors: dict[str, torch.Tensor]
    policy_target: torch.Tensor  # (ACTION_SPACE_SIZE,)
    value_target: torch.Tensor  # (NUM_PLAYERS,)


@dataclass
class TrainingBatch:
    """Collated batch of training examples, ready for GPU.

    Fields
    ------
    node_features  : (batch, 54, node_feat_dim)
    edge_index     : (2, num_edges)          — shared graph topology
    edge_features  : (batch, num_edges, edge_feat_dim)
    flat_features  : (batch, flat_feat_dim)
    action_masks   : (batch, 337)
    policy_targets : (batch, 337)
    value_targets  : (batch, 4)
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    flat_features: torch.Tensor
    action_masks: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor

    def to(self, device: str | torch.device) -> TrainingBatch:
        return TrainingBatch(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def pin_memory(self) -> TrainingBatch:
        return TrainingBatch(
            **{f.name: getattr(self, f.name).pin_memory() for f in fields(self)}
        )

    @property
    def batch_size(self) -> int:
        return self.node_features.size(0)


class ReplayBuffer:
    """Fixed-size circular replay buffer for AlphaZero training data.

    Stores ``(state_tensors, policy_target, value_target)`` tuples.
    Fixed-shape targets live in pre-allocated numpy arrays; variable-shape
    state dicts are kept in a plain list.  All mutating operations are
    guarded by a :class:`threading.Lock`.
    """

    def __init__(self, capacity: int = 1_000_000) -> None:
        self._capacity = capacity
        self._size = 0
        self._write_idx = 0
        self._lock = threading.Lock()

        self._policies = np.zeros((capacity, ACTION_SPACE_SIZE), dtype=np.float32)
        self._values = np.zeros((capacity, NUM_PLAYERS), dtype=np.float32)
        self._states: list[dict[str, torch.Tensor] | None] = [None] * capacity

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def push(
        self,
        state_tensors: dict[str, torch.Tensor],
        policy: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Add a single training example.  Thread-safe."""
        with self._lock:
            self._write_single(state_tensors, policy, value)

    def push_game(self, game_data: list[TrainingExample]) -> None:
        """Add all positions from a completed game in one lock acquisition."""
        with self._lock:
            for ex in game_data:
                self._write_single(ex.state_tensors, ex.policy_target, ex.value_target)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> TrainingBatch:
        """Sample a random minibatch and return collated tensors."""
        with self._lock:
            if self._size < batch_size:
                raise ValueError(
                    f"Not enough data: buffer contains {self._size} examples, "
                    f"requested {batch_size}"
                )
            indices = np.random.choice(self._size, size=batch_size, replace=False)
            states = [self._states[i] for i in indices]
            policies = self._policies[indices].copy()
            values = self._values[indices].copy()

        return _collate(states, policies, values)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize buffer contents to disk via :func:`torch.save`."""
        with self._lock:
            torch.save(
                {
                    "capacity": self._capacity,
                    "size": self._size,
                    "write_idx": self._write_idx,
                    "policies": self._policies[: self._size],
                    "values": self._values[: self._size],
                    "states": [self._states[i] for i in range(self._size)],
                },
                path,
            )

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        """Deserialize a previously saved buffer."""
        data = torch.load(path, weights_only=False, map_location="cpu")
        buf = cls(capacity=data["capacity"])
        size: int = data["size"]
        buf._policies[:size] = data["policies"]
        buf._values[:size] = data["values"]
        for i, state in enumerate(data["states"]):
            buf._states[i] = state
        buf._size = size
        buf._write_idx = data["write_idx"]
        return buf

    # ------------------------------------------------------------------
    # Internals (caller must hold ``self._lock``)
    # ------------------------------------------------------------------

    def _write_single(
        self,
        state_tensors: dict[str, torch.Tensor],
        policy: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        idx = self._write_idx
        self._states[idx] = {k: v.detach().cpu() for k, v in state_tensors.items()}
        self._policies[idx] = policy.detach().cpu().numpy()
        self._values[idx] = value.detach().cpu().numpy()
        self._write_idx = (idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)


# ----------------------------------------------------------------------
# Collation
# ----------------------------------------------------------------------


def _collate(
    states: list[dict[str, torch.Tensor] | None],
    policies: np.ndarray,
    values: np.ndarray,
) -> TrainingBatch:
    """Stack individual examples into a single :class:`TrainingBatch`."""
    typed: list[dict[str, torch.Tensor]] = states  # type: ignore[assignment]
    return TrainingBatch(
        node_features=torch.stack([s["node_features"] for s in typed]),
        edge_index=typed[0]["edge_index"],  # fixed topology, shared across batch
        edge_features=torch.stack([s["edge_features"] for s in typed]),
        flat_features=torch.stack([s["flat_features"] for s in typed]),
        action_masks=torch.stack([s["action_masks"] for s in typed]),
        policy_targets=torch.from_numpy(policies),
        value_targets=torch.from_numpy(values),
    )
