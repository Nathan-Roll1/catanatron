from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from hexzero.config import HexaZeroConfig

from .replay_buffer import NUM_PLAYERS, TrainingExample, ReplayBuffer

if TYPE_CHECKING:
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.encoder.state_encoder import StateEncoder
    from hexzero.model.network import HexaZeroNet

logger = logging.getLogger(__name__)


# ======================================================================
# Data containers
# ======================================================================


@dataclass
class GameRecord:
    """Complete record of a self-play game."""

    examples: list[TrainingExample]
    winner: int  # player index who won, -1 for draw / timeout
    num_turns: int
    game_seed: int
    mcts_values: list[float] = field(default_factory=list)

    def to_training_examples(self) -> list[TrainingExample]:
        return list(self.examples)


@dataclass
class SelfPlayStats:
    """Aggregate statistics for a batch of self-play games."""

    games_played: int = 0
    total_positions: int = 0
    avg_game_length: float = 0.0
    win_distribution: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_PLAYERS, dtype=np.float64)
    )
    avg_mcts_value: float = 0.0
    positions_per_second: float = 0.0

    def __repr__(self) -> str:
        wins = ", ".join(f"P{i}={int(w)}" for i, w in enumerate(self.win_distribution))
        return (
            f"SelfPlayStats(games={self.games_played}, "
            f"positions={self.total_positions}, "
            f"avg_len={self.avg_game_length:.1f}, "
            f"wins=[{wins}], "
            f"pos/s={self.positions_per_second:.1f})"
        )


# ======================================================================
# Single worker
# ======================================================================


class SelfPlayWorker:
    """Generates training data through self-play games.

    Each worker plays complete games using MCTS to select moves.  All
    player seats share the same neural network (pure self-play).
    """

    def __init__(
        self,
        worker_id: int,
        network: HexaZeroNet,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder,
        config: HexaZeroConfig,
        device: str = "cpu",
    ) -> None:
        self._id = worker_id
        self._network = network
        self._encoder = state_encoder
        self._action_encoder = action_encoder
        self._config = config
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def play_game(self, seed: int | None = None) -> GameRecord:
        """Play one complete self-play game and return its record.

        1. Reset game environment.
        2. Loop until terminal / max-length:
           a. Encode state, run MCTS, select action via temperature schedule.
           b. Optionally resign if root value is below threshold.
           c. Record ``(state, policy, current_player)`` tuple.
           d. Step the game.
        3. Assign value targets using actual outcome (rotated per position).
        """
        from hexzero.game.interface import CatanGame
        from hexzero.mcts.search import MCTSSearch

        game = CatanGame(config=self._config.game)
        game.reset(seed=seed)

        mcts = MCTSSearch(
            network=self._network,
            encoder=self._encoder,
            action_encoder=self._action_encoder,
            config=self._config.mcts,
            device=self._device,
        )

        history: list[tuple[dict[str, torch.Tensor], torch.Tensor, int]] = []
        mcts_values: list[float] = []

        while (
            not game.is_terminal()
            and game.turn_number < self._config.selfplay.max_game_length
        ):
            current_player = game.current_player()

            state_view = game.get_state_view()
            state_tensors = self._encoder.encode(state_view)
            # Move to CPU immediately to avoid accumulating on GPU
            state_tensors = {k: v.detach().cpu() for k, v in state_tensors.items()}

            result = mcts.search(game)

            # Temperature schedule: explore early, exploit later
            temperature = (
                self._config.mcts.temperature_init
                if game.turn_number < self._config.mcts.temperature_threshold
                else self._config.mcts.temperature_final
            )
            action_space_idx = mcts.select_action(
                result.action_probs, temperature, game.turn_number
            )

            # Optional resignation
            if self._config.selfplay.resign_enabled:
                player_value = float(result.root_value[current_player])
                if player_value < self._config.selfplay.resign_threshold:
                    logger.debug(
                        "Worker %d: player %d resigns (value=%.3f)",
                        self._id,
                        current_player,
                        player_value,
                    )
                    break

            # Store action mask alongside state tensors for training
            legal = game.get_legal_actions()
            action_mask = self._action_encoder.get_action_mask(legal)
            state_tensors["action_masks"] = action_mask

            policy = torch.from_numpy(result.action_probs).float()
            history.append((state_tensors, policy, current_player))
            mcts_values.append(float(result.root_value[current_player]))

            # Convert action_space_index to legal action list index
            legal_idx = 0
            for i, a in enumerate(legal):
                if self._action_encoder.encode(a) == action_space_idx:
                    legal_idx = i
                    break
            game.step(legal_idx)

        # Determine winner
        winner: int = game.winner() if game.is_terminal() else -1  # type: ignore[assignment]
        if winner is None:
            winner = -1

        examples = _build_examples(history, winner, self._config.game.num_players)

        return GameRecord(
            examples=examples,
            winner=winner,
            num_turns=game.turn_number,
            game_seed=seed if seed is not None else -1,
            mcts_values=mcts_values,
        )

    def play_games(
        self, num_games: int, start_seed: int = 0
    ) -> list[GameRecord]:
        """Play *num_games* sequentially, using deterministic seeds."""
        records: list[GameRecord] = []
        for i in range(num_games):
            record = self.play_game(seed=start_seed + i)
            records.append(record)
            logger.info(
                "Worker %d game %d/%d: %d turns, winner=%d",
                self._id,
                i + 1,
                num_games,
                record.num_turns,
                record.winner,
            )
        return records


# ======================================================================
# Manager
# ======================================================================


class SelfPlayManager:
    """Coordinates multiple self-play workers and feeds a replay buffer.

    Responsibilities:
    - Spawn workers (threaded; GIL is released during numpy / CUDA ops).
    - Collect :class:`GameRecord` objects.
    - Push training examples to the :class:`ReplayBuffer`.
    - Track running statistics.
    """

    def __init__(
        self,
        network: HexaZeroNet,
        replay_buffer: ReplayBuffer,
        config: HexaZeroConfig,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder,
    ) -> None:
        self._network = network
        self._buffer = replay_buffer
        self._config = config
        self._encoder = state_encoder
        self._action_encoder = action_encoder
        self._games_played = 0
        self._total_positions = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_games(self, num_games: int) -> SelfPlayStats:
        """Generate *num_games* self-play games and push results to buffer."""
        self._network.eval()
        start = time.monotonic()

        num_workers = min(self._config.selfplay.num_workers, num_games)
        device = self._config.device
        all_records: list[GameRecord] = []

        if num_workers <= 1:
            worker = self._make_worker(0, device)
            all_records = worker.play_games(num_games, self._games_played)
        else:
            chunks = _distribute(num_games, num_workers)
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = []
                seed_offset = self._games_played
                for wid, count in enumerate(chunks):
                    worker = self._make_worker(wid, device)
                    futures.append(
                        pool.submit(worker.play_games, count, seed_offset)
                    )
                    seed_offset += count
                for fut in as_completed(futures):
                    all_records.extend(fut.result())

        # Feed replay buffer
        total_positions = 0
        for record in all_records:
            self._buffer.push_game(record.examples)
            total_positions += len(record.examples)

        self._games_played += len(all_records)
        self._total_positions += total_positions

        elapsed = time.monotonic() - start
        stats = _compute_stats(all_records, total_positions, elapsed)
        logger.info("Self-play iteration: %s", stats)
        return stats

    def run_iteration(self) -> SelfPlayStats:
        """Run one iteration (``games_per_iteration`` games)."""
        return self.generate_games(self._config.selfplay.games_per_iteration)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_worker(self, worker_id: int, device: str) -> SelfPlayWorker:
        return SelfPlayWorker(
            worker_id=worker_id,
            network=self._network,
            state_encoder=self._encoder,
            action_encoder=self._action_encoder,
            config=self._config,
            device=device,
        )


# ======================================================================
# Helpers
# ======================================================================


def _build_examples(
    history: list[tuple[dict[str, torch.Tensor], torch.Tensor, int]],
    winner: int,
    num_players: int,
) -> list[TrainingExample]:
    """Convert per-move history into :class:`TrainingExample` objects.

    Value targets are one-hot vectors *rotated* so that index 0 always
    corresponds to the player-to-move at that position.  This matches the
    network's value head which predicts from the current player's viewpoint.
    """
    examples: list[TrainingExample] = []
    for state_tensors, policy, player_at_turn in history:
        if winner < 0:
            value_target = torch.zeros(num_players, dtype=torch.float32)
        else:
            value_target = torch.zeros(num_players, dtype=torch.float32)
            rotated_idx = (winner - player_at_turn) % num_players
            value_target[rotated_idx] = 1.0
        examples.append(
            TrainingExample(
                state_tensors=state_tensors,
                policy_target=policy,
                value_target=value_target,
            )
        )
    return examples


def _distribute(total: int, num_workers: int) -> list[int]:
    """Divide *total* items as evenly as possible across *num_workers*."""
    base, remainder = divmod(total, num_workers)
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


def _compute_stats(
    records: list[GameRecord],
    total_positions: int,
    elapsed: float,
) -> SelfPlayStats:
    n = len(records)
    if n == 0:
        return SelfPlayStats()

    lengths = [r.num_turns for r in records]
    wins = np.zeros(NUM_PLAYERS, dtype=np.float64)
    for r in records:
        if 0 <= r.winner < NUM_PLAYERS:
            wins[r.winner] += 1

    all_mcts_vals = [v for r in records for v in r.mcts_values]
    avg_mcts = float(np.mean(all_mcts_vals)) if all_mcts_vals else 0.0

    return SelfPlayStats(
        games_played=n,
        total_positions=total_positions,
        avg_game_length=float(np.mean(lengths)),
        win_distribution=wins,
        avg_mcts_value=avg_mcts,
        positions_per_second=total_positions / max(elapsed, 1e-9),
    )
