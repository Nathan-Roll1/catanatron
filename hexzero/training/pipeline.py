"""Full AlphaZero training pipeline for HexaZero.

Each iteration:
  1. Self-play  — generate games with MCTS + current network.
  2. Training   — sample from the replay buffer and update the network.
  3. Evaluation — pit the new network against the AB2 baseline.
  4. Checkpoint — save the model if it has improved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from hexzero.config import HexaZeroConfig
from hexzero.training.trainer import Trainer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Iteration result
# ---------------------------------------------------------------------------


@dataclass
class IterationResult:
    iteration: int
    selfplay_stats: dict[str, Any] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)
    eval_elo: float | None = None
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Orchestrates the full AlphaZero loop.

    Concrete self-play and evaluation components are plugged in lazily via
    ``initialize``; this module provides the outer loop and ties the pieces
    together.
    """

    def __init__(self, config: HexaZeroConfig) -> None:
        self.config = config

        self.network: Any = None
        self.trainer: Trainer | None = None
        self.replay_buffer: Any = None
        self.inference_server: Any = None

        self._iteration: int = 0
        self._best_elo: float = config.elo.initial_elo
        self._log_dir = Path(config.log_dir)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build all components.  Call once before :meth:`run`."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log.info("Initialising HexaZero training pipeline")
        log.info("  device : %s", self.config.device)
        log.info("  log_dir: %s", self._log_dir)

    def _ensure_initialized(self) -> None:
        if self.network is None:
            raise RuntimeError(
                "Pipeline not initialised — call initialize() first."
            )

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_iteration(self, iteration: int) -> IterationResult:
        """Execute one self-play → train → evaluate cycle."""
        self._ensure_initialized()
        t0 = time.monotonic()
        result = IterationResult(iteration=iteration)

        # 1. Self-play ------------------------------------------------
        log.info("=== Iteration %d: self-play ===", iteration)
        selfplay_stats = self._run_selfplay()
        result.selfplay_stats = selfplay_stats

        # 2. Training -------------------------------------------------
        log.info("=== Iteration %d: training ===", iteration)
        assert self.trainer is not None
        training_metrics = self.trainer.train_iteration(self.replay_buffer)
        result.training_metrics = training_metrics
        log.info(
            "  loss=%.4f  vloss=%.4f  ploss=%.4f  vacc=%.3f",
            training_metrics.get("total_loss", 0),
            training_metrics.get("value_loss", 0),
            training_metrics.get("policy_loss", 0),
            training_metrics.get("value_accuracy", 0),
        )

        # 3. Evaluation -----------------------------------------------
        if iteration % self.config.training.checkpoint_interval == 0:
            log.info("=== Iteration %d: evaluation ===", iteration)
            elo = self.evaluate_current()
            result.eval_elo = elo
            if elo > self._best_elo:
                log.info("  New best ELO: %.1f (was %.1f)", elo, self._best_elo)
                self._best_elo = elo
                self._save(iteration, elo, is_best=True)
            else:
                log.info("  ELO %.1f did not beat best %.1f", elo, self._best_elo)

        # 4. Periodic checkpoint --------------------------------------
        if iteration % self.config.training.checkpoint_interval == 0:
            self._save(iteration, result.eval_elo)

        result.elapsed_seconds = time.monotonic() - t0
        self._iteration = iteration
        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, num_iterations: int = 1000, resume_from: int = 0) -> None:
        """Run the full training loop for *num_iterations* iterations."""
        self._ensure_initialized()
        log.info(
            "Starting training: %d iterations from %d",
            num_iterations,
            resume_from,
        )
        for it in range(resume_from, resume_from + num_iterations):
            result = self.run_iteration(it)
            log.info(
                "Iteration %d done in %.1fs  loss=%.4f  elo=%s",
                it,
                result.elapsed_seconds,
                result.training_metrics.get("total_loss", float("nan")),
                f"{result.eval_elo:.0f}" if result.eval_elo is not None else "—",
            )

    # ------------------------------------------------------------------
    # Evaluation stub
    # ------------------------------------------------------------------

    def evaluate_current(self) -> float:
        """Evaluate the current network against the AB2 baseline.

        Returns an ELO estimate.  The actual arena logic will be implemented
        once the MCTS player is integrated.
        """
        log.warning("evaluate_current() not yet implemented — returning initial ELO")
        return self._best_elo

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_selfplay(self) -> dict[str, Any]:
        """Generate self-play games.  Stub until MCTS player is ready."""
        log.warning("_run_selfplay() not yet implemented")
        return {"games": 0}

    def _save(
        self,
        iteration: int,
        elo: float | None,
        is_best: bool = False,
    ) -> None:
        assert self.trainer is not None
        ckpt_dir = self._log_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        path = ckpt_dir / f"iteration_{iteration:06d}.pt"
        extra: dict[str, Any] = {}
        if elo is not None:
            extra["elo_rating"] = elo

        self.trainer.save_checkpoint(str(path), iteration, extra_state=extra)

        if is_best:
            best_path = ckpt_dir / "best.pt"
            self.trainer.save_checkpoint(str(best_path), iteration, extra_state=extra)
            log.info("  Saved best checkpoint to %s", best_path)
