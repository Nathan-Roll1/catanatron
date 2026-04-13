"""Training loop for HexaZero.

Handles optimizer setup, learning-rate scheduling, mixed-precision training,
gradient clipping, checkpointing, and metric tracking.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler

from hexzero.config import TrainingConfig
from hexzero.training.loss import HexaZeroLoss

if TYPE_CHECKING:
    from hexzero.model.network import HexaZeroNet

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Learning-rate scheduler: cosine annealing with linear warmup
# ---------------------------------------------------------------------------


def _warmup_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


class _WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """LambdaLR wrapper that implements cosine decay with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self._warmup = warmup_steps
        self._total = total_steps
        base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._base_lrs = base_lrs

        def lr_lambda(step: int) -> float:
            return _warmup_cosine_lr(step, warmup_steps, total_steps, 1.0)

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Manages the full training lifecycle for HexaZero."""

    def __init__(
        self,
        network: HexaZeroNet,
        config: TrainingConfig,
        device: str = "cuda",
        total_steps: int | None = None,
    ) -> None:
        self.network = network
        self.config = config
        self.device = torch.device(device)
        self.network.to(self.device)

        self.criterion = HexaZeroLoss(
            value_weight=config.value_loss_weight,
            policy_weight=config.policy_loss_weight,
        )

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self._total_steps = total_steps or 100_000
        self.scheduler = _WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.lr_warmup_steps,
            total_steps=self._total_steps,
        )

        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))
        self._global_step: int = 0
        self._max_grad_norm: float = 1.0

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Run one gradient-update step.  Returns scalar metrics."""
        self.network.train()

        inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        targets = {
            "policy_targets": inputs.pop("policy_targets"),
            "value_targets": inputs.pop("value_targets"),
            "action_masks": inputs["action_masks"],
        }

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=self.device.type,
            enabled=(self.device.type == "cuda"),
        ):
            predictions = self.network(inputs)
            losses = self.criterion(predictions, targets)

        total_loss: Tensor = losses["total_loss"]
        self.scaler.scale(total_loss).backward()

        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.network.parameters(), self._max_grad_norm
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self._global_step += 1

        return {
            "total_loss": losses["total_loss"].item(),
            "value_loss": losses["value_loss"].item(),
            "policy_loss": losses["policy_loss"].item(),
            "value_accuracy": losses["value_accuracy"].item(),
            "policy_entropy": losses["policy_entropy"].item(),
            "learning_rate": self.current_lr,
            "gradient_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
        }

    # ------------------------------------------------------------------
    # Epoch / iteration helpers
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        replay_buffer: Any,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """Sample random batches from *replay_buffer* for one full pass.

        ``replay_buffer`` must expose ``__len__`` and ``sample(batch_size)``
        returning a dict of tensors matching the expected batch format.
        """
        bs = batch_size or self.config.batch_size
        num_batches = max(len(replay_buffer) // bs, 1)
        accum: dict[str, float] = {}

        for _ in range(num_batches):
            batch = replay_buffer.sample(bs)
            metrics = self.train_step(batch)
            for k, v in metrics.items():
                accum[k] = accum.get(k, 0.0) + v

        return {k: v / num_batches for k, v in accum.items()}

    def train_iteration(
        self,
        replay_buffer: Any,
        num_epochs: int | None = None,
    ) -> dict[str, float]:
        """Full training iteration (multiple epochs) after a self-play phase."""
        epochs = num_epochs or self.config.num_epochs_per_iteration
        accum: dict[str, float] = {}

        t0 = time.monotonic()
        for epoch_idx in range(epochs):
            epoch_metrics = self.train_epoch(replay_buffer)
            for k, v in epoch_metrics.items():
                accum[k] = accum.get(k, 0.0) + v

            log.info(
                "epoch %d/%d  loss=%.4f  vloss=%.4f  ploss=%.4f  vacc=%.3f  lr=%.2e",
                epoch_idx + 1,
                epochs,
                epoch_metrics["total_loss"],
                epoch_metrics["value_loss"],
                epoch_metrics["policy_loss"],
                epoch_metrics["value_accuracy"],
                epoch_metrics["learning_rate"],
            )

        elapsed = time.monotonic() - t0
        avg = {k: v / epochs for k, v in accum.items()}
        avg["epoch_seconds"] = elapsed / epochs
        avg["total_seconds"] = elapsed
        avg["global_step"] = float(self._global_step)
        return avg

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str,
        iteration: int,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        """Persist all training state to *path*."""
        ckpt: dict[str, Any] = {
            "iteration": iteration,
            "global_step": self._global_step,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "config": asdict(self.config),
        }
        if extra_state:
            ckpt.update(extra_state)

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, out)
        log.info("Saved checkpoint to %s (iteration %d)", path, iteration)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        network: HexaZeroNet,
        config: TrainingConfig,
        device: str = "cuda",
    ) -> tuple[Trainer, int]:
        """Restore a *Trainer* from a checkpoint.  Returns ``(trainer, iteration)``."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["model_state_dict"])

        trainer = cls(network, config, device=device)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        trainer._global_step = ckpt.get("global_step", 0)

        iteration: int = ckpt["iteration"]
        log.info("Loaded checkpoint from %s (iteration %d)", path, iteration)
        return trainer, iteration

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def global_step(self) -> int:
        return self._global_step
