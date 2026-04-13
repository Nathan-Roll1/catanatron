"""Batched GPU inference server for MCTS evaluation.

For single-threaded MCTS the ``evaluate`` / ``evaluate_batch`` helpers run a
direct forward pass.  For parallel MCTS, ``start_server`` spins up a
background thread that collects individual requests into GPU-friendly batches.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from hexzero.model.network import HexaZeroNet

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal request type for the async queue
# ---------------------------------------------------------------------------


@dataclass
class _InferenceRequest:
    tensors: dict[str, Tensor]
    future: Future = field(default_factory=Future)


# ---------------------------------------------------------------------------
# BatchInferenceServer
# ---------------------------------------------------------------------------


class BatchInferenceServer:
    """Collects MCTS evaluation requests and processes them in GPU batches."""

    def __init__(
        self,
        network: HexaZeroNet,
        device: str = "cuda",
        max_batch_size: int = 64,
        max_wait_ms: float = 1.0,
    ) -> None:
        self.network = network
        self.device = torch.device(device)
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0

        self.network.to(self.device)
        self.network.eval()

        self._queue: list[_InferenceRequest] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Synchronous helpers (single-threaded MCTS)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self, state_tensors: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Run a single unbatched evaluation.

        Adds a leading batch dimension, runs the forward pass, then squeezes
        the batch dimension back out.  Returns ``policy_probs`` and ``value``.
        """
        batched = {
            k: v.unsqueeze(0).to(self.device, non_blocking=True)
            for k, v in state_tensors.items()
        }
        out = self._forward(batched)
        return {k: v.squeeze(0) for k, v in out.items()}

    @torch.no_grad()
    def evaluate_batch(
        self, batch_tensors: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Evaluate a pre-batched set of positions."""
        on_device = {
            k: v.to(self.device, non_blocking=True)
            for k, v in batch_tensors.items()
        }
        return self._forward(on_device)

    # ------------------------------------------------------------------
    # Async server (parallel MCTS)
    # ------------------------------------------------------------------

    def start_server(self) -> None:
        """Launch background thread that drains the request queue."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._server_loop, daemon=True, name="batch-inference"
        )
        self._thread.start()
        log.info(
            "Batch inference server started (batch=%d, wait=%.1fms)",
            self.max_batch_size,
            self.max_wait_s * 1000,
        )

    def stop_server(self) -> None:
        """Signal the background thread to stop and wait for it."""
        if not self._running:
            return
        self._running = False
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("Batch inference server stopped")

    def submit_async(
        self, state_tensors: dict[str, Tensor]
    ) -> Future:
        """Enqueue a single position for batched evaluation.

        Returns a :class:`~concurrent.futures.Future` whose result will be
        a ``dict[str, Tensor]`` with ``policy_probs`` and ``value``.
        """
        req = _InferenceRequest(tensors=state_tensors)
        with self._lock:
            self._queue.append(req)
            if len(self._queue) >= self.max_batch_size:
                self._event.set()
        return req.future

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _server_loop(self) -> None:
        while self._running:
            self._event.wait(timeout=self.max_wait_s)
            self._event.clear()

            with self._lock:
                batch_reqs = self._queue[: self.max_batch_size]
                self._queue = self._queue[self.max_batch_size :]

            if not batch_reqs:
                continue

            try:
                self._process_batch(batch_reqs)
            except Exception:
                log.exception("Error in batch inference")
                for req in batch_reqs:
                    if not req.future.done():
                        req.future.cancel()

    @torch.no_grad()
    def _process_batch(self, requests: list[_InferenceRequest]) -> None:
        keys = requests[0].tensors.keys()
        collated: dict[str, Tensor] = {}
        for key in keys:
            stacked = torch.stack([r.tensors[key] for r in requests])
            collated[key] = stacked.to(self.device, non_blocking=True)

        results = self._forward(collated)

        for i, req in enumerate(requests):
            single = {k: v[i].cpu() for k, v in results.items()}
            req.future.set_result(single)

    def _forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        self.network.eval()
        with torch.no_grad():
            out = self.network(batch)
        return {
            "policy_probs": out["policy_probs"],
            "value": out["value"],
        }

    def update_network(self, network: HexaZeroNet) -> None:
        """Hot-swap the network weights (e.g. after a training iteration)."""
        self.network = network
        self.network.to(self.device)
        self.network.eval()
