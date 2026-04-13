from __future__ import annotations

import logging
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hexzero.config import NetworkConfig

from .gnn import BoardEncoder
from .heads import PolicyHeadA, SpatialPolicyHead, PolicyHeadB, ValueHead
from .trunk import ResNetTrunk

log = logging.getLogger(__name__)


class HexaZeroNet(nn.Module):
    """Complete HexaZero network.

    Combines a GNN board encoder, a deep residual trunk, and three prediction
    heads (active-turn policy, trade acceptance, win-probability value).

    Forward input — ``dict`` with keys:

    * ``node_features``  : ``(B, 54, node_feat_dim)``
    * ``edge_index``     : ``(2, E_per_graph)`` — shared COO topology
    * ``edge_features``  : ``(B, E_per_graph, edge_feat_dim)``
    * ``flat_features``  : ``(B, flat_feat_dim)``
    * ``action_mask``    : ``(B, action_space_size)`` — optional boolean mask

    Forward output — ``dict`` with keys:

    * ``policy_logits``       : ``(B, action_space_size)`` — masked raw logits
    * ``policy_probs``        : ``(B, action_space_size)`` — after masked softmax
    * ``trade_accept_prob``   : ``(B, 1)``
    * ``value``               : ``(B, 4)`` — per-seat win probabilities
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        self.board_encoder = BoardEncoder.from_config(config)

        trunk_input_dim = config.gnn_output_dim + config.flat_feature_dim
        self.trunk = ResNetTrunk(
            input_dim=trunk_input_dim,
            trunk_channels=config.trunk_channels,
            num_blocks=config.trunk_blocks,
            activation=config.trunk_activation,
        )

        road_pairs, tile_nodes = self._compute_topology()
        self.policy_head = SpatialPolicyHead(
            trunk_channels=config.trunk_channels,
            gnn_hidden_dim=config.gnn_hidden_dim,
            action_space_size=config.action_space_size,
            road_node_pairs=road_pairs,
            tile_node_indices=tile_nodes,
        )
        self.trade_head = PolicyHeadB(trunk_channels=config.trunk_channels)
        self.value_head = ValueHead(
            trunk_channels=config.trunk_channels,
            hidden_dim=config.value_head_hidden,
        )

        self.gradient_checkpointing = False

        self._init_weights()
        log.info("HexaZeroNet: %s trainable parameters", f"{self.num_parameters:,}")

    # ------------------------------------------------------------------
    # Board topology (static, computed once)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_topology() -> tuple[torch.Tensor, torch.Tensor]:
        """Build road-node and tile-node mappings in compact (54-node) space."""
        from hexzero.encoder.action_encoder import ActionEncoder
        from hexzero.game.interface import CatanGame

        ae = ActionEncoder()
        g = CatanGame(seed=0)
        g.reset()
        se = g.make_state_encoder()

        full_to_compact = {}
        for ci, fi in enumerate(se._land):
            full_to_compact[int(fi)] = ci

        road_pairs = torch.zeros(72, 2, dtype=torch.long)
        for i in range(72):
            a, b = ae._idx_to_edge[i]
            road_pairs[i, 0] = full_to_compact[int(a)]
            road_pairs[i, 1] = full_to_compact[int(b)]

        tile_nodes = torch.from_numpy(se._ltiles.copy()).long()  # (19, 6)

        return road_pairs, tile_nodes

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for head in (self.trade_head, self.value_head):
            head.fc_out.weight.data.mul_(0.01)
        self.policy_head.global_fc_out.weight.data.mul_(0.01)
        for scorer in (self.policy_head.settlement_scorer,
                       self.policy_head.city_scorer,
                       self.policy_head.road_scorer,
                       self.policy_head.robber_scorer):
            scorer[-1].weight.data.mul_(0.01)

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        board_emb, node_emb = self.board_encoder(
            batch["node_features"],
            batch["edge_index"],
            batch["edge_features"],
        )

        combined = torch.cat([board_emb, batch["flat_features"]], dim=-1)

        if self.gradient_checkpointing and self.training:
            trunk_out: torch.Tensor = torch.utils.checkpoint.checkpoint(
                self.trunk, combined, use_reentrant=False
            )
        else:
            trunk_out = self.trunk(combined)

        raw_logits = self.policy_head(trunk_out, node_emb)
        trade_accept_prob = self.trade_head(trunk_out)
        value = self.value_head(trunk_out)

        mask = batch.get("action_mask")
        if mask is not None:
            masked_logits = raw_logits.masked_fill(~mask.bool(), -1e9)
        else:
            masked_logits = raw_logits

        policy_probs = F.softmax(masked_logits, dim=-1)
        policy_probs = torch.nan_to_num(policy_probs, nan=0.0)

        return {
            "raw_policy_logits": raw_logits,
            "policy_logits": masked_logits,
            "policy_probs": policy_probs,
            "trade_accept_prob": trade_accept_prob,
            "value": value,
        }

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        """Inference-mode forward pass returning NumPy arrays on CPU."""
        was_training = self.training
        self.eval()
        out = self.forward(batch)
        if was_training:
            self.train()
        return {k: v.cpu().numpy() for k, v in out.items()}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> HexaZeroNet:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = NetworkConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model
