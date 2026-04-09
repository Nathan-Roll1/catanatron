"""Compact dual-head network for human move prediction (~200k params).

Same architectural pattern as HexaZeroNet (GNN encoder + ResNet trunk +
spatial policy head + value head) but with all dimensions shrunk to fit
a ~200k parameter budget.  Designed for fast training on Apple Silicon
(MPS) with fp16 autocast.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================

@dataclass
class SmallNetworkConfig:
    # GNN encoder
    gnn_layers: int = 3
    gnn_hidden_dim: int = 48
    gnn_output_dim: int = 80

    # Trunk
    trunk_blocks: int = 4
    trunk_channels: int = 80
    trunk_activation: str = "mish"

    # Heads
    policy_hidden_dim: int = 80
    scorer_hidden_dim: int = 32
    value_head_hidden: int = 64

    # Input dims (fixed by state encoder)
    node_feature_dim: int = 18
    edge_feature_dim: int = 5
    flat_feature_dim: int = 115
    action_space_size: int = 397  # 337 base + 60 trade offers (1:1, 1:2, 2:1)


# ======================================================================
# GNN layers
# ======================================================================

class EdgeConvLayer(nn.Module):
    """Message-passing layer matching hexzero's EdgeConvLayer interface."""

    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        messages = self.msg_mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
        # index_add_ requires matching dtypes; force fp32 for scatter
        agg = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        agg.index_add_(0, dst, messages.to(agg.dtype))
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + out)


class SmallBoardEncoder(nn.Module):
    """Compact GNN board encoder."""

    def __init__(self, cfg: SmallNetworkConfig) -> None:
        super().__init__()
        H = cfg.gnn_hidden_dim
        self.node_proj = nn.Sequential(nn.Linear(cfg.node_feature_dim, H), nn.Mish())
        self.edge_proj = nn.Linear(cfg.edge_feature_dim, H)
        self.layers = nn.ModuleList([EdgeConvLayer(H, H) for _ in range(cfg.gnn_layers)])
        self.output_proj = nn.Sequential(
            nn.Linear(2 * H, cfg.gnn_output_dim),
            nn.Mish(),
            nn.Linear(cfg.gnn_output_dim, cfg.gnn_output_dim),
        )
        self._cached_ei: torch.Tensor | None = None
        self._cache_key: tuple = (-1, -1, torch.device("cpu"))

    def _batched_edge_index(self, ei: torch.Tensor, B: int, N: int) -> torch.Tensor:
        key = (B, N, ei.device)
        if key != self._cache_key or self._cached_ei is None:
            offsets = torch.arange(B, device=ei.device) * N
            self._cached_ei = (ei.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)
            self._cache_key = key
        return self._cached_ei

    @torch.autocast("mps", enabled=False)
    @torch.autocast("cuda", enabled=False)
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GNN runs in fp32 — scatter ops (index_add_) need matching dtypes."""
        B, N, _ = node_features.shape
        E = edge_features.shape[1]

        nf = node_features.float()
        ef = edge_features.float()

        h = self.node_proj(nf)
        e = self.edge_proj(ef)

        h_flat = h.reshape(B * N, -1)
        e_flat = e.reshape(B * E, -1)
        ei = self._batched_edge_index(edge_index, B, N)

        for layer in self.layers:
            h_flat = layer(h_flat, ei, e_flat)

        node_emb = h_flat.reshape(B, N, -1)
        mean_pool = node_emb.mean(dim=1)
        max_pool = node_emb.max(dim=1).values
        board_emb = self.output_proj(torch.cat([mean_pool, max_pool], dim=-1))
        return board_emb, node_emb


# ======================================================================
# Trunk
# ======================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn2(self.fc2(self.act(self.bn1(self.fc1(x))))) + x)


class SmallTrunk(nn.Module):
    def __init__(self, input_dim: int, channels: int, num_blocks: int) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, channels), nn.BatchNorm1d(channels), nn.Mish(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x


# ======================================================================
# Heads
# ======================================================================

class SmallSpatialPolicyHead(nn.Module):
    """Compact spatial policy head with configurable hidden dimensions."""

    def __init__(self, cfg: SmallNetworkConfig,
                 road_pairs: torch.Tensor, tile_nodes: torch.Tensor) -> None:
        super().__init__()
        T = cfg.trunk_channels
        H = cfg.gnn_hidden_dim
        PH = cfg.policy_hidden_dim
        SH = cfg.scorer_hidden_dim

        self.trunk_norm = nn.LayerNorm(T)
        self.node_norm = nn.LayerNorm(H)

        self.global_fc = nn.Sequential(
            nn.Linear(T, PH), nn.BatchNorm1d(PH), nn.Mish(),
            nn.Linear(PH, cfg.action_space_size),
        )

        ctx_dim = T + H
        self.settlement_scorer = nn.Sequential(nn.Linear(ctx_dim, SH), nn.Mish(), nn.Linear(SH, 1))
        self.city_scorer = nn.Sequential(nn.Linear(ctx_dim, SH), nn.Mish(), nn.Linear(SH, 1))
        self.road_scorer = nn.Sequential(nn.Linear(T + 2 * H, SH), nn.Mish(), nn.Linear(SH, 1))
        self.robber_scorer = nn.Sequential(nn.Linear(ctx_dim, SH), nn.Mish(), nn.Linear(SH, 5))

        self.register_buffer("road_pairs", road_pairs)
        self.register_buffer("tile_nodes", tile_nodes)

    def forward(self, trunk_out: torch.Tensor,
                node_emb: torch.Tensor) -> torch.Tensor:
        B, N, H = node_emb.shape
        tn = self.trunk_norm(trunk_out)
        nn_ = self.node_norm(node_emb)

        global_logits = self.global_fc(tn)

        ctx = torch.cat([tn.unsqueeze(1).expand(-1, N, -1), nn_], dim=-1)
        sett = self.settlement_scorer(ctx).squeeze(-1)
        city = self.city_scorer(ctx).squeeze(-1)

        src = nn_[:, self.road_pairs[:, 0], :]
        dst = nn_[:, self.road_pairs[:, 1], :]
        road_ctx = torch.cat([tn.unsqueeze(1).expand(-1, 72, -1), src, dst], dim=-1)
        road = self.road_scorer(road_ctx).squeeze(-1)

        tile_emb = nn_[:, self.tile_nodes, :].mean(dim=2)
        tile_ctx = torch.cat([tn.unsqueeze(1).expand(-1, 19, -1), tile_emb], dim=-1)
        robber = self.robber_scorer(tile_ctx).reshape(B, 95)

        return torch.cat([
            global_logits[:, :5], sett, city, road, robber, global_logits[:, 280:],
        ], dim=1)  # total = 5 + 54 + 54 + 72 + 95 + (action_space-280) = action_space


class SmallValueHead(nn.Module):
    def __init__(self, trunk_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(trunk_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.Mish()
        self.fc_out = nn.Linear(hidden_dim, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.bn(self.fc1(x))))


# ======================================================================
# Full network
# ======================================================================

class HumanBotNet(nn.Module):
    """Compact dual-head Catan network (~200k params).

    Same forward interface as HexaZeroNet:
      Input:  dict with node_features, edge_index, edge_features, flat_features, action_mask
      Output: dict with policy_logits, policy_probs, value
    """

    def __init__(self, config: SmallNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or SmallNetworkConfig()
        cfg = self.config

        self.board_encoder = SmallBoardEncoder(cfg)

        trunk_input = cfg.gnn_output_dim + cfg.flat_feature_dim
        self.trunk = SmallTrunk(trunk_input, cfg.trunk_channels, cfg.trunk_blocks)

        road_pairs, tile_nodes = self._compute_topology()
        self.policy_head = SmallSpatialPolicyHead(cfg, road_pairs, tile_nodes)
        self.value_head = SmallValueHead(cfg.trunk_channels, cfg.value_head_hidden)

        self._init_weights()
        log.info("HumanBotNet: %s params", f"{self.num_parameters:,}")

    @staticmethod
    def _compute_topology() -> tuple[torch.Tensor, torch.Tensor]:
        from hexzero.encoder.action_encoder import ActionEncoder
        from hexzero.game.interface import CatanGame

        ae = ActionEncoder()
        g = CatanGame(seed=0)
        g.reset()
        se = g.make_state_encoder()

        full_to_compact = {int(fi): ci for ci, fi in enumerate(se._land)}

        road_pairs = torch.zeros(72, 2, dtype=torch.long)
        for i in range(72):
            a, b = ae._idx_to_edge[i]
            road_pairs[i, 0] = full_to_compact[int(a)]
            road_pairs[i, 1] = full_to_compact[int(b)]

        tile_nodes = torch.from_numpy(se._ltiles.copy()).long()
        return road_pairs, tile_nodes

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.value_head.fc_out.weight.data.mul_(0.01)
        for scorer in (self.policy_head.settlement_scorer,
                       self.policy_head.city_scorer,
                       self.policy_head.road_scorer,
                       self.policy_head.robber_scorer):
            scorer[-1].weight.data.mul_(0.01)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        board_emb, node_emb = self.board_encoder(
            batch["node_features"], batch["edge_index"], batch["edge_features"],
        )
        combined = torch.cat([board_emb, batch["flat_features"]], dim=-1)
        trunk_out = self.trunk(combined)

        raw_logits = self.policy_head(trunk_out, node_emb)
        value = self.value_head(trunk_out)

        mask = batch.get("action_mask")
        if mask is not None:
            fill_val = -6e4 if raw_logits.dtype == torch.float16 else -1e9
            masked_logits = raw_logits.masked_fill(~mask.bool(), fill_val)
        else:
            masked_logits = raw_logits

        policy_probs = F.softmax(masked_logits, dim=-1)
        policy_probs = torch.nan_to_num(policy_probs, nan=0.0)

        return {
            "raw_policy_logits": raw_logits,
            "policy_logits": masked_logits,
            "policy_probs": policy_probs,
            "value": value,
        }

    @torch.inference_mode()
    def predict(self, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        was_training = self.training
        self.eval()
        out = self.forward(batch)
        if was_training:
            self.train()
        return {k: v.cpu().numpy() for k, v in out.items()}

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        torch.save({
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
            "metadata": metadata or {},
            "model_type": "HumanBotNet",
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> HumanBotNet:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = SmallNetworkConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model
