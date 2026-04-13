from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHeadA(nn.Module):
    """Active-turn policy head producing raw logits over the full action space.

    Action masking is applied externally by :class:`HexaZeroNet`, not here.

    Shape: ``(batch, trunk_channels)`` -> ``(batch, action_space_size)``
    """

    def __init__(
        self,
        trunk_channels: int = 256,
        hidden_dim: int = 256,
        action_space_size: int = 337,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(trunk_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.Mish()
        self.fc_out = nn.Linear(hidden_dim, action_space_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.bn(self.fc1(x))))


class SpatialPolicyHead(nn.Module):
    """Policy head that uses per-node GNN embeddings for spatial actions.

    Settlements/cities are scored by attending each node embedding against the
    trunk context. Roads are scored from endpoint node pairs. Robber tiles are
    scored from pooled adjacent-node embeddings. Non-spatial actions (roll, end
    turn, dev cards, trades) use the global trunk vector.

    Action-space layout (337 total):
        [0,5)     5   singletons        -> global MLP
        [5,59)    54  settlements       -> node attention
        [59,113)  54  cities            -> node attention
        [113,185) 72  roads             -> endpoint pair scoring
        [185,280) 95  robber (19*5)     -> tile-pooled attention
        [280,337) 57  discard/yop/etc   -> global MLP
    """

    def __init__(
        self,
        trunk_channels: int = 256,
        gnn_hidden_dim: int = 128,
        action_space_size: int = 337,
        road_node_pairs: torch.Tensor | None = None,
        tile_node_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        ctx_dim = trunk_channels + gnn_hidden_dim

        # Normalize inputs to spatial scorers (prevents init magnitude explosion)
        self.trunk_norm = nn.LayerNorm(trunk_channels)
        self.node_norm = nn.LayerNorm(gnn_hidden_dim)

        # Global fallback for non-spatial actions
        self.global_fc1 = nn.Linear(trunk_channels, 256)
        self.global_bn = nn.BatchNorm1d(256)
        self.global_act = nn.Mish()
        self.global_fc_out = nn.Linear(256, action_space_size)

        # Node-level scorers (shared MLP applied per node)
        self.settlement_scorer = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.Mish(), nn.Linear(64, 1),
        )
        self.city_scorer = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.Mish(), nn.Linear(64, 1),
        )

        # Road scorer: concat two endpoint embeddings + trunk
        self.road_scorer = nn.Sequential(
            nn.Linear(trunk_channels + 2 * gnn_hidden_dim, 64),
            nn.Mish(),
            nn.Linear(64, 1),
        )

        # Robber scorer: pooled tile embedding + trunk -> 5 steal options
        self.robber_scorer = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.Mish(), nn.Linear(64, 5),
        )

        # Static topology buffers
        if road_node_pairs is not None:
            self.register_buffer("road_pairs", road_node_pairs)
        if tile_node_indices is not None:
            self.register_buffer("tile_nodes", tile_node_indices)

    def forward(
        self, trunk_out: torch.Tensor, node_embeddings: torch.Tensor
    ) -> torch.Tensor:
        B, N, H = node_embeddings.shape

        # ── Normalize inputs ──────────────────────────────────────────
        trunk_n = self.trunk_norm(trunk_out)   # (B, T) normalized
        node_n = self.node_norm(node_embeddings)  # (B, N, H) normalized

        # ── Global base scores ────────────────────────────────────────
        global_logits = self.global_fc_out(
            self.global_act(self.global_bn(self.global_fc1(trunk_n)))
        )  # (B, 337)

        # ── Settlement scores (indices 5-59) ──────────────────────────
        trunk_exp = trunk_n.unsqueeze(1).expand(-1, N, -1)     # (B, 54, T)
        ctx = torch.cat([trunk_exp, node_n], dim=-1)           # (B, 54, T+H)
        sett_scores = self.settlement_scorer(ctx).squeeze(-1)  # (B, 54)

        # ── City scores (indices 59-113) ──────────────────────────────
        city_scores = self.city_scorer(ctx).squeeze(-1)        # (B, 54)

        # ── Road scores (indices 113-185) ─────────────────────────────
        src = node_n[:, self.road_pairs[:, 0], :]              # (B, 72, H)
        dst = node_n[:, self.road_pairs[:, 1], :]              # (B, 72, H)
        trunk_r = trunk_n.unsqueeze(1).expand(-1, 72, -1)      # (B, 72, T)
        road_ctx = torch.cat([trunk_r, src, dst], dim=-1)      # (B, 72, T+2H)
        road_scores = self.road_scorer(road_ctx).squeeze(-1)   # (B, 72)

        # ── Robber scores (indices 185-280) ───────────────────────────
        tile_emb = node_n[:, self.tile_nodes, :].mean(dim=2)   # (B, 19, H)
        trunk_t = trunk_n.unsqueeze(1).expand(-1, 19, -1)      # (B, 19, T)
        tile_ctx = torch.cat([trunk_t, tile_emb], dim=-1)      # (B, 19, T+H)
        robber_scores = self.robber_scorer(tile_ctx).reshape(B, 95)    # (B, 95)

        # ── Concatenate all parts (no in-place ops) ───────────────────
        return torch.cat([
            global_logits[:, :5],       # [0,5)    singletons
            sett_scores,                # [5,59)   settlements
            city_scores,                # [59,113) cities
            road_scores,                # [113,185) roads
            robber_scores,              # [185,280) robber
            global_logits[:, 280:],     # [280,337) discard/yop/monopoly/trade
        ], dim=1)


class PolicyHeadB(nn.Module):
    """Reactive trade-acceptance head producing P(accept) in [0, 1].

    Shape: ``(batch, trunk_channels)`` -> ``(batch, 1)``
    """

    def __init__(self, trunk_channels: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(trunk_channels, 64)
        self.act = nn.Mish()
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc_out(self.act(self.fc1(x))))


class ValueHead(nn.Module):
    """Value head predicting per-seat win logits.

    Index 0 always corresponds to the current player (rotated by the encoder).
    Outputs raw logits; apply softmax externally for probabilities.

    Shape: ``(batch, trunk_channels)`` -> ``(batch, num_players)``
    """

    NUM_PLAYERS: int = 4

    def __init__(
        self,
        trunk_channels: int = 256,
        hidden_dim: int = 256,
        num_players: int = NUM_PLAYERS,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(trunk_channels, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.Mish()
        self.fc_out = nn.Linear(hidden_dim, num_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.act(self.bn(self.fc1(x))))
