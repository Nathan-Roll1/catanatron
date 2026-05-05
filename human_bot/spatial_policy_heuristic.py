from __future__ import annotations

import torch
import torch.nn as nn

AD = 337
FD = 115
NF = 18
EF = 5


class SpatialPolicyHeuristic(nn.Module):
    """Cheap policy scorer with raw spatial features and no GNN.

    It scores every action slot directly:
      - all actions get a flat-feature linear score + action bias
      - settlement/city slots get an extra node-feature linear score
      - road slots get an extra raw edge-feature score
      - robber slots get an extra tile-node-average score
    """

    def __init__(self, tile_nodes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("tile_nodes", tile_nodes.long())
        self.flat = nn.Linear(FD, AD)
        self.settlement = nn.Linear(NF, 1)
        self.city = nn.Linear(NF, 1)
        self.road = nn.Linear(2 * EF, 1)
        self.robber = nn.Linear(NF, 5)

    def forward(self, nf: torch.Tensor, ef: torch.Tensor,
                ff: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.flat(ff)

        logits[:, 5:59] = logits[:, 5:59] + self.settlement(nf).squeeze(-1)
        logits[:, 59:113] = logits[:, 59:113] + self.city(nf).squeeze(-1)

        # StateEncoder orders directed edges as [a->b, b->a] for each
        # undirected edge, matching ActionEncoder's sorted edge order.
        e0 = ef[:, 0::2, :]
        e1 = ef[:, 1::2, :]
        road_feat = torch.cat([e0, e1], dim=-1)
        logits[:, 113:185] = logits[:, 113:185] + self.road(road_feat).squeeze(-1)

        tile_feat = nf[:, self.tile_nodes, :].mean(dim=2)
        logits[:, 185:280] = logits[:, 185:280] + self.robber(tile_feat).reshape(-1, 95)

        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits


class SpatialPolicyHeuristicMLP(nn.Module):
    """Nonlinear cheap scorer, still without graph message passing."""

    def __init__(self, tile_nodes: torch.Tensor, hidden: int = 128) -> None:
        super().__init__()
        self.register_buffer("tile_nodes", tile_nodes.long())
        self.flat = nn.Sequential(
            nn.Linear(FD, hidden),
            nn.ReLU(),
            nn.Linear(hidden, AD),
        )
        self.settlement = nn.Sequential(nn.Linear(NF, 48), nn.ReLU(), nn.Linear(48, 1))
        self.city = nn.Sequential(nn.Linear(NF, 48), nn.ReLU(), nn.Linear(48, 1))
        self.road = nn.Sequential(nn.Linear(2 * EF, 32), nn.ReLU(), nn.Linear(32, 1))
        self.robber = nn.Sequential(nn.Linear(NF, 48), nn.ReLU(), nn.Linear(48, 5))

    def forward(self, nf: torch.Tensor, ef: torch.Tensor,
                ff: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.flat(ff)
        logits[:, 5:59] = logits[:, 5:59] + self.settlement(nf).squeeze(-1)
        logits[:, 59:113] = logits[:, 59:113] + self.city(nf).squeeze(-1)

        road_feat = torch.cat([ef[:, 0::2, :], ef[:, 1::2, :]], dim=-1)
        logits[:, 113:185] = logits[:, 113:185] + self.road(road_feat).squeeze(-1)

        tile_feat = nf[:, self.tile_nodes, :].mean(dim=2)
        logits[:, 185:280] = logits[:, 185:280] + self.robber(tile_feat).reshape(-1, 95)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits
