from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from hexzero.config import NetworkConfig


class EdgeConvLayer(nn.Module):
    """Single message-passing layer with edge features.

    For each directed edge (i -> j) with edge feature e_ij:

    1. message:    m_ij = MLP_msg(h_i || h_j || e_ij)
    2. aggregate:  M_j  = Σ_{i ∈ N(j)} m_ij
    3. update:     h_j' = LayerNorm(h_j + MLP_update(h_j || M_j))
    """

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

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Args:
            x:          (N, H)  node embeddings where N = batch_size * num_nodes.
            edge_index: (2, E)  COO source / target indices.
            edge_attr:  (E, H)  projected edge features.

        Returns:
            Updated node embeddings with the same shape as *x*.
        """
        src, dst = edge_index

        messages = self.msg_mlp(
            torch.cat([x[src], x[dst], edge_attr], dim=-1)
        )

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)

        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return self.norm(x + out)


class BoardEncoder(nn.Module):
    """Full GNN encoder for the Catan board graph.

    Pipeline::

        node_features (B,N,F_n) -> node_proj  ─┐
        edge_features (B,E,F_e) -> edge_proj  ─┤
                                                ├──► K × EdgeConvLayer
                                                │
                                                ├──► mean+max pool -> output_proj -> board_embedding  (B, D_out)
                                                └──► node_embeddings  (B, N, H)

    Because the board topology (edge_index) is identical for every game,
    all graphs in a batch share the same connectivity.  We exploit this by
    flattening the batch into a single large graph with per-batch node
    offsets, running message passing once, and reshaping back.
    """

    def __init__(
        self,
        node_feat_dim: int = 12,
        edge_feat_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.Mish(),
        )
        self.edge_proj = nn.Linear(edge_feat_dim, hidden_dim)

        self.layers = nn.ModuleList(
            [EdgeConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim),
        )

        self._init_weights()

        self._cached_ei: Tensor | None = None
        self._cache_key: tuple[int, int, torch.device] = (-1, -1, torch.device("cpu"))

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Batched edge-index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_edge_index(
        edge_index: Tensor, batch_size: int, num_nodes: int
    ) -> Tensor:
        """Replicate *edge_index* for every graph in the batch.

        Each replica's node indices are offset by ``i * num_nodes`` so that
        the flat node tensor can be indexed directly.

        Returns:
            Long tensor of shape ``(2, batch_size * edges_per_graph)``.
        """
        offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
        # (2,1,E) + (1,B,1) -> (2,B,E) -> (2, B*E)
        return (edge_index.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)

    def _batched_edge_index(
        self, edge_index: Tensor, batch_size: int, num_nodes: int
    ) -> Tensor:
        """Return (and cache) the batched edge index."""
        key = (batch_size, num_nodes, edge_index.device)
        if key != self._cache_key or self._cached_ei is None:
            self._cached_ei = self._expand_edge_index(
                edge_index, batch_size, num_nodes
            )
            self._cache_key = key
        return self._cached_ei

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            node_features: ``(B, N, node_feat_dim)``
            edge_index:    ``(2, E_per_graph)`` COO — shared across the batch.
            edge_features: ``(B, E_per_graph, edge_feat_dim)``

        Returns:
            board_embedding: ``(B, output_dim)`` — dense graph-level vector.
            node_embeddings: ``(B, N, hidden_dim)`` — per-node representations.
        """
        B, N, _ = node_features.shape
        E = edge_features.shape[1]

        # Project raw features into the hidden space.
        h = self.node_proj(node_features)  # (B, N, H)
        e = self.edge_proj(edge_features)  # (B, E, H)

        # Flatten batch into one big graph for message passing.
        h_flat = h.reshape(B * N, -1)
        e_flat = e.reshape(B * E, -1)
        ei = self._batched_edge_index(edge_index, B, N)

        for layer in self.layers:
            h_flat = layer(h_flat, ei, e_flat)

        # Reshape back to per-graph tensors.
        node_embeddings = h_flat.reshape(B, N, -1)

        # Global graph-level pooling: concat(mean, max) -> project.
        mean_pool = node_embeddings.mean(dim=1)
        max_pool = node_embeddings.max(dim=1).values
        board_embedding = self.output_proj(
            torch.cat([mean_pool, max_pool], dim=-1)
        )

        return board_embedding, node_embeddings

    # ------------------------------------------------------------------
    # Config convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: NetworkConfig) -> BoardEncoder:
        """Construct from a :class:`~hexzero.config.NetworkConfig`."""
        return cls(
            node_feat_dim=cfg.node_feature_dim,
            edge_feat_dim=cfg.edge_feature_dim,
            hidden_dim=cfg.gnn_hidden_dim,
            output_dim=cfg.gnn_output_dim,
            num_layers=cfg.gnn_layers,
        )
