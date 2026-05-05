"""State encoder for HexaZero: converts C engine state to GNN-ready tensors."""

from __future__ import annotations

import numpy as np
import torch
from typing import Protocol

# ---------------------------------------------------------------------------
# Constants matching C engine (catan_types.h, board.h)
# ---------------------------------------------------------------------------

TOTAL_NODES = 96
NUM_LAND_TILES = 19
NUM_RESOURCES = 5
NUM_PLAYERS = 4
NUM_PROMPTS = 7  # ActionPrompt enum: 0–6

# Player-state field indices (must match C enum in catan_types.h)
_VP = 0
_ROADS_AVAIL = 1
_SETT_AVAIL = 2
_CITIES_AVAIL = 3
_HAS_ROAD = 4
_HAS_ARMY = 5
_HAS_ROLLED = 6
_HAS_PLAYED_DEV = 7
_LONGEST_ROAD = 9
_RES = 14       # 14..18  wood, brick, sheep, wheat, ore
_DEV = 19       # 19..23  knight, yop, monopoly, road_building, vp
_PLAYED = 24    # 24..28  played dev cards (same order)

_FEAT_PER_PLAYER = 24
_FEAT_BANK = 6
_FEAT_PHASE = 13  # 7 prompt one-hot + 5 binary flags + 1 num_turns

# Dice probability for each sum 2..12 (indices 0–1 are zero).
_DICE_PROB = np.zeros(13, dtype=np.float32)
for _a in range(1, 7):
    for _b in range(1, 7):
        _DICE_PROB[_a + _b] += 1.0 / 36.0


# ---------------------------------------------------------------------------
# Structural type for the game state exposed by the C engine
# ---------------------------------------------------------------------------

class StateView(Protocol):
    """Duck-typed interface for game state objects from the C engine."""

    current_player: int
    num_turns: int
    is_initial_build_phase: bool
    current_prompt: int
    num_players: int
    player_state: np.ndarray      # (4, 29)
    buildings: np.ndarray          # (96,) int8
    road_owners: np.ndarray        # (96, 3) int8
    color_to_index: np.ndarray     # (4,) int32 — Color enum -> seat index
    robber_coord: tuple[int, int, int]
    resource_bank: np.ndarray      # (5,)
    dev_deck_size: int
    tile_resources: np.ndarray     # (19,) int8
    tile_numbers: np.ndarray       # (19,) int8
    tile_nodes: np.ndarray         # (19, 6) int32
    is_discarding: bool
    is_road_building: bool
    is_moving_knight: bool
    is_resolving_trade: bool


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """Convert a Catan :class:`StateView` into tensors for a GNN + MLP network.

    Create **one instance per game**.  The graph topology (``edge_index``) is
    identical for all games on the same map and is pre-computed once.  Port
    assignments are game-specific and cached at construction time.

    Returned dict keys
    ------------------
    ``node_features``   (N, 18)   per-node features for the GNN
    ``edge_index``      (2, E)    COO adjacency (bidirectional)
    ``edge_features``   (E, 5)    per-directed-edge road ownership
    ``flat_features``   (F,)      global state vector  (F = 115)

    Node feature layout (18 dims)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    0‥4   building one-hot  [empty, own_sett, own_city, foe_sett, foe_city]
    5‥9   resource production  (dice-probability–weighted, per resource)
    10‥16 port one-hot  [none, wood, brick, sheep, wheat, ore, generic]
    17    robber adjacency  (1 if any adjacent tile has the robber)

    Edge feature layout (5 dims)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    0     no road
    1     own road
    2‥4   enemy roads  (player offsets 1, 2, 3 relative to current player)

    Flat feature layout (115 dims)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    0‥95   4 × 24 per-player features  (current player always slot 0)
    96‥101 bank  (5 resources / 19, dev_deck / 25)
    102‥114 phase  (prompt one-hot[7], 5 flags, num_turns / 1000)
    """

    NODE_FEATURE_DIM: int = 18
    EDGE_FEATURE_DIM: int = 5
    FLAT_FEATURE_DIM: int = (
        _FEAT_PER_PLAYER * NUM_PLAYERS + _FEAT_BANK + _FEAT_PHASE
    )  # 115

    def __init__(
        self,
        tile_nodes: np.ndarray,
        tile_coords: np.ndarray,
        static_adj: np.ndarray,
        adj_count: np.ndarray,
        port_map: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        tile_nodes : ndarray, shape (19, 6)
            Node IDs for each land tile.  Static for a given map topology.
        tile_coords : ndarray, shape (19, 3)
            Cube coordinates ``(x, y, z)`` per land tile (for robber matching).
        static_adj : ndarray, shape (96, 3)
            ``STATIC_ADJ`` exported from the C engine.
        adj_count : ndarray, shape (96,)
            ``STATIC_ADJ_COUNT`` – valid neighbour count per node.
        port_map : ndarray, shape (96,), optional
            Port resource type per *global* node ID.
            ``-1`` = no port, ``0–4`` = specific resource, ``5`` = generic 3:1.
        """
        tn = np.asarray(tile_nodes, dtype=np.int32)
        tc = np.asarray(tile_coords, dtype=np.int32)
        sa = np.asarray(static_adj, dtype=np.int32)

        # -- land nodes & global ↔ local mapping --------------------------
        land = np.unique(tn.ravel())
        self.num_nodes: int = int(len(land))
        self._land: np.ndarray = land                    # (N,) global IDs

        g2l = np.full(TOTAL_NODES, -1, dtype=np.int32)
        g2l[land] = np.arange(self.num_nodes, dtype=np.int32)
        self._g2l: np.ndarray = g2l

        # tile layout in local-node space
        self._ltiles: np.ndarray = g2l[tn]               # (19, 6)
        self._tcoords: np.ndarray = tc                    # (19, 3)

        # -- edge_index (COO, bidirectional) -------------------------------
        # Two nodes are adjacent iff they are consecutive in some tile's
        # 6-node ring (with wrap-around: position 5 ↔ position 0).
        edge_set: set[tuple[int, int]] = set()
        for t in range(NUM_LAND_TILES):
            for i in range(6):
                a, b = int(tn[t, i]), int(tn[t, (i + 1) % 6])
                edge_set.add((min(a, b), max(a, b)))

        src_list: list[int] = []
        dst_list: list[int] = []
        for ga, gb in sorted(edge_set):
            la, lb = int(g2l[ga]), int(g2l[gb])
            src_list += [la, lb]
            dst_list += [lb, la]

        self._edge_index = torch.tensor(
            [src_list, dst_list], dtype=torch.long
        )
        self.num_edges: int = self._edge_index.shape[1]
        self.num_undirected_edges: int = len(edge_set)

        # -- road-owner lookup table ---------------------------------------
        # For directed edge k we need road_owners[global_src_k, adj_idx_k].
        esrc = land[np.asarray(src_list)]
        edst = land[np.asarray(dst_list)]
        adj_at_src = sa[esrc]                             # (E, 3)
        match = adj_at_src == edst[:, None]                # (E, 3)
        if not match.any(axis=1).all():
            bad = np.where(~match.any(axis=1))[0]
            raise ValueError(
                f"Land edges at directed-edge indices {bad.tolist()} "
                f"have no matching entry in static_adj."
            )
        self._road_src: np.ndarray = esrc.astype(np.intp)
        self._road_adj: np.ndarray = np.argmax(match, axis=1).astype(np.intp)

        # -- port one-hot (cached, game-specific) --------------------------
        port_oh = np.zeros((self.num_nodes, 7), dtype=np.float32)
        if port_map is not None:
            pm = np.asarray(port_map, dtype=np.int8)[land]
            has = pm >= 0
            port_oh[~has, 0] = 1.0
            rows = np.where(has)[0]
            port_oh[rows, pm[has].astype(np.intp) + 1] = 1.0
        else:
            port_oh[:, 0] = 1.0
        self._port_oh = torch.from_numpy(port_oh)
        self._port_oh_np = port_oh

    # ------------------------------------------------------------------
    # Single-state encoding
    # ------------------------------------------------------------------

    def encode(self, state: StateView) -> dict[str, torch.Tensor]:
        """Encode one game state into CPU float32 tensors.

        The returned ``edge_index`` is a **shared** reference that is the
        same object for every call.  Do not modify it in-place.
        """
        nf_np, ef_np, flat_np = self._encode_numpy(state)
        return {
            "node_features": torch.from_numpy(nf_np),
            "edge_index": self._edge_index,
            "edge_features": torch.from_numpy(ef_np),
            "flat_features": torch.from_numpy(flat_np),
        }

    def encode_into(self, state: StateView,
                    nf_out: np.ndarray, ef_out: np.ndarray,
                    flat_out: np.ndarray) -> None:
        """Encode directly into pre-allocated numpy buffers (zero-alloc)."""
        self._encode_numpy_into(state, nf_out, ef_out, flat_out)

    def _encode_numpy(self, state: StateView):
        """Encode into fresh numpy arrays."""
        N, E = self.num_nodes, self.num_edges
        nf = np.zeros((N, self.NODE_FEATURE_DIM), dtype=np.float32)
        ef = np.zeros((E, self.EDGE_FEATURE_DIM), dtype=np.float32)
        flat = np.zeros(self.FLAT_FEATURE_DIM, dtype=np.float32)
        self._encode_numpy_into(state, nf, ef, flat)
        return nf, ef, flat

    def _encode_numpy_into(self, state: StateView,
                           nf: np.ndarray, ef: np.ndarray,
                           flat: np.ndarray) -> None:
        """Core encoder: writes into pre-allocated numpy arrays."""
        cp = state.current_player
        N = self.num_nodes

        nf[:] = 0.0
        ef[:] = 0.0
        flat[:] = 0.0

        # ── node features (N, 18) ────────────────────────────────────
        bld = state.buildings[self._land]
        occ = bld >= 0
        col_raw = np.where(occ, bld >> 2, -1)  # Color enum (0-3)
        # Map Color -> seat index using color_to_index
        c2i = state.color_to_index
        col = np.where(occ, c2i[col_raw.clip(0, 3)], -1)  # seat index
        typ = np.where(occ, bld & 0x3, -1)
        own = occ & (col == cp)
        foe = occ & (col != cp)
        s_mask = typ == 0
        c_mask = typ == 1

        nf[:, 0] = ~occ
        nf[:, 1] = own & s_mask
        nf[:, 2] = own & c_mask
        nf[:, 3] = foe & s_mask
        nf[:, 4] = foe & c_mask

        tres = np.asarray(state.tile_resources)
        tnum = np.asarray(state.tile_numbers)
        valid = (tres >= 0) & (tnum > 0)

        if valid.any():
            for ti in np.where(valid)[0]:
                prob = _DICE_PROB[tnum[ti]]
                r = int(tres[ti])
                for ni in self._ltiles[ti]:
                    nf[ni, 5 + r] += prob

        nf[:, 10:17] = self._port_oh_np

        rc = np.asarray(state.robber_coord, dtype=np.int32)
        rmatch = np.all(self._tcoords == rc, axis=1)
        rt = np.where(rmatch)[0]
        if len(rt):
            nf[self._ltiles[rt[0]], 17] = 1.0

        # Determine real player count (same logic used by flat-feature block
        # below) so the edge "enemy road" channels match the flat convention
        # in mixed 2/3/4-player games.
        n_real = int(getattr(state, "num_players", NUM_PLAYERS))
        if n_real <= 0 or n_real > NUM_PLAYERS:
            n_real = NUM_PLAYERS

        # ── edge features (E, 5) ─────────────────────────────────────
        rc_raw = state.road_owners[self._road_src, self._road_adj]  # Color enum
        # Map Color -> seat index for road owners
        rc_arr = np.where(rc_raw >= 0, c2i[rc_raw.clip(0, 3)], -1)  # seat index
        ef[:, 0] = rc_arr < 0
        ef[:, 1] = rc_arr == cp
        # Enemy road channels rotate by `n_real` (not 4) so that in 2p/3p
        # games, the single/two real opponents land in a stable channel
        # relative to turn order instead of cycling through phantom seats.
        # Channels beyond the real opponent count remain all zeros (same
        # convention as flat-feature unused-seat padding).
        for k in range(1, NUM_PLAYERS):
            if k < n_real:
                ef[:, 1 + k] = rc_arr == (cp + k) % n_real
            else:
                ef[:, 1 + k] = 0.0

        # ── flat features (115) ──────────────────────────────────────
        # Rotate so the current player is always slot 0, then the other
        # real seats follow in turn order. Unused trailing slots stay 0.
        # Works for 2/3/4-player games — 2p has slots {0, 1} populated,
        # 3p has {0, 1, 2}, 4p has all four.
        rot_real = [(cp + i) % n_real for i in range(n_real)]
        ps = state.player_state[rot_real].astype(np.float32)

        o = 0
        for p in range(NUM_PLAYERS):
            if p < n_real:
                flat[o] = ps[p, _VP] / 10.0
                flat[o+1:o+6] = ps[p, _RES:_RES+5] / 19.0
                flat[o+6:o+11] = ps[p, _DEV:_DEV+5] / 14.0
                flat[o+11:o+16] = ps[p, _PLAYED:_PLAYED+5]
                flat[o+16] = ps[p, _HAS_ROAD]
                flat[o+17] = ps[p, _HAS_ARMY]
                flat[o+18] = ps[p, _HAS_ROLLED]
                flat[o+19] = ps[p, _HAS_PLAYED_DEV]
                flat[o+20] = ps[p, _ROADS_AVAIL] / 15.0
                flat[o+21] = ps[p, _SETT_AVAIL] / 5.0
                flat[o+22] = ps[p, _CITIES_AVAIL] / 4.0
                flat[o+23] = ps[p, _LONGEST_ROAD] / 15.0
            # else: leave zeros for unused trailing seats
            o += _FEAT_PER_PLAYER

        o1 = NUM_PLAYERS * _FEAT_PER_PLAYER
        flat[o1:o1+5] = np.asarray(state.resource_bank, dtype=np.float32) / 19.0
        flat[o1+5] = state.dev_deck_size / 25.0

        o2 = o1 + _FEAT_BANK
        prompt = int(state.current_prompt)
        if 0 <= prompt < NUM_PROMPTS:
            flat[o2 + prompt] = 1.0
        flat[o2+7] = float(state.is_initial_build_phase)
        flat[o2+8] = float(state.is_discarding)
        flat[o2+9] = float(state.is_road_building)
        flat[o2+10] = float(state.is_moving_knight)
        flat[o2+11] = float(state.is_resolving_trade)
        flat[o2+12] = state.num_turns / 1000.0

    # ------------------------------------------------------------------
    # Batch encoding (PyG-compatible)
    # ------------------------------------------------------------------

    def encode_batch(
        self, states: list[StateView],
    ) -> dict[str, torch.Tensor]:
        """Encode multiple states with PyG-compatible graph batching.

        Returns an extra ``batch`` key of shape ``(B·N,)`` mapping each
        node to its graph index within the batch.
        """
        B = len(states)
        if B == 0:
            return self._empty_batch()

        N = self.num_nodes
        E = self.num_edges

        all_nf = torch.empty(B * N, self.NODE_FEATURE_DIM)
        all_ef = torch.empty(B * E, self.EDGE_FEATURE_DIM)
        all_ff = torch.empty(B, self.FLAT_FEATURE_DIM)

        for i, s in enumerate(states):
            enc = self.encode(s)
            all_nf[i * N : (i + 1) * N] = enc["node_features"]
            all_ef[i * E : (i + 1) * E] = enc["edge_features"]
            all_ff[i] = enc["flat_features"]

        # Replicate edge_index with per-graph node offsets
        base = self._edge_index.repeat(1, B)              # (2, B·E)
        offsets = (
            torch.arange(B, dtype=torch.long).repeat_interleave(E) * N
        )
        batched_ei = base + offsets.unsqueeze(0)

        batch_vec = torch.arange(B, dtype=torch.long).repeat_interleave(N)

        return {
            "node_features": all_nf,
            "edge_index": batched_ei,
            "edge_features": all_ef,
            "flat_features": all_ff,
            "batch": batch_vec,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_batch(self) -> dict[str, torch.Tensor]:
        return {
            "node_features": torch.empty(0, self.NODE_FEATURE_DIM),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "edge_features": torch.empty(0, self.EDGE_FEATURE_DIM),
            "flat_features": torch.empty(0, self.FLAT_FEATURE_DIM),
            "batch": torch.empty(0, dtype=torch.long),
        }
