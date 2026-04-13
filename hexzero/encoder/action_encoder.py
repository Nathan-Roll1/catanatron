"""Action encoder for HexaZero: maps C Action structs to neural-network action indices.

Provides a fixed-size discrete action space (337 slots) covering every legal
Catan action.  The encoder translates between the C engine's ``Action`` struct
representation and flat integer indices suitable for a neural-network policy head.

Usage::

    encoder = ActionEncoder()           # standard base map
    idx = encoder.encode(action)        # Action -> int
    action = encoder.decode(idx, color) # int -> Action
    mask = encoder.get_action_mask(legal_actions)  # list[Action] -> Tensor
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# C engine constants  (mirrors catan_types.h enum values)
# ---------------------------------------------------------------------------

AT_ROLL: int = 0
AT_MOVE_ROBBER: int = 1
AT_DISCARD_RESOURCE: int = 2
AT_BUILD_ROAD: int = 3
AT_BUILD_SETTLEMENT: int = 4
AT_BUILD_CITY: int = 5
AT_BUY_DEVELOPMENT_CARD: int = 6
AT_PLAY_KNIGHT_CARD: int = 7
AT_PLAY_YEAR_OF_PLENTY: int = 8
AT_PLAY_MONOPOLY: int = 9
AT_PLAY_ROAD_BUILDING: int = 10
AT_MARITIME_TRADE: int = 11
AT_OFFER_TRADE: int = 12
AT_ACCEPT_TRADE: int = 13
AT_REJECT_TRADE: int = 14
AT_CONFIRM_TRADE: int = 15
AT_CANCEL_TRADE: int = 16
AT_END_TURN: int = 17

NUM_NODES: int = 54
TOTAL_NODES: int = 96
NUM_EDGES: int = 72
NUM_LAND_TILES: int = 19
NUM_RESOURCES: int = 5
MAX_PLAYERS: int = 4


# ---------------------------------------------------------------------------
# Lightweight Action type matching C struct layout
# ---------------------------------------------------------------------------

class Action(NamedTuple):
    """``typedef struct { Color color; ActionType type; int32_t value[5]; } Action;``"""

    color: int
    type: int
    value: tuple[int, int, int, int, int]


# ---------------------------------------------------------------------------
# Base-map topology helpers  (deterministic for all seeds)
# ---------------------------------------------------------------------------

_UNIT_VECTORS = (
    (1, -1, 0), (0, -1, 1), (-1, 0, 1),
    (-1, 1, 0), (0, 1, -1), (1, 0, -1),
)

# EdgeRef -> pair of NodeRefs that form the edge
_EDGE_NODEREFS = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 1))

# When a neighbor tile is found in direction *d*, which of the current tile's
# NodeRefs / EdgeRefs are shared with which of the neighbor's.
#   ((my_noderef_a, nb_noderef_a), (my_noderef_b, nb_noderef_b),
#    (my_edgeref, nb_edgeref))
_NEIGHBOR_SHARING = (
    ((1, 5), (2, 4), (0, 3)),  # DIR_EAST
    ((3, 5), (2, 0), (1, 4)),  # DIR_SOUTHEAST
    ((3, 1), (4, 0), (2, 5)),  # DIR_SOUTHWEST
    ((5, 1), (4, 2), (3, 0)),  # DIR_WEST
    ((0, 2), (5, 3), (4, 1)),  # DIR_NORTHWEST
    ((0, 4), (1, 3), (5, 2)),  # DIR_NORTHEAST
)

# Coordinates of all 37 tiles in the standard base-map topology,
# in the same insertion order as the C ``build_map`` function.
_BASE_TOPO_COORDS: tuple[tuple[int, int, int], ...] = (
    # 19 land tiles (center, ring 1, ring 2)
    (0, 0, 0),
    (1, -1, 0), (0, -1, 1), (-1, 0, 1), (-1, 1, 0), (0, 1, -1), (1, 0, -1),
    (2, -2, 0), (1, -2, 1), (0, -2, 2), (-1, -1, 2), (-2, 0, 2),
    (-2, 1, 1), (-2, 2, 0), (-1, 2, -1), (0, 2, -2), (1, 1, -2),
    (2, 0, -2), (2, -1, -1),
    # 18 water / port tiles (ring 3)
    (3, -3, 0), (2, -3, 1), (1, -3, 2), (0, -3, 3), (-1, -2, 3), (-2, -1, 3),
    (-3, 0, 3), (-3, 1, 2), (-3, 2, 1), (-3, 3, 0), (-2, 3, -1), (-1, 3, -2),
    (0, 3, -3), (1, 2, -3), (2, 1, -3), (3, 0, -3), (3, -1, -2), (3, -2, -1),
)

_NUM_LAND_ENTRIES = 19


def _build_base_map_data() -> (
    tuple[list[int], list[tuple[int, int]], list[tuple[int, int, int]]]
):
    """Replicate the C ``build_map`` topology walk for the standard base map.

    Returns
    -------
    land_node_ids : list[int]
        Sorted list of 54 land-node board IDs.
    edges : list[tuple[int, int]]
        Sorted list of 72 canonical ``(a, b)`` edges with ``a < b``.
    tile_coordinates : list[tuple[int, int, int]]
        The 19 land-tile cube coordinates in tile-index order.
    """
    placed: dict[
        tuple[int, int, int],
        tuple[list[int], list[tuple[int, int]]],
    ] = {}
    next_id = 0

    land_node_set: set[int] = set()
    edge_set: set[tuple[int, int]] = set()
    tile_coords: list[tuple[int, int, int]] = []

    for idx, coord in enumerate(_BASE_TOPO_COORDS):
        nodes = [-1] * 6
        edges: list[tuple[int, int]] = [(-1, -1)] * 6

        for d in range(6):
            ux, uy, uz = _UNIT_VECTORS[d]
            nb = (coord[0] + ux, coord[1] + uy, coord[2] + uz)
            if nb not in placed:
                continue
            nb_nodes, nb_edges = placed[nb]
            (ma, na), (mb, nb_ref), (me, ne) = _NEIGHBOR_SHARING[d]
            nodes[ma] = nb_nodes[na]
            nodes[mb] = nb_nodes[nb_ref]
            edges[me] = nb_edges[ne]

        for i in range(6):
            if nodes[i] == -1:
                nodes[i] = next_id
                next_id += 1

        for i in range(6):
            if edges[i] == (-1, -1):
                a_ref, b_ref = _EDGE_NODEREFS[i]
                edges[i] = (nodes[a_ref], nodes[b_ref])

        placed[coord] = (nodes, edges)

        if idx < _NUM_LAND_ENTRIES:
            tile_coords.append(coord)
            land_node_set.update(nodes)
            for a, b in edges:
                edge_set.add((min(a, b), max(a, b)))

    return sorted(land_node_set), sorted(edge_set), tile_coords


# ---------------------------------------------------------------------------
# Pre-built decode tables for Year-of-Plenty and Maritime Trade
# ---------------------------------------------------------------------------

# 15 pairs (r1, r2) with r1 <= r2
_YOP_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (r1, r2) for r1 in range(NUM_RESOURCES) for r2 in range(r1, NUM_RESOURCES)
)

# 20 ordered (give, receive) with give != receive
_MARITIME_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (g, r) for g in range(NUM_RESOURCES) for r in range(NUM_RESOURCES) if r != g
)

# Types occupying a single fixed slot (used by decode for indices 0-4)
_SINGLE_SLOT_TYPES: tuple[int, ...] = (
    AT_ROLL,
    AT_END_TURN,
    AT_BUY_DEVELOPMENT_CARD,
    AT_PLAY_KNIGHT_CARD,
    AT_PLAY_ROAD_BUILDING,
)


# ---------------------------------------------------------------------------
# ActionEncoder
# ---------------------------------------------------------------------------

class ActionEncoder:
    """Fixed-size encoder between C ``Action`` structs and NN action indices.

    Action-space layout (337 total):

    ======  ====  ============================
    Offset  Size  Action type
    ======  ====  ============================
         0     1  AT_ROLL
         1     1  AT_END_TURN
         2     1  AT_BUY_DEVELOPMENT_CARD
         3     1  AT_PLAY_KNIGHT_CARD
         4     1  AT_PLAY_ROAD_BUILDING
         5    54  AT_BUILD_SETTLEMENT (by node)
        59    54  AT_BUILD_CITY (by node)
       113    72  AT_BUILD_ROAD (by edge)
       185    95  AT_MOVE_ROBBER (19 tiles x 5)
       280     5  AT_DISCARD_RESOURCE
       285    20  AT_PLAY_YEAR_OF_PLENTY
       305     5  AT_PLAY_MONOPOLY
       310    20  AT_MARITIME_TRADE
       330     1  AT_ACCEPT_TRADE
       331     1  AT_REJECT_TRADE
       332     1  AT_CANCEL_TRADE
       333     4  AT_CONFIRM_TRADE (by player)
    ======  ====  ============================
    """

    # Constant offsets (class-level to avoid per-instance storage overhead)
    _ROLL: int = 0
    _END_TURN: int = 1
    _BUY_DEV: int = 2
    _KNIGHT: int = 3
    _ROAD_BUILDING: int = 4
    _SETTLEMENT: int = 5
    _CITY: int = _SETTLEMENT + NUM_NODES                      # 59
    _ROAD: int = _CITY + NUM_NODES                             # 113
    _ROBBER: int = _ROAD + NUM_EDGES                           # 185
    _DISCARD: int = _ROBBER + NUM_LAND_TILES * 5               # 280
    _YOP: int = _DISCARD + NUM_RESOURCES                       # 285
    _MONOPOLY: int = _YOP + 20                                 # 305
    _MARITIME: int = _MONOPOLY + NUM_RESOURCES                 # 310
    _ACCEPT: int = _MARITIME + 20                              # 330
    _REJECT: int = _ACCEPT + 1                                 # 331
    _CANCEL: int = _REJECT + 1                                 # 332
    _CONFIRM: int = _CANCEL + 1                                # 333

    ACTION_SPACE_SIZE: int = _CONFIRM + MAX_PLAYERS            # 337

    _NO_STEAL: int = MAX_PLAYERS  # sentinel steal slot (4 = "no steal")
    _NUM_YOP_PAIRS: int = 15

    def __init__(
        self,
        land_node_ids: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        tile_coordinates: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Build lookup tables for action encoding / decoding.

        Parameters
        ----------
        land_node_ids
            Sorted list of 54 land-node board IDs (from ``CatanMap.land_nodes``).
            If *None*, defaults are computed from the standard base-map topology.
        edges
            List of 72 canonical ``(a, b)`` edge pairs with ``a < b``.
        tile_coordinates
            List of 19 ``(x, y, z)`` cube coordinates for each land tile.
        """
        if land_node_ids is None or edges is None or tile_coordinates is None:
            dn, de, dt = _build_base_map_data()
            land_node_ids = land_node_ids if land_node_ids is not None else dn
            edges = edges if edges is not None else de
            tile_coordinates = tile_coordinates if tile_coordinates is not None else dt

        if len(land_node_ids) != NUM_NODES:
            raise ValueError(
                f"Expected {NUM_NODES} land nodes, got {len(land_node_ids)}"
            )
        if len(edges) != NUM_EDGES:
            raise ValueError(f"Expected {NUM_EDGES} edges, got {len(edges)}")
        if len(tile_coordinates) != NUM_LAND_TILES:
            raise ValueError(
                f"Expected {NUM_LAND_TILES} tile coordinates, "
                f"got {len(tile_coordinates)}"
            )

        # --- Node mapping: board node_id -> compact index (0-53) ----------
        self._node_to_compact = np.full(TOTAL_NODES, -1, dtype=np.int32)
        self._compact_to_node = np.array(land_node_ids, dtype=np.int32)
        for compact_idx, node_id in enumerate(land_node_ids):
            self._node_to_compact[node_id] = compact_idx

        # --- Edge mapping: canonical (a, b) -> edge index (0-71) ----------
        self._edge_lut = np.full(
            (TOTAL_NODES, TOTAL_NODES), -1, dtype=np.int16
        )
        self._idx_to_edge = np.empty((NUM_EDGES, 2), dtype=np.int32)
        for edge_idx, (a, b) in enumerate(edges):
            self._edge_lut[a, b] = edge_idx
            self._idx_to_edge[edge_idx] = (a, b)

        # --- Tile coordinate mapping: (x,y,z) -> tile index (0-18) --------
        self._coord_to_tile: dict[tuple[int, int, int], int] = {
            c: i for i, c in enumerate(tile_coordinates)
        }
        self._tile_to_coord = np.array(tile_coordinates, dtype=np.int32)

        # --- Year-of-Plenty pair LUT: (r1, r2) -> pair_index (0-14) ------
        self._yop_lut = np.full(
            (NUM_RESOURCES, NUM_RESOURCES), -1, dtype=np.int16
        )
        for pi, (r1, r2) in enumerate(_YOP_PAIRS):
            self._yop_lut[r1, r2] = pi

        # --- Maritime trade LUT: (give, receive) -> trade_index (0-19) ----
        self._mar_lut = np.full(
            (NUM_RESOURCES, NUM_RESOURCES), -1, dtype=np.int16
        )
        for ti, (g, r) in enumerate(_MARITIME_PAIRS):
            self._mar_lut[g, r] = ti

    # ------------------------------------------------------------------
    # Encode: Action -> int
    # ------------------------------------------------------------------

    def encode(self, action: object) -> int:
        """Convert a C ``Action`` struct (or NamedTuple) to an action-space index.

        Accepts both ctypes Action structs (attribute access) and
        NamedTuple Action values. The action must have ``.type`` and
        ``.value`` attributes.

        Raises
        ------
        ValueError
            If the action type is unsupported (e.g. ``AT_OFFER_TRADE``).
        """
        t: int = action.type   # type: ignore[union-attr]
        v = action.value       # type: ignore[union-attr]

        # Hot-path types first (most frequent in MCTS legal-action lists)
        if t == AT_BUILD_SETTLEMENT:
            return self._SETTLEMENT + int(self._node_to_compact[v[0]])
        if t == AT_BUILD_ROAD:
            a, b = (v[0], v[1]) if v[0] < v[1] else (v[1], v[0])
            return self._ROAD + int(self._edge_lut[a, b])
        if t == AT_BUILD_CITY:
            return self._CITY + int(self._node_to_compact[v[0]])
        if t == AT_ROLL:
            return self._ROLL
        if t == AT_END_TURN:
            return self._END_TURN
        if t == AT_MOVE_ROBBER:
            tile = self._coord_to_tile[(v[0], v[1], v[2])]
            steal = v[3] if v[3] >= 0 else self._NO_STEAL
            return self._ROBBER + tile * 5 + steal
        if t == AT_BUY_DEVELOPMENT_CARD:
            return self._BUY_DEV
        if t == AT_MARITIME_TRADE:
            return self._MARITIME + int(self._mar_lut[v[0], v[4]])
        if t == AT_PLAY_KNIGHT_CARD:
            return self._KNIGHT
        if t == AT_PLAY_ROAD_BUILDING:
            return self._ROAD_BUILDING
        if t == AT_DISCARD_RESOURCE:
            return self._DISCARD + v[0]
        if t == AT_PLAY_YEAR_OF_PLENTY:
            if v[1] < 0:
                return self._YOP + self._NUM_YOP_PAIRS + v[0]
            return self._YOP + int(self._yop_lut[v[0], v[1]])
        if t == AT_PLAY_MONOPOLY:
            return self._MONOPOLY + v[0]
        if t == AT_ACCEPT_TRADE:
            return self._ACCEPT
        if t == AT_REJECT_TRADE:
            return self._REJECT
        if t == AT_CANCEL_TRADE:
            return self._CANCEL
        if t == AT_CONFIRM_TRADE:
            return self._CONFIRM + v[4]

        raise ValueError(f"Unsupported action type {t}")

    # ------------------------------------------------------------------
    # Decode: int -> Action
    # ------------------------------------------------------------------

    def decode(self, index: int, color: int) -> Action:
        """Convert an action-space index back to a C ``Action`` struct.

        Notes
        -----
        * Maritime trade actions are decoded with 4:1 rate format
          (``[give, give, give, give, receive]``).
        * Trade accept / reject / cancel carry zeroed value fields;
          match against the legal-actions list for the full struct.
        """
        _z = (0, 0, 0, 0, 0)

        # --- Single-slot actions (indices 0-4) ---
        if index < self._SETTLEMENT:
            return Action(color, _SINGLE_SLOT_TYPES[index], _z)

        # --- BUILD_SETTLEMENT (indices 5-58) ---
        if index < self._CITY:
            nid = int(self._compact_to_node[index - self._SETTLEMENT])
            return Action(color, AT_BUILD_SETTLEMENT, (nid, 0, 0, 0, 0))

        # --- BUILD_CITY (indices 59-112) ---
        if index < self._ROAD:
            nid = int(self._compact_to_node[index - self._CITY])
            return Action(color, AT_BUILD_CITY, (nid, 0, 0, 0, 0))

        # --- BUILD_ROAD (indices 113-184) ---
        if index < self._ROBBER:
            ei = index - self._ROAD
            a = int(self._idx_to_edge[ei, 0])
            b = int(self._idx_to_edge[ei, 1])
            return Action(color, AT_BUILD_ROAD, (a, b, 0, 0, 0))

        # --- MOVE_ROBBER (indices 185-279) ---
        if index < self._DISCARD:
            off = index - self._ROBBER
            tile, steal = divmod(off, 5)
            x = int(self._tile_to_coord[tile, 0])
            y = int(self._tile_to_coord[tile, 1])
            z = int(self._tile_to_coord[tile, 2])
            rob = steal if steal < self._NO_STEAL else -1
            return Action(color, AT_MOVE_ROBBER, (x, y, z, rob, 0))

        # --- DISCARD_RESOURCE (indices 280-284) ---
        if index < self._YOP:
            return Action(
                color, AT_DISCARD_RESOURCE, (index - self._DISCARD, 0, 0, 0, 0)
            )

        # --- PLAY_YEAR_OF_PLENTY (indices 285-304) ---
        if index < self._MONOPOLY:
            yi = index - self._YOP
            if yi < self._NUM_YOP_PAIRS:
                r1, r2 = _YOP_PAIRS[yi]
                return Action(
                    color, AT_PLAY_YEAR_OF_PLENTY, (r1, r2, 0, 0, 0)
                )
            r = yi - self._NUM_YOP_PAIRS
            return Action(color, AT_PLAY_YEAR_OF_PLENTY, (r, -1, 0, 0, 0))

        # --- PLAY_MONOPOLY (indices 305-309) ---
        if index < self._MARITIME:
            return Action(
                color, AT_PLAY_MONOPOLY, (index - self._MONOPOLY, 0, 0, 0, 0)
            )

        # --- MARITIME_TRADE (indices 310-329) ---
        if index < self._ACCEPT:
            g, r = _MARITIME_PAIRS[index - self._MARITIME]
            return Action(color, AT_MARITIME_TRADE, (g, g, g, g, r))

        # --- ACCEPT / REJECT / CANCEL (indices 330-332) ---
        if index == self._ACCEPT:
            return Action(color, AT_ACCEPT_TRADE, _z)
        if index == self._REJECT:
            return Action(color, AT_REJECT_TRADE, _z)
        if index == self._CANCEL:
            return Action(color, AT_CANCEL_TRADE, _z)

        # --- CONFIRM_TRADE (indices 333-336) ---
        if index < self.ACTION_SPACE_SIZE:
            accepting = index - self._CONFIRM
            return Action(color, AT_CONFIRM_TRADE, (0, 0, 0, 0, accepting))

        raise ValueError(f"Index {index} out of range [0, {self.ACTION_SPACE_SIZE})")

    # ------------------------------------------------------------------
    # Batch encoding
    # ------------------------------------------------------------------

    def encode_batch(self, actions: list[Action]) -> np.ndarray:
        """Encode multiple actions into a numpy ``int32`` array of indices."""
        n = len(actions)
        out = np.empty(n, dtype=np.int32)
        enc = self.encode  # local ref avoids repeated attribute lookup
        for i in range(n):
            out[i] = enc(actions[i])
        return out

    # ------------------------------------------------------------------
    # Masking & policy targets
    # ------------------------------------------------------------------

    def get_action_mask(self, legal_actions: list[Action]) -> torch.Tensor:
        """Create a binary mask of shape ``(ACTION_SPACE_SIZE,)``.

        Returns a ``float32`` tensor: ``1.0`` for legal actions, ``0.0`` elsewhere.
        """
        mask = torch.zeros(self.ACTION_SPACE_SIZE, dtype=torch.float32)
        if legal_actions:
            enc = self.encode
            mask[[enc(a) for a in legal_actions]] = 1.0
        return mask

    def get_policy_target(
        self,
        legal_actions: list[Action],
        visit_counts: np.ndarray,
    ) -> torch.Tensor:
        """Convert MCTS visit counts to a probability vector over the full action space.

        ``visit_counts[i]`` corresponds to ``legal_actions[i]``.
        The output sums to 1.0 across legal positions.
        """
        target = torch.zeros(self.ACTION_SPACE_SIZE, dtype=torch.float32)
        total = visit_counts.sum()
        if total > 0:
            indices = self.encode_batch(legal_actions)
            probs = (visit_counts / total).astype(np.float32)
            idx_tensor = torch.from_numpy(indices.astype(np.int64))
            target.scatter_(0, idx_tensor, torch.from_numpy(probs))
        return target

    @staticmethod
    def masked_softmax(
        logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mask then softmax.  Masked (illegal) positions get ``-inf``."""
        return torch.softmax(
            logits.masked_fill(mask == 0.0, float("-inf")), dim=-1
        )
