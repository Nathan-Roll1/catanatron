"""High-performance Catan game interface wrapping the C engine via ctypes.

Designed for RL/MCTS workloads:
- CatanMap allocated once, shared across all clones (immutable after build)
- Legal actions cached from game_execute (avoids regeneration)
- clone() is a struct copy with shared map pointer
- Pre-allocated action buffers avoid per-step allocation
- Numpy extraction uses from_address for zero-copy reads
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from hexzero.bindings.lib_loader import load_library
from hexzero.bindings.structs import (
    Action,
    CatanMap,
    Game,
    RngState,
    COLOR_NONE,
    MAP_BASE,
    MAX_ACTIONS,
    MAX_DEGREE,
    MAX_PLAYERS,
    NPLACE_OFFICIAL_SPIRAL,
    NPLACE_RANDOM_BALANCED,
    NUM_LAND_TILES,
    NUM_PLAYER_STATE_FIELDS,
    NUM_RESOURCES,
    TOTAL_NODES,
)
from hexzero.config import GameConfig

# Pre-built ctypes array types for efficient buffer operations
_ActionArray = Action * MAX_ACTIONS
_PSFlat = ctypes.c_int * (MAX_PLAYERS * NUM_PLAYER_STATE_FIELDS)
_ROFlat = ctypes.c_int8 * (TOTAL_NODES * MAX_DEGREE)
_ACTION_SIZE = ctypes.sizeof(Action)


@dataclass(frozen=True)
class _MapInfo:
    """Cached immutable map data extracted once from the C CatanMap struct."""

    tile_resources: np.ndarray  # (19,) int32
    tile_numbers: np.ndarray  # (19,) int32
    tile_nodes: np.ndarray  # (19, 6) int32


@dataclass
class StateView:
    """Structured snapshot of game state for neural network encoding.

    All mutable arrays (player_state, buildings, road_owners, resource_bank)
    are independent copies safe to retain and batch across steps.
    Map arrays (tile_*) are shared read-only views from the cached map data.
    """

    current_player: int
    num_turns: int
    is_initial_build_phase: bool
    current_prompt: int
    num_players: int

    player_state: np.ndarray  # (4, 29) int32
    buildings: np.ndarray  # (96,) int8 — packed color<<2|type, or -1
    road_owners: np.ndarray  # (96, 3) int8 — color per adj slot, or -1
    color_to_index: np.ndarray  # (4,) int32 — maps Color enum -> seat index
    robber_coord: tuple[int, int, int]

    resource_bank: np.ndarray  # (5,) int32
    dev_deck_size: int

    tile_resources: np.ndarray  # (19,) int32 [read-only, shared]
    tile_numbers: np.ndarray  # (19,) int32 [read-only, shared]
    tile_nodes: np.ndarray  # (19, 6) int32 [read-only, shared]

    is_discarding: bool
    is_road_building: bool
    is_moving_knight: bool
    is_resolving_trade: bool


class CatanGame:
    """High-performance Catan game interface wrapping the C engine.

    Typical usage for self-play::

        game = CatanGame(seed=42)
        state = game.reset()
        while not game.is_terminal():
            idx = select_action(state, game.get_legal_action_count())
            state, done = game.step(idx)

    For MCTS simulation::

        sim = game.clone()          # cheap struct copy
        while not sim.is_terminal():
            action = sim.get_legal_actions()[random_idx]
            sim.apply_action_direct(action)
    """

    __slots__ = (
        "_lib",
        "_game",
        "_map_obj",
        "_map_ptr",
        "_map_info",
        "_action_buf",
        "_action_count",
        "_num_players",
        "_config",
        "_nplace",
    )

    def __init__(
        self,
        seed: int = 0,
        num_players: int = 4,
        config: GameConfig | None = None,
        random_board: bool = False,
    ):
        self._lib: ctypes.CDLL = load_library()
        self._config = config or GameConfig(num_players=num_players)
        self._num_players = self._config.num_players

        self._action_buf = _ActionArray()
        self._action_count = ctypes.c_int(0)

        self._nplace = NPLACE_RANDOM_BALANCED if random_board else NPLACE_OFFICIAL_SPIRAL

        self._map_obj = CatanMap()
        rng = RngState()
        self._lib.rng_init(ctypes.byref(rng), ctypes.c_uint64(seed))
        self._lib.build_map(
            ctypes.byref(self._map_obj),
            MAP_BASE,
            self._nplace,
            ctypes.byref(rng),
        )
        self._map_ptr = ctypes.pointer(self._map_obj)
        self._map_info = _extract_map_info(self._map_obj)

        self._game = Game()
        self._init_game(seed)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_game(self, seed: int) -> None:
        """Initialize the C Game struct and generate initial legal actions."""
        colors = (ctypes.c_int * MAX_PLAYERS)()
        for i in range(self._num_players):
            colors[i] = i
        for i in range(self._num_players, MAX_PLAYERS):
            colors[i] = COLOR_NONE

        self._lib.game_init_with_map(
            ctypes.byref(self._game),
            self._map_ptr,
            self._num_players,
            colors,
            ctypes.c_uint64(seed),
            self._config.discard_limit,
            self._config.friendly_robber,
            self._config.vps_to_win,
        )

        self._action_count.value = self._lib.generate_playable_actions(
            ctypes.byref(self._game.state),
            self._action_buf,
            MAX_ACTIONS,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> StateView:
        """Reset to initial state.  Returns the starting observation.

        When *seed* is provided a fresh CatanMap is allocated so that
        existing clones of the previous game remain valid.
        """
        if seed is not None:
            self._map_obj = CatanMap()
            rng = RngState()
            self._lib.rng_init(ctypes.byref(rng), ctypes.c_uint64(seed))
            self._lib.build_map(
                ctypes.byref(self._map_obj),
                MAP_BASE,
                self._nplace,
                ctypes.byref(rng),
            )
            self._map_ptr = ctypes.pointer(self._map_obj)
            self._map_info = _extract_map_info(self._map_obj)
        else:
            seed = int(self._game.seed)

        self._init_game(seed)
        return self.get_state_view()

    def step(self, action_index: int) -> tuple[StateView, bool]:
        """Apply an action by index into the current legal action list.

        Returns ``(state_view, done)`` where *done* is ``True`` when a
        player has won or the turn limit has been reached.

        Raises:
            RuntimeError: If the game is already terminal.
            IndexError: If *action_index* is out of range.
        """
        if self.is_terminal():
            raise RuntimeError("Cannot step: game is already terminal")

        n = self._action_count.value
        if not 0 <= action_index < n:
            raise IndexError(
                f"action_index {action_index} out of range [0, {n})"
            )

        self._lib.game_execute(
            ctypes.byref(self._game),
            self._action_buf[action_index],
            self._action_buf,
            ctypes.byref(self._action_count),
        )

        done = self.is_terminal()
        return self.get_state_view(), done

    def get_legal_actions(self) -> list[Action]:
        """Return independent copies of the currently legal actions."""
        n = self._action_count.value
        return [
            Action.from_buffer_copy(self._action_buf, i * _ACTION_SIZE)
            for i in range(n)
        ]

    def get_legal_action_count(self) -> int:
        """Return the number of currently legal actions."""
        return self._action_count.value

    def current_player(self) -> int:
        """Return the current player's seat index (0-based)."""
        return self._game.state.current_player_index

    def winner(self) -> int | None:
        """Return the winning player's seat index, or ``None``."""
        color: int = self._lib.game_winning_color(ctypes.byref(self._game))
        if color == COLOR_NONE:
            return None
        state = self._game.state
        for i in range(state.num_players):
            if state.colors[i] == color:
                return i
        return None  # pragma: no cover

    def is_terminal(self) -> bool:
        """``True`` when a player has won or the turn limit was reached."""
        if self._lib.game_winning_color(ctypes.byref(self._game)) != COLOR_NONE:
            return True
        return self._game.state.num_turns >= self._config.turns_limit

    def clone(self) -> CatanGame:
        """Deep-copy the game state for MCTS simulation.

        The CatanMap is shared (immutable, pointer copy).  The action
        buffer is partially copied (only the live portion).
        """
        new = object.__new__(CatanGame)
        new._lib = self._lib
        new._config = self._config
        new._num_players = self._num_players
        new._map_obj = self._map_obj
        new._map_ptr = self._map_ptr
        new._map_info = self._map_info

        new._game = Game()
        new._action_buf = _ActionArray()
        n = self._action_count.value
        new._action_count = ctypes.c_int(n)

        self._lib.game_copy(
            ctypes.byref(new._game), ctypes.byref(self._game)
        )

        if n > 0:
            ctypes.memmove(new._action_buf, self._action_buf, _ACTION_SIZE * n)

        return new

    def get_state_view(self) -> StateView:
        """Return a structured snapshot of the current game state.

        Mutable board/player arrays are copied.  Map arrays are shared
        read-only references (the underlying map is immutable).
        """
        state = self._game.state
        board = state.board

        ps = _PSFlat.from_address(ctypes.addressof(state.player_state))
        player_state = np.frombuffer(ps, dtype=np.int32).reshape(
            MAX_PLAYERS, NUM_PLAYER_STATE_FIELDS
        ).copy()

        buildings = np.ctypeslib.as_array(board.buildings).copy()

        ro = _ROFlat.from_address(ctypes.addressof(board.road_owner))
        road_owners = np.frombuffer(ro, dtype=np.int8).reshape(
            TOTAL_NODES, MAX_DEGREE
        ).copy()

        rc = board.robber_coordinate

        resource_bank = np.array(state.resource_freqdeck, dtype=np.int32)

        c2i = np.array([state.color_to_index[i] for i in range(MAX_PLAYERS)],
                       dtype=np.int32)

        mi = self._map_info
        return StateView(
            current_player=state.current_player_index,
            num_turns=state.num_turns,
            is_initial_build_phase=bool(state.is_initial_build_phase),
            current_prompt=int(state.current_prompt),
            num_players=int(state.num_players),
            player_state=player_state,
            buildings=buildings,
            road_owners=road_owners,
            color_to_index=c2i,
            robber_coord=(rc.x, rc.y, rc.z),
            resource_bank=resource_bank,
            dev_deck_size=state.dev_deck_size,
            tile_resources=mi.tile_resources,
            tile_numbers=mi.tile_numbers,
            tile_nodes=mi.tile_nodes,
            is_discarding=bool(state.is_discarding),
            is_road_building=bool(state.is_road_building),
            is_moving_knight=bool(state.is_moving_knight),
            is_resolving_trade=bool(state.is_resolving_trade),
        )

    def state_view_from_struct(self, game_struct) -> StateView:
        """Build a StateView from an arbitrary Game struct sharing this map."""
        state = game_struct.state
        board = state.board

        ps = _PSFlat.from_address(ctypes.addressof(state.player_state))
        player_state = np.frombuffer(ps, dtype=np.int32).reshape(
            MAX_PLAYERS, NUM_PLAYER_STATE_FIELDS
        ).copy()

        buildings = np.ctypeslib.as_array(board.buildings).copy()

        ro = _ROFlat.from_address(ctypes.addressof(board.road_owner))
        road_owners = np.frombuffer(ro, dtype=np.int8).reshape(
            TOTAL_NODES, MAX_DEGREE
        ).copy()

        rc = board.robber_coordinate
        resource_bank = np.array(state.resource_freqdeck, dtype=np.int32)
        c2i = np.array([state.color_to_index[i] for i in range(MAX_PLAYERS)],
                       dtype=np.int32)

        mi = self._map_info
        return StateView(
            current_player=state.current_player_index,
            num_turns=state.num_turns,
            is_initial_build_phase=bool(state.is_initial_build_phase),
            current_prompt=int(state.current_prompt),
            num_players=int(state.num_players),
            player_state=player_state,
            buildings=buildings,
            road_owners=road_owners,
            color_to_index=c2i,
            robber_coord=(rc.x, rc.y, rc.z),
            resource_bank=resource_bank,
            dev_deck_size=state.dev_deck_size,
            tile_resources=mi.tile_resources,
            tile_numbers=mi.tile_numbers,
            tile_nodes=mi.tile_nodes,
            is_discarding=bool(state.is_discarding),
            is_road_building=bool(state.is_road_building),
            is_moving_knight=bool(state.is_moving_knight),
            is_resolving_trade=bool(state.is_resolving_trade),
        )

    def apply_action_direct(self, action: Action) -> None:
        """Apply a raw C Action struct directly (for MCTS playouts).

        Updates the internal legal action cache.
        """
        self._lib.game_execute(
            ctypes.byref(self._game),
            action,
            self._action_buf,
            ctypes.byref(self._action_count),
        )

    @property
    def turn_number(self) -> int:
        """Current turn number."""
        return self._game.state.num_turns

    @property
    def num_players(self) -> int:
        """Number of players in this game."""
        return self._num_players

    def make_state_encoder(self):
        """Create a StateEncoder configured for this game's map."""
        from hexzero.bindings.lib_loader import get_static_adjacency
        from hexzero.encoder.state_encoder import StateEncoder

        mi = self._map_info
        cmap = self._map_obj

        tile_coords = np.array(
            [(cmap.land_tile_coords[i].x, cmap.land_tile_coords[i].y, cmap.land_tile_coords[i].z)
             for i in range(NUM_LAND_TILES)],
            dtype=np.int32,
        )
        static_adj, adj_count = get_static_adjacency(self._lib)

        port_map = np.full(TOTAL_NODES, -1, dtype=np.int8)
        for pi in range(6):
            for ni in range(cmap.port_nodes_count[pi]):
                node = cmap.port_nodes[pi][ni]
                if 0 <= node < TOTAL_NODES:
                    port_map[node] = pi

        return StateEncoder(
            tile_nodes=mi.tile_nodes,
            tile_coords=tile_coords,
            static_adj=static_adj,
            adj_count=adj_count,
            port_map=port_map,
        )

    def __repr__(self) -> str:
        return (
            f"CatanGame(turn={self.turn_number}, "
            f"player={self.current_player()}, "
            f"actions={self.get_legal_action_count()}, "
            f"terminal={self.is_terminal()})"
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_map_info(cmap: CatanMap) -> _MapInfo:
    """Extract immutable tile data from a CatanMap into cached numpy arrays."""
    resources = np.empty(NUM_LAND_TILES, dtype=np.int32)
    numbers = np.empty(NUM_LAND_TILES, dtype=np.int32)
    nodes = np.empty((NUM_LAND_TILES, 6), dtype=np.int32)
    for i in range(NUM_LAND_TILES):
        t = cmap.land_tiles[i]
        resources[i] = t.resource
        numbers[i] = t.number
        for j in range(6):
            nodes[i, j] = t.nodes[j]
    resources.flags.writeable = False
    numbers.flags.writeable = False
    nodes.flags.writeable = False
    return _MapInfo(resources, numbers, nodes)
