"""Load the compiled shared library and wire up ctypes function prototypes."""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from ctypes import POINTER, c_bool, c_double, c_int, c_uint64

from .structs import (
    Action,
    Board,
    CatanMap,
    DecideFn,
    Game,
    RngState,
    SearchCtx,
    SearchResult,
    State,
    ValueFn,
)

_LIB_DIR = Path(__file__).resolve().parent / "lib"

_lib_cache: ctypes.CDLL | None = None


def _lib_path() -> Path:
    name = "libcatan.dylib" if platform.system() == "Darwin" else "libcatan.so"
    path = _LIB_DIR / name
    if not path.exists():
        from .build_lib import build
        build()
    if not path.exists():
        raise FileNotFoundError(
            f"Shared library not found at {path} even after build attempt."
        )
    return path


def load_library() -> ctypes.CDLL:
    """Load libcatan and register all function signatures.

    The library is loaded once and cached for the process lifetime.
    """
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    lib = ctypes.CDLL(str(_lib_path()))
    _bind_rng(lib)
    _bind_map(lib)
    _bind_board(lib)
    _bind_state(lib)
    _bind_game(lib)
    _bind_actions(lib)
    _bind_value(lib)
    _bind_search(lib)
    _lib_cache = lib
    return lib


# ---------------------------------------------------------------------------
# Per-header prototype bindings
# ---------------------------------------------------------------------------

def _bind_rng(lib: ctypes.CDLL) -> None:
    lib.rng_init.argtypes = [POINTER(RngState), c_uint64]
    lib.rng_init.restype = None


def _bind_map(lib: ctypes.CDLL) -> None:
    lib.build_map.argtypes = [POINTER(CatanMap), c_int, c_int, POINTER(RngState)]
    lib.build_map.restype = None


def _bind_board(lib: ctypes.CDLL) -> None:
    lib.board_init_static_graph.argtypes = [POINTER(CatanMap)]
    lib.board_init_static_graph.restype = None

    lib.board_init.argtypes = [POINTER(Board), POINTER(CatanMap)]
    lib.board_init.restype = None


def _bind_state(lib: ctypes.CDLL) -> None:
    lib.state_init.argtypes = [
        POINTER(State),       # s
        c_int,                # num_players
        POINTER(c_int),       # colors[]
        POINTER(CatanMap),    # map
        c_int,                # discard_limit
        c_bool,               # friendly_robber
        c_int,                # vps_to_win
        POINTER(RngState),    # rng
    ]
    lib.state_init.restype = None


def _bind_game(lib: ctypes.CDLL) -> None:
    lib.game_init_with_map.argtypes = [
        POINTER(Game),        # g
        POINTER(CatanMap),    # map
        c_int,                # num_players
        POINTER(c_int),       # colors[]
        c_uint64,             # seed
        c_int,                # discard_limit
        c_bool,               # friendly_robber
        c_int,                # vps_to_win
    ]
    lib.game_init_with_map.restype = None

    lib.game_copy.argtypes = [POINTER(Game), POINTER(Game)]
    lib.game_copy.restype = None

    lib.game_winning_color.argtypes = [POINTER(Game)]
    lib.game_winning_color.restype = c_int

    lib.game_execute.argtypes = [
        POINTER(Game),        # g
        Action,               # action (by value)
        POINTER(Action),      # action_buf
        POINTER(c_int),       # action_count
    ]
    lib.game_execute.restype = None

    lib.game_play.argtypes = [POINTER(Game), DecideFn]
    lib.game_play.restype = c_int


def _bind_actions(lib: ctypes.CDLL) -> None:
    lib.generate_playable_actions.argtypes = [
        POINTER(State),
        POINTER(Action),
        c_int,
    ]
    lib.generate_playable_actions.restype = c_int


def _bind_value(lib: ctypes.CDLL) -> None:
    lib.base_value_fn.argtypes = [POINTER(Game), c_int]
    lib.base_value_fn.restype = c_double


def _bind_search(lib: ctypes.CDLL) -> None:
    lib.alphabeta_search.argtypes = [
        POINTER(SearchCtx),   # ctx
        POINTER(Game),        # g
        POINTER(Action),      # actions
        c_int,                # num_actions
        c_int,                # depth
        c_double,             # alpha
        c_double,             # beta
        c_int,                # bot_color
        ValueFn,              # eval_fn
    ]
    lib.alphabeta_search.restype = SearchResult

    lib.apply_action.argtypes = [POINTER(State), Action, POINTER(RngState)]
    lib.apply_action.restype = None


def get_static_adjacency(lib: ctypes.CDLL | None = None) -> tuple:
    """Extract STATIC_ADJ and STATIC_ADJ_COUNT global arrays from the C engine.

    Must be called AFTER at least one board_init_static_graph() call
    (which happens inside game_init_with_map).

    Returns (static_adj, adj_count) as numpy-compatible lists.
    """
    import numpy as np
    from .structs import TOTAL_NODES, MAX_DEGREE

    if lib is None:
        lib = load_library()

    IntArray = c_int * (TOTAL_NODES * MAX_DEGREE)
    IntArrayFlat = c_int * TOTAL_NODES

    adj_sym = IntArray.in_dll(lib, "STATIC_ADJ")
    cnt_sym = IntArrayFlat.in_dll(lib, "STATIC_ADJ_COUNT")

    adj = np.frombuffer(adj_sym, dtype=np.int32).reshape(TOTAL_NODES, MAX_DEGREE).copy()
    cnt = np.frombuffer(cnt_sym, dtype=np.int32).copy()

    return adj, cnt
