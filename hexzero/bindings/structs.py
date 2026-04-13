"""Ctypes struct definitions matching the C engine byte-for-byte."""

from __future__ import annotations

import ctypes
from ctypes import (
    Structure,
    POINTER,
    c_bool,
    c_double,
    c_int,
    c_int8,
    c_int32,
    c_uint32,
    c_uint64,
    c_void_p,
)

# ---------------------------------------------------------------------------
# Constants (mirror C #defines)
# ---------------------------------------------------------------------------

MAX_PLAYERS = 4
NUM_RESOURCES = 5
NUM_DEV_TYPES = 5
NUM_NODES = 54
NUM_EDGES = 72
NUM_LAND_TILES = 19
NUM_PORTS = 9
MAX_ACTIONS = 128
DEV_DECK_SIZE = 25
MAX_DEV_DECK = 25
TURNS_LIMIT = 1000
MT_N = 624
MAX_TOPO_TILES = 64
MAX_ADJ_TILES = 3
TOTAL_NODES = 96
MAX_DEGREE = 3
MAX_COMPONENTS = 8
MAX_ROAD_EDGES = 72
NUM_PLAYER_STATE_FIELDS = 29
MAX_SEARCH_DEPTH = 24

# ---------------------------------------------------------------------------
# Enum constants (all represented as c_int in structs)
# ---------------------------------------------------------------------------

COLOR_RED = 0
COLOR_BLUE = 1
COLOR_ORANGE = 2
COLOR_WHITE = 3
COLOR_NONE = -1

RES_WOOD = 0
RES_BRICK = 1
RES_SHEEP = 2
RES_WHEAT = 3
RES_ORE = 4
RES_NONE = -1

DEV_KNIGHT = 0
DEV_YEAR_OF_PLENTY = 1
DEV_MONOPOLY = 2
DEV_ROAD_BUILDING = 3
DEV_VICTORY_POINT = 4
DEV_NONE = -1

BLD_SETTLEMENT = 0
BLD_CITY = 1
BLD_ROAD = 2

AT_ROLL = 0
AT_MOVE_ROBBER = 1
AT_DISCARD_RESOURCE = 2
AT_BUILD_ROAD = 3
AT_BUILD_SETTLEMENT = 4
AT_BUILD_CITY = 5
AT_BUY_DEVELOPMENT_CARD = 6
AT_PLAY_KNIGHT_CARD = 7
AT_PLAY_YEAR_OF_PLENTY = 8
AT_PLAY_MONOPOLY = 9
AT_PLAY_ROAD_BUILDING = 10
AT_MARITIME_TRADE = 11
AT_OFFER_TRADE = 12
AT_ACCEPT_TRADE = 13
AT_REJECT_TRADE = 14
AT_CONFIRM_TRADE = 15
AT_CANCEL_TRADE = 16
AT_END_TURN = 17

PROMPT_BUILD_INITIAL_SETTLEMENT = 0
PROMPT_BUILD_INITIAL_ROAD = 1
PROMPT_PLAY_TURN = 2
PROMPT_DISCARD = 3
PROMPT_MOVE_ROBBER = 4
PROMPT_DECIDE_TRADE = 5
PROMPT_DECIDE_ACCEPTEES = 6

DIR_EAST = 0
DIR_SOUTHEAST = 1
DIR_SOUTHWEST = 2
DIR_WEST = 3
DIR_NORTHWEST = 4
DIR_NORTHEAST = 5

NREF_NORTH = 0
NREF_NORTHEAST = 1
NREF_SOUTHEAST = 2
NREF_SOUTH = 3
NREF_SOUTHWEST = 4
NREF_NORTHWEST = 5

PS_VICTORY_POINTS = 0
PS_ROADS_AVAILABLE = 1
PS_SETTLEMENTS_AVAILABLE = 2
PS_CITIES_AVAILABLE = 3
PS_HAS_ROAD = 4
PS_HAS_ARMY = 5
PS_HAS_ROLLED = 6
PS_HAS_PLAYED_DEV_CARD_IN_TURN = 7
PS_ACTUAL_VICTORY_POINTS = 8
PS_LONGEST_ROAD_LENGTH = 9
PS_KNIGHT_OWNED_AT_START = 10
PS_MONOPOLY_OWNED_AT_START = 11
PS_YEAR_OF_PLENTY_OWNED_AT_START = 12
PS_ROAD_BUILDING_OWNED_AT_START = 13
PS_WOOD_IN_HAND = 14
PS_BRICK_IN_HAND = 15
PS_SHEEP_IN_HAND = 16
PS_WHEAT_IN_HAND = 17
PS_ORE_IN_HAND = 18
PS_KNIGHT_IN_HAND = 19
PS_YEAR_OF_PLENTY_IN_HAND = 20
PS_MONOPOLY_IN_HAND = 21
PS_ROAD_BUILDING_IN_HAND = 22
PS_VICTORY_POINT_IN_HAND = 23
PS_PLAYED_KNIGHT = 24
PS_PLAYED_YEAR_OF_PLENTY = 25
PS_PLAYED_MONOPOLY = 26
PS_PLAYED_ROAD_BUILDING = 27
PS_PLAYED_VICTORY_POINT = 28

MAP_BASE = 0
MAP_MINI = 1
MAP_TOURNAMENT = 2

NPLACE_OFFICIAL_SPIRAL = 0
NPLACE_RANDOM = 1

# ---------------------------------------------------------------------------
# Struct definitions — declaration order mirrors the C header dependency chain
# ---------------------------------------------------------------------------


class Coordinate(Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("z", c_int),
    ]

    def __repr__(self) -> str:
        return f"Coordinate({self.x}, {self.y}, {self.z})"


class Action(Structure):
    _fields_ = [
        ("color", c_int),
        ("type", c_int),
        ("value", c_int32 * 5),
    ]

    def __repr__(self) -> str:
        vals = list(self.value)
        return f"Action(color={self.color}, type={self.type}, value={vals})"


class ActionRecord(Structure):
    _fields_ = [
        ("action", Action),
        ("result", c_int32 * 2),
    ]


class RngState(Structure):
    _fields_ = [
        ("mt", c_uint32 * MT_N),
        ("mti", c_int),
    ]


class LandTile(Structure):
    _fields_ = [
        ("id", c_int),
        ("resource", c_int),
        ("number", c_int),
        ("nodes", c_int * 6),
        ("edges", (c_int * 2) * 6),
    ]


class Port(Structure):
    _fields_ = [
        ("id", c_int),
        ("resource", c_int),
        ("direction", c_int),
        ("nodes", c_int * 6),
        ("edges", (c_int * 2) * 6),
    ]


class CatanMap(Structure):
    _fields_ = [
        ("land_tiles", LandTile * NUM_LAND_TILES),
        ("num_land_tiles", c_int),
        ("ports", Port * NUM_PORTS),
        ("num_ports", c_int),
        ("land_nodes", c_int * NUM_NODES),
        ("num_land_nodes", c_int),
        ("adjacent_tiles", (c_int * MAX_ADJ_TILES) * NUM_NODES),
        ("adjacent_tiles_count", c_int * NUM_NODES),
        ("port_nodes", (c_int * 10) * 6),
        ("port_nodes_count", c_int * 6),
        ("dice_probas", c_double * 13),
        ("land_tile_coords", Coordinate * NUM_LAND_TILES),
    ]


class Board(Structure):
    _fields_ = [
        ("map", POINTER(CatanMap)),
        ("buildings", c_int8 * TOTAL_NODES),
        ("road_owner", (c_int8 * MAX_DEGREE) * TOTAL_NODES),
        ("cc_sets", ((c_uint64 * 2) * MAX_COMPONENTS) * MAX_PLAYERS),
        ("cc_count", c_int * MAX_PLAYERS),
        ("buildable", c_uint64 * 2),
        ("road_lengths", c_int * MAX_PLAYERS),
        ("road_color", c_int),
        ("road_length", c_int),
        ("robber_coordinate", Coordinate),
    ]


class State(Structure):
    _fields_ = [
        ("board", Board),
        ("num_players", c_int),
        ("colors", c_int * MAX_PLAYERS),
        ("color_to_index", c_int * MAX_PLAYERS),
        ("player_state", (c_int * NUM_PLAYER_STATE_FIELDS) * MAX_PLAYERS),
        ("resource_freqdeck", c_int * NUM_RESOURCES),
        ("development_listdeck", c_int * MAX_DEV_DECK),
        ("dev_deck_size", c_int),
        ("settlements", (c_int * 5) * MAX_PLAYERS),
        ("settlement_count", c_int * MAX_PLAYERS),
        ("cities", (c_int * 4) * MAX_PLAYERS),
        ("city_count", c_int * MAX_PLAYERS),
        ("roads", ((c_int * 2) * 15) * MAX_PLAYERS),
        ("road_count", c_int * MAX_PLAYERS),
        ("num_action_records", c_int),
        ("num_turns", c_int),
        ("current_player_index", c_int),
        ("current_turn_index", c_int),
        ("current_prompt", c_int),
        ("is_initial_build_phase", c_bool),
        ("is_discarding", c_bool),
        ("discard_counts", c_int * MAX_PLAYERS),
        ("is_moving_knight", c_bool),
        ("is_road_building", c_bool),
        ("free_roads_available", c_int),
        ("is_resolving_trade", c_bool),
        ("current_trade", c_int * 11),
        ("acceptees", c_bool * MAX_PLAYERS),
        ("discard_limit", c_int),
        ("friendly_robber", c_bool),
        ("vps_to_win", c_int),
    ]


class Game(Structure):
    _fields_ = [
        ("state", State),
        ("map", POINTER(CatanMap)),
        ("rng", RngState),
        ("eval_ctx", c_void_p),
        ("seed", c_uint64),
        ("vps_to_win", c_int),
    ]


class SearchResult(Structure):
    _fields_ = [
        ("action", Action),
        ("value", c_double),
    ]


class SearchCtx(Structure):
    _fields_ = [
        ("pool", Game * MAX_SEARCH_DEPTH),
        ("actions", (Action * MAX_ACTIONS) * MAX_SEARCH_DEPTH),
        ("depth_counter", c_int),
        ("user_data", c_void_p),
    ]


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

DecideFn = ctypes.CFUNCTYPE(Action, POINTER(State), POINTER(Action), c_int)
ValueFn = ctypes.CFUNCTYPE(c_double, POINTER(Game), c_int)
