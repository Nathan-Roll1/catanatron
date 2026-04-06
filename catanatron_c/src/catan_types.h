#ifndef CATAN_TYPES_H
#define CATAN_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_PLAYERS     4
#define NUM_RESOURCES   5
#define NUM_DEV_TYPES   5
#define NUM_NODES       54
#define NUM_EDGES       72
#define NUM_LAND_TILES  19
#define NUM_PORTS       9
#define MAX_ACTIONS     128
#define DEV_DECK_SIZE   25
#define TURNS_LIMIT     1000

typedef enum {
    COLOR_RED    = 0,
    COLOR_BLUE   = 1,
    COLOR_ORANGE = 2,
    COLOR_WHITE  = 3,
    COLOR_NONE   = -1
} Color;

typedef enum {
    RES_WOOD  = 0,
    RES_BRICK = 1,
    RES_SHEEP = 2,
    RES_WHEAT = 3,
    RES_ORE   = 4,
    RES_NONE  = -1
} Resource;

typedef enum {
    DEV_KNIGHT         = 0,
    DEV_YEAR_OF_PLENTY = 1,
    DEV_MONOPOLY       = 2,
    DEV_ROAD_BUILDING  = 3,
    DEV_VICTORY_POINT  = 4,
    DEV_NONE           = -1
} DevCardType;

typedef enum {
    BLD_SETTLEMENT = 0,
    BLD_CITY       = 1,
    BLD_ROAD       = 2
} BuildingType;

typedef enum {
    AT_ROLL = 0,
    AT_MOVE_ROBBER,
    AT_DISCARD_RESOURCE,
    AT_BUILD_ROAD,
    AT_BUILD_SETTLEMENT,
    AT_BUILD_CITY,
    AT_BUY_DEVELOPMENT_CARD,
    AT_PLAY_KNIGHT_CARD,
    AT_PLAY_YEAR_OF_PLENTY,
    AT_PLAY_MONOPOLY,
    AT_PLAY_ROAD_BUILDING,
    AT_MARITIME_TRADE,
    AT_OFFER_TRADE,
    AT_ACCEPT_TRADE,
    AT_REJECT_TRADE,
    AT_CONFIRM_TRADE,
    AT_CANCEL_TRADE,
    AT_END_TURN
} ActionType;

typedef enum {
    PROMPT_BUILD_INITIAL_SETTLEMENT = 0,
    PROMPT_BUILD_INITIAL_ROAD,
    PROMPT_PLAY_TURN,
    PROMPT_DISCARD,
    PROMPT_MOVE_ROBBER,
    PROMPT_DECIDE_TRADE,
    PROMPT_DECIDE_ACCEPTEES
} ActionPrompt;

typedef enum {
    DIR_EAST = 0,
    DIR_SOUTHEAST,
    DIR_SOUTHWEST,
    DIR_WEST,
    DIR_NORTHWEST,
    DIR_NORTHEAST
} Direction;

typedef enum {
    NREF_NORTH = 0,
    NREF_NORTHEAST,
    NREF_SOUTHEAST,
    NREF_SOUTH,
    NREF_SOUTHWEST,
    NREF_NORTHWEST
} NodeRef;

typedef struct {
    int x, y, z;
} Coordinate;

typedef struct {
    Color    color;
    ActionType type;
    int32_t  value[5];
} Action;

typedef struct {
    Action   action;
    int32_t  result[2];
} ActionRecord;

/* Player state field indices -- player_state[player_idx][field] */
enum {
    PS_VICTORY_POINTS = 0,
    PS_ROADS_AVAILABLE,
    PS_SETTLEMENTS_AVAILABLE,
    PS_CITIES_AVAILABLE,
    PS_HAS_ROAD,
    PS_HAS_ARMY,
    PS_HAS_ROLLED,
    PS_HAS_PLAYED_DEV_CARD_IN_TURN,
    PS_ACTUAL_VICTORY_POINTS,
    PS_LONGEST_ROAD_LENGTH,
    PS_KNIGHT_OWNED_AT_START,
    PS_MONOPOLY_OWNED_AT_START,
    PS_YEAR_OF_PLENTY_OWNED_AT_START,
    PS_ROAD_BUILDING_OWNED_AT_START,
    PS_WOOD_IN_HAND,
    PS_BRICK_IN_HAND,
    PS_SHEEP_IN_HAND,
    PS_WHEAT_IN_HAND,
    PS_ORE_IN_HAND,
    PS_KNIGHT_IN_HAND,
    PS_YEAR_OF_PLENTY_IN_HAND,
    PS_MONOPOLY_IN_HAND,
    PS_ROAD_BUILDING_IN_HAND,
    PS_VICTORY_POINT_IN_HAND,
    PS_PLAYED_KNIGHT,
    PS_PLAYED_YEAR_OF_PLENTY,
    PS_PLAYED_MONOPOLY,
    PS_PLAYED_ROAD_BUILDING,
    PS_PLAYED_VICTORY_POINT,
    NUM_PLAYER_STATE_FIELDS
};

#define PS_RESOURCE_IN_HAND(r) (PS_WOOD_IN_HAND + (r))
#define PS_DEV_IN_HAND(d)      (PS_KNIGHT_IN_HAND + (d))
#define PS_PLAYED_DEV(d)       (PS_PLAYED_KNIGHT + (d))

static const int ROAD_COST[5]       = {1, 1, 0, 0, 0};
static const int SETTLEMENT_COST[5] = {1, 1, 1, 1, 0};
static const int CITY_COST[5]       = {0, 0, 0, 2, 3};
static const int DEV_CARD_COST[5]   = {0, 0, 1, 1, 1};

static const int PLAYER_INIT[NUM_PLAYER_STATE_FIELDS] = {
    0,  /* VICTORY_POINTS */
    15, /* ROADS_AVAILABLE */
    5,  /* SETTLEMENTS_AVAILABLE */
    4,  /* CITIES_AVAILABLE */
    0,  /* HAS_ROAD */
    0,  /* HAS_ARMY */
    0,  /* HAS_ROLLED */
    0,  /* HAS_PLAYED_DEV_CARD_IN_TURN */
    0,  /* ACTUAL_VICTORY_POINTS */
    0,  /* LONGEST_ROAD_LENGTH */
    0, 0, 0, 0, /* _OWNED_AT_START flags */
    0, 0, 0, 0, 0, /* resources in hand */
    0, 0, 0, 0, 0, /* dev cards in hand */
    0, 0, 0, 0, 0  /* played dev cards */
};

static inline bool coord_eq(Coordinate a, Coordinate b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

static inline bool action_eq(Action a, Action b) {
    if (a.color != b.color || a.type != b.type) return false;
    for (int i = 0; i < 5; i++)
        if (a.value[i] != b.value[i]) return false;
    return true;
}

#endif
