#ifndef STATE_H
#define STATE_H

#include "catan_types.h"
#include "board.h"

#define MAX_ACTION_RECORDS 2048
#define MAX_DEV_DECK       25

typedef struct {
    Board   board;

    int     num_players;
    Color   colors[MAX_PLAYERS]; /* seating order (shuffled) */
    int     color_to_index[MAX_PLAYERS]; /* Color -> seat index */

    int     player_state[MAX_PLAYERS][NUM_PLAYER_STATE_FIELDS];

    int     resource_freqdeck[NUM_RESOURCES]; /* bank */
    int     development_listdeck[MAX_DEV_DECK];
    int     dev_deck_size;

    /* Per-color building lists (for fast iteration) */
    int     settlements[MAX_PLAYERS][5]; /* max 5 settlements */
    int     settlement_count[MAX_PLAYERS];
    int     cities[MAX_PLAYERS][4];      /* max 4 cities */
    int     city_count[MAX_PLAYERS];
    int     roads[MAX_PLAYERS][15][2];   /* max 15 roads */
    int     road_count[MAX_PLAYERS];

    int     num_action_records; /* count only; records stored externally */

    int     num_turns;
    int     current_player_index;
    int     current_turn_index;

    ActionPrompt current_prompt;
    bool    is_initial_build_phase;
    bool    is_discarding;
    int     discard_counts[MAX_PLAYERS];
    bool    is_moving_knight;
    bool    is_road_building;
    int     free_roads_available;

    bool    is_resolving_trade;
    int     current_trade[11];
    bool    acceptees[MAX_PLAYERS];

    int     discard_limit;
    bool    friendly_robber;
    int     vps_to_win;
} State;

/* State management */
void state_init(State *s, int num_players, Color colors[], CatanMap *map,
                int discard_limit, bool friendly_robber, int vps_to_win,
                RngState *rng);

/* state_copy: just memcpy the whole struct */
static inline void state_copy(State *dst, const State *src) {
    *dst = *src;
    /* board.map pointer is shared (immutable), no need to deep copy */
}

/* Player state accessors */
#define PS(s, color, field) ((s)->player_state[(s)->color_to_index[(int)(color)]][(field)])
#define PS_IDX(s, idx, field) ((s)->player_state[(idx)][(field)])

/* Current player color */
static inline Color state_current_color(const State *s) {
    return s->colors[s->current_player_index];
}

/* Freqdeck helpers */
static inline bool freqdeck_contains(const int a[5], const int b[5]) {
    return a[0]>=b[0] && a[1]>=b[1] && a[2]>=b[2] && a[3]>=b[3] && a[4]>=b[4];
}
static inline void freqdeck_subtract(int dst[5], const int b[5]) {
    for (int i = 0; i < 5; i++) dst[i] -= b[i];
}
static inline void freqdeck_add(int dst[5], const int b[5]) {
    for (int i = 0; i < 5; i++) dst[i] += b[i];
}
static inline int freqdeck_total(const int d[5]) {
    return d[0]+d[1]+d[2]+d[3]+d[4];
}
static inline int player_num_resources(const State *s, int idx) {
    return s->player_state[idx][PS_WOOD_IN_HAND]
         + s->player_state[idx][PS_BRICK_IN_HAND]
         + s->player_state[idx][PS_SHEEP_IN_HAND]
         + s->player_state[idx][PS_WHEAT_IN_HAND]
         + s->player_state[idx][PS_ORE_IN_HAND];
}
static inline int player_num_devs(const State *s, int idx) {
    return s->player_state[idx][PS_KNIGHT_IN_HAND]
         + s->player_state[idx][PS_YEAR_OF_PLENTY_IN_HAND]
         + s->player_state[idx][PS_MONOPOLY_IN_HAND]
         + s->player_state[idx][PS_ROAD_BUILDING_IN_HAND]
         + s->player_state[idx][PS_VICTORY_POINT_IN_HAND];
}

/* Get player freqdeck (resource hand) */
static inline void player_get_hand(const State *s, int idx, int out[5]) {
    out[0] = s->player_state[idx][PS_WOOD_IN_HAND];
    out[1] = s->player_state[idx][PS_BRICK_IN_HAND];
    out[2] = s->player_state[idx][PS_SHEEP_IN_HAND];
    out[3] = s->player_state[idx][PS_WHEAT_IN_HAND];
    out[4] = s->player_state[idx][PS_ORE_IN_HAND];
}

#endif
