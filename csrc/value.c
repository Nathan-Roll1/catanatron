/*
 * Hand-crafted value function for Catan position evaluation.
 * Matches the Python base_fn with inline production calculation.
 */

#include "value.h"
#include <math.h>

static const double DICE_P[13] = {
    0, 0,
    1.0/36, 2.0/36, 3.0/36, 4.0/36, 5.0/36, 6.0/36,
    5.0/36, 4.0/36, 3.0/36, 2.0/36, 1.0/36
};

static const double W_VPS       = 3e14;
static const double W_PROD      = 1e8;
static const double W_EPROD     = -1e8;
static const double W_TILES     = 1.0;
static const double W_BUILDABLE = 1e3;
static const double W_ROAD      = 10.0;
static const double W_SYNERGY   = 1e2;
static const double W_HAND      = 1.0;
static const double W_DISCARD   = -5.0;
static const double W_DEVS      = 10.0;
static const double W_ARMY      = 10.1;
static const double VARIETY_BONUS = 4.0 * (2.778 / 100.0);

static void compute_production(State *s, Color color, double *prod_out, int *variety_out) {
    Board *b = &s->board;
    CatanMap *map = b->map;
    Coordinate robber = b->robber_coordinate;
    int idx = s->color_to_index[(int)color];

    double total = 0.0;
    int variety = 0;
    double res_prod[5] = {0};

    /* Settlements */
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[tile_idx];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
            res_prod[(int)t->resource] += DICE_P[t->number];
        }
    }

    /* Cities */
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[tile_idx];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
            res_prod[(int)t->resource] += 2.0 * DICE_P[t->number];
        }
    }

    for (int r = 0; r < 5; r++) {
        total += res_prod[r];
        if (res_prod[r] > 0) variety++;
    }

    *prod_out = total;
    *variety_out = variety;
}

double base_value_fn(Game *g, Color p0_color) {
    State *s = &g->state;
    int idx = s->color_to_index[(int)p0_color];

    double p0_prod; int p0_var;
    compute_production(s, p0_color, &p0_prod, &p0_var);
    double production = p0_prod + p0_var * VARIETY_BONUS;

    /* Find enemy (first non-self color) */
    Color enemy = COLOR_NONE;
    for (int i = 0; i < s->num_players; i++) {
        if (s->colors[i] != p0_color) { enemy = s->colors[i]; break; }
    }
    double e_prod; int e_var;
    if (enemy != COLOR_NONE) {
        compute_production(s, enemy, &e_prod, &e_var);
    } else {
        e_prod = 0; e_var = 0;
    }

    int *ps = s->player_state[idx];
    int lr = ps[PS_LONGEST_ROAD_LENGTH];

    int wheat = ps[PS_WHEAT_IN_HAND], ore = ps[PS_ORE_IN_HAND];
    int sheep = ps[PS_SHEEP_IN_HAND], brick = ps[PS_BRICK_IN_HAND], wood = ps[PS_WOOD_IN_HAND];
    double d_city = (fmax(2-wheat,0) + fmax(3-ore,0)) / 5.0;
    double d_settle = (fmax(1-wheat,0) + fmax(1-sheep,0) + fmax(1-brick,0) + fmax(1-wood,0)) / 4.0;
    double hand_synergy = (2 - d_city - d_settle) / 2.0;

    int num_in_hand = wood + brick + sheep + wheat + ore;

    /* Owned tiles */
    bool tile_seen[NUM_LAND_TILES] = {false};
    int num_tiles = 0;
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < s->board.map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = s->board.map->adjacent_tiles[node][ti];
            if (!tile_seen[tile_idx]) { tile_seen[tile_idx] = true; num_tiles++; }
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < s->board.map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = s->board.map->adjacent_tiles[node][ti];
            if (!tile_seen[tile_idx]) { tile_seen[tile_idx] = true; num_tiles++; }
        }
    }

    /* Count buildable nodes via bitset popcount (avoids iterating 96 nodes) */
    uint64_t reachable[2] = {0, 0};
    for (int i = 0; i < s->board.cc_count[(int)p0_color]; i++)
        bs_or(reachable, reachable, s->board.cc_sets[(int)p0_color][i]);
    uint64_t avail[2];
    bs_and(avail, reachable, s->board.buildable);
    int num_buildable = __builtin_popcountll(avail[0]) + __builtin_popcountll(avail[1]);
    double lr_factor = (num_buildable == 0) ? W_ROAD : 0.1;

    int num_devs = ps[PS_KNIGHT_IN_HAND] + ps[PS_YEAR_OF_PLENTY_IN_HAND]
                 + ps[PS_MONOPOLY_IN_HAND] + ps[PS_ROAD_BUILDING_IN_HAND]
                 + ps[PS_VICTORY_POINT_IN_HAND];
    int army = ps[PS_PLAYED_KNIGHT];

    return (double)(
        ps[PS_VICTORY_POINTS] * W_VPS
        + production * W_PROD
        + e_prod * W_EPROD
        + hand_synergy * W_SYNERGY
        + num_buildable * W_BUILDABLE
        + num_tiles * W_TILES
        + num_in_hand * W_HAND
        + (num_in_hand > 7 ? W_DISCARD : 0)
        + lr * lr_factor
        + num_devs * W_DEVS
        + army * W_ARMY
    );
}
