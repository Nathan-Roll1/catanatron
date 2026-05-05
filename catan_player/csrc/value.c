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

static const double W2_CITY_DIST     = 1.15e7;
static const double W2_SETTLE_DIST   = 9.0e6;
static const double W2_DEV_DIST      = 8.0e6;
static const double W2_ROAD_DIST     = 4.0e6;
static const double W2_ROLL_PROGRESS = 2.4e7;
static const double W2_DEV_TACTICAL  = 2.5e6;
static const double W2_RACE          = 4.0e6;
static const double W2_THREAT        = 3e14;
static const double W2_TERMINAL      = 1.0e18;

typedef struct {
    double score;
    double threat;
} KnownFutureEval;

typedef struct {
    double progress;
    double city_ready;
    double settle_ready;
    double dev_ready;
    double road_ready;
} RollProgressEval;

static inline int imax(int a, int b) { return a > b ? a : b; }
static inline int imin(int a, int b) { return a < b ? a : b; }
static inline double dmax0(double x) { return x > 0.0 ? x : 0.0; }

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

static void compute_resource_production(State *s, Color color, double out[5]) {
    Board *b = &s->board;
    CatanMap *map = b->map;
    Coordinate robber = b->robber_coordinate;
    int idx = s->color_to_index[(int)color];

    for (int r = 0; r < 5; r++) out[r] = 0.0;

    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[tile_idx];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
            out[(int)t->resource] += DICE_P[t->number];
        }
    }

    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[tile_idx];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
            out[(int)t->resource] += 2.0 * DICE_P[t->number];
        }
    }
}

static void production_summary(const double prod_res[5], double *total_out, int *variety_out) {
    double total = 0.0;
    int variety = 0;
    for (int r = 0; r < 5; r++) {
        total += prod_res[r];
        if (prod_res[r] > 0.0) variety++;
    }
    *total_out = total;
    *variety_out = variety;
}

static int count_owned_tiles(State *s, int idx) {
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
    return num_tiles;
}

static int count_buildable_nodes(State *s, Color color) {
    uint64_t reachable[2] = {0, 0};
    for (int i = 0; i < s->board.cc_count[(int)color]; i++)
        bs_or(reachable, reachable, s->board.cc_sets[(int)color][i]);
    uint64_t avail[2];
    bs_and(avail, reachable, s->board.buildable);
    return __builtin_popcountll(avail[0]) + __builtin_popcountll(avail[1]);
}

static int count_buildable_edges(State *s, Color color) {
    int edges[MAX_ROAD_EDGES][2];
    return board_buildable_edges(&s->board, color, edges, MAX_ROAD_EDGES);
}

static void trade_rates_for_player(State *s, Color color, int rates[5], bool ports[6]) {
    for (int r = 0; r < 5; r++) rates[r] = 4;
    board_get_player_port_resources(&s->board, color, ports);
    if (ports[5]) {
        for (int r = 0; r < 5; r++) rates[r] = 3;
    }
    for (int r = 0; r < 5; r++) {
        if (ports[r]) rates[r] = 2;
    }
}

static int conversion_distance(const int hand[5], const int cost[5], const int rates[5]) {
    int deficit = 0;
    int trade_capacity = 0;
    for (int r = 0; r < 5; r++) {
        int missing = cost[r] - hand[r];
        if (missing > 0) deficit += missing;
        int surplus = hand[r] - cost[r];
        if (surplus > 0) trade_capacity += surplus / rates[r];
    }
    return deficit - imin(deficit, trade_capacity);
}

static int player_known_vp(State *s, int idx) {
    int *ps = s->player_state[idx];
    int public_vp = ps[PS_VICTORY_POINTS];
    int actual_vp = ps[PS_ACTUAL_VICTORY_POINTS];
    int public_plus_vp_cards = public_vp + ps[PS_VICTORY_POINT_IN_HAND];
    return imax(actual_vp, imax(public_vp, public_plus_vp_cards));
}

static bool player_can_play_dev_now(State *s, int idx, DevCardType dev) {
    int *ps = s->player_state[idx];
    if (ps[PS_HAS_PLAYED_DEV_CARD_IN_TURN]) return false;
    if (ps[PS_DEV_IN_HAND(dev)] <= 0) return false;

    int owned_at_start_field;
    switch (dev) {
        case DEV_KNIGHT:         owned_at_start_field = PS_KNIGHT_OWNED_AT_START; break;
        case DEV_MONOPOLY:       owned_at_start_field = PS_MONOPOLY_OWNED_AT_START; break;
        case DEV_YEAR_OF_PLENTY: owned_at_start_field = PS_YEAR_OF_PLENTY_OWNED_AT_START; break;
        case DEV_ROAD_BUILDING:  owned_at_start_field = PS_ROAD_BUILDING_OWNED_AT_START; break;
        default: return false;
    }
    return ps[owned_at_start_field] > 0;
}

static int yop_distance_help(State *s, const int hand[5], const int cost[5]) {
    int help = 0;
    for (int r = 0; r < 5; r++) {
        int missing = cost[r] - hand[r];
        if (missing > 0 && s->resource_freqdeck[r] > 0)
            help += imin(missing, s->resource_freqdeck[r]);
    }
    return imin(help, 2);
}

static int monopoly_distance_help(State *s, int idx, const int hand[5], const int cost[5]) {
    int best = 0;
    for (int r = 0; r < 5; r++) {
        int missing = cost[r] - hand[r];
        if (missing <= 0) continue;
        int held_by_others = 0;
        for (int p = 0; p < s->num_players; p++) {
            if (p == idx) continue;
            held_by_others += s->player_state[p][PS_RESOURCE_IN_HAND(r)];
        }
        best = imax(best, imin(missing, held_by_others));
    }
    return best;
}

static int assisted_distance(State *s, int idx, const int hand[5], const int cost[5],
                             const int rates[5], bool yop_now, bool monopoly_now) {
    int dist = conversion_distance(hand, cost, rates);
    if (yop_now) dist = imax(0, dist - yop_distance_help(s, hand, cost));
    if (monopoly_now) dist = imax(0, dist - monopoly_distance_help(s, idx, hand, cost));
    return dist;
}

static void dev_deck_counts(State *s, int counts[5]) {
    for (int d = 0; d < 5; d++) counts[d] = 0;
    for (int i = 0; i < s->dev_deck_size; i++) {
        int d = s->development_listdeck[i];
        if (d >= 0 && d < 5) counts[d]++;
    }
}

static void compute_roll_payouts(State *s, int number, int payout[MAX_PLAYERS][5]) {
    Board *b = &s->board;
    CatanMap *map = b->map;
    int totals[5] = {0, 0, 0, 0, 0};

    for (int p = 0; p < MAX_PLAYERS; p++)
        for (int r = 0; r < 5; r++)
            payout[p][r] = 0;

    if (number == 7) return;

    for (int t = 0; t < map->num_land_tiles; t++) {
        LandTile *tile = &map->land_tiles[t];
        if (tile->number != number || tile->resource == RES_NONE) continue;
        if (coord_eq(map->land_tile_coords[t], b->robber_coordinate)) continue;

        int res = (int)tile->resource;
        for (int ni = 0; ni < 6; ni++) {
            int node = tile->nodes[ni];
            if (b->buildings[node] < 0) continue;
            Color nc = (Color)(b->buildings[node] >> 2);
            BuildingType bt = (BuildingType)(b->buildings[node] & 3);
            int ci = s->color_to_index[(int)nc];
            int amount = (bt == BLD_CITY) ? 2 : 1;
            payout[ci][res] += amount;
            totals[res] += amount;
        }
    }

    for (int r = 0; r < 5; r++) {
        if (totals[r] <= s->resource_freqdeck[r]) continue;
        for (int p = 0; p < MAX_PLAYERS; p++)
            payout[p][r] = 0;
    }
}

static int peek_next_roll_sum(const Game *g) {
    RngState rng = g->rng;
    int d1 = rng_randint(&rng, 1, 6);
    int d2 = rng_randint(&rng, 1, 6);
    return d1 + d2;
}

static int exact_roll_sum_for_eval(const Game *g, int idx) {
    const State *s = &g->state;
    if (idx < 0 || idx >= s->num_players) return -1;
    if (s->current_prompt != PROMPT_PLAY_TURN) return -1;
    if (s->current_player_index != idx) return -1;
    if (s->player_state[idx][PS_HAS_ROLLED]) return -1;
    return peek_next_roll_sum(g);
}

static RollProgressEval roll_progress(State *s, int idx, const int hand[5],
                                       const int rates[5],
                                       bool can_city, int city_dist,
                                       bool can_settle, int settle_dist,
                                       bool can_dev, int dev_dist,
                                       bool can_road, int road_dist,
                                       int exact_sum) {
    RollProgressEval out = {0.0, 0.0, 0.0, 0.0, 0.0};
    int payout[MAX_PLAYERS][5];

    int first_sum = (exact_sum >= 2 && exact_sum <= 12) ? exact_sum : 2;
    int last_sum = (exact_sum >= 2 && exact_sum <= 12) ? exact_sum : 12;
    for (int sum = first_sum; sum <= last_sum; sum++) {
        compute_roll_payouts(s, sum, payout);
        int next_hand[5];
        for (int r = 0; r < 5; r++)
            next_hand[r] = hand[r] + payout[idx][r];

        double p = (exact_sum >= 2 && exact_sum <= 12) ? 1.0 : DICE_P[sum];
        if (can_city) {
            int nd = conversion_distance(next_hand, CITY_COST, rates);
            out.progress += p * 1.30 * dmax0((double)(city_dist - nd));
            if (nd == 0) out.city_ready += p;
        }
        if (can_settle) {
            int nd = conversion_distance(next_hand, SETTLEMENT_COST, rates);
            out.progress += p * 1.00 * dmax0((double)(settle_dist - nd));
            if (nd == 0) out.settle_ready += p;
        }
        if (can_dev) {
            int nd = conversion_distance(next_hand, DEV_CARD_COST, rates);
            out.progress += p * 0.70 * dmax0((double)(dev_dist - nd));
            if (nd == 0) out.dev_ready += p;
        }
        if (can_road) {
            int nd = conversion_distance(next_hand, ROAD_COST, rates);
            out.progress += p * 0.40 * dmax0((double)(road_dist - nd));
            if (nd == 0) out.road_ready += p;
        }
    }
    return out;
}

static KnownFutureEval known_future_player_eval(Game *g, int idx,
                                                const int dev_counts[5],
                                                bool use_exact_roll) {
    State *s = &g->state;
    Color color = s->colors[idx];
    int *ps = s->player_state[idx];

    double prod_res[5];
    compute_resource_production(s, color, prod_res);
    double prod_total;
    int prod_variety;
    production_summary(prod_res, &prod_total, &prod_variety);
    double production = prod_total + prod_variety * VARIETY_BONUS;

    int hand[5];
    player_get_hand(s, idx, hand);
    int num_in_hand = hand[0] + hand[1] + hand[2] + hand[3] + hand[4];

    bool ports[6];
    int rates[5];
    trade_rates_for_player(s, color, rates, ports);

    int num_tiles = count_owned_tiles(s, idx);
    int num_buildable = count_buildable_nodes(s, color);
    int num_road_edges = count_buildable_edges(s, color);

    bool can_city_target = ps[PS_CITIES_AVAILABLE] > 0 && s->settlement_count[idx] > 0;
    bool has_settlement_spot = s->is_initial_build_phase || num_buildable > 0;
    bool can_settle_target = ps[PS_SETTLEMENTS_AVAILABLE] > 0;
    bool can_dev_target = s->dev_deck_size > 0;
    bool can_road_target = ps[PS_ROADS_AVAILABLE] > 0 && num_road_edges > 0;

    bool yop_now = player_can_play_dev_now(s, idx, DEV_YEAR_OF_PLENTY);
    bool monopoly_now = player_can_play_dev_now(s, idx, DEV_MONOPOLY);
    bool knight_now = player_can_play_dev_now(s, idx, DEV_KNIGHT);
    bool road_building_now = player_can_play_dev_now(s, idx, DEV_ROAD_BUILDING);

    int city_dist = can_city_target
        ? conversion_distance(hand, CITY_COST, rates) : 99;
    int settle_dist = can_settle_target
        ? conversion_distance(hand, SETTLEMENT_COST, rates) : 99;
    int dev_dist = can_dev_target
        ? conversion_distance(hand, DEV_CARD_COST, rates) : 99;
    int road_dist = can_road_target
        ? conversion_distance(hand, ROAD_COST, rates) : 99;

    int city_dist_assisted = can_city_target
        ? assisted_distance(s, idx, hand, CITY_COST, rates, yop_now, monopoly_now) : 99;
    int settle_dist_assisted = can_settle_target
        ? assisted_distance(s, idx, hand, SETTLEMENT_COST, rates, yop_now, monopoly_now) : 99;
    int dev_dist_assisted = can_dev_target
        ? assisted_distance(s, idx, hand, DEV_CARD_COST, rates, yop_now, monopoly_now) : 99;
    int road_dist_assisted = can_road_target
        ? assisted_distance(s, idx, hand, ROAD_COST, rates, yop_now, monopoly_now) : 99;

    double score = 0.0;
    int known_vp = player_known_vp(s, idx);
    score += (double)known_vp * W_VPS;
    score += production * W_PROD;
    score += num_buildable * W_BUILDABLE;
    score += num_tiles * W_TILES;
    score += num_in_hand * W_HAND;
    score += (num_in_hand > 7 ? W_DISCARD : 0.0);

    if (can_city_target)
        score += (5 - imin(city_dist_assisted, 5)) * W2_CITY_DIST;
    if (can_settle_target) {
        double place_factor = has_settlement_spot ? 1.0 : (num_road_edges > 0 ? 0.35 : 0.0);
        score += place_factor * (4 - imin(settle_dist_assisted, 4)) * W2_SETTLE_DIST;
    }
    if (can_dev_target)
        score += (3 - imin(dev_dist_assisted, 3)) * W2_DEV_DIST;
    if (can_road_target)
        score += (2 - imin(road_dist_assisted, 2)) * W2_ROAD_DIST;

    int exact_sum = use_exact_roll ? exact_roll_sum_for_eval(g, idx) : -1;
    RollProgressEval roll_eval = roll_progress(
        s, idx, hand, rates,
        can_city_target, city_dist,
        can_settle_target && has_settlement_spot, settle_dist,
        can_dev_target, dev_dist,
        can_road_target, road_dist,
        exact_sum);
    double roll_scale = 0.35;
    if (state_current_color(s) == color &&
        s->current_prompt == PROMPT_PLAY_TURN &&
        !ps[PS_HAS_ROLLED]) {
        roll_scale = 1.10;
    }
    score += roll_eval.progress * roll_scale * W2_ROLL_PROGRESS;

    int num_devs = player_num_devs(s, idx);
    score += num_devs * W_DEVS;
    score += ps[PS_KNIGHT_IN_HAND] * W2_DEV_TACTICAL;
    score += ps[PS_YEAR_OF_PLENTY_IN_HAND] * (W2_DEV_TACTICAL * 1.30);
    score += ps[PS_MONOPOLY_IN_HAND] * (W2_DEV_TACTICAL * 1.15);
    score += ps[PS_ROAD_BUILDING_IN_HAND] * (can_road_target ? W2_DEV_TACTICAL * 1.25
                                                            : W2_DEV_TACTICAL * 0.40);

    double vp_dev_prob = 0.0;
    if (s->dev_deck_size > 0)
        vp_dev_prob = (double)dev_counts[DEV_VICTORY_POINT] / (double)s->dev_deck_size;
    if (can_dev_target && dev_dist_assisted == 0)
        score += vp_dev_prob * 5.0e7;

    int army_holder = -1;
    int army_holder_size = 0;
    for (int p = 0; p < s->num_players; p++) {
        if (s->player_state[p][PS_HAS_ARMY]) {
            army_holder = p;
            army_holder_size = s->player_state[p][PS_PLAYED_KNIGHT];
            break;
        }
    }
    int army_goal = 3;
    if (army_holder >= 0 && army_holder != idx)
        army_goal = army_holder_size + 1;
    int future_knights = ps[PS_PLAYED_KNIGHT] + ps[PS_KNIGHT_IN_HAND];
    int army_gap = ps[PS_HAS_ARMY] ? 0 : imax(0, army_goal - future_knights);
    score += ps[PS_PLAYED_KNIGHT] * W_ARMY;
    score += (5 - imin(army_gap, 5)) * W2_RACE;
    if (ps[PS_HAS_ARMY]) score += 6.0e6;
    bool can_take_army_now = !ps[PS_HAS_ARMY] && knight_now &&
                             ps[PS_PLAYED_KNIGHT] + 1 >= army_goal;

    int lr = ps[PS_LONGEST_ROAD_LENGTH];
    int road_goal = ps[PS_HAS_ROAD] ? 0 : imax(5, s->board.road_length + 1);
    int road_now_gain = (can_road_target && road_dist_assisted == 0) ? 1 : 0;
    int rb_gain = road_building_now ? imin(2, ps[PS_ROADS_AVAILABLE]) : 0;
    int road_gap = ps[PS_HAS_ROAD] ? 0 : imax(0, road_goal - (lr + road_now_gain + rb_gain));
    score += lr * W_ROAD;
    score += (5 - imin(road_gap, 5)) * (W2_RACE * 0.90);
    if (ps[PS_HAS_ROAD]) score += 6.0e6;
    bool can_take_road_now = !ps[PS_HAS_ROAD] &&
                             (lr + road_now_gain + rb_gain) >= road_goal;

    int target_vp = s->vps_to_win > 0 ? s->vps_to_win : g->vps_to_win;
    if (target_vp <= 0) target_vp = 10;

    bool can_gain_vp_now =
        (can_city_target && city_dist_assisted == 0) ||
        (can_settle_target && has_settlement_spot && settle_dist_assisted == 0);
    bool can_roll_into_vp =
        (roll_eval.city_ready > 0.0 && can_city_target) ||
        (roll_eval.settle_ready > 0.0 && can_settle_target && has_settlement_spot);

    double threat = 0.0;
    if (known_vp >= target_vp) {
        threat += 2.0 * W2_THREAT;
    } else {
        if (known_vp + 1 >= target_vp && can_gain_vp_now)
            threat += 0.70 * W2_THREAT;
        if (known_vp + 2 >= target_vp && (can_take_army_now || can_take_road_now))
            threat += 0.70 * W2_THREAT;
        if (known_vp + 1 >= target_vp && can_dev_target && dev_dist_assisted == 0)
            threat += 0.25 * vp_dev_prob * W2_THREAT;
        if (state_current_color(s) == color && !ps[PS_HAS_ROLLED] &&
            known_vp + 1 >= target_vp && can_roll_into_vp) {
            double roll_win_p = roll_eval.city_ready + roll_eval.settle_ready;
            if (roll_win_p > 1.0) roll_win_p = 1.0;
            threat += 0.30 * roll_win_p * W2_THREAT;
        }
        if (known_vp >= target_vp - 1)
            threat += 0.06 * W2_THREAT;
        if (known_vp >= target_vp - 2 && (can_take_army_now || can_take_road_now))
            threat += 0.12 * W2_THREAT;
    }
    score += threat;

    KnownFutureEval out = {score, threat};
    return out;
}

static double opponent_feature_value(Game *g, int idx) {
    State *s = &g->state;
    Color c = s->colors[idx];
    double prod;
    int var;
    compute_production(s, c, &prod, &var);
    double production = prod + var * VARIETY_BONUS;

    int *ps = s->player_state[idx];
    int wheat = ps[PS_WHEAT_IN_HAND], ore = ps[PS_ORE_IN_HAND];
    int sheep = ps[PS_SHEEP_IN_HAND], brick = ps[PS_BRICK_IN_HAND], wood = ps[PS_WOOD_IN_HAND];
    double d_city = (fmax(2-wheat,0) + fmax(3-ore,0)) / 5.0;
    double d_settle = (fmax(1-wheat,0) + fmax(1-sheep,0) + fmax(1-brick,0) + fmax(1-wood,0)) / 4.0;
    double hand_synergy = (2 - d_city - d_settle) / 2.0;
    int num_in_hand = wood + brick + sheep + wheat + ore;
    int num_devs = ps[PS_KNIGHT_IN_HAND] + ps[PS_YEAR_OF_PLENTY_IN_HAND]
                 + ps[PS_MONOPOLY_IN_HAND] + ps[PS_ROAD_BUILDING_IN_HAND]
                 + ps[PS_VICTORY_POINT_IN_HAND];

    return (double)(
        ps[PS_VICTORY_POINTS] * W_VPS
        + production * W_PROD
        + hand_synergy * W_SYNERGY
        + num_in_hand * W_HAND
        + (num_in_hand > 7 ? W_DISCARD : 0)
        + ps[PS_LONGEST_ROAD_LENGTH] * W_ROAD
        + num_devs * W_DEVS
        + ps[PS_PLAYED_KNIGHT] * W_ARMY
    );
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

double base_value_fn_enemy_full(Game *g, Color p0_color) {
    double v = base_value_fn(g, p0_color);
    State *s = &g->state;
    double pressure = 0.0;
    for (int i = 0; i < s->num_players; i++) {
        if (s->colors[i] == p0_color) continue;
        pressure += opponent_feature_value(g, i);
    }
    return v - 0.10 * pressure;
}

double base_value_fn_known_future(Game *g, Color p0_color) {
    return base_value_fn_known_future_exact(g, p0_color, false);
}

double base_value_fn_known_future_exact(Game *g, Color p0_color, bool use_exact_roll) {
    return base_value_fn_known_future_profile(g, p0_color, use_exact_roll, 0);
}

double base_value_fn_known_future_profile(Game *g, Color p0_color,
                                          bool use_exact_roll,
                                          int profile) {
    Color winner = game_winning_color(g);
    if (winner == p0_color) return W2_TERMINAL;
    if (winner != COLOR_NONE) return -W2_TERMINAL;

    State *s = &g->state;
    int idx = s->color_to_index[(int)p0_color];
    int deck_counts[5];
    dev_deck_counts(s, deck_counts);

    KnownFutureEval self = known_future_player_eval(g, idx, deck_counts,
                                                    use_exact_roll);

    double opp_sum = 0.0;
    double opp_leader = 0.0;
    double opp_threat_sum = 0.0;
    double opp_threat_leader = 0.0;
    int opp_count = 0;
    int self_known_vp = player_known_vp(s, idx);
    int opp_leader_vp = 0;

    for (int i = 0; i < s->num_players; i++) {
        if (i == idx) continue;
        KnownFutureEval opp = known_future_player_eval(g, i, deck_counts,
                                                       use_exact_roll);
        int opp_vp = player_known_vp(s, i);
        opp_sum += opp.score;
        opp_threat_sum += opp.threat;
        if (opp_count == 0 || opp.score > opp_leader) {
            opp_leader = opp.score;
            opp_leader_vp = opp_vp;
        }
        if (opp_count == 0 || opp.threat > opp_threat_leader)
            opp_threat_leader = opp.threat;
        opp_count++;
    }

    if (opp_count == 0) return self.score;

    double sum_w = 0.18;
    double leader_w = 0.22;
    double threat_leader_w = 0.70;
    double threat_rest_w = 0.20;
    if (profile == 1) {
        sum_w = 0.12;
        leader_w = 0.36;
        threat_leader_w = 0.85;
        threat_rest_w = 0.18;
    } else if (profile == 2) {
        sum_w = 0.18;
        leader_w = 0.22;
        threat_leader_w = 1.10;
        threat_rest_w = 0.30;
    } else if (profile == 3) {
        sum_w = 0.08;
        leader_w = 0.12;
        threat_leader_w = 0.50;
        threat_rest_w = 0.10;
    } else if (profile == 4) {
        sum_w = 0.15;
        leader_w = 0.30;
        threat_leader_w = 0.78;
        threat_rest_w = 0.18;
    } else if (profile == 5) {
        sum_w = 0.12;
        leader_w = 0.30;
        threat_leader_w = 0.70;
        threat_rest_w = 0.20;
    } else if (profile == 6) {
        int target_vp = s->vps_to_win > 0 ? s->vps_to_win : g->vps_to_win;
        if (target_vp <= 0) target_vp = 10;
        bool urgent_leader = opp_leader_vp >= target_vp - 2 ||
                             opp_leader_vp >= self_known_vp + 2;
        if (urgent_leader) {
            sum_w = 0.12;
            leader_w = 0.36;
            threat_leader_w = 0.85;
            threat_rest_w = 0.18;
        }
    } else if (profile == 7) {
        int target_vp = s->vps_to_win > 0 ? s->vps_to_win : g->vps_to_win;
        if (target_vp <= 0) target_vp = 10;
        bool urgent_leader = opp_leader_vp >= target_vp - 2 ||
                             opp_leader_vp >= self_known_vp + 2;
        bool self_close = self_known_vp >= target_vp - 2;
        bool early_engine = self_known_vp <= 5 && opp_leader_vp <= 5;
        if (urgent_leader) {
            sum_w = 0.12;
            leader_w = 0.36;
            threat_leader_w = 0.85;
            threat_rest_w = 0.18;
        } else if (self_close) {
            sum_w = 0.08;
            leader_w = 0.18;
            threat_leader_w = 0.55;
            threat_rest_w = 0.10;
        } else if (early_engine) {
            sum_w = 0.06;
            leader_w = 0.10;
            threat_leader_w = 0.42;
            threat_rest_w = 0.06;
        } else {
            sum_w = 0.15;
            leader_w = 0.30;
            threat_leader_w = 0.78;
            threat_rest_w = 0.18;
        }
    } else if (profile == 8) {
        int target_vp = s->vps_to_win > 0 ? s->vps_to_win : g->vps_to_win;
        if (target_vp <= 0) target_vp = 10;
        bool urgent_leader = opp_leader_vp >= target_vp - 2 ||
                             opp_leader_vp >= self_known_vp + 2;
        if (urgent_leader) {
            sum_w = 0.10;
            leader_w = 0.48;
            threat_leader_w = 0.95;
            threat_rest_w = 0.12;
        } else {
            sum_w = 0.10;
            leader_w = 0.42;
            threat_leader_w = 0.82;
            threat_rest_w = 0.14;
        }
    }

    double pressure = sum_w * opp_sum + leader_w * opp_leader;
    double threat_pressure = threat_leader_w * opp_threat_leader
                           + threat_rest_w * (opp_threat_sum - opp_threat_leader);
    return self.score - pressure - threat_pressure;
}
