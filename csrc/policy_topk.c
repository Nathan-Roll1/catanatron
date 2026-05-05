/* policy_topk.c — Pure C policy callback for deep_search.
 *
 * Replaces the Python policy callback with a single C function that:
 *   1. Encodes game state to nf/ef/ff buffers (state_encode.c)
 *   2. Builds the action mask from legal actions
 *   3. Runs nn_forward (libnn)
 *   4. Sorts legal actions by their policy logit
 *   5. Returns top-K indices (positions in `actions` array)
 *
 * This eliminates ALL Python overhead in the deep_search inner loop.
 */

#include "policy_topk.h"
#include "actions.h"
#include "state_encode.h"
#include "value.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Action type enum values (must match catan_types.h) */
#define AT_ROLL                  0
#define AT_MOVE_ROBBER           1
#define AT_DISCARD_RESOURCE      2
#define AT_BUILD_ROAD            3
#define AT_BUILD_SETTLEMENT      4
#define AT_BUILD_CITY            5
#define AT_BUY_DEVELOPMENT_CARD  6
#define AT_PLAY_KNIGHT_CARD      7
#define AT_PLAY_YEAR_OF_PLENTY   8
#define AT_PLAY_MONOPOLY         9
#define AT_PLAY_ROAD_BUILDING   10
#define AT_MARITIME_TRADE       11
#define AT_OFFER_TRADE          12
#define AT_ACCEPT_TRADE         13
#define AT_REJECT_TRADE         14
#define AT_CONFIRM_TRADE        15
#define AT_CANCEL_TRADE         16
#define AT_END_TURN             17

/* Policy-space slot starts (must match Python ActionEncoder constants) */
#define SLOT_ROLL          0
#define SLOT_END_TURN      1
#define SLOT_BUY_DEV       2
#define SLOT_KNIGHT        3
#define SLOT_ROAD_BUILDING 4
#define SLOT_SETTLEMENT    5     /* + node_to_compact[v[0]] */
#define SLOT_CITY         59     /* + node_to_compact[v[0]] */
#define SLOT_ROAD        113     /* + edge_lut[min(v[0],v[1])][max(v[0],v[1])] */
#define SLOT_ROBBER      185     /* + tile*5 + steal_idx */
#define SLOT_DISCARD     280     /* + v[0] */
#define SLOT_YOP         285     /* + yop_lut (unused for top-K, see note) */
#define SLOT_MONOPOLY    305     /* + v[0] */
#define SLOT_MARITIME    310     /* + mar_lut[v[0]][v[4]] */
#define SLOT_ACCEPT      330
#define SLOT_REJECT      331
#define SLOT_CANCEL      332
#define SLOT_CONFIRM     333     /* + v[4] (color) */

#define NO_STEAL_IDX 4
#define POLICY_AD 337   /* size of policy logit space (NN AD) */

/* Encode action to flat policy-space index. Returns -1 if unencodable. */
int policy_action_encode(const NNModel *m, const Action *a) {
    int t = a->type;
    const int32_t *v = a->value;

    /* Hot path: most common types first */
    if (t == AT_BUILD_SETTLEMENT) {
        int compact = m->node_to_compact[v[0]];
        if (compact < 0) return -1;
        return SLOT_SETTLEMENT + compact;
    }
    if (t == AT_BUILD_ROAD) {
        int a0 = v[0], a1 = v[1];
        int lo = a0 < a1 ? a0 : a1;
        int hi = a0 < a1 ? a1 : a0;
        if (lo < 0 || hi >= 96) return -1;
        int e = m->edge_lut[lo][hi];
        if (e < 0) return -1;
        return SLOT_ROAD + e;
    }
    if (t == AT_BUILD_CITY) {
        int compact = m->node_to_compact[v[0]];
        if (compact < 0) return -1;
        return SLOT_CITY + compact;
    }
    if (t == AT_ROLL) return SLOT_ROLL;
    if (t == AT_END_TURN) return SLOT_END_TURN;
    if (t == AT_MOVE_ROBBER) {
        int x = v[0], y = v[1], z = v[2];
        /* coord_to_tile is offset by +3 in each axis */
        if (x < -3 || x > 3 || y < -3 || y > 3 || z < -3 || z > 3) return -1;
        int tile = m->coord_to_tile[x + 3][y + 3][z + 3];
        if (tile < 0) return -1;
        int steal = (v[3] >= 0) ? v[3] : NO_STEAL_IDX;
        return SLOT_ROBBER + tile * 5 + steal;
    }
    if (t == AT_BUY_DEVELOPMENT_CARD) return SLOT_BUY_DEV;
    if (t == AT_MARITIME_TRADE) {
        int give = v[0], recv = v[4];
        if (give < 0 || give >= 5 || recv < 0 || recv >= 5) return -1;
        int idx = m->mar_lut[give][recv];
        if (idx < 0) return -1;
        return SLOT_MARITIME + idx;
    }
    if (t == AT_PLAY_KNIGHT_CARD) return SLOT_KNIGHT;
    if (t == AT_PLAY_ROAD_BUILDING) return SLOT_ROAD_BUILDING;
    if (t == AT_DISCARD_RESOURCE) {
        if (v[0] < 0 || v[0] >= 5) return -1;
        return SLOT_DISCARD + v[0];
    }
    if (t == AT_PLAY_MONOPOLY) {
        if (v[0] < 0 || v[0] >= 5) return -1;
        return SLOT_MONOPOLY + v[0];
    }
    if (t == AT_PLAY_YEAR_OF_PLENTY) {
        /* YOP encoding requires a YOP-pairs LUT not in NNModel.
         * This is rare and unlikely to be in top-K. Skip. */
        return -1;
    }
    if (t == AT_ACCEPT_TRADE) return SLOT_ACCEPT;
    if (t == AT_REJECT_TRADE) return SLOT_REJECT;
    if (t == AT_CANCEL_TRADE) return SLOT_CANCEL;
    if (t == AT_CONFIRM_TRADE) {
        if (v[4] < 0 || v[4] >= 4) return -1;
        return SLOT_CONFIRM + v[4];
    }
    return -1;  /* AT_OFFER_TRADE and unknowns */
}

/* nn_policy_only signature (declared here, defined in nn.c) */
extern void nn_policy_only(const NNModel *m,
                           const float node_feat[NN_NODES][NN_NODE_FEAT],
                           const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                           const float flat_feat[NN_FLAT_DIM],
                           const float mask[NN_MASK_DIM],
                           float policy_out[NN_MASK_DIM]);

typedef struct {
    float logit;
    int idx;
} ScoredAction;

typedef struct {
    int checked;
    int enabled;
    float bias[POLICY_AD];
    float *weights;  /* [POLICY_AD][ENC_FLAT_FEAT_DIM] */
} HeuristicPolicy;

static HeuristicPolicy g_hpol = {0, 0, {0}, NULL};

typedef struct {
    int checked;
    int enabled;
    int use_value_tiebreak;
    int variant;
    int flags;
} AlgoPolicy;

static AlgoPolicy g_apol = {0, 0, 0, 0, 0};

static void algo_policy_add_flag(int variant) {
    if (variant > 0 && variant <= 5) g_apol.flags |= (1 << variant);
}

static int algo_policy_has_flag(int variant) {
    return (g_apol.flags & (1 << variant)) != 0;
}

void policy_algo_configure(int flags, int use_value_tiebreak) {
    g_apol.checked = 1;
    g_apol.enabled = 1;
    g_apol.variant = 0;
    g_apol.flags = flags;
    g_apol.use_value_tiebreak = use_value_tiebreak ? 1 : 0;
}

static void algo_policy_try_enable(void) {
    if (g_apol.checked) return;
    g_apol.checked = 1;
    const char *mode = getenv("CATAN_POLICY_ALGO");
    if (mode && mode[0] && mode[0] != '0') {
        g_apol.enabled = 1;
        const char *vb = getenv("CATAN_POLICY_ALGO_VALUE");
        g_apol.use_value_tiebreak = (vb && vb[0] && vb[0] != '0');
        const char *var = getenv("CATAN_POLICY_ALGO_VARIANT");
        if (var && var[0]) {
            g_apol.variant = atoi(var);
            if (g_apol.variant < 0 || g_apol.variant > 5) g_apol.variant = 0;
            algo_policy_add_flag(g_apol.variant);
        }
        const char *flags = getenv("CATAN_POLICY_ALGO_FLAGS");
        if (flags && flags[0]) {
            if (strchr(flags, ',')) {
                const char *p = flags;
                while (*p) {
                    algo_policy_add_flag(atoi(p));
                    while (*p && *p != ',') p++;
                    if (*p == ',') p++;
                }
            } else {
                int mask = atoi(flags);
                for (int i = 1; i <= 5; i++) {
                    if (mask & (1 << i)) algo_policy_add_flag(i);
                }
            }
        }
    }
}

static void heuristic_policy_try_load(void) {
    if (g_hpol.checked) return;
    g_hpol.checked = 1;

    const char *path = getenv("CATAN_POLICY_HEURISTIC");
    if (!path || !path[0]) return;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "policy heuristic: failed to open %s\n", path);
        return;
    }

    char magic[4];
    uint32_t ver = 0, fd = 0, ad = 0;
    if (fread(magic, 1, 4, f) != 4 ||
        fread(&ver, sizeof(uint32_t), 1, f) != 1 ||
        fread(&fd, sizeof(uint32_t), 1, f) != 1 ||
        fread(&ad, sizeof(uint32_t), 1, f) != 1 ||
        memcmp(magic, "HPOL", 4) != 0 ||
        ver != 1 || fd != ENC_FLAT_FEAT_DIM || ad != POLICY_AD) {
        fprintf(stderr, "policy heuristic: bad header in %s\n", path);
        fclose(f);
        return;
    }

    float *weights = (float *)malloc((size_t)POLICY_AD * ENC_FLAT_FEAT_DIM * sizeof(float));
    if (!weights) {
        fclose(f);
        return;
    }
    size_t nb = fread(g_hpol.bias, sizeof(float), POLICY_AD, f);
    size_t nw = fread(weights, sizeof(float), (size_t)POLICY_AD * ENC_FLAT_FEAT_DIM, f);
    fclose(f);
    if (nb != POLICY_AD || nw != (size_t)POLICY_AD * ENC_FLAT_FEAT_DIM) {
        fprintf(stderr, "policy heuristic: truncated file %s\n", path);
        free(weights);
        return;
    }

    g_hpol.weights = weights;
    g_hpol.enabled = 1;
    fprintf(stderr, "policy heuristic: loaded %s\n", path);
}

static inline float heuristic_policy_score(int pidx, const float *ff) {
    if (pidx < 0 || pidx >= POLICY_AD || !g_hpol.enabled) return -1e9f;
    const float *w = g_hpol.weights + (size_t)pidx * ENC_FLAT_FEAT_DIM;
    float s = g_hpol.bias[pidx];
    for (int i = 0; i < ENC_FLAT_FEAT_DIM; i++) s += w[i] * ff[i];
    return s;
}

static inline double dice_p(const CatanMap *map, int number) {
    if (number < 2 || number > 12) return 0.0;
    return map->dice_probas[number];
}

static int current_idx(const State *s) {
    return s->current_player_index;
}

static double resource_need_score(const State *s, int pidx, int r) {
    const int *ps = s->player_state[pidx];
    int have = ps[PS_RESOURCE_IN_HAND(r)];
    double need = 0.0;
    if (r == RES_WHEAT) need += (ps[PS_WHEAT_IN_HAND] < 2) ? 2.2 : 0.5;
    if (r == RES_ORE)   need += (ps[PS_ORE_IN_HAND]   < 3) ? 2.0 : 0.4;
    if (r == RES_SHEEP) need += (ps[PS_SHEEP_IN_HAND] < 1) ? 1.5 : 0.2;
    if (r == RES_WOOD)  need += (ps[PS_WOOD_IN_HAND]  < 1) ? 1.2 : 0.1;
    if (r == RES_BRICK) need += (ps[PS_BRICK_IN_HAND] < 1) ? 1.2 : 0.1;
    if (have == 0) need += 0.8;
    if (have >= 3) need -= 0.5 * (have - 2);
    return need;
}

static double node_prod_score(const Game *g, int node, Color color) {
    const State *s = &g->state;
    const CatanMap *map = s->board.map;
    double prod = 0.0, variety = 0.0;
    int seen[NUM_RESOURCES] = {0};
    for (int i = 0; i < map->adjacent_tiles_count[node]; i++) {
        int ti = map->adjacent_tiles[node][i];
        const LandTile *t = &map->land_tiles[ti];
        if (t->resource == RES_NONE || t->number == 0) continue;
        if (coord_eq(map->land_tile_coords[ti], s->board.robber_coordinate)) continue;
        double p = dice_p(map, t->number);
        prod += p;
        if (!seen[(int)t->resource]) {
            seen[(int)t->resource] = 1;
            variety += 0.015;
        }
    }

    bool ports[6] = {false};
    board_get_player_port_resources((Board *)&s->board, color, ports);
    double port_bonus = 0.0;
    for (int r = 0; r < 5; r++) {
        if (ports[r]) port_bonus += seen[r] ? 0.05 : 0.015;
    }
    if (ports[5]) port_bonus += 0.025;
    return prod * 100.0 + variety * 100.0 + port_bonus * 100.0;
}

static double node_need_prod_score(const Game *g, int node, Color color) {
    const State *s = &g->state;
    const CatanMap *map = s->board.map;
    int pi = s->color_to_index[(int)color];
    double prod = 0.0, need_prod = 0.0;
    int seen[NUM_RESOURCES] = {0};
    for (int i = 0; i < map->adjacent_tiles_count[node]; i++) {
        int ti = map->adjacent_tiles[node][i];
        const LandTile *t = &map->land_tiles[ti];
        if (t->resource == RES_NONE || t->number == 0) continue;
        if (coord_eq(map->land_tile_coords[ti], s->board.robber_coordinate)) continue;
        int r = (int)t->resource;
        double p = dice_p(map, t->number);
        prod += p;
        need_prod += p * resource_need_score(s, pi, r);
        seen[r] = 1;
    }
    double diversity = 0.0;
    for (int r = 0; r < NUM_RESOURCES; r++) {
        if (seen[r] && s->player_state[pi][PS_RESOURCE_IN_HAND(r)] == 0) diversity += 0.025;
    }
    return prod * 85.0 + need_prod * 65.0 + diversity * 100.0;
}

static int hand_total_for_player(const State *s, int pi) {
    int total = 0;
    for (int r = 0; r < NUM_RESOURCES; r++) total += s->player_state[pi][PS_RESOURCE_IN_HAND(r)];
    return total;
}

static int best_enemy_vp(const State *s, int pi) {
    int best = 0;
    for (int i = 0; i < s->num_players; i++) {
        if (i == pi) continue;
        int vp = s->player_state[i][PS_VICTORY_POINTS];
        if (vp > best) best = vp;
    }
    return best;
}

static int best_enemy_road(const State *s, int pi) {
    int best = 0;
    for (int i = 0; i < s->num_players; i++) {
        if (i == pi) continue;
        int lr = s->player_state[i][PS_LONGEST_ROAD_LENGTH];
        if (lr > best) best = lr;
    }
    return best;
}

static int can_afford(const State *s, int pi, const int cost[5]) {
    for (int r = 0; r < NUM_RESOURCES; r++) {
        if (s->player_state[pi][PS_RESOURCE_IN_HAND(r)] < cost[r]) return 0;
    }
    return 1;
}

static int tile_index_from_coord(const State *s, int x, int y, int z) {
    const CatanMap *map = s->board.map;
    for (int i = 0; i < map->num_land_tiles; i++) {
        Coordinate c = map->land_tile_coords[i];
        if (c.x == x && c.y == y && c.z == z) return i;
    }
    return -1;
}

static double robber_score(const Game *g, const Action *a, Color color) {
    const State *s = &g->state;
    const CatanMap *map = s->board.map;
    int ti = tile_index_from_coord(s, a->value[0], a->value[1], a->value[2]);
    if (ti < 0) return 0.0;
    const LandTile *t = &map->land_tiles[ti];
    double p = dice_p(map, t->number);
    double score = 0.0;
    for (int j = 0; j < 6; j++) {
        int node = t->nodes[j];
        Color nc = board_get_node_color((Board *)&s->board, node);
        if (nc == COLOR_NONE) continue;
        double mult = board_get_node_building((Board *)&s->board, node) == BLD_CITY ? 2.0 : 1.0;
        if (nc == color) score -= 140.0 * p * mult;
        else {
            int ni = s->color_to_index[(int)nc];
            int vp = (ni >= 0) ? s->player_state[ni][PS_VICTORY_POINTS] : 0;
            score += 170.0 * p * mult + 2.5 * vp;
        }
    }
    if (a->value[3] >= 0 && a->value[3] < MAX_PLAYERS) {
        int steal_idx = s->color_to_index[a->value[3]];
        if (steal_idx >= 0) {
            score += 0.4 * player_num_resources(s, steal_idx);
            score += 1.5 * s->player_state[steal_idx][PS_VICTORY_POINTS];
        }
    }
    return score;
}

static double road_score(const Game *g, const Action *a, Color color) {
    const State *s = &g->state;
    int a0 = a->value[0], a1 = a->value[1];
    double score = 3.0;
    if (bs_test(s->board.buildable, a0)) score += node_prod_score(g, a0, color) * 0.35;
    if (bs_test(s->board.buildable, a1)) score += node_prod_score(g, a1, color) * 0.35;
    int idx = s->color_to_index[(int)color];
    if (idx >= 0) {
        score += 0.8 * s->player_state[idx][PS_LONGEST_ROAD_LENGTH];
        if (s->player_state[idx][PS_ROADS_AVAILABLE] <= 2) score -= 5.0;
    }
    return score;
}

static double road_connectivity_score(const Game *g, const Action *a, Color color) {
    const State *s = &g->state;
    double score = 0.0;
    for (int end = 0; end < 2; end++) {
        int node = a->value[end];
        int own_adj = 0, empty_adj = 0;
        for (int ai = 0; ai < STATIC_ADJ_COUNT[node]; ai++) {
            int nb = STATIC_ADJ[node][ai];
            int rai = board_adj_index(node, nb);
            if (rai < 0) continue;
            int owner = s->board.road_owner[node][rai];
            if (owner == color) own_adj++;
            if (owner == COLOR_NONE) empty_adj++;
        }
        score += 4.0 * own_adj + 1.5 * empty_adj;
        if (bs_test(s->board.buildable, node)) score += 0.5 * node_prod_score(g, node, color);
    }
    return score;
}

static double algo_base_score(const Game *g, const Action *a) {
    const State *s = &g->state;
    Color color = state_current_color(s);
    int pi = current_idx(s);
    const int *ps = s->player_state[pi];
    int vp = ps[PS_VICTORY_POINTS];

    if (a->type == AT_ROLL) return 100000.0;
    if (a->type == AT_ACCEPT_TRADE) return -20.0;
    if (a->type == AT_REJECT_TRADE) return 10.0;
    if (a->type == AT_CANCEL_TRADE) return 8.0;
    if (a->type == AT_CONFIRM_TRADE) return 5.0;

    double score = 0.0;
    switch (a->type) {
    case AT_BUILD_CITY:
        score = 900.0 + 6.0 * node_prod_score(g, a->value[0], color);
        if (algo_policy_has_flag(1)) score += 4.0 * node_need_prod_score(g, a->value[0], color);
        if (vp >= g->vps_to_win - 1) score += 100000.0;
        break;
    case AT_BUILD_SETTLEMENT:
        score = 820.0 + 5.0 * node_prod_score(g, a->value[0], color);
        if (algo_policy_has_flag(1)) score += 5.0 * node_need_prod_score(g, a->value[0], color);
        if (s->is_initial_build_phase) score += 220.0;
        if (vp >= g->vps_to_win - 1) score += 100000.0;
        break;
    case AT_BUILD_ROAD:
        score = 380.0 + road_score(g, a, color);
        if (algo_policy_has_flag(4)) {
            int own_lr = ps[PS_LONGEST_ROAD_LENGTH];
            int enemy_lr = best_enemy_road(s, pi);
            score += road_connectivity_score(g, a, color);
            if (own_lr >= 4 && own_lr + 1 >= enemy_lr) score += 65.0;
            if (ps[PS_ROADS_AVAILABLE] <= 3) score -= 35.0;
        }
        if (s->is_initial_build_phase) score += 120.0;
        break;
    case AT_BUY_DEVELOPMENT_CARD:
        score = 520.0 + 12.0 * vp + 4.0 * ps[PS_ORE_IN_HAND] + 3.0 * ps[PS_WHEAT_IN_HAND];
        if (ps[PS_SHEEP_IN_HAND] > 1 && ps[PS_WHEAT_IN_HAND] > 1 && ps[PS_ORE_IN_HAND] > 1) score += 60.0;
        if (algo_policy_has_flag(3)) {
            if (vp >= 7) score += 110.0;
            if (ps[PS_PLAYED_KNIGHT] >= 2) score += 80.0;
            if (vp < 4 && can_afford(s, pi, SETTLEMENT_COST)) score -= 140.0;
            if (vp < 5 && can_afford(s, pi, CITY_COST)) score -= 100.0;
            if (s->dev_deck_size <= 5) score -= 45.0;
        }
        break;
    case AT_PLAY_KNIGHT_CARD:
        score = 610.0 + 16.0 * ps[PS_PLAYED_KNIGHT] + robber_score(g, a, color);
        if (algo_policy_has_flag(2)) score += 6.0 * best_enemy_vp(s, pi);
        break;
    case AT_PLAY_ROAD_BUILDING:
        score = 570.0 + 20.0 * (15 - ps[PS_ROADS_AVAILABLE]);
        break;
    case AT_PLAY_YEAR_OF_PLENTY:
        score = 540.0 + 50.0 * resource_need_score(s, pi, a->value[0])
                    + 50.0 * resource_need_score(s, pi, a->value[1]);
        break;
    case AT_PLAY_MONOPOLY: {
        int r = a->value[0];
        int enemy_have = 0;
        for (int i = 0; i < s->num_players; i++) {
            if (i == pi) continue;
            enemy_have += s->player_state[i][PS_RESOURCE_IN_HAND(r)];
        }
        score = 500.0 + 42.0 * enemy_have + 20.0 * resource_need_score(s, pi, r);
        break;
    }
    case AT_MARITIME_TRADE:
        score = 250.0 + 70.0 * resource_need_score(s, pi, a->value[4])
                    - 25.0 * resource_need_score(s, pi, a->value[0])
                    + 3.0 * ps[PS_RESOURCE_IN_HAND(a->value[0])];
        if (algo_policy_has_flag(5)) {
            int give = a->value[0], recv = a->value[4];
            int hand = hand_total_for_player(s, pi);
            score += 55.0 * resource_need_score(s, pi, recv);
            score -= 45.0 * resource_need_score(s, pi, give);
            if (hand > 7) score += 18.0 * (hand - 7);
            if (ps[PS_RESOURCE_IN_HAND(give)] <= 2) score -= 35.0;
        }
        break;
    case AT_MOVE_ROBBER:
        score = 650.0 + robber_score(g, a, color);
        if (algo_policy_has_flag(2)) {
            int steal_idx = (a->value[3] >= 0 && a->value[3] < MAX_PLAYERS)
                ? s->color_to_index[a->value[3]]
                : -1;
            if (steal_idx >= 0) {
                int leader = best_enemy_vp(s, pi);
                if (s->player_state[steal_idx][PS_VICTORY_POINTS] >= leader) score += 55.0;
                score += 8.0 * s->player_state[steal_idx][PS_VICTORY_POINTS];
            }
        }
        break;
    case AT_DISCARD_RESOURCE: {
        int r = a->value[0];
        score = 100.0 + 35.0 * ps[PS_RESOURCE_IN_HAND(r)]
                    - 45.0 * resource_need_score(s, pi, r);
        break;
    }
    case AT_END_TURN:
        score = -100.0;
        if (ps[PS_HAS_ROLLED]) score += 200.0;
        if (algo_policy_has_flag(5)) {
            int hand = hand_total_for_player(s, pi);
            if (hand > 7) score -= 35.0 * (hand - 7);
            if (can_afford(s, pi, CITY_COST) || can_afford(s, pi, SETTLEMENT_COST)) score -= 120.0;
            if (can_afford(s, pi, DEV_CARD_COST) && vp >= 7) score -= 70.0;
        }
        break;
    default:
        score = -1000.0;
        break;
    }

    if (g_apol.use_value_tiebreak &&
        (a->type == AT_BUILD_CITY || a->type == AT_BUILD_SETTLEMENT ||
         a->type == AT_BUILD_ROAD || a->type == AT_BUY_DEVELOPMENT_CARD)) {
        Game child;
        Action tmp[MAX_ACTIONS];
        int ntmp = 0;
        game_copy(&child, g);
        game_execute(&child, *a, tmp, &ntmp);
        double v = base_value_fn(&child, color) * 1e-8;
        if (game_winning_color(&child) == color) v += 100000.0;
        score += v;
    }

    return score;
}

/* Sort descending by logit (qsort comparator) */
static int scored_cmp_desc(const void *a, const void *b) {
    const ScoredAction *sa = (const ScoredAction *)a;
    const ScoredAction *sb = (const ScoredAction *)b;
    if (sa->logit > sb->logit) return -1;
    if (sa->logit < sb->logit) return 1;
    return sa->idx - sb->idx;  /* stable for ties: lower idx first */
}

/* Insertion sort top-K: for small K (<=16) much faster than full qsort.
 * Maintains a max-heap of size k with min at top, replacing it when a
 * better candidate arrives. Then sort the final heap descending. */
static int top_k_select(const ScoredAction *all, int n, int k,
                         ScoredAction *out_sorted) {
    if (n == 0) return 0;
    if (k > n) k = n;

    /* For small k, use a simple heap. Push first k, then for each remaining,
     * if better than min, replace and re-heapify. At end, sort descending. */
    /* Simple approach: copy first k, then for each remaining, check against
     * the current minimum in the heap. Use linear-min for tiny k. */

    /* Copy first k */
    for (int i = 0; i < k; i++) out_sorted[i] = all[i];
    /* Find min in out_sorted[0..k] */
    int min_idx = 0;
    for (int i = 1; i < k; i++) {
        if (out_sorted[i].logit < out_sorted[min_idx].logit) min_idx = i;
    }
    /* For each remaining */
    for (int i = k; i < n; i++) {
        if (all[i].logit > out_sorted[min_idx].logit) {
            out_sorted[min_idx] = all[i];
            /* Recompute min */
            min_idx = 0;
            for (int j = 1; j < k; j++) {
                if (out_sorted[j].logit < out_sorted[min_idx].logit) min_idx = j;
            }
        }
    }
    /* Sort the top-k descending */
    qsort(out_sorted, k, sizeof(ScoredAction), scored_cmp_desc);
    return k;
}

static void force_include_action_type(ScoredAction *selected, int *n_top, int k,
                                      const ScoredAction *scored,
                                      const Action *actions, int n_scored,
                                      ActionType type) {
    int wanted = -1;
    for (int i = 0; i < n_scored; i++) {
        if (actions[scored[i].idx].type == type) {
            wanted = i;
            break;
        }
    }
    if (wanted < 0) return;

    int action_idx = scored[wanted].idx;
    for (int i = 0; i < *n_top; i++) {
        if (selected[i].idx == action_idx) return;
    }

    if (*n_top < k) {
        selected[*n_top] = scored[wanted];
        (*n_top)++;
    } else if (*n_top > 0) {
        selected[*n_top - 1] = scored[wanted];
    }
    qsort(selected, *n_top, sizeof(ScoredAction), scored_cmp_desc);
}

int policy_top_k_ex(const StateEncoderC *enc, const NNModel *m,
                    const Game *g, const Action *actions, int n_actions,
                    int k, int *out_indices,
                    float *nf, float *ef, float *ff, float *mk, float *out,
                    int use_algo_policy) {
    if (n_actions <= 0 || k <= 0) return 0;
    algo_policy_try_enable();
    heuristic_policy_try_load();

    if (use_algo_policy) {
        ScoredAction scored[256];
        int n_scored = n_actions < 256 ? n_actions : 256;
        for (int i = 0; i < n_scored; i++) {
            scored[i].logit = (float)algo_base_score(g, &actions[i]);
            scored[i].idx = i;
        }
        if (k > n_scored) k = n_scored;
        if (k > 64) k = 64;
        ScoredAction selected[64];
        int n_top = top_k_select(scored, n_scored, k, selected);
        force_include_action_type(selected, &n_top, k, scored, actions, n_scored, AT_END_TURN);
        force_include_action_type(selected, &n_top, k, scored, actions, n_scored, AT_REJECT_TRADE);
        force_include_action_type(selected, &n_top, k, scored, actions, n_scored, AT_CANCEL_TRADE);
        for (int i = 0; i < n_top; i++) out_indices[i] = selected[i].idx;
        return n_top;
    }

    /* 1. Encode state */
    encode_state(enc, g, nf, ef, ff);

    /* 2. Build action mask from legal actions, tracking each action's
     *    policy index alongside. We compute scored entries directly. */
    memset(mk, 0, NN_MASK_DIM * sizeof(float));

    ScoredAction scored[256];  /* MAX_ACTIONS = 128 in catanatron */
    int n_scored = 0;
    int policy_idx_per_action[256];  /* policy index per action (-1 if unencodable) */
    for (int i = 0; i < n_actions && i < 256; i++) {
        int pidx = policy_action_encode(m, &actions[i]);
        policy_idx_per_action[i] = pidx;
        if (pidx >= 0 && pidx < NN_MASK_DIM) {
            mk[pidx] = 1.0f;
        }
    }

    NNOutput *nn_out = (NNOutput *)out;
    if (!g_hpol.enabled) {
        /* 3. Run policy-only NN forward. Top-k does not use value logits. */
        nn_policy_only(m,
                       (const float (*)[NN_NODE_FEAT])nf,
                       (const float (*)[NN_EDGE_FEAT])ef,
                       ff, mk, nn_out->policy);
    }

    /* 4. Score each legal action by its policy logit */
    for (int i = 0; i < n_actions; i++) {
        int pidx = policy_idx_per_action[i];
        if (g_hpol.enabled) {
            scored[n_scored].logit = heuristic_policy_score(pidx, ff);
        } else if (pidx >= 0 && pidx < POLICY_AD) {
            scored[n_scored].logit = nn_out->policy[pidx];
        } else {
            scored[n_scored].logit = -1e9f;
        }
        scored[n_scored].idx = i;
        n_scored++;
    }

    /* 5. Top-K selection */
    if (k > n_scored) k = n_scored;
    if (k > 64) k = 64;
    ScoredAction selected[64];
    int n_top = top_k_select(scored, n_scored, k, selected);
    for (int i = 0; i < n_top; i++) {
        out_indices[i] = selected[i].idx;
    }
    return n_top;
}

int policy_top_k(const StateEncoderC *enc, const NNModel *m,
                 const Game *g, const Action *actions, int n_actions,
                 int k, int *out_indices,
                 float *nf, float *ef, float *ff, float *mk, float *out) {
    algo_policy_try_enable();
    return policy_top_k_ex(enc, m, g, actions, n_actions, k, out_indices,
                           nf, ef, ff, mk, out, g_apol.enabled);
}
