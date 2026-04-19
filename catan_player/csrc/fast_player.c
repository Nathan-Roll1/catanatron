/*
 * fast_player.c — Pure C Catan player: NN policy + AB2 value search.
 *
 * ABt30 search: neural network policy selects top-5 candidates, each is
 * rolled out 30 plies with policy argmax, then base_value_fn scores the leaf.
 *
 * Build:
 *   cc -O3 -march=native -flto -Icsrc csrc/fast_player.c csrc/nn.c \
 *      csrc/rng.c csrc/map.c csrc/board.c csrc/state.c csrc/actions.c \
 *      csrc/apply_action.c csrc/game.c csrc/value.c csrc/search.c \
 *      -o catan_player -lm [-framework Accelerate]
 *
 * Usage:
 *   ./catan_player                      # default: ABt30, seed 42
 *   ./catan_player --seed 777           # custom seed
 *   ./catan_player --depth 20           # shallower search
 *   ./catan_player --games 100          # multi-game benchmark
 *   ./catan_player --verbose            # print every action
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <libgen.h>

#include "nn.h"
#include "game.h"
#include "actions.h"
#include "value.h"
#include "catan_types.h"

static const float DICE_PROB[13] = {
    0, 0,
    1.0f/36, 2.0f/36, 3.0f/36, 4.0f/36, 5.0f/36, 6.0f/36,
    5.0f/36, 4.0f/36, 3.0f/36, 2.0f/36, 1.0f/36
};

static const char *RES_NAMES[5] = {"Wood", "Brick", "Sheep", "Wheat", "Ore"};

/*
 * Action index layout (337 total, must match Python ActionEncoder):
 *   0       ROLL
 *   1       END_TURN
 *   2       BUY_DEV
 *   3       KNIGHT
 *   4       ROAD_BUILDING
 *   5-58    SETTLEMENT (54 compact nodes)
 *   59-112  CITY (54 compact nodes)
 *   113-184 ROAD (72 edges)
 *   185-279 MOVE_ROBBER (19 tiles x 5: slots 0-3=steal player, 4=no steal)
 *   280-284 DISCARD (5 resources)
 *   285-304 YEAR_OF_PLENTY (15 ordered pairs + 5 half-picks)
 *   305-309 MONOPOLY (5 resources)
 *   310-329 MARITIME (20 give/receive pairs)
 *   330-336 Trade accept/reject/cancel/confirm (unused by policy)
 */

/* YoP pair LUT: (r1,r2) with r1<=r2 -> pair index 0-14 */
static const int YOP_LUT[5][5] = {
    { 0,  1,  2,  3,  4},
    { 1,  5,  6,  7,  8},
    { 2,  6,  9, 10, 11},
    { 3,  7, 10, 12, 13},
    { 4,  8, 11, 13, 14},
};

#define FEAT_PER_PLAYER 24
#define FEAT_BANK       6
#define FEAT_PHASE      13
#define AD              337


/* ── State Encoding ─────────────────────────────────────────────── */

static void encode_state(const Game *g, const NNModel *m,
                         float nf[NN_NODES][NN_NODE_FEAT],
                         float ef[NN_MAX_EDGES][NN_EDGE_FEAT],
                         float ff[NN_FLAT_DIM]) {
    const State *s = &g->state;
    int cp = s->current_player_index;

    memset(nf, 0, sizeof(float) * NN_NODES * NN_NODE_FEAT);
    memset(ef, 0, sizeof(float) * NN_MAX_EDGES * NN_EDGE_FEAT);
    memset(ff, 0, sizeof(float) * NN_FLAT_DIM);

    for (int i = 0; i < NN_NODES; i++) {
        int gid = m->land_nodes[i];
        int8_t bld = s->board.buildings[gid];
        if (bld >= 0) {
            int color = bld >> 2, type = bld & 0x3;
            int seat = s->color_to_index[color];
            int own = (seat == cp);
            if (own && type == 0)       nf[i][1] = 1.0f;
            else if (own && type == 1)  nf[i][2] = 1.0f;
            else if (!own && type == 0) nf[i][3] = 1.0f;
            else if (!own && type == 1) nf[i][4] = 1.0f;
        } else {
            nf[i][0] = 1.0f;
        }
    }

    for (int ti = 0; ti < 19; ti++) {
        int res = s->board.map->land_tiles[ti].resource;
        int num = s->board.map->land_tiles[ti].number;
        if (res < 0 || num <= 0 || num > 12) continue;
        float prob = DICE_PROB[num];
        for (int ni = 0; ni < 6; ni++) {
            int lid = m->node_to_compact[m->tile_nodes[ti][ni]];
            if (lid >= 0 && lid < NN_NODES)
                nf[lid][5 + res] += prob;
        }
    }

    for (int i = 0; i < NN_NODES; i++) nf[i][10] = 1.0f;
    for (int pi = 0; pi < s->board.map->num_ports; pi++) {
        const Port *port = &s->board.map->ports[pi];
        int ptype = (port->resource < 0) ? 5 : port->resource;
        for (int ni = 0; ni < 6; ni++) {
            int lid = m->node_to_compact[port->nodes[ni]];
            if (lid >= 0 && lid < NN_NODES) {
                nf[lid][10] = 0.0f;
                nf[lid][11 + ptype] = 1.0f;
            }
        }
    }

    Coordinate rc = s->board.robber_coordinate;
    for (int ti = 0; ti < 19; ti++) {
        Coordinate tc = s->board.map->land_tile_coords[ti];
        if (tc.x == rc.x && tc.y == rc.y && tc.z == rc.z) {
            for (int ni = 0; ni < 6; ni++) {
                int lid = m->node_to_compact[m->tile_nodes[ti][ni]];
                if (lid >= 0 && lid < NN_NODES) nf[lid][17] = 1.0f;
            }
            break;
        }
    }

    int ne = m->num_edges;
    for (int e = 0; e < ne; e++) {
        int src_g = m->land_nodes[m->edge_src[e]];
        int dst_g = m->land_nodes[m->edge_dst[e]];
        int adj_idx = -1;
        for (int j = 0; j < MAX_DEGREE; j++)
            if (STATIC_ADJ[src_g][j] == dst_g) { adj_idx = j; break; }
        if (adj_idx < 0) { ef[e][0] = 1.0f; continue; }
        int8_t ro = s->board.road_owner[src_g][adj_idx];
        if (ro < 0) {
            ef[e][0] = 1.0f;
        } else {
            int seat = s->color_to_index[(int)ro];
            if (seat == cp) ef[e][1] = 1.0f;
            else ef[e][1 + ((seat - cp + 4) % 4)] = 1.0f;
        }
    }

    int o = 0;
    for (int p = 0; p < 4; p++) {
        int seat = (cp + p) % 4;
        const int *ps = s->player_state[seat];
        ff[o]    = ps[PS_VICTORY_POINTS] / 10.0f;
        for (int r = 0; r < 5; r++) ff[o+1+r] = ps[PS_WOOD_IN_HAND+r] / 19.0f;
        for (int d = 0; d < 5; d++) ff[o+6+d] = ps[PS_KNIGHT_IN_HAND+d] / 14.0f;
        for (int d = 0; d < 5; d++) ff[o+11+d] = (float)ps[PS_PLAYED_KNIGHT+d];
        ff[o+16] = (float)ps[PS_HAS_ROAD];
        ff[o+17] = (float)ps[PS_HAS_ARMY];
        ff[o+18] = (float)ps[PS_HAS_ROLLED];
        ff[o+19] = (float)ps[PS_HAS_PLAYED_DEV_CARD_IN_TURN];
        ff[o+20] = ps[PS_ROADS_AVAILABLE] / 15.0f;
        ff[o+21] = ps[PS_SETTLEMENTS_AVAILABLE] / 5.0f;
        ff[o+22] = ps[PS_CITIES_AVAILABLE] / 4.0f;
        ff[o+23] = ps[PS_LONGEST_ROAD_LENGTH] / 15.0f;
        o += FEAT_PER_PLAYER;
    }

    int o1 = 4 * FEAT_PER_PLAYER;
    for (int r = 0; r < 5; r++) ff[o1+r] = s->resource_freqdeck[r] / 19.0f;
    ff[o1+5] = s->dev_deck_size / 25.0f;

    int o2 = o1 + FEAT_BANK;
    if (s->current_prompt >= 0 && s->current_prompt < 7)
        ff[o2 + s->current_prompt] = 1.0f;
    ff[o2+7]  = s->is_initial_build_phase ? 1.0f : 0.0f;
    ff[o2+8]  = s->is_discarding ? 1.0f : 0.0f;
    ff[o2+9]  = s->is_road_building ? 1.0f : 0.0f;
    ff[o2+10] = s->is_moving_knight ? 1.0f : 0.0f;
    ff[o2+11] = s->is_resolving_trade ? 1.0f : 0.0f;
    ff[o2+12] = s->num_turns / 1000.0f;
}


/* ── Action Encoding (must match Python ActionEncoder) ──────────── */

static int action_to_idx(const NNModel *m, Action a) {
    switch (a.type) {
    case AT_ROLL:                return 0;
    case AT_END_TURN:            return 1;
    case AT_BUY_DEVELOPMENT_CARD: return 2;
    case AT_PLAY_KNIGHT_CARD:    return 3;
    case AT_PLAY_ROAD_BUILDING:  return 4;
    case AT_BUILD_SETTLEMENT: {
        int lid = m->node_to_compact[a.value[0]];
        return (lid >= 0) ? 5 + lid : -1;
    }
    case AT_BUILD_CITY: {
        int lid = m->node_to_compact[a.value[0]];
        return (lid >= 0) ? 59 + lid : -1;
    }
    case AT_BUILD_ROAD: {
        int e = m->edge_lut[a.value[0]][a.value[1]];
        return (e >= 0) ? 113 + e : -1;
    }
    case AT_MOVE_ROBBER: {
        int x = a.value[0]+3, y = a.value[1]+3, z = a.value[2]+3;
        if (x < 0 || x >= 7 || y < 0 || y >= 7 || z < 0 || z >= 7) return -1;
        int ti = m->coord_to_tile[x][y][z];
        if (ti < 0) return -1;
        int si = (a.value[3] < 0) ? 4 : a.value[3];
        return 185 + ti * 5 + si;
    }
    case AT_DISCARD_RESOURCE:
        return (a.value[0] >= 0 && a.value[0] < 5) ? 280 + a.value[0] : -1;
    case AT_PLAY_YEAR_OF_PLENTY: {
        int r1 = a.value[0], r2 = a.value[1];
        if (r1 < 0 || r1 >= 5) return -1;
        if (r2 < 0) return 285 + 15 + r1;
        if (r2 >= 5) return -1;
        return 285 + YOP_LUT[r1][r2];
    }
    case AT_PLAY_MONOPOLY:
        return (a.value[0] >= 0 && a.value[0] < 5) ? 305 + a.value[0] : -1;
    case AT_MARITIME_TRADE: {
        int give = a.value[0], get = a.value[4];
        if (give < 0 || give >= 5 || get < 0 || get >= 5) return -1;
        int mi = m->mar_lut[give][get];
        return (mi >= 0) ? 310 + mi : -1;
    }
    default: return -1;
    }
}

static void encode_action_mask(const NNModel *m, const Action *le, int n_le,
                               float mask[NN_MASK_DIM]) {
    memset(mask, 0, sizeof(float) * NN_MASK_DIM);
    for (int i = 0; i < n_le; i++) {
        int idx = action_to_idx(m, le[i]);
        if (idx >= 0 && idx < NN_MASK_DIM) mask[idx] = 1.0f;
    }
}


/* ── Action Formatting ──────────────────────────────────────────── */

static const char *action_name(int type) {
    switch (type) {
    case AT_ROLL: return "ROLL";
    case AT_MOVE_ROBBER: return "ROBBER";
    case AT_DISCARD_RESOURCE: return "DISCARD";
    case AT_BUILD_ROAD: return "ROAD";
    case AT_BUILD_SETTLEMENT: return "SETTLEMENT";
    case AT_BUILD_CITY: return "CITY";
    case AT_BUY_DEVELOPMENT_CARD: return "BUY_DEV";
    case AT_PLAY_KNIGHT_CARD: return "KNIGHT";
    case AT_PLAY_YEAR_OF_PLENTY: return "YEAR_OF_PLENTY";
    case AT_PLAY_MONOPOLY: return "MONOPOLY";
    case AT_PLAY_ROAD_BUILDING: return "ROAD_BUILDING";
    case AT_MARITIME_TRADE: return "MARITIME";
    case AT_END_TURN: return "END_TURN";
    default: return "UNKNOWN";
    }
}

static void format_action(char *buf, size_t sz, Action a) {
    switch (a.type) {
    case AT_BUILD_SETTLEMENT:
    case AT_BUILD_CITY:
        snprintf(buf, sz, "%s(node=%d)", action_name(a.type), a.value[0]);
        break;
    case AT_BUILD_ROAD:
        snprintf(buf, sz, "ROAD(%d-%d)", a.value[0], a.value[1]);
        break;
    case AT_MOVE_ROBBER:
        if (a.value[3] >= 0)
            snprintf(buf, sz, "ROBBER((%d,%d,%d),steal_P%d)",
                     a.value[0], a.value[1], a.value[2], a.value[3]);
        else
            snprintf(buf, sz, "ROBBER((%d,%d,%d),no_steal)",
                     a.value[0], a.value[1], a.value[2]);
        break;
    case AT_MARITIME_TRADE: {
        const char *give = (a.value[0] >= 0 && a.value[0] < 5) ? RES_NAMES[a.value[0]] : "?";
        const char *get  = (a.value[4] >= 0 && a.value[4] < 5) ? RES_NAMES[a.value[4]] : "?";
        snprintf(buf, sz, "MARITIME(%s->%s)", give, get);
        break;
    }
    case AT_DISCARD_RESOURCE:
        snprintf(buf, sz, "DISCARD(%s)",
                 (a.value[0] >= 0 && a.value[0] < 5) ? RES_NAMES[a.value[0]] : "?");
        break;
    default:
        snprintf(buf, sz, "%s", action_name(a.type));
        break;
    }
}

static bool is_interesting(int type) {
    return type == AT_BUILD_SETTLEMENT || type == AT_BUILD_CITY ||
           type == AT_BUILD_ROAD || type == AT_BUY_DEVELOPMENT_CARD ||
           type == AT_MOVE_ROBBER || type == AT_PLAY_KNIGHT_CARD ||
           type == AT_MARITIME_TRADE;
}


/* ── Search Heuristics ──────────────────────────────────────────── */

static float action_bonus(Action a) {
    switch (a.type) {
    case AT_BUILD_CITY:            return 1.0f;
    case AT_BUILD_SETTLEMENT:      return 0.4f;
    case AT_BUILD_ROAD:            return 0.05f;
    case AT_BUY_DEVELOPMENT_CARD:  return 0.1f;
    default: return 0.0f;
    }
}

static int fix_robber_steal(int chosen, const Action *le, int n_le) {
    Action act = le[chosen];
    if (act.type != AT_MOVE_ROBBER || act.value[3] >= 0)
        return chosen;
    int tx = act.value[0], ty = act.value[1], tz = act.value[2];
    for (int i = 0; i < n_le; i++) {
        if (le[i].type == AT_MOVE_ROBBER && le[i].value[3] >= 0 &&
            le[i].value[0] == tx && le[i].value[1] == ty && le[i].value[2] == tz)
            return i;
    }
    for (int i = 0; i < n_le; i++) {
        if (le[i].type == AT_MOVE_ROBBER && le[i].value[3] >= 0)
            return i;
    }
    return chosen;
}


/* ── Search ─────────────────────────────────────────────────────── */

static void policy_step(const NNModel *m, Game *gc,
                        float nf_buf[NN_NODES][NN_NODE_FEAT],
                        float ef_buf[NN_MAX_EDGES][NN_EDGE_FEAT],
                        float ff_buf[NN_FLAT_DIM],
                        float mk_buf[NN_MASK_DIM]) {
    Action acts[MAX_ACTIONS];
    int n = generate_playable_actions(&gc->state, acts, MAX_ACTIONS);
    if (n == 0) return;
    if (n == 1) { int dn; game_execute(gc, acts[0], acts, &dn); return; }

    encode_state(gc, m, nf_buf, ef_buf, ff_buf);
    encode_action_mask(m, acts, n, mk_buf);
    NNOutput out;
    nn_forward(m, nf_buf, ef_buf, ff_buf, mk_buf, &out);

    int best_i = 0;
    float best_v = -1e30f;
    for (int i = 0; i < n; i++) {
        int idx = action_to_idx(m, acts[i]);
        float v = (idx >= 0 && idx < AD) ? out.policy[idx] : -1e30f;
        if (v > best_v) { best_v = v; best_i = i; }
    }
    int dn;
    game_execute(gc, acts[best_i], acts, &dn);
}

static int abt_search(const NNModel *m, Game *g,
                      const Action *le, int n_le, int depth, int top_k,
                      float nf_buf[NN_NODES][NN_NODE_FEAT],
                      float ef_buf[NN_MAX_EDGES][NN_EDGE_FEAT],
                      float ff_buf[NN_FLAT_DIM],
                      float mk_buf[NN_MASK_DIM]) {
    int seat = g->state.current_player_index;
    Color seat_color = g->state.colors[seat];

    encode_state(g, m, nf_buf, ef_buf, ff_buf);
    encode_action_mask(m, le, n_le, mk_buf);
    NNOutput out;
    nn_forward(m, nf_buf, ef_buf, ff_buf, mk_buf, &out);

    float scores[MAX_ACTIONS];
    for (int i = 0; i < n_le; i++) {
        int idx = action_to_idx(m, le[i]);
        scores[i] = (idx >= 0 && idx < AD) ? out.policy[idx] : -1e30f;
    }

    int cands[MAX_ACTIONS];
    int n_cands = n_le;
    for (int i = 0; i < n_le; i++) cands[i] = i;

    if (n_le > top_k && depth >= 2) {
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < n_le; j++)
                if (scores[cands[j]] > scores[cands[best]]) best = j;
            int tmp = cands[i]; cands[i] = cands[best]; cands[best] = tmp;
        }
        n_cands = top_k;
    }

    int best_pos = 0;
    float best_val = -1e30f;

    for (int p = 0; p < n_cands; p++) {
        int ci = cands[p];
        Game gc;
        game_copy(&gc, g);
        Action tmp_acts[MAX_ACTIONS];
        int tmp_n;
        game_execute(&gc, le[ci], tmp_acts, &tmp_n);

        for (int ply = 2; ply <= depth; ply++) {
            if (game_winning_color(&gc) != COLOR_NONE) break;
            policy_step(m, &gc, nf_buf, ef_buf, ff_buf, mk_buf);
        }

        float v;
        Color w = game_winning_color(&gc);
        if (w != COLOR_NONE)
            v = (w == seat_color) ? 10.0f : -10.0f;
        else
            v = (float)base_value_fn(&gc, seat_color);
        v += action_bonus(le[ci]);

        if (v > best_val) { best_val = v; best_pos = p; }
    }

    int chosen = cands[best_pos];
    return fix_robber_steal(chosen, le, n_le);
}


/* ── AB2 2-ply greedy search ──────────────────────────────────── */

static int ab2_choose(Game *g, const Action *le, int n_le) {
    Color bc = g->state.colors[g->state.current_player_index];
    int best_i = 0;
    double best_v = -1e30;
    Game ch, ch2;
    Action ca[MAX_ACTIONS], ca2[MAX_ACTIONS];
    int cn, cn2;

    for (int i = 0; i < n_le; i++) {
        game_copy(&ch, g);
        game_execute(&ch, le[i], ca, &cn);
        double v;
        if (cn > 0 && game_winning_color(&ch) == COLOR_NONE) {
            if (cn > 1) {
                Color bc2 = ch.state.colors[ch.state.current_player_index];
                int brj = 0;
                double brv = -1e30;
                for (int j = 0; j < cn; j++) {
                    game_copy(&ch2, &ch);
                    game_execute(&ch2, ca[j], ca2, &cn2);
                    double rv = base_value_fn(&ch2, bc2);
                    if (rv > brv) { brv = rv; brj = j; }
                }
                game_copy(&ch2, &ch);
                game_execute(&ch2, ca[brj], ca2, &cn2);
                v = base_value_fn(&ch2, bc);
            } else {
                game_execute(&ch, ca[0], ca, &cn);
                v = base_value_fn(&ch, bc);
            }
        } else {
            v = base_value_fn(&ch, bc);
        }
        if (v > best_v) { best_v = v; best_i = i; }
    }
    return best_i;
}


/* ── Main ───────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    uint64_t seed_base = 42;
    int num_games = 1;
    int search_depth = 30;
    int top_k = 5;
    bool verbose = false;
    bool vs_ab2 = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i+1 < argc)
            seed_base = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--games") == 0 && i+1 < argc)
            num_games = atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i+1 < argc)
            search_depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i+1 < argc)
            top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--verbose") == 0)
            verbose = true;
        else if (strcmp(argv[i], "--vs-ab2") == 0)
            vs_ab2 = true;
        else if (argv[i][0] != '-')
            seed_base = (uint64_t)atoll(argv[i]);
    }

    /* Resolve weights path relative to executable */
    char exe_dir[1024];
    strncpy(exe_dir, argv[0], sizeof(exe_dir) - 1);
    char *dir = dirname(exe_dir);
    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/weights/model.bin", dir);

    NNModel *model = (NNModel *)calloc(1, sizeof(NNModel));
    if (!model) { fprintf(stderr, "OOM\n"); return 1; }
    if (nn_load(model, weights_path) != 0) {
        snprintf(weights_path, sizeof(weights_path), "weights/model.bin");
        if (nn_load(model, weights_path) != 0) {
            fprintf(stderr, "Failed to load weights/model.bin\n");
            free(model); return 1;
        }
    }

    float (*nf)[NN_NODE_FEAT] = calloc(NN_NODES, sizeof(*nf));
    float (*ef)[NN_EDGE_FEAT] = calloc(NN_MAX_EDGES, sizeof(*ef));
    float *ff = calloc(NN_FLAT_DIM, sizeof(float));
    float *mk = calloc(NN_MASK_DIM, sizeof(float));
    Action *actions = calloc(MAX_ACTIONS, sizeof(Action));

    CatanMap map;
    RngState map_rng;
    Color colors[4] = {0, 1, 2, 3};
    int wins[4] = {0};
    int total_turns = 0;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gi = 0; gi < num_games; gi++) {
        uint64_t seed = seed_base + gi;
        rng_init(&map_rng, seed);
        build_map(&map, MAP_BASE, 0, &map_rng);

        Game g;
        game_init_with_map(&g, &map, 4, colors, seed, 7, false, 10);
        int decisions = 0;

        while (game_winning_color(&g) == COLOR_NONE && g.state.num_turns < 1000) {
            int n_act = generate_playable_actions(&g.state, actions, MAX_ACTIONS);
            if (n_act == 0) break;

            int cp = g.state.current_player_index;
            bool is_ab2_seat = vs_ab2 && (cp == 1 || cp == 3);
            int chosen;
            if (n_act == 1) {
                chosen = 0;
                if (verbose) {
                    char buf[128];
                    format_action(buf, sizeof(buf), actions[0]);
                    printf("  T%3d P%d %s (forced)\n", g.state.num_turns, cp, buf);
                }
            } else if (is_ab2_seat) {
                chosen = ab2_choose(&g, actions, n_act);
                decisions++;
            } else {
                chosen = abt_search(model, &g, actions, n_act,
                                    search_depth, top_k, nf, ef, ff, mk);
                decisions++;
                if (verbose || is_interesting(actions[chosen].type)) {
                    char buf[128];
                    format_action(buf, sizeof(buf), actions[chosen]);
                    printf("  T%3d P%d %s\n", g.state.num_turns, cp, buf);
                }
            }

            int next_n;
            game_execute(&g, actions[chosen], actions, &next_n);
        }

        Color winner = game_winning_color(&g);
        if (winner != COLOR_NONE) wins[g.state.color_to_index[winner]]++;
        total_turns += g.state.num_turns;

        if (num_games == 1) {
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            printf("\n=============================================\n");
            printf("  Search: ABt%d (top-%d)\n", search_depth, top_k);
            printf("  Seed:   %llu\n", (unsigned long long)seed);
            printf("  Winner: Player %d\n",
                   winner != COLOR_NONE ? g.state.color_to_index[winner] : -1);
            printf("  Turns:  %d\n", g.state.num_turns);
            printf("  Time:   %.1fs (%d decisions)\n", elapsed, decisions);
            printf("=============================================\n");
        }
    }

    if (num_games > 1) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("%d games in %.1fs (%.1f games/sec)\n",
               num_games, elapsed, num_games / elapsed);
        printf("Avg turns: %.0f\n", (double)total_turns / num_games);
        if (vs_ab2) {
            int nn_w = wins[0] + wins[2];
            int ab_w = wins[1] + wins[3];
            printf("NN(P0+P2)=%d  AB2(P1+P3)=%d  WR=%.0f%%\n",
                   nn_w, ab_w, 100.0 * nn_w / (nn_w + ab_w + 1e-8));
        } else {
            printf("Wins: P0=%d P1=%d P2=%d P3=%d\n",
                   wins[0], wins[1], wins[2], wins[3]);
        }
    }

    free(nf); free(ef); free(ff); free(mk); free(actions); free(model);
    return 0;
}
