/* state_encode.c — C port of hexzero/encoder/state_encoder.py
 *
 * Goal: produce byte-identical output to the Python encoder for any
 * (Game state) input, so we can swap it in transparently.
 *
 * Key design: pre-compute all per-map lookup tables once in
 * state_encoder_init(), then encode_state() is pure arithmetic over the
 * Game struct.
 */

#include "state_encode.h"
#include "actions.h"
#include "state.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PLAYERS 4
#define NUM_RESOURCES 5
#define NUM_PROMPTS 7
#define FEAT_PER_PLAYER 24
#define FEAT_BANK 6
#define FEAT_PHASE 13

/* Player-state field indices (must match catan_types.h enum order) */
#define _VP            0
#define _ROADS_AVAIL   1
#define _SETT_AVAIL    2
#define _CITIES_AVAIL  3
#define _HAS_ROAD      4
#define _HAS_ARMY      5
#define _HAS_ROLLED    6
#define _HAS_PLAYED    7
#define _LONGEST_ROAD  9
#define _RES          14   /* 14..18 wood,brick,sheep,wheat,ore */
#define _DEV          19   /* 19..23 dev cards in hand */
#define _PLAYED       24   /* 24..28 played dev cards */

/* Dice probability for each sum 2..12 */
static const float DICE_PROB[13] = {
    0.0f, 0.0f,
    1.0f/36, 2.0f/36, 3.0f/36, 4.0f/36, 5.0f/36, 6.0f/36,
    5.0f/36, 4.0f/36, 3.0f/36, 2.0f/36, 1.0f/36
};

/* ===== Initialization =====
 * Build lookup tables from the game's map. The map provides:
 *   - land_tiles[19]: each has nodes[6], coords (via land_tile_coords[i])
 *   - ports[9]: each has resource + nodes
 * STATIC_ADJ gives us node adjacency.
 *
 * Steps:
 *   1. Collect unique land node IDs (sorted) → local_to_global, land_to_local
 *   2. Compute ltiles[t][k] = land_to_local[map.land_tiles[t].nodes[k]]
 *   3. Copy tile_coords from map.land_tile_coords
 *   4. Build directed edge list from tile rings (consecutive nodes share edge)
 *      For each undirected edge (a,b) add two directed (a→b, b→a)
 *      Sort by (min(a,b), max(a,b)) to match Python's sorted iteration order
 *   5. For each directed edge (s,d), find adj_idx so STATIC_ADJ[s][adj_idx]==d
 *   6. Build port_oh from ports[].resource and ports[].nodes
 */

static int int_cmp(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return x - y;
}

typedef struct { int a, b; } Pair;
static int pair_cmp(const void *p, const void *q) {
    const Pair *pa = (const Pair *)p, *pb = (const Pair *)q;
    if (pa->a != pb->a) return pa->a - pb->a;
    return pa->b - pb->b;
}

void state_encoder_init(StateEncoderC *enc, const Game *g, int num_players) {
    memset(enc, 0, sizeof(*enc));
    enc->n_real_players = (num_players >= 2 && num_players <= 4) ? num_players : 4;

    CatanMap *map = g->state.board.map;
    if (!map) return;

    /* Initialize this dylib's STATIC_ADJ if not yet done. Idempotent:
     * board_init_static_graph() returns immediately if already initialized.
     * Each shared library has its own copy of STATIC_ADJ (it's a global in
     * board.c that gets statically linked into every dylib that includes it). */
    board_init_static_graph(map);

    /* 1. Collect unique land node IDs */
    int land_buf[TOTAL_NODES];
    int n_land = 0;
    {
        int seen[TOTAL_NODES] = {0};
        for (int t = 0; t < NUM_LAND_TILES; t++) {
            for (int k = 0; k < 6; k++) {
                int nid = map->land_tiles[t].nodes[k];
                if (nid >= 0 && nid < TOTAL_NODES && !seen[nid]) {
                    seen[nid] = 1;
                    land_buf[n_land++] = nid;
                }
            }
        }
    }
    qsort(land_buf, n_land, sizeof(int), int_cmp);
    enc->N = n_land;

    for (int i = 0; i < TOTAL_NODES; i++) enc->land_to_local[i] = -1;
    for (int i = 0; i < n_land; i++) {
        enc->local_to_global[i] = land_buf[i];
        enc->land_to_local[land_buf[i]] = i;
    }

    /* 2. ltiles in local-node space */
    for (int t = 0; t < NUM_LAND_TILES; t++) {
        for (int k = 0; k < 6; k++) {
            int gnode = map->land_tiles[t].nodes[k];
            enc->ltiles[t][k] = (gnode >= 0 && gnode < TOTAL_NODES) ?
                                 enc->land_to_local[gnode] : -1;
        }
    }

    /* 3. Tile coords */
    for (int t = 0; t < NUM_LAND_TILES; t++) {
        enc->tile_coords[t][0] = map->land_tile_coords[t].x;
        enc->tile_coords[t][1] = map->land_tile_coords[t].y;
        enc->tile_coords[t][2] = map->land_tile_coords[t].z;
    }

    /* 4. Build directed edge list in canonical Python order:
     *    For each tile, consecutive ring positions share an edge.
     *    Collect unique undirected edges into a set, sort by (min,max), then
     *    emit (a,b) and (b,a) for each in that sorted order. */
    Pair undir_edges[NUM_LAND_TILES * 6];
    int n_undir = 0;
    {
        /* Use sorted list with dedup (small N, O(N^2) is fine) */
        for (int t = 0; t < NUM_LAND_TILES; t++) {
            for (int i = 0; i < 6; i++) {
                int a = map->land_tiles[t].nodes[i];
                int b = map->land_tiles[t].nodes[(i + 1) % 6];
                if (a < 0 || b < 0) continue;
                int lo = a < b ? a : b;
                int hi = a < b ? b : a;
                int dup = 0;
                for (int k = 0; k < n_undir; k++) {
                    if (undir_edges[k].a == lo && undir_edges[k].b == hi) {
                        dup = 1; break;
                    }
                }
                if (!dup) {
                    undir_edges[n_undir].a = lo;
                    undir_edges[n_undir].b = hi;
                    n_undir++;
                }
            }
        }
    }
    qsort(undir_edges, n_undir, sizeof(Pair), pair_cmp);

    /* 5. Emit directed edges in Python order: for each (a,b), emit a→b then b→a.
     *    Python: src_list += [la, lb]; dst_list += [lb, la]
     *    So directed edge 2k is (a→b), 2k+1 is (b→a). */
    enc->E = n_undir * 2;
    if (enc->E > ENC_NUM_EDGES) {
        fprintf(stderr, "encoder_init: E=%d exceeds ENC_NUM_EDGES=%d\n",
                enc->E, ENC_NUM_EDGES);
        enc->E = ENC_NUM_EDGES;
    }

    /* For each directed edge, find adj index in STATIC_ADJ[src] of dst.
     * road_src_global[k] = global src node, road_adj_idx[k] = which adj slot. */
    for (int k = 0; k < n_undir; k++) {
        int ga = undir_edges[k].a;
        int gb = undir_edges[k].b;

        /* Edge a→b */
        int adj_ab = -1;
        for (int j = 0; j < MAX_DEGREE; j++) {
            if (STATIC_ADJ[ga][j] == gb) { adj_ab = j; break; }
        }
        enc->road_src_global[2*k]     = ga;
        enc->road_adj_idx[2*k]        = adj_ab;

        /* Edge b→a */
        int adj_ba = -1;
        for (int j = 0; j < MAX_DEGREE; j++) {
            if (STATIC_ADJ[gb][j] == ga) { adj_ba = j; break; }
        }
        enc->road_src_global[2*k + 1] = gb;
        enc->road_adj_idx[2*k + 1]    = adj_ba;
    }

    /* 6. Port one-hot from ports[].
     * port_oh[local][k]: 0 = no port, 1..5 = wood/brick/sheep/wheat/ore, 6 = generic
     * For each port, mark its 2 inward-facing nodes (port_dir_to_noderefs).
     * Default: column 0 = 1 (no port). */
    for (int i = 0; i < enc->N; i++) {
        for (int k = 0; k < 7; k++) enc->port_oh[i][k] = 0.0f;
        enc->port_oh[i][0] = 1.0f;
    }

    /* Direction → noderefs (mirrors map.c port_dir_to_noderefs) */
    static const int port_dir_to_nrefs[6][2] = {
        {2, 1},  /* DIR_EAST: SE, NE */
        {3, 2},  /* DIR_SOUTHEAST: S, SE */
        {4, 3},  /* DIR_SOUTHWEST: SW, S */
        {5, 4},  /* DIR_WEST: NW, SW */
        {0, 5},  /* DIR_NORTHWEST: N, NW */
        {1, 0},  /* DIR_NORTHEAST: NE, N */
    };

    for (int p = 0; p < map->num_ports; p++) {
        const Port *pt = &map->ports[p];
        int d = pt->direction;
        if (d < 0 || d >= 6) continue;
        int na = port_dir_to_nrefs[d][0];
        int nb = port_dir_to_nrefs[d][1];
        int gna = pt->nodes[na];
        int gnb = pt->nodes[nb];

        /* Resource: -1 = generic (slot 6), 0..4 = specific (slot 1..5) */
        int slot;
        if (pt->resource == RES_NONE) {
            slot = 6;
        } else if (pt->resource >= 0 && pt->resource < 5) {
            slot = 1 + pt->resource;
        } else {
            continue;  /* unknown port type */
        }

        for (int q = 0; q < 2; q++) {
            int gn = (q == 0) ? gna : gnb;
            if (gn < 0 || gn >= TOTAL_NODES) continue;
            int local = enc->land_to_local[gn];
            if (local < 0) continue;  /* port nodes might not be land */
            enc->port_oh[local][0] = 0.0f;
            for (int s = 1; s < 7; s++) enc->port_oh[local][s] = 0.0f;
            enc->port_oh[local][slot] = 1.0f;
        }
    }
}

/* ===== Encode state =====
 * Output:
 *   nf[N][18]   - per-node features
 *   ef[E][5]    - per-directed-edge features
 *   flat[115]   - global state vector
 *
 * Memory layout: row-major. nf[n][f] = nf[n * 18 + f].
 */
void encode_state_full(const StateEncoderC *enc, const Game *g,
                       float *nf, float *ef, float *flat) {
    const State *st = &g->state;
    const int N = enc->N;
    const int E = enc->E;
    int cp = st->current_player_index;

    /* Zero buffers */
    memset(nf, 0, N * ENC_NODE_FEAT_DIM * sizeof(float));
    memset(ef, 0, E * ENC_EDGE_FEAT_DIM * sizeof(float));
    memset(flat, 0, ENC_FLAT_FEAT_DIM * sizeof(float));

    /* ---- Node features (N, 18) ---- */
    /* 0..4: building one-hot [empty, own_sett, own_city, foe_sett, foe_city] */
    for (int local = 0; local < N; local++) {
        int gnode = enc->local_to_global[local];
        int8_t bld = st->board.buildings[gnode];
        float *row = nf + local * ENC_NODE_FEAT_DIM;
        if (bld < 0) {
            row[0] = 1.0f;  /* empty */
            continue;
        }
        int color_raw = bld >> 2;
        int typ = bld & 0x3;
        int seat = (color_raw >= 0 && color_raw < MAX_PLAYERS) ?
                    st->color_to_index[color_raw] : -1;
        int is_own = (seat == cp);
        if (is_own) {
            if (typ == 0) row[1] = 1.0f;       /* own settlement */
            else if (typ == 1) row[2] = 1.0f;  /* own city */
        } else {
            if (typ == 0) row[3] = 1.0f;       /* foe settlement */
            else if (typ == 1) row[4] = 1.0f;  /* foe city */
        }
    }

    /* 5..9: resource production (dice-prob weighted, per resource) */
    const CatanMap *map = st->board.map;
    if (map) {
        for (int t = 0; t < NUM_LAND_TILES; t++) {
            int res = map->land_tiles[t].resource;
            int num = map->land_tiles[t].number;
            if (res < 0 || res >= 5 || num <= 0 || num > 12) continue;
            float prob = DICE_PROB[num];
            for (int k = 0; k < 6; k++) {
                int local = enc->ltiles[t][k];
                if (local < 0 || local >= N) continue;
                nf[local * ENC_NODE_FEAT_DIM + 5 + res] += prob;
            }
        }
    }

    /* 10..16: port one-hot */
    for (int local = 0; local < N; local++) {
        for (int k = 0; k < 7; k++) {
            nf[local * ENC_NODE_FEAT_DIM + 10 + k] = enc->port_oh[local][k];
        }
    }

    /* 17: robber adjacency */
    {
        int rx = st->board.robber_coordinate.x;
        int ry = st->board.robber_coordinate.y;
        int rz = st->board.robber_coordinate.z;
        for (int t = 0; t < NUM_LAND_TILES; t++) {
            if (enc->tile_coords[t][0] == rx &&
                enc->tile_coords[t][1] == ry &&
                enc->tile_coords[t][2] == rz) {
                for (int k = 0; k < 6; k++) {
                    int local = enc->ltiles[t][k];
                    if (local >= 0 && local < N) {
                        nf[local * ENC_NODE_FEAT_DIM + 17] = 1.0f;
                    }
                }
                break;  /* Python takes first match: rt[0] */
            }
        }
    }

    /* ---- Edge features (E, 5) ---- */
    int n_real = enc->n_real_players;

    for (int e = 0; e < E; e++) {
        int gsrc = enc->road_src_global[e];
        int adj  = enc->road_adj_idx[e];
        if (gsrc < 0 || adj < 0) {
            ef[e * ENC_EDGE_FEAT_DIM + 0] = 1.0f;  /* no road */
            continue;
        }
        int8_t color_raw = st->board.road_owner[gsrc][adj];
        int seat = (color_raw >= 0 && color_raw < MAX_PLAYERS) ?
                    st->color_to_index[color_raw] : -1;
        if (seat < 0) {
            ef[e * ENC_EDGE_FEAT_DIM + 0] = 1.0f;  /* no road */
        } else if (seat == cp) {
            ef[e * ENC_EDGE_FEAT_DIM + 1] = 1.0f;  /* own road */
        } else {
            /* Enemy road: rotate by n_real */
            for (int k = 1; k < MAX_PLAYERS; k++) {
                if (k < n_real && seat == ((cp + k) % n_real)) {
                    ef[e * ENC_EDGE_FEAT_DIM + 1 + k] = 1.0f;
                    break;
                }
            }
        }
    }

    /* ---- Flat features (115) ---- */
    int o = 0;
    for (int p = 0; p < MAX_PLAYERS; p++) {
        if (p < n_real) {
            int real_seat = (cp + p) % n_real;
            const int *ps = st->player_state[real_seat];
            flat[o + 0]  = (float)ps[_VP] / 10.0f;
            flat[o + 1]  = (float)ps[_RES + 0] / 19.0f;
            flat[o + 2]  = (float)ps[_RES + 1] / 19.0f;
            flat[o + 3]  = (float)ps[_RES + 2] / 19.0f;
            flat[o + 4]  = (float)ps[_RES + 3] / 19.0f;
            flat[o + 5]  = (float)ps[_RES + 4] / 19.0f;
            flat[o + 6]  = (float)ps[_DEV + 0] / 14.0f;
            flat[o + 7]  = (float)ps[_DEV + 1] / 14.0f;
            flat[o + 8]  = (float)ps[_DEV + 2] / 14.0f;
            flat[o + 9]  = (float)ps[_DEV + 3] / 14.0f;
            flat[o + 10] = (float)ps[_DEV + 4] / 14.0f;
            flat[o + 11] = (float)ps[_PLAYED + 0];
            flat[o + 12] = (float)ps[_PLAYED + 1];
            flat[o + 13] = (float)ps[_PLAYED + 2];
            flat[o + 14] = (float)ps[_PLAYED + 3];
            flat[o + 15] = (float)ps[_PLAYED + 4];
            flat[o + 16] = (float)ps[_HAS_ROAD];
            flat[o + 17] = (float)ps[_HAS_ARMY];
            flat[o + 18] = (float)ps[_HAS_ROLLED];
            flat[o + 19] = (float)ps[_HAS_PLAYED];
            flat[o + 20] = (float)ps[_ROADS_AVAIL] / 15.0f;
            flat[o + 21] = (float)ps[_SETT_AVAIL] / 5.0f;
            flat[o + 22] = (float)ps[_CITIES_AVAIL] / 4.0f;
            flat[o + 23] = (float)ps[_LONGEST_ROAD] / 15.0f;
        }
        o += FEAT_PER_PLAYER;
    }

    /* Bank: 5 resources / 19, dev_deck / 25 */
    int o1 = MAX_PLAYERS * FEAT_PER_PLAYER;  /* 96 */
    for (int r = 0; r < 5; r++) {
        flat[o1 + r] = (float)st->resource_freqdeck[r] / 19.0f;
    }
    flat[o1 + 5] = (float)st->dev_deck_size / 25.0f;

    /* Phase: prompt one-hot[7], 5 flags, num_turns/1000 */
    int o2 = o1 + FEAT_BANK;  /* 102 */
    int prompt = st->current_prompt;
    if (prompt >= 0 && prompt < NUM_PROMPTS) {
        flat[o2 + prompt] = 1.0f;
    }
    flat[o2 + 7]  = st->is_initial_build_phase ? 1.0f : 0.0f;
    flat[o2 + 8]  = st->is_discarding ? 1.0f : 0.0f;
    flat[o2 + 9]  = st->is_road_building ? 1.0f : 0.0f;
    flat[o2 + 10] = st->is_moving_knight ? 1.0f : 0.0f;
    flat[o2 + 11] = st->is_resolving_trade ? 1.0f : 0.0f;
    flat[o2 + 12] = (float)st->num_turns / 1000.0f;
}
