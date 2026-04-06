/*
 * Board: mutable game board state for Catan.
 * Tracks buildings, roads, connected components, longest road.
 * Uses bitsets for node sets and flat arrays for all storage.
 */

#include "board.h"
#include <string.h>
#include <stdlib.h>

int  STATIC_ADJ[TOTAL_NODES][MAX_DEGREE];
int  STATIC_ADJ_COUNT[TOTAL_NODES];
volatile bool STATIC_GRAPH_INIT = false;

void board_init_static_graph(CatanMap *map) {
    __sync_synchronize();
    if (STATIC_GRAPH_INIT) return;
    memset(STATIC_ADJ_COUNT, 0, sizeof(STATIC_ADJ_COUNT));
    memset(STATIC_ADJ, -1, sizeof(STATIC_ADJ));

    /* Build adjacency from ALL tiles (land + port + water).
     * We iterate all edges from all tiles and add both directions. */
    bool added[TOTAL_NODES][TOTAL_NODES];
    memset(added, 0, sizeof(added));

    /* We need to rebuild from the map's tile edge data.
     * For the base map, tiles are stored during build_map.
     * We iterate land_tiles and ports and add their edges. */

    /* Helper: add edge if not already present */
    #define ADD_EDGE(a, b) do { \
        if ((a) >= 0 && (b) >= 0 && (a) < TOTAL_NODES && (b) < TOTAL_NODES && !added[a][b]) { \
            int _ca = STATIC_ADJ_COUNT[a]; \
            int _cb = STATIC_ADJ_COUNT[b]; \
            if (_ca < MAX_DEGREE) STATIC_ADJ[a][_ca] = (b); \
            if (_cb < MAX_DEGREE) STATIC_ADJ[b][_cb] = (a); \
            STATIC_ADJ_COUNT[a] = _ca + 1; \
            STATIC_ADJ_COUNT[b] = _cb + 1; \
            added[a][b] = true; added[b][a] = true; \
        } \
    } while(0)

    for (int i = 0; i < map->num_land_tiles; i++) {
        for (int e = 0; e < 6; e++) {
            int a = map->land_tiles[i].edges[e][0];
            int b = map->land_tiles[i].edges[e][1];
            ADD_EDGE(a, b);
        }
    }
    for (int i = 0; i < map->num_ports; i++) {
        for (int e = 0; e < 6; e++) {
            int a = map->ports[i].edges[e][0];
            int b = map->ports[i].edges[e][1];
            ADD_EDGE(a, b);
        }
    }
    #undef ADD_EDGE

    /* Sort each adjacency list for deterministic iteration */
    for (int n = 0; n < TOTAL_NODES; n++) {
        int cnt = STATIC_ADJ_COUNT[n];
        for (int i = 0; i < cnt - 1; i++)
            for (int j = i + 1; j < cnt; j++)
                if (STATIC_ADJ[n][i] > STATIC_ADJ[n][j]) {
                    int tmp = STATIC_ADJ[n][i];
                    STATIC_ADJ[n][i] = STATIC_ADJ[n][j];
                    STATIC_ADJ[n][j] = tmp;
                }
    }

    __sync_synchronize(); STATIC_GRAPH_INIT = true; __sync_synchronize();
}

int board_adj_index(int from, int to) {
    for (int i = 0; i < STATIC_ADJ_COUNT[from]; i++)
        if (STATIC_ADJ[from][i] == to) return i;
    return -1;
}

void board_init(Board *b, CatanMap *map) {
    memset(b, 0, sizeof(Board));
    b->map = map;
    memset(b->buildings, -1, sizeof(b->buildings));

    for (int i = 0; i < TOTAL_NODES; i++)
        for (int j = 0; j < MAX_DEGREE; j++)
            b->road_owner[i][j] = (int8_t)COLOR_NONE;

    b->road_color = COLOR_NONE;
    b->road_length = 0;

    /* Initialize buildable set to all land nodes */
    bs_zero(b->buildable);
    for (int i = 0; i < map->num_land_nodes; i++)
        bs_set(b->buildable, map->land_nodes[i]);

    /* Find desert tile for robber */
    for (int i = 0; i < map->num_land_tiles; i++) {
        if (map->land_tiles[i].resource == RES_NONE) {
            b->robber_coordinate = map->land_tile_coords[i];
            break;
        }
    }

    board_init_static_graph(map);
}

void board_copy(Board *dst, const Board *src) {
    memcpy(dst, src, sizeof(Board));
}

/* ---- DFS walk: find all nodes connected to start via roads of given color ---- */

static void dfs_walk(Board *b, int start, Color color, uint64_t result[2]) {
    bs_zero(result);
    int stack[TOTAL_NODES];
    int sp = 0;
    stack[sp++] = start;

    while (sp > 0) {
        int n = stack[--sp];
        if (bs_test(result, n)) continue;
        bs_set(result, n);

        if (board_is_enemy_node(b, n, color)) continue;

        for (int i = 0; i < STATIC_ADJ_COUNT[n]; i++) {
            int v = STATIC_ADJ[n][i];
            if (bs_test(result, v)) continue;
            if (b->road_owner[n][i] == (int8_t)color)
                stack[sp++] = v;
        }
    }
}

/* ---- Find which component contains a node ---- */

static int find_component(Board *b, Color color, int node_id) {
    for (int i = 0; i < b->cc_count[color]; i++)
        if (bs_test(b->cc_sets[color][i], node_id))
            return i;
    return -1;
}

/* ---- Longest acyclic path (DFS) ---- */

int longest_acyclic_path(Board *b, const uint64_t node_set[2], Color color) {
    int best = 0;

    /* For each start node in node_set, DFS for longest road */
    for (int start = 0; start < TOTAL_NODES; start++) {
        if (!bs_test(node_set, start)) continue;

        /* DFS stack: (node, edge_visited_bitset, path_length) */
        typedef struct { int node; uint64_t visited_edges[2]; int len; } Frame;
        Frame stack[256];
        int sp = 0;
        stack[sp++] = (Frame){start, {0, 0}, 0};

        while (sp > 0) {
            Frame f = stack[--sp];
            bool expanded = false;

            for (int i = 0; i < STATIC_ADJ_COUNT[f.node]; i++) {
                int neighbor = STATIC_ADJ[f.node][i];

                if (b->road_owner[f.node][i] != (int8_t)color)
                    continue;
                if (board_is_enemy_node(b, neighbor, color))
                    continue;

                /* Canonical edge ID for visited tracking */
                int ea = f.node < neighbor ? f.node : neighbor;
                int eb = f.node < neighbor ? neighbor : f.node;
                int edge_bit = ea * 3 + board_adj_index(ea, eb); /* unique per edge */
                if (edge_bit >= 128) continue;

                if (bs_test(f.visited_edges, edge_bit))
                    continue;

                Frame nf;
                nf.node = neighbor;
                bs_copy(nf.visited_edges, f.visited_edges);
                bs_set(nf.visited_edges, edge_bit);
                nf.len = f.len + 1;
                stack[sp++] = nf;
                expanded = true;
            }

            if (!expanded && f.len > best)
                best = f.len;
        }
    }
    return best;
}

/* ---- Build Settlement ---- */

void board_build_settlement(Board *b, Color color, int node_id, bool initial_build_phase) {
    b->buildings[node_id] = (int8_t)((color << 2) | BLD_SETTLEMENT);

    if (initial_build_phase) {
        int ci = b->cc_count[color];
        bs_zero(b->cc_sets[color][ci]);
        bs_set(b->cc_sets[color][ci], node_id);
        b->cc_count[color]++;
    } else {
        /* Check if we plow an opponent's road component */
        for (int other = 0; other < MAX_PLAYERS; other++) {
            if (other == (int)color) continue;

            /* Count roads of 'other' color at this node */
            int edge_count = 0;
            int edge_neighbors[MAX_DEGREE];
            for (int i = 0; i < STATIC_ADJ_COUNT[node_id]; i++) {
                if (b->road_owner[node_id][i] == (int8_t)other) {
                    edge_neighbors[edge_count++] = STATIC_ADJ[node_id][i];
                }
            }

            if (edge_count == 2) {
                /* Plowed: split the component */
                int a = edge_neighbors[0], c = edge_neighbors[1];
                int comp_idx = find_component(b, (Color)other, node_id);
                if (comp_idx < 0) continue;

                uint64_t a_set[2], c_set[2];
                dfs_walk(b, a, (Color)other, a_set);
                dfs_walk(b, c, (Color)other, c_set);

                /* Remove old component, add two new ones */
                int last = b->cc_count[other] - 1;
                if (comp_idx != last) {
                    bs_copy(b->cc_sets[other][comp_idx], b->cc_sets[other][last]);
                }
                b->cc_count[other]--;

                int ci = b->cc_count[other];
                bs_copy(b->cc_sets[other][ci], a_set);
                b->cc_count[other]++;
                ci = b->cc_count[other];
                bs_copy(b->cc_sets[other][ci], c_set);
                b->cc_count[other]++;

                /* Recompute road length for plowed color */
                int max_len = 0;
                for (int j = 0; j < b->cc_count[other]; j++) {
                    int len = longest_acyclic_path(b, b->cc_sets[other][j], (Color)other);
                    if (len > max_len) max_len = len;
                }
                b->road_lengths[other] = max_len;

                /* Update global longest road */
                int gl_color = COLOR_NONE, gl_len = 0;
                for (int p = 0; p < MAX_PLAYERS; p++) {
                    if (b->road_lengths[p] > gl_len) {
                        gl_len = b->road_lengths[p];
                        gl_color = p;
                    }
                }
                b->road_color = gl_color;
                b->road_length = gl_len;
            }
        }
    }

    /* Remove node and all neighbors from buildable */
    bs_clear(b->buildable, node_id);
    for (int i = 0; i < STATIC_ADJ_COUNT[node_id]; i++)
        bs_clear(b->buildable, STATIC_ADJ[node_id][i]);

}

/* ---- Build Road ---- */

void board_build_road(Board *b, Color color, int edge_a, int edge_b) {
    int ai = board_adj_index(edge_a, edge_b);
    int bi = board_adj_index(edge_b, edge_a);
    b->road_owner[edge_a][ai] = (int8_t)color;
    b->road_owner[edge_b][bi] = (int8_t)color;

    /* Find connected components for a and b */
    int a_idx = find_component(b, color, edge_a);
    int b_idx = find_component(b, color, edge_b);

    uint64_t *component = NULL;

    if (a_idx < 0 && !board_is_enemy_node(b, edge_a, color) && b_idx >= 0) {
        bs_set(b->cc_sets[color][b_idx], edge_a);
        component = b->cc_sets[color][b_idx];
    } else if (b_idx < 0 && !board_is_enemy_node(b, edge_b, color) && a_idx >= 0) {
        bs_set(b->cc_sets[color][a_idx], edge_b);
        component = b->cc_sets[color][a_idx];
    } else if (a_idx >= 0 && b_idx >= 0 && a_idx != b_idx) {
        /* Merge both components */
        bs_or(b->cc_sets[color][a_idx], b->cc_sets[color][a_idx], b->cc_sets[color][b_idx]);
        component = b->cc_sets[color][a_idx];
        /* Remove b_idx */
        int last = b->cc_count[color] - 1;
        if (b_idx != last) {
            bs_copy(b->cc_sets[color][b_idx], b->cc_sets[color][last]);
        }
        b->cc_count[color]--;
    } else {
        int chosen = (a_idx >= 0) ? a_idx : b_idx;
        if (chosen >= 0) component = b->cc_sets[color][chosen];
    }

    /* Compute longest path in this component */
    if (component) {
        int candidate = longest_acyclic_path(b, component, color);
        if (candidate > b->road_lengths[color])
            b->road_lengths[color] = candidate;
        if (candidate >= 5 && candidate > b->road_length) {
            b->road_color = (int)color;
            b->road_length = candidate;
        }
    }

}

/* ---- Build City ---- */

void board_build_city(Board *b, Color color, int node_id) {
    b->buildings[node_id] = (int8_t)((color << 2) | BLD_CITY);
}

/* ---- Buildable Node IDs ---- */

int board_buildable_node_ids(Board *b, Color color, bool initial_build_phase,
                              int *out, int max_out) {
    int count = 0;
    if (initial_build_phase) {
        for (int n = 0; n < TOTAL_NODES && count < max_out; n++) {
            if (bs_test(b->buildable, n))
                out[count++] = n;
        }
    } else {
        /* Intersection of connected component nodes with buildable */
        uint64_t reachable[2] = {0, 0};
        for (int i = 0; i < b->cc_count[color]; i++)
            bs_or(reachable, reachable, b->cc_sets[color][i]);

        uint64_t result[2];
        bs_and(result, reachable, b->buildable);

        for (int n = 0; n < TOTAL_NODES && count < max_out; n++) {
            if (bs_test(result, n))
                out[count++] = n;
        }
    }
    return count;
}

/* ---- Buildable Edges ---- */

int board_buildable_edges(Board *b, Color color, int out[][2], int max_out) {
    uint64_t reachable[2] = {0, 0};
    for (int i = 0; i < b->cc_count[color]; i++)
        bs_or(reachable, reachable, b->cc_sets[color][i]);

    /* Precompute land node lookup as bitset */
    uint64_t land_bs[2] = {0, 0};
    for (int i = 0; i < b->map->num_land_nodes; i++)
        bs_set(land_bs, b->map->land_nodes[i]);

    int count = 0;
    bool seen[TOTAL_NODES][MAX_DEGREE];
    memset(seen, 0, sizeof(seen));

    for (int n = 0; n < TOTAL_NODES; n++) {
        if (!bs_test(reachable, n) || !bs_test(land_bs, n)) continue;

        for (int i = 0; i < STATIC_ADJ_COUNT[n]; i++) {
            int v = STATIC_ADJ[n][i];
            if (!bs_test(land_bs, v)) continue;

            int vi = board_adj_index(v, n);
            if (seen[v][vi]) continue;
            seen[n][i] = true;

            if (b->road_owner[n][i] != (int8_t)COLOR_NONE) continue;

            if (count < max_out) {
                out[count][0] = n < v ? n : v;
                out[count][1] = n < v ? v : n;
                count++;
            }
        }
    }
    return count;
}

/* ---- Port Resources ---- */

void board_get_player_port_resources(Board *b, Color color, bool out[6]) {
    memset(out, 0, 6 * sizeof(bool));
    for (int r = 0; r < 6; r++) {
        for (int j = 0; j < b->map->port_nodes_count[r]; j++) {
            int nid = b->map->port_nodes[r][j];
            if (board_get_node_color(b, nid) == color) {
                out[r] = true;
                break;
            }
        }
    }
}
