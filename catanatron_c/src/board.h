#ifndef BOARD_H
#define BOARD_H

#include "catan_types.h"
#include "map.h"

#define TOTAL_NODES    96  /* all nodes in static graph (land + water + port) */
#define MAX_DEGREE      3  /* max neighbors per node in the hex grid */
#define MAX_COMPONENTS  8  /* max connected components per color */
#define MAX_ROAD_EDGES 72

typedef struct {
    CatanMap *map;

    /* buildings[node_id] = packed (color<<2)|bld_type, or -1 if empty */
    int8_t buildings[TOTAL_NODES];

    /* roads: road_owner[node][adj_idx] = color or COLOR_NONE(-1) */
    int8_t road_owner[TOTAL_NODES][MAX_DEGREE];

    /* connected_components[color][comp_idx] = bitset of node IDs */
    uint64_t cc_sets[MAX_PLAYERS][MAX_COMPONENTS][2]; /* 128-bit bitset (96 nodes) */
    int      cc_count[MAX_PLAYERS];

    /* board_buildable_ids: bitset of nodes where settlements can be placed */
    uint64_t buildable[2]; /* 128-bit bitset */

    /* road length tracking */
    int  road_lengths[MAX_PLAYERS];
    int  road_color;   /* Color or COLOR_NONE */
    int  road_length;

    Coordinate robber_coordinate;
} Board;

/* Static graph adjacency -- precomputed, shared by all boards */
extern int    STATIC_ADJ[TOTAL_NODES][MAX_DEGREE];
extern int    STATIC_ADJ_COUNT[TOTAL_NODES];
extern volatile bool STATIC_GRAPH_INIT;

void board_init_static_graph(CatanMap *map);
void board_init(Board *b, CatanMap *map);
void board_copy(Board *dst, const Board *src);

/* Building operations -- return (prev_road_color, new_road_color) */
void board_build_settlement(Board *b, Color color, int node_id, bool initial_build_phase);
void board_build_road(Board *b, Color color, int edge_a, int edge_b);
void board_build_city(Board *b, Color color, int node_id);

/* Queries */
int  board_buildable_node_ids(Board *b, Color color, bool initial_build_phase,
                              int *out, int max_out);
int  board_buildable_edges(Board *b, Color color, int out[][2], int max_out);
void board_get_player_port_resources(Board *b, Color color, bool out[6]);

/* Helpers */
static inline Color board_get_node_color(Board *b, int node_id) {
    int8_t v = b->buildings[node_id];
    return (v < 0) ? COLOR_NONE : (Color)(v >> 2);
}
static inline BuildingType board_get_node_building(Board *b, int node_id) {
    return (BuildingType)(b->buildings[node_id] & 0x3);
}
static inline bool board_is_enemy_node(Board *b, int node_id, Color color) {
    Color nc = board_get_node_color(b, node_id);
    return nc != COLOR_NONE && nc != color;
}

/* Bitset helpers for 128-bit sets (nodes 0..95) */
static inline void bs_set(uint64_t s[2], int bit) { s[bit/64] |= (1ULL << (bit%64)); }
static inline void bs_clear(uint64_t s[2], int bit) { s[bit/64] &= ~(1ULL << (bit%64)); }
static inline bool bs_test(const uint64_t s[2], int bit) { return (s[bit/64] >> (bit%64)) & 1; }
static inline void bs_or(uint64_t dst[2], const uint64_t a[2], const uint64_t b[2]) {
    dst[0] = a[0] | b[0]; dst[1] = a[1] | b[1];
}
static inline void bs_and(uint64_t dst[2], const uint64_t a[2], const uint64_t b[2]) {
    dst[0] = a[0] & b[0]; dst[1] = a[1] & b[1];
}
static inline bool bs_empty(const uint64_t s[2]) { return s[0] == 0 && s[1] == 0; }
static inline void bs_zero(uint64_t s[2]) { s[0] = 0; s[1] = 0; }
static inline void bs_copy(uint64_t dst[2], const uint64_t src[2]) { dst[0]=src[0]; dst[1]=src[1]; }

int longest_acyclic_path(Board *b, const uint64_t node_set[2], Color color);

/* Road adjacency index lookup */
int board_adj_index(int from, int to);

/* Is there a road of `color` on edge (a, b)? */
static inline bool board_is_friendly_road(Board *b, int a, int adj_idx, Color color) {
    return b->road_owner[a][adj_idx] == (int8_t)color;
}

#endif
