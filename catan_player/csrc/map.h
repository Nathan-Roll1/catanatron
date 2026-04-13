#ifndef MAP_H
#define MAP_H

#include "catan_types.h"
#include "rng.h"

#define MAX_TOPO_TILES  64
#define MAX_ADJ_TILES    3

typedef struct {
    int       id;
    Resource  resource;   /* RES_NONE for desert */
    int       number;     /* 0 for desert */
    int       nodes[6];   /* indexed by NodeRef: N,NE,SE,S,SW,NW */
    int       edges[6][2]; /* indexed by EdgeRef: E,SE,SW,W,NW,NE; each is (node_a, node_b) */
} LandTile;

typedef struct {
    int       id;
    Resource  resource;   /* RES_NONE for 3:1 port */
    Direction direction;
    int       nodes[6];
    int       edges[6][2];
} Port;

typedef struct {
    LandTile  land_tiles[NUM_LAND_TILES];
    int       num_land_tiles;
    Port      ports[NUM_PORTS];
    int       num_ports;

    /* land_nodes: all node IDs that appear on land tiles. Stored as sorted array. */
    int       land_nodes[NUM_NODES];
    int       num_land_nodes;

    /* adjacent_tiles[node_id] = list of land tile indices (into land_tiles[]) */
    int       adjacent_tiles[NUM_NODES][MAX_ADJ_TILES];
    int       adjacent_tiles_count[NUM_NODES];

    /* port_nodes: for each resource (0-4) and generic (index 5), store node IDs */
    int       port_nodes[6][10];
    int       port_nodes_count[6];

    /* dice probabilities (precomputed) */
    double    dice_probas[13]; /* index 2..12 */

    /* coordinate of each land tile (for robber placement) */
    Coordinate land_tile_coords[NUM_LAND_TILES];
} CatanMap;

void build_map(CatanMap *map, int map_type, int number_placement, RngState *rng);

/* map_type constants */
#define MAP_BASE       0
#define MAP_MINI       1
#define MAP_TOURNAMENT 2

/* number_placement constants */
#define NPLACE_OFFICIAL_SPIRAL 0
#define NPLACE_RANDOM          1

#endif
