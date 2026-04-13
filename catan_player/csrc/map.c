/*
 * Map generation for Catan, matching Python's catanatron exactly.
 *
 * The topology defines coordinates and tile types in insertion order.
 * Tiles are placed one by one; when a new tile is placed, it shares
 * nodes and edges with already-placed neighbors. New nodes get
 * incrementing IDs.
 */

#include "map.h"
#include "rng.h"
#include <string.h>
#include <stdlib.h>

/* ---- Coordinate helpers ---- */

static const Coordinate UNIT_VECTORS[6] = {
    /* DIR_EAST */      { 1, -1,  0},
    /* DIR_SOUTHEAST */ { 0, -1,  1},
    /* DIR_SOUTHWEST */ {-1,  0,  1},
    /* DIR_WEST */      {-1,  1,  0},
    /* DIR_NORTHWEST */ { 0,  1, -1},
    /* DIR_NORTHEAST */ { 1,  0, -1},
};

static Coordinate coord_add(Coordinate a, Coordinate b) {
    return (Coordinate){a.x + b.x, a.y + b.y, a.z + b.z};
}

/* ---- Topology entry ---- */

typedef enum { TOPO_LAND, TOPO_WATER, TOPO_PORT } TopoType;

typedef struct {
    Coordinate coord;
    TopoType   type;
    Direction  port_dir; /* only for TOPO_PORT */
} TopoEntry;

/* Base map topology (37 entries: 19 land + 18 water/port) in Python dict insertion order */
static const TopoEntry BASE_TOPOLOGY[] = {
    /* center */
    {{ 0, 0, 0}, TOPO_LAND, 0},
    /* first ring */
    {{ 1,-1, 0}, TOPO_LAND, 0},
    {{ 0,-1, 1}, TOPO_LAND, 0},
    {{-1, 0, 1}, TOPO_LAND, 0},
    {{-1, 1, 0}, TOPO_LAND, 0},
    {{ 0, 1,-1}, TOPO_LAND, 0},
    {{ 1, 0,-1}, TOPO_LAND, 0},
    /* second ring */
    {{ 2,-2, 0}, TOPO_LAND, 0},
    {{ 1,-2, 1}, TOPO_LAND, 0},
    {{ 0,-2, 2}, TOPO_LAND, 0},
    {{-1,-1, 2}, TOPO_LAND, 0},
    {{-2, 0, 2}, TOPO_LAND, 0},
    {{-2, 1, 1}, TOPO_LAND, 0},
    {{-2, 2, 0}, TOPO_LAND, 0},
    {{-1, 2,-1}, TOPO_LAND, 0},
    {{ 0, 2,-2}, TOPO_LAND, 0},
    {{ 1, 1,-2}, TOPO_LAND, 0},
    {{ 2, 0,-2}, TOPO_LAND, 0},
    {{ 2,-1,-1}, TOPO_LAND, 0},
    /* third ring (water + ports) */
    {{ 3,-3, 0}, TOPO_PORT, DIR_WEST},
    {{ 2,-3, 1}, TOPO_WATER, 0},
    {{ 1,-3, 2}, TOPO_PORT, DIR_NORTHWEST},
    {{ 0,-3, 3}, TOPO_WATER, 0},
    {{-1,-2, 3}, TOPO_PORT, DIR_NORTHWEST},
    {{-2,-1, 3}, TOPO_WATER, 0},
    {{-3, 0, 3}, TOPO_PORT, DIR_NORTHEAST},
    {{-3, 1, 2}, TOPO_WATER, 0},
    {{-3, 2, 1}, TOPO_PORT, DIR_EAST},
    {{-3, 3, 0}, TOPO_WATER, 0},
    {{-2, 3,-1}, TOPO_PORT, DIR_EAST},
    {{-1, 3,-2}, TOPO_WATER, 0},
    {{ 0, 3,-3}, TOPO_PORT, DIR_SOUTHEAST},
    {{ 1, 2,-3}, TOPO_WATER, 0},
    {{ 2, 1,-3}, TOPO_PORT, DIR_SOUTHWEST},
    {{ 3, 0,-3}, TOPO_WATER, 0},
    {{ 3,-1,-2}, TOPO_PORT, DIR_SOUTHWEST},
    {{ 3,-2,-1}, TOPO_WATER, 0},
};
#define BASE_TOPO_LEN 37

/* Mini map topology (19 entries: 7 land + 12 water) */
static const TopoEntry MINI_TOPOLOGY[] = {
    {{ 0, 0, 0}, TOPO_LAND, 0},
    {{ 1,-1, 0}, TOPO_LAND, 0},
    {{ 0,-1, 1}, TOPO_LAND, 0},
    {{-1, 0, 1}, TOPO_LAND, 0},
    {{-1, 1, 0}, TOPO_LAND, 0},
    {{ 0, 1,-1}, TOPO_LAND, 0},
    {{ 1, 0,-1}, TOPO_LAND, 0},
    {{ 2,-2, 0}, TOPO_WATER, 0},
    {{ 1,-2, 1}, TOPO_WATER, 0},
    {{ 0,-2, 2}, TOPO_WATER, 0},
    {{-1,-1, 2}, TOPO_WATER, 0},
    {{-2, 0, 2}, TOPO_WATER, 0},
    {{-2, 1, 1}, TOPO_WATER, 0},
    {{-2, 2, 0}, TOPO_WATER, 0},
    {{-1, 2,-1}, TOPO_WATER, 0},
    {{ 0, 2,-2}, TOPO_WATER, 0},
    {{ 1, 1,-2}, TOPO_WATER, 0},
    {{ 2, 0,-2}, TOPO_WATER, 0},
    {{ 2,-1,-1}, TOPO_WATER, 0},
};
#define MINI_TOPO_LEN 19

/* Resource templates */
static const Resource BASE_TILE_RESOURCES[] = {
    RES_WOOD, RES_WOOD, RES_WOOD, RES_WOOD,
    RES_BRICK, RES_BRICK, RES_BRICK,
    RES_SHEEP, RES_SHEEP, RES_SHEEP, RES_SHEEP,
    RES_WHEAT, RES_WHEAT, RES_WHEAT, RES_WHEAT,
    RES_ORE, RES_ORE, RES_ORE,
    RES_NONE
};
#define BASE_NUM_TILE_RES 19

static const int BASE_NUMBERS[] = {2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12};
#define BASE_NUM_NUMBERS 18

static const Resource BASE_PORT_RESOURCES[] = {
    RES_WOOD, RES_BRICK, RES_SHEEP, RES_WHEAT, RES_ORE,
    RES_NONE, RES_NONE, RES_NONE, RES_NONE
};
#define BASE_NUM_PORT_RES 9

static const int BASE_SPIRAL_NUMBERS[] = {5,2,6,3,8,10,9,12,11,4,8,10,9,4,5,6,3,11};

static const Resource MINI_TILE_RESOURCES[] = {
    RES_WOOD, RES_NONE, RES_BRICK, RES_SHEEP, RES_WHEAT, RES_WHEAT, RES_ORE
};
#define MINI_NUM_TILE_RES 7

static const int MINI_NUMBERS[] = {3,4,5,6,8,9,10};
#define MINI_NUM_NUMBERS 7

/* ---- Placed tile storage during initialization ---- */

typedef struct {
    Coordinate coord;
    int        nodes[6];
    int        edges[6][2];
    bool       placed;
} PlacedTile;




/* Get NodeRef endpoints of an EdgeRef */
static void edge_to_noderefs(int edge_ref, int *a, int *b) {
    /* EdgeRef order: E=0, SE=1, SW=2, W=3, NW=4, NE=5 */
    /* NodeRef order: N=0, NE=1, SE=2, S=3, SW=4, NW=5 */
    static const int edge_nodes[6][2] = {
        {1, 2}, /* E:  NE-SE */
        {2, 3}, /* SE: SE-S  */
        {3, 4}, /* SW: S-SW  */
        {4, 5}, /* W:  SW-NW */
        {5, 0}, /* NW: NW-N  */
        {0, 1}, /* NE: N-NE  */
    };
    *a = edge_nodes[edge_ref][0];
    *b = edge_nodes[edge_ref][1];
}

/* Port direction -> which two NodeRefs face inward */
static void port_dir_to_noderefs(Direction dir, int *a, int *b) {
    /* Matches Python's PORT_DIRECTION_TO_NODEREFS */
    switch (dir) {
        case DIR_WEST:      *a = NREF_NORTHWEST;  *b = NREF_SOUTHWEST; break;
        case DIR_NORTHWEST: *a = NREF_NORTH;      *b = NREF_NORTHWEST; break;
        case DIR_NORTHEAST: *a = NREF_NORTHEAST;  *b = NREF_NORTH;     break;
        case DIR_EAST:      *a = NREF_SOUTHEAST;  *b = NREF_NORTHEAST; break;
        case DIR_SOUTHEAST: *a = NREF_SOUTH;      *b = NREF_SOUTHEAST; break;
        case DIR_SOUTHWEST: *a = NREF_SOUTHWEST;  *b = NREF_SOUTH;     break;
        default: *a = 0; *b = 0; break;
    }
}

static int find_placed(PlacedTile *placed, int placed_count, Coordinate c) {
    for (int i = 0; i < placed_count; i++)
        if (placed[i].placed && coord_eq(placed[i].coord, c))
            return i;
    return -1;
}

static void get_nodes_and_edges(PlacedTile *placed, int placed_count, const TopoEntry *entries, int entry_idx,
                                int nodes[6], int edges[6][2], int *node_autoinc) {
    for (int i = 0; i < 6; i++) nodes[i] = -1;
    for (int i = 0; i < 6; i++) { edges[i][0] = -1; edges[i][1] = -1; }

    Coordinate coord = entries[entry_idx].coord;

    for (int d = 0; d < 6; d++) {
        Coordinate neighbor_coord = coord_add(coord, UNIT_VECTORS[d]);
        int ni = find_placed(placed, placed_count, neighbor_coord);
        if (ni < 0) continue;

        PlacedTile *nb = &placed[ni];
        switch (d) {
            case DIR_EAST:
                nodes[NREF_NORTHEAST] = nb->nodes[NREF_NORTHWEST];
                nodes[NREF_SOUTHEAST] = nb->nodes[NREF_SOUTHWEST];
                edges[DIR_EAST][0] = nb->edges[DIR_WEST][0];
                edges[DIR_EAST][1] = nb->edges[DIR_WEST][1];
                break;
            case DIR_SOUTHEAST:
                nodes[NREF_SOUTH]     = nb->nodes[NREF_NORTHWEST];
                nodes[NREF_SOUTHEAST] = nb->nodes[NREF_NORTH];
                edges[DIR_SOUTHEAST][0] = nb->edges[DIR_NORTHWEST][0];
                edges[DIR_SOUTHEAST][1] = nb->edges[DIR_NORTHWEST][1];
                break;
            case DIR_SOUTHWEST:
                nodes[NREF_SOUTH]     = nb->nodes[NREF_NORTHEAST];
                nodes[NREF_SOUTHWEST] = nb->nodes[NREF_NORTH];
                edges[DIR_SOUTHWEST][0] = nb->edges[DIR_NORTHEAST][0];
                edges[DIR_SOUTHWEST][1] = nb->edges[DIR_NORTHEAST][1];
                break;
            case DIR_WEST:
                nodes[NREF_NORTHWEST] = nb->nodes[NREF_NORTHEAST];
                nodes[NREF_SOUTHWEST] = nb->nodes[NREF_SOUTHEAST];
                edges[DIR_WEST][0] = nb->edges[DIR_EAST][0];
                edges[DIR_WEST][1] = nb->edges[DIR_EAST][1];
                break;
            case DIR_NORTHWEST:
                nodes[NREF_NORTH]     = nb->nodes[NREF_SOUTHEAST];
                nodes[NREF_NORTHWEST] = nb->nodes[NREF_SOUTH];
                edges[DIR_NORTHWEST][0] = nb->edges[DIR_SOUTHEAST][0];
                edges[DIR_NORTHWEST][1] = nb->edges[DIR_SOUTHEAST][1];
                break;
            case DIR_NORTHEAST:
                nodes[NREF_NORTH]     = nb->nodes[NREF_SOUTHWEST];
                nodes[NREF_NORTHEAST] = nb->nodes[NREF_SOUTH];
                edges[DIR_NORTHEAST][0] = nb->edges[DIR_SOUTHWEST][0];
                edges[DIR_NORTHEAST][1] = nb->edges[DIR_SOUTHWEST][1];
                break;
        }
    }

    for (int i = 0; i < 6; i++) {
        if (nodes[i] == -1) nodes[i] = (*node_autoinc)++;
    }
    for (int i = 0; i < 6; i++) {
        if (edges[i][0] == -1) {
            int a, b;
            edge_to_noderefs(i, &a, &b);
            edges[i][0] = nodes[a];
            edges[i][1] = nodes[b];
        }
    }
}

/* ---- Spiral walk for official number placement ---- */


typedef struct {
    Coordinate coords[NUM_LAND_TILES + 7]; /* enough for any map */
    int count;
} CoordList;

static void spiral_land_coordinates(const TopoEntry *topo, int topo_len,
                                     Coordinate start, CoordList *out) {
    out->count = 0;

    /* Build set of land coordinates */
    Coordinate land_coords[32];
    int num_land = 0;
    for (int i = 0; i < topo_len; i++)
        if (topo[i].type == TOPO_LAND)
            land_coords[num_land++] = topo[i].coord;

    #define IS_LAND(c) ({ bool _f = false; \
        for (int _i = 0; _i < num_land; _i++) \
            if (coord_eq(land_coords[_i], (c))) { _f = true; break; } _f; })

    /* directions in reversed order (Python: list(Direction); directions.reverse()) */
    /* Direction enum: E=0, SE=1, SW=2, W=3, NW=4, NE=5 */
    /* reversed: NE=5, NW=4, W=3, SW=2, SE=1, E=0 */
    int dirs[6] = {5, 4, 3, 2, 1, 0};

    /* Find initial direction */
    int dir_idx = -1;
    for (int i = 0; i < 6; i++) {
        int prev_i = (i - 1 + 6) % 6;
        Coordinate next = coord_add(start, UNIT_VECTORS[dirs[i]]);
        Coordinate prev = coord_add(start, UNIT_VECTORS[dirs[prev_i]]);
        if (IS_LAND(next) && !IS_LAND(prev)) {
            dir_idx = i;
            break;
        }
    }
    if (dir_idx < 0) return;

    bool visited[32] = {false};
    #define VISITED(c) ({ int _vi = -1; \
        for (int _i = 0; _i < num_land; _i++) \
            if (coord_eq(land_coords[_i], (c))) { _vi = _i; break; } \
        (_vi >= 0 && visited[_vi]); })
    #define MARK_VISITED(c) do { \
        for (int _i = 0; _i < num_land; _i++) \
            if (coord_eq(land_coords[_i], (c))) { visited[_i] = true; break; } \
    } while(0)

    Coordinate coord = start;
    while (out->count < num_land) {
        if (!VISITED(coord)) {
            MARK_VISITED(coord);
            out->coords[out->count++] = coord;
        }
        Coordinate next = coord_add(coord, UNIT_VECTORS[dirs[dir_idx]]);
        if (IS_LAND(next) && !VISITED(next)) {
            coord = next;
            continue;
        }
        dir_idx = (dir_idx + 1) % 6;
    }
    #undef IS_LAND
    #undef VISITED
    #undef MARK_VISITED
}

/* ---- Build dice probabilities ---- */

static void init_dice_probas(double probas[13]) {
    memset(probas, 0, 13 * sizeof(double));
    for (int i = 1; i <= 6; i++)
        for (int j = 1; j <= 6; j++)
            probas[i + j] += 1.0 / 36.0;
}

/* ---- Main build_map ---- */

void build_map(CatanMap *map, int map_type, int number_placement, RngState *rng) {
    memset(map, 0, sizeof(CatanMap));
    init_dice_probas(map->dice_probas);

    const TopoEntry *topo;
    int topo_len;
    const Resource *tile_res_template;
    int num_tile_res;
    const int *numbers_template;
    int num_numbers;
    const Resource *port_res_template;
    int num_port_res;

    if (map_type == MAP_MINI) {
        topo = MINI_TOPOLOGY;    topo_len = MINI_TOPO_LEN;
        tile_res_template = MINI_TILE_RESOURCES; num_tile_res = MINI_NUM_TILE_RES;
        numbers_template = MINI_NUMBERS;         num_numbers = MINI_NUM_NUMBERS;
        port_res_template = NULL;                num_port_res = 0;
    } else {
        topo = BASE_TOPOLOGY;    topo_len = BASE_TOPO_LEN;
        tile_res_template = BASE_TILE_RESOURCES; num_tile_res = BASE_NUM_TILE_RES;
        numbers_template = BASE_NUMBERS;         num_numbers = BASE_NUM_NUMBERS;
        port_res_template = BASE_PORT_RESOURCES; num_port_res = BASE_NUM_PORT_RES;
    }

    /* Shuffle in Python order: port_resources, tile_resources, numbers */
    int shuffled_port_res[16];
    for (int i = 0; i < num_port_res; i++) shuffled_port_res[i] = port_res_template[i];
    if (num_port_res > 0) {
        int tmp[16]; rng_sample_int(rng, shuffled_port_res, num_port_res, tmp, num_port_res);
        memcpy(shuffled_port_res, tmp, num_port_res * sizeof(int));
    }

    int shuffled_tile_res[32];
    for (int i = 0; i < num_tile_res; i++) shuffled_tile_res[i] = tile_res_template[i];
    { int tmp[32]; rng_sample_int(rng, shuffled_tile_res, num_tile_res, tmp, num_tile_res);
      memcpy(shuffled_tile_res, tmp, num_tile_res * sizeof(int)); }

    int shuffled_numbers[32];
    for (int i = 0; i < num_numbers; i++) shuffled_numbers[i] = numbers_template[i];
    { int tmp[32]; rng_sample_int(rng, shuffled_numbers, num_numbers, tmp, num_numbers);
      memcpy(shuffled_numbers, tmp, num_numbers * sizeof(int)); }

    /* Python pops from end of shuffled lists, so we use indices that decrement */
    int tile_res_idx = num_tile_res - 1;
    int numbers_idx = num_numbers - 1;
    int port_res_idx = num_port_res - 1;

    /* Place tiles in topology order */
    PlacedTile placed[MAX_TOPO_TILES]; memset(placed, 0, sizeof(placed));
    int placed_count = 0;
    int node_autoinc = 0;
    int tile_autoinc = 0;
    int port_autoinc = 0;

    map->num_land_tiles = 0;
    map->num_ports = 0;

    for (int t = 0; t < topo_len; t++) {
        int nodes[6];
        int edges[6][2];
        get_nodes_and_edges(placed, placed_count, topo, t, nodes, edges, &node_autoinc);

        /* Record in placed array */
        placed[placed_count].coord = topo[t].coord;
        memcpy(placed[placed_count].nodes, nodes, sizeof(nodes));
        memcpy(placed[placed_count].edges, edges, sizeof(edges));
        placed[placed_count].placed = true;
        placed_count++;

        if (topo[t].type == TOPO_LAND) {
            Resource res = (Resource)shuffled_tile_res[tile_res_idx--];
            int num = 0;
            if (res != RES_NONE) {
                num = shuffled_numbers[numbers_idx--];
            }
            LandTile *lt = &map->land_tiles[map->num_land_tiles];
            lt->id = tile_autoinc++;
            lt->resource = res;
            lt->number = num;
            memcpy(lt->nodes, nodes, sizeof(nodes));
            memcpy(lt->edges, edges, sizeof(edges));
            map->land_tile_coords[map->num_land_tiles] = topo[t].coord;
            map->num_land_tiles++;
        } else if (topo[t].type == TOPO_PORT) {
            Resource res = (Resource)shuffled_port_res[port_res_idx--];
            Port *p = &map->ports[map->num_ports];
            p->id = port_autoinc++;
            p->resource = res;
            p->direction = topo[t].port_dir;
            memcpy(p->nodes, nodes, sizeof(nodes));
            memcpy(p->edges, edges, sizeof(edges));
            map->num_ports++;
        }
        /* TOPO_WATER: nothing to store */
    }

    /* Official spiral number placement: override numbers */
    if (number_placement == NPLACE_OFFICIAL_SPIRAL && map_type != MAP_TOURNAMENT) {
        Coordinate start = (map_type == MAP_BASE)
            ? (Coordinate){2, -2, 0}
            : (Coordinate){1, -1, 0};

        CoordList spiral;
        spiral_land_coordinates(topo, topo_len, start, &spiral);

        int num_idx = 0;
        for (int s = 0; s < spiral.count; s++) {
            /* Find which land tile has this coordinate */
            for (int i = 0; i < map->num_land_tiles; i++) {
                if (coord_eq(map->land_tile_coords[i], spiral.coords[s])) {
                    if (map->land_tiles[i].resource != RES_NONE) {
                        map->land_tiles[i].number = BASE_SPIRAL_NUMBERS[num_idx++];
                    }
                    break;
                }
            }
        }
    }

    /* Build land_nodes set */
    bool node_is_land[256] = {false};
    map->num_land_nodes = 0;
    for (int i = 0; i < map->num_land_tiles; i++) {
        for (int n = 0; n < 6; n++) {
            int nid = map->land_tiles[i].nodes[n];
            if (nid >= 0 && !node_is_land[nid]) {
                node_is_land[nid] = true;
                map->land_nodes[map->num_land_nodes++] = nid;
            }
        }
    }
    /* Sort land_nodes */
    for (int i = 0; i < map->num_land_nodes - 1; i++)
        for (int j = i + 1; j < map->num_land_nodes; j++)
            if (map->land_nodes[i] > map->land_nodes[j]) {
                int tmp = map->land_nodes[i];
                map->land_nodes[i] = map->land_nodes[j];
                map->land_nodes[j] = tmp;
            }

    /* Build adjacent_tiles */
    memset(map->adjacent_tiles_count, 0, sizeof(map->adjacent_tiles_count));
    for (int i = 0; i < map->num_land_tiles; i++) {
        for (int n = 0; n < 6; n++) {
            int nid = map->land_tiles[i].nodes[n];
            if (nid >= 0 && nid < NUM_NODES) {
                int c = map->adjacent_tiles_count[nid];
                if (c < MAX_ADJ_TILES) {
                    map->adjacent_tiles[nid][c] = i;
                    map->adjacent_tiles_count[nid] = c + 1;
                }
            }
        }
    }

    /* Build port_nodes */
    memset(map->port_nodes_count, 0, sizeof(map->port_nodes_count));
    for (int i = 0; i < map->num_ports; i++) {
        Port *p = &map->ports[i];
        int res_idx = (p->resource == RES_NONE) ? 5 : (int)p->resource;
        int na, nb;
        port_dir_to_noderefs(p->direction, &na, &nb);
        int node_a = p->nodes[na];
        int node_b = p->nodes[nb];
        int c = map->port_nodes_count[res_idx];
        map->port_nodes[res_idx][c] = node_a;
        map->port_nodes[res_idx][c + 1] = node_b;
        map->port_nodes_count[res_idx] = c + 2;
    }
}
