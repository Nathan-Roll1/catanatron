#include <stdio.h>
#include "map.h"
#include "rng.h"

int main(void) {
    CatanMap map;
    rng_seed(42);
    build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

    printf("=== LAND TILES ===\n");
    for (int i = 0; i < map.num_land_tiles; i++) {
        LandTile *t = &map.land_tiles[i];
        printf("  tile %d: res=%d num=%d nodes=[%d, %d, %d, %d, %d, %d]\n",
               t->id, t->resource, t->number,
               t->nodes[0], t->nodes[1], t->nodes[2],
               t->nodes[3], t->nodes[4], t->nodes[5]);
    }

    printf("\n=== LAND NODES (%d) ===\n[", map.num_land_nodes);
    for (int i = 0; i < map.num_land_nodes; i++) {
        printf("%d%s", map.land_nodes[i], i < map.num_land_nodes - 1 ? ", " : "");
    }
    printf("]\n");

    printf("\n=== PORTS (%d) ===\n", map.num_ports);
    for (int i = 0; i < map.num_ports; i++) {
        Port *p = &map.ports[i];
        printf("  port %d: res=%d dir=%d nodes=[%d, %d, %d, %d, %d, %d]\n",
               p->id, p->resource, p->direction,
               p->nodes[0], p->nodes[1], p->nodes[2],
               p->nodes[3], p->nodes[4], p->nodes[5]);
    }

    printf("\n=== PORT_NODES ===\n");
    const char *port_names[] = {"WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "3:1"};
    for (int r = 0; r < 6; r++) {
        if (map.port_nodes_count[r] > 0) {
            printf("  %s: [", port_names[r]);
            for (int j = 0; j < map.port_nodes_count[r]; j++)
                printf("%d%s", map.port_nodes[r][j], j < map.port_nodes_count[r]-1 ? ", " : "");
            printf("]\n");
        }
    }

    printf("\n=== ADJACENT_TILES (node 0-5) ===\n");
    for (int n = 0; n < 6; n++) {
        printf("  node %d: tiles [", n);
        for (int j = 0; j < map.adjacent_tiles_count[n]; j++)
            printf("%d%s", map.land_tiles[map.adjacent_tiles[n][j]].id,
                   j < map.adjacent_tiles_count[n]-1 ? ", " : "");
        printf("]\n");
    }

    return 0;
}
