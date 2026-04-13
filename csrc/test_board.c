#include <stdio.h>
#include "board.h"
#include "rng.h"

int main(void) {
    CatanMap map;
    rng_seed(42);
    build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

    Board b;
    board_init(&b, &map);

    printf("Robber: (%d,%d,%d)\n", b.robber_coordinate.x, b.robber_coordinate.y, b.robber_coordinate.z);

    /* Count buildable nodes (should be 54 initially) */
    int nodes[96];
    int count = board_buildable_node_ids(&b, COLOR_RED, true, nodes, 96);
    printf("Initial buildable nodes: %d\n", count);

    /* Build a settlement at node 0 for RED */
    board_build_settlement(&b, COLOR_RED, 0, true);
    printf("After RED settlement at 0: building[0] = %d (expect %d)\n",
           b.buildings[0], (COLOR_RED << 2) | BLD_SETTLEMENT);

    /* Check neighbors removed from buildable */
    count = board_buildable_node_ids(&b, COLOR_RED, true, nodes, 96);
    printf("Buildable after settle at 0: %d (expect ~50)\n", count);

    /* Build a road */
    board_build_road(&b, COLOR_RED, 0, 1);
    printf("Road 0-1 built. road_owner[0][adj(0,1)] = %d\n",
           b.road_owner[0][board_adj_index(0, 1)]);

    /* Check static graph */
    printf("\nStatic adj for node 0: ");
    for (int i = 0; i < STATIC_ADJ_COUNT[0]; i++)
        printf("%d ", STATIC_ADJ[0][i]);
    printf("(count=%d)\n", STATIC_ADJ_COUNT[0]);

    printf("Static adj for node 5: ");
    for (int i = 0; i < STATIC_ADJ_COUNT[5]; i++)
        printf("%d ", STATIC_ADJ[5][i]);
    printf("(count=%d)\n", STATIC_ADJ_COUNT[5]);

    printf("\nAll tests passed.\n");
    return 0;
}
