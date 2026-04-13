#include <stdio.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    Color colors[] = {COLOR_RED, COLOR_BLUE};

    printf("Playing 10 random games...\n");
    for (int seed = 0; seed < 10; seed++) {
        CatanMap map;
        rng_seed((uint64_t)seed);
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);
        Game g;
        game_init_with_map(&g, &map, 2, colors, (uint64_t)seed, 7, false, 10);
        Color winner = game_play(&g, random_player_decide);
        const char *names[] = {"RED", "BLUE", "ORANGE", "WHITE"};
        printf("  seed=%d: winner=%s turns=%d\n", seed,
               winner == COLOR_NONE ? "NONE" : names[winner], g.state.num_turns);
    }

    printf("\nBenchmarking 10000 random games...\n");
    clock_t start = clock();
    int wins[4] = {0};
    for (int seed = 0; seed < 10000; seed++) {
        CatanMap map;
        rng_seed((uint64_t)seed);
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);
        Game g;
        game_init_with_map(&g, &map, 2, colors, (uint64_t)seed, 7, false, 10);
        Color w = game_play(&g, random_player_decide);
        if (w != COLOR_NONE) wins[w]++;
    }
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("  10000 games in %.3fs (%.0f games/sec)\n", elapsed, 10000.0/elapsed);
    printf("  RED=%d BLUE=%d\n", wins[COLOR_RED], wins[COLOR_BLUE]);

    printf("\nsizeof(State): %zu  sizeof(Game): %zu\n", sizeof(State), sizeof(Game));

    return 0;
}
