/*
 * Find the exact seed that crashes AB:3 search.
 * Run single-threaded, print seed before each game, crash reveals the bad one.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

int main(void) {
    RngState tmp; rng_init(&tmp, 0);
    CatanMap tm; build_map(&tm, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp);
    board_init_static_graph(&tm);

    SearchCtx *ctx = (SearchCtx*)malloc(sizeof(SearchCtx));
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    for (int gi = 0; gi < 1000000; gi++) {
        uint64_t seed = (uint64_t)1777000000ULL + gi;
        fprintf(stderr, "\rseed=%llu game=%d", (unsigned long long)seed, gi);

        RngState rng; rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);
        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        int ab3_seat = gi % 4;

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Action a;
            if (n == 1) { a = acts[0]; }
            else {
                Color cur = state_current_color(&game.state);
                int seat = -1;
                for (int i = 0; i < 4; i++) if (colors[i] == cur) { seat = i; break; }
                int depth = (seat == ab3_seat) ? 3 : 2;
                ctx->depth_counter = 0;
                Game cp; game_copy(&cp, &game);
                SearchResult sr = alphabeta_search(ctx, &cp, acts, n, depth, -1e30, 1e30, cur, base_value_fn);
                a = (sr.action.type || sr.action.color) ? sr.action : acts[0];
            }
            game_execute(&game, a, acts, &n);
        }
    }
    free(ctx);
    printf("\nNo crash in 1M games\n");
    return 0;
}
