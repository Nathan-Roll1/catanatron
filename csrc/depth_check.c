#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

/* Count how many nodes each depth explores */
static int node_count = 0;

static double counting_eval(Game *g, Color c) {
    node_count++;
    return base_value_fn(g, c);
}

int main(void) {
    RngState rng; rng_init(&rng, 0);
    CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
    board_init_static_graph(&map);
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    Game game; game_init_with_map(&game, &map, 4, colors, 42, 7, false, 10);

    /* Play 30 random ticks to get to mid-game */
    Action acts[MAX_ACTIONS];
    int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
    for (int i = 0; i < 30 && game_winning_color(&game) == COLOR_NONE; i++) {
        Action a = acts[rng_choice_index(&game.rng, n)];
        game_execute(&game, a, acts, &n);
    }

    printf("State: turn=%d, prompt=%d, actions=%d\n",
           game.state.num_turns, game.state.current_prompt, n);

    SearchCtx *ctx = (SearchCtx*)malloc(sizeof(SearchCtx));

    for (int depth = 1; depth <= 5; depth++) {
        node_count = 0;
        ctx->depth_counter = 0;
        Game cp; game_copy(&cp, &game);
        SearchResult sr = alphabeta_search(ctx, &cp, acts, n, depth, -1e30, 1e30,
                                           state_current_color(&game.state), counting_eval);
        printf("AB:%d -> %d leaf evals, action type=%d\n", depth, node_count, sr.action.type);
    }

    printf("\nMAX_SEARCH_DEPTH = %d\n", MAX_SEARCH_DEPTH);
    printf("sizeof(SearchCtx) = %zu\n", sizeof(SearchCtx));
    free(ctx);
    return 0;
}
