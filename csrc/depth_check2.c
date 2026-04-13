#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

static int node_count = 0;
static double counting_eval(Game *g, Color c) { node_count++; return base_value_fn(g, c); }

int main(void) {
    RngState rng; rng_init(&rng, 0);
    CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
    board_init_static_graph(&map);
    Color colors[2] = {COLOR_RED, COLOR_BLUE};
    
    /* Try multiple seeds to find a state with many actions */
    for (int seed = 0; seed < 20; seed++) {
        Game game; game_init_with_map(&game, &map, 2, colors, seed, 7, false, 10);
        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        /* Play until we have a PLAY_TURN prompt with 5+ actions */
        for (int tick = 0; tick < 200 && game_winning_color(&game) == COLOR_NONE; tick++) {
            if (game.state.current_prompt == PROMPT_PLAY_TURN && n > 5) {
                printf("seed=%d tick=%d turn=%d n_actions=%d prompt=%d\n",
                       seed, tick, game.state.num_turns, n, game.state.current_prompt);
                SearchCtx *ctx = (SearchCtx*)malloc(sizeof(SearchCtx));
                for (int depth = 1; depth <= 4; depth++) {
                    node_count = 0;
                    ctx->depth_counter = 0;
                    Game cp; game_copy(&cp, &game);
                    alphabeta_search(ctx, &cp, acts, n, depth, -1e30, 1e30,
                                     state_current_color(&game.state), counting_eval);
                    printf("  AB:%d -> %5d nodes\n", depth, node_count);
                }
                free(ctx);
                printf("\n");
                break;
            }
            Action a = acts[rng_choice_index(&game.rng, n)];
            game_execute(&game, a, acts, &n);
        }
    }
    return 0;
}
