/* Thread-safe alpha-beta search. All mutable state in SearchCtx. */

#include "search.h"
#include "apply_action.h"
#include "actions.h"
#include <math.h>
#include <string.h>

static int action_order(ActionType t) {
    switch (t) {
        case AT_BUILD_CITY: return 0; case AT_BUILD_SETTLEMENT: return 1;
        case AT_BUY_DEVELOPMENT_CARD: return 2; case AT_BUILD_ROAD: return 3;
        case AT_PLAY_KNIGHT_CARD: return 4; case AT_PLAY_MONOPOLY: return 5;
        case AT_PLAY_YEAR_OF_PLENTY: return 6; case AT_PLAY_ROAD_BUILDING: return 7;
        case AT_MARITIME_TRADE: return 8; case AT_MOVE_ROBBER: return 9;
        case AT_END_TURN: return 10; case AT_ROLL: return 11; default: return 20;
    }
}
static int action_cmp(const void *a, const void *b) {
    return action_order(((Action*)a)->type) - action_order(((Action*)b)->type);
}

SearchResult alphabeta_search(SearchCtx *ctx, Game *g, Action *actions, int num_actions,
                               int depth, double alpha, double beta,
                               Color bot_color, ValueFn eval_fn) {
    SearchResult result = {.value = 0, .action = {0}};

    if (depth <= 0 || game_winning_color(g) != COLOR_NONE ||
        ctx->depth_counter >= 6) {
        result.value = eval_fn(g, bot_color);
        return result;
    }

    qsort(actions, num_actions, sizeof(Action), action_cmp);

    int pool_idx = ctx->depth_counter++;
    Game *child = &ctx->pool[pool_idx];
    Action *child_actions = ctx->actions[pool_idx];
    bool maximizing = (state_current_color(&g->state) == bot_color);

    if (maximizing) {
        result.value = -1e30;
        for (int i = 0; i < num_actions; i++) {
            int child_depth = (actions[i].type == AT_ROLL) ? depth : depth - 1;
            int child_n;
            game_copy(child, g);
            game_execute(child, actions[i], child_actions, &child_n);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta, bot_color, eval_fn);
            if (sr.value > result.value) { result.value = sr.value; result.action = actions[i]; }
            alpha = fmax(alpha, result.value);
            if (alpha >= beta) break;
        }
    } else {
        result.value = 1e30;
        for (int i = 0; i < num_actions; i++) {
            int child_depth = (actions[i].type == AT_ROLL) ? depth : depth - 1;
            int child_n;
            game_copy(child, g);
            game_execute(child, actions[i], child_actions, &child_n);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta, bot_color, eval_fn);
            if (sr.value < result.value) { result.value = sr.value; result.action = actions[i]; }
            beta = fmin(beta, result.value);
            if (beta <= alpha) break;
        }
    }

    ctx->depth_counter = pool_idx;
    return result;
}

Action random_player_decide(State *s, Action *actions, int n, RngState *rng) {
    (void)s;
    return actions[rng_choice_index(rng, n)];
}
