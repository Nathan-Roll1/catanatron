#ifndef SEARCH_H
#define SEARCH_H

#include "game.h"
#include "value.h"

typedef double (*ValueFn)(Game *g, Color color);

typedef struct {
    Action action;
    double value;
} SearchResult;

/* Thread-safe search context -- caller allocates on heap or stack */
#define MAX_SEARCH_DEPTH 24

typedef struct {
    Game   pool[MAX_SEARCH_DEPTH];
    Action actions[MAX_SEARCH_DEPTH][MAX_ACTIONS];
    int    depth_counter;
    const void *user_data;  /* opaque pointer for custom eval context */
} SearchCtx;

SearchResult alphabeta_search(SearchCtx *ctx, Game *g, Action *actions, int num_actions,
                               int depth, double alpha, double beta,
                               Color bot_color, ValueFn eval_fn);

/* Known-future alpha-beta: applies every action through game_execute. */
SearchResult alphabeta_search_deterministic(SearchCtx *ctx, Game *g,
                               Action *actions, int num_actions,
                               int depth, double alpha, double beta,
                               Color bot_color, ValueFn eval_fn);

/* Known-future MaxN-style search: each turn owner maximizes their own
 * evaluator component, matching selfish FFA play better than paranoid AB. */
SearchResult maxn_search_deterministic(SearchCtx *ctx, Game *g,
                               Action *actions, int num_actions,
                               int depth, ValueFn eval_fn);

Action random_player_decide(State *s, Action *actions, int n, RngState *rng);

#endif
