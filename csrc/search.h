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
    int    same_turn_mode;   /* If nonzero, search terminates at turn boundaries
                                (mirrors Python SameTurnAlphaBetaPlayer).
                                Avoids paranoid-minimax pathology in multi-
                                player Catan. Set via alphabeta_search_same_turn. */
    const void *user_data;  /* opaque pointer for custom eval context */
} SearchCtx;

SearchResult alphabeta_search(SearchCtx *ctx, Game *g, Action *actions, int num_actions,
                               int depth, double alpha, double beta,
                               Color bot_color, ValueFn eval_fn);

/* Same-turn variant: only searches within the bot's current turn, evaluates
 * with eval_fn as soon as another player becomes current. Equivalent to
 * Python catanatron's SameTurnAlphaBetaPlayer. This avoids the paranoid-
 * minimax pathology that makes standard alphabeta_search non-monotonic in
 * depth for 3+-player games. Internally sets ctx->same_turn_mode then
 * invokes alphabeta_search. */
SearchResult alphabeta_search_same_turn(SearchCtx *ctx, Game *g, Action *actions,
                                         int num_actions, int depth,
                                         double alpha, double beta,
                                         Color bot_color, ValueFn eval_fn);

Action random_player_decide(State *s, Action *actions, int n, RngState *rng);

#endif
