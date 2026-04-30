/* Thread-safe alpha-beta search. All mutable state in SearchCtx. */

#include "search.h"
#include "apply_action.h"
#include "actions.h"
#include "state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

/* Dice probabilities for sums 2..12 (sum_ * 36 = numerator) */
static const double DICE_P_SUM[13] = {
    0, 0,
    1.0/36, 2.0/36, 3.0/36, 4.0/36, 5.0/36, 6.0/36,
    5.0/36, 4.0/36, 3.0/36, 2.0/36, 1.0/36
};

/* Forward decl */
static double eval_action_expected(SearchCtx *ctx, Game *g, Action a,
                                    int child_depth, double alpha, double beta,
                                    Color bot_color, ValueFn eval_fn,
                                    Game *child, Action *child_actions);
static double eval_action_deterministic(SearchCtx *ctx, Game *g, Action a,
                                         int child_depth, double alpha,
                                         double beta, Color bot_color,
                                         ValueFn eval_fn, Game *child,
                                         Action *child_actions);

SearchResult alphabeta_search(SearchCtx *ctx, Game *g, Action *actions, int num_actions,
                               int depth, double alpha, double beta,
                               Color bot_color, ValueFn eval_fn) {
    SearchResult result = {.value = 0, .action = {0}};

    if (depth <= 0 || game_winning_color(g) != COLOR_NONE ||
        ctx->depth_counter >= MAX_SEARCH_DEPTH) {
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
            double v = eval_action_expected(ctx, g, actions[i], child_depth,
                                             alpha, beta, bot_color, eval_fn,
                                             child, child_actions);
            if (v > result.value) { result.value = v; result.action = actions[i]; }
            alpha = fmax(alpha, result.value);
            if (alpha >= beta) break;
        }
    } else {
        result.value = 1e30;
        for (int i = 0; i < num_actions; i++) {
            int child_depth = (actions[i].type == AT_ROLL) ? depth : depth - 1;
            double v = eval_action_expected(ctx, g, actions[i], child_depth,
                                             alpha, beta, bot_color, eval_fn,
                                             child, child_actions);
            if (v < result.value) { result.value = v; result.action = actions[i]; }
            beta = fmin(beta, result.value);
            if (beta <= alpha) break;
        }
    }

    ctx->depth_counter = pool_idx;
    return result;
}

SearchResult alphabeta_search_deterministic(SearchCtx *ctx, Game *g,
                               Action *actions, int num_actions,
                               int depth, double alpha, double beta,
                               Color bot_color, ValueFn eval_fn) {
    SearchResult result = {.value = 0, .action = {0}};

    if (depth <= 0 || game_winning_color(g) != COLOR_NONE ||
        ctx->depth_counter >= MAX_SEARCH_DEPTH) {
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
            double v = eval_action_deterministic(ctx, g, actions[i], child_depth,
                                                 alpha, beta, bot_color,
                                                 eval_fn, child, child_actions);
            if (v > result.value) { result.value = v; result.action = actions[i]; }
            alpha = fmax(alpha, result.value);
            if (alpha >= beta) break;
        }
    } else {
        result.value = 1e30;
        for (int i = 0; i < num_actions; i++) {
            int child_depth = (actions[i].type == AT_ROLL) ? depth : depth - 1;
            double v = eval_action_deterministic(ctx, g, actions[i], child_depth,
                                                 alpha, beta, bot_color,
                                                 eval_fn, child, child_actions);
            if (v < result.value) { result.value = v; result.action = actions[i]; }
            beta = fmin(beta, result.value);
            if (beta <= alpha) break;
        }
    }

    ctx->depth_counter = pool_idx;
    return result;
}

typedef struct {
    Action action;
    double values[MAX_PLAYERS];
} MaxNResult;

static MaxNResult maxn_recurse_deterministic(SearchCtx *ctx, Game *g,
                                             Action *actions, int num_actions,
                                             int depth, ValueFn eval_fn) {
    MaxNResult result;
    memset(&result, 0, sizeof(result));

    Color winner = game_winning_color(g);
    if (depth <= 0 || winner != COLOR_NONE ||
        ctx->depth_counter >= MAX_SEARCH_DEPTH || num_actions <= 0) {
        for (int p = 0; p < g->state.num_players; p++) {
            result.values[p] = eval_fn(g, g->state.colors[p]);
        }
        return result;
    }

    qsort(actions, num_actions, sizeof(Action), action_cmp);

    int pool_idx = ctx->depth_counter++;
    Game *child = &ctx->pool[pool_idx];
    Action *child_actions = ctx->actions[pool_idx];
    int cp = g->state.current_player_index;

    double best = -1e300;
    int have_best = 0;
    for (int i = 0; i < num_actions; i++) {
        int child_depth = (actions[i].type == AT_ROLL) ? depth : depth - 1;
        game_copy(child, g);
        int child_n = 0;
        game_execute(child, actions[i], child_actions, &child_n);

        MaxNResult child_result = maxn_recurse_deterministic(
            ctx, child, child_actions, child_n, child_depth, eval_fn);
        double v = child_result.values[cp];
        if (!have_best || v > best) {
            best = v;
            have_best = 1;
            result = child_result;
            result.action = actions[i];
        }
    }

    ctx->depth_counter = pool_idx;
    return result;
}

SearchResult maxn_search_deterministic(SearchCtx *ctx, Game *g,
                               Action *actions, int num_actions,
                               int depth, ValueFn eval_fn) {
    SearchResult out = {.value = 0, .action = {0}};
    if (num_actions <= 0) return out;
    int root_idx = g->state.current_player_index;
    MaxNResult r = maxn_recurse_deterministic(ctx, g, actions, num_actions,
                                              depth, eval_fn);
    out.action = r.action;
    out.value = r.values[root_idx];
    return out;
}

/* Compute expected value of taking `a` from `g`, expanding chance nodes
 * (ROLL: 11 dice outcomes, BUY_DEVELOPMENT_CARD: deck composition,
 * MOVE_ROBBER with steal: 5 resource outcomes). Mirrors Python's
 * tree_search_utils.execute_spectrum. */
static double eval_action_expected(SearchCtx *ctx, Game *g, Action a,
                                    int child_depth, double alpha, double beta,
                                    Color bot_color, ValueFn eval_fn,
                                    Game *child, Action *child_actions) {
    if (a.type == AT_ROLL) {
        double expected = 0.0;
        for (int sum = 2; sum <= 12; sum++) {
            game_copy(child, g);
            apply_roll_forced(&child->state, a, sum);
            int child_n = generate_playable_actions(&child->state,
                                                     child_actions, MAX_ACTIONS);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta,
                                                bot_color, eval_fn);
            expected += DICE_P_SUM[sum] * sr.value;
        }
        return expected;
    }

    if (a.type == AT_BUY_DEVELOPMENT_CARD) {
        /* Build imagined deck: real deck + opponent face-down cards */
        int counts[5] = {0};
        for (int i = 0; i < g->state.dev_deck_size; i++) {
            int c = g->state.development_listdeck[i];
            if (c >= 0 && c < 5) counts[c]++;
        }
        int self_idx = g->state.color_to_index[(int)a.color];
        for (int p = 0; p < g->state.num_players; p++) {
            if (p == self_idx) continue;
            for (int c = 0; c < 5; c++) {
                int n = g->state.player_state[p][PS_DEV_IN_HAND(c)];
                if (n > 0) counts[c] += n;
            }
        }
        int total = 0;
        for (int c = 0; c < 5; c++) total += counts[c];
        if (total == 0) {
            /* Fallback to deterministic top-of-deck */
            game_copy(child, g);
            int child_n;
            game_execute(child, a, child_actions, &child_n);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta,
                                                bot_color, eval_fn);
            return sr.value;
        }
        double expected = 0.0;
        for (int c = 0; c < 5; c++) {
            if (counts[c] == 0) continue;
            double p = (double)counts[c] / (double)total;
            game_copy(child, g);
            apply_buy_dev_card_forced(&child->state, a, c);
            int child_n = generate_playable_actions(&child->state,
                                                     child_actions, MAX_ACTIONS);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta,
                                                bot_color, eval_fn);
            expected += p * sr.value;
        }
        return expected;
    }

    if (a.type == AT_MOVE_ROBBER && a.value[3] != COLOR_NONE) {
        /* Check if anyone to actually steal from */
        int rc = a.value[3];
        int ri = g->state.color_to_index[rc];
        int hand_total = 0;
        for (int r = 0; r < 5; r++) {
            int amt = g->state.player_state[ri][PS_RESOURCE_IN_HAND(r)];
            if (amt > 0) hand_total += amt;
        }
        if (hand_total == 0) {
            /* Nothing to steal — deterministic */
            game_copy(child, g);
            int child_n;
            game_execute(child, a, child_actions, &child_n);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta,
                                                bot_color, eval_fn);
            return sr.value;
        }
        /* Python uses 1/5 per resource (uniform), even if the victim has 0
         * of some resources (impossible imagined outcomes are silently ignored
         * by apply_move_robber_forced). */
        double expected = 0.0;
        for (int r = 0; r < 5; r++) {
            game_copy(child, g);
            apply_move_robber_forced(&child->state, a, r);
            int child_n = generate_playable_actions(&child->state,
                                                     child_actions, MAX_ACTIONS);
            SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                                child_depth, alpha, beta,
                                                bot_color, eval_fn);
            expected += 0.2 * sr.value;
        }
        return expected;
    }

    /* Deterministic action */
    game_copy(child, g);
    int child_n;
    game_execute(child, a, child_actions, &child_n);
    SearchResult sr = alphabeta_search(ctx, child, child_actions, child_n,
                                        child_depth, alpha, beta,
                                        bot_color, eval_fn);
    return sr.value;
}

/* Deterministic known-future transition: every action is applied through
 * game_execute so the copied Game.rng, dev deck order, and robber steal RNG
 * define the sole child state. */
static double eval_action_deterministic(SearchCtx *ctx, Game *g, Action a,
                                         int child_depth, double alpha,
                                         double beta, Color bot_color,
                                         ValueFn eval_fn, Game *child,
                                         Action *child_actions) {
    game_copy(child, g);
    int child_n;
    game_execute(child, a, child_actions, &child_n);
    SearchResult sr = alphabeta_search_deterministic(ctx, child, child_actions,
                                                     child_n, child_depth,
                                                     alpha, beta,
                                                     bot_color, eval_fn);
    return sr.value;
}

Action random_player_decide(State *s, Action *actions, int n, RngState *rng) {
    (void)s;
    return actions[rng_choice_index(rng, n)];
}
