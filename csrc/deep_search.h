#ifndef DEEP_SEARCH_H
#define DEEP_SEARCH_H

#include "game.h"
#include "search.h"
#include "value.h"

/* Deep recursive minimax search with NN policy pruning at branching points.
 *
 * Algorithm matches the Python SuperBotV3 exactly:
 *   1. Fast-forward through opponent turns + forced moves until our turn.
 *   2. If terminal: return ±1.0 (win/loss) or 0.0 (draw).
 *   3. If depth_left == 0: return normalized AB-leaf value in [-1, 1].
 *   4. Otherwise: branch over top-K policy candidates (K from schedule),
 *      recursively evaluate each, return max. Alpha-beta pruning.
 *
 * Performance notes:
 *   - Inner game_copy is the only memory allocation; everything else is on stack.
 *   - State hash for leaf cache is computed in C (vs ~200us in Python).
 *   - Single FFI call per decision (vs thousands in pure Python).
 */

typedef struct DeepSearchCtx DeepSearchCtx;

#define DS_MAX_DEPTH      24
#define DS_MAX_K          16
#define DS_MAX_K_SCHEDULE 16

/* Caller-supplied policy callback: fills out[K] with the legal action indices
 * sorted by NN policy logit (descending). Returns the number actually written
 * (<= min(num_actions, k)). */
typedef int (*PolicyTopKFn)(void *userdata, Game *g, Action *actions, int n,
                             int k, int *out_indices);

DeepSearchCtx *deep_search_create(int log2_cache_size, void *userdata,
                                   PolicyTopKFn policy_fn);

/* Alternative constructor: use the built-in C policy_top_k (encode +
 * nn_forward + top-K) instead of a Python callback. The DeepSearchCtx
 * takes ownership of nothing — caller manages nn_model and enc lifetime.
 *
 * This eliminates ALL Python overhead from the inner loop. */
struct StateEncoderC;  /* fwd decl */
struct NNModel;         /* fwd decl */
DeepSearchCtx *deep_search_create_c(int log2_cache_size,
                                     struct NNModel *nn_model,
                                     struct StateEncoderC *enc);

void deep_search_destroy(DeepSearchCtx *ctx);

/* Configure the search. Top-K schedule: schedule[i] = K used at depth i.
 * If our_depth > len, the last value is used for deeper levels. */
void deep_search_configure(DeepSearchCtx *ctx,
                           int our_depth,
                           const int *top_k_schedule, int schedule_len,
                           int opponent_ab_depth,
                           double time_budget_sec);

/* Leaf eval mode:
 *   0=original base_value_fn
 *   1=top-2 enemy visible VP penalty
 *   2=all enemy VP+production at 0.1x
 *   3=leader full feature pressure at 0.1x
 *   4=all enemy full feature pressure at 0.1x */
void deep_search_set_leaf_mode(DeepSearchCtx *ctx, int leaf_mode);
void deep_search_set_algo_policy(DeepSearchCtx *ctx, int enabled);
/* Opponent moves during fast-forward:
 *   0=AB search (default)
 *   1=H-S heuristic policy top-1
 *   2=H-S top-6, choose best one-ply leaf value */
void deep_search_set_opponent_model(DeepSearchCtx *ctx, int model);

/* Iterative deepening at root. start_depth must be >= 1. Each iteration
 * reorders root candidates by previous-iteration values for stronger
 * alpha-beta cuts in the next iteration. Set 0 to disable. */
void deep_search_set_iterative(DeepSearchCtx *ctx, int enabled, int start_depth);

/* Critical-state extension: when any opponent has VP >= threshold, allow up
 * to `extra_depth` extra plies on the path. Set 0 to disable. */
void deep_search_set_critical_extension(DeepSearchCtx *ctx,
                                         int vp_threshold, int extra_depth);

/* Run search from `game` for `our_color`. Returns value in [-1, 1].
 * On entry, `game` should be at our turn; if not, the search will fast-forward.
 * `out_best_action_index` is filled with the index into `root_candidates`
 * of the best branch (or -1 if no candidates). */
double deep_search_root(DeepSearchCtx *ctx, Game *game, Color our_color,
                         int *root_candidate_indices, int n_root_candidates,
                         int *out_best_idx);

/* Same as deep_search_root, but ALSO fills out_values[i] with the exact
 * minimax value for each root candidate, evaluated with the FULL alpha-beta
 * window (no root-level pruning), so values are exact and comparable across
 * candidates. Costs more CPU than the pruned version but produces a denser
 * training signal: each candidate's value is exact, not just bounded.
 *
 * `out_values` must have length >= n_root_candidates.
 * Returns best value (max over out_values).
 *
 * Internal AB pruning still applies (alpha/beta inside recursion), so
 * out_values[i] is the true minimax value of candidate i, computed without
 * help from previously-found candidates. */
double deep_search_root_full(DeepSearchCtx *ctx, Game *game, Color our_color,
                              int *root_candidate_indices, int n_root_candidates,
                              int *out_best_idx, double *out_values);

/* Stats for diagnostics. */
typedef struct {
    long n_calls;
    long n_leaves;
    long n_pruned;
    long n_terminal_short;
    long n_cache_hits;       /* leaf eval cache */
    long n_cache_misses;
    long n_pcache_hits;      /* policy cache */
    long n_pcache_misses;    /* policy cache miss = NN forward in policy_top_k */
    long n_root_early_exits; /* deep_search_root stopped after proven win */
} DeepSearchStats;

void deep_search_get_stats(DeepSearchCtx *ctx, DeepSearchStats *out);
void deep_search_reset_stats(DeepSearchCtx *ctx);

#endif
