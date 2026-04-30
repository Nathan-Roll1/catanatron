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
#include "nn.h"
#include "state_encode.h"
DeepSearchCtx *deep_search_create_c(int log2_cache_size,
                                     NNModel *nn_model,
                                     StateEncoderC *enc);
DeepSearchCtx *deep_search_clone_config(const DeepSearchCtx *src);

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
 *   4=opponent-aware full pressure at 0.1x
 *   5=known-future race/threat value function
 *   6=Leaf5 plus exact next-roll peek at pre-roll leaves
 *   7=Leaf5 FFA leader-heavy pressure
 *   8=Leaf5 FFA threat-heavy pressure
 *   9=Leaf5 FFA low-pressure/selfish profile
 *   10=Leaf5 FFA moderate leader/threat pressure
 *   11=Leaf5 FFA moderate leader-only pressure
 *   12=Leaf5 FFA dynamic urgent-leader pressure
 *   13=Leaf5 FFA adaptive early-selfish/late-leader pressure
 *   14=Leaf7 with exact next-roll leaf awareness
 *   16=Leaf5 FFA high leader-pressure profile */
void deep_search_set_leaf_mode(DeepSearchCtx *ctx, int leaf_mode);

/* Recursive search enhancements, all optional and independently testable.
 * TT entries store exact/lower/upper bounds and a best move for ordering. */
void deep_search_set_tt_bits(DeepSearchCtx *ctx, int tt_bits);
void deep_search_set_search_enhancements(DeepSearchCtx *ctx,
                                         int pvs_enabled,
                                         int lmr_enabled);
void deep_search_set_iterative_deepening(DeepSearchCtx *ctx, int enabled);
void deep_search_set_candidate_rescue(DeepSearchCtx *ctx, int enabled);
void deep_search_set_leaf_extension(DeepSearchCtx *ctx, int enabled);

/* Use the built-in hand-coded no-ML policy for move ordering. */
void deep_search_set_algo_policy(DeepSearchCtx *ctx, int enabled);

/* Opponent moves during fast-forward:
 *   0=AB2 search (default)
 *   1=H-S heuristic policy top-1
 *   2=H-S top-6, choose best one-ply leaf value
 *   3=deterministic AB2 search using known Game.rng future
 *   4=deterministic AB2 with known-future evaluator
 *   5=deterministic MaxN/selfish search
 *   6=deterministic MaxN/selfish search with known-future evaluator
 *   7=nested deterministic H-S opponent model, width 2
 *   8=nested deterministic H-S opponent model, width 3
 *   9=nested deterministic H-S opponent model, width 4 */
void deep_search_set_opponent_model(DeepSearchCtx *ctx, int model);

/* Run search from `game` for `our_color`. Returns value in [-1, 1].
 * On entry, `game` should be at our turn; if not, the search will fast-forward.
 * `out_best_action_index` is filled with the index into `root_candidates`
 * of the best branch (or -1 if no candidates). */
double deep_search_root(DeepSearchCtx *ctx, Game *game, Color our_color,
                         int *root_candidate_indices, int n_root_candidates,
                         int *out_best_idx);
double deep_search_root_scores(DeepSearchCtx *ctx, Game *game, Color our_color,
                               int *root_candidate_indices,
                               int n_root_candidates,
                               int *out_best_idx,
                               double *out_values,
                               int *out_valid);
double deep_search_root_parallel(DeepSearchCtx *ctx, Game *game, Color our_color,
                                 int *root_candidate_indices,
                                 int n_root_candidates,
                                 int *out_best_idx,
                                 DeepSearchCtx **workers,
                                 int n_workers);
double deep_search_root_parallel_scores(DeepSearchCtx *ctx, Game *game,
                                        Color our_color,
                                        int *root_candidate_indices,
                                        int n_root_candidates,
                                        int *out_best_idx,
                                        DeepSearchCtx **workers,
                                        int n_workers,
                                        double *out_values,
                                        int *out_valid);

/* Stats for diagnostics. */
typedef struct {
    long n_calls;
    long n_leaves;
    long n_pruned;
    long n_terminal_short;
    long n_cache_hits;       /* leaf eval cache */
    long n_cache_misses;
    long n_pcache_hits;      /* policy cache */
    long n_pcache_misses;
} DeepSearchStats;

void deep_search_get_stats(DeepSearchCtx *ctx, DeepSearchStats *out);
void deep_search_reset_stats(DeepSearchCtx *ctx);
void deep_search_clear_caches(DeepSearchCtx *ctx);

#endif
