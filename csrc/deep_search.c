/* deep_search.c — Pure C implementation of SuperBotV3's recursive minimax
 * search. Designed for single-threaded performance on Apple Silicon.
 *
 * Compile:
 *   cc -shared -O3 -march=native -flto -fPIC -o libdeep.dylib \
 *      deep_search.c rng.c map.c board.c state.c actions.c \
 *      apply_action.c game.c value.c search.c \
 *      -lm -framework Accelerate
 */

#include "deep_search.h"
#include "actions.h"
#include "apply_action.h"
#include "board.h"
#include "policy_topk.h"
#include "state_encode.h"
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VALUE_SCALE   3e15
#define WIN_VAL        1.0
#define LOSS_VAL      -1.0
#define MAX_TURNS_RUN  500

struct DeepSearchCtx {
    /* Two policy modes:
     *   1. Python callback path: policy_fn != NULL, used for legacy compat.
     *   2. Pure-C path: c_nn_model + c_enc set, no Python crossings. */
    void *userdata;
    PolicyTopKFn policy_fn;

    /* Pure-C policy state (used when policy_fn == NULL). */
    NNModel *c_nn_model;
    StateEncoderC *c_enc;
    /* Scratch buffers for C policy_top_k */
    float c_nf_buf[ENC_NUM_NODES * ENC_NODE_FEAT_DIM];
    float c_ef_buf[ENC_NUM_EDGES * ENC_EDGE_FEAT_DIM];
    float c_ff_buf[ENC_FLAT_FEAT_DIM];
    float c_mk_buf[NN_MASK_DIM];
    float c_out_buf[4 + NN_MASK_DIM];

    /* Search configuration */
    int our_depth;
    int top_k_schedule[DS_MAX_K_SCHEDULE];
    int schedule_len;
    int opp_ab_depth;
    double time_budget_sec;
    double deadline_clock;
    int leaf_mode;
    int algo_policy;
    int opponent_model;  /* 0=AB search, 1=H-S top-1, 2=H-S one-ply leaf */
    int iter_enabled;
    int iter_start_depth;
    int crit_vp_threshold;
    int crit_extra_depth;

    /* Recursive game/action stack to avoid malloc per-node. */
    Game game_pool[DS_MAX_DEPTH];
    Action action_pool[DS_MAX_DEPTH][MAX_ACTIONS];
    int top_pool[DS_MAX_DEPTH][DS_MAX_K];

    /* Re-used opponent AB search context */
    SearchCtx ab_ctx;
    Action ab_buf[MAX_ACTIONS];

    /* Leaf cache (open addressing, always-replace) */
    uint64_t *cache_keys;
    double   *cache_vals;
    long      cache_size;
    long      cache_mask;

    /* Policy cache: hash → top-K legal action indices.
     * Indexed in same hash space; separate buckets. Stored as uint8 since
     * legal action counts rarely exceed 64. k=255 means empty. */
    uint64_t *pcache_keys;
    uint8_t  *pcache_data;   /* layout: [k, idx0, idx1, ..., idx_{DS_MAX_K-1}] per bucket */
    long      pcache_size;
    long      pcache_mask;

    /* Stats */
    DeepSearchStats stats;
};

/* ===== Time helpers ===== */
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ===== State hash (must match base_value_fn inputs) =====
 * Optimized: read 8 bytes at a time into a single uint64 for the bulk
 * arrays (buildings 96 bytes = 12 words, roads 288 bytes = 36 words).
 * Falls back to byte iteration for the trailing few bytes if any.
 */
static inline uint64_t state_hash(const Game *g) {
    const State *st = &g->state;
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint64_t P = 0x100000001b3ULL;

    /* Buildings (96 bytes = 12 × uint64) */
    {
        const uint64_t *p = (const uint64_t *)(const void *)st->board.buildings;
        for (int i = 0; i < 12; i++) h = (h ^ p[i]) * P;
    }

    /* Roads (288 bytes = 36 × uint64) */
    {
        const uint64_t *p = (const uint64_t *)(const void *)st->board.road_owner;
        for (int i = 0; i < 36; i++) h = (h ^ p[i]) * P;
    }

    /* Robber */
    h = (h ^ ((uint64_t)(st->board.robber_coordinate.x & 0xff)
              | ((uint64_t)(st->board.robber_coordinate.y & 0xff) << 8)
              | ((uint64_t)(st->board.robber_coordinate.z & 0xff) << 16))) * P;

    /* Per-player relevant fields. Pack into a single uint64 per player
     * to amortize the FNV step. */
    for (int p = 0; p < MAX_PLAYERS; p++) {
        const int *ps = st->player_state[p];
        /* Pack VP, longest road, played knight, and 5 resource counts (low byte each) */
        uint64_t pack = (uint64_t)(ps[0] & 0xff)               /* VP */
                      | ((uint64_t)(ps[9] & 0xff) << 8)         /* longest road */
                      | ((uint64_t)(ps[24] & 0xff) << 16)       /* played knight */
                      | ((uint64_t)(ps[14] & 0xff) << 24)       /* wood */
                      | ((uint64_t)(ps[15] & 0xff) << 32)       /* brick */
                      | ((uint64_t)(ps[16] & 0xff) << 40)       /* sheep */
                      | ((uint64_t)(ps[17] & 0xff) << 48)       /* wheat */
                      | ((uint64_t)(ps[18] & 0xff) << 56);      /* ore */
        h = (h ^ pack) * P;
        /* Dev cards in hand: 5 fields (19..23) */
        uint64_t pack2 = (uint64_t)(ps[19] & 0xff)
                       | ((uint64_t)(ps[20] & 0xff) << 8)
                       | ((uint64_t)(ps[21] & 0xff) << 16)
                       | ((uint64_t)(ps[22] & 0xff) << 24)
                       | ((uint64_t)(ps[23] & 0xff) << 32)
                       | ((uint64_t)p << 56);  /* mix in player index */
        h = (h ^ pack2) * P;
    }

    /* Current player + prompt + turn number */
    h = (h ^ ((uint64_t)(st->current_player_index & 0xff)
              | ((uint64_t)(st->current_prompt & 0xff) << 8)
              | ((uint64_t)(st->num_turns & 0xffff) << 16))) * P;
    return h;
}

/* Policy top-K cache: keyed by full state hash + depth + branching parameters.
 * Stores indices into the scratch `actions` array produced by generate_playable_actions.
 */
static inline uint64_t policy_pcache_key(const Game *g, int depth_idx, int k_cap, int n_actions) {
    uint64_t h = state_hash(g);
    h ^= ((uint64_t)(depth_idx & 0x1f) << 59);
    h ^= ((uint64_t)(k_cap & 0x1f) << 54);
    h ^= ((uint64_t)(n_actions & 0xff) << 46);
    return h;
}

static int policy_pcache_try_get(DeepSearchCtx *ctx, uint64_t ph, int *top, int expect_top) {
    if (ctx->pcache_size <= 0 || ph == 0) return -1;
    long pidx = (long)(ph & ctx->pcache_mask);
    if (ctx->pcache_keys[pidx] != ph) return -1;
    uint8_t *pd = ctx->pcache_data + pidx * (1 + DS_MAX_K);
    int nk = (int)pd[0];
    if (nk <= 0 || nk > DS_MAX_K) return -1;
    if (nk != expect_top) return -1;
    ctx->stats.n_pcache_hits++;
    for (int i = 0; i < nk; i++) top[i] = (int)pd[1 + i];
    return nk;
}

static void policy_pcache_put(DeepSearchCtx *ctx, uint64_t ph, const int *top, int n_top) {
    if (ctx->pcache_size <= 0 || ph == 0 || n_top <= 0) return;
    if (n_top > DS_MAX_K) n_top = DS_MAX_K;
    long pidx = (long)(ph & ctx->pcache_mask);
    ctx->pcache_keys[pidx] = ph;
    uint8_t *pd = ctx->pcache_data + pidx * (1 + DS_MAX_K);
    pd[0] = (uint8_t)n_top;
    for (int i = 0; i < n_top; i++) pd[1 + i] = (uint8_t)top[i];
}

static inline double leaf_value_no_cache(Game *g, Color our_color, int leaf_mode) {
    Color w = game_winning_color(g);
    if (w != COLOR_NONE) {
        return (w == our_color) ? WIN_VAL : LOSS_VAL;
    }

    double raw;
    if (leaf_mode == 1) {
        raw = base_value_fn_enemy_vp(g, our_color);
    } else if (leaf_mode == 2) {
        raw = base_value_fn_enemy_all_vp_prod(g, our_color);
    } else if (leaf_mode == 3) {
        raw = base_value_fn_enemy_leader(g, our_color);
    } else if (leaf_mode == 4) {
        raw = base_value_fn_enemy_full(g, our_color);
    } else {
        raw = base_value_fn(g, our_color);
    }
    double v = raw / VALUE_SCALE;
    if (v > 0.99) v = 0.99;
    if (v < -0.99) v = -0.99;
    return v;
}

/* ===== Leaf evaluation (with cache) ===== */
static inline double leaf_value(DeepSearchCtx *ctx, Game *g, Color our_color) {
    /* Terminal */
    Color w = game_winning_color(g);
    if (w != COLOR_NONE) {
        return (w == our_color) ? WIN_VAL : LOSS_VAL;
    }

    uint64_t h = 0;
    if (ctx->cache_size > 0) {
        h = state_hash(g);
        long idx = (long)(h & ctx->cache_mask);
        if (ctx->cache_keys[idx] == h && h != 0) {
            ctx->stats.n_cache_hits++;
            return ctx->cache_vals[idx];
        }
        ctx->stats.n_cache_misses++;
    }

    double v = leaf_value_no_cache(g, our_color, ctx->leaf_mode);

    if (ctx->cache_size > 0 && h != 0) {
        long idx = (long)(h & ctx->cache_mask);
        ctx->cache_keys[idx] = h;
        ctx->cache_vals[idx] = v;
    }
    return v;
}

/* ===== Opponent AB2 choose (matches Python _ab2_choose) ===== */
static inline int ab2_choose_idx(DeepSearchCtx *ctx, Game *g,
                                  Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    Color cp = g->state.colors[g->state.current_player_index];
    /* Copy actions to ab_buf since alphabeta_search may reorder them */
    for (int i = 0; i < n; i++) ctx->ab_buf[i] = actions[i];
    SearchResult res = alphabeta_search(&ctx->ab_ctx, g, ctx->ab_buf, n,
                                         ctx->opp_ab_depth, -1e30, 1e30,
                                         cp, base_value_fn);
    /* Find the chosen action's index in the original `actions` */
    for (int i = 0; i < n; i++) {
        if (memcmp(&res.action, &actions[i], sizeof(Action)) == 0) {
            return i;
        }
    }
    return 0;
}

static inline int hs_policy_choose_idx(DeepSearchCtx *ctx, Game *g,
                                       Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;

    int top[DS_MAX_K];
    int k = 6;
    if (k > n) k = n;
    int n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, actions, n, k,
                                top, ctx->c_nf_buf, ctx->c_ef_buf, ctx->c_ff_buf,
                                ctx->c_mk_buf, ctx->c_out_buf, 1);
    if (n_top <= 0) return 0;
    return top[0];
}

static inline int hs_leaf_choose_idx(DeepSearchCtx *ctx, Game *g,
                                     Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;

    int top[DS_MAX_K];
    int k = 6;
    if (k > n) k = n;
    int n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, actions, n, k,
                                top, ctx->c_nf_buf, ctx->c_ef_buf, ctx->c_ff_buf,
                                ctx->c_mk_buf, ctx->c_out_buf, 1);
    if (n_top <= 0) return 0;

    Color cp = g->state.colors[g->state.current_player_index];
    int best_idx = top[0];
    double best_v = -2.0;
    for (int i = 0; i < n_top; i++) {
        Game child;
        game_copy(&child, g);
        int dummy;
        game_execute(&child, actions[top[i]], ctx->ab_buf, &dummy);
        double v = leaf_value_no_cache(&child, cp, ctx->leaf_mode);
        if (v > best_v) {
            best_v = v;
            best_idx = top[i];
        }
    }
    return best_idx;
}

/* ===== Fast-forward through opponent turns =====
 * Returns:
 *   -1 if game terminated (no actions to take)
 *   N>=2 if our turn with N legal actions (filled into scratch_actions)
 *   1 should never happen (forced moves auto-applied)
 */
static int fast_forward_to_us(DeepSearchCtx *ctx, Game *g, Color our_color,
                                Action *scratch_actions) {
    int n;
    int safety = 0;
    while (safety++ < 200) {
        if (game_winning_color(g) != COLOR_NONE) return -1;
        if (g->state.num_turns >= MAX_TURNS_RUN) return -1;
        Color cp = g->state.colors[g->state.current_player_index];
        n = generate_playable_actions(&g->state, scratch_actions, MAX_ACTIONS);
        if (n == 0) return -1;
        if (n == 1) {
            int dummy;
            game_execute(g, scratch_actions[0], scratch_actions, &dummy);
            continue;
        }
        if (cp == our_color) {
            return n;  /* our turn, scratch already populated */
        }
        int chosen;
        if (ctx->opponent_model == 2 && ctx->c_enc != NULL) {
            chosen = hs_leaf_choose_idx(ctx, g, scratch_actions, n);
        } else if (ctx->opponent_model == 1 && ctx->c_enc != NULL) {
            chosen = hs_policy_choose_idx(ctx, g, scratch_actions, n);
        } else {
            chosen = ab2_choose_idx(ctx, g, scratch_actions, n);
        }
        int dummy;
        game_execute(g, scratch_actions[chosen], scratch_actions, &dummy);
    }
    return -1;
}

/* ===== Recursive deep search =====
 * `ext_left` is the remaining critical-state extension budget for this branch.
 * When the position has any opponent at VP >= crit_vp_threshold, we extend
 * the search by 1 ply (consuming one unit of `ext_left`).
 */
static double deep_search_recurse(DeepSearchCtx *ctx, Game *g,
                                    Color our_color, int depth_left,
                                    int depth_idx, double alpha, double beta,
                                    int ext_left) {
    /* Time budget check (cheap) */
    if ((ctx->stats.n_calls & 0xff) == 0) {
        if (now_sec() >= ctx->deadline_clock) {
            return leaf_value(ctx, g, our_color);
        }
    }
    ctx->stats.n_calls++;

    if (depth_idx >= DS_MAX_DEPTH - 1) {
        ctx->stats.n_leaves++;
        return leaf_value(ctx, g, our_color);
    }

    Action *scratch = ctx->action_pool[depth_idx];

    int n = fast_forward_to_us(ctx, g, our_color, scratch);
    if (n < 0) {
        ctx->stats.n_leaves++;
        return leaf_value(ctx, g, our_color);
    }

    if (depth_left == 0) {
        /* Critical-state extension: if any opponent is close to winning, take
         * one extra ply on this branch. Only allowed when budget is positive. */
        if (ext_left > 0 && ctx->crit_extra_depth > 0) {
            int max_enemy_vp = 0;
            const State *st = &g->state;
            for (int i = 0; i < st->num_players; i++) {
                if (st->colors[i] == our_color) continue;
                int vp = st->player_state[i][PS_VICTORY_POINTS];
                if (vp > max_enemy_vp) max_enemy_vp = vp;
            }
            if (max_enemy_vp >= ctx->crit_vp_threshold) {
                depth_left = 1;
                ext_left--;
            }
        }
        if (depth_left == 0) {
            ctx->stats.n_leaves++;
            return leaf_value(ctx, g, our_color);
        }
    }

    /* Terminal-win shortcut: if any move wins immediately, return WIN_VAL. */
    for (int i = 0; i < n; i++) {
        Game *child = &ctx->game_pool[depth_idx];
        game_copy(child, g);
        int dummy;
        game_execute(child, scratch[i], ctx->ab_buf, &dummy);
        if (game_winning_color(child) == our_color) {
            ctx->stats.n_terminal_short++;
            return WIN_VAL;
        }
    }

    /* Get top-K from policy. Two modes:
     *   - C path (c_nn_model set): pure-C policy_top_k, zero FFI.
     *   - Python path (policy_fn set): callback into Python NN. */
    int k_idx = depth_idx < ctx->schedule_len ? depth_idx : ctx->schedule_len - 1;
    int k = ctx->top_k_schedule[k_idx];
    if (k > n) k = n;
    if (k > DS_MAX_K) k = DS_MAX_K;
    int *top = ctx->top_pool[depth_idx];
    int n_top = -1;
    if (ctx->c_nn_model != NULL && ctx->c_enc != NULL) {
        uint64_t pkey = policy_pcache_key(g, depth_idx, k, n);
        n_top = policy_pcache_try_get(ctx, pkey, top, k);
        if (n_top < 0) {
            ctx->stats.n_pcache_misses++;
            n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, scratch, n, k,
                                    top, ctx->c_nf_buf, ctx->c_ef_buf, ctx->c_ff_buf,
                                    ctx->c_mk_buf, ctx->c_out_buf,
                                    ctx->algo_policy);
            if (n_top > 0) policy_pcache_put(ctx, pkey, top, n_top);
        }
    } else {
        n_top = ctx->policy_fn(ctx->userdata, g, scratch, n, k, top);
    }

    if (n_top == 0) {
        ctx->stats.n_leaves++;
        return leaf_value(ctx, g, our_color);
    }

    double best_v = -2.0;
    for (int i = 0; i < n_top; i++) {
        Game *child = &ctx->game_pool[depth_idx];
        game_copy(child, g);
        int dummy;
        game_execute(child, scratch[top[i]], ctx->ab_buf, &dummy);

        double v = deep_search_recurse(ctx, child, our_color,
                                         depth_left - 1, depth_idx + 1,
                                         alpha, beta, ext_left);
        if (v > best_v) best_v = v;
        if (v > alpha) alpha = v;
        if (alpha >= beta) {
            ctx->stats.n_pruned += (n_top - i - 1);
            break;
        }
        if (best_v >= WIN_VAL - 1e-6) break;
    }
    return best_v;
}

/* ===== Public API ===== */
static DeepSearchCtx *_deep_search_create_common(int log2_cache_size) {
    DeepSearchCtx *ctx = calloc(1, sizeof(DeepSearchCtx));
    if (!ctx) return NULL;
    ctx->our_depth = 4;
    ctx->schedule_len = 1;
    ctx->top_k_schedule[0] = 5;
    ctx->opp_ab_depth = 2;
    ctx->time_budget_sec = 5.0;
    ctx->iter_enabled = 0;
    ctx->iter_start_depth = 2;
    ctx->crit_vp_threshold = 100;  /* effectively disabled by default */
    ctx->crit_extra_depth = 0;
    if (log2_cache_size > 0) {
        ctx->cache_size = 1L << log2_cache_size;
        ctx->cache_mask = ctx->cache_size - 1;
        ctx->cache_keys = calloc(ctx->cache_size, sizeof(uint64_t));
        ctx->cache_vals = calloc(ctx->cache_size, sizeof(double));
        ctx->pcache_size = ctx->cache_size;
        ctx->pcache_mask = ctx->pcache_size - 1;
        ctx->pcache_keys = calloc(ctx->pcache_size, sizeof(uint64_t));
        ctx->pcache_data = calloc(ctx->pcache_size, (1 + DS_MAX_K) * sizeof(uint8_t));
        if (!ctx->cache_keys || !ctx->cache_vals
            || !ctx->pcache_keys || !ctx->pcache_data) {
            free(ctx->cache_keys); free(ctx->cache_vals);
            free(ctx->pcache_keys); free(ctx->pcache_data);
            free(ctx);
            return NULL;
        }
    }
    return ctx;
}

DeepSearchCtx *deep_search_create(int log2_cache_size, void *userdata,
                                    PolicyTopKFn policy_fn) {
    DeepSearchCtx *ctx = _deep_search_create_common(log2_cache_size);
    if (!ctx) return NULL;
    ctx->userdata = userdata;
    ctx->policy_fn = policy_fn;
    ctx->c_nn_model = NULL;
    ctx->c_enc = NULL;
    return ctx;
}

DeepSearchCtx *deep_search_create_c(int log2_cache_size,
                                     struct NNModel *nn_model,
                                     struct StateEncoderC *enc) {
    DeepSearchCtx *ctx = _deep_search_create_common(log2_cache_size);
    if (!ctx) return NULL;
    ctx->userdata = NULL;
    ctx->policy_fn = NULL;
    ctx->c_nn_model = (NNModel *)nn_model;
    ctx->c_enc = (StateEncoderC *)enc;
    return ctx;
}

void deep_search_destroy(DeepSearchCtx *ctx) {
    if (!ctx) return;
    free(ctx->cache_keys);
    free(ctx->cache_vals);
    free(ctx->pcache_keys);
    free(ctx->pcache_data);
    free(ctx);
}

void deep_search_configure(DeepSearchCtx *ctx,
                           int our_depth,
                           const int *top_k_schedule, int schedule_len,
                           int opponent_ab_depth,
                           double time_budget_sec) {
    ctx->our_depth = our_depth;
    if (schedule_len > DS_MAX_K_SCHEDULE) schedule_len = DS_MAX_K_SCHEDULE;
    for (int i = 0; i < schedule_len; i++) {
        int k = top_k_schedule[i];
        if (k < 1) k = 1;
        if (k > DS_MAX_K) k = DS_MAX_K;
        ctx->top_k_schedule[i] = k;
    }
    ctx->schedule_len = schedule_len;
    ctx->opp_ab_depth = opponent_ab_depth;
    ctx->time_budget_sec = time_budget_sec;
}

void deep_search_set_leaf_mode(DeepSearchCtx *ctx, int leaf_mode) {
    if (!ctx) return;
    ctx->leaf_mode = leaf_mode;
    if (ctx->cache_keys && ctx->cache_size > 0)
        memset(ctx->cache_keys, 0, (size_t)ctx->cache_size * sizeof(uint64_t));
    if (ctx->cache_vals && ctx->cache_size > 0)
        memset(ctx->cache_vals, 0, (size_t)ctx->cache_size * sizeof(double));
}

void deep_search_set_algo_policy(DeepSearchCtx *ctx, int enabled) {
    if (!ctx) return;
    ctx->algo_policy = enabled ? 1 : 0;
}

void deep_search_set_opponent_model(DeepSearchCtx *ctx, int model) {
    if (!ctx) return;
    ctx->opponent_model = model;
}

void deep_search_set_iterative(DeepSearchCtx *ctx, int enabled, int start_depth) {
    if (!ctx) return;
    ctx->iter_enabled = enabled ? 1 : 0;
    ctx->iter_start_depth = start_depth;
}

void deep_search_set_critical_extension(DeepSearchCtx *ctx,
                                         int vp_threshold, int extra_depth) {
    if (!ctx) return;
    ctx->crit_vp_threshold = vp_threshold;
    ctx->crit_extra_depth = extra_depth;
}

double deep_search_root(DeepSearchCtx *ctx, Game *game, Color our_color,
                         int *root_candidate_indices, int n_root_candidates,
                         int *out_best_idx) {
    if (out_best_idx) *out_best_idx = -1;
    if (n_root_candidates <= 0) return 0.0;

    deep_search_reset_stats(ctx);
    ctx->deadline_clock = now_sec() + ctx->time_budget_sec;
    ctx->ab_ctx.depth_counter = 0;
    ctx->ab_ctx.same_turn_mode = 0;

    /* Initialize STATIC_ADJ in libdeep's address space using the map
     * pointer from the Game (which was allocated by libcatan but the
     * map data structure is well-defined). board_init_static_graph()
     * is idempotent (no-op after first call) so this is cheap. */
    if (game->state.board.map) {
        board_init_static_graph(game->state.board.map);
    }

    /* Get root legal actions to map indices */
    Action root_actions[MAX_ACTIONS];
    int n_actions = generate_playable_actions(&game->state, root_actions, MAX_ACTIONS);
    if (n_actions == 0) return 0.0;

    /* Iterative deepening: search at increasing depths and reorder
     * candidates by value between iterations for better alpha-beta cuts. */
    if (ctx->iter_enabled && ctx->our_depth >= 2) {
        int n = n_root_candidates;
        int order[MAX_ACTIONS];
        double values[MAX_ACTIONS];
        for (int i = 0; i < n; i++) order[i] = root_candidate_indices[i];

        int target_depth = ctx->our_depth;
        int start_depth = ctx->iter_start_depth > 0 ? ctx->iter_start_depth : 2;
        if (start_depth > target_depth) start_depth = target_depth;

        int best_pi = 0;
        double best_v = -2.0;
        int best_chosen_ci = -1;
        int have_completed = 0;

        for (int d_iter = start_depth; d_iter <= target_depth; d_iter++) {
            double iter_alpha = -2.0;
            double iter_best = -2.0;
            int iter_best_pi = 0;
            int completed = 1;
            int win_break = 0;

            for (int i = 0; i < n; i++) {
                values[i] = -2.0;
                int ci = order[i];
                if (ci < 0 || ci >= n_actions) continue;

                if (now_sec() >= ctx->deadline_clock) {
                    completed = 0;
                    break;
                }

                Game child;
                game_copy(&child, game);
                int dummy;
                game_execute(&child, root_actions[ci], ctx->ab_buf, &dummy);

                double v = deep_search_recurse(ctx, &child, our_color,
                                                 d_iter - 1, 1,
                                                 iter_alpha, 2.0,
                                                 ctx->crit_extra_depth);
                values[i] = v;
                if (v > iter_best) {
                    iter_best = v;
                    iter_best_pi = i;
                }
                if (v > iter_alpha) iter_alpha = v;
                if (iter_best >= WIN_VAL - 1e-6) {
                    win_break = 1;
                    ctx->stats.n_root_early_exits++;
                    break;
                }
            }

            if (completed) {
                best_v = iter_best;
                best_chosen_ci = order[iter_best_pi];
                have_completed = 1;
                if (win_break) break;

                /* Sort (order, values) by values descending for next iteration. */
                for (int i = 1; i < n; i++) {
                    int j = i;
                    while (j > 0 && values[j] > values[j - 1]) {
                        double tv = values[j]; values[j] = values[j - 1]; values[j - 1] = tv;
                        int to = order[j]; order[j] = order[j - 1]; order[j - 1] = to;
                        j--;
                    }
                }
            } else {
                /* Time ran out mid-iteration: keep last completed best. */
                break;
            }
        }

        if (!have_completed) {
            /* Fall back to a depth-1 pass if everything timed out. */
            best_v = -2.0;
            for (int i = 0; i < n_root_candidates; i++) {
                int ci = root_candidate_indices[i];
                if (ci < 0 || ci >= n_actions) continue;
                Game child;
                game_copy(&child, game);
                int dummy;
                game_execute(&child, root_actions[ci], ctx->ab_buf, &dummy);
                double v = leaf_value(ctx, &child, our_color);
                if (v > best_v) { best_v = v; best_pi = i; }
            }
        } else {
            best_pi = 0;
            for (int k = 0; k < n_root_candidates; k++) {
                if (root_candidate_indices[k] == best_chosen_ci) {
                    best_pi = k;
                    break;
                }
            }
        }

        if (out_best_idx) *out_best_idx = best_pi;
        return best_v;
    }

    double best_v = -2.0;
    int best_pi = 0;
    double alpha = -2.0;
    double beta = 2.0;

    for (int i = 0; i < n_root_candidates; i++) {
        int ci = root_candidate_indices[i];
        if (ci < 0 || ci >= n_actions) continue;

        Game child;
        game_copy(&child, game);
        int dummy;
        game_execute(&child, root_actions[ci], ctx->ab_buf, &dummy);

        double v = deep_search_recurse(ctx, &child, our_color,
                                         ctx->our_depth - 1, 1,
                                         alpha, 2.0,
                                         ctx->crit_extra_depth);
        if (v > best_v) {
            best_v = v;
            best_pi = i;
        }
        if (v > alpha) alpha = v;
        if (best_v >= WIN_VAL - 1e-6) {
            ctx->stats.n_root_early_exits++;
            break;
        }
        if (now_sec() >= ctx->deadline_clock) break;
    }

    if (out_best_idx) *out_best_idx = best_pi;
    return best_v;
}


/* deep_search_root_full: same as deep_search_root but evaluates every
 * candidate with a FULL window (no root-level alpha-beta pruning) and
 * fills out_values[] with each candidate's exact minimax value.
 *
 * Returns best value (= max(out_values)). Untouched entries (those past
 * the deadline) are left at -2.0 so the caller can detect timeouts. */
double deep_search_root_full(DeepSearchCtx *ctx, Game *game, Color our_color,
                              int *root_candidate_indices, int n_root_candidates,
                              int *out_best_idx, double *out_values) {
    if (out_best_idx) *out_best_idx = -1;
    if (out_values) {
        for (int i = 0; i < n_root_candidates; i++) out_values[i] = -2.0;
    }
    if (n_root_candidates <= 0) return 0.0;

    deep_search_reset_stats(ctx);
    ctx->deadline_clock = now_sec() + ctx->time_budget_sec;
    ctx->ab_ctx.depth_counter = 0;
    ctx->ab_ctx.same_turn_mode = 0;

    if (game->state.board.map) {
        board_init_static_graph(game->state.board.map);
    }

    Action root_actions[MAX_ACTIONS];
    int n_actions = generate_playable_actions(&game->state, root_actions, MAX_ACTIONS);
    if (n_actions == 0) return 0.0;

    double best_v = -2.0;
    int best_pi = 0;

    for (int i = 0; i < n_root_candidates; i++) {
        int ci = root_candidate_indices[i];
        if (ci < 0 || ci >= n_actions) continue;

        Game child;
        game_copy(&child, game);
        int dummy;
        game_execute(&child, root_actions[ci], ctx->ab_buf, &dummy);

        /* FULL window per candidate (alpha=-2, beta=2): no root-level pruning.
         * Internal pruning inside deep_search_recurse still applies but the
         * returned value is exact for this root branch. */
        double v = deep_search_recurse(ctx, &child, our_color,
                                         ctx->our_depth - 1, 1,
                                         -2.0, 2.0,
                                         ctx->crit_extra_depth);
        if (out_values) out_values[i] = v;
        if (v > best_v) {
            best_v = v;
            best_pi = i;
        }
        if (best_v >= WIN_VAL - 1e-6) {
            ctx->stats.n_root_early_exits++;
            break;
        }
        if (now_sec() >= ctx->deadline_clock) break;
    }

    if (out_best_idx) *out_best_idx = best_pi;
    return best_v;
}

void deep_search_get_stats(DeepSearchCtx *ctx, DeepSearchStats *out) {
    *out = ctx->stats;
}

void deep_search_reset_stats(DeepSearchCtx *ctx) {
    memset(&ctx->stats, 0, sizeof(ctx->stats));
}
