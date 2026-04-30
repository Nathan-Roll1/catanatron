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
#include "state_hash.h"
#include "state_encode.h"
#include "nn.h"
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VALUE_SCALE   3e15
#define WIN_VAL        1.0
#define LOSS_VAL      -1.0
#define MAX_TURNS_RUN  500
#define PVS_EPS        1e-6

typedef enum {
    TT_EMPTY = 0,
    TT_EXACT = 1,
    TT_LOWER = 2,
    TT_UPPER = 3
} TTFlag;

typedef struct {
    uint64_t key;
    double value;
    int depth;
    uint8_t flag;
    uint8_t best_valid;
    Action best_action;
} TTEntry;

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
    int opponent_model;
    int cache_bits;
    int tt_bits;
    int pvs_enabled;
    int lmr_enabled;
    int id_enabled;
    int candidate_rescue;
    int leaf_extension;

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

    /* Recursive transposition table: bounds + best action ordering. */
    TTEntry  *tt_entries;
    long      tt_size;
    long      tt_mask;

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

/* ===== Recursive transposition table ===== */
static inline uint64_t tt_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x ? x : 0x9e3779b97f4a7c15ULL;
}

static inline uint64_t tt_key(const Game *g, Color our_color) {
    uint64_t h = game_dynamic_hash(g);
    h ^= 0x6eed0e9da4d94a4fULL + ((uint64_t)(our_color + 17) << 32);
    return tt_mix64(h);
}

static inline int tt_probe(DeepSearchCtx *ctx, uint64_t key, int depth_left,
                           double *alpha, double *beta, double *out_value,
                           Action *out_best, int *out_has_best) {
    if (out_has_best) *out_has_best = 0;
    if (ctx->tt_size <= 0 || ctx->tt_entries == NULL) return 0;

    TTEntry *entry = &ctx->tt_entries[(long)(key & ctx->tt_mask)];
    if (entry->flag == TT_EMPTY || entry->key != key) return 0;

    if (entry->best_valid && out_best && out_has_best) {
        *out_best = entry->best_action;
        *out_has_best = 1;
    }

    if (entry->depth < depth_left) return 0;

    if (entry->flag == TT_EXACT) {
        *out_value = entry->value;
        return 1;
    }
    if (entry->flag == TT_LOWER) {
        if (entry->value > *alpha) *alpha = entry->value;
    } else if (entry->flag == TT_UPPER) {
        if (entry->value < *beta) *beta = entry->value;
    }
    if (*alpha >= *beta) {
        *out_value = entry->value;
        return 1;
    }
    return 0;
}

static inline void tt_store(DeepSearchCtx *ctx, uint64_t key, int depth_left,
                            double value, double alpha_orig, double beta_orig,
                            const Action *best_action, int best_valid) {
    if (ctx->tt_size <= 0 || ctx->tt_entries == NULL) return;

    TTEntry *entry = &ctx->tt_entries[(long)(key & ctx->tt_mask)];
    if (entry->flag != TT_EMPTY && entry->key != key && entry->depth > depth_left) {
        return;
    }

    entry->key = key;
    entry->value = value;
    entry->depth = depth_left;
    if (value <= alpha_orig) {
        entry->flag = TT_UPPER;
    } else if (value >= beta_orig) {
        entry->flag = TT_LOWER;
    } else {
        entry->flag = TT_EXACT;
    }
    entry->best_valid = best_valid ? 1U : 0U;
    if (best_valid && best_action) {
        entry->best_action = *best_action;
    }
}

static inline void move_tt_best_to_front(Action *actions, int n_actions,
                                         int *top, int *n_top,
                                         const Action *best_action) {
    if (!best_action || !top || !n_top || *n_top <= 0) return;

    int top_pos = -1;
    for (int i = 0; i < *n_top; i++) {
        if (top[i] >= 0 && top[i] < n_actions &&
            action_eq(actions[top[i]], *best_action)) {
            top_pos = i;
            break;
        }
    }

    if (top_pos < 0) {
        return;
    }

    int best_idx = top[top_pos];
    for (int i = top_pos; i > 0; i--) {
        top[i] = top[i - 1];
    }
    top[0] = best_idx;
}

static inline int top_contains_action(Action *actions, int *top, int n_top,
                                      const Action *action) {
    for (int i = 0; i < n_top; i++) {
        if (action_eq(actions[top[i]], *action)) return 1;
    }
    return 0;
}

/* ===== Leaf evaluation (with cache) ===== */
static inline double leaf_value_uncached(DeepSearchCtx *ctx, Game *g,
                                         Color our_color) {
    /* Terminal */
    Color w = game_winning_color(g);
    if (w != COLOR_NONE) {
        return (w == our_color) ? WIN_VAL : LOSS_VAL;
    }

    double raw;
    if (ctx->leaf_mode == 6) {
        raw = base_value_fn_known_future_exact(g, our_color, true);
    } else if (ctx->leaf_mode == 16) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 8);
    } else if (ctx->leaf_mode == 14) {
        raw = base_value_fn_known_future_profile(g, our_color, true, 1);
    } else if (ctx->leaf_mode == 13) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 7);
    } else if (ctx->leaf_mode == 12) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 6);
    } else if (ctx->leaf_mode == 11) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 5);
    } else if (ctx->leaf_mode == 10) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 4);
    } else if (ctx->leaf_mode == 9) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 3);
    } else if (ctx->leaf_mode == 8) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 2);
    } else if (ctx->leaf_mode == 7) {
        raw = base_value_fn_known_future_profile(g, our_color, false, 1);
    } else if (ctx->leaf_mode == 5) {
        raw = base_value_fn_known_future(g, our_color);
    } else if (ctx->leaf_mode == 4) {
        raw = base_value_fn_enemy_full(g, our_color);
    } else {
        raw = base_value_fn(g, our_color);
    }
    double v = raw / VALUE_SCALE;
    if (v > 0.99) v = 0.99;
    if (v < -0.99) v = -0.99;
    return v;
}

static inline double leaf_value(DeepSearchCtx *ctx, Game *g, Color our_color) {
    uint64_t h = 0;
    if (ctx->cache_size > 0) {
        h = state_hash(g);
        h ^= ((uint64_t)(our_color + 2) * 0x9e3779b97f4a7c15ULL);
        long idx = (long)(h & ctx->cache_mask);
        if (ctx->cache_keys[idx] == h && h != 0) {
            ctx->stats.n_cache_hits++;
            return ctx->cache_vals[idx];
        }
        ctx->stats.n_cache_misses++;
    }

    double v = leaf_value_uncached(ctx, g, our_color);

    if (ctx->cache_size > 0 && h != 0) {
        long idx = (long)(h & ctx->cache_mask);
        ctx->cache_keys[idx] = h;
        ctx->cache_vals[idx] = v;
    }
    return v;
}

static inline double leaf_extension_value(DeepSearchCtx *ctx, Game *g,
                                          Color our_color,
                                          Action *actions, int n,
                                          int depth_idx) {
    double best_v = leaf_value(ctx, g, our_color);
    if (!ctx->leaf_extension || n <= 0) return best_v;

    for (int i = 0; i < n; i++) {
        Game *child = &ctx->game_pool[depth_idx];
        game_copy(child, g);
        int dummy;
        game_execute(child, actions[i], ctx->ab_buf, &dummy);
        double v = leaf_value(ctx, child, our_color);
        if (v > best_v) best_v = v;
        if (best_v >= WIN_VAL - 1e-6) break;
    }
    return best_v;
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

static inline int deterministic_ab2_choose_idx(DeepSearchCtx *ctx, Game *g,
                                               Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    Color cp = g->state.colors[g->state.current_player_index];
    for (int i = 0; i < n; i++) ctx->ab_buf[i] = actions[i];
    SearchResult res = alphabeta_search_deterministic(&ctx->ab_ctx, g,
                                                       ctx->ab_buf, n,
                                                       ctx->opp_ab_depth,
                                                       -1e30, 1e30,
                                                       cp, base_value_fn);
    for (int i = 0; i < n; i++) {
        if (memcmp(&res.action, &actions[i], sizeof(Action)) == 0) {
            return i;
        }
    }
    return 0;
}

static inline int deterministic_kf_ab2_choose_idx(DeepSearchCtx *ctx, Game *g,
                                                  Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    Color cp = g->state.colors[g->state.current_player_index];
    for (int i = 0; i < n; i++) ctx->ab_buf[i] = actions[i];
    SearchResult res = alphabeta_search_deterministic(&ctx->ab_ctx, g,
                                                       ctx->ab_buf, n,
                                                       ctx->opp_ab_depth,
                                                       -1e30, 1e30,
                                                       cp,
                                                       base_value_fn_known_future);
    for (int i = 0; i < n; i++) {
        if (memcmp(&res.action, &actions[i], sizeof(Action)) == 0) {
            return i;
        }
    }
    return 0;
}

static inline int deterministic_maxn_choose_idx(DeepSearchCtx *ctx, Game *g,
                                                Action *actions, int n,
                                                ValueFn eval_fn) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    for (int i = 0; i < n; i++) ctx->ab_buf[i] = actions[i];
    SearchResult res = maxn_search_deterministic(&ctx->ab_ctx, g,
                                                  ctx->ab_buf, n,
                                                  ctx->opp_ab_depth,
                                                  eval_fn);
    for (int i = 0; i < n; i++) {
        if (memcmp(&res.action, &actions[i], sizeof(Action)) == 0) {
            return i;
        }
    }
    return 0;
}

static inline int heuristic_choose_idx(DeepSearchCtx *ctx, Game *g,
                                       Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    int top[DS_MAX_K];
    int n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, actions, n, 1,
                                top, ctx->c_nf_buf, ctx->c_ef_buf,
                                ctx->c_ff_buf, ctx->c_mk_buf,
                                ctx->c_out_buf, 1);
    return n_top > 0 ? top[0] : 0;
}

static inline int heuristic_leaf_choose_idx(DeepSearchCtx *ctx, Game *g,
                                            Action *actions, int n) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    int k = n < 6 ? n : 6;
    int top[DS_MAX_K];
    int n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, actions, n, k,
                                top, ctx->c_nf_buf, ctx->c_ef_buf,
                                ctx->c_ff_buf, ctx->c_mk_buf,
                                ctx->c_out_buf, 1);
    if (n_top <= 0) return 0;

    Color opp_color = g->state.colors[g->state.current_player_index];
    double best_v = -2.0;
    int best_idx = top[0];
    for (int i = 0; i < n_top; i++) {
        Game child;
        int dummy;
        game_copy(&child, g);
        game_execute(&child, actions[top[i]], ctx->ab_buf, &dummy);
        double v = leaf_value(ctx, &child, opp_color);
        if (v > best_v) {
            best_v = v;
            best_idx = top[i];
        }
    }
    return best_idx;
}

static inline double nested_hs_next_turn_value(DeepSearchCtx *ctx, Game *start,
                                               Color actor_color, int width) {
    Game sim;
    Action actions[MAX_ACTIONS];
    int n = 0;
    game_copy(&sim, start);
    n = generate_playable_actions(&sim.state, actions, MAX_ACTIONS);

    int safety = 0;
    while (safety++ < 120) {
        Color winner = game_winning_color(&sim);
        if (winner != COLOR_NONE) {
            return (winner == actor_color) ? WIN_VAL : LOSS_VAL;
        }
        if (sim.state.num_turns >= MAX_TURNS_RUN || n <= 0) {
            return leaf_value_uncached(ctx, &sim, actor_color);
        }

        if (n == 1) {
            game_execute(&sim, actions[0], actions, &n);
            continue;
        }

        Color cp = state_current_color(&sim.state);
        if (cp == actor_color) {
            int k = width < n ? width : n;
            if (k > DS_MAX_K) k = DS_MAX_K;
            int top[DS_MAX_K];
            int profile = ctx->algo_policy > 0 ? ctx->algo_policy : 1;
            int n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model,
                                        &sim, actions, n, k, top,
                                        ctx->c_nf_buf, ctx->c_ef_buf,
                                        ctx->c_ff_buf, ctx->c_mk_buf,
                                        ctx->c_out_buf, profile);
            if (n_top <= 0) {
                return leaf_value_uncached(ctx, &sim, actor_color);
            }

            double best_v = -2.0;
            for (int i = 0; i < n_top; i++) {
                Game child;
                Action child_actions[MAX_ACTIONS];
                int child_n = 0;
                game_copy(&child, &sim);
                game_execute(&child, actions[top[i]], child_actions, &child_n);
                double v = leaf_value_uncached(ctx, &child, actor_color);
                if (v > best_v) best_v = v;
            }
            return best_v;
        }

        int chosen = deterministic_ab2_choose_idx(ctx, &sim, actions, n);
        if (chosen < 0 || chosen >= n) chosen = 0;
        game_execute(&sim, actions[chosen], actions, &n);
    }

    return leaf_value_uncached(ctx, &sim, actor_color);
}

static inline int nested_hs_choose_idx(DeepSearchCtx *ctx, Game *g,
                                       Action *actions, int n, int width) {
    if (n == 0) return -1;
    if (n == 1) return 0;
    if (width < 1) width = 1;
    if (width > DS_MAX_K) width = DS_MAX_K;

    int k = width < n ? width : n;
    int top[DS_MAX_K];
    int profile = ctx->algo_policy > 0 ? ctx->algo_policy : 1;
    int n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, actions, n, k,
                                top, ctx->c_nf_buf, ctx->c_ef_buf,
                                ctx->c_ff_buf, ctx->c_mk_buf,
                                ctx->c_out_buf, profile);
    if (n_top <= 0) {
        return deterministic_ab2_choose_idx(ctx, g, actions, n);
    }

    Color actor_color = state_current_color(&g->state);
    double best_v = -2.0;
    int best_idx = top[0];
    for (int i = 0; i < n_top; i++) {
        Game child;
        Action child_actions[MAX_ACTIONS];
        int child_n = 0;
        game_copy(&child, g);
        game_execute(&child, actions[top[i]], child_actions, &child_n);
        double v = nested_hs_next_turn_value(ctx, &child, actor_color, width);
        if (v > best_v) {
            best_v = v;
            best_idx = top[i];
        }
    }
    return best_idx;
}

static inline int opponent_choose_idx(DeepSearchCtx *ctx, Game *g,
                                      Action *actions, int n) {
    if (ctx->opponent_model == 1) {
        return heuristic_choose_idx(ctx, g, actions, n);
    }
    if (ctx->opponent_model == 2) {
        return heuristic_leaf_choose_idx(ctx, g, actions, n);
    }
    if (ctx->opponent_model == 3) {
        return deterministic_ab2_choose_idx(ctx, g, actions, n);
    }
    if (ctx->opponent_model == 4) {
        return deterministic_kf_ab2_choose_idx(ctx, g, actions, n);
    }
    if (ctx->opponent_model == 5) {
        return deterministic_maxn_choose_idx(ctx, g, actions, n, base_value_fn);
    }
    if (ctx->opponent_model == 6) {
        return deterministic_maxn_choose_idx(ctx, g, actions, n,
                                             base_value_fn_known_future);
    }
    if (ctx->opponent_model == 7) {
        return nested_hs_choose_idx(ctx, g, actions, n, 2);
    }
    if (ctx->opponent_model == 8) {
        return nested_hs_choose_idx(ctx, g, actions, n, 3);
    }
    if (ctx->opponent_model == 9) {
        return nested_hs_choose_idx(ctx, g, actions, n, 4);
    }
    return ab2_choose_idx(ctx, g, actions, n);
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
        int chosen = opponent_choose_idx(ctx, g, scratch_actions, n);
        int dummy;
        game_execute(g, scratch_actions[chosen], scratch_actions, &dummy);
    }
    return -1;
}

/* ===== Recursive deep search ===== */
static double deep_search_recurse(DeepSearchCtx *ctx, Game *g,
                                    Color our_color, int depth_left,
                                    int depth_idx, double alpha, double beta) {
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
        ctx->stats.n_leaves++;
        return leaf_extension_value(ctx, g, our_color, scratch, n, depth_idx);
    }

    double alpha_orig = alpha;
    double beta_orig = beta;
    uint64_t node_key = 0;
    Action tt_best_action;
    int tt_has_best = 0;
    if (ctx->tt_size > 0) {
        node_key = tt_key(g, our_color);
        double tt_value;
        if (tt_probe(ctx, node_key, depth_left, &alpha, &beta, &tt_value,
                     &tt_best_action, &tt_has_best)) {
            return tt_value;
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
            if (ctx->tt_size > 0) {
                tt_store(ctx, node_key, depth_left, WIN_VAL,
                         alpha_orig, beta_orig, &scratch[i], 1);
            }
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
    int n_top;
    if (ctx->algo_policy || (ctx->c_nn_model != NULL && ctx->c_enc != NULL)) {
        n_top = policy_top_k_ex(ctx->c_enc, ctx->c_nn_model, g, scratch, n, k,
                                top, ctx->c_nf_buf, ctx->c_ef_buf, ctx->c_ff_buf,
                                ctx->c_mk_buf, ctx->c_out_buf,
                                ctx->algo_policy);
    } else {
        n_top = ctx->policy_fn(ctx->userdata, g, scratch, n, k, top);
    }

    if (n_top == 0) {
        ctx->stats.n_leaves++;
        return leaf_value(ctx, g, our_color);
    }
    if (tt_has_best) {
        move_tt_best_to_front(scratch, n, top, &n_top, &tt_best_action);
    }
    if (ctx->candidate_rescue && n_top < n) {
        double rescue_v = -2.0;
        int rescue_idx = -1;
        for (int i = 0; i < n; i++) {
            if (top_contains_action(scratch, top, n_top, &scratch[i])) continue;
            Game *child = &ctx->game_pool[depth_idx];
            game_copy(child, g);
            int dummy;
            game_execute(child, scratch[i], ctx->ab_buf, &dummy);
            double v = leaf_value(ctx, child, our_color);
            if (v > rescue_v) {
                rescue_v = v;
                rescue_idx = i;
            }
        }
        if (rescue_idx >= 0) {
            if (n_top < k && n_top < DS_MAX_K) {
                top[n_top++] = rescue_idx;
            } else if (n_top > 0) {
                top[n_top - 1] = rescue_idx;
            }
        }
    }

    double best_v = -2.0;
    Action best_action;
    int best_valid = 0;
    int searched = 0;
    for (int i = 0; i < n_top; i++) {
        int action_idx = top[i];
        int reduction = 0;
        if (ctx->lmr_enabled && searched > 0 && i >= 2 && depth_left >= 3) {
            reduction = 1;
        }
        int child_depth = depth_left - 1 - reduction;
        if (child_depth < 0) child_depth = 0;

        Game *child = &ctx->game_pool[depth_idx];
        game_copy(child, g);
        int dummy;
        game_execute(child, scratch[action_idx], ctx->ab_buf, &dummy);

        double v;
        if (ctx->pvs_enabled && searched > 0 && alpha + PVS_EPS < beta) {
            double scout_beta = alpha + PVS_EPS;
            if (scout_beta > beta) scout_beta = beta;
            v = deep_search_recurse(ctx, child, our_color,
                                    child_depth, depth_idx + 1,
                                    alpha, scout_beta);
            if (v > alpha && v < beta) {
                game_copy(child, g);
                game_execute(child, scratch[action_idx], ctx->ab_buf, &dummy);
                v = deep_search_recurse(ctx, child, our_color,
                                        child_depth, depth_idx + 1,
                                        alpha, beta);
            }
        } else {
            v = deep_search_recurse(ctx, child, our_color,
                                    child_depth, depth_idx + 1,
                                    alpha, beta);
        }
        if (reduction && v > alpha) {
            game_copy(child, g);
            game_execute(child, scratch[action_idx], ctx->ab_buf, &dummy);
            v = deep_search_recurse(ctx, child, our_color,
                                    depth_left - 1, depth_idx + 1,
                                    alpha, beta);
        }

        if (v > best_v) {
            best_v = v;
            best_action = scratch[action_idx];
            best_valid = 1;
        }
        if (v > alpha) alpha = v;
        searched++;
        if (alpha >= beta) {
            ctx->stats.n_pruned += (n_top - i - 1);
            break;
        }
        if (best_v >= WIN_VAL - 1e-6) break;
    }
    if (ctx->tt_size > 0) {
        tt_store(ctx, node_key, depth_left, best_v, alpha_orig, beta_orig,
                 &best_action, best_valid);
    }
    return best_v;
}

static double deep_search_eval_root_candidate(DeepSearchCtx *ctx,
                                              const Game *game,
                                              const Action *root_action,
                                              Color our_color,
                                              int depth_left,
                                              int depth_idx,
                                              double alpha,
                                              double beta) {
    int start_depth = depth_left;
    if (ctx->id_enabled && ctx->tt_size > 0 && depth_left > 1) {
        start_depth = 1;
    }

    double value = 0.0;
    for (int d = start_depth; d <= depth_left; d++) {
        Game child;
        game_copy(&child, game);
        int dummy;
        game_execute(&child, *root_action, ctx->ab_buf, &dummy);

        double a = (d == depth_left) ? alpha : -2.0;
        double b = (d == depth_left) ? beta : 2.0;
        value = deep_search_recurse(ctx, &child, our_color, d, depth_idx, a, b);
        if (now_sec() >= ctx->deadline_clock) break;
    }
    return value;
}

/* ===== Public API ===== */
static DeepSearchCtx *_deep_search_create_common(int log2_cache_size) {
    DeepSearchCtx *ctx = calloc(1, sizeof(DeepSearchCtx));
    if (!ctx) return NULL;
    ctx->cache_bits = log2_cache_size;
    ctx->our_depth = 4;
    ctx->schedule_len = 1;
    ctx->top_k_schedule[0] = 5;
    ctx->opp_ab_depth = 2;
    ctx->time_budget_sec = 5.0;
    if (log2_cache_size > 0) {
        ctx->cache_size = 1L << log2_cache_size;
        ctx->cache_mask = ctx->cache_size - 1;
        ctx->cache_keys = calloc(ctx->cache_size, sizeof(uint64_t));
        ctx->cache_vals = calloc(ctx->cache_size, sizeof(double));
        if (!ctx->cache_keys || !ctx->cache_vals) {
            free(ctx->cache_keys); free(ctx->cache_vals);
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
                                     NNModel *nn_model,
                                     StateEncoderC *enc) {
    DeepSearchCtx *ctx = _deep_search_create_common(log2_cache_size);
    if (!ctx) return NULL;
    ctx->userdata = NULL;
    ctx->policy_fn = NULL;
    ctx->c_nn_model = nn_model;
    ctx->c_enc = enc;
    return ctx;
}

void deep_search_destroy(DeepSearchCtx *ctx) {
    if (!ctx) return;
    free(ctx->cache_keys);
    free(ctx->cache_vals);
    free(ctx->pcache_keys);
    free(ctx->pcache_data);
    free(ctx->tt_entries);
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

void deep_search_set_algo_policy(DeepSearchCtx *ctx, int enabled) {
    if (!ctx) return;
    ctx->algo_policy = enabled < 0 ? 0 : enabled;
}

void deep_search_set_leaf_mode(DeepSearchCtx *ctx, int leaf_mode) {
    if (!ctx) return;
    ctx->leaf_mode = leaf_mode;
}

void deep_search_set_tt_bits(DeepSearchCtx *ctx, int tt_bits) {
    if (!ctx) return;

    free(ctx->tt_entries);
    ctx->tt_entries = NULL;
    ctx->tt_size = 0;
    ctx->tt_mask = 0;
    ctx->tt_bits = 0;

    if (tt_bits <= 0) return;
    if (tt_bits > 24) tt_bits = 24;
    if (tt_bits >= (int)(8 * sizeof(long) - 1)) return;

    long size = 1L << tt_bits;
    TTEntry *entries = (TTEntry *)calloc((size_t)size, sizeof(TTEntry));
    if (!entries) return;

    ctx->tt_entries = entries;
    ctx->tt_size = size;
    ctx->tt_mask = size - 1;
    ctx->tt_bits = tt_bits;
}

void deep_search_set_search_enhancements(DeepSearchCtx *ctx,
                                         int pvs_enabled,
                                         int lmr_enabled) {
    if (!ctx) return;
    ctx->pvs_enabled = pvs_enabled ? 1 : 0;
    ctx->lmr_enabled = lmr_enabled ? 1 : 0;
}

void deep_search_set_iterative_deepening(DeepSearchCtx *ctx, int enabled) {
    if (!ctx) return;
    ctx->id_enabled = enabled ? 1 : 0;
}

void deep_search_set_candidate_rescue(DeepSearchCtx *ctx, int enabled) {
    if (!ctx) return;
    ctx->candidate_rescue = enabled ? 1 : 0;
}

void deep_search_set_leaf_extension(DeepSearchCtx *ctx, int enabled) {
    if (!ctx) return;
    ctx->leaf_extension = enabled ? 1 : 0;
}

DeepSearchCtx *deep_search_clone_config(const DeepSearchCtx *src) {
    if (!src) return NULL;
    DeepSearchCtx *ctx = _deep_search_create_common(src->cache_bits);
    if (!ctx) return NULL;
    ctx->userdata = src->userdata;
    ctx->policy_fn = src->policy_fn;
    ctx->c_nn_model = src->c_nn_model;
    ctx->c_enc = src->c_enc;
    ctx->our_depth = src->our_depth;
    ctx->schedule_len = src->schedule_len;
    for (int i = 0; i < src->schedule_len; i++) {
        ctx->top_k_schedule[i] = src->top_k_schedule[i];
    }
    ctx->opp_ab_depth = src->opp_ab_depth;
    ctx->time_budget_sec = src->time_budget_sec;
    ctx->leaf_mode = src->leaf_mode;
    ctx->algo_policy = src->algo_policy;
    ctx->opponent_model = src->opponent_model;
    ctx->pvs_enabled = src->pvs_enabled;
    ctx->lmr_enabled = src->lmr_enabled;
    ctx->id_enabled = src->id_enabled;
    ctx->candidate_rescue = src->candidate_rescue;
    ctx->leaf_extension = src->leaf_extension;
    deep_search_set_tt_bits(ctx, src->tt_bits);
    return ctx;
}

void deep_search_set_opponent_model(DeepSearchCtx *ctx, int model) {
    if (!ctx) return;
    if (model < 0 || model > 9) model = 0;
    ctx->opponent_model = model;
}

double deep_search_root(DeepSearchCtx *ctx, Game *game, Color our_color,
                         int *root_candidate_indices, int n_root_candidates,
                         int *out_best_idx) {
    if (out_best_idx) *out_best_idx = -1;
    if (n_root_candidates <= 0) return 0.0;

    ctx->deadline_clock = now_sec() + ctx->time_budget_sec;
    ctx->ab_ctx.depth_counter = 0;

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

    double best_v = -2.0;
    int best_pi = 0;
    double alpha = -2.0;
    double beta = 2.0;

    for (int i = 0; i < n_root_candidates; i++) {
        int ci = root_candidate_indices[i];
        if (ci < 0 || ci >= n_actions) continue;

        double v = deep_search_eval_root_candidate(ctx, game, &root_actions[ci],
                                                   our_color,
                                                   ctx->our_depth - 1, 1,
                                                   alpha, 2.0);
        if (v > best_v) {
            best_v = v;
            best_pi = i;
        }
        if (v > alpha) alpha = v;
        if (now_sec() >= ctx->deadline_clock) break;
    }

    if (out_best_idx) *out_best_idx = best_pi;
    return best_v;
}

double deep_search_root_scores(DeepSearchCtx *ctx, Game *game, Color our_color,
                               int *root_candidate_indices,
                               int n_root_candidates,
                               int *out_best_idx,
                               double *out_values,
                               int *out_valid) {
    if (out_best_idx) *out_best_idx = -1;
    for (int i = 0; i < n_root_candidates; i++) {
        if (out_values) out_values[i] = -2.0;
        if (out_valid) out_valid[i] = 0;
    }
    if (n_root_candidates <= 0) return 0.0;

    ctx->deadline_clock = now_sec() + ctx->time_budget_sec;
    ctx->ab_ctx.depth_counter = 0;

    if (game->state.board.map) {
        board_init_static_graph(game->state.board.map);
    }

    Action root_actions[MAX_ACTIONS];
    int n_actions = generate_playable_actions(&game->state, root_actions, MAX_ACTIONS);
    if (n_actions == 0) return 0.0;

    double best_v = -2.0;
    int best_pi = -1;

    for (int i = 0; i < n_root_candidates; i++) {
        int ci = root_candidate_indices[i];
        if (ci < 0 || ci >= n_actions) continue;

        double v = deep_search_eval_root_candidate(ctx, game, &root_actions[ci],
                                                   our_color,
                                                   ctx->our_depth - 1, 1,
                                                   -2.0, 2.0);
        if (out_values) out_values[i] = v;
        if (out_valid) out_valid[i] = 1;
        if (best_pi < 0 || v > best_v) {
            best_v = v;
            best_pi = i;
        }
        if (now_sec() >= ctx->deadline_clock) break;
    }

    if (best_pi < 0) {
        best_pi = 0;
        best_v = 0.0;
    }
    if (out_best_idx) *out_best_idx = best_pi;
    return best_v;
}

typedef struct {
    DeepSearchCtx *ctx;
    const Game *game;
    const Action *root_actions;
    int n_actions;
    const int *root_candidate_indices;
    int n_root_candidates;
    Color our_color;
    double deadline_clock;
    volatile int *next_index;
    double *values;
    int *valid;
} RootWorkerArgs;

static void *deep_search_root_worker(void *arg_ptr) {
    RootWorkerArgs *arg = (RootWorkerArgs *)arg_ptr;
    DeepSearchCtx *ctx = arg->ctx;
    ctx->deadline_clock = arg->deadline_clock;
    ctx->ab_ctx.depth_counter = 0;
    deep_search_reset_stats(ctx);

    for (;;) {
        int i = __sync_fetch_and_add(arg->next_index, 1);
        if (i >= arg->n_root_candidates) break;
        if (now_sec() >= arg->deadline_clock) break;

        int ci = arg->root_candidate_indices[i];
        if (ci < 0 || ci >= arg->n_actions) {
            arg->values[i] = -2.0;
            arg->valid[i] = 0;
            continue;
        }

        arg->values[i] = deep_search_eval_root_candidate(ctx, arg->game,
                                                         &arg->root_actions[ci],
                                                         arg->our_color,
                                                         ctx->our_depth - 1, 1,
                                                         -2.0, 2.0);
        arg->valid[i] = 1;
    }
    return NULL;
}

double deep_search_root_parallel(DeepSearchCtx *ctx, Game *game, Color our_color,
                                 int *root_candidate_indices,
                                 int n_root_candidates,
                                 int *out_best_idx,
                                 DeepSearchCtx **workers,
                                 int n_workers) {
    return deep_search_root_parallel_scores(ctx, game, our_color,
                                            root_candidate_indices,
                                            n_root_candidates,
                                            out_best_idx,
                                            workers, n_workers,
                                            NULL, NULL);
}

double deep_search_root_parallel_scores(DeepSearchCtx *ctx, Game *game,
                                        Color our_color,
                                        int *root_candidate_indices,
                                        int n_root_candidates,
                                        int *out_best_idx,
                                        DeepSearchCtx **workers,
                                        int n_workers,
                                        double *out_values,
                                        int *out_valid) {
    for (int i = 0; i < n_root_candidates; i++) {
        if (out_values) out_values[i] = -2.0;
        if (out_valid) out_valid[i] = 0;
    }
    if (n_workers <= 1 || n_root_candidates <= 1 || workers == NULL) {
        if (out_values || out_valid) {
            return deep_search_root_scores(ctx, game, our_color,
                                           root_candidate_indices,
                                           n_root_candidates, out_best_idx,
                                           out_values, out_valid);
        }
        return deep_search_root(ctx, game, our_color, root_candidate_indices,
                                n_root_candidates, out_best_idx);
    }
    if (out_best_idx) *out_best_idx = -1;
    if (n_root_candidates <= 0) return 0.0;
    if (n_workers > n_root_candidates) n_workers = n_root_candidates;
    if (n_workers > DS_MAX_K) n_workers = DS_MAX_K;

    double deadline = now_sec() + ctx->time_budget_sec;
    ctx->deadline_clock = deadline;

    if (game->state.board.map) {
        board_init_static_graph(game->state.board.map);
    }

    Action root_actions[MAX_ACTIONS];
    int n_actions = generate_playable_actions(&game->state, root_actions, MAX_ACTIONS);
    if (n_actions == 0) return 0.0;

    double values[DS_MAX_K];
    int valid[DS_MAX_K];
    for (int i = 0; i < DS_MAX_K; i++) {
        values[i] = -2.0;
        valid[i] = 0;
    }

    volatile int next_index = 0;
    pthread_t threads[DS_MAX_K];
    RootWorkerArgs args[DS_MAX_K];
    int launched = 0;

    for (int t = 0; t < n_workers; t++) {
        DeepSearchCtx *wctx = workers[t];
        if (wctx == NULL) continue;
        args[t].ctx = wctx;
        args[t].game = game;
        args[t].root_actions = root_actions;
        args[t].n_actions = n_actions;
        args[t].root_candidate_indices = root_candidate_indices;
        args[t].n_root_candidates = n_root_candidates;
        args[t].our_color = our_color;
        args[t].deadline_clock = deadline;
        args[t].next_index = &next_index;
        args[t].values = values;
        args[t].valid = valid;
        if (pthread_create(&threads[launched], NULL, deep_search_root_worker, &args[t]) == 0) {
            launched++;
        }
    }

    for (int t = 0; t < launched; t++) {
        pthread_join(threads[t], NULL);
    }

    memset(&ctx->stats, 0, sizeof(ctx->stats));
    for (int t = 0; t < n_workers; t++) {
        if (!workers[t]) continue;
        DeepSearchStats st;
        deep_search_get_stats(workers[t], &st);
        ctx->stats.n_calls += st.n_calls;
        ctx->stats.n_leaves += st.n_leaves;
        ctx->stats.n_pruned += st.n_pruned;
        ctx->stats.n_terminal_short += st.n_terminal_short;
        ctx->stats.n_cache_hits += st.n_cache_hits;
        ctx->stats.n_cache_misses += st.n_cache_misses;
        ctx->stats.n_pcache_hits += st.n_pcache_hits;
        ctx->stats.n_pcache_misses += st.n_pcache_misses;
    }

    double best_v = -2.0;
    int best_pi = -1;
    for (int i = 0; i < n_root_candidates; i++) {
        if (!valid[i]) continue;
        if (out_values) out_values[i] = values[i];
        if (out_valid) out_valid[i] = 1;
        if (best_pi < 0 || values[i] > best_v) {
            best_v = values[i];
            best_pi = i;
        }
    }
    if (best_pi < 0) {
        best_pi = 0;
        best_v = 0.0;
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

void deep_search_clear_caches(DeepSearchCtx *ctx) {
    if (!ctx) return;
    if (ctx->cache_keys && ctx->cache_size > 0) {
        memset(ctx->cache_keys, 0, (size_t)ctx->cache_size * sizeof(uint64_t));
    }
    if (ctx->pcache_keys && ctx->pcache_size > 0) {
        memset(ctx->pcache_keys, 0, (size_t)ctx->pcache_size * sizeof(uint64_t));
    }
    if (ctx->tt_entries && ctx->tt_size > 0) {
        memset(ctx->tt_entries, 0, (size_t)ctx->tt_size * sizeof(TTEntry));
    }
}
