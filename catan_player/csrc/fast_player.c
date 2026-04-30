/*
 * fast_player.c -- minimal standalone C runner for three agents:
 *   - m2_0ply: M2 neural policy argmax, no search
 *   - H-S: strongest validated no-ML heuristic search bot
 *   - AB2: strong depth-2 alpha-beta/expectimax baseline
 */

#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "actions.h"
#include "board.h"
#include "catan_types.h"
#include "deep_search.h"
#include "game.h"
#include "map.h"
#include "nn.h"
#include "policy_topk.h"
#include "search.h"
#include "state_encode.h"
#include "value.h"

typedef enum {
    AGENT_M2_0PLY = 0,
    AGENT_HS = 1,
    AGENT_AB2 = 2,
    AGENT_HS_PLUS = 3,
} AgentKind;

typedef struct {
    int depth;
    int k[DS_MAX_K_SCHEDULE];
    int k_len;
    int leaf_mode;
    int opp_ab_depth;
    int opponent_model;
    int cache_bits;
    int tt_bits;
    int pvs_enabled;
    int lmr_enabled;
    int id_enabled;
    int rescue_enabled;
    int leaf_extend_enabled;
    int policy_profile;
    int root_rollout;
    int root_ensemble;
    double time_budget_sec;
    int root_workers;
} HSConfig;

typedef struct {
    AgentKind kind;
    HSConfig hs_cfg;
    DeepSearchCtx *leaf_ctx;
    DeepSearchCtx *root_workers[DS_MAX_K];
    int n_root_workers;
    SearchCtx *ab_ctx;
    float nf[ENC_NUM_NODES * ENC_NODE_FEAT_DIM];
    float ef[ENC_NUM_EDGES * ENC_EDGE_FEAT_DIM];
    float ff[ENC_FLAT_FEAT_DIM];
    float mk[NN_MASK_DIM];
    float out[4 + NN_MASK_DIM];
} AgentRuntime;

typedef struct {
    const char *name;
    AgentKind kind;
    HSConfig hs_cfg;
} ArenaVariant;

static const int HS_DEPTH = 6;
static const int HS_K[6] = {6, 4, 2, 2, 2, 2};
static const int HS_K_LEN = 6;
static const int HS_LEAF_MODE = 4;
static const int HS_OPP_AB_DEPTH = 2;
static const int HS_CACHE_BITS = 20;
static const int AB2_DEPTH = 2;

static const char *opponent_model_name(int m);

static int default_plus_workers(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 2) return 1;
    if (n > DS_MAX_K) return DS_MAX_K;
    return (int)n;
}

static void hs_config_default(HSConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->depth = HS_DEPTH;
    cfg->k_len = HS_K_LEN;
    for (int i = 0; i < HS_K_LEN; i++) cfg->k[i] = HS_K[i];
    cfg->leaf_mode = HS_LEAF_MODE;
    cfg->opp_ab_depth = HS_OPP_AB_DEPTH;
    cfg->opponent_model = 0;
    cfg->cache_bits = HS_CACHE_BITS;
    cfg->tt_bits = 0;
    cfg->pvs_enabled = 0;
    cfg->lmr_enabled = 0;
    cfg->id_enabled = 0;
    cfg->rescue_enabled = 0;
    cfg->leaf_extend_enabled = 0;
    cfg->policy_profile = 1;
    cfg->root_rollout = 0;
    cfg->root_ensemble = 0;
    cfg->time_budget_sec = 5.0;
    cfg->root_workers = 1;
}

static void hs_plus_config_default(HSConfig *cfg) {
    static const int k[] = {6, 4, 2, 2, 2, 2};
    memset(cfg, 0, sizeof(*cfg));
    cfg->depth = 6;
    cfg->k_len = (int)(sizeof(k) / sizeof(k[0]));
    for (int i = 0; i < cfg->k_len; i++) cfg->k[i] = k[i];
    cfg->leaf_mode = 7;
    cfg->opp_ab_depth = 2;
    cfg->opponent_model = 3;
    cfg->cache_bits = 20;
    cfg->tt_bits = 0;
    cfg->pvs_enabled = 0;
    cfg->lmr_enabled = 0;
    cfg->id_enabled = 0;
    cfg->rescue_enabled = 0;
    cfg->leaf_extend_enabled = 0;
    cfg->policy_profile = 2;
    cfg->root_rollout = 0;
    cfg->root_ensemble = 0;
    cfg->time_budget_sec = 5.0;
    cfg->root_workers = default_plus_workers();
}

static bool parse_k_schedule_arg(const char *s, HSConfig *cfg) {
    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s", s);
    int vals[DS_MAX_K_SCHEDULE];
    int n = 0;
    char *tok = strtok(tmp, ",");
    while (tok != NULL && n < DS_MAX_K_SCHEDULE) {
        int v = atoi(tok);
        if (v < 1) return false;
        if (v > DS_MAX_K) v = DS_MAX_K;
        vals[n++] = v;
        tok = strtok(NULL, ",");
    }
    if (tok != NULL || n == 0) return false;
    cfg->k_len = n;
    for (int i = 0; i < n; i++) cfg->k[i] = vals[i];
    return true;
}

static void print_hs_config(const char *label, const HSConfig *cfg) {
    printf("  %s: depth=%d k=", label, cfg->depth);
    for (int i = 0; i < cfg->k_len; i++) {
        if (i) printf(",");
        printf("%d", cfg->k[i]);
    }
    printf(" leaf=%d opp_model=%s opp_ab=%d cache=2^%d tt=%s",
           cfg->leaf_mode, opponent_model_name(cfg->opponent_model),
           cfg->opp_ab_depth, cfg->cache_bits,
           cfg->tt_bits > 0 ? "2^" : "off");
    if (cfg->tt_bits > 0) printf("%d", cfg->tt_bits);
    printf(" pvs=%d lmr=%d id=%d rescue=%d leaf_ext=%d policy=%d rollout=%d ensemble=%d time=%.1fs workers=%d\n",
           cfg->pvs_enabled, cfg->lmr_enabled, cfg->id_enabled,
           cfg->rescue_enabled, cfg->leaf_extend_enabled,
           cfg->policy_profile, cfg->root_rollout, cfg->root_ensemble,
           cfg->time_budget_sec, cfg->root_workers);
}

static int parse_opponent_model(const char *s) {
    if (strcmp(s, "ab2") == 0 || strcmp(s, "AB2") == 0) return 0;
    if (strcmp(s, "hs") == 0 || strcmp(s, "h-s") == 0 ||
        strcmp(s, "algo") == 0 || strcmp(s, "policy") == 0) return 1;
    if (strcmp(s, "hs-leaf") == 0 || strcmp(s, "h-s-leaf") == 0 ||
        strcmp(s, "algo-leaf") == 0 || strcmp(s, "leaf") == 0) return 2;
    if (strcmp(s, "det-ab2") == 0 || strcmp(s, "deterministic-ab2") == 0 ||
        strcmp(s, "known-ab2") == 0 || strcmp(s, "known-future-ab2") == 0) return 3;
    if (strcmp(s, "det-kf-ab2") == 0 || strcmp(s, "det-leaf5-ab2") == 0 ||
        strcmp(s, "known-future-eval-ab2") == 0) return 4;
    if (strcmp(s, "det-maxn") == 0 || strcmp(s, "maxn") == 0 ||
        strcmp(s, "selfish") == 0) return 5;
    if (strcmp(s, "det-kf-maxn") == 0 || strcmp(s, "maxn-kf") == 0 ||
        strcmp(s, "selfish-kf") == 0) return 6;
    if (strcmp(s, "nested-hs2") == 0 || strcmp(s, "det-hs2") == 0 ||
        strcmp(s, "hs-nested2") == 0) return 7;
    if (strcmp(s, "nested-hs3") == 0 || strcmp(s, "det-hs3") == 0 ||
        strcmp(s, "nested-hs") == 0 || strcmp(s, "hs-nested") == 0) return 8;
    if (strcmp(s, "nested-hs4") == 0 || strcmp(s, "det-hs4") == 0 ||
        strcmp(s, "hs-nested4") == 0) return 9;
    return -1;
}

static const char *opponent_model_name(int m) {
    if (m == 1) return "hs";
    if (m == 2) return "hs-leaf";
    if (m == 3) return "det-ab2";
    if (m == 4) return "det-kf-ab2";
    if (m == 5) return "det-maxn";
    if (m == 6) return "det-kf-maxn";
    if (m == 7) return "nested-hs2";
    if (m == 8) return "nested-hs3";
    if (m == 9) return "nested-hs4";
    return "ab2";
}

static const char *agent_name(AgentKind k) {
    switch (k) {
    case AGENT_M2_0PLY: return "m2_0ply";
    case AGENT_HS: return "H-S";
    case AGENT_AB2: return "AB2";
    case AGENT_HS_PLUS: return "H-S+";
    }
    return "unknown";
}

static void variant_base_plus(HSConfig *cfg) {
    hs_plus_config_default(cfg);
}

static void variant_set_opp_model(HSConfig *cfg, const char *name) {
    int model = parse_opponent_model(name);
    if (model >= 0) cfg->opponent_model = model;
}

static bool configure_arena_variant(const char *name, ArenaVariant *out) {
    memset(out, 0, sizeof(*out));
    out->name = name;
    out->kind = AGENT_HS_PLUS;
    variant_base_plus(&out->hs_cfg);

    if (strcmp(name, "ab2") == 0 || strcmp(name, "AB2") == 0) {
        out->kind = AGENT_AB2;
        return true;
    }
    if (strcmp(name, "h-s") == 0 || strcmp(name, "hs") == 0 ||
        strcmp(name, "H-S") == 0) {
        out->kind = AGENT_HS;
        hs_config_default(&out->hs_cfg);
        return true;
    }
    if (strcmp(name, "default") == 0 || strcmp(name, "leaf7-policy2") == 0) {
        return true;
    }
    if (strcmp(name, "old-default") == 0) {
        out->hs_cfg.leaf_mode = 5;
        out->hs_cfg.policy_profile = 1;
        return true;
    }
    if (strcmp(name, "leaf5-policy2") == 0) {
        out->hs_cfg.leaf_mode = 5;
        out->hs_cfg.policy_profile = 2;
        return true;
    }
    if (strcmp(name, "leaf7-policy1") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 1;
        return true;
    }
    if (strcmp(name, "leaf7-policy3") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 3;
        return true;
    }
    if (strcmp(name, "leaf7-policy4") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 4;
        return true;
    }
    if (strcmp(name, "leaf7-policy5") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 5;
        return true;
    }
    if (strcmp(name, "leaf7-policy6") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 6;
        return true;
    }
    if (strcmp(name, "leaf7-policy7") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 7;
        return true;
    }
    if (strcmp(name, "leaf7-policy8") == 0) {
        out->hs_cfg.leaf_mode = 7;
        out->hs_cfg.policy_profile = 8;
        return true;
    }
    if (strcmp(name, "leaf7") == 0) {
        out->hs_cfg.leaf_mode = 7;
        return true;
    }
    if (strcmp(name, "leaf8") == 0 || strcmp(name, "leaf8-policy2") == 0) {
        out->hs_cfg.leaf_mode = 8;
        return true;
    }
    if (strcmp(name, "leaf12") == 0) {
        out->hs_cfg.leaf_mode = 12;
        return true;
    }
    if (strcmp(name, "leaf13") == 0) {
        out->hs_cfg.leaf_mode = 13;
        return true;
    }
    if (strcmp(name, "leaf14") == 0) {
        out->hs_cfg.leaf_mode = 14;
        return true;
    }
    if (strcmp(name, "leaf16") == 0) {
        out->hs_cfg.leaf_mode = 16;
        return true;
    }
    if (strcmp(name, "pvs") == 0) {
        out->hs_cfg.pvs_enabled = 1;
        return true;
    }
    if (strcmp(name, "tt") == 0) {
        out->hs_cfg.tt_bits = 18;
        return true;
    }
    if (strcmp(name, "ttid") == 0) {
        out->hs_cfg.tt_bits = 18;
        out->hs_cfg.id_enabled = 1;
        return true;
    }
    if (strcmp(name, "tt-pvs-noid") == 0) {
        out->hs_cfg.tt_bits = 18;
        out->hs_cfg.pvs_enabled = 1;
        return true;
    }
    if (strcmp(name, "tt-pvs") == 0) {
        out->hs_cfg.tt_bits = 18;
        out->hs_cfg.pvs_enabled = 1;
        out->hs_cfg.id_enabled = 1;
        return true;
    }
    if (strcmp(name, "tt-pvs-lmr") == 0) {
        out->hs_cfg.tt_bits = 18;
        out->hs_cfg.pvs_enabled = 1;
        out->hs_cfg.lmr_enabled = 1;
        out->hs_cfg.id_enabled = 1;
        return true;
    }
    if (strcmp(name, "leaf8-ttid") == 0) {
        out->hs_cfg.leaf_mode = 8;
        out->hs_cfg.tt_bits = 18;
        out->hs_cfg.id_enabled = 1;
        return true;
    }
    if (strcmp(name, "k7") == 0 || strcmp(name, "k7-pvs") == 0 ||
        strcmp(name, "leaf8-k7") == 0 || strcmp(name, "leaf8-k7-pvs") == 0) {
        static const int k[] = {7, 4, 2, 2, 2, 2};
        out->hs_cfg.k_len = 6;
        for (int i = 0; i < 6; i++) out->hs_cfg.k[i] = k[i];
        if (strcmp(name, "leaf8-k7") == 0 || strcmp(name, "leaf8-k7-pvs") == 0)
            out->hs_cfg.leaf_mode = 8;
        if (strcmp(name, "k7-pvs") == 0 || strcmp(name, "leaf8-k7-pvs") == 0)
            out->hs_cfg.pvs_enabled = 1;
        return true;
    }
    if (strcmp(name, "k6-5") == 0 || strcmp(name, "k6-5-pvs") == 0) {
        static const int k[] = {6, 5, 2, 2, 2, 2};
        out->hs_cfg.k_len = 6;
        for (int i = 0; i < 6; i++) out->hs_cfg.k[i] = k[i];
        if (strcmp(name, "k6-5-pvs") == 0) out->hs_cfg.pvs_enabled = 1;
        return true;
    }
    if (strcmp(name, "depth6-k6-4-3") == 0) {
        static const int k[] = {6, 4, 3, 2, 2, 2};
        out->hs_cfg.k_len = 6;
        for (int i = 0; i < 6; i++) out->hs_cfg.k[i] = k[i];
        return true;
    }
    if (strcmp(name, "depth7-tight") == 0) {
        static const int k[] = {6, 4, 2, 2, 1, 1, 1};
        out->hs_cfg.depth = 7;
        out->hs_cfg.k_len = 7;
        for (int i = 0; i < 7; i++) out->hs_cfg.k[i] = k[i];
        return true;
    }
    if (strcmp(name, "det-maxn") == 0) {
        variant_set_opp_model(&out->hs_cfg, "det-maxn");
        return true;
    }
    if (strcmp(name, "det-kf-maxn") == 0) {
        variant_set_opp_model(&out->hs_cfg, "det-kf-maxn");
        return true;
    }
    if (strcmp(name, "det-kf-ab2") == 0) {
        variant_set_opp_model(&out->hs_cfg, "det-kf-ab2");
        return true;
    }
    if (strcmp(name, "nested-hs2") == 0 || strcmp(name, "nested-hs3") == 0 ||
        strcmp(name, "nested-hs4") == 0 || strcmp(name, "nested-hs") == 0) {
        variant_set_opp_model(&out->hs_cfg,
                              strcmp(name, "nested-hs") == 0 ? "nested-hs3" : name);
        return true;
    }
    if (strcmp(name, "opp-hs") == 0) {
        variant_set_opp_model(&out->hs_cfg, "hs");
        return true;
    }
    if (strcmp(name, "opp-hs-leaf") == 0) {
        variant_set_opp_model(&out->hs_cfg, "hs-leaf");
        return true;
    }
    if (strcmp(name, "rescue") == 0) {
        out->hs_cfg.rescue_enabled = 1;
        return true;
    }
    if (strcmp(name, "leaf-extend") == 0) {
        out->hs_cfg.leaf_extend_enabled = 1;
        return true;
    }
    if (strcmp(name, "root-rollout") == 0) {
        out->hs_cfg.root_rollout = 1;
        return true;
    }
    if (strcmp(name, "root-ensemble1") == 0) {
        out->hs_cfg.root_ensemble = 1;
        return true;
    }
    if (strcmp(name, "root-ensemble2") == 0) {
        out->hs_cfg.root_ensemble = 2;
        return true;
    }
    if (strcmp(name, "root-scoreblend3") == 0) {
        out->hs_cfg.root_ensemble = 3;
        return true;
    }
    if (strcmp(name, "root-scoreblend4") == 0) {
        out->hs_cfg.root_ensemble = 4;
        return true;
    }
    return false;
}

static bool parse_agent(const char *s, AgentKind *out) {
    if (strcmp(s, "m2_0ply") == 0 || strcmp(s, "m2") == 0 || strcmp(s, "policy") == 0) {
        *out = AGENT_M2_0PLY;
        return true;
    }
    if (strcmp(s, "H-S") == 0 || strcmp(s, "h-s") == 0 ||
        strcmp(s, "hs") == 0 || strcmp(s, "heuristic") == 0 ||
        strcmp(s, "leaf0_search") == 0 || strcmp(s, "leaf0") == 0 ||
        strcmp(s, "search") == 0) {
        *out = AGENT_HS;
        return true;
    }
    if (strcmp(s, "H-S+") == 0 || strcmp(s, "h-s+") == 0 ||
        strcmp(s, "hs+") == 0 || strcmp(s, "hsp") == 0 ||
        strcmp(s, "heuristic+") == 0 || strcmp(s, "tuned") == 0 ||
        strcmp(s, "search+") == 0) {
        *out = AGENT_HS_PLUS;
        return true;
    }
    if (strcmp(s, "AB2") == 0 || strcmp(s, "ab2") == 0 ||
        strcmp(s, "strong_ab2") == 0 || strcmp(s, "full_ab2") == 0) {
        *out = AGENT_AB2;
        return true;
    }
    return false;
}

static const char *action_name(ActionType type) {
    switch (type) {
    case AT_ROLL: return "ROLL";
    case AT_MOVE_ROBBER: return "ROBBER";
    case AT_DISCARD_RESOURCE: return "DISCARD";
    case AT_BUILD_ROAD: return "ROAD";
    case AT_BUILD_SETTLEMENT: return "SETTLEMENT";
    case AT_BUILD_CITY: return "CITY";
    case AT_BUY_DEVELOPMENT_CARD: return "BUY_DEV";
    case AT_PLAY_KNIGHT_CARD: return "KNIGHT";
    case AT_PLAY_YEAR_OF_PLENTY: return "YEAR_OF_PLENTY";
    case AT_PLAY_MONOPOLY: return "MONOPOLY";
    case AT_PLAY_ROAD_BUILDING: return "ROAD_BUILDING";
    case AT_MARITIME_TRADE: return "MARITIME";
    case AT_ACCEPT_TRADE: return "ACCEPT_TRADE";
    case AT_REJECT_TRADE: return "REJECT_TRADE";
    case AT_CANCEL_TRADE: return "CANCEL_TRADE";
    case AT_CONFIRM_TRADE: return "CONFIRM_TRADE";
    case AT_END_TURN: return "END_TURN";
    default: return "UNKNOWN";
    }
}

static int fix_robber_steal(int chosen, const Action *actions, int n_actions) {
    Action act = actions[chosen];
    if (act.type != AT_MOVE_ROBBER || act.value[3] >= 0) return chosen;
    int tx = act.value[0], ty = act.value[1], tz = act.value[2];
    for (int i = 0; i < n_actions; i++) {
        if (actions[i].type == AT_MOVE_ROBBER && actions[i].value[3] >= 0 &&
            actions[i].value[0] == tx && actions[i].value[1] == ty && actions[i].value[2] == tz) {
            return i;
        }
    }
    for (int i = 0; i < n_actions; i++) {
        if (actions[i].type == AT_MOVE_ROBBER && actions[i].value[3] >= 0) return i;
    }
    return chosen;
}

static int immediate_win_idx(Game *g, const Action *actions, int n_actions) {
    Color us = state_current_color(&g->state);
    for (int i = 0; i < n_actions; i++) {
        Game child;
        Action tmp[MAX_ACTIONS];
        int ntmp = 0;
        game_copy(&child, g);
        game_execute(&child, actions[i], tmp, &ntmp);
        if (game_winning_color(&child) == us) return i;
    }
    return -1;
}

static double hs_root_leaf_value(const HSConfig *cfg, Game *g, Color us) {
    Color winner = game_winning_color(g);
    if (winner == us) return 1.0e18;
    if (winner != COLOR_NONE) return -1.0e18;
    if (cfg->leaf_mode == 16) return base_value_fn_known_future_profile(g, us, false, 8);
    if (cfg->leaf_mode == 14) return base_value_fn_known_future_profile(g, us, true, 1);
    if (cfg->leaf_mode == 13) return base_value_fn_known_future_profile(g, us, false, 7);
    if (cfg->leaf_mode == 12) return base_value_fn_known_future_profile(g, us, false, 6);
    if (cfg->leaf_mode == 11) return base_value_fn_known_future_profile(g, us, false, 5);
    if (cfg->leaf_mode == 10) return base_value_fn_known_future_profile(g, us, false, 4);
    if (cfg->leaf_mode == 9) return base_value_fn_known_future_profile(g, us, false, 3);
    if (cfg->leaf_mode == 8) return base_value_fn_known_future_profile(g, us, false, 2);
    if (cfg->leaf_mode == 7) return base_value_fn_known_future_profile(g, us, false, 1);
    if (cfg->leaf_mode == 6) return base_value_fn_known_future_exact(g, us, true);
    if (cfg->leaf_mode == 5) return base_value_fn_known_future(g, us);
    if (cfg->leaf_mode == 4) return base_value_fn_enemy_full(g, us);
    return base_value_fn(g, us);
}

static bool top_contains_idx(const int *top, int n_top, int idx) {
    for (int i = 0; i < n_top; i++) {
        if (top[i] == idx) return true;
    }
    return false;
}

static void rescue_root_candidate(AgentRuntime *rt, Game *g,
                                  const Action *actions, int n_actions,
                                  int *top, int *n_top, int k_root) {
    if (!rt->hs_cfg.rescue_enabled || *n_top >= n_actions) return;

    Color us = state_current_color(&g->state);
    double best_v = -1.0e300;
    int best_idx = -1;
    for (int i = 0; i < n_actions; i++) {
        if (top_contains_idx(top, *n_top, i)) continue;
        Game child;
        Action tmp[MAX_ACTIONS];
        int ntmp = 0;
        game_copy(&child, g);
        game_execute(&child, actions[i], tmp, &ntmp);
        double v = hs_root_leaf_value(&rt->hs_cfg, &child, us);
        if (v > best_v) {
            best_v = v;
            best_idx = i;
        }
    }
    if (best_idx < 0) return;
    if (*n_top < k_root && *n_top < DS_MAX_K) {
        top[(*n_top)++] = best_idx;
    } else if (*n_top > 0) {
        top[*n_top - 1] = best_idx;
    }
}

static int greedy_rollout_choose(AgentRuntime *rt, Game *g,
                                 Action *actions, int n_actions) {
    int win = immediate_win_idx(g, actions, n_actions);
    if (win >= 0) return win;

    int k = n_actions < 6 ? n_actions : 6;
    int top[DS_MAX_K];
    int n_top = policy_top_k_ex(NULL, NULL, g, actions, n_actions, k, top,
                                rt->nf, rt->ef, rt->ff, rt->mk, rt->out,
                                rt->hs_cfg.policy_profile);
    if (n_top <= 0) return 0;

    Color cp = state_current_color(&g->state);
    double best_v = -1.0e300;
    int best = top[0];
    for (int i = 0; i < n_top; i++) {
        Game child;
        Action tmp[MAX_ACTIONS];
        int ntmp = 0;
        game_copy(&child, g);
        game_execute(&child, actions[top[i]], tmp, &ntmp);
        double v = hs_root_leaf_value(&rt->hs_cfg, &child, cp);
        if (v > best_v) {
            best_v = v;
            best = top[i];
        }
    }
    return best;
}

static double rollout_root_value(AgentRuntime *rt, Game *root,
                                 const Action *root_action, Color us) {
    Game sim;
    Action actions[MAX_ACTIONS];
    int n_actions = 0;
    game_copy(&sim, root);
    game_execute(&sim, *root_action, actions, &n_actions);

    int decisions = 0;
    while (game_winning_color(&sim) == COLOR_NONE &&
           sim.state.num_turns < 500 &&
           decisions < 1200) {
        n_actions = generate_playable_actions(&sim.state, actions, MAX_ACTIONS);
        if (n_actions <= 0) break;

        int chosen = 0;
        if (n_actions > 1) {
            chosen = greedy_rollout_choose(rt, &sim, actions, n_actions);
            decisions++;
        }
        game_execute(&sim, actions[chosen], actions, &n_actions);
    }

    Color winner = game_winning_color(&sim);
    int us_idx = sim.state.color_to_index[(int)us];
    int us_vp = us_idx >= 0 ? sim.state.player_state[us_idx][PS_ACTUAL_VICTORY_POINTS] : 0;
    int leader_vp = 0;
    for (int p = 0; p < sim.state.num_players; p++) {
        if (p == us_idx) continue;
        int vp = sim.state.player_state[p][PS_ACTUAL_VICTORY_POINTS];
        if (vp > leader_vp) leader_vp = vp;
    }
    if (winner == us) return 1.0e9 - 1000.0 * (double)sim.state.num_turns + us_vp;
    if (winner != COLOR_NONE) return -1.0e9 + 1000.0 * (double)us_vp - leader_vp;
    return 1000.0 * (double)us_vp - (double)leader_vp;
}

static int choose_hs_rollout(AgentRuntime *rt, Game *g,
                             Action *actions, int n_actions,
                             int *top, int n_top) {
    Color us = state_current_color(&g->state);
    double best_v = -1.0e300;
    int best_pos = 0;
    for (int i = 0; i < n_top; i++) {
        double v = rollout_root_value(rt, g, &actions[top[i]], us);
        if (v > best_v) {
            best_v = v;
            best_pos = i;
        }
    }
    return fix_robber_steal(top[best_pos], actions, n_actions);
}

static void runtime_set_leaf_mode(AgentRuntime *rt, int leaf_mode) {
    if (rt->leaf_ctx) {
        deep_search_set_leaf_mode(rt->leaf_ctx, leaf_mode);
        deep_search_clear_caches(rt->leaf_ctx);
    }
    for (int i = 0; i < rt->n_root_workers; i++) {
        if (!rt->root_workers[i]) continue;
        deep_search_set_leaf_mode(rt->root_workers[i], leaf_mode);
        deep_search_clear_caches(rt->root_workers[i]);
    }
}

static int search_best_pos(AgentRuntime *rt, Game *g, Color us,
                           int *top, int n_top) {
    int best_pos = 0;
    if (rt->n_root_workers > 1) {
        deep_search_root_parallel(rt->leaf_ctx, g, us, top, n_top, &best_pos,
                                  rt->root_workers, rt->n_root_workers);
    } else {
        deep_search_root(rt->leaf_ctx, g, us, top, n_top, &best_pos);
    }
    if (best_pos < 0 || best_pos >= n_top) best_pos = 0;
    return best_pos;
}

static int search_scores(AgentRuntime *rt, Game *g, Color us,
                         int *top, int n_top,
                         double *values, int *valid) {
    int best_pos = 0;
    if (rt->n_root_workers > 1) {
        deep_search_root_parallel_scores(rt->leaf_ctx, g, us, top, n_top,
                                         &best_pos,
                                         rt->root_workers, rt->n_root_workers,
                                         values, valid);
    } else {
        deep_search_root_scores(rt->leaf_ctx, g, us, top, n_top,
                                &best_pos, values, valid);
    }
    if (best_pos < 0 || best_pos >= n_top) best_pos = 0;
    return best_pos;
}

static int choose_hs_ensemble(AgentRuntime *rt, Game *g,
                              Action *actions, int n_actions,
                              int *top, int n_top) {
    Color us = state_current_color(&g->state);
    int original_leaf = rt->hs_cfg.leaf_mode;
    int profile_pool[5] = {original_leaf, 8, 7, 12, 5};
    int weight_pool[5] = {2, 1, 1, 1, 1};
    int target_profiles = (rt->hs_cfg.root_ensemble == 2 ||
                           rt->hs_cfg.root_ensemble >= 4) ? 4 : 3;
    int profiles[4];
    int weights[4];
    int n_profiles = 0;
    for (int i = 0; i < 5 && n_profiles < target_profiles; i++) {
        bool seen = false;
        for (int j = 0; j < n_profiles; j++) {
            if (profiles[j] == profile_pool[i]) {
                seen = true;
                break;
            }
        }
        if (seen) continue;
        profiles[n_profiles] = profile_pool[i];
        weights[n_profiles] = weight_pool[i];
        n_profiles++;
    }

    int votes[DS_MAX_K] = {0};
    int default_pos = 0;
    double score_sum[DS_MAX_K] = {0.0};
    int score_valid[DS_MAX_K] = {0};
    for (int i = 0; i < n_profiles; i++) {
        runtime_set_leaf_mode(rt, profiles[i]);
        int pos;
        if (rt->hs_cfg.root_ensemble >= 3) {
            double values[DS_MAX_K];
            int valid[DS_MAX_K];
            pos = search_scores(rt, g, us, top, n_top, values, valid);
            for (int j = 0; j < n_top; j++) {
                if (!valid[j]) continue;
                score_sum[j] += (double)weights[i] * values[j];
                score_valid[j] = 1;
            }
        } else {
            pos = search_best_pos(rt, g, us, top, n_top);
        }
        if (i == 0) default_pos = pos;
        votes[pos] += weights[i];
    }
    runtime_set_leaf_mode(rt, original_leaf);

    int best_pos = default_pos;
    if (rt->hs_cfg.root_ensemble >= 3) {
        double best_score = score_valid[best_pos] ? score_sum[best_pos] : -1.0e300;
        int best_vote = votes[best_pos];
        for (int i = 0; i < n_top; i++) {
            if (!score_valid[i]) continue;
            if (score_sum[i] > best_score ||
                (score_sum[i] == best_score && votes[i] > best_vote)) {
                best_score = score_sum[i];
                best_vote = votes[i];
                best_pos = i;
            }
        }
        return fix_robber_steal(top[best_pos], actions, n_actions);
    }

    int best_vote = votes[default_pos];
    for (int i = 0; i < n_top; i++) {
        if (votes[i] > best_vote) {
            best_vote = votes[i];
            best_pos = i;
        }
    }
    return fix_robber_steal(top[best_pos], actions, n_actions);
}

static int choose_m2_0ply(AgentRuntime *rt, NNModel *model, const StateEncoderC *enc,
                          Game *g, Action *actions, int n_actions) {
    int win = immediate_win_idx(g, actions, n_actions);
    if (win >= 0) return win;

    encode_state_full(enc, g, rt->nf, rt->ef, rt->ff);
    memset(rt->mk, 0, sizeof(rt->mk));

    int pidx[256];
    for (int i = 0; i < n_actions && i < 256; i++) {
        pidx[i] = policy_action_encode(model, &actions[i]);
        if (pidx[i] >= 0 && pidx[i] < NN_MASK_DIM) rt->mk[pidx[i]] = 1.0f;
    }

    NNOutput nn_out;
    nn_forward(model,
               (const float (*)[NN_NODE_FEAT])rt->nf,
               (const float (*)[NN_EDGE_FEAT])rt->ef,
               rt->ff, rt->mk, &nn_out);

    int best = 0;
    float best_logit = -1e30f;
    for (int i = 0; i < n_actions && i < 256; i++) {
        float logit = (pidx[i] >= 0 && pidx[i] < 337) ? nn_out.policy[pidx[i]] : -1e30f;
        if (logit > best_logit) {
            best_logit = logit;
            best = i;
        }
    }
    return fix_robber_steal(best, actions, n_actions);
}

static int choose_hs(AgentRuntime *rt, Game *g, Action *actions, int n_actions) {
    int win = immediate_win_idx(g, actions, n_actions);
    if (win >= 0) return win;

    int k_root = rt->hs_cfg.k[0];
    if (k_root > n_actions) k_root = n_actions;

    int top[DS_MAX_K];
    int n_top = policy_top_k_ex(NULL, NULL, g, actions, n_actions, k_root, top,
                                rt->nf, rt->ef, rt->ff, rt->mk, rt->out,
                                rt->hs_cfg.policy_profile);
    if (n_top <= 0) return 0;
    rescue_root_candidate(rt, g, actions, n_actions, top, &n_top, k_root);

    if (rt->hs_cfg.root_rollout == 1) {
        return choose_hs_rollout(rt, g, actions, n_actions, top, n_top);
    }
    if (rt->hs_cfg.root_ensemble > 0) {
        return choose_hs_ensemble(rt, g, actions, n_actions, top, n_top);
    }

    Color us = state_current_color(&g->state);
    int best_pos = search_best_pos(rt, g, us, top, n_top);
    return fix_robber_steal(top[best_pos], actions, n_actions);
}

static int choose_ab2(AgentRuntime *rt, Game *g, Action *actions, int n_actions) {
    int win = immediate_win_idx(g, actions, n_actions);
    if (win >= 0) return win;

    Action acts_copy[MAX_ACTIONS];
    memcpy(acts_copy, actions, (size_t)n_actions * sizeof(Action));
    memset(rt->ab_ctx, 0, sizeof(*rt->ab_ctx));
    Color us = state_current_color(&g->state);
    SearchResult r = alphabeta_search(rt->ab_ctx, g, acts_copy, n_actions,
                                      AB2_DEPTH, -1e30, 1e30,
                                      us, base_value_fn);
    for (int i = 0; i < n_actions; i++) {
        if (memcmp(&actions[i], &r.action, sizeof(Action)) == 0) {
            return fix_robber_steal(i, actions, n_actions);
        }
    }
    return 0;
}

static bool needs_model(AgentKind a, AgentKind b, bool uses_opponent) {
    return a == AGENT_M2_0PLY || (uses_opponent && b == AGENT_M2_0PLY);
}

static int load_model_if_needed(NNModel **model_out, const char *weights_path, bool needed) {
    *model_out = NULL;
    if (!needed) return 0;
    NNModel *model = (NNModel *)calloc(1, sizeof(NNModel));
    if (!model) {
        fprintf(stderr, "OOM allocating NNModel\n");
        return -1;
    }
    if (nn_load(model, weights_path) != 0) {
        fprintf(stderr, "Failed to load model weights: %s\n", weights_path);
        free(model);
        return -1;
    }
    *model_out = model;
    return 0;
}

static void init_runtime_with_config(AgentRuntime *rt, AgentKind kind,
                                     const HSConfig *cfg) {
    memset(rt, 0, sizeof(*rt));
    rt->kind = kind;
    if (kind == AGENT_HS || kind == AGENT_HS_PLUS) {
        rt->hs_cfg = *cfg;
        if (rt->hs_cfg.depth < 1) rt->hs_cfg.depth = 1;
        if (rt->hs_cfg.k_len < 1) rt->hs_cfg.k_len = 1;
        if (rt->hs_cfg.root_workers < 1) rt->hs_cfg.root_workers = 1;
        if (rt->hs_cfg.root_workers > DS_MAX_K) rt->hs_cfg.root_workers = DS_MAX_K;
        if (rt->hs_cfg.policy_profile < 1) rt->hs_cfg.policy_profile = 1;

        rt->leaf_ctx = deep_search_create_c(rt->hs_cfg.cache_bits, NULL, NULL);
        if (!rt->leaf_ctx) {
            fprintf(stderr, "deep_search_create_c failed\n");
            exit(1);
        }
        deep_search_configure(rt->leaf_ctx, rt->hs_cfg.depth,
                              rt->hs_cfg.k, rt->hs_cfg.k_len,
                              rt->hs_cfg.opp_ab_depth,
                              rt->hs_cfg.time_budget_sec);
        deep_search_set_leaf_mode(rt->leaf_ctx, rt->hs_cfg.leaf_mode);
        deep_search_set_algo_policy(rt->leaf_ctx, rt->hs_cfg.policy_profile);
        deep_search_set_opponent_model(rt->leaf_ctx, rt->hs_cfg.opponent_model);
        deep_search_set_tt_bits(rt->leaf_ctx, rt->hs_cfg.tt_bits);
        deep_search_set_search_enhancements(rt->leaf_ctx,
                                            rt->hs_cfg.pvs_enabled,
                                            rt->hs_cfg.lmr_enabled);
        deep_search_set_iterative_deepening(rt->leaf_ctx,
                                            rt->hs_cfg.id_enabled);
        deep_search_set_candidate_rescue(rt->leaf_ctx,
                                         rt->hs_cfg.rescue_enabled);
        deep_search_set_leaf_extension(rt->leaf_ctx,
                                       rt->hs_cfg.leaf_extend_enabled);

        if (rt->hs_cfg.root_workers > 1) {
            rt->n_root_workers = rt->hs_cfg.root_workers;
            for (int i = 0; i < rt->n_root_workers; i++) {
                rt->root_workers[i] = deep_search_clone_config(rt->leaf_ctx);
                if (!rt->root_workers[i]) {
                    fprintf(stderr, "deep_search_clone_config failed\n");
                    exit(1);
                }
            }
        }
    } else if (kind == AGENT_AB2) {
        rt->ab_ctx = (SearchCtx *)calloc(1, sizeof(SearchCtx));
        if (!rt->ab_ctx) {
            fprintf(stderr, "AB2 SearchCtx allocation failed\n");
            exit(1);
        }
    }
}

static void init_runtime(AgentRuntime *rt, AgentKind kind,
                         const HSConfig *hs_cfg,
                         const HSConfig *plus_cfg) {
    if (kind == AGENT_HS || kind == AGENT_HS_PLUS) {
        const HSConfig *cfg = (kind == AGENT_HS_PLUS) ? plus_cfg : hs_cfg;
        init_runtime_with_config(rt, kind, cfg);
        return;
    }
    init_runtime_with_config(rt, kind, hs_cfg);
}

static void destroy_runtime(AgentRuntime *rt) {
    for (int i = 0; i < rt->n_root_workers; i++) {
        if (rt->root_workers[i]) deep_search_destroy(rt->root_workers[i]);
        rt->root_workers[i] = NULL;
    }
    rt->n_root_workers = 0;
    if (rt->leaf_ctx) deep_search_destroy(rt->leaf_ctx);
    rt->leaf_ctx = NULL;
    free(rt->ab_ctx);
    rt->ab_ctx = NULL;
}

static void default_weights_path(char *out, size_t n, const char *argv0) {
    char tmp[1024];
    strncpy(tmp, argv0, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    char *dir = dirname(tmp);
    snprintf(out, n, "%s/weights/model.bin", dir);
}

static bool parse_arena_arg(const char *arg,
                            char names[4][64],
                            ArenaVariant variants[4]) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", arg);
    int n = 0;
    char *tok = strtok(tmp, ",");
    while (tok != NULL && n < 4) {
        while (*tok == ' ') tok++;
        snprintf(names[n], 64, "%s", tok);
        size_t len = strlen(names[n]);
        while (len > 0 && names[n][len - 1] == ' ') {
            names[n][--len] = '\0';
        }
        if (names[n][0] == '\0') return false;
        if (!configure_arena_variant(names[n], &variants[n])) return false;
        variants[n].name = names[n];
        n++;
        tok = strtok(NULL, ",");
    }
    return n == 4 && tok == NULL;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--agent h-s|h-s+|ab2|m2_0ply] [--games N] [--seed S]\n"
        "          [--h2h|--ffa --opponent h-s|h-s+|ab2|m2_0ply] [--weights PATH] [--verbose]\n"
        "          [--arena v0,v1,v2,v3]\n"
        "          [--plus-depth N] [--plus-k K[,K...]] [--plus-time-ms N]\n"
        "          [--plus-workers N] [--plus-leaf-mode N] [--plus-opp-model ab2|det-ab2|det-kf-ab2|det-maxn|det-kf-maxn|nested-hs2|nested-hs3|nested-hs4|hs|hs-leaf]\n"
        "          [--plus-opp-depth N] [--plus-tt-bits N] [--plus-pvs 0|1] [--plus-lmr 0|1] [--plus-id 0|1]\n"
        "          [--plus-rescue 0|1] [--plus-leaf-extend 0|1] [--plus-policy-profile N] [--plus-root-rollout N]\n"
        "          [--plus-root-ensemble N]\n"
        "          [--hs-tt-bits N] [--hs-pvs 0|1] [--hs-lmr 0|1] [--hs-id 0|1]\n"
        "          [--hs-rescue 0|1] [--hs-leaf-extend 0|1] [--hs-policy-profile N] [--hs-root-rollout N]\n"
        "          [--hs-root-ensemble N]\n\n"
        "Agents:\n"
        "  m2_0ply       M2 neural policy argmax, no search\n"
        "  H-S           strongest validated no-ML heuristic search setup\n"
        "  H-S+          wider parallel H-S variant for local-compute sweeps\n"
        "  AB2           depth-2 alpha-beta with expectimax over chance nodes\n",
        argv0);
}

int main(int argc, char **argv) {
    AgentKind agent = AGENT_HS;
    AgentKind opponent = AGENT_M2_0PLY;
    uint64_t seed_base = 42;
    int games = 1;
    bool h2h = false;
    bool ffa = false;
    bool arena = false;
    bool verbose = false;
    char weights_path[1024];
    char arena_arg[512] = "";
    char arena_names[4][64] = {{0}};
    ArenaVariant arena_variants[4];
    HSConfig hs_cfg;
    HSConfig plus_cfg;
    hs_config_default(&hs_cfg);
    hs_plus_config_default(&plus_cfg);
    default_weights_path(weights_path, sizeof(weights_path), argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--agent") == 0 && i + 1 < argc) {
            if (!parse_agent(argv[++i], &agent)) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--opponent") == 0 && i + 1 < argc) {
            if (!parse_agent(argv[++i], &opponent)) {
                usage(argv[0]);
                return 2;
            }
        } else if (strcmp(argv[i], "--arena") == 0 && i + 1 < argc) {
            snprintf(arena_arg, sizeof(arena_arg), "%s", argv[++i]);
            arena = true;
        } else if (strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            games = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_base = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            snprintf(weights_path, sizeof(weights_path), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--plus-depth") == 0 && i + 1 < argc) {
            plus_cfg.depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-k") == 0 && i + 1 < argc) {
            if (!parse_k_schedule_arg(argv[++i], &plus_cfg)) {
                fprintf(stderr, "Invalid --plus-k schedule\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--plus-time-ms") == 0 && i + 1 < argc) {
            plus_cfg.time_budget_sec = atof(argv[++i]) / 1000.0;
        } else if (strcmp(argv[i], "--plus-workers") == 0 && i + 1 < argc) {
            plus_cfg.root_workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-leaf-mode") == 0 && i + 1 < argc) {
            plus_cfg.leaf_mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-opp-model") == 0 && i + 1 < argc) {
            int model = parse_opponent_model(argv[++i]);
            if (model < 0) {
                fprintf(stderr, "Invalid --plus-opp-model\n");
                return 2;
            }
            plus_cfg.opponent_model = model;
        } else if (strcmp(argv[i], "--plus-opp-depth") == 0 && i + 1 < argc) {
            plus_cfg.opp_ab_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-cache-bits") == 0 && i + 1 < argc) {
            plus_cfg.cache_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-tt-bits") == 0 && i + 1 < argc) {
            plus_cfg.tt_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-pvs") == 0 && i + 1 < argc) {
            plus_cfg.pvs_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--plus-lmr") == 0 && i + 1 < argc) {
            plus_cfg.lmr_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--plus-id") == 0 && i + 1 < argc) {
            plus_cfg.id_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--plus-rescue") == 0 && i + 1 < argc) {
            plus_cfg.rescue_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--plus-leaf-extend") == 0 && i + 1 < argc) {
            plus_cfg.leaf_extend_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--plus-policy-profile") == 0 && i + 1 < argc) {
            plus_cfg.policy_profile = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-root-rollout") == 0 && i + 1 < argc) {
            plus_cfg.root_rollout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plus-root-ensemble") == 0 && i + 1 < argc) {
            plus_cfg.root_ensemble = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-depth") == 0 && i + 1 < argc) {
            hs_cfg.depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-k") == 0 && i + 1 < argc) {
            if (!parse_k_schedule_arg(argv[++i], &hs_cfg)) {
                fprintf(stderr, "Invalid --hs-k schedule\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--hs-time-ms") == 0 && i + 1 < argc) {
            hs_cfg.time_budget_sec = atof(argv[++i]) / 1000.0;
        } else if (strcmp(argv[i], "--hs-workers") == 0 && i + 1 < argc) {
            hs_cfg.root_workers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-leaf-mode") == 0 && i + 1 < argc) {
            hs_cfg.leaf_mode = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-opp-model") == 0 && i + 1 < argc) {
            int model = parse_opponent_model(argv[++i]);
            if (model < 0) {
                fprintf(stderr, "Invalid --hs-opp-model\n");
                return 2;
            }
            hs_cfg.opponent_model = model;
        } else if (strcmp(argv[i], "--hs-opp-depth") == 0 && i + 1 < argc) {
            hs_cfg.opp_ab_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-cache-bits") == 0 && i + 1 < argc) {
            hs_cfg.cache_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-tt-bits") == 0 && i + 1 < argc) {
            hs_cfg.tt_bits = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-pvs") == 0 && i + 1 < argc) {
            hs_cfg.pvs_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--hs-lmr") == 0 && i + 1 < argc) {
            hs_cfg.lmr_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--hs-id") == 0 && i + 1 < argc) {
            hs_cfg.id_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--hs-rescue") == 0 && i + 1 < argc) {
            hs_cfg.rescue_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--hs-leaf-extend") == 0 && i + 1 < argc) {
            hs_cfg.leaf_extend_enabled = atoi(argv[++i]) ? 1 : 0;
        } else if (strcmp(argv[i], "--hs-policy-profile") == 0 && i + 1 < argc) {
            hs_cfg.policy_profile = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-root-rollout") == 0 && i + 1 < argc) {
            hs_cfg.root_rollout = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hs-root-ensemble") == 0 && i + 1 < argc) {
            hs_cfg.root_ensemble = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--h2h") == 0) {
            h2h = true;
        } else if (strcmp(argv[i], "--ffa") == 0) {
            ffa = true;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    if (games < 1) games = 1;
    if (h2h && ffa) {
        fprintf(stderr, "--h2h and --ffa are mutually exclusive\n");
        return 2;
    }
    if (arena && (h2h || ffa)) {
        fprintf(stderr, "--arena cannot be combined with --h2h or --ffa\n");
        return 2;
    }
    if (arena && !parse_arena_arg(arena_arg, arena_names, arena_variants)) {
        fprintf(stderr, "Invalid --arena list; expected four known variants\n");
        return 2;
    }

    NNModel *model = NULL;
    bool uses_opponent = h2h || ffa || arena;
    bool arena_needs_model = false;
    if (arena) {
        for (int i = 0; i < 4; i++) {
            if (arena_variants[i].kind == AGENT_M2_0PLY) arena_needs_model = true;
        }
    }
    if (load_model_if_needed(&model, weights_path,
                             arena ? arena_needs_model
                                   : needs_model(agent, opponent, uses_opponent)) != 0) {
        return 1;
    }

    if (arena) {
        printf("catan_player: arena=%s,%s,%s,%s games=%d seed=%llu\n",
               arena_variants[0].name, arena_variants[1].name,
               arena_variants[2].name, arena_variants[3].name,
               games, (unsigned long long)seed_base);
    } else {
        printf("catan_player: agent=%s", agent_name(agent));
        if (uses_opponent) printf(" opponent=%s", agent_name(opponent));
        printf(" games=%d seed=%llu\n", games, (unsigned long long)seed_base);
        if (agent == AGENT_HS || (uses_opponent && opponent == AGENT_HS)) {
            print_hs_config("H-S", &hs_cfg);
        }
        if (agent == AGENT_HS_PLUS || (uses_opponent && opponent == AGENT_HS_PLUS)) {
            print_hs_config("H-S+", &plus_cfg);
        }
    }

    int seat_wins[4] = {0};
    int arena_wins[4] = {0};
    int team_wins = 0, opp_wins = 0;
    int primary_wins = 0, field_wins = 0;
    int total_turns = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gi = 0; gi < games; gi++) {
        uint64_t seed = seed_base + (uint64_t)gi;
        RngState map_rng;
        CatanMap map;
        rng_init(&map_rng, seed);
        build_map(&map, MAP_BASE, 0, &map_rng);

        Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_WHITE, COLOR_ORANGE};
        Game g;
        game_init_with_map(&g, &map, 4, colors, seed, 7, false, 10);

        StateEncoderC enc;
        state_encoder_init(&enc, &g, 4);

        AgentRuntime rt[4];
        int team_parity = gi & 1;
        int primary_seat = gi & 3;
        for (int s = 0; s < 4; s++) {
            AgentKind k = agent;
            if (arena) {
                init_runtime_with_config(&rt[s], arena_variants[s].kind,
                                         &arena_variants[s].hs_cfg);
                continue;
            } else if (h2h) {
                bool team_seat = (s == team_parity || s == team_parity + 2);
                k = team_seat ? agent : opponent;
            } else if (ffa) {
                k = (s == primary_seat) ? agent : opponent;
            }
            init_runtime(&rt[s], k, &hs_cfg, &plus_cfg);
        }

        Action actions[MAX_ACTIONS];
        int decisions = 0;
        while (game_winning_color(&g) == COLOR_NONE && g.state.num_turns < 500) {
            int n_actions = generate_playable_actions(&g.state, actions, MAX_ACTIONS);
            if (n_actions <= 0) break;

            int cp = g.state.current_player_index;
            int chosen = 0;
            if (n_actions > 1) {
                if (rt[cp].kind == AGENT_M2_0PLY) {
                    chosen = choose_m2_0ply(&rt[cp], model, &enc, &g, actions, n_actions);
                } else if (rt[cp].kind == AGENT_HS || rt[cp].kind == AGENT_HS_PLUS) {
                    chosen = choose_hs(&rt[cp], &g, actions, n_actions);
                } else {
                    chosen = choose_ab2(&rt[cp], &g, actions, n_actions);
                }
                decisions++;
            }

            if (verbose) {
                printf("  T%3d P%d %-12s %s\n",
                       g.state.num_turns, cp, agent_name(rt[cp].kind),
                       action_name(actions[chosen].type));
            }

            int next_n = 0;
            game_execute(&g, actions[chosen], actions, &next_n);
        }

        Color winner = game_winning_color(&g);
        int wi = winner == COLOR_NONE ? -1 : g.state.color_to_index[(int)winner];
        if (wi >= 0) seat_wins[wi]++;
        if (arena && wi >= 0) arena_wins[wi]++;
        total_turns += g.state.num_turns;

        int vp[4] = {
            g.state.player_state[0][PS_ACTUAL_VICTORY_POINTS],
            g.state.player_state[1][PS_ACTUAL_VICTORY_POINTS],
            g.state.player_state[2][PS_ACTUAL_VICTORY_POINTS],
            g.state.player_state[3][PS_ACTUAL_VICTORY_POINTS],
        };

        if (h2h && wi >= 0) {
            bool team_win = (wi == team_parity || wi == team_parity + 2);
            if (team_win) team_wins++;
            else opp_wins++;
        } else if (ffa && wi >= 0) {
            if (wi == primary_seat) primary_wins++;
            else field_wins++;
        }

        if (arena) {
            printf("ARENA: seed=%llu winner=P%d winner_variant=%s variants=[%s,%s,%s,%s] actualVP=[%d %d %d %d] turns=%d decisions=%d\n",
                   (unsigned long long)seed, wi,
                   wi >= 0 ? arena_variants[wi].name : "none",
                   arena_variants[0].name, arena_variants[1].name,
                   arena_variants[2].name, arena_variants[3].name,
                   vp[0], vp[1], vp[2], vp[3], g.state.num_turns, decisions);
            fflush(stdout);
        } else if (games == 1 || h2h || ffa) {
            printf("[%d/%d] seed=%llu winner=P%d actualVP=[%d %d %d %d] turns=%d decisions=%d",
                   gi + 1, games, (unsigned long long)seed, wi,
                   vp[0], vp[1], vp[2], vp[3], g.state.num_turns, decisions);
            if (h2h) {
                printf(" team=%s seats=[%d,%d] opp=%s",
                       agent_name(agent), team_parity, team_parity + 2,
                       agent_name(opponent));
            } else if (ffa) {
                printf(" primary=%s seat=%d field=%s",
                       agent_name(agent), primary_seat, agent_name(opponent));
            }
            printf("\n");
            fflush(stdout);
        }

        for (int s = 0; s < 4; s++) destroy_runtime(&rt[s]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("\n%d games in %.2fs (%.2f games/s), avg turns %.1f\n",
           games, elapsed, games / elapsed, (double)total_turns / games);
    if (h2h) {
        int decided = team_wins + opp_wins;
        printf("H2H: %s=%d %s=%d WR=%.1f%%\n",
               agent_name(agent), team_wins, agent_name(opponent), opp_wins,
               decided ? 100.0 * team_wins / decided : 0.0);
    } else if (ffa) {
        int decided = primary_wins + field_wins;
        printf("FFA: %s=%d field_%s=%d WR=%.1f%%\n",
               agent_name(agent), primary_wins,
               agent_name(opponent), field_wins,
               decided ? 100.0 * primary_wins / decided : 0.0);
    } else if (arena) {
        printf("Arena wins: %s=%d %s=%d %s=%d %s=%d\n",
               arena_variants[0].name, arena_wins[0],
               arena_variants[1].name, arena_wins[1],
               arena_variants[2].name, arena_wins[2],
               arena_variants[3].name, arena_wins[3]);
    } else {
        printf("Seat wins: P0=%d P1=%d P2=%d P3=%d\n",
               seat_wins[0], seat_wins[1], seat_wins[2], seat_wins[3]);
    }

    free(model);
    return 0;
}
