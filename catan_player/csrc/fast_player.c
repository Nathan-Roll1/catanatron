/*
 * fast_player.c -- minimal standalone C runner for two agents:
 *   - m2_0ply: M2 neural policy argmax, no search
 *   - H-S: strongest validated no-ML heuristic search bot
 */

#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "actions.h"
#include "board.h"
#include "catan_types.h"
#include "deep_search.h"
#include "game.h"
#include "map.h"
#include "nn.h"
#include "policy_topk.h"
#include "state_encode.h"

typedef enum {
    AGENT_M2_0PLY = 0,
    AGENT_HS = 1,
} AgentKind;

typedef struct {
    AgentKind kind;
    DeepSearchCtx *leaf_ctx;
    float nf[ENC_NUM_NODES * ENC_NODE_FEAT_DIM];
    float ef[ENC_NUM_EDGES * ENC_EDGE_FEAT_DIM];
    float ff[ENC_FLAT_FEAT_DIM];
    float mk[NN_MASK_DIM];
    float out[4 + NN_MASK_DIM];
} AgentRuntime;

static const int HS_DEPTH = 5;
static const int HS_K[5] = {6, 4, 2, 2, 2};
static const int HS_OPP_AB_DEPTH = 2;
static const int HS_CACHE_BITS = 20;

static const char *agent_name(AgentKind k) {
    return k == AGENT_M2_0PLY ? "m2_0ply" : "H-S";
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

    int k_root = HS_K[0];
    if (k_root > n_actions) k_root = n_actions;

    int top[16];
    int n_top = policy_top_k_ex(NULL, NULL, g, actions, n_actions, k_root, top,
                                rt->nf, rt->ef, rt->ff, rt->mk, rt->out, 1);
    if (n_top <= 0) return 0;

    int best_pos = 0;
    Color us = state_current_color(&g->state);
    deep_search_root(rt->leaf_ctx, g, us, top, n_top, &best_pos);
    if (best_pos < 0 || best_pos >= n_top) best_pos = 0;
    return fix_robber_steal(top[best_pos], actions, n_actions);
}

static bool needs_model(AgentKind a, AgentKind b, bool h2h) {
    return a == AGENT_M2_0PLY || (h2h && b == AGENT_M2_0PLY);
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

static void init_runtime(AgentRuntime *rt, AgentKind kind) {
    memset(rt, 0, sizeof(*rt));
    rt->kind = kind;
    if (kind == AGENT_HS) {
        rt->leaf_ctx = deep_search_create_c(HS_CACHE_BITS, NULL, NULL);
        if (!rt->leaf_ctx) {
            fprintf(stderr, "deep_search_create_c failed\n");
            exit(1);
        }
        deep_search_configure(rt->leaf_ctx, HS_DEPTH, HS_K, 5,
                              HS_OPP_AB_DEPTH, 5.0);
        deep_search_set_algo_policy(rt->leaf_ctx, 1);
    }
}

static void destroy_runtime(AgentRuntime *rt) {
    if (rt->leaf_ctx) deep_search_destroy(rt->leaf_ctx);
    rt->leaf_ctx = NULL;
}

static void default_weights_path(char *out, size_t n, const char *argv0) {
    char tmp[1024];
    strncpy(tmp, argv0, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    char *dir = dirname(tmp);
    snprintf(out, n, "%s/weights/model.bin", dir);
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage: %s [--agent h-s|m2_0ply] [--games N] [--seed S]\n"
        "          [--h2h --opponent h-s|m2_0ply] [--weights PATH] [--verbose]\n\n"
        "Agents:\n"
        "  m2_0ply       M2 neural policy argmax, no search\n"
        "  H-S           strongest validated no-ML heuristic search setup\n",
        argv0);
}

int main(int argc, char **argv) {
    AgentKind agent = AGENT_HS;
    AgentKind opponent = AGENT_M2_0PLY;
    uint64_t seed_base = 42;
    int games = 1;
    bool h2h = false;
    bool verbose = false;
    char weights_path[1024];
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
        } else if (strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            games = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed_base = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            snprintf(weights_path, sizeof(weights_path), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--h2h") == 0) {
            h2h = true;
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

    NNModel *model = NULL;
    if (load_model_if_needed(&model, weights_path, needs_model(agent, opponent, h2h)) != 0) {
        return 1;
    }

    printf("catan_player: agent=%s", agent_name(agent));
    if (h2h) printf(" opponent=%s", agent_name(opponent));
    printf(" games=%d seed=%llu\n", games, (unsigned long long)seed_base);

    int seat_wins[4] = {0};
    int team_wins = 0, opp_wins = 0;
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
        for (int s = 0; s < 4; s++) {
            AgentKind k = agent;
            if (h2h) {
                bool team_seat = (s == team_parity || s == team_parity + 2);
                k = team_seat ? agent : opponent;
            }
            init_runtime(&rt[s], k);
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
                } else {
                    chosen = choose_hs(&rt[cp], &g, actions, n_actions);
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
        }

        if (games == 1 || h2h) {
            printf("[%d/%d] seed=%llu winner=P%d actualVP=[%d %d %d %d] turns=%d decisions=%d",
                   gi + 1, games, (unsigned long long)seed, wi,
                   vp[0], vp[1], vp[2], vp[3], g.state.num_turns, decisions);
            if (h2h) {
                printf(" team=%s seats=[%d,%d] opp=%s",
                       agent_name(agent), team_parity, team_parity + 2,
                       agent_name(opponent));
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
    } else {
        printf("Seat wins: P0=%d P1=%d P2=%d P3=%d\n",
               seat_wins[0], seat_wins[1], seat_wins[2], seat_wins[3]);
    }

    free(model);
    return 0;
}
