/*
 * SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer
 * for the 11 value function weights. Each iteration:
 *   1. Perturb all weights by +delta or -delta (random signs)
 *   2. Play N games: theta+delta vs theta-delta
 *   3. Estimate gradient from win rate difference
 *   4. Update theta in the gradient direction
 *
 * This is the standard approach for tuning chess/game engine parameters.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define NUM_PARAMS 11
#define GAMES_PER_ITER 200  /* 100 per side */
#define NUM_ITERATIONS 500
#define AB_DEPTH 2

/* Current weights (mutable during optimization) */
static double theta[NUM_PARAMS];
static double plus_weights[NUM_PARAMS];
static double minus_weights[NUM_PARAMS];

/* Which weight set to use: 0=plus, 1=minus */
static int active_weights = 0;

static const char *param_names[NUM_PARAMS] = {
    "vps", "prod", "eprod", "tiles", "buildable",
    "road", "synergy", "hand", "discard", "devs", "army"
};

/* Initial values matching current defaults */
static const double INIT[NUM_PARAMS] = {
    3e14, 1e8, -1e8, 1.0, 1e3,
    10.0, 1e2, 1.0, -5.0, 10.0, 10.1
};

/* Perturbation sizes (proportional to parameter magnitude) */
static const double DELTA[NUM_PARAMS] = {
    3e12, 1e6, 1e6, 0.5, 1e2,
    2.0, 1e1, 0.5, 1.0, 2.0, 2.0
};

/* Learning rates */
static const double LR[NUM_PARAMS] = {
    3e12, 1e6, 1e6, 0.3, 50.0,
    1.0, 5.0, 0.3, 0.5, 1.0, 1.0
};

/* Override value function to use tunable weights */
static const double DICE_P[13] = {
    0, 0,
    1.0/36, 2.0/36, 3.0/36, 4.0/36, 5.0/36, 6.0/36,
    5.0/36, 4.0/36, 3.0/36, 2.0/36, 1.0/36
};

static double eval_with_weights(Game *g, Color p0_color, const double w[NUM_PARAMS]) {
    State *s = &g->state;
    int idx = s->color_to_index[(int)p0_color];
    Board *b = &s->board;
    CatanMap *map = b->map;
    Coordinate robber = b->robber_coordinate;

    /* Production */
    double res_prod[5] = {0};
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[tile_idx];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
            res_prod[(int)t->resource] += DICE_P[t->number];
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[tile_idx];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
            res_prod[(int)t->resource] += 2.0 * DICE_P[t->number];
        }
    }
    double total_prod = 0; int variety = 0;
    for (int r = 0; r < 5; r++) { total_prod += res_prod[r]; if (res_prod[r] > 0) variety++; }
    double production = total_prod + variety * 4.0 * (2.778/100.0);

    /* Enemy production */
    Color enemy = COLOR_NONE;
    for (int i = 0; i < s->num_players; i++)
        if (s->colors[i] != p0_color) { enemy = s->colors[i]; break; }
    double e_prod = 0;
    if (enemy != COLOR_NONE) {
        int ei = s->color_to_index[(int)enemy];
        double eres[5] = {0};
        for (int si = 0; si < s->settlement_count[ei]; si++) {
            int node = s->settlements[ei][si];
            for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
                int tile_idx = map->adjacent_tiles[node][ti];
                LandTile *t = &map->land_tiles[tile_idx];
                if (t->resource == RES_NONE || t->number == 0) continue;
                if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
                eres[(int)t->resource] += DICE_P[t->number];
            }
        }
        for (int ci = 0; ci < s->city_count[ei]; ci++) {
            int node = s->cities[ei][ci];
            for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
                int tile_idx = map->adjacent_tiles[node][ti];
                LandTile *t = &map->land_tiles[tile_idx];
                if (t->resource == RES_NONE || t->number == 0) continue;
                if (coord_eq(map->land_tile_coords[tile_idx], robber)) continue;
                eres[(int)t->resource] += 2.0 * DICE_P[t->number];
            }
        }
        for (int r = 0; r < 5; r++) e_prod += eres[r];
    }

    int *ps = s->player_state[idx];
    int wheat = ps[PS_WHEAT_IN_HAND], ore = ps[PS_ORE_IN_HAND];
    int sheep = ps[PS_SHEEP_IN_HAND], brick = ps[PS_BRICK_IN_HAND], wood = ps[PS_WOOD_IN_HAND];
    double d_city = (fmax(2-wheat,0) + fmax(3-ore,0)) / 5.0;
    double d_settle = (fmax(1-wheat,0) + fmax(1-sheep,0) + fmax(1-brick,0) + fmax(1-wood,0)) / 4.0;
    double hand_synergy = (2 - d_city - d_settle) / 2.0;
    int num_in_hand = wood + brick + sheep + wheat + ore;

    bool tile_seen[NUM_LAND_TILES] = {false}; int num_tiles = 0;
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            if (!tile_seen[tile_idx]) { tile_seen[tile_idx] = true; num_tiles++; }
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int tile_idx = map->adjacent_tiles[node][ti];
            if (!tile_seen[tile_idx]) { tile_seen[tile_idx] = true; num_tiles++; }
        }
    }

    int buildable_buf[TOTAL_NODES];
    int num_buildable = board_buildable_node_ids(&s->board, p0_color, false, buildable_buf, TOTAL_NODES);
    double lr_factor = (num_buildable == 0) ? w[5] : 0.1;

    int num_devs = ps[PS_KNIGHT_IN_HAND] + ps[PS_YEAR_OF_PLENTY_IN_HAND]
                 + ps[PS_MONOPOLY_IN_HAND] + ps[PS_ROAD_BUILDING_IN_HAND]
                 + ps[PS_VICTORY_POINT_IN_HAND];

    return ps[PS_VICTORY_POINTS] * w[0]
         + production * w[1]
         + e_prod * w[2]
         + num_tiles * w[3]
         + num_buildable * w[4]
         + ps[PS_LONGEST_ROAD_LENGTH] * lr_factor
         + hand_synergy * w[6]
         + num_in_hand * w[7]
         + (num_in_hand > 7 ? w[8] : 0)
         + num_devs * w[9]
         + ps[PS_PLAYED_KNIGHT] * w[10];
}

/* Patched value function that reads from active_weights */
double tunable_value_fn(Game *g, Color c) {
    return eval_with_weights(g, c, active_weights == 0 ? plus_weights : minus_weights);
}

/* Play a match: plus_weights (RED) vs minus_weights (BLUE).
 * Returns: score for plus side (1.0 win, 0.5 draw, 0.0 loss) */
static double play_match(int seed) {
    Color colors[2] = {COLOR_RED, COLOR_BLUE};
    CatanMap map;
    rng_seed((uint64_t)seed);
    build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

    Game game;
    game_init_with_map(&game, &map, 2, colors, (uint64_t)seed, 7, false, 10);

    Action actions[MAX_ACTIONS];
    int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

    while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
        Color cur = state_current_color(&game.state);
        Action action;
        if (n == 1) {
            action = actions[0];
        } else {
            active_weights = (cur == COLOR_RED) ? 0 : 1;
            /* Inline AB search using tunable eval */
            double deadline = (double)clock()/CLOCKS_PER_SEC + 120.0;
            Game copy;
            game_copy(&copy, &game);
            SearchResult sr = alphabeta_search(&copy, actions, n,
                AB_DEPTH, -1e30, 1e30, deadline, cur);
            action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
        }
        game_execute(&game, action, actions, &n);
    }

    Color w = game_winning_color(&game);
    if (w == COLOR_RED) return 1.0;
    if (w == COLOR_BLUE) return 0.0;
    return 0.5;
}

int main(void) {
    memcpy(theta, INIT, sizeof(INIT));

    double best_theta[NUM_PARAMS];
    memcpy(best_theta, theta, sizeof(theta));
    double best_score = 0.5;

    printf("SPSA Optimization: %d params, %d games/iter, %d iterations\n",
           NUM_PARAMS, GAMES_PER_ITER, NUM_ITERATIONS);
    printf("Initial weights:\n  ");
    for (int i = 0; i < NUM_PARAMS; i++) printf("%s=%.4g ", param_names[i], theta[i]);
    printf("\n\n");

    clock_t total_start = clock();
    int total_games = 0;

    /* Use C stdlib rand for perturbation signs (separate from game RNG) */
    srand(12345);

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        /* Generate random perturbation direction */
        int signs[NUM_PARAMS];
        for (int i = 0; i < NUM_PARAMS; i++)
            signs[i] = (rand() % 2) * 2 - 1;  /* +1 or -1 */

        /* Decay schedule */
        double ck = 1.0 / pow(iter + 1, 0.17);
        double ak = 1.0 / pow(iter + 1 + 50, 0.6);

        for (int i = 0; i < NUM_PARAMS; i++) {
            plus_weights[i]  = theta[i] + ck * DELTA[i] * signs[i];
            minus_weights[i] = theta[i] - ck * DELTA[i] * signs[i];
        }

        /* Play matches: half with plus=RED, half with plus=BLUE */
        double plus_score = 0;
        int half = GAMES_PER_ITER / 2;
        for (int g = 0; g < half; g++) {
            int seed = iter * GAMES_PER_ITER + g;
            plus_score += play_match(seed);
        }
        /* Swap sides */
        double tmp[NUM_PARAMS];
        memcpy(tmp, plus_weights, sizeof(tmp));
        memcpy(plus_weights, minus_weights, sizeof(tmp));
        memcpy(minus_weights, tmp, sizeof(tmp));
        for (int g = half; g < GAMES_PER_ITER; g++) {
            int seed = iter * GAMES_PER_ITER + g;
            plus_score += (1.0 - play_match(seed));
        }
        /* Restore */
        memcpy(tmp, plus_weights, sizeof(tmp));
        memcpy(plus_weights, minus_weights, sizeof(tmp));
        memcpy(minus_weights, tmp, sizeof(tmp));

        double win_rate = plus_score / GAMES_PER_ITER;
        double gradient_signal = (win_rate - 0.5) * 4.0;  /* scale to [-2, 2] */

        /* Update theta */
        for (int i = 0; i < NUM_PARAMS; i++) {
            theta[i] += ak * LR[i] * gradient_signal * signs[i];
        }

        total_games += GAMES_PER_ITER;

        if (win_rate > best_score) {
            best_score = win_rate;
            memcpy(best_theta, plus_weights, sizeof(theta));
        }

        if ((iter + 1) % 10 == 0) {
            double elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
            printf("iter %3d: plus_wr=%.3f grad=%.3f  [%d games, %.0fs, %.0f g/s]\n",
                   iter + 1, win_rate, gradient_signal, total_games, elapsed,
                   total_games / elapsed);
            if ((iter + 1) % 50 == 0) {
                printf("  theta: ");
                for (int i = 0; i < NUM_PARAMS; i++)
                    printf("%s=%.4g ", param_names[i], theta[i]);
                printf("\n");
            }
        }
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;

    printf("\n==========================================\n");
    printf("  SPSA Optimization Complete\n");
    printf("  %d iterations, %d total games, %.1fs\n",
           NUM_ITERATIONS, total_games, total_elapsed);
    printf("==========================================\n");
    printf("\nOptimized weights:\n");
    for (int i = 0; i < NUM_PARAMS; i++)
        printf("  %-12s = %20.6f  (was %20.6f,  delta=%+.4g)\n",
               param_names[i], theta[i], INIT[i], theta[i] - INIT[i]);

    printf("\nC code:\n");
    printf("static const double W_VPS       = %.6g;\n", theta[0]);
    printf("static const double W_PROD      = %.6g;\n", theta[1]);
    printf("static const double W_EPROD     = %.6g;\n", theta[2]);
    printf("static const double W_TILES     = %.6g;\n", theta[3]);
    printf("static const double W_BUILDABLE = %.6g;\n", theta[4]);
    printf("static const double W_ROAD      = %.6g;\n", theta[5]);
    printf("static const double W_SYNERGY   = %.6g;\n", theta[6]);
    printf("static const double W_HAND      = %.6g;\n", theta[7]);
    printf("static const double W_DISCARD   = %.6g;\n", theta[8]);
    printf("static const double W_DEVS      = %.6g;\n", theta[9]);
    printf("static const double W_ARMY      = %.6g;\n", theta[10]);

    return 0;
}
