/*
 * Fine-grained SPSA: search within 1% of current values.
 * 1000 games per evaluation, 100 iterations = 100k total games.
 * Uses seeds 1M+ to be fully disjoint from all prior runs.
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
#define GAMES_PER_EVAL 1000
#define NUM_ITERS 100
#define AB_DEPTH 2

static const double DICE_P[13] = {
    0, 0, 1.0/36, 2.0/36, 3.0/36, 4.0/36, 5.0/36, 6.0/36,
    5.0/36, 4.0/36, 3.0/36, 2.0/36, 1.0/36
};

static const char *pnames[NUM_PARAMS] = {
    "vps", "prod", "eprod", "tiles", "buildable",
    "road", "synergy", "hand", "discard", "devs", "army"
};

static double theta[NUM_PARAMS] = {
    3e14, 1e8, -1e8, 1.0, 1e3, 10.0, 1e2, 1.0, -5.0, 10.0, 10.1
};

static double plus_w[NUM_PARAMS], minus_w[NUM_PARAMS];
static const double *match_weights[2]; /* [0]=RED weights, [1]=BLUE weights */

static double eval_w(Game *g, Color p0, const double w[NUM_PARAMS]) {
    State *s = &g->state;
    int idx = s->color_to_index[(int)p0];
    Board *b = &s->board; CatanMap *map = b->map;
    Coordinate robber = b->robber_coordinate;

    double rp[5] = {0};
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int t2 = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[t2];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[t2], robber)) continue;
            rp[(int)t->resource] += DICE_P[t->number];
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int t2 = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[t2];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[t2], robber)) continue;
            rp[(int)t->resource] += 2.0 * DICE_P[t->number];
        }
    }
    double tp = 0; int var = 0;
    for (int r = 0; r < 5; r++) { tp += rp[r]; if (rp[r] > 0) var++; }
    double prod = tp + var * 4.0 * (2.778/100.0);

    Color enemy = COLOR_NONE;
    for (int i = 0; i < s->num_players; i++)
        if (s->colors[i] != p0) { enemy = s->colors[i]; break; }
    double ep = 0;
    if (enemy != COLOR_NONE) {
        int ei = s->color_to_index[(int)enemy];
        for (int si = 0; si < s->settlement_count[ei]; si++) {
            int node = s->settlements[ei][si];
            for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
                int t2 = map->adjacent_tiles[node][ti];
                LandTile *t = &map->land_tiles[t2];
                if (t->resource == RES_NONE || t->number == 0) continue;
                if (coord_eq(map->land_tile_coords[t2], robber)) continue;
                ep += DICE_P[t->number];
            }
        }
        for (int ci = 0; ci < s->city_count[ei]; ci++) {
            int node = s->cities[ei][ci];
            for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
                int t2 = map->adjacent_tiles[node][ti];
                LandTile *t = &map->land_tiles[t2];
                if (t->resource == RES_NONE || t->number == 0) continue;
                if (coord_eq(map->land_tile_coords[t2], robber)) continue;
                ep += 2.0 * DICE_P[t->number];
            }
        }
    }

    int *ps = s->player_state[idx];
    int wh = ps[PS_WHEAT_IN_HAND], or2 = ps[PS_ORE_IN_HAND];
    int sh = ps[PS_SHEEP_IN_HAND], br = ps[PS_BRICK_IN_HAND], wo = ps[PS_WOOD_IN_HAND];
    double dc = (fmax(2-wh,0) + fmax(3-or2,0)) / 5.0;
    double ds = (fmax(1-wh,0) + fmax(1-sh,0) + fmax(1-br,0) + fmax(1-wo,0)) / 4.0;
    double hs = (2 - dc - ds) / 2.0;
    int nih = wo+br+sh+wh+or2;

    bool ts[NUM_LAND_TILES] = {false}; int nt = 0;
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int t2 = map->adjacent_tiles[node][ti];
            if (!ts[t2]) { ts[t2] = true; nt++; }
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int t2 = map->adjacent_tiles[node][ti];
            if (!ts[t2]) { ts[t2] = true; nt++; }
        }
    }

    int buf[TOTAL_NODES];
    int nb = board_buildable_node_ids(&s->board, p0, false, buf, TOTAL_NODES);
    double lrf = (nb == 0) ? w[5] : 0.1;
    int nd = ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]
            +ps[PS_MONOPOLY_IN_HAND]+ps[PS_ROAD_BUILDING_IN_HAND]
            +ps[PS_VICTORY_POINT_IN_HAND];

    return ps[PS_VICTORY_POINTS]*w[0] + prod*w[1] + ep*w[2]
         + nt*w[3] + nb*w[4] + ps[PS_LONGEST_ROAD_LENGTH]*lrf
         + hs*w[6] + nih*w[7] + (nih>7 ? w[8] : 0)
         + nd*w[9] + ps[PS_PLAYED_KNIGHT]*w[10];
}

/* Search uses this -- pick weights based on which player is evaluating */
double fine_value_fn(Game *g, Color c) {
    Color rc = g->state.colors[0];
    return eval_w(g, c, (c == rc) ? match_weights[0] : match_weights[1]);
}

/* Play one game, return 1.0 if RED wins, 0.0 if BLUE, 0.5 draw */
static double play_one(int seed) {
    Color colors[2] = {COLOR_RED, COLOR_BLUE};
    CatanMap map;
    rng_seed((uint64_t)seed);
    build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);
    Game game;
    game_init_with_map(&game, &map, 2, colors, (uint64_t)seed, 7, false, 10);
    Action actions[MAX_ACTIONS];
    int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

    while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
        Action action;
        if (n == 1) { action = actions[0]; }
        else {
            double dl = (double)clock()/CLOCKS_PER_SEC + 120.0;
            Game cp; game_copy(&cp, &game);
            SearchResult sr = alphabeta_search(&cp, actions, n, AB_DEPTH, -1e30, 1e30, dl,
                                                state_current_color(&game.state));
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
    printf("Fine SPSA: 1%% perturbations, %d games/eval, %d iters = %dk total\n\n",
           GAMES_PER_EVAL, NUM_ITERS, GAMES_PER_EVAL * NUM_ITERS / 1000);

    srand(99999);
    clock_t t0 = clock();
    int total_games = 0;

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        /* Random perturbation signs */
        int signs[NUM_PARAMS];
        for (int i = 0; i < NUM_PARAMS; i++) signs[i] = (rand() % 2) * 2 - 1;

        /* 1% perturbation of |theta| */
        for (int i = 0; i < NUM_PARAMS; i++) {
            double delta = fabs(theta[i]) * 0.01 * signs[i];
            plus_w[i]  = theta[i] + delta;
            minus_w[i] = theta[i] - delta;
        }

        /* Play GAMES_PER_EVAL: half plus=RED, half plus=BLUE */
        double plus_score = 0;
        int half = GAMES_PER_EVAL / 2;

        /* plus=RED vs minus=BLUE */
        match_weights[0] = plus_w;
        match_weights[1] = minus_w;
        for (int g = 0; g < half; g++)
            plus_score += play_one(1000000 + iter * GAMES_PER_EVAL + g);

        /* Swap: minus=RED vs plus=BLUE */
        match_weights[0] = minus_w;
        match_weights[1] = plus_w;
        for (int g = half; g < GAMES_PER_EVAL; g++)
            plus_score += (1.0 - play_one(1000000 + iter * GAMES_PER_EVAL + g));

        double wr = plus_score / GAMES_PER_EVAL;
        double grad = (wr - 0.5) * 4.0;
        total_games += GAMES_PER_EVAL;

        /* Learning rate: 0.1% of |theta| * gradient */
        for (int i = 0; i < NUM_PARAMS; i++) {
            double step = fabs(theta[i]) * 0.001 * grad * signs[i];
            theta[i] += step;
        }

        double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("iter %3d: wr=%.3f grad=%+.3f  [%dk games, %.0fs, %.0f g/s]",
               iter + 1, wr, grad, total_games/1000, elapsed, total_games/elapsed);

        if ((iter + 1) % 10 == 0) {
            printf("\n  theta:");
            for (int i = 0; i < NUM_PARAMS; i++) {
                double pct = (theta[i] - (i==2 ? -1e8 : (i==8 ? -5.0 :
                    (double[]){3e14,1e8,0,1,1e3,10,1e2,1,-5,10,10.1}[i])))
                    / fabs((double[]){3e14,1e8,-1e8,1,1e3,10,1e2,1,-5,10,10.1}[i]) * 100;
                printf(" %s=%+.2f%%", pnames[i], pct);
            }
            printf("\n");
        } else {
            printf("\n");
        }
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
    printf("\n==========================================\n");
    printf("  Fine SPSA Complete: %dk games in %.0fs\n", total_games/1000, elapsed);
    printf("==========================================\n");
    printf("\nFinal weights (change from original):\n");
    double orig[NUM_PARAMS] = {3e14, 1e8, -1e8, 1.0, 1e3, 10.0, 1e2, 1.0, -5.0, 10.0, 10.1};
    for (int i = 0; i < NUM_PARAMS; i++) {
        double pct = (theta[i] - orig[i]) / fabs(orig[i]) * 100;
        printf("  %-12s = %20.6f  (%+.4f%%)\n", pnames[i], theta[i], pct);
    }

    printf("\nC code:\n");
    for (int i = 0; i < NUM_PARAMS; i++)
        printf("  %s = %.10g\n", pnames[i], theta[i]);

    return 0;
}
