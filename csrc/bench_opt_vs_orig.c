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
#define N_GAMES 1000
#define AB_DEPTH 2

static const double DICE_P[13] = {
    0, 0, 1.0/36, 2.0/36, 3.0/36, 4.0/36, 5.0/36, 6.0/36,
    5.0/36, 4.0/36, 3.0/36, 2.0/36, 1.0/36
};

static const double ORIG[NUM_PARAMS] = {
    3e14, 1e8, -1e8, 1.0, 1e3, 10.0, 1e2, 1.0, -5.0, 10.0, 10.1
};
static const double OPT[NUM_PARAMS] = {
    3.00023e+14, 1.00003e+08, -1.0001e+08, 0.993348, 991.919,
    9.90644, 98.5322, 1.07375, -4.95745, 9.84593, 10.414
};

static const double *active_w;

static double eval_w(Game *g, Color p0, const double w[NUM_PARAMS]) {
    State *s = &g->state;
    int idx = s->color_to_index[(int)p0];
    Board *b = &s->board;
    CatanMap *map = b->map;
    Coordinate robber = b->robber_coordinate;

    double res_prod[5] = {0};
    for (int si = 0; si < s->settlement_count[idx]; si++) {
        int node = s->settlements[idx][si];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int ti2 = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[ti2];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[ti2], robber)) continue;
            res_prod[(int)t->resource] += DICE_P[t->number];
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int ti2 = map->adjacent_tiles[node][ti];
            LandTile *t = &map->land_tiles[ti2];
            if (t->resource == RES_NONE || t->number == 0) continue;
            if (coord_eq(map->land_tile_coords[ti2], robber)) continue;
            res_prod[(int)t->resource] += 2.0 * DICE_P[t->number];
        }
    }
    double total_prod = 0; int variety = 0;
    for (int r = 0; r < 5; r++) { total_prod += res_prod[r]; if (res_prod[r] > 0) variety++; }
    double production = total_prod + variety * 4.0 * (2.778/100.0);

    Color enemy = COLOR_NONE;
    for (int i = 0; i < s->num_players; i++)
        if (s->colors[i] != p0) { enemy = s->colors[i]; break; }
    double e_prod = 0;
    if (enemy != COLOR_NONE) {
        int ei = s->color_to_index[(int)enemy];
        for (int si = 0; si < s->settlement_count[ei]; si++) {
            int node = s->settlements[ei][si];
            for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
                int ti2 = map->adjacent_tiles[node][ti];
                LandTile *t = &map->land_tiles[ti2];
                if (t->resource == RES_NONE || t->number == 0) continue;
                if (coord_eq(map->land_tile_coords[ti2], robber)) continue;
                e_prod += DICE_P[t->number];
            }
        }
        for (int ci = 0; ci < s->city_count[ei]; ci++) {
            int node = s->cities[ei][ci];
            for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
                int ti2 = map->adjacent_tiles[node][ti];
                LandTile *t = &map->land_tiles[ti2];
                if (t->resource == RES_NONE || t->number == 0) continue;
                if (coord_eq(map->land_tile_coords[ti2], robber)) continue;
                e_prod += 2.0 * DICE_P[t->number];
            }
        }
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
            int ti2 = map->adjacent_tiles[node][ti];
            if (!tile_seen[ti2]) { tile_seen[ti2] = true; num_tiles++; }
        }
    }
    for (int ci = 0; ci < s->city_count[idx]; ci++) {
        int node = s->cities[idx][ci];
        for (int ti = 0; ti < map->adjacent_tiles_count[node]; ti++) {
            int ti2 = map->adjacent_tiles[node][ti];
            if (!tile_seen[ti2]) { tile_seen[ti2] = true; num_tiles++; }
        }
    }

    int buf[TOTAL_NODES];
    int num_buildable = board_buildable_node_ids(&s->board, p0, false, buf, TOTAL_NODES);
    double lr_factor = (num_buildable == 0) ? w[5] : 0.1;
    int num_devs = ps[PS_KNIGHT_IN_HAND] + ps[PS_YEAR_OF_PLENTY_IN_HAND]
                 + ps[PS_MONOPOLY_IN_HAND] + ps[PS_ROAD_BUILDING_IN_HAND]
                 + ps[PS_VICTORY_POINT_IN_HAND];

    return ps[PS_VICTORY_POINTS] * w[0] + production * w[1] + e_prod * w[2]
         + num_tiles * w[3] + num_buildable * w[4] + ps[PS_LONGEST_ROAD_LENGTH] * lr_factor
         + hand_synergy * w[6] + num_in_hand * w[7] + (num_in_hand > 7 ? w[8] : 0)
         + num_devs * w[9] + ps[PS_PLAYED_KNIGHT] * w[10];
}

/* Globally track which player uses which weights */
static const double *player_weights[2];

double patched_value_fn(Game *g, Color c) {
    Color red_c = g->state.colors[0];
    /* Determine which weight set this color uses based on original assignment */
    return eval_w(g, c, (c == red_c) ? player_weights[0] : player_weights[1]);
}

int main(void) {
    int opt_wins = 0, orig_wins = 0, draws = 0;
    int opt_wins_as_red = 0, opt_wins_as_blue = 0;
    Color colors[2] = {COLOR_RED, COLOR_BLUE};
    clock_t t0 = clock();

    /* Seed range 500000+ to be completely disjoint from training */
    for (int gi = 0; gi < N_GAMES; gi++) {
        int seed = 500000 + gi;

        /* Alternate: even = opt is RED, odd = opt is BLUE */
        if (gi % 2 == 0) {
            player_weights[0] = OPT;   /* RED = optimized */
            player_weights[1] = ORIG;  /* BLUE = original */
        } else {
            player_weights[0] = ORIG;  /* RED = original */
            player_weights[1] = OPT;   /* BLUE = optimized */
        }

        CatanMap map;
        rng_seed((uint64_t)seed);
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 2, colors, (uint64_t)seed, 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Action action;
            if (n == 1) {
                action = actions[0];
            } else {
                Color cur = state_current_color(&game.state);
                double deadline = (double)clock()/CLOCKS_PER_SEC + 120.0;
                Game copy;
                game_copy(&copy, &game);

                /* Temporarily patch active_w for search eval */
                Color red_c = game.state.colors[0];
                active_w = (cur == red_c) ? player_weights[0] : player_weights[1];

                SearchResult sr = alphabeta_search(&copy, actions, n,
                    AB_DEPTH, -1e30, 1e30, deadline, cur);
                action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
            }
            game_execute(&game, action, actions, &n);
        }

        Color winner = game_winning_color(&game);
        bool opt_is_red = (gi % 2 == 0);
        Color opt_color = opt_is_red ? COLOR_RED : COLOR_BLUE;

        if (winner == opt_color) {
            opt_wins++;
            if (opt_is_red) opt_wins_as_red++; else opt_wins_as_blue++;
        } else if (winner != COLOR_NONE) {
            orig_wins++;
        } else {
            draws++;
        }

        if ((gi + 1) % 100 == 0) {
            double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;
            printf("  [%4d/%d] opt=%d orig=%d draws=%d  opt_wr=%.1f%%  (%.0fs)\n",
                   gi + 1, N_GAMES, opt_wins, orig_wins, draws,
                   100.0 * opt_wins / (gi + 1), elapsed);
        }
    }

    double elapsed = (double)(clock() - t0) / CLOCKS_PER_SEC;

    printf("\n==========================================\n");
    printf("  Optimized vs Original -- %d games\n", N_GAMES);
    printf("  (seeds 500000-%d, disjoint from training)\n", 500000 + N_GAMES - 1);
    printf("==========================================\n");
    printf("  Optimized: %d wins (%.1f%%)\n", opt_wins, 100.0 * opt_wins / N_GAMES);
    printf("  Original:  %d wins (%.1f%%)\n", orig_wins, 100.0 * orig_wins / N_GAMES);
    printf("  Draws:     %d\n", draws);
    printf("\n  Position control:\n");
    printf("    Opt as RED:  %d / %d wins\n", opt_wins_as_red, N_GAMES / 2);
    printf("    Opt as BLUE: %d / %d wins\n", opt_wins_as_blue, N_GAMES / 2);
    printf("\n  Time: %.1fs (%.0f games/sec)\n", elapsed, N_GAMES / elapsed);

    return 0;
}
