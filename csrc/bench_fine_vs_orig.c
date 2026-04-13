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

static const double ORIG_W[NUM_PARAMS] = {
    3e14, 1e8, -1e8, 1.0, 1e3, 10.0, 1e2, 1.0, -5.0, 10.0, 10.1
};

/* Fine-tuned from 100k game SPSA */
static const double FINE_W[NUM_PARAMS] = {
    2.999651438e+14, 100010781.2, -99987581.37, 0.9999718064, 998.9483594,
    9.999158095, 99.97078487, 0.9998438182, -5.00001903, 9.997398398, 10.10585774
};

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
    int wh=ps[PS_WHEAT_IN_HAND], or2=ps[PS_ORE_IN_HAND];
    int sh=ps[PS_SHEEP_IN_HAND], br=ps[PS_BRICK_IN_HAND], wo=ps[PS_WOOD_IN_HAND];
    double dc = (fmax(2-wh,0)+fmax(3-or2,0))/5.0;
    double ds = (fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.0;
    double hs = (2-dc-ds)/2.0;
    int nih = wo+br+sh+wh+or2;

    bool ts[NUM_LAND_TILES]={false}; int nt=0;
    for (int si=0; si<s->settlement_count[idx]; si++) {
        int node=s->settlements[idx][si];
        for (int ti=0; ti<map->adjacent_tiles_count[node]; ti++) {
            int t2=map->adjacent_tiles[node][ti];
            if(!ts[t2]){ts[t2]=true;nt++;}
        }
    }
    for (int ci=0; ci<s->city_count[idx]; ci++) {
        int node=s->cities[idx][ci];
        for (int ti=0; ti<map->adjacent_tiles_count[node]; ti++) {
            int t2=map->adjacent_tiles[node][ti];
            if(!ts[t2]){ts[t2]=true;nt++;}
        }
    }
    int buf[TOTAL_NODES];
    int nb=board_buildable_node_ids(&s->board,p0,false,buf,TOTAL_NODES);
    double lrf=(nb==0)?w[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]
          +ps[PS_MONOPOLY_IN_HAND]+ps[PS_ROAD_BUILDING_IN_HAND]
          +ps[PS_VICTORY_POINT_IN_HAND];

    return ps[PS_VICTORY_POINTS]*w[0]+prod*w[1]+ep*w[2]+nt*w[3]+nb*w[4]
          +ps[PS_LONGEST_ROAD_LENGTH]*lrf+hs*w[6]+nih*w[7]
          +(nih>7?w[8]:0)+nd*w[9]+ps[PS_PLAYED_KNIGHT]*w[10];
}

/* Per-seat weight assignment */
static const double *seat_weights[4];

double seat_value_fn(Game *g, Color c) {
    int seat = -1;
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    for (int i = 0; i < 4; i++) if (colors[i] == c) { seat = i; break; }
    return eval_w(g, c, seat_weights[seat]);
}

int main(void) {
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    int fine_wins = 0, orig_wins = 0;
    int fine_seats = 0, orig_seats = 0;
    clock_t t0 = clock();

    /* 6 seat combos for 2-of-4 */
    int combos[6][4] = {
        {1,1,0,0}, {1,0,1,0}, {1,0,0,1},
        {0,1,1,0}, {0,1,0,1}, {0,0,1,1}
    }; /* 1 = fine-tuned, 0 = original */

    /* Seeds 2M+ totally fresh */
    for (int gi = 0; gi < N_GAMES; gi++) {
        int *combo = combos[gi % 6];
        for (int i = 0; i < 4; i++) {
            seat_weights[i] = combo[i] ? FINE_W : ORIG_W;
            if (combo[i]) fine_seats++; else orig_seats++;
        }

        int seed = 2000000 + gi;
        CatanMap map;
        rng_seed((uint64_t)seed);
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 4, colors, (uint64_t)seed, 7, false, 10);
        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Action action;
            if (n == 1) { action = actions[0]; }
            else {
                Color cur = state_current_color(&game.state);
                double dl = (double)clock()/CLOCKS_PER_SEC + 120.0;
                Game cp; game_copy(&cp, &game);
                SearchResult sr = alphabeta_search(&cp, actions, n, AB_DEPTH,
                    -1e30, 1e30, dl, cur);
                action = (sr.action.type!=0||sr.action.color!=0) ? sr.action : actions[0];
            }
            game_execute(&game, action, actions, &n);
        }

        Color winner = game_winning_color(&game);
        if (winner != COLOR_NONE) {
            int ws = -1;
            for (int i = 0; i < 4; i++) if (colors[i] == winner) { ws = i; break; }
            if (combo[ws]) fine_wins++; else orig_wins++;
        }

        if ((gi+1) % 100 == 0) {
            double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
            double fwr = 100.0*fine_wins/fine_seats*4;
            double owr = 100.0*orig_wins/orig_seats*4;
            printf("  [%4d/%d] fine=%.1f%% orig=%.1f%%  (%d vs %d wins, %.0fs)\n",
                   gi+1, N_GAMES, fwr, owr, fine_wins, orig_wins, elapsed);
        }
    }

    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
    double fwr = 100.0*fine_wins/fine_seats*4;
    double owr = 100.0*orig_wins/orig_seats*4;

    printf("\n==========================================\n");
    printf("  Fine-tuned vs Original -- %d 4p games\n", N_GAMES);
    printf("  Seeds 2000000-%d (completely fresh)\n", 2000000+N_GAMES-1);
    printf("==========================================\n");
    printf("  Fine-tuned: %d wins / %d seats = %.2f%% per player\n",
           fine_wins, fine_seats, 100.0*fine_wins/fine_seats);
    printf("  Original:   %d wins / %d seats = %.2f%% per player\n",
           orig_wins, orig_seats, 100.0*orig_wins/orig_seats);
    printf("  Baseline: 25.00%%\n");
    printf("\n  Difference: %+.2f pp\n", 100.0*fine_wins/fine_seats - 100.0*orig_wins/orig_seats);
    printf("  Time: %.1fs (%.0f games/sec)\n", elapsed, N_GAMES/elapsed);

    return 0;
}
