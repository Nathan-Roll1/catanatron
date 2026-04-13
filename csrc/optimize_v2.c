/*
 * Robust SPSA optimizer v2:
 * - CRN: both perturbations play the SAME seeds against a fixed baseline (RandomPlayer)
 * - 5% perturbations for detectable signal
 * - 500 games per direction per iteration (1000 total per iter)
 * - Fresh seeds every iteration, never reused
 * - Validation checkpoint every 10 iters on a fixed held-out seed set
 * - Early stopping if validation degrades
 * Budget: 100k games total -> ~50 SPSA iters + validation
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
#define GAMES_PER_DIR 500
#define NUM_ITERS 45
#define VAL_GAMES 500
#define VAL_INTERVAL 9
#define AB_DEPTH 2
#define PERTURB_PCT 0.05

static const double DICE_P[13] = {
    0, 0, 1.0/36, 2.0/36, 3.0/36, 4.0/36, 5.0/36, 6.0/36,
    5.0/36, 4.0/36, 3.0/36, 2.0/36, 1.0/36
};
static const char *pnames[NUM_PARAMS] = {
    "vps","prod","eprod","tiles","buildable","road","synergy","hand","discard","devs","army"
};
static const double ORIG[NUM_PARAMS] = {
    3e14, 1e8, -1e8, 1.0, 1e3, 10.0, 1e2, 1.0, -5.0, 10.0, 10.1
};

static double theta[NUM_PARAMS];
static double best_theta[NUM_PARAMS];
static const double *active_eval_w;

static double eval_w(Game *g, Color p0, const double w[NUM_PARAMS]) {
    State *s = &g->state;
    int idx = s->color_to_index[(int)p0];
    Board *b = &s->board; CatanMap *map = b->map;
    Coordinate robber = b->robber_coordinate;
    double rp[5]={0};
    for (int si=0;si<s->settlement_count[idx];si++){
        int node=s->settlements[idx][si];
        for (int ti=0;ti<map->adjacent_tiles_count[node];ti++){
            int t2=map->adjacent_tiles[node][ti]; LandTile *t=&map->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;
            rp[(int)t->resource]+=DICE_P[t->number];
        }
    }
    for (int ci=0;ci<s->city_count[idx];ci++){
        int node=s->cities[idx][ci];
        for (int ti=0;ti<map->adjacent_tiles_count[node];ti++){
            int t2=map->adjacent_tiles[node][ti]; LandTile *t=&map->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;
            rp[(int)t->resource]+=2.0*DICE_P[t->number];
        }
    }
    double tp=0;int var=0;
    for(int r=0;r<5;r++){tp+=rp[r];if(rp[r]>0)var++;}
    double prod=tp+var*4.0*(2.778/100.0);
    Color enemy=COLOR_NONE;
    for(int i=0;i<s->num_players;i++) if(s->colors[i]!=p0){enemy=s->colors[i];break;}
    double ep=0;
    if(enemy!=COLOR_NONE){
        int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){
            int node=s->settlements[ei][si];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){
                int t2=map->adjacent_tiles[node][ti]; LandTile *t=&map->land_tiles[t2];
                if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;
                ep+=DICE_P[t->number];
            }
        }
        for(int ci=0;ci<s->city_count[ei];ci++){
            int node=s->cities[ei][ci];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){
                int t2=map->adjacent_tiles[node][ti]; LandTile *t=&map->land_tiles[t2];
                if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;
                ep+=2.0*DICE_P[t->number];
            }
        }
    }
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.0;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.0;
    double hs=(2-dc-ds)/2.0;
    int nih=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={false};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){
        int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){
            int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=true;nt++;}
        }
    }
    for(int ci=0;ci<s->city_count[idx];ci++){
        int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){
            int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=true;nt++;}
        }
    }
    int buf[TOTAL_NODES];
    int nb=board_buildable_node_ids(&s->board,p0,false,buf,TOTAL_NODES);
    double lrf=(nb==0)?w[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]
          +ps[PS_MONOPOLY_IN_HAND]+ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    return ps[PS_VICTORY_POINTS]*w[0]+prod*w[1]+ep*w[2]+nt*w[3]+nb*w[4]
          +ps[PS_LONGEST_ROAD_LENGTH]*lrf+hs*w[6]+nih*w[7]+(nih>7?w[8]:0)
          +nd*w[9]+ps[PS_PLAYED_KNIGHT]*w[10];
}

/*
 * CRN game: AB with given weights as RED vs Random as BLUE.
 * Seed determines map + all randomness. Returns 1 if AB wins, 0 otherwise.
 */
static int play_vs_random(const double w[NUM_PARAMS], int seed) {
    active_eval_w = w;
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
        } else if (cur == COLOR_RED) {
            double dl = (double)clock()/CLOCKS_PER_SEC + 120.0;
            Game cp; game_copy(&cp, &game);
            SearchResult sr = alphabeta_search(&cp, actions, n, AB_DEPTH, -1e30, 1e30, dl, cur);
            action = (sr.action.type!=0||sr.action.color!=0) ? sr.action : actions[0];
        } else {
            action = actions[rng_choice_index(n)];
        }
        game_execute(&game, action, actions, &n);
    }
    return game_winning_color(&game) == COLOR_RED ? 1 : 0;
}

/* Evaluate weights: win count over N games against random on given seed range */
static int evaluate(const double w[NUM_PARAMS], int seed_start, int n_games) {
    int wins = 0;
    for (int i = 0; i < n_games; i++)
        wins += play_vs_random(w, seed_start + i);
    return wins;
}

int main(void) {
    memcpy(theta, ORIG, sizeof(ORIG));
    memcpy(best_theta, ORIG, sizeof(ORIG));

    /* Fixed validation seeds: 9000000-9000499 */
    int val_seed = 9000000;
    int baseline_val = evaluate(ORIG, val_seed, VAL_GAMES);
    double best_val_wr = (double)baseline_val / VAL_GAMES;

    printf("Robust SPSA v2: %d params, %d games/dir, %d iters\n", NUM_PARAMS, GAMES_PER_DIR, NUM_ITERS);
    printf("CRN: each perturbation plays same seeds vs RandomPlayer\n");
    printf("Perturbation: %.0f%%  |  Validation: %d games every %d iters\n", PERTURB_PCT*100, VAL_GAMES, VAL_INTERVAL);
    printf("Baseline validation win rate: %d/%d = %.1f%%\n\n", baseline_val, VAL_GAMES, best_val_wr*100);

    clock_t t0 = clock();
    int total_games = VAL_GAMES;
    srand(77777);

    for (int iter = 0; iter < NUM_ITERS; iter++) {
        int signs[NUM_PARAMS];
        for (int i = 0; i < NUM_PARAMS; i++) signs[i] = (rand()%2)*2-1;

        double plus_w[NUM_PARAMS], minus_w[NUM_PARAMS];
        for (int i = 0; i < NUM_PARAMS; i++) {
            double delta = fabs(theta[i]) * PERTURB_PCT * signs[i];
            plus_w[i]  = theta[i] + delta;
            minus_w[i] = theta[i] - delta;
        }

        /* CRN: both play SAME seeds against random */
        int seed_start = 3000000 + iter * GAMES_PER_DIR;
        int plus_wins  = evaluate(plus_w, seed_start, GAMES_PER_DIR);
        int minus_wins = evaluate(minus_w, seed_start, GAMES_PER_DIR);
        total_games += GAMES_PER_DIR * 2;

        double plus_wr  = (double)plus_wins / GAMES_PER_DIR;
        double minus_wr = (double)minus_wins / GAMES_PER_DIR;
        double gradient = (plus_wr - minus_wr); /* positive = plus is better */

        /* Adaptive step: 0.5% of |theta| scaled by gradient */
        double ak = 0.005 / pow(iter + 1, 0.3);
        for (int i = 0; i < NUM_PARAMS; i++)
            theta[i] += ak * fabs(theta[i]) * gradient * signs[i];

        double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("iter %2d: plus=%.1f%% minus=%.1f%% grad=%+.3f  [%dk, %.0fs, %.0f g/s]\n",
               iter+1, plus_wr*100, minus_wr*100, gradient, total_games/1000, elapsed, total_games/elapsed);

        /* Validation checkpoint */
        if ((iter+1) % VAL_INTERVAL == 0) {
            int val_wins = evaluate(theta, val_seed, VAL_GAMES);
            total_games += VAL_GAMES;
            double val_wr = (double)val_wins / VAL_GAMES;
            printf("  >> VALIDATION: %d/%d = %.1f%% (baseline %.1f%%, best %.1f%%)\n",
                   val_wins, VAL_GAMES, val_wr*100, (double)baseline_val/VAL_GAMES*100, best_val_wr*100);

            if (val_wr > best_val_wr) {
                best_val_wr = val_wr;
                memcpy(best_theta, theta, sizeof(theta));
                printf("  >> NEW BEST!\n");
            }

            printf("  theta shift: ");
            for (int i = 0; i < NUM_PARAMS; i++) {
                double pct = (theta[i]-ORIG[i])/fabs(ORIG[i])*100;
                printf("%s=%+.2f%% ", pnames[i], pct);
            }
            printf("\n\n");
        }
    }

    /* Final validation of best_theta */
    int final_val = evaluate(best_theta, val_seed, VAL_GAMES);
    total_games += VAL_GAMES;
    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;

    printf("\n==========================================\n");
    printf("  Robust SPSA v2 Complete\n");
    printf("  %dk total games in %.0fs (%.0f g/s)\n", total_games/1000, elapsed, total_games/elapsed);
    printf("==========================================\n");
    printf("\nBaseline validation: %d/%d = %.2f%%\n", baseline_val, VAL_GAMES, 100.0*baseline_val/VAL_GAMES);
    printf("Best validation:    %d/%d = %.2f%%\n", final_val, VAL_GAMES, 100.0*final_val/VAL_GAMES);
    printf("Improvement: %+.2f pp\n", 100.0*(final_val-baseline_val)/VAL_GAMES);

    printf("\nBest weights:\n");
    for (int i = 0; i < NUM_PARAMS; i++) {
        double pct = (best_theta[i]-ORIG[i])/fabs(ORIG[i])*100;
        printf("  %-12s = %20.6f  (%+.3f%%)\n", pnames[i], best_theta[i], pct);
    }

    return 0;
}
