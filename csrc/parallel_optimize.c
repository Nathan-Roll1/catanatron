/*
 * Parallel parameter optimization: 4 parameter variants compete in each game.
 * Each game has 4 seats, each with a different weight vector.
 * Run 16 games in parallel (16 threads), each game is independent.
 * After all games, rank variants by win rate, keep top, mutate rest.
 *
 * Architecture:
 * - 4 "contestants" with different weight vectors
 * - Each game: all 4 contestants play, rotating seats across games
 * - 1000 games = 250 games per seat assignment = solid stats
 * - Each contestant's eval is selected by seat -> weight vector mapping
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define AB_DEPTH 3
#define TOTAL_GAMES 1000
#define NUM_THREADS 16
#define GAMES_PER_THREAD (TOTAL_GAMES / NUM_THREADS)
#define NUM_VARIANTS 4
#define NUM_W 11

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBONUS=4.*(2.778/100.);

/* 4 weight vectors competing */
static double variants[NUM_VARIANTS][NUM_W];

static const char *wnames[NUM_W]={"vps","prod","eprod","tiles","buildable",
    "road","synergy","hand","discard","devs","army"};

/* Thread-local: which variant is in which seat for the current game */
typedef struct {
    int thread_id;
    int n_games;
    int seed_base;
    int seat_to_variant[4]; /* not used -- we rotate per game */
    int variant_wins[NUM_VARIANTS];
} ThreadWork;

/* Per-thread RNG */
static __thread uint32_t tl_mt[624];
static __thread int tl_mti = 625;

static void tl_seed(uint64_t s) {
    uint32_t key[1]={(uint32_t)(s&0xffffffff)};
    tl_mt[0]=19650218UL&0xffffffffUL;
    for(tl_mti=1;tl_mti<624;tl_mti++){tl_mt[tl_mti]=(1812433253UL*(tl_mt[tl_mti-1]^(tl_mt[tl_mti-1]>>30))+tl_mti)&0xffffffffUL;}
    int i=1,j=0,k=624;
    for(;k;k--){tl_mt[i]=(tl_mt[i]^((tl_mt[i-1]^(tl_mt[i-1]>>30))*1664525UL))+key[j]+j;tl_mt[i]&=0xffffffffUL;i++;j++;if(i>=624){tl_mt[0]=tl_mt[623];i=1;}if(j>=1)j=0;}
    for(k=623;k;k--){tl_mt[i]=(tl_mt[i]^((tl_mt[i-1]^(tl_mt[i-1]>>30))*1566083941UL))-i;tl_mt[i]&=0xffffffffUL;i++;if(i>=624){tl_mt[0]=tl_mt[623];i=1;}}
    tl_mt[0]=0x80000000UL;
}

/* Value function that reads weights from the variant assigned to this color's seat */
static __thread int game_seat_variant[4]; /* seat -> variant index */

static double variant_eval(Game *g, Color c) {
    State *s=&g->state; int idx=s->color_to_index[(int)c];
    Board *b=&s->board; CatanMap *map=b->map; Coordinate robber=b->robber_coordinate;

    /* Determine which variant this color uses */
    Color allc[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(allc[i]==c){seat=i;break;}
    const double *W = variants[game_seat_variant[seat]];

    double rp[5]={0};
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=2.*DP[t->number];}}
    double tp=0;int var=0;for(int r=0;r<5;r++){tp+=rp[r];if(rp[r]>0)var++;}
    double prod=tp+var*VBONUS;

    Color enemy=COLOR_NONE;for(int i=0;i<s->num_players;i++)if(s->colors[i]!=c){enemy=s->colors[i];break;}
    double ep=0;
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=DP[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=2.*DP[t->number];}}}

    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    double syn=(2-dc-ds)/2.;int nih=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    uint64_t reach[2]={0,0};
    for(int i=0;i<s->board.cc_count[(int)c];i++)bs_or(reach,reach,s->board.cc_sets[(int)c][i]);
    uint64_t avail[2];bs_and(avail,reach,s->board.buildable);
    int nb=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    double lrf=(nb==0)?W[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];

    return ps[PS_VICTORY_POINTS]*W[0]+prod*W[1]+ep*W[2]+nt*W[3]+nb*W[4]
          +ps[PS_LONGEST_ROAD_LENGTH]*lrf+syn*W[6]+nih*W[7]+(nih>7?W[8]:0)
          +nd*W[9]+ps[PS_PLAYED_KNIGHT]*W[10];
}

static void *worker(void *arg) {
    ThreadWork *tw = (ThreadWork*)arg;
    memset(tw->variant_wins, 0, sizeof(tw->variant_wins));

    for (int gi = 0; gi < tw->n_games; gi++) {
        /* Rotate seat assignments: game gi assigns variant (gi+seat)%4 to each seat */
        for (int s = 0; s < 4; s++)
            game_seat_variant[s] = (gi + s) % NUM_VARIANTS;

        int seed = tw->seed_base + gi;
        rng_seed((uint64_t)seed);
        CatanMap map;
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
        Game game;
        game_init_with_map(&game, &map, 4, colors, (uint64_t)seed, 7, false, 10);

        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        char rng_buf[2600];

        while (game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT) {
            Action a;
            if (n==1) { a=acts[0]; }
            else {
                Color cur=state_current_color(&game.state);
                rng_save_state(rng_buf);
                Game cp; game_copy(&cp, &game);
                SearchResult sr=alphabeta_search(&cp,acts,n,AB_DEPTH,-1e30,1e30,9e99,cur,variant_eval);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];
            }
            game_execute(&game,a,acts,&n);
        }

        Color w=game_winning_color(&game);
        if (w!=COLOR_NONE) {
            int ws=-1;for(int i=0;i<4;i++)if(colors[i]==w){ws=i;break;}
            int winner_variant = game_seat_variant[ws];
            tw->variant_wins[winner_variant]++;
        }
    }
    return NULL;
}

int main(void) {
    srand(7777);

    /* Variant 0: baseline */
    double baseline[NUM_W]={3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};
    memcpy(variants[0], baseline, sizeof(baseline));

    /* Variants 1-3: perturb one parameter each by +10% */
    memcpy(variants[1], baseline, sizeof(baseline));
    variants[1][1] *= 1.1;  /* prod +10% */

    memcpy(variants[2], baseline, sizeof(baseline));
    variants[2][2] *= 1.1;  /* eprod +10% (less negative) */

    memcpy(variants[3], baseline, sizeof(baseline));
    variants[3][10] *= 1.5;  /* army +50% */

    printf("=== Parallel 4-way Tournament ===\n");
    printf("4 variants compete in each game, 1000 games, %d threads\n", NUM_THREADS);
    printf("AB:%d, rotating seats\n\n", AB_DEPTH);
    printf("Variants:\n");
    for (int v=0;v<NUM_VARIANTS;v++){
        printf("  V%d: ", v);
        for(int i=0;i<NUM_W;i++){
            if(variants[v][i]!=baseline[i])
                printf("%s=%.4g(%+.0f%%) ",wnames[i],variants[v][i],
                    (variants[v][i]-baseline[i])/fabs(baseline[i])*100);
        }
        if(v==0) printf("(baseline)");
        printf("\n");
    }
    printf("\n");

    /* Pre-init maps (must be sequential for global RNG) */
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);

    pthread_t threads[NUM_THREADS];
    ThreadWork work[NUM_THREADS];

    for(int i=0;i<NUM_THREADS;i++){
        work[i].thread_id=i;
        work[i].n_games=GAMES_PER_THREAD;
        work[i].seed_base=700000000+i*100000;
    }

    for(int i=0;i<NUM_THREADS;i++)
        pthread_create(&threads[i],NULL,worker,&work[i]);
    for(int i=0;i<NUM_THREADS;i++)
        pthread_join(threads[i],NULL);

    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    /* Aggregate wins */
    int total_wins[NUM_VARIANTS]={0};
    for(int i=0;i<NUM_THREADS;i++)
        for(int v=0;v<NUM_VARIANTS;v++)
            total_wins[v]+=work[i].variant_wins[v];

    printf("==========================================\n");
    printf("  Results: %d games, %.1fs (%.0f g/s)\n", TOTAL_GAMES, el, TOTAL_GAMES/el);
    printf("==========================================\n");
    for(int v=0;v<NUM_VARIANTS;v++){
        double wr=100.*total_wins[v]/TOTAL_GAMES;
        printf("  V%d: %3d wins = %5.1f%%%s\n", v, total_wins[v], wr,
               wr>27?" <<<":"");
    }
    printf("  Baseline (random): 25.0%%\n");

    return 0;
}
