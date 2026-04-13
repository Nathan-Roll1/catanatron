/*
 * Parallel evolutionary optimizer.
 * 4 weight vectors compete in every game (one per seat, rotating).
 * Each generation: 1000 games across 16 threads, rank by win rate.
 * Top 1 survives unchanged. Others mutated from top 2.
 * 100 generations = 100k games.
 * Track parameter drift across generations.
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

#define AB_DEPTH 2
#define GAMES_PER_GEN 1000
#define NUM_THREADS 16
#define GAMES_PER_THREAD (GAMES_PER_GEN / NUM_THREADS)
#define NUM_VARIANTS 4
#define NUM_W 11
#define NUM_GENS 100

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBONUS=4.*(2.778/100.);
static const char *wn[NUM_W]={"vps","prod","eprod","tiles","build","road","syn","hand","disc","devs","army"};
static const double ORIG[NUM_W]={3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};

static double variants[NUM_VARIANTS][NUM_W];
static __thread int game_seat_variant[4];

typedef struct {
    int n_games; int seed_base;
    int variant_wins[NUM_VARIANTS];
} ThreadWork;

static double variant_eval(Game *g, Color c) {
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *map=b->map;Coordinate robber=b->robber_coordinate;
    Color allc[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(allc[i]==c){seat=i;break;}
    const double *W=variants[game_seat_variant[seat]];
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
          +ps[PS_LONGEST_ROAD_LENGTH]*lrf+syn*W[6]+nih*W[7]+(nih>7?W[8]:0)+nd*W[9]+ps[PS_PLAYED_KNIGHT]*W[10];
}

static void *worker(void *arg) {
    ThreadWork *tw=(ThreadWork*)arg;
    memset(tw->variant_wins,0,sizeof(tw->variant_wins));
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    for(int gi=0;gi<tw->n_games;gi++){
        for(int s=0;s<4;s++) game_seat_variant[s]=(gi+s)%NUM_VARIANTS;
        int seed=tw->seed_base+gi;
        rng_seed((uint64_t)seed);
        CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        char rng_buf[2600];
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                rng_save_state(rng_buf);Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,AB_DEPTH,-1e30,1e30,9e99,cur,variant_eval);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==w){ws=i;break;}
            tw->variant_wins[game_seat_variant[ws]]++;}
    }
    return NULL;
}

static void mutate(double dst[NUM_W], const double src[NUM_W]) {
    memcpy(dst, src, sizeof(double)*NUM_W);
    /* Pick 1-2 random parameters to perturb */
    int n_changes = 1 + (rand()%2);
    for (int c=0; c<n_changes; c++) {
        int p = rand()%NUM_W;
        double factor = 1.0 + ((rand()%200 - 100) / 1000.0); /* +/- 10% */
        dst[p] *= factor;
    }
}

int main(void) {
    srand(31415);
    struct timespec T0,T1;
    clock_gettime(CLOCK_MONOTONIC,&T0);

    printf("=== Parallel Evolutionary Optimizer ===\n");
    printf("%d variants, %d games/gen, %d gens = %dk games, %d threads, AB:%d\n\n",
           NUM_VARIANTS,GAMES_PER_GEN,NUM_GENS,NUM_GENS*GAMES_PER_GEN/1000,NUM_THREADS,AB_DEPTH);

    /* Initialize: V0=baseline, V1-3=small random perturbations */
    for(int v=0;v<NUM_VARIANTS;v++) memcpy(variants[v],ORIG,sizeof(ORIG));
    for(int v=1;v<NUM_VARIANTS;v++) mutate(variants[v],ORIG);

    double best_ever[NUM_W]; memcpy(best_ever,ORIG,sizeof(ORIG));
    double best_ever_wr=25.0;

    /* Header */
    printf("gen  V0%%   V1%%   V2%%   V3%%   best%%  |");
    for(int i=0;i<NUM_W;i++) printf(" %s",wn[i]);
    printf("\n");

    for(int gen=0;gen<NUM_GENS;gen++){
        pthread_t threads[NUM_THREADS];
        ThreadWork work[NUM_THREADS];
        for(int i=0;i<NUM_THREADS;i++){
            work[i].n_games=GAMES_PER_THREAD;
            work[i].seed_base=gen*1000000+i*100000+500000000;
        }
        for(int i=0;i<NUM_THREADS;i++) pthread_create(&threads[i],NULL,worker,&work[i]);
        for(int i=0;i<NUM_THREADS;i++) pthread_join(threads[i],NULL);

        int wins[NUM_VARIANTS]={0};
        for(int i=0;i<NUM_THREADS;i++)for(int v=0;v<NUM_VARIANTS;v++)wins[v]+=work[i].variant_wins[v];

        /* Rank */
        int rank[NUM_VARIANTS]={0,1,2,3};
        for(int i=0;i<NUM_VARIANTS-1;i++)for(int j=i+1;j<NUM_VARIANTS;j++)
            if(wins[rank[j]]>wins[rank[i]]){int t=rank[i];rank[i]=rank[j];rank[j]=t;}

        double top_wr=100.*wins[rank[0]]/GAMES_PER_GEN;
        if(top_wr>best_ever_wr){best_ever_wr=top_wr;memcpy(best_ever,variants[rank[0]],sizeof(ORIG));}

        /* Print progress */
        if((gen+1)%5==0 || gen==0){
            printf("%3d  ",gen+1);
            for(int v=0;v<NUM_VARIANTS;v++) printf("%4.1f%% ",100.*wins[v]/GAMES_PER_GEN);
            printf("%4.1f  |",best_ever_wr);
            /* Print top variant's parameter deltas */
            for(int i=0;i<NUM_W;i++){
                double pct=(variants[rank[0]][i]-ORIG[i])/fabs(ORIG[i])*100;
                if(fabs(pct)>0.5) printf(" %+.1f",pct); else printf("    .");
            }
            printf("\n");
        }

        /* Evolve: rank[0] survives. rank[1] survives. rank[2-3] mutated from top 2 */
        double new_v[NUM_VARIANTS][NUM_W];
        memcpy(new_v[0], variants[rank[0]], sizeof(ORIG)); /* champion */
        memcpy(new_v[1], variants[rank[1]], sizeof(ORIG)); /* runner-up */
        mutate(new_v[2], variants[rank[rand()%2]]);        /* mutant of top */
        mutate(new_v[3], variants[rank[rand()%2]]);        /* mutant of top */
        memcpy(variants, new_v, sizeof(variants));
    }

    clock_gettime(CLOCK_MONOTONIC,&T1);
    double total_el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;

    printf("\n==========================================\n");
    printf("  %dk games in %.1fs (%.0f g/s)\n",NUM_GENS*GAMES_PER_GEN/1000,total_el,
           NUM_GENS*GAMES_PER_GEN/total_el);
    printf("==========================================\n");
    printf("\nBest ever (%.1f%%):\n",best_ever_wr);
    for(int i=0;i<NUM_W;i++){
        double pct=(best_ever[i]-ORIG[i])/fabs(ORIG[i])*100;
        printf("  %-8s = %15.6g  (%+.3f%%)\n",wn[i],best_ever[i],pct);
    }
    printf("\nCurrent champion:\n");
    for(int i=0;i<NUM_W;i++){
        double pct=(variants[0][i]-ORIG[i])/fabs(ORIG[i])*100;
        printf("  %-8s = %15.6g  (%+.3f%%)\n",wn[i],variants[0][i],pct);
    }

    return 0;
}
