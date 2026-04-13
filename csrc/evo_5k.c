/*
 * Parallel evolutionary optimizer. Clean, no TLS, no global mutable state.
 * 18 threads, 4 variants compete per game, 80 gens x 1008 games = 80k train + 10k validate.
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
#include "value.h"

#define NT 1
#define GAMES_PER_GEN 1000
#define GPT (GAMES_PER_GEN/NT)
#define NUM_V 4
#define NUM_W 11
#define NUM_GENS 5
#define AB_DEPTH 2

static const double ORIG[NUM_W]={3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};
static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VB=4.*(2.778/100.);
static const char *wn[NUM_W]={"vps","prod","eprod","tiles","build","road","syn","hand","disc","devs","army"};

typedef struct {
    int tid, n; int seed_base;
    double W[NUM_V][NUM_W]; /* thread-local copy of weights */
    int wins[NUM_V];
} TW;

static double eval_w(Game *g, Color c, const double *W) {
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *map=b->map;Coordinate robber=b->robber_coordinate;
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
    double prod=tp+var*VB;
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

static double variant_eval(Game *g, Color c) {
    int idx = g->state.color_to_index[(int)c];
    if (idx < 0 || idx >= NUM_V || !g->eval_ctx) return base_value_fn(g, c);
    const double (*W)[NUM_W] = (const double(*)[NUM_W])g->eval_ctx;
    return eval_w(g, c, W[idx]);
}

static void *worker(void *arg) {
    TW *tw=(TW*)arg;
    memset(tw->wins,0,sizeof(tw->wins));
    RngState rng; SearchCtx ctx;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    for(int gi=0;gi<tw->n;gi++){
        uint64_t seed=(uint64_t)tw->seed_base+gi;
        rng_init(&rng,seed);CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&rng);
        Game game;game_init_with_map(&game,&map,4,colors,seed,7,false,10);
        game.eval_ctx = tw->W; /* point to thread-local weights */
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                ctx.depth_counter=0;Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,AB_DEPTH,-1e30,1e30,cur,variant_eval);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE){int wi=game.state.color_to_index[(int)w]; tw->wins[wi]++;}
    }
    return NULL;
}

static void mutate(double dst[NUM_W], const double src[NUM_W]) {
    memcpy(dst,src,sizeof(double)*NUM_W);
    int p=rand()%NUM_W;
    dst[p]*=1.0+((rand()%200-100)/1000.);
}

int main(void) {
    srand(31415);
    struct timespec T0,T1;clock_gettime(CLOCK_MONOTONIC,&T0);

    /* Pre-init static graph */
    RngState tmp;rng_init(&tmp,0);CatanMap tmp_map;
    build_map(&tmp_map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&tmp);
    board_init_static_graph(&tmp_map);

    printf("=== Parallel Evo: %d gens x %d games, %d threads, AB:%d ===\n\n",
           NUM_GENS,GAMES_PER_GEN,NT,AB_DEPTH);

    double V[NUM_V][NUM_W];
    for(int v=0;v<NUM_V;v++) memcpy(V[v],ORIG,sizeof(ORIG));
    for(int v=1;v<NUM_V;v++) mutate(V[v],ORIG);
    double best_ever[NUM_W];memcpy(best_ever,ORIG,sizeof(ORIG));double best_wr=25.;
    int tg=0;

    for(int gen=0;gen<NUM_GENS;gen++){
        /* Rotate variants across player indices each gen for position fairness */
        double Vrot[NUM_V][NUM_W];
        for(int v=0;v<NUM_V;v++) memcpy(Vrot[v],V[(v+gen)%NUM_V],sizeof(ORIG));

        pthread_t threads[NT]; TW work[NT];
        for(int i=0;i<NT;i++){
            work[i].tid=i;work[i].n=GPT;work[i].seed_base=gen*1000000+i*GPT+100000000;
            memcpy(work[i].W,Vrot,sizeof(Vrot));
        }
        for(int i=0;i<NT;i++) pthread_create(&threads[i],NULL,worker,&work[i]);
        for(int i=0;i<NT;i++) pthread_join(threads[i],NULL);
        tg+=GAMES_PER_GEN;

        /* Aggregate wins and unrotate */
        int uwins[NUM_V]={0};
        for(int i=0;i<NT;i++) for(int v=0;v<NUM_V;v++) uwins[(v+gen)%NUM_V]+=work[i].wins[v];

        int rank[4]={0,1,2,3};
        for(int i=0;i<3;i++)for(int j=i+1;j<4;j++)
            if(uwins[rank[j]]>uwins[rank[i]]){int t=rank[i];rank[i]=rank[j];rank[j]=t;}
        double tw2=100.*uwins[rank[0]]/GAMES_PER_GEN;
        if(tw2>best_wr){best_wr=tw2;memcpy(best_ever,V[rank[0]],sizeof(ORIG));}

        if((gen+1)%10==0){
            clock_gettime(CLOCK_MONOTONIC,&T1);
            double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
            printf("gen %2d: ",gen+1);
            for(int v=0;v<4;v++) printf("V%d=%4.1f%% ",v,100.*uwins[v]/GAMES_PER_GEN);
            printf("best=%.1f%% [%dk, %.0fs, %.0f g/s] | ",best_wr,tg/1000,el,tg/el);
            for(int i=0;i<NUM_W;i++){double p=(V[rank[0]][i]-ORIG[i])/fabs(ORIG[i])*100;
                if(fabs(p)>0.5)printf("%s%+.1f ",wn[i],p);}
            printf("\n");
        }

        double nv[4][NUM_W];
        memcpy(nv[0],V[rank[0]],sizeof(ORIG));
        memcpy(nv[1],V[rank[1]],sizeof(ORIG));
        mutate(nv[2],V[rank[rand()%2]]);
        mutate(nv[3],V[rank[rand()%2]]);
        memcpy(V,nv,sizeof(V));
    }

    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
    printf("\n=== %dk games in %.0fs (%.0f g/s) ===\n",tg/1000,el,tg/el);
    printf("Best (%.1f%%):\n",best_wr);
    for(int i=0;i<NUM_W;i++){double p=(best_ever[i]-ORIG[i])/fabs(ORIG[i])*100;
        printf("  %-8s = %15.6g  (%+.3f%%)\n",wn[i],best_ever[i],p);}
    return 0;
}
