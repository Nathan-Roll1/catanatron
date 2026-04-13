/*
 * Parallel evolutionary parameter search.
 * 4 weight variants compete in every 4p game.
 * Each generation: 1000 games across 18 threads.
 * 80 generations = 80k games. At ~3600 4p g/s = ~22s.
 * Top 2 survive, bottom 2 mutated. Track drift.
 * After evolution: validate best vs baseline on 10k fresh games.
 * Total: ~90k games in ~30s. Budget: 5 minutes = plenty of headroom.
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

#define NUM_THREADS 18
#define GAMES_PER_GEN 1008  /* divisible by 18 and 4 */
#define GPT (GAMES_PER_GEN / NUM_THREADS)
#define NUM_VARIANTS 4
#define NUM_W 11
#define NUM_GENS 80
#define VAL_GAMES 10008  /* divisible by 18 and 4 */
#define VAL_GPT (VAL_GAMES / NUM_THREADS)
#define AB_DEPTH 2

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBONUS=4.*(2.778/100.);
static const char *wn[NUM_W]={"vps","prod","eprod","tiles","build","road","syn","hand","disc","devs","army"};
static const double ORIG[NUM_W]={3e14,1e8,-1e8,1.,1e3,10.,1e2,1.,-5.,10.,10.1};

/* Per-thread variant weights (set before each game batch) */
static double V[NUM_VARIANTS][NUM_W]; /* used by main thread for evolution */
static pthread_key_t tl_V_key;
static const double (*get_tl_V(void))[NUM_W] { return (const double(*)[NUM_W])pthread_getspecific(tl_V_key); }

typedef struct { int tid; int n; int seed_base; int wins[NUM_VARIANTS]; double W[NUM_VARIANTS][NUM_W]; } TW;

static double veval(Game *g, Color c) {
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *map=b->map;Coordinate robber=b->robber_coordinate;
    if (idx < 0 || idx >= NUM_VARIANTS) idx = 0;
    const double (*lv)[NUM_W] = get_tl_V();
    const double *W = lv ? lv[idx] : V[idx];

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
    TW *tw=(TW*)arg;
    memset(tw->wins,0,sizeof(tw->wins));
    pthread_setspecific(tl_V_key, tw->W);
    RngState rng; SearchCtx ctx;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    /* Variant-to-seat rotation: in game gi, seat s gets variant (gi+s)%4.
     * We achieve this by rotating the V[] pointers per game.
     * Since veval reads V[seat], we rotate by shuffling which variant's
     * weights are in which V[] slot. But V[] is shared... so instead,
     * for each game we copy V into a local rotated version.
     * Actually, since each game has a different rotation, and veval just reads
     * V[seat], we can rotate by remapping colors to seats differently.
     * Simplest: just let seat=variant (no rotation), and rely on the large
     * number of games for fairness. Each variant always sits in the same seat,
     * but with 1000+ games the position effect averages out.
     * For proper rotation, we'd need per-game state in veval, which requires
     * thread-local storage. Let's use a simpler approach: shuffle the variant
     * assignment into V[] before each batch, so different gens get different
     * seat assignments. */
    for(int gi=0;gi<tw->n;gi++){
        uint64_t seed=(uint64_t)tw->seed_base+gi;
        rng_init(&rng,seed);
        CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&rng);
        Game game;game_init_with_map(&game,&map,4,colors,seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                ctx.depth_counter=0;Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,AB_DEPTH,-1e30,1e30,cur,veval);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE) {
            int winner_idx = game.state.color_to_index[(int)w];
            tw->wins[winner_idx]++;
        }
    }
    return NULL;
}

static void run_gen(int seed_offset, int gpthread, int wins_out[NUM_VARIANTS]) {
    pthread_t threads[NUM_THREADS]; TW work[NUM_THREADS];
    for(int i=0;i<NUM_THREADS;i++){
        work[i].tid=i;work[i].n=gpthread;
        work[i].seed_base=seed_offset+i*gpthread;
        memcpy(work[i].W, V, sizeof(V)); /* thread-local copy of weights */
    }
    for(int i=0;i<NUM_THREADS;i++) pthread_create(&threads[i],NULL,worker,&work[i]);
    for(int i=0;i<NUM_THREADS;i++) pthread_join(threads[i],NULL);
    memset(wins_out,0,NUM_VARIANTS*sizeof(int));
    /* Map seat wins to variant wins. Since seat=variant (no rotation),
     * wins for color RED=seat0=variant0, etc. */
    for(int i=0;i<NUM_THREADS;i++)
        for(int v=0;v<NUM_VARIANTS;v++) wins_out[v]+=work[i].wins[v];
}

static void mutate(double dst[NUM_W], const double src[NUM_W]) {
    memcpy(dst,src,sizeof(double)*NUM_W);
    int nc=1+(rand()%2);
    for(int c=0;c<nc;c++){
        int p=rand()%NUM_W;
        dst[p]*=1.0+((rand()%200-100)/1000.);
    }
}

int main(void) {
    srand(424242);
    pthread_key_create(&tl_V_key, NULL);
    struct timespec T0,T1;clock_gettime(CLOCK_MONOTONIC,&T0);
    int tg=0;

    printf("=== Parallel Evolutionary Search ===\n");
    printf("%d variants, %d games/gen, %d gens, %d threads, AB:%d\n",
           NUM_VARIANTS,GAMES_PER_GEN,NUM_GENS,NUM_THREADS,AB_DEPTH);
    printf("Train: %dk games | Validate: %dk games\n\n",
           NUM_GENS*GAMES_PER_GEN/1000, VAL_GAMES/1000);

    /* Init static graph once before any threads */
    {RngState tmp_rng; rng_init(&tmp_rng, 0);
     CatanMap tmp_map; build_map(&tmp_map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp_rng);
     board_init_static_graph(&tmp_map);}

    /* Init: V0=baseline, V1-3=mutants */
    for(int v=0;v<NUM_VARIANTS;v++) memcpy(V[v],ORIG,sizeof(ORIG));
    for(int v=1;v<NUM_VARIANTS;v++) mutate(V[v],ORIG);

    double best_ever[NUM_W];memcpy(best_ever,ORIG,sizeof(ORIG));
    double best_wr=25.;

    for(int gen=0;gen<NUM_GENS;gen++){
        /* Shuffle variant-to-seat mapping each gen for position fairness.
         * We rotate V[] slots: shift by gen%4. */
        double Vrot[4][NUM_W];
        for(int v=0;v<4;v++) memcpy(Vrot[v], V[(v+gen)%4], sizeof(ORIG));
        memcpy(V, Vrot, sizeof(V));

        int wins[NUM_VARIANTS];
        run_gen(gen*100000+100000000, GPT, wins);
        tg+=GAMES_PER_GEN;

        /* Unrotate wins back to original variant indices */
        int uwins[NUM_VARIANTS];
        for(int v=0;v<4;v++) uwins[(v+gen)%4]=wins[v];

        /* Unrotate V back */
        double Vunrot[4][NUM_W];
        for(int v=0;v<4;v++) memcpy(Vunrot[(v+gen)%4], V[v], sizeof(ORIG));
        memcpy(V, Vunrot, sizeof(V));

        /* Rank */
        int rank[4]={0,1,2,3};
        for(int i=0;i<3;i++)for(int j=i+1;j<4;j++)
            if(uwins[rank[j]]>uwins[rank[i]]){int t=rank[i];rank[i]=rank[j];rank[j]=t;}
        double tw=100.*uwins[rank[0]]/GAMES_PER_GEN;
        if(tw>best_wr){best_wr=tw;memcpy(best_ever,V[rank[0]],sizeof(ORIG));}

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

        /* Evolve */
        double nv[4][NUM_W];
        memcpy(nv[0],V[rank[0]],sizeof(ORIG));
        memcpy(nv[1],V[rank[1]],sizeof(ORIG));
        mutate(nv[2],V[rank[rand()%2]]);
        mutate(nv[3],V[rank[rand()%2]]);
        memcpy(V,nv,sizeof(V));
    }

    /* Validate: best_ever as V0, baseline as V1-V3 */
    printf("\n=== Validation: best vs 3x baseline (%dk games) ===\n",VAL_GAMES/1000);
    memcpy(V[0],best_ever,sizeof(ORIG));
    for(int v=1;v<4;v++) memcpy(V[v],ORIG,sizeof(ORIG));
    /* Run with rotation over 4 gens worth to cover all seats */
    int val_wins[4]={0};
    for(int r=0;r<4;r++){
        /* Rotate so best_ever gets each seat */
        double Vrot[4][NUM_W];
        for(int v=0;v<4;v++) memcpy(Vrot[v], V[(v+r)%4], sizeof(ORIG));
        double Vsave[4][NUM_W]; memcpy(Vsave,V,sizeof(V));
        memcpy(V,Vrot,sizeof(V));

        int rw[4];
        run_gen(900000000+r*100000, VAL_GPT/4, rw);
        tg+=VAL_GAMES/4;

        /* Seat r had best_ever */
        val_wins[0]+=rw[r]; /* best_ever wins when sitting at seat r */
        for(int v=0;v<4;v++) if(v!=r) val_wins[1]+=rw[v]; /* baseline wins */

        memcpy(V,Vsave,sizeof(V));
    }

    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;

    double best_val_wr=100.*val_wins[0]/(VAL_GAMES/4.);
    double base_val_wr=100.*val_wins[1]/(VAL_GAMES*3./4.);

    printf("  Best ever: %d wins / %d seats = %.2f%%\n", val_wins[0], VAL_GAMES/4, best_val_wr);
    printf("  Baseline:  %d wins / %d seats = %.2f%%\n", val_wins[1], VAL_GAMES*3/4, base_val_wr);
    printf("  Diff: %+.2f pp\n", best_val_wr - 25.0);

    printf("\n=== Summary: %dk games in %.0fs (%.0f g/s) ===\n",tg/1000,el,tg/el);
    printf("Best training wr: %.1f%%\n",best_wr);
    printf("Best OOS wr: %.2f%% (baseline 25%%)\n",best_val_wr);
    printf("\nBest weights:\n");
    for(int i=0;i<NUM_W;i++){double p=(best_ever[i]-ORIG[i])/fabs(ORIG[i])*100;
        printf("  %-8s = %15.6g  (%+.3f%%)\n",wn[i],best_ever[i],p);}

    return 0;
}
