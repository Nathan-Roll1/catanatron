/*
 * 1M games per variant, CRN, paired CI via per-thread aggregation.
 * CI: compute mean and variance of per-thread win-rate differences,
 * then SE = sd / sqrt(NT). This captures the true paired variance.
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

#define NT 18
#define AB_DEPTH 2
#define NV 4
#define GAMES_PER_VAR 999972  /* divisible by NT (55554 per thread) */
#define GPT (GAMES_PER_VAR / NT)

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBN=4.*(2.778/100.);
static const char *VNAMES[NV]={
    "threat_aware",
    "endgame_subtle",
    "robber_subtle",
    "threat+robber combo",
};

static int g_variant=-1, g_seat=-1;

typedef struct {
    double rp[5],prod,ep[4],ep_total,safe_prod;
    int evps[4],max_evps,n_en,vps,hand,buildable,tiles,road,army,devs;
    double synergy;
} F;

static F xf(Game *g, Color c) {
    F f={0};State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *m=b->map;Coordinate rob=b->robber_coordinate;
    for(int si=0;si<s->settlement_count[idx];si++){int nd=s->settlements[idx][si];
        if(nd<0||nd>=NUM_NODES)continue;
        for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;double p=DP[t->number];
            f.rp[(int)t->resource]+=p;if(!coord_eq(m->land_tile_coords[t2],rob))f.safe_prod+=p;}}
    for(int ci=0;ci<s->city_count[idx];ci++){int nd=s->cities[idx][ci];
        if(nd<0||nd>=NUM_NODES)continue;
        for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;double p=2.*DP[t->number];
            f.rp[(int)t->resource]+=p;if(!coord_eq(m->land_tile_coords[t2],rob))f.safe_prod+=p;}}
    double tp=0;int vr=0;for(int r=0;r<5;r++){tp+=f.rp[r];if(f.rp[r]>0)vr++;}
    f.prod=tp+vr*VBN;f.n_en=0;
    for(int i=0;i<s->num_players;i++){if(s->colors[i]==c)continue;int ei=i;double ep=0;
        for(int si=0;si<s->settlement_count[ei];si++){int nd=s->settlements[ei][si];
            if(nd<0||nd>=NUM_NODES)continue;
            for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
                if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
                if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(m->land_tile_coords[t2],rob))continue;ep+=DP[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int nd=s->cities[ei][ci];
            if(nd<0||nd>=NUM_NODES)continue;
            for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
                if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
                if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(m->land_tile_coords[t2],rob))continue;ep+=2.*DP[t->number];}}
        f.ep[f.n_en]=ep;f.ep_total+=ep;
        f.evps[f.n_en]=s->player_state[ei][PS_VICTORY_POINTS];
        if(f.evps[f.n_en]>f.max_evps)f.max_evps=f.evps[f.n_en];f.n_en++;}
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    f.synergy=(2-dc-ds)/2.;f.hand=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int nd=s->settlements[idx][si];
        if(nd<0||nd>=NUM_NODES)continue;for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){
            int t2=m->adjacent_tiles[nd][ti];if(t2>=0&&t2<NUM_LAND_TILES&&!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int nd=s->cities[idx][ci];
        if(nd<0||nd>=NUM_NODES)continue;for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){
            int t2=m->adjacent_tiles[nd][ti];if(t2>=0&&t2<NUM_LAND_TILES&&!ts[t2]){ts[t2]=1;nt++;}}}
    uint64_t reach[2]={0,0};
    for(int i=0;i<s->board.cc_count[(int)c];i++)bs_or(reach,reach,s->board.cc_sets[(int)c][i]);
    uint64_t avail[2];bs_and(avail,reach,s->board.buildable);
    f.buildable=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    f.road=s->player_state[idx][PS_LONGEST_ROAD_LENGTH];f.vps=s->player_state[idx][PS_VICTORY_POINTS];
    f.army=s->player_state[idx][PS_PLAYED_KNIGHT];
    f.devs=s->player_state[idx][PS_KNIGHT_IN_HAND]+s->player_state[idx][PS_YEAR_OF_PLENTY_IN_HAND]
          +s->player_state[idx][PS_MONOPOLY_IN_HAND]+s->player_state[idx][PS_ROAD_BUILDING_IN_HAND]
          +s->player_state[idx][PS_VICTORY_POINT_IN_HAND];
    return f;
}

static double base_sc(F *f){
    double lrf=(f->buildable==0)?10.:0.1;
    double ep0=f->n_en>0?f->ep[0]:0;
    return f->vps*3e14+f->prod*1e8+ep0*(-1e8)+f->tiles+f->buildable*1e3
          +f->road*lrf+f->synergy*1e2+f->hand+(f->hand>7?-5.:0)+f->devs*10.+f->army*10.1;
}

static double var_sc(F *f, int v){
    double b=base_sc(f);
    switch(v){
    case 0: for(int i=0;i<f->n_en;i++) b-=f->ep[i]*f->evps[i]*1e7; return b;
    case 1: if(f->vps>=7){b+=f->prod*5e6;b+=f->buildable*500;} return b;
    case 2: b+=f->safe_prod*2e6; return b;
    case 3: for(int i=0;i<f->n_en;i++) b-=f->ep[i]*f->evps[i]*1e7;
            b+=f->safe_prod*2e6; return b;
    }
    return b;
}

static double eval_fn(Game *g, Color c){
    F f=xf(g,c);
    Color cs[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(cs[i]==c){seat=i;break;}
    if(seat==g_seat&&g_variant>=0) return var_sc(&f,g_variant);
    return base_sc(&f);
}

typedef struct {
    int tid, n;
    uint64_t sb;
    int var_wins, base_wins, games;
    double var_wr, base_wr, diff; /* per-thread rates for CI */
} WA;

static void run_half(WA *w, int use_var) {
    RngState rng; SearchCtx ctx;
    Color cs[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int *out = use_var ? &w->var_wins : &w->base_wins;
    *out = 0;
    int sv = g_variant; if (!use_var) g_variant = -1;
    for (int gi = 0; gi < w->n; gi++) {
        g_seat = gi % 4;
        uint64_t seed = w->sb + (uint64_t)w->tid * 100000ULL + gi;
        rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, cs, seed, 7, false, 10);
        Action acts[MAX_ACTIONS]; int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        while (game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT) {
            Action a; if (n==1) a=acts[0];
            else { Color cur=state_current_color(&game.state); ctx.depth_counter=0;
                Game cp; game_copy(&cp, &game);
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,AB_DEPTH,-1e30,1e30,cur,eval_fn);
                a=(sr.action.type||sr.action.color)?sr.action:acts[0]; }
            game_execute(&game, a, acts, &n);
        }
        Color w2 = game_winning_color(&game);
        if (w2!=COLOR_NONE && game.state.color_to_index[(int)w2]==g_seat) (*out)++;
    }
    g_seat = -1;
    if (!use_var) g_variant = sv;
}

static void *worker(void *a) {
    WA *w = (WA*)a;
    run_half(w, 1);
    run_half(w, 0);
    w->games = w->n;
    w->var_wr = (double)w->var_wins / w->n;
    w->base_wr = (double)w->base_wins / w->n;
    w->diff = w->var_wr - w->base_wr;
    return NULL;
}

int main(void) {
    srand(time(NULL));
    RngState tmp; rng_init(&tmp, 0);
    CatanMap tm; build_map(&tm, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp);
    board_init_static_graph(&tm);

    printf("=== 1M-Game Structural Validation ===\n");
    printf("%d variants x %dk games each, CRN, %d threads\n", NV, GAMES_PER_VAR/1000, NT);
    printf("CI: per-thread paired difference, t-distribution with %d df\n\n", NT-1);

    struct timespec T0, T1;
    clock_gettime(CLOCK_MONOTONIC, &T0);
    int tg = 0;

    for (int v = 0; v < NV; v++) {
        g_variant = v;
        uint64_t seed = (uint64_t)time(NULL) * 1000003ULL + v * 7919ULL;

        pthread_t th[NT]; WA wa[NT];
        for (int i = 0; i < NT; i++) {
            wa[i].tid = i; wa[i].n = GPT; wa[i].sb = seed;
        }
        for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, worker, &wa[i]);
        for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);

        int tw = 0, bw = 0;
        double diffs[NT];
        for (int i = 0; i < NT; i++) {
            tw += wa[i].var_wins;
            bw += wa[i].base_wins;
            diffs[i] = wa[i].diff;
        }
        tg += GAMES_PER_VAR * 2;

        /* Paired CI: mean and SE of per-thread differences */
        double mean_diff = 0;
        for (int i = 0; i < NT; i++) mean_diff += diffs[i];
        mean_diff /= NT;

        double var_diff = 0;
        for (int i = 0; i < NT; i++) var_diff += (diffs[i] - mean_diff) * (diffs[i] - mean_diff);
        var_diff /= (NT - 1);

        double se = sqrt(var_diff / NT);
        double t_crit = 2.110; /* t(17, 0.025) for 95% CI */
        double ci95 = t_crit * se * 100; /* in percentage points */
        double wr = 25.0 + mean_diff * 100;

        clock_gettime(CLOCK_MONOTONIC, &T1);
        double el = (T1.tv_sec-T0.tv_sec) + (T1.tv_nsec-T0.tv_nsec) / 1e9;

        int sig = fabs(mean_diff * 100) > ci95;
        printf("  %-26s %6.3f%% (%+.3f pp) 95%% CI:[%+.3f, %+.3f] pp %s\n",
               VNAMES[v], wr, wr - 25,
               (mean_diff * 100) - ci95, (mean_diff * 100) + ci95,
               sig ? "*** SIGNIFICANT ***" : "");
        printf("    tw=%d bw=%d n=%d  per-thread diffs: [", tw, bw, GAMES_PER_VAR);
        for (int i = 0; i < NT; i++) printf("%.3f%s", diffs[i]*100, i<NT-1?",":"");
        printf("]\n");
        printf("    [%dk, %.0fs, %.0f g/s]\n\n", tg/1000, el, tg/el);
    }
    g_variant = -1;

    clock_gettime(CLOCK_MONOTONIC, &T1);
    double el = (T1.tv_sec-T0.tv_sec) + (T1.tv_nsec-T0.tv_nsec) / 1e9;
    printf("=== Total: %dM games in %.0fs (%.0f g/s) ===\n", tg/1000000, el, tg/el);
    return 0;
}
