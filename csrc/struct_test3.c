/*
 * Round 3: genuinely different strategic ideas, not weight tweaks.
 * 100k games each, CRN, paired CI.
 *
 * Ideas that change HOW the bot reasons, not how much it values things:
 * 0. wheat_ore_prio: extra bonus for wheat+ore production specifically (city path)
 * 1. port_leverage: value resources you can trade 2:1 more than others
 * 2. deny_spots: penalize when enemies have more buildable spots than you
 * 3. road_to_port: bonus for road length when you DON'T have a port yet
 * 4. city_over_settle: when you have 3+ settlements, prefer city over new settle
 * 5. knight_tempo: bonus for playing knights early (before largest army race)
 * 6. resource_balance: penalize having 0 production in any resource
 * 7. block_leader: extra penalty for the LEADING enemy's production only
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
#define NV 8
#define GAMES 100008
#define GPT (GAMES/NT)

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBN=4.*(2.778/100.);
static const char *VNAMES[NV]={
    "wheat_ore_prio",
    "port_leverage",
    "deny_spots",
    "road_to_port",
    "city_over_settle",
    "knight_tempo",
    "resource_balance",
    "block_leader",
};

static int g_variant=-1, g_seat=-1;

typedef struct {
    double rp[5],prod;
    double ep_per[4],ep_total;
    int evps[4],max_evps,leader_idx,n_en;
    int vps,hand,buildable,tiles,road,army,devs;
    int settlements,cities;
    double synergy;
    int has_port[6]; /* 0-4 resource ports, 5=3:1 */
    int enemy_buildable[4];
} F;

static F xf(Game *g, Color c) {
    F f={0};State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *m=b->map;Coordinate rob=b->robber_coordinate;
    for(int si=0;si<s->settlement_count[idx];si++){int nd=s->settlements[idx][si];
        if(nd<0||nd>=NUM_NODES)continue;
        for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;
            if(!coord_eq(m->land_tile_coords[t2],rob))f.rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int nd=s->cities[idx][ci];
        if(nd<0||nd>=NUM_NODES)continue;
        for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;
            if(!coord_eq(m->land_tile_coords[t2],rob))f.rp[(int)t->resource]+=2.*DP[t->number];}}
    double tp=0;int vr=0;for(int r=0;r<5;r++){tp+=f.rp[r];if(f.rp[r]>0)vr++;}
    f.prod=tp+vr*VBN;
    /* Ports */
    bool ports[6];board_get_player_port_resources(b,c,ports);
    for(int i=0;i<6;i++) f.has_port[i]=ports[i];
    /* Enemies */
    f.n_en=0;f.leader_idx=-1;
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
        f.ep_per[f.n_en]=ep;f.ep_total+=ep;
        int ev=s->player_state[ei][PS_VICTORY_POINTS];
        f.evps[f.n_en]=ev;
        if(ev>f.max_evps){f.max_evps=ev;f.leader_idx=f.n_en;}
        /* Enemy buildable */
        uint64_t er[2]={0,0};
        for(int j=0;j<s->board.cc_count[(int)s->colors[i]];j++)
            bs_or(er,er,s->board.cc_sets[(int)s->colors[i]][j]);
        uint64_t ea[2];bs_and(ea,er,s->board.buildable);
        f.enemy_buildable[f.n_en]=__builtin_popcountll(ea[0])+__builtin_popcountll(ea[1]);
        f.n_en++;
    }
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
    f.tiles=nt;
    uint64_t reach[2]={0,0};
    for(int i=0;i<s->board.cc_count[(int)c];i++)bs_or(reach,reach,s->board.cc_sets[(int)c][i]);
    uint64_t avail[2];bs_and(avail,reach,s->board.buildable);
    f.buildable=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    f.road=ps[PS_LONGEST_ROAD_LENGTH];f.vps=ps[PS_VICTORY_POINTS];f.army=ps[PS_PLAYED_KNIGHT];
    f.devs=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    f.settlements=s->settlement_count[idx];f.cities=s->city_count[idx];
    return f;
}

static double base_sc(F *f){
    double lrf=(f->buildable==0)?10.:0.1;
    double ep0=f->n_en>0?f->ep_per[0]:0;
    return f->vps*3e14+f->prod*1e8+ep0*(-1e8)+f->tiles+f->buildable*1e3
          +f->road*lrf+f->synergy*1e2+f->hand+(f->hand>7?-5.:0)+f->devs*10.+f->army*10.1;
}

static double var_sc(F *f, int v){
    double b=base_sc(f);
    switch(v){
    case 0: /* wheat_ore_prio: bonus for wheat+ore production (city materials) */
        b += (f->rp[3] + f->rp[4]) * 3e7; /* wheat(3) + ore(4) */
        return b;
    case 1: /* port_leverage: resources you have 2:1 port for are worth more */
        for(int r=0;r<5;r++)
            if(f->has_port[r]) b += f->rp[r] * 5e7;
        return b;
    case 2: /* deny_spots: penalize when enemies have more buildable than you */
        {int max_eb=0;for(int i=0;i<f->n_en;i++)if(f->enemy_buildable[i]>max_eb)max_eb=f->enemy_buildable[i];
         if(max_eb > f->buildable) b -= (max_eb - f->buildable) * 500;
        }
        return b;
    case 3: /* road_to_port: road value increases if you have no port */
        {int any_port=0;for(int i=0;i<6;i++)if(f->has_port[i])any_port=1;
         if(!any_port) b += f->road * 50;
        }
        return b;
    case 4: /* city_over_settle: when 3+ settlements, value cities more */
        if(f->settlements >= 3) b += f->cities * 5e7;
        return b;
    case 5: /* knight_tempo: bonus for knights played in early/mid game */
        if(f->vps < 7) b += f->army * 1e7;
        return b;
    case 6: /* resource_balance: penalty for having 0 production in any resource */
        {int zeros=0;for(int r=0;r<5;r++)if(f->rp[r]==0)zeros++;
         b -= zeros * 2e7;
        }
        return b;
    case 7: /* block_leader: extra penalty for leading enemy's production */
        if(f->leader_idx>=0 && f->max_evps > f->vps)
            b -= f->ep_per[f->leader_idx] * 5e7;
        return b;
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

typedef struct{int tid,n;uint64_t sb;int vw,bw;double diff;}WA;

static void run_half(WA *w,int use_var){
    RngState rng;SearchCtx ctx;
    Color cs[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int *out=use_var?&w->vw:&w->bw;*out=0;
    int sv=g_variant;if(!use_var)g_variant=-1;
    for(int gi=0;gi<w->n;gi++){
        g_seat=gi%4;uint64_t seed=w->sb+(uint64_t)w->tid*100000ULL+gi;
        rng_init(&rng,seed);CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&rng);
        Game game;game_init_with_map(&game,&map,4,cs,seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1)a=acts[0];
            else{Color cur=state_current_color(&game.state);ctx.depth_counter=0;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,AB_DEPTH,-1e30,1e30,cur,eval_fn);
                a=(sr.action.type||sr.action.color)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w2=game_winning_color(&game);
        if(w2!=COLOR_NONE&&game.state.color_to_index[(int)w2]==g_seat)(*out)++;
    }
    g_seat=-1;if(!use_var)g_variant=sv;
}

static void *worker(void *a){WA *w=(WA*)a;run_half(w,1);run_half(w,0);
    w->diff=(double)w->vw/w->n-(double)w->bw/w->n;return NULL;}

int main(void){
    srand(time(NULL));
    RngState tmp;rng_init(&tmp,0);CatanMap tm;build_map(&tm,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&tmp);
    board_init_static_graph(&tm);
    printf("=== Structural Tests Round 3: 100k games each, CRN, paired CI ===\n\n");
    struct timespec T0,T1;clock_gettime(CLOCK_MONOTONIC,&T0);int tg=0;

    typedef struct{int id;double wr,ci;}R;
    R results[NV];

    for(int v=0;v<NV;v++){
        g_variant=v;uint64_t seed=(uint64_t)time(NULL)*1000003ULL+v*131071ULL;
        pthread_t th[NT];WA wa[NT];
        for(int i=0;i<NT;i++){wa[i].tid=i;wa[i].n=GPT;wa[i].sb=seed;}
        for(int i=0;i<NT;i++)pthread_create(&th[i],NULL,worker,&wa[i]);
        for(int i=0;i<NT;i++)pthread_join(th[i],NULL);
        int tw=0,bw=0;double diffs[NT];
        for(int i=0;i<NT;i++){tw+=wa[i].vw;bw+=wa[i].bw;diffs[i]=wa[i].diff;}
        tg+=GAMES*2;
        double md=0;for(int i=0;i<NT;i++)md+=diffs[i];md/=NT;
        double vd=0;for(int i=0;i<NT;i++)vd+=(diffs[i]-md)*(diffs[i]-md);vd/=(NT-1);
        double se=sqrt(vd/NT);double ci=2.110*se*100;
        double wr=25.+md*100;
        results[v]=(R){v,wr,ci};
        clock_gettime(CLOCK_MONOTONIC,&T1);
        double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
        int sig=fabs(md*100)>ci;
        printf("  %-22s %6.2f%% (%+.2f pp) CI:[%+.2f,%+.2f]%s  [%dk,%.0fs,%.0f g/s]\n",
               VNAMES[v],wr,wr-25,md*100-ci,md*100+ci,sig?" ***":"",tg/1000,el,tg/el);
    }
    g_variant=-1;
    /* Sort */
    for(int i=0;i<NV-1;i++)for(int j=i+1;j<NV;j++)if(results[j].wr>results[i].wr){R t=results[i];results[i]=results[j];results[j]=t;}
    printf("\n=== Ranked ===\n");
    for(int i=0;i<NV;i++){R *r=&results[i];
        printf("  %d. %-22s %+.2f pp  CI:[%+.2f,%+.2f]%s\n",i+1,VNAMES[r->id],r->wr-25,
               (r->wr-25)-r->ci,(r->wr-25)+r->ci,fabs(r->wr-25)>r->ci?" ***":"");}
    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
    printf("\nTotal: %dM games in %.0fs (%.0f g/s)\n",tg/1000000,el,tg/el);
    return 0;
}
