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
#define GAMES 999972
#define GPT (GAMES/NT)

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBN=4.*(2.778/100.);
static int g_var=-1, g_seat=-1;

static double eval_fn(Game *g, Color c){
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *m=b->map;Coordinate rob=b->robber_coordinate;
    double rp[5]={0};
    for(int si=0;si<s->settlement_count[idx];si++){int nd=s->settlements[idx][si];
        if(nd<0||nd>=NUM_NODES)continue;
        for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;
            if(!coord_eq(m->land_tile_coords[t2],rob))rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int nd=s->cities[idx][ci];
        if(nd<0||nd>=NUM_NODES)continue;
        for(int ti=0;ti<m->adjacent_tiles_count[nd];ti++){int t2=m->adjacent_tiles[nd][ti];
            if(t2<0||t2>=NUM_LAND_TILES)continue;LandTile *t=&m->land_tiles[t2];
            if(t->resource==RES_NONE||t->number==0)continue;
            if(!coord_eq(m->land_tile_coords[t2],rob))rp[(int)t->resource]+=2.*DP[t->number];}}
    double tp=0;int vr=0;for(int r=0;r<5;r++){tp+=rp[r];if(rp[r]>0)vr++;}
    double prod=tp+vr*VBN;
    Color enemy=COLOR_NONE;for(int i=0;i<s->num_players;i++)if(s->colors[i]!=c){enemy=s->colors[i];break;}
    double ep=0;
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
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
                if(coord_eq(m->land_tile_coords[t2],rob))continue;ep+=2.*DP[t->number];}}}
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    double syn=(2-dc-ds)/2.;int nih=wo+br+sh+wh+or2;
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
    int nb=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    double lrf=(nb==0)?10.:0.1;
    int nd2=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
           +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    int road=ps[PS_LONGEST_ROAD_LENGTH];
    double val=ps[PS_VICTORY_POINTS]*3e14+prod*1e8+ep*(-1e8)+nt+nb*1e3
              +road*lrf+syn*1e2+nih+(nih>7?-5.:0)+nd2*10.+ps[PS_PLAYED_KNIGHT]*10.1;
    /* road_to_port: value roads more when you have no port */
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if(seat==g_seat && g_var==1){
        bool ports[6];board_get_player_port_resources(b,c,ports);
        int any=0;for(int i=0;i<6;i++)if(ports[i])any=1;
        if(!any) val+=road*50;
    }
    return val;
}

typedef struct{int tid,n;uint64_t sb;int vw,bw;double diff;}WA;
static void run_half(WA *w,int uv){
    RngState rng;SearchCtx ctx;
    Color cs[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int *out=uv?&w->vw:&w->bw;*out=0;
    int sv=g_var;if(!uv)g_var=-1;
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
    g_seat=-1;if(!uv)g_var=sv;
}
static void *worker(void *a){WA *w=(WA*)a;run_half(w,1);run_half(w,0);
    w->diff=(double)w->vw/w->n-(double)w->bw/w->n;return NULL;}

int main(void){
    srand(time(NULL));
    RngState tmp;rng_init(&tmp,0);CatanMap tm;build_map(&tm,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&tmp);
    board_init_static_graph(&tm);
    g_var=1;
    printf("=== road_to_port: 1M games, CRN, paired CI ===\n\n");
    uint64_t seed=(uint64_t)time(NULL)*1000003ULL;
    pthread_t th[NT];WA wa[NT];
    for(int i=0;i<NT;i++){wa[i].tid=i;wa[i].n=GPT;wa[i].sb=seed;}
    struct timespec T0,T1;clock_gettime(CLOCK_MONOTONIC,&T0);
    for(int i=0;i<NT;i++)pthread_create(&th[i],NULL,worker,&wa[i]);
    for(int i=0;i<NT;i++)pthread_join(th[i],NULL);
    int tw=0,bw=0;double diffs[NT];
    for(int i=0;i<NT;i++){tw+=wa[i].vw;bw+=wa[i].bw;diffs[i]=wa[i].diff;}
    double md=0;for(int i=0;i<NT;i++)md+=diffs[i];md/=NT;
    double vd=0;for(int i=0;i<NT;i++)vd+=(diffs[i]-md)*(diffs[i]-md);vd/=(NT-1);
    double se=sqrt(vd/NT);double ci=2.110*se*100;
    double wr=25.+md*100;
    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
    printf("  Win rate:  %.3f%%\n",wr);
    printf("  Effect:    %+.3f pp\n",wr-25);
    printf("  95%% CI:    [%+.3f, %+.3f] pp\n",md*100-ci,md*100+ci);
    printf("  Significant: %s\n",fabs(md*100)>ci?"YES ***":"no");
    printf("\n  var_wins=%d base_wins=%d n=%d\n",tw,bw,GAMES);
    printf("  Per-thread diffs: ");
    for(int i=0;i<NT;i++)printf("%.2f%s",diffs[i]*100,i<NT-1?" ":"");
    printf("\n  %.0fs, %.0f g/s\n",el,GAMES*2/el);
    return 0;
}
