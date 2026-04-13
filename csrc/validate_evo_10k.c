#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define AB_DEPTH 2
#define GAMES 10000
#define NUM_W 11

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double VBONUS=4.*(2.778/100.);

static const double ORIG[NUM_W]={3e14, 1e8, -1e8, 1., 1e3, 10., 1e2, 1., -5., 10., 10.1};
static const double EVO[NUM_W]={
    3e14,           /* vps: unchanged */
    9.62954e+07,    /* prod: -3.7% */
    -9.08e+07,      /* eprod: +9.2% (less penalty) */
    1.,             /* tiles: unchanged */
    949.,           /* buildable: -5.1% */
    10.8653,        /* road: +8.7% */
    100.9,          /* synergy: +0.9% */
    1.,             /* hand: unchanged */
    -5.,            /* discard: unchanged */
    10.4,           /* devs: +4.0% */
    10.1            /* army: unchanged */
};

static int active_variant[4]; /* seat -> 0=orig, 1=evo */

static double eval_with(Game *g, Color c, const double W[NUM_W]) {
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

static double mixed_eval(Game *g, Color c) {
    Color allc[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(allc[i]==c){seat=i;break;}
    return eval_with(g, c, active_variant[seat] ? EVO : ORIG);
}

int main(void) {
    /* 6 seat combos for placing 2 evo among 4 seats */
    int combos[6][4]={
        {1,1,0,0},{1,0,1,0},{1,0,0,1},
        {0,1,1,0},{0,1,0,1},{0,0,1,1}
    };
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int evo_wins=0, orig_wins=0, evo_seats=0, orig_seats=0;

    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);

    /* Seeds 900M+ -- completely fresh, never used in training */
    for(int gi=0;gi<GAMES;gi++){
        int *combo=combos[gi%6];
        for(int s=0;s<4;s++){
            active_variant[s]=combo[s];
            if(combo[s])evo_seats++;else orig_seats++;
        }
        int seed=900000000+gi;
        rng_seed((uint64_t)seed);
        CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        char rb[2600];
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                rng_save_state(rb);Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,AB_DEPTH,-1e30,1e30,9e99,cur,mixed_eval);
                rng_restore_state(rb);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==w){ws=i;break;}
            if(combo[ws])evo_wins++;else orig_wins++;}
        if((gi+1)%200==0){
            clock_gettime(CLOCK_MONOTONIC,&t1);
            double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            printf("  [%d/%d] evo=%.1f%% orig=%.1f%%  (%.0fs)\n",
                gi+1,GAMES,100.*evo_wins/evo_seats,100.*orig_wins/orig_seats,el);
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n==========================================\n");
    printf("  2x EVO vs 2x ORIG -- %d games (seeds 900M+)\n", GAMES);
    printf("==========================================\n");
    printf("  EVO params: prod=-3.7%%, eprod=+9.2%%, build=-5.1%%,\n");
    printf("              road=+8.7%%, devs=+4.0%%\n\n");
    printf("  EVO:  %d wins / %d seats = %.2f%%\n", evo_wins, evo_seats, 100.*evo_wins/evo_seats);
    printf("  ORIG: %d wins / %d seats = %.2f%%\n", orig_wins, orig_seats, 100.*orig_wins/orig_seats);
    printf("  Baseline: 25.00%%\n");
    printf("  Difference: %+.2f pp\n", 100.*evo_wins/evo_seats - 100.*orig_wins/orig_seats);
    printf("  Time: %.1fs (%.0f games/sec)\n", el, GAMES/el);

    return 0;
}
