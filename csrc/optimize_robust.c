/*
 * Robust nonlinear feature search with multiple-testing correction.
 *
 * Strategy:
 * 1. Screen 15 candidates at 2000 games each (Bonferroni threshold: >27.2%)
 * 2. Top 3 survive to validation: 5000 games each on fresh seeds
 * 3. Survivor must beat 25% at p<0.01 (>26.4% at N=5000)
 *
 * New candidates include features the linear search can't capture:
 *  - Port-production synergy (2:1 port + matching resource production)
 *  - Opponent VP proximity (react more when opponents near 10)
 *  - Settlement count interaction with production
 *  - City ratio (cities / total buildings)
 *  - Resource concentration (Herfindahl index on production)
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define AB_DEPTH 2
#define SCREEN_GAMES 2000
#define VALIDATE_GAMES 5000
#define NUM_CANDIDATES 15
#define TOP_K 3

static const double DICE_P[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double W[11]={3e14,1e8,-1e8,1.0,1e3,10.0,1e2,1.0,-5.0,10.0,10.1};

static int active_seat = -1;
static int active_cand = -1;
static double active_weight = 0;

typedef struct {
    double prod, eprod, synergy, hand, buildable, tiles, vps, army, devs, road;
    double city_ratio;       /* cities / (cities+settlements) */
    double prod_hhi;         /* Herfindahl of production across 5 resources */
    double max_enemy_vps;    /* highest opponent VP count */
    double port_prod_match;  /* production of resources for which we have 2:1 port */
    int    settlements, cities;
} Features;

static Features extract(Game *g, Color p0) {
    Features f={0};
    State *s=&g->state; int idx=s->color_to_index[(int)p0];
    Board *b=&s->board; CatanMap *map=b->map; Coordinate robber=b->robber_coordinate;
    
    double rp[5]={0};
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=DICE_P[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=2.*DICE_P[t->number];}}
    double tp=0;int var=0;for(int r=0;r<5;r++){tp+=rp[r];if(rp[r]>0)var++;}
    f.prod=tp+var*4.*(2.778/100.);

    /* Production HHI */
    if(tp>0){double sum_sq=0;for(int r=0;r<5;r++){double s2=rp[r]/tp;sum_sq+=s2*s2;}f.prod_hhi=sum_sq;}
    else f.prod_hhi=1.0;

    /* Port-production match: sum production of resources where we have 2:1 port */
    bool ports[6]; board_get_player_port_resources(b,p0,ports);
    f.port_prod_match=0;
    for(int r=0;r<5;r++) if(ports[r]) f.port_prod_match+=rp[r];

    Color enemy=COLOR_NONE;
    for(int i=0;i<s->num_players;i++)if(s->colors[i]!=p0){enemy=s->colors[i];break;}
    f.eprod=0;
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;f.eprod+=DICE_P[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;f.eprod+=2.*DICE_P[t->number];}}}

    /* Max enemy VPs */
    f.max_enemy_vps=0;
    for(int i=0;i<s->num_players;i++){
        if(s->colors[i]==p0)continue;
        double v=s->player_state[i][PS_VICTORY_POINTS];
        if(v>f.max_enemy_vps) f.max_enemy_vps=v;
    }

    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    f.synergy=(2-dc-ds)/2.;f.hand=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    f.tiles=nt;
    int buf[TOTAL_NODES];f.buildable=board_buildable_node_ids(&s->board,p0,false,buf,TOTAL_NODES);
    f.road=ps[PS_LONGEST_ROAD_LENGTH];f.vps=ps[PS_VICTORY_POINTS];
    f.army=ps[PS_PLAYED_KNIGHT];
    f.devs=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    f.settlements=s->settlement_count[idx]; f.cities=s->city_count[idx];
    f.city_ratio=(f.settlements+f.cities>0)?(double)f.cities/(f.settlements+f.cities):0;
    return f;
}

static const char *cand_names[NUM_CANDIDATES]={
    "port_prod_match",      /* 0: bonus for having 2:1 port + matching production */
    "1/prod_hhi",           /* 1: reward diverse production (low HHI) */
    "max_enemy_vps*eprod",  /* 2: penalize enemy prod more when they're close to winning */
    "city_ratio*prod",      /* 3: cities amplify production value */
    "settlements*buildable",/* 4: expansion potential scales with existing network */
    "vps*army",             /* 5: army is more valuable with more VPs */
    "(10-vps)*buildable",   /* 6: expansion matters more when far from winning */
    "prod*prod",            /* 7: superlinear production */
    "army*army",            /* 8: superlinear army */
    "hand*synergy",         /* 9: resources more valuable when right mix */
    "port_match*synergy",   /* 10: port+matching resources+right cards */
    "enemy_vps_sq",         /* 11: penalize when opponent is close to winning */
    "prod*(10-max_evps)",   /* 12: production matters more when opponents are behind */
    "devs*army",            /* 13: dev cards + army interaction */
    "tiles*prod",           /* 14: production concentration on owned tiles */
};

static double compute_cand(Features *f, int id) {
    switch(id){
        case 0: return f->port_prod_match;
        case 1: return (f->prod_hhi>0)?(1.0/f->prod_hhi):0;
        case 2: return f->max_enemy_vps * f->eprod;
        case 3: return f->city_ratio * f->prod;
        case 4: return f->settlements * f->buildable;
        case 5: return f->vps * f->army;
        case 6: return (10.0 - f->vps) * f->buildable;
        case 7: return f->prod * f->prod;
        case 8: return f->army * f->army;
        case 9: return f->hand * f->synergy;
        case 10: return f->port_prod_match * f->synergy;
        case 11: return f->max_enemy_vps * f->max_enemy_vps;
        case 12: return f->prod * (10.0 - f->max_enemy_vps);
        case 13: return f->devs * f->army;
        case 14: return f->tiles * f->prod;
        default: return 0;
    }
}

double nl_value_fn(Game *g, Color c) {
    Features f=extract(g,c);
    int idx=g->state.color_to_index[(int)c]; int *ps=g->state.player_state[idx];
    double lrf=(f.buildable==0)?W[5]:0.1;
    double val=ps[PS_VICTORY_POINTS]*W[0]+f.prod*W[1]+f.eprod*W[2]+f.tiles*W[3]
              +f.buildable*W[4]+f.road*lrf+f.synergy*W[6]+f.hand*W[7]
              +(f.hand>7?W[8]:0)+f.devs*W[9]+f.army*W[10];
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if(seat==active_seat && active_cand>=0)
        val+=active_weight*compute_cand(&f,active_cand);
    return val;
}

static int run_tournament(int cand, double weight, int seed_base, int n_games) {
    active_cand=cand; active_weight=weight;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int wins=0;
    for(int gi=0;gi<n_games;gi++){
        active_seat=gi%4;
        int seed=seed_base+gi;
        CatanMap map;rng_seed((uint64_t)seed);
        build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action actions[MAX_ACTIONS];
        int n=generate_playable_actions(&game.state,actions,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action action;
            if(n==1){action=actions[0];}
            else{Color cur=state_current_color(&game.state);
                double dl=(double)clock()/CLOCKS_PER_SEC+120.;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,actions,n,AB_DEPTH,-1e30,1e30,dl,cur);
                action=(sr.action.type!=0||sr.action.color!=0)?sr.action:actions[0];}
            game_execute(&game,action,actions,&n);
        }
        Color winner=game_winning_color(&game);
        if(winner!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==winner){ws=i;break;}
            if(ws==active_seat)wins++;}
    }
    active_cand=-1;active_seat=-1;
    return wins;
}

int main(void) {
    printf("=== Robust Nonlinear Feature Search ===\n");
    printf("Phase 1: Screen %d candidates x 3 weights @ %d games each\n", NUM_CANDIDATES, SCREEN_GAMES);
    printf("Phase 2: Validate top %d @ %d games on fresh seeds\n", TOP_K, VALIDATE_GAMES);
    printf("Bonferroni threshold (15*3=45 tests, p<0.05): >27.2%%\n\n");

    clock_t t0=clock(); int total_games=0;

    typedef struct{int id;double w;double wr;int wins;}R;
    R screen[NUM_CANDIDATES];

    /* Screen: try 3 weight scales per candidate */
    double try_w[]={1e5, 1e6, 1e7};
    int n_try=3;

    for(int c=0;c<NUM_CANDIDATES;c++){
        int best_wins=0;double best_w=0;
        for(int wi=0;wi<n_try;wi++){
            /* Try both signs */
            for(int sign=0;sign<2;sign++){
                double w=(sign==0?1:-1)*try_w[wi];
                int seed_base=10000000+c*100000+wi*10000+sign*5000;
                int wins=run_tournament(c,w,seed_base,SCREEN_GAMES);
                total_games+=SCREEN_GAMES;
                if(wins>best_wins){best_wins=wins;best_w=w;}
            }
        }
        double wr=100.*best_wins/(double)SCREEN_GAMES;
        screen[c]=(R){c,best_w,wr,best_wins};
        double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
        char flag = wr>27.2?'*':' ';
        printf("  [%2d] %-24s best_w=%+.0e  %d/%d = %5.1f%% %c  [%dk, %.0fs]\n",
               c,cand_names[c],best_w,best_wins,SCREEN_GAMES,wr,flag,total_games/1000,elapsed);
    }

    /* Sort by win rate */
    for(int i=0;i<NUM_CANDIDATES-1;i++)for(int j=i+1;j<NUM_CANDIDATES;j++)
        if(screen[j].wr>screen[i].wr){R tmp=screen[i];screen[i]=screen[j];screen[j]=tmp;}

    printf("\n=== Phase 1 Results (sorted) ===\n");
    printf("  Baseline: 25.0%%  |  Bonferroni threshold: 27.2%%\n");
    for(int i=0;i<NUM_CANDIDATES;i++){
        R *r=&screen[i];
        printf("  %2d. %-24s %5.1f%%  w=%+.0e%s\n",
               i+1,cand_names[r->id],r->wr,r->w,r->wr>27.2?" <<<":"");
    }

    /* Phase 2: validate top K */
    printf("\n=== Phase 2: Validation (%d games, fresh seeds) ===\n", VALIDATE_GAMES);
    printf("  Required: >26.4%% (p<0.01 at N=%d)\n\n", VALIDATE_GAMES);

    for(int i=0;i<TOP_K&&i<NUM_CANDIDATES;i++){
        R *r=&screen[i];
        int seed_base=20000000+r->id*100000;
        int wins=run_tournament(r->id,r->w,seed_base,VALIDATE_GAMES);
        total_games+=VALIDATE_GAMES;
        double wr=100.*wins/(double)VALIDATE_GAMES;
        double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
        char *verdict=(wr>26.4)?"SIGNIFICANT":"not significant";
        printf("  %-24s %d/%d = %.2f%%  (%s)  [%dk, %.0fs]\n",
               cand_names[r->id],wins,VALIDATE_GAMES,wr,verdict,total_games/1000,elapsed);
    }

    double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("\n  Total: %dk games in %.0fs (%.0f g/s)\n",total_games/1000,elapsed,total_games/elapsed);
    return 0;
}
