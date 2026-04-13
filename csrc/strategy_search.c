/*
 * Strategy optimization: test structural changes to the value function
 * and search algorithm. Each variant gets 1 seat in a 4-player AB:2 game
 * against 3 baseline seats. Baseline = 25%.
 *
 * Test ideas that change HOW the bot plays, not just weights:
 * 1. Consider ALL enemies, not just the first
 * 2. Scale enemy production penalty by their VP count
 * 3. Bonus for ore+wheat production (city materials)
 * 4. Penalty for resource concentration (vulnerable to robber)  
 * 5. Bonus for having 3+ resource types producing
 * 6. Value cities > settlements more aggressively
 * 7. Penalize long road without enough settlements
 * 8. Scale hand value by game phase (early vs late)
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
#define GAMES 1000

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double W_VPS=3e14,W_PROD=1e8,W_EPROD=-1e8,W_TILES=1.,W_BUILDABLE=1e3,
    W_ROAD=10.,W_SYNERGY=1e2,W_HAND=1.,W_DISCARD=-5.,W_DEVS=10.,W_ARMY=10.1;
static const double VBONUS=4.*(2.778/100.);

static int active_seat=-1;
static int active_variant=-1;

typedef struct {
    double rp[5]; double prod; double eprod_total; double eprod_per[4];
    int vps; int enemy_vps[4]; int max_evps; int num_enemies;
    double synergy; int hand; int buildable; int tiles; int road;
    int army; int devs; int settlements; int cities; int idx;
} F;

static F extract(Game *g, Color p0) {
    F f={0}; State *s=&g->state; f.idx=s->color_to_index[(int)p0];
    Board *b=&s->board; CatanMap *map=b->map; Coordinate robber=b->robber_coordinate;
    for(int si=0;si<s->settlement_count[f.idx];si++){int node=s->settlements[f.idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;f.rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[f.idx];ci++){int node=s->cities[f.idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;f.rp[(int)t->resource]+=2.*DP[t->number];}}
    double tp=0;int var=0;for(int r=0;r<5;r++){tp+=f.rp[r];if(f.rp[r]>0)var++;}
    f.prod=tp+var*VBONUS;
    
    f.num_enemies=0;
    for(int i=0;i<s->num_players;i++){if(s->colors[i]==p0)continue;
        int ei=i; double ep=0;
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=DP[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=2.*DP[t->number];}}
        f.eprod_per[f.num_enemies]=ep; f.eprod_total+=ep;
        f.enemy_vps[f.num_enemies]=s->player_state[ei][PS_VICTORY_POINTS];
        if(f.enemy_vps[f.num_enemies]>f.max_evps)f.max_evps=f.enemy_vps[f.num_enemies];
        f.num_enemies++;
    }

    int *ps=s->player_state[f.idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    f.synergy=(2-dc-ds)/2.; f.hand=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[f.idx];si++){int node=s->settlements[f.idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[f.idx];ci++){int node=s->cities[f.idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    f.tiles=nt;
    uint64_t reach[2]={0,0};
    for(int i=0;i<s->board.cc_count[(int)p0];i++)bs_or(reach,reach,s->board.cc_sets[(int)p0][i]);
    uint64_t avail[2];bs_and(avail,reach,s->board.buildable);
    f.buildable=__builtin_popcountll(avail[0])+__builtin_popcountll(avail[1]);
    f.road=ps[PS_LONGEST_ROAD_LENGTH]; f.vps=ps[PS_VICTORY_POINTS];
    f.army=ps[PS_PLAYED_KNIGHT];
    f.devs=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    f.settlements=s->settlement_count[f.idx]; f.cities=s->city_count[f.idx];
    return f;
}

static double baseline_eval(F *f) {
    double lrf=(f->buildable==0)?W_ROAD:0.1;
    /* Baseline uses only first enemy */
    double ep=f->num_enemies>0?f->eprod_per[0]:0;
    return f->vps*W_VPS+f->prod*W_PROD+ep*W_EPROD+f->tiles*W_TILES+f->buildable*W_BUILDABLE
          +f->road*lrf+f->synergy*W_SYNERGY+f->hand*W_HAND+(f->hand>7?W_DISCARD:0)
          +f->devs*W_DEVS+f->army*W_ARMY;
}

static double variant_eval(F *f, int variant) {
    double base = baseline_eval(f);
    double extra = 0;

    switch(variant) {
    case 0: /* All enemies: penalize avg enemy production, not just first */
        if(f->num_enemies>0){
            double avg_ep = f->eprod_total / f->num_enemies;
            double first_ep = f->eprod_per[0];
            extra = (avg_ep - first_ep) * W_EPROD; /* add penalty for other enemies */
        }
        break;
    case 1: /* Scale enemy penalty by their VPs -- worry more about leaders */
        for(int i=0;i<f->num_enemies;i++)
            extra += f->eprod_per[i] * f->enemy_vps[i] * (-1e7);
        break;
    case 2: /* Ore+wheat production bonus (city materials) */
        extra = (f->rp[3] + f->rp[4]) * 5e7; /* wheat + ore */
        break;
    case 3: /* Penalty for production concentration (robber vulnerability) */
        if(f->prod>0){double max_r=0;for(int r=0;r<5;r++)if(f->rp[r]>max_r)max_r=f->rp[r];
            extra = -(max_r/f->prod) * 2e7; /* higher concentration = worse */}
        break;
    case 4: /* Bonus for 4+ resource types producing (trade flexibility) */
        {int types=0;for(int r=0;r<5;r++)if(f->rp[r]>0)types++;
         if(types>=4) extra=5e7; if(types>=5) extra=1e8;}
        break;
    case 5: /* Value city count with increasing returns */
        extra = f->cities * f->cities * 5e7;
        break;
    case 6: /* Penalize road without buildable nodes (wasted roads) */
        if(f->road>5 && f->buildable==0) extra=-5e7;
        break;
    case 7: /* Early game: value production more. Late game: value VPs more */
        {double phase=(double)f->vps/10.; /* 0=early, 1=winning */
         extra = f->prod * (1.-phase) * 5e7 + f->vps * phase * 1e14;}
        break;
    case 8: /* Combo: all enemies + city bonus + ore/wheat */
        if(f->num_enemies>0){double avg_ep=f->eprod_total/f->num_enemies;
            extra=(avg_ep-f->eprod_per[0])*W_EPROD;}
        extra += f->cities*f->cities*3e7 + (f->rp[3]+f->rp[4])*3e7;
        break;
    case 9: /* Aggressive: extra penalty for leading opponent */
        if(f->max_evps>=7) extra = f->max_evps * (-5e13);
        break;
    }
    return base + extra;
}

double strat_value_fn(Game *g, Color c) {
    F f=extract(g,c);
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if(seat==active_seat && active_variant>=0)
        return variant_eval(&f, active_variant);
    return baseline_eval(&f);
}

static int run_tournament(int variant, int n_games, int seed_base) {
    active_variant=variant;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int wins=0;
    for(int gi=0;gi<n_games;gi++){
        active_seat=gi%4;
        int seed=seed_base+gi;
        CatanMap map;rng_seed((uint64_t)seed);build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,AB_DEPTH,-1e30,1e30,9e99,cur);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==w){ws=i;break;}
            if(ws==active_seat)wins++;}
    }
    active_variant=-1;active_seat=-1;
    return wins;
}

static const char *vnames[]={"all_enemies","vp_scaled_eprod","ore_wheat_bonus",
    "anti_concentration","diversity_bonus","city_squared","road_waste_penalty",
    "phase_scaling","combo_all+city+orewheat","aggressive_vs_leader"};
#define NUM_V 10

int main(void) {
    printf("=== Strategy Search: 10 variants, %d 4p games each ===\n", GAMES);
    printf("1 seat variant vs 3 baseline. Baseline = 25%%\n");
    printf("p<0.05 threshold at N=%d: >27.7%%\n\n", GAMES);

    clock_t t0=clock(); int tg=0;
    typedef struct{int id;double wr;int wins;}R;
    R results[NUM_V];

    for(int v=0;v<NUM_V;v++){
        int seed_base=50000000+v*10000;
        int wins=run_tournament(v,GAMES,seed_base);
        tg+=GAMES;
        double wr=100.*wins/(double)GAMES;
        results[v]=(R){v,wr,wins};
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  [%2d] %-30s %d/%d = %5.1f%%%s  [%dk, %.0fs]\n",
               v,vnames[v],wins,GAMES,wr,wr>27.7?" <<<":"",tg/1000,el);
    }

    /* Sort */
    for(int i=0;i<NUM_V-1;i++)for(int j=i+1;j<NUM_V;j++)
        if(results[j].wr>results[i].wr){R t=results[i];results[i]=results[j];results[j]=t;}

    printf("\n=== Ranked ===\n");
    for(int i=0;i<NUM_V;i++){
        R *r=&results[i];
        printf("  %2d. %-30s %5.1f%%  %+.1f pp%s\n",
               i+1,vnames[r->id],r->wr,r->wr-25,r->wr>27.7?" ***":"");
    }

    /* Validate top 2 at 3000 games on fresh seeds */
    printf("\n=== Validation (3000 games, fresh seeds) ===\n");
    for(int i=0;i<2;i++){
        int v=results[i].id;
        int wins=run_tournament(v,3000,70000000+v*100000);
        tg+=3000;
        double wr=100.*wins/3000.;
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  %-30s %d/3000 = %.1f%% (%+.1f pp)  [%dk, %.0fs]\n",
               vnames[v],wins,wr,wr-25,tg/1000,el);
    }

    return 0;
}
