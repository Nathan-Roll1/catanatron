/*
 * Deep optimization: 
 * 1. Bake in port_match*synergy (confirmed +2pp)
 * 2. Fine-tune its weight with binary search (5000 games per test)
 * 3. Search for a SECOND interaction on top of the first
 * 4. Final validation: 10000 games, both features, vs pure baseline
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

static const double DICE_P[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double W[11]={3e14,1e8,-1e8,1.0,1e3,10.0,1e2,1.0,-5.0,10.0,10.1};

static int active_seat = -1;
static int use_enhanced = 0; /* 0=baseline only, 1=feature1 only, 2=feature1+feature2 */
static double w_feat1 = -1e6;
static int feat2_id = -1;
static double w_feat2 = 0;

typedef struct {
    double prod, eprod, synergy, hand, buildable, tiles, vps, army, devs, road;
    double port_prod_match, prod_hhi, max_enemy_vps, city_ratio;
    int settlements, cities;
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
    if(tp>0){double ss=0;for(int r=0;r<5;r++){double s2=rp[r]/tp;ss+=s2*s2;}f.prod_hhi=ss;}else f.prod_hhi=1.;
    bool ports[6];board_get_player_port_resources(b,p0,ports);
    f.port_prod_match=0;for(int r=0;r<5;r++)if(ports[r])f.port_prod_match+=rp[r];
    Color enemy=COLOR_NONE;for(int i=0;i<s->num_players;i++)if(s->colors[i]!=p0){enemy=s->colors[i];break;}
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;f.eprod+=DICE_P[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;f.eprod+=2.*DICE_P[t->number];}}}
    f.max_enemy_vps=0;for(int i=0;i<s->num_players;i++){if(s->colors[i]==p0)continue;
        double v=s->player_state[i][PS_VICTORY_POINTS];if(v>f.max_enemy_vps)f.max_enemy_vps=v;}
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    f.synergy=(2-dc-ds)/2.;f.hand=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    f.tiles=nt;int buf[TOTAL_NODES];f.buildable=board_buildable_node_ids(&s->board,p0,false,buf,TOTAL_NODES);
    f.road=ps[PS_LONGEST_ROAD_LENGTH];f.vps=ps[PS_VICTORY_POINTS];f.army=ps[PS_PLAYED_KNIGHT];
    f.devs=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    f.settlements=s->settlement_count[idx];f.cities=s->city_count[idx];
    f.city_ratio=(f.settlements+f.cities>0)?(double)f.cities/(f.settlements+f.cities):0;
    return f;
}

static double compute_feat2(Features *f, int id) {
    switch(id){
        case 0: return f->port_prod_match;
        case 1: return (f->prod_hhi>0)?(1./f->prod_hhi):0;
        case 2: return f->max_enemy_vps * f->eprod;
        case 3: return f->city_ratio * f->prod;
        case 4: return f->settlements * f->buildable;
        case 5: return f->vps * f->army;
        case 6: return (10. - f->vps) * f->buildable;
        case 7: return f->prod * f->prod;
        case 8: return f->devs * f->army;
        case 9: return f->max_enemy_vps * f->max_enemy_vps;
        default: return 0;
    }
}
static const char *f2names[]={"port_prod","1/hhi","evps*eprod","city*prod","settle*build",
    "vps*army","(10-vps)*build","prod^2","devs*army","evps^2"};
#define NUM_F2 10

double nl_value_fn(Game *g, Color c) {
    Features f=extract(g,c);
    int idx=g->state.color_to_index[(int)c];int *ps=g->state.player_state[idx];
    double lrf=(f.buildable==0)?W[5]:0.1;
    double val=ps[PS_VICTORY_POINTS]*W[0]+f.prod*W[1]+f.eprod*W[2]+f.tiles*W[3]
              +f.buildable*W[4]+f.road*lrf+f.synergy*W[6]+f.hand*W[7]
              +(f.hand>7?W[8]:0)+f.devs*W[9]+f.army*W[10];
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if(seat==active_seat && use_enhanced>=1)
        val += w_feat1 * f.port_prod_match * f.synergy;
    if(seat==active_seat && use_enhanced>=2 && feat2_id>=0)
        val += w_feat2 * compute_feat2(&f, feat2_id);
    return val;
}

static int run_games(int n_games, int seed_base) {
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int wins=0;
    for(int gi=0;gi<n_games;gi++){
        active_seat=gi%4; int seed=seed_base+gi;
        CatanMap map;rng_seed((uint64_t)seed);build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action actions[MAX_ACTIONS];int n=generate_playable_actions(&game.state,actions,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action action;if(n==1){action=actions[0];}
            else{Color cur=state_current_color(&game.state);double dl=(double)clock()/CLOCKS_PER_SEC+120.;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,actions,n,AB_DEPTH,-1e30,1e30,dl,cur);
                action=(sr.action.type!=0||sr.action.color!=0)?sr.action:actions[0];}
            game_execute(&game,action,actions,&n);}
        Color winner=game_winning_color(&game);
        if(winner!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==winner){ws=i;break;}
            if(ws==active_seat)wins++;}
    }
    active_seat=-1;
    return wins;
}

int main(void) {
    clock_t t0=clock(); int total_games=0;

    /* === PHASE 1: Tune w_feat1 weight === */
    printf("=== Phase 1: Tune port_match*synergy weight ===\n");
    printf("Testing 5 weight values, 3000 games each\n\n");
    double test_w1[]={-5e5, -1e6, -2e6, -5e6, -1e7};
    int best_w1_wins=0; double best_w1=0;
    use_enhanced=1; feat2_id=-1;
    for(int i=0;i<5;i++){
        w_feat1=test_w1[i];
        int wins=run_games(3000, 30000000+i*10000);
        total_games+=3000;
        double wr=100.*wins/3000.;
        double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  w=%+.0e: %d/3000 = %.1f%%  [%dk, %.0fs]\n", test_w1[i],wins,wr,total_games/1000,elapsed);
        if(wins>best_w1_wins){best_w1_wins=wins;best_w1=test_w1[i];}
    }
    w_feat1=best_w1;
    printf("  Best: w=%+.0e (%.1f%%)\n\n", best_w1, 100.*best_w1_wins/3000.);

    /* === PHASE 2: Search for second feature === */
    printf("=== Phase 2: Search for 2nd feature (on top of feat1) ===\n");
    printf("Testing %d candidates at 3 weights, 2000 games each\n\n", NUM_F2);
    use_enhanced=2;
    typedef struct{int id;double w;double wr;int wins;}R;
    R screen2[NUM_F2];
    double try2[]={1e5,1e6,1e7};
    for(int c=0;c<NUM_F2;c++){
        int bw=0;double bwt=0;
        for(int wi=0;wi<3;wi++){for(int sign=0;sign<2;sign++){
            double w2=(sign?-1:1)*try2[wi]; feat2_id=c; w_feat2=w2;
            int wins=run_games(2000, 40000000+c*100000+wi*10000+sign*5000);
            total_games+=2000;
            if(wins>bw){bw=wins;bwt=w2;}
        }}
        double wr=100.*bw/2000.;
        screen2[c]=(R){c,bwt,wr,bw};
        double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  %-20s best_w=%+.0e %d/2000=%.1f%%%s [%dk, %.0fs]\n",
               f2names[c],bwt,bw,wr,wr>27.0?" <<<":"",total_games/1000,elapsed);
    }
    /* Sort */
    for(int i=0;i<NUM_F2-1;i++)for(int j=i+1;j<NUM_F2;j++)
        if(screen2[j].wr>screen2[i].wr){R tmp=screen2[i];screen2[i]=screen2[j];screen2[j]=tmp;}
    
    printf("\n  Top 3:\n");
    for(int i=0;i<3;i++) printf("    %s w=%+.0e %.1f%%\n",f2names[screen2[i].id],screen2[i].w,screen2[i].wr);

    /* === PHASE 3: Validate top combo on 10000 fresh games === */
    printf("\n=== Phase 3: Final validation (10000 games) ===\n");
    
    /* Test enhanced (feat1 + best feat2) vs pure baseline */
    feat2_id=screen2[0].id; w_feat2=screen2[0].w;
    
    /* Enhanced */
    use_enhanced=2;
    int enh_wins=run_games(10000, 50000000);
    total_games+=10000;
    double enh_wr=100.*enh_wins/10000.;
    
    /* Feat1 only */
    use_enhanced=1;
    int f1_wins=run_games(10000, 60000000);
    total_games+=10000;
    double f1_wr=100.*f1_wins/10000.;

    double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("\n==========================================\n");
    printf("  Final Results (10000 games each, fresh seeds)\n");
    printf("==========================================\n");
    printf("  Baseline (no features):    25.00%% (by definition)\n");
    printf("  + port_match*synergy:      %.2f%% (%+.2f pp)\n", f1_wr, f1_wr-25);
    printf("  + port_match*syn + %s: %.2f%% (%+.2f pp)\n", 
           f2names[screen2[0].id], enh_wr, enh_wr-25);
    printf("\n  feat1 weight: %+.0e\n", w_feat1);
    printf("  feat2 (%s) weight: %+.0e\n", f2names[screen2[0].id], w_feat2);
    printf("\n  Total: %dk games in %.0fs (%.0f g/s)\n",total_games/1000,elapsed,total_games/elapsed);
    
    return 0;
}
