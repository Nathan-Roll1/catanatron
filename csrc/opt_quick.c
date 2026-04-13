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
static int active_seat=-1; static double feat1_w=-1e6; static int use_feat1=0;

double opt_value_fn(Game *g, Color c) {
    State *s=&g->state;int idx=s->color_to_index[(int)c];
    Board *b=&s->board;CatanMap *map=b->map;Coordinate robber=b->robber_coordinate;
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
    double prod=tp+var*4.*(2.778/100.);
    Color enemy=COLOR_NONE;for(int i=0;i<s->num_players;i++)if(s->colors[i]!=c){enemy=s->colors[i];break;}
    double ep=0;
    if(enemy!=COLOR_NONE){int ei=s->color_to_index[(int)enemy];
        for(int si=0;si<s->settlement_count[ei];si++){int node=s->settlements[ei][si];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=DICE_P[t->number];}}
        for(int ci=0;ci<s->city_count[ei];ci++){int node=s->cities[ei][ci];
            for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
                LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
                if(coord_eq(map->land_tile_coords[t2],robber))continue;ep+=2.*DICE_P[t->number];}}}
    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    double syn=(2-dc-ds)/2.;int nih=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    int buf[TOTAL_NODES];int nb=board_buildable_node_ids(&s->board,c,false,buf,TOTAL_NODES);
    double lrf=(nb==0)?W[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    double val=ps[PS_VICTORY_POINTS]*W[0]+prod*W[1]+ep*W[2]+nt*W[3]+nb*W[4]
              +ps[PS_LONGEST_ROAD_LENGTH]*lrf+syn*W[6]+nih*W[7]+(nih>7?W[8]:0)+nd*W[9]+ps[PS_PLAYED_KNIGHT]*W[10];

    /* Port-production-synergy interaction */
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if(seat==active_seat && use_feat1){
        bool ports[6];board_get_player_port_resources(b,c,ports);
        double pm=0;for(int r=0;r<5;r++)if(ports[r])pm+=rp[r];
        val+=feat1_w*pm*syn;
    }
    return val;
}

static int run_batch(int n_games, int seed_base) {
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int wins=0;
    for(int gi=0;gi<n_games;gi++){
        active_seat=gi%4;int seed=seed_base+gi;
        CatanMap map;rng_seed((uint64_t)seed);build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action actions[MAX_ACTIONS];int n=generate_playable_actions(&game.state,actions,MAX_ACTIONS);
        int ticks=0;
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT&&ticks<5000){
            Action action;if(n==1){action=actions[0];}
            else{Color cur=state_current_color(&game.state);double dl=(double)clock()/CLOCKS_PER_SEC+120.;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,actions,n,AB_DEPTH,-1e30,1e30,dl,cur);
                action=(sr.action.type!=0||sr.action.color!=0)?sr.action:actions[0];}
            game_execute(&game,action,actions,&n);ticks++;
        }
        Color winner=game_winning_color(&game);
        if(winner!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==winner){ws=i;break;}
            if(ws==active_seat)wins++;}
    }
    active_seat=-1;return wins;
}

int main(void) {
    clock_t t0=clock();int tg=0;

    /* Phase 1: Tune weight for port_match*synergy */
    printf("=== Phase 1: Tune port_match*synergy weight (2000 games each) ===\n");
    double wts[]={-3e5,-5e5,-1e6,-2e6,-5e6};
    int best_w_idx=0;int best_wins=0;
    use_feat1=1;
    for(int i=0;i<5;i++){
        feat1_w=wts[i];
        int wins=run_batch(2000,30000000+i*5000);tg+=2000;
        double wr=100.*wins/2000.;
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  w=%+.0e: %d/2000 = %.1f%%  [%dk, %.0fs]\n",wts[i],wins,wr,tg/1000,el);
        if(wins>best_wins){best_wins=wins;best_w_idx=i;}
    }
    feat1_w=wts[best_w_idx];
    printf("  Best: w=%+.0e (%.1f%%)\n\n",feat1_w,100.*best_wins/2000.);

    /* Phase 2: Validate feat1 at best weight, 10000 games */
    printf("=== Phase 2: Validate (10000 games, seeds 80M+) ===\n");
    use_feat1=1;
    int v_enhanced=run_batch(10000,80000000);tg+=10000;
    use_feat1=0;
    int v_baseline=run_batch(10000,90000000);tg+=10000;

    double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    double e_wr=100.*v_enhanced/10000.;
    double b_wr=100.*v_baseline/10000.;
    printf("  Enhanced (port_match*syn w=%+.0e): %d/10000 = %.2f%%\n",feat1_w,v_enhanced,e_wr);
    printf("  Baseline:                          %d/10000 = %.2f%%\n",v_baseline,b_wr);
    printf("  Improvement: %+.2f pp\n",e_wr-b_wr);
    printf("\n  Total: %dk games in %.0fs (%.0f g/s)\n",tg/1000,el,tg/el);

    return 0;
}
