#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define AB_DEPTH 2
#define N_GAMES 2000

static const double DICE_P[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double W[11]={3e14,1e8,-1e8,1.0,1e3,10.0,1e2,1.0,-5.0,10.0,10.1};

static int active_seat = -1;

double nl_value_fn(Game *g, Color c) {
    State *s=&g->state; int idx=s->color_to_index[(int)c];
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
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    double hs=(2-dc-ds)/2.;int nih=wo+br+sh+wh+or2;
    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    int buf[TOTAL_NODES];int nb=board_buildable_node_ids(&s->board,c,false,buf,TOTAL_NODES);
    double lrf=(nb==0)?W[5]:0.1;
    int nd=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]+ps[PS_MONOPOLY_IN_HAND]
          +ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    
    double val = ps[PS_VICTORY_POINTS]*W[0]+prod*W[1]+ep*W[2]+nt*W[3]+nb*W[4]
                +ps[PS_LONGEST_ROAD_LENGTH]*lrf+hs*W[6]+nih*W[7]+(nih>7?W[8]:0)
                +nd*W[9]+ps[PS_PLAYED_KNIGHT]*W[10];
    
    /* Add vps*prod interaction for the enhanced seat */
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if (seat == active_seat)
        val += 1e7 * ps[PS_VICTORY_POINTS] * prod;
    
    return val;
}

int main(void) {
    printf("Validating vps*prod interaction (w=+1e7)\n");
    printf("1 seat enhanced vs 3 baseline, %d 4p games, fresh seeds 7M+\n\n", N_GAMES);
    
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int enhanced_wins=0;
    clock_t t0=clock();
    
    for(int gi=0;gi<N_GAMES;gi++){
        active_seat = gi % 4;
        int seed = 7000000 + gi;
        CatanMap map;rng_seed((uint64_t)seed);
        build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action actions[MAX_ACTIONS];
        int n=generate_playable_actions(&game.state,actions,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT){
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
            if(ws==active_seat)enhanced_wins++;}
        
        if((gi+1)%500==0){
            double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
            double wr=100.*enhanced_wins/(gi+1);
            printf("  [%d/%d] enhanced=%d wins (%.1f%%)  %.0fs\n",gi+1,N_GAMES,enhanced_wins,wr,elapsed);
        }
    }
    
    active_seat=-1;
    double elapsed=(double)(clock()-t0)/CLOCKS_PER_SEC;
    double wr=100.*enhanced_wins/(double)N_GAMES;
    
    printf("\n==========================================\n");
    printf("  vps*prod (w=+1e7) validation: %d games\n", N_GAMES);
    printf("==========================================\n");
    printf("  Enhanced seat: %d / %d wins = %.2f%%\n", enhanced_wins, N_GAMES, wr);
    printf("  Baseline:      25.00%%\n");
    printf("  Improvement:   %+.2f pp\n", wr - 25.0);
    printf("  Time: %.0fs (%.0f games/sec)\n", elapsed, N_GAMES/elapsed);
    
    return 0;
}
