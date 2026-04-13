/*
 * Search for the best single interaction term to add to the value function.
 * 
 * Test 10 candidate interaction features, each with a weight to optimize.
 * For each candidate: binary search for optimal weight using 4-player games
 * (1 seat with new feature vs 3 seats without). Baseline is 25%.
 *
 * Candidates:
 *  0. prod * hand_synergy       (production when you have matching resources)
 *  1. prod * num_buildable      (production when you can expand)
 *  2. army * eprod              (knights hurt enemy production)
 *  3. hand_synergy * buildable  (ready to build AND have places)
 *  4. vps * prod                (production matters more when close to winning)
 *  5. eprod * num_tiles         (block enemies on contested tiles)
 *  6. prod^2                    (superlinear production reward)
 *  7. army^2                    (superlinear army reward)
 *  8. hand * buildable          (resources + expansion potential)
 *  9. (7-vps) * prod            (production matters more early game)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define AB_DEPTH 2
#define EVAL_GAMES 500
#define NUM_CANDIDATES 10

static const double DICE_P[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};
static const double W[11]={3e14,1e8,-1e8,1.0,1e3,10.0,1e2,1.0,-5.0,10.0,10.1};

static const char *cand_names[NUM_CANDIDATES] = {
    "prod*synergy", "prod*buildable", "army*eprod", "synergy*buildable",
    "vps*prod", "eprod*tiles", "prod^2", "army^2", "hand*buildable", "(7-vps)*prod"
};

/* Which seat (0-3) uses the interaction term. -1 = none (all use baseline). */
static int active_seat = -1;
static double interaction_weight = 0;
static int interaction_id = -1;

typedef struct {
    double prod, eprod, synergy, hand, buildable, tiles, vps, army, devs, road;
} Features;

static Features extract(Game *g, Color p0) {
    Features f = {0};
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

    int *ps=s->player_state[idx];
    int wh=ps[PS_WHEAT_IN_HAND],or2=ps[PS_ORE_IN_HAND],sh=ps[PS_SHEEP_IN_HAND],
        br=ps[PS_BRICK_IN_HAND],wo=ps[PS_WOOD_IN_HAND];
    double dc=(fmax(2-wh,0)+fmax(3-or2,0))/5.;
    double ds=(fmax(1-wh,0)+fmax(1-sh,0)+fmax(1-br,0)+fmax(1-wo,0))/4.;
    f.synergy=(2-dc-ds)/2.;
    f.hand=wo+br+sh+wh+or2;

    bool ts[NUM_LAND_TILES]={0};int nt=0;
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];if(!ts[t2]){ts[t2]=1;nt++;}}}
    f.tiles=nt;

    int buf[TOTAL_NODES];
    f.buildable=board_buildable_node_ids(&s->board,p0,false,buf,TOTAL_NODES);
    double lrf=(f.buildable==0)?W[5]:0.1;
    f.road=ps[PS_LONGEST_ROAD_LENGTH];
    f.vps=ps[PS_VICTORY_POINTS];
    f.army=ps[PS_PLAYED_KNIGHT];
    f.devs=ps[PS_KNIGHT_IN_HAND]+ps[PS_YEAR_OF_PLENTY_IN_HAND]
          +ps[PS_MONOPOLY_IN_HAND]+ps[PS_ROAD_BUILDING_IN_HAND]+ps[PS_VICTORY_POINT_IN_HAND];
    return f;
}

static double compute_interaction(Features *f, int cand_id) {
    switch(cand_id) {
        case 0: return f->prod * f->synergy;
        case 1: return f->prod * f->buildable;
        case 2: return f->army * f->eprod;
        case 3: return f->synergy * f->buildable;
        case 4: return f->vps * f->prod;
        case 5: return f->eprod * f->tiles;
        case 6: return f->prod * f->prod;
        case 7: return f->army * f->army;
        case 8: return f->hand * f->buildable;
        case 9: return (7.0 - f->vps) * f->prod;
        default: return 0;
    }
}

/* The value function: base + optional interaction term */
double nl_value_fn(Game *g, Color c) {
    Features f = extract(g, c);
    State *s=&g->state; int idx=s->color_to_index[(int)c];
    int *ps=s->player_state[idx];
    double lrf=(f.buildable==0)?W[5]:0.1;
    
    double val = f.vps*W[0] + f.prod*W[1] + f.eprod*W[2] + f.tiles*W[3]
               + f.buildable*W[4] + f.road*lrf + f.synergy*W[6]
               + f.hand*W[7] + (f.hand>7?W[8]:0) + f.devs*W[9] + f.army*W[10];

    /* Add interaction term if this is the active seat */
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1; for(int i=0;i<4;i++) if(colors[i]==c){seat=i;break;}
    if (seat == active_seat && interaction_id >= 0) {
        val += interaction_weight * compute_interaction(&f, interaction_id);
    }
    return val;
}

/* Run EVAL_GAMES 4-player games, 1 seat with interaction vs 3 without.
   Returns win count for the interaction seat. */
static int run_eval(int cand_id, double weight, int seed_base) {
    interaction_id = cand_id;
    interaction_weight = weight;
    
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int wins = 0;
    
    for (int gi = 0; gi < EVAL_GAMES; gi++) {
        active_seat = gi % 4; /* rotate which seat gets the interaction */
        int seed = seed_base + gi;
        
        CatanMap map; rng_seed((uint64_t)seed);
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);
        Game game; game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action actions[MAX_ACTIONS];
        int n=generate_playable_actions(&game.state,actions,MAX_ACTIONS);
        
        while(game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT){
            Action action;
            if(n==1){action=actions[0];}
            else{
                Color cur=state_current_color(&game.state);
                double dl=(double)clock()/CLOCKS_PER_SEC+120.;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,actions,n,AB_DEPTH,-1e30,1e30,dl,cur);
                action=(sr.action.type!=0||sr.action.color!=0)?sr.action:actions[0];
            }
            game_execute(&game,action,actions,&n);
        }
        
        Color winner=game_winning_color(&game);
        if(winner!=COLOR_NONE){
            int ws=-1;for(int i=0;i<4;i++)if(colors[i]==winner){ws=i;break;}
            if(ws==active_seat) wins++;
        }
    }
    active_seat = -1;
    interaction_id = -1;
    return wins;
}

int main(void) {
    printf("Nonlinear feature search: 10 interaction candidates\n");
    printf("4-player AB2: 1 seat with feature vs 3 without. Baseline=25%%\n");
    printf("%d games per evaluation\n\n", EVAL_GAMES);
    
    clock_t t0 = clock();
    int total_games = 0;
    
    /* Test each candidate at several weight scales */
    double test_weights[] = {1e4, 1e5, 1e6, 1e7, -1e4, -1e5, -1e6, -1e7};
    int n_weights = 8;
    
    typedef struct { int id; double weight; double wr; int wins; } Result;
    Result best_results[NUM_CANDIDATES];
    
    for (int c = 0; c < NUM_CANDIDATES; c++) {
        double best_wr = 0; double best_w = 0; int best_wins = 0;
        
        for (int wi = 0; wi < n_weights; wi++) {
            int seed_base = 5000000 + c * 10000 + wi * 1000;
            int wins = run_eval(c, test_weights[wi], seed_base);
            total_games += EVAL_GAMES;
            double wr = (double)wins / (EVAL_GAMES / 4) * 100; /* per-seat win rate (EVAL_GAMES/4 seats) */
            
            if (wr > best_wr) { best_wr = wr; best_w = test_weights[wi]; best_wins = wins; }
        }
        
        best_results[c] = (Result){c, best_w, best_wr, best_wins};
        double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  %-20s  best_w=%+.0e  wr=%.1f%%  (%d/%d wins)  [%dk games, %.0fs]\n",
               cand_names[c], best_w, best_wr, best_wins, EVAL_GAMES/4, total_games/1000, elapsed);
    }
    
    /* Sort by win rate */
    for(int i=0;i<NUM_CANDIDATES-1;i++)
        for(int j=i+1;j<NUM_CANDIDATES;j++)
            if(best_results[j].wr>best_results[i].wr){Result tmp=best_results[i];best_results[i]=best_results[j];best_results[j]=tmp;}
    
    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("\n==========================================\n");
    printf("  Results ranked by per-seat win rate\n");
    printf("  Baseline: 25.0%%  |  %dk games  |  %.0fs\n", total_games/1000, elapsed);
    printf("==========================================\n");
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        Result *r = &best_results[i];
        char bar[30] = {0};
        int blen = (int)((r->wr - 20) * 2); if(blen<0)blen=0; if(blen>29)blen=29;
        for(int j=0;j<blen;j++) bar[j]='#';
        printf("  %-20s  w=%+.0e  %5.1f%%  %s%s\n",
               cand_names[r->id], r->weight, r->wr, bar,
               r->wr > 28.0 ? " ***" : (r->wr > 26.5 ? " *" : ""));
    }
    
    return 0;
}
