/*
 * Compact evolutionary optimizer with proper value function injection.
 * Tests 8 micro-feature candidates, 5000 games each, validates top 2 OOS.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define AB_DEPTH 2
#define SCREEN_GAMES 1000
#define VALIDATE_GAMES 2000
#define NUM_CANDS 8

static const double DP[13]={0,0,1./36,2./36,3./36,4./36,5./36,6./36,5./36,4./36,3./36,2./36,1./36};

static int active_seat = -1;
static int active_cand = -1;
static double active_w = 0;

static double compute_micro(Game *g, Color c, int cand_id) {
    State *s=&g->state; int idx=s->color_to_index[(int)c];
    Board *b=&s->board; CatanMap *map=b->map; Coordinate robber=b->robber_coordinate;
    
    double rp[5]={0};
    for(int si=0;si<s->settlement_count[idx];si++){int node=s->settlements[idx][si];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=DP[t->number];}}
    for(int ci=0;ci<s->city_count[idx];ci++){int node=s->cities[idx][ci];
        for(int ti=0;ti<map->adjacent_tiles_count[node];ti++){int t2=map->adjacent_tiles[node][ti];
            LandTile *t=&map->land_tiles[t2];if(t->resource==RES_NONE||t->number==0)continue;
            if(coord_eq(map->land_tile_coords[t2],robber))continue;rp[(int)t->resource]+=2.*DP[t->number];}}
    int *ps=s->player_state[idx];

    switch(cand_id) {
    case 0: /* ore+wheat prod (city materials) */
        return rp[3]+rp[4];
    case 1: /* wood+brick prod (road/settle materials) */
        return rp[0]+rp[1];
    case 2: /* producing resource diversity (count) */
        {int c2=0;for(int r=0;r<5;r++)if(rp[r]>0)c2++;return c2;}
    case 3: /* max enemy VP gap */
        {int maxev=0;for(int i=0;i<s->num_players;i++){if((int)s->colors[i]==(int)c)continue;
            int ev=s->player_state[i][PS_VICTORY_POINTS];if(ev>maxev)maxev=ev;}
         return fmax(0,maxev-ps[PS_VICTORY_POINTS]);}
    case 4: /* cities squared */
        return s->city_count[idx]*s->city_count[idx];
    case 5: /* cards over discard limit */
        {int h=ps[PS_WOOD_IN_HAND]+ps[PS_BRICK_IN_HAND]+ps[PS_SHEEP_IN_HAND]+ps[PS_WHEAT_IN_HAND]+ps[PS_ORE_IN_HAND];
         return fmax(0,h-7);}
    case 6: /* army approaching threshold */
        return fmax(0,ps[PS_PLAYED_KNIGHT]-1);
    case 7: /* production per settlement */
        {double tp=0;for(int r=0;r<5;r++)tp+=rp[r];
         int ns=s->settlement_count[idx]+s->city_count[idx];
         return ns>0?tp/ns:0;}
    default: return 0;
    }
}

static const char *cand_names[NUM_CANDS]={
    "ore_wheat_prod","wood_brick_prod","diversity","leader_gap",
    "cities_sq","cards_over_7","army_threshold","prod_per_bldg"
};

static double enhanced_eval(Game *g, Color c) {
    double base = base_value_fn(g, c);
    if (active_cand < 0 || active_seat < 0) return base;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat=-1;for(int i=0;i<4;i++)if(colors[i]==c){seat=i;break;}
    if(seat < 0 || seat != active_seat) return base;
    double micro = compute_micro(g, c, active_cand);
    if (!isfinite(micro)) return base;
    return base + active_w * micro;
}

static int run_4p(int cand, double w, int n_games, int seed_base) {
    active_cand=cand; active_w=w;
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
                char rng_buf[2600]; rng_save_state(rng_buf);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,AB_DEPTH,-1e30,1e30,9e99,cur,enhanced_eval);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color winner=game_winning_color(&game);
        if(winner!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==winner){ws=i;break;}
            if(ws==active_seat)wins++;}
    }
    active_cand=-1;active_seat=-1;
    return wins;
}

int main(void) {
    printf("=== Feature Search with Proper Value Injection ===\n");
    printf("%d candidates, %d screen games, %d validation games\n", NUM_CANDS, SCREEN_GAMES, VALIDATE_GAMES);
    printf("Threshold (p<0.01 at N=%d): >26.6%%\n\n", SCREEN_GAMES);

    clock_t t0=clock(); int tg=0;
    typedef struct{int id;double w;double wr;int wins;}R;
    R results[NUM_CANDS];

    double try_w[]={1e5,1e6,1e7,-1e5,-1e6,-1e7};

    for(int c=0;c<NUM_CANDS;c++){
        int bw=0;double bwt=0;
        for(int wi=0;wi<6;wi++){
            int wins=run_4p(c,try_w[wi],SCREEN_GAMES,100000000+c*1000000+wi*100000);
            tg+=SCREEN_GAMES;
            if(wins>bw){bw=wins;bwt=try_w[wi];}
        }
        double wr=100.*bw/(double)SCREEN_GAMES;
        results[c]=(R){c,bwt,wr,bw};
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  %-20s best_w=%+.0e %d/%d = %5.1f%%%s [%dk, %.0fs, %.0f g/s]\n",
               cand_names[c],bwt,bw,SCREEN_GAMES,wr,wr>26.6?" <<<":"",tg/1000,el,tg/el);
    }

    /* Sort */
    for(int i=0;i<NUM_CANDS-1;i++)for(int j=i+1;j<NUM_CANDS;j++)
        if(results[j].wr>results[i].wr){R t=results[i];results[i]=results[j];results[j]=t;}

    printf("\n=== Top 3 -> Validation (%d OOS games) ===\n", VALIDATE_GAMES);
    for(int i=0;i<3;i++){
        R *r=&results[i];
        int wins=run_4p(r->id,r->w,VALIDATE_GAMES,200000000+r->id*1000000);
        tg+=VALIDATE_GAMES;
        double wr=100.*wins/(double)VALIDATE_GAMES;
        double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
        printf("  %-20s w=%+.0e: %d/%d = %.1f%% (%+.1f pp)%s  [%dk, %.0fs]\n",
               cand_names[r->id],r->w,wins,VALIDATE_GAMES,wr,wr-25,
               wr>26.6?" ** SIGNIFICANT **":"",tg/1000,el);
    }

    double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("\nTotal: %dk games in %.0fs (%.0f g/s)\n",tg/1000,el,tg/el);
    return 0;
}
