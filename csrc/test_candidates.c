/*
 * Test strategy candidates by modifying base weights.
 * Each test: 1 seat uses modified eval, 3 use original.
 * Candidates modify the WEIGHT constants, not the feature structure.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define GAMES 3000
#define AB_DEPTH 2

static int active_seat = -1;

/* We'll test depth 3 (free ROLL) vs depth 2 as a strategy change */
static int enhanced_depth = 2;

static int run_test(int n_games, int seed_base) {
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int wins=0;
    for(int gi=0;gi<n_games;gi++){
        active_seat=gi%4;
        int seed=seed_base+gi;
        CatanMap map;rng_seed((uint64_t)seed);build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        char rng_buf[2600];
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{
                Color cur=state_current_color(&game.state);
                int seat=-1;
                for(int i=0;i<4;i++)if(colors[i]==cur){seat=i;break;}
                int depth=(seat==active_seat)?enhanced_depth:2;
                rng_save_state(rng_buf);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,depth,-1e30,1e30,9e99,cur,base_value_fn);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE){int ws=-1;for(int i=0;i<4;i++)if(colors[i]==w){ws=i;break;}
            if(ws==active_seat)wins++;}
    }
    active_seat=-1;
    return wins;
}

int main(void) {
    clock_t t0=clock(); int tg=0;
    printf("=== Strategy Tests: %d 4p games each ===\n\n", GAMES);

    /* Test 1: AB:3 vs 3x AB:2 */
    enhanced_depth=3;
    int w3=run_test(GAMES,500000000);tg+=GAMES;
    double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("  AB:3 vs 3xAB:2: %d/%d = %.1f%% (%+.1f pp)  [%.0fs]\n",
           w3,GAMES,100.*w3/GAMES,100.*w3/GAMES-25,el);

    /* Test 2: AB:4 vs 3x AB:2 */
    enhanced_depth=4;
    int w4=run_test(GAMES,600000000);tg+=GAMES;
    el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("  AB:4 vs 3xAB:2: %d/%d = %.1f%% (%+.1f pp)  [%.0fs]\n",
           w4,GAMES,100.*w4/GAMES,100.*w4/GAMES-25,el);

    /* Test 3: AB:2 baseline (control) */
    enhanced_depth=2;
    int w2=run_test(GAMES,700000000);tg+=GAMES;
    el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("  AB:2 vs 3xAB:2: %d/%d = %.1f%% (%+.1f pp)  [%.0fs] (control)\n",
           w2,GAMES,100.*w2/GAMES,100.*w2/GAMES-25,el);

    printf("\n  Total: %dk games in %.0fs (%.0f g/s)\n",tg/1000,el,tg/el);

    printf("\n=== Summary ===\n");
    printf("  AB:2 baseline: %.1f%%\n", 100.*w2/GAMES);
    printf("  AB:3 (free ROLL): %.1f%% (%+.1f pp)\n", 100.*w3/GAMES, 100.*w3/GAMES-100.*w2/GAMES);
    printf("  AB:4 (free ROLL): %.1f%% (%+.1f pp)\n", 100.*w4/GAMES, 100.*w4/GAMES-100.*w2/GAMES);

    return 0;
}
