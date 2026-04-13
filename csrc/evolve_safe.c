/* Minimal evolve: test each candidate separately in its own process-like reset */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define AB_DEPTH 2
#define GAMES 1000

/* Use baseline eval directly -- no custom function pointer */
static int run_1seat_vs_3baseline(int seed_base, int n_games) {
    /* Just run standard AB:2 vs AB:2 vs AB:2 vs AB:2, count seat 0 wins */
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int seat0_wins=0;
    for(int gi=0;gi<n_games;gi++){
        int seed=seed_base+gi;
        CatanMap map;rng_seed((uint64_t)seed);build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        char rng_buf[2600];
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                rng_save_state(rng_buf);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,AB_DEPTH,-1e30,1e30,9e99,cur,base_value_fn);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w==colors[gi%4]) seat0_wins++; /* rotating "seat 0" */
    }
    return seat0_wins;
}

int main(void) {
    printf("=== Stability Test: 4p AB:2 baseline ===\n");
    clock_t t0=clock();
    
    /* First, verify 4p baseline works at all */
    printf("Running %d 4-player AB:2 games...\n", GAMES);
    int wins = run_1seat_vs_3baseline(400000000, GAMES);
    double el=(double)(clock()-t0)/CLOCKS_PER_SEC;
    printf("  Seat wins: %d/%d = %.1f%% (baseline 25%%)\n", wins, GAMES, 100.*wins/GAMES);
    printf("  Time: %.1fs (%.0f g/s)\n", el, GAMES/el);
    
    return 0;
}
