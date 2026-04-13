#include <stdio.h>
#include <string.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

int main(void) {
    Color colors[2]={COLOR_RED,COLOR_BLUE};
    int N=1000;
    int wins[4]={0};
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int gi=0;gi<N;gi++){
        CatanMap map;rng_seed((uint64_t)(gi+999000000));
        build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,2,colors,(uint64_t)(gi+999000000),7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        char rng_buf[2600];
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{Color cur=state_current_color(&game.state);
                rng_save_state(rng_buf);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,3,-1e30,1e30,9e99,cur,base_value_fn);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE)wins[(int)w]++;
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("1000 AB:3 vs AB:3 (2p): %.1fs = %.0f games/sec\n",el,N/el);
    printf("RED=%d BLUE=%d\n",wins[0],wins[1]);
    return 0;
}
