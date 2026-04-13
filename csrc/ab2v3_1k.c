#include <stdio.h>
#include <string.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define GAMES 1000

int main(void) {
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    int combos[6][4]={
        {3,3,2,2},{3,2,3,2},{3,2,2,3},
        {2,3,3,2},{2,3,2,3},{2,2,3,3}
    };
    int ab2_wins=0, ab3_wins=0, ab2_seats=0, ab3_seats=0;
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);

    for(int gi=0;gi<GAMES;gi++){
        int *combo=combos[gi%6];
        for(int i=0;i<4;i++){if(combo[i]==2)ab2_seats++;else ab3_seats++;}

        int seed=800000000+gi;
        CatanMap map;rng_seed((uint64_t)seed);build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,4,colors,(uint64_t)seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        char rng_buf[2600];

        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a;if(n==1){a=acts[0];}
            else{
                Color cur=state_current_color(&game.state);
                int seat=-1;for(int i=0;i<4;i++)if(colors[i]==cur){seat=i;break;}
                int depth=combo[seat];
                rng_save_state(rng_buf);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,acts,n,depth,-1e30,1e30,9e99,cur,base_value_fn);
                rng_restore_state(rng_buf);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];
            }
            game_execute(&game,a,acts,&n);
        }
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE){
            int ws=-1;for(int i=0;i<4;i++)if(colors[i]==w){ws=i;break;}
            if(combo[ws]==2)ab2_wins++;else ab3_wins++;
        }
        if((gi+1)%200==0){
            clock_gettime(CLOCK_MONOTONIC,&t1);
            double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
            printf("  [%d/%d] AB2=%.1f%% AB3=%.1f%%  (%.0fs)\n",
                gi+1,GAMES,100.*ab2_wins/ab2_seats,100.*ab3_wins/ab3_seats,el);
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double el=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    printf("\n==========================================\n");
    printf("  2x AB:2 vs 2x AB:3 -- %d games\n", GAMES);
    printf("==========================================\n");
    printf("  AB:2: %d wins / %d seats = %.1f%%\n", ab2_wins, ab2_seats, 100.*ab2_wins/ab2_seats);
    printf("  AB:3: %d wins / %d seats = %.1f%%\n", ab3_wins, ab3_seats, 100.*ab3_wins/ab3_seats);
    printf("  Baseline: 25.0%%\n");
    printf("  Time: %.1fs (%.0f games/sec)\n", el, GAMES/el);

    return 0;
}
