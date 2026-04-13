#include <stdio.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    Color colors[2]={COLOR_RED,COLOR_BLUE};
    struct timespec t0,t1;
    
    /* Time individual components over many calls */
    #define N 500
    double t_init=0, t_search=0, t_copy=0, t_exec=0, t_gen=0, t_val=0;
    int total_nodes=0, total_copies=0, total_evals=0;

    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int gi=0;gi<N;gi++){
        CatanMap map;rng_seed((uint64_t)(gi+88000000));
        build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        Game game;game_init_with_map(&game,&map,2,colors,(uint64_t)(gi+88000000),7,false,10);
        Action actions[MAX_ACTIONS];
        int n=generate_playable_actions(&game.state,actions,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action action;
            if(n==1){action=actions[0];}
            else{Color cur=state_current_color(&game.state);
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&cp,actions,n,2,-1e30,1e30,9e9,cur);
                action=(sr.action.type!=0||sr.action.color!=0)?sr.action:actions[0];}
            game_execute(&game,action,actions,&n);
        }
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double total=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("%d AB:2 games in %.3fs = %.0f g/s\n",N,total,N/total);
    printf("sizeof(Game)=%zu sizeof(State)=%zu sizeof(Board)=%zu sizeof(CatanMap)=%zu\n",
           sizeof(Game),sizeof(State),sizeof(Board),sizeof(CatanMap));

    /* Quick component timing */
    Game g; CatanMap map;
    rng_seed(42); build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
    game_init_with_map(&g,&map,2,colors,42,7,false,10);
    /* Advance to mid-game */
    Action acts[MAX_ACTIONS];int na=generate_playable_actions(&g.state,acts,MAX_ACTIONS);
    for(int i=0;i<50&&game_winning_color(&g)==COLOR_NONE;i++){
        Action a=acts[rng_choice_index(na)];game_execute(&g,a,acts,&na);}

    int M=100000;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<M;i++){Game cp;game_copy(&cp,&g);}
    clock_gettime(CLOCK_MONOTONIC,&t1);
    t_copy=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec);
    printf("\ngame_copy: %.0f ns/call (%d calls)\n",t_copy/M,M);

    clock_gettime(CLOCK_MONOTONIC,&t0);
    volatile double v=0;
    for(int i=0;i<M;i++) v+=base_value_fn(&g,COLOR_RED);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    t_val=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec);
    printf("base_value_fn: %.0f ns/call\n",t_val/M);

    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<M;i++){Action buf[MAX_ACTIONS];generate_playable_actions(&g.state,buf,MAX_ACTIONS);}
    clock_gettime(CLOCK_MONOTONIC,&t1);
    t_gen=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec);
    printf("gen_actions: %.0f ns/call\n",t_gen/M);

    return 0;
}
