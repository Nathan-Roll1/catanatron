#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define NT 18
#define GAMES 50004
#define GPT (GAMES/NT)

/* Simple: 1 seat AB:3 vs 3 seats AB:2. Rotate seat each game. */
typedef struct { int tid,n; uint64_t sb; int ab3_wins,total; } WA;

static void *worker(void *a) {
    WA *w=(WA*)a; w->ab3_wins=0; w->total=0;
    RngState rng;
    SearchCtx *ctx = (SearchCtx*)malloc(sizeof(SearchCtx));
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};

    for(int gi=0;gi<w->n;gi++){
        int ab3_seat = gi % 4;
        uint64_t seed=w->sb+(uint64_t)w->tid*100000ULL+gi;
        rng_init(&rng,seed);CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&rng);
        Game game;game_init_with_map(&game,&map,4,colors,seed,7,false,10);
        Action acts[MAX_ACTIONS];int n=generate_playable_actions(&game.state,acts,MAX_ACTIONS);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<TURNS_LIMIT){
            Action a2;if(n==1){a2=acts[0];}
            else{
                Color cur=state_current_color(&game.state);
                int seat=-1;for(int i=0;i<4;i++)if(colors[i]==cur){seat=i;break;}
                int depth=(seat==ab3_seat)?3:2;
                ctx->depth_counter=0;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(ctx,&cp,acts,n,depth,-1e30,1e30,cur,base_value_fn);
                a2=(sr.action.type||sr.action.color)?sr.action:acts[0];
            }
            game_execute(&game,a2,acts,&n);
        }
        Color wc=game_winning_color(&game);
        w->total++;
        if(wc!=COLOR_NONE&&game.state.color_to_index[(int)wc]==ab3_seat) w->ab3_wins++;
    }
    free(ctx);
    return NULL;
}

int main(void){
    srand(time(NULL));
    RngState tmp;rng_init(&tmp,0);CatanMap tm;build_map(&tm,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&tmp);
    board_init_static_graph(&tm);

    printf("=== AB:3 vs 3x AB:2: %dk games, %d threads ===\n\n",GAMES/1000,NT);
    uint64_t seed=(uint64_t)time(NULL)*1000003ULL;
    pthread_t th[NT];WA wa[NT];
    for(int i=0;i<NT;i++){wa[i].tid=i;wa[i].n=GPT;wa[i].sb=seed;}
    struct timespec T0,T1;clock_gettime(CLOCK_MONOTONIC,&T0);
    for(int i=0;i<NT;i++)pthread_create(&th[i],NULL,worker,&wa[i]);
    for(int i=0;i<NT;i++)pthread_join(th[i],NULL);
    clock_gettime(CLOCK_MONOTONIC,&T1);
    double el=(T1.tv_sec-T0.tv_sec)+(T1.tv_nsec-T0.tv_nsec)/1e9;
    int tw=0,tt=0;for(int i=0;i<NT;i++){tw+=wa[i].ab3_wins;tt+=wa[i].total;}
    double wr=100.*tw/tt;
    printf("  AB:3 wins: %d / %d = %.2f%%\n",tw,tt,wr);
    printf("  Baseline (random): 25.00%%\n");
    printf("  Effect: %+.2f pp\n",wr-25);
    printf("  %.0fs, %.0f g/s\n",el,GAMES/el);
    return 0;
}
