#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"
#define NT 18
typedef struct { int tid; int wins[4]; } TW;
static void *w(void *a) {
    TW *t=(TW*)a; memset(t->wins,0,16);
    RngState rng; SearchCtx ctx;
    Color c[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    for(int gi=0;gi<56;gi++){
        uint64_t seed=99999ULL*t->tid+gi;
        rng_init(&rng,seed);CatanMap map;build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&rng);
        Game game;game_init_with_map(&game,&map,4,c,seed,7,false,10);
        Action acts[128];int n=generate_playable_actions(&game.state,acts,128);
        while(game_winning_color(&game)==COLOR_NONE&&game.state.num_turns<1000){
            Action a;if(n==1)a=acts[0];
            else{Color cur=state_current_color(&game.state);ctx.depth_counter=0;
                Game cp;game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,2,-1e30,1e30,cur,base_value_fn);
                a=(sr.action.type||sr.action.color)?sr.action:acts[0];}
            game_execute(&game,a,acts,&n);}
        Color wc=game_winning_color(&game);
        if(wc!=COLOR_NONE)t->wins[game.state.color_to_index[(int)wc]]++;
    }
    return NULL;
}
int main(void){
    RngState r;rng_init(&r,0);CatanMap m;build_map(&m,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&r);board_init_static_graph(&m);
    pthread_t th[NT];TW tw[NT];
    for(int i=0;i<NT;i++){tw[i].tid=i;pthread_create(&th[i],NULL,w,&tw[i]);}
    for(int i=0;i<NT;i++)pthread_join(th[i],NULL);
    int tot[4]={0};for(int i=0;i<NT;i++)for(int j=0;j<4;j++)tot[j]+=tw[i].wins[j];
    int s=tot[0]+tot[1]+tot[2]+tot[3];
    printf("P0=%d(%.1f%%) P1=%d(%.1f%%) P2=%d(%.1f%%) P3=%d(%.1f%%) total=%d\n",
        tot[0],100.*tot[0]/s,tot[1],100.*tot[1]/s,tot[2],100.*tot[2]/s,tot[3],100.*tot[3]/s,s);
    return 0;
}
