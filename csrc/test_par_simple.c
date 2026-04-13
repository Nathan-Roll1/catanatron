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

static void *worker(void *arg) {
    TW *tw = (TW*)arg;
    memset(tw->wins, 0, sizeof(tw->wins));
    RngState rng; SearchCtx ctx;
    Color colors[4]={COLOR_RED,COLOR_BLUE,COLOR_ORANGE,COLOR_WHITE};
    for (int gi = 0; gi < 50; gi++) {
        uint64_t seed = (uint64_t)tw->tid * 10000 + gi;
        rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);
        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        while (game_winning_color(&game)==COLOR_NONE && game.state.num_turns<TURNS_LIMIT) {
            Action a;
            if (n==1) a=acts[0];
            else { Color cur=state_current_color(&game.state);
                ctx.depth_counter=0; Game cp; game_copy(&cp,&game);
                SearchResult sr=alphabeta_search(&ctx,&cp,acts,n,2,-1e30,1e30,cur,base_value_fn);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0]; }
            game_execute(&game,a,acts,&n);
        }
        Color w=game_winning_color(&game);
        if(w!=COLOR_NONE) tw->wins[game.state.color_to_index[(int)w]]++;
    }
    return NULL;
}

int main(void) {
    printf("Init static graph...\n");
    RngState rng; rng_init(&rng,0);
    CatanMap map; build_map(&map,MAP_BASE,NPLACE_OFFICIAL_SPIRAL,&rng);
    board_init_static_graph(&map);
    printf("Running %d threads x 50 4p AB:2 games...\n", NT);

    pthread_t threads[NT]; TW work[NT];
    for(int i=0;i<NT;i++) { work[i].tid=i; pthread_create(&threads[i],NULL,worker,&work[i]); }
    for(int i=0;i<NT;i++) pthread_join(threads[i],NULL);

    int total=0;
    for(int i=0;i<NT;i++) for(int v=0;v<4;v++) total+=work[i].wins[v];
    printf("Total wins: %d / %d games\n", total, NT*50);
    printf("Success!\n");
    return 0;
}
