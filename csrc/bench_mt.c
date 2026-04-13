#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define NUM_THREADS 16
#define GAMES_PER_THREAD 625

typedef struct {
    int n;
    Game *games;
    CatanMap *maps;
    int wins[4];
} Work;

static void *worker(void *arg) {
    Work *w = (Work*)arg;
    memset(w->wins,0,sizeof(w->wins));
    for (int gi=0;gi<w->n;gi++){
        Game *g=&w->games[gi];
        Action acts[MAX_ACTIONS];
        int n=generate_playable_actions(&g->state,acts,MAX_ACTIONS);
        while(game_winning_color(g)==COLOR_NONE&&g->state.num_turns<TURNS_LIMIT){
            Action a;
            if(n==1){a=acts[0];}else{
                Color cur=state_current_color(&g->state);
                Game cp;game_copy(&cp,g);
                SearchResult sr=alphabeta_search(&cp,acts,n,2,-1e30,1e30,9e99,cur);
                a=(sr.action.type!=0||sr.action.color!=0)?sr.action:acts[0];}
            game_execute(g,a,acts,&n);}
        Color winner=game_winning_color(g);
        if(winner!=COLOR_NONE) w->wins[(int)winner]++;
    }
    return NULL;
}

int main(void) {
    int total = NUM_THREADS * GAMES_PER_THREAD;
    printf("=== Max Throughput: %dk AB:2 games, %d threads ===\n\n", total/1000, NUM_THREADS);

    /* Allocate and initialize games sequentially */
    CatanMap *maps = malloc(total*sizeof(CatanMap));
    Game *games = malloc(total*sizeof(Game));
    Color colors[2]={COLOR_RED,COLOR_BLUE};
    
    printf("Initializing...\n");
    for(int i=0;i<total;i++){
        rng_seed((uint64_t)(i+300000000));
        build_map(&maps[i],MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        game_init_with_map(&games[i],&maps[i],2,colors,(uint64_t)(i+300000000),7,false,10);
    }

    /* Single-thread baseline */
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    Work single={.n=1000,.games=games,.maps=maps};
    worker(&single);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double st=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("Single-thread: 1000 games in %.2fs = %.0f g/s\n\n", st, 1000/st);

    /* Re-init the 1000 games consumed */
    for(int i=0;i<1000;i++){
        rng_seed((uint64_t)(i+300000000));
        build_map(&maps[i],MAP_BASE,NPLACE_OFFICIAL_SPIRAL);
        game_init_with_map(&games[i],&maps[i],2,colors,(uint64_t)(i+300000000),7,false,10);
    }

    /* Multi-thread */
    printf("Running %d threads x %d games...\n", NUM_THREADS, GAMES_PER_THREAD);
    pthread_t threads[NUM_THREADS];
    Work work[NUM_THREADS];
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<NUM_THREADS;i++){
        work[i].n=GAMES_PER_THREAD;
        work[i].games=&games[i*GAMES_PER_THREAD];
        work[i].maps=&maps[i*GAMES_PER_THREAD];
        pthread_create(&threads[i],NULL,worker,&work[i]);
    }
    for(int i=0;i<NUM_THREADS;i++) pthread_join(threads[i],NULL);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double mt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    int tr=0,tb=0;
    for(int i=0;i<NUM_THREADS;i++){tr+=work[i].wins[0];tb+=work[i].wins[1];}
    double gps=total/mt;

    printf("\n==========================================\n");
    printf("  %dk AB:2 vs AB:2\n", total/1000);
    printf("==========================================\n");
    printf("  Single-thread: %.0f games/sec\n", 1000/st);
    printf("  %d threads:     %.0f games/sec\n", NUM_THREADS, gps);
    printf("  Scaling:        %.1fx\n", gps/(1000/st));
    printf("  Wall time:      %.2fs\n", mt);
    printf("  RED=%d BLUE=%d\n", tr, tb);
    printf("\n  Per hour: %.1fM games\n", gps*3600/1e6);

    free(maps); free(games);
    return 0;
}
