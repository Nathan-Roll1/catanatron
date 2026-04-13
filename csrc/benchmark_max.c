#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

#define NUM_THREADS 16
#define GAMES_PER_THREAD 625
#define TOTAL_GAMES (NUM_THREADS * GAMES_PER_THREAD)

typedef struct {
    int thread_id;
    int n_games;
    Game *games;       /* pre-initialized games */
    int wins[4];
    double elapsed;
} ThreadWork;

static void *worker(void *arg) {
    ThreadWork *w = (ThreadWork *)arg;
    memset(w->wins, 0, sizeof(w->wins));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gi = 0; gi < w->n_games; gi++) {
        Game *game = &w->games[gi];

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game->state, actions, MAX_ACTIONS);

        while (game_winning_color(game) == COLOR_NONE && game->state.num_turns < TURNS_LIMIT) {
            Action action;
            if (n == 1) {
                action = actions[0];
            } else {
                Color cur = state_current_color(&game->state);
                double dl = (double)clock() / CLOCKS_PER_SEC + 120.0;
                Game cp;
                game_copy(&cp, game);
                SearchResult sr = alphabeta_search(&cp, actions, n,
                    2, -1e30, 1e30, dl, cur);
                action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
            }
            game_execute(game, action, actions, &n);
        }

        Color winner = game_winning_color(game);
        if (winner != COLOR_NONE) w->wins[(int)winner]++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    w->elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    return NULL;
}

int main(void) {
    printf("=== Maximum Throughput: AB:2 vs AB:2 ===\n");
    printf("Apple M5 Max, 18 cores\n");
    printf("%d threads x %d games = %dk total\n\n", NUM_THREADS, GAMES_PER_THREAD, TOTAL_GAMES/1000);

    /* Single-threaded baseline */
    printf("Single-threaded baseline (500 games)...\n");
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        Color colors[2] = {COLOR_RED, COLOR_BLUE};
        int bwins[4] = {0};
        for (int gi = 0; gi < 500; gi++) {
            CatanMap map;
            rng_seed((uint64_t)(gi + 99000000));
            build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);
            Game game;
            game_init_with_map(&game, &map, 2, colors, (uint64_t)(gi + 99000000), 7, false, 10);
            Action actions[MAX_ACTIONS];
            int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);
            while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
                Action action;
                if (n == 1) { action = actions[0]; }
                else {
                    Color cur = state_current_color(&game.state);
                    double dl = (double)clock()/CLOCKS_PER_SEC + 120.0;
                    Game cp; game_copy(&cp, &game);
                    SearchResult sr = alphabeta_search(&cp, actions, n, 2, -1e30, 1e30, dl, cur);
                    action = (sr.action.type!=0||sr.action.color!=0) ? sr.action : actions[0];
                }
                game_execute(&game, action, actions, &n);
            }
            Color w = game_winning_color(&game);
            if (w != COLOR_NONE) bwins[(int)w]++;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double bt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        printf("  500 games in %.2fs = %.0f games/sec\n\n", bt, 500.0/bt);
    }

    /* Pre-initialize all games sequentially (RNG is global, must be serial) */
    printf("Initializing %dk games...\n", TOTAL_GAMES/1000);
    Game *all_games = (Game *)malloc(TOTAL_GAMES * sizeof(Game));
    Color colors[2] = {COLOR_RED, COLOR_BLUE};

    struct timespec init0, init1;
    clock_gettime(CLOCK_MONOTONIC, &init0);
    for (int i = 0; i < TOTAL_GAMES; i++) {
        CatanMap map;
        rng_seed((uint64_t)(i + 200000000));
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);
        game_init_with_map(&all_games[i], &map, 2, colors, (uint64_t)(i + 200000000), 7, false, 10);
    }
    clock_gettime(CLOCK_MONOTONIC, &init1);
    double init_time = (init1.tv_sec-init0.tv_sec)+(init1.tv_nsec-init0.tv_nsec)/1e9;
    printf("  Init: %.2fs\n\n", init_time);

    /* Run in parallel */
    printf("Running %d threads...\n", NUM_THREADS);
    pthread_t threads[NUM_THREADS];
    ThreadWork work[NUM_THREADS];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < NUM_THREADS; i++) {
        work[i].thread_id = i;
        work[i].n_games = GAMES_PER_THREAD;
        work[i].games = &all_games[i * GAMES_PER_THREAD];
        pthread_create(&threads[i], NULL, worker, &work[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    int total_red = 0, total_blue = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        total_red += work[i].wins[COLOR_RED];
        total_blue += work[i].wins[COLOR_BLUE];
        printf("  Thread %2d: %.1fs, %d+%d wins\n", i, work[i].elapsed,
               work[i].wins[COLOR_RED], work[i].wins[COLOR_BLUE]);
    }

    double gps = TOTAL_GAMES / wall;
    printf("\n==========================================\n");
    printf("  %dk AB:2 vs AB:2 games\n", TOTAL_GAMES/1000);
    printf("==========================================\n");
    printf("  Wall time:     %.2f seconds\n", wall);
    printf("  Throughput:    %.0f games/sec\n", gps);
    printf("  RED: %d  BLUE: %d\n", total_red, total_blue);
    printf("\n  Projections:\n");
    printf("    1 minute:    %dk games\n", (int)(gps*60/1000));
    printf("    1 hour:      %.1fM games\n", gps*3600/1e6);
    printf("    1 day:       %.0fM games\n", gps*86400/1e6);

    free(all_games);
    return 0;
}
