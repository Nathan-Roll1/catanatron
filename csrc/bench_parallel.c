/*
 * Thread-safe parallel benchmark. Zero global mutable state.
 * Each thread owns: its own RngState, SearchCtx, games, maps.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define NUM_THREADS 16
#define GAMES_PER_THREAD 625
#define TOTAL_GAMES (NUM_THREADS * GAMES_PER_THREAD)
#define AB_DEPTH 2
#define NUM_PLAYERS 4

typedef struct {
    int thread_id;
    int wins[4];
    double elapsed;
} ThreadResult;

static void *worker(void *arg) {
    ThreadResult *res = (ThreadResult*)arg;
    memset(res->wins, 0, sizeof(res->wins));

    /* Thread-local state: RNG, search context, maps */
    RngState rng;
    SearchCtx ctx;
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int gi = 0; gi < GAMES_PER_THREAD; gi++) {
        uint64_t seed = (uint64_t)res->thread_id * 1000000 + gi + 400000000ULL;

        /* Build map with thread-local RNG */
        rng_init(&rng, seed);
        CatanMap map;
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);

        /* Init game (uses its own embedded RNG) */
        Game game;
        game_init_with_map(&game, &map, NUM_PLAYERS, colors, seed, 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Action action;
            if (n == 1) {
                action = actions[0];
            } else {
                Color cur = state_current_color(&game.state);
                ctx.depth_counter = 0;
                Game cp; game_copy(&cp, &game);
                SearchResult sr = alphabeta_search(&ctx, &cp, actions, n,
                    AB_DEPTH, -1e30, 1e30, cur, base_value_fn);
                action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
            }
            game_execute(&game, action, actions, &n);
        }

        Color w = game_winning_color(&game);
        if (w != COLOR_NONE) res->wins[(int)w]++;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    res->elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    return NULL;
}

int main(void) {
    printf("=== Thread-Safe Parallel Benchmark ===\n");
    printf("%d threads x %d 4p-AB:%d games = %dk total\n\n",
           NUM_THREADS, GAMES_PER_THREAD, AB_DEPTH, TOTAL_GAMES/1000);

    /* Single-thread baseline */
    struct timespec st0, st1;
    clock_gettime(CLOCK_MONOTONIC, &st0);
    ThreadResult single = {.thread_id = 99};
    worker(&single);
    clock_gettime(CLOCK_MONOTONIC, &st1);
    double single_time = (st1.tv_sec-st0.tv_sec)+(st1.tv_nsec-st0.tv_nsec)/1e9;
    double single_gps = GAMES_PER_THREAD / single_time;
    printf("Single thread: %d games in %.2fs = %.0f g/s\n\n", GAMES_PER_THREAD, single_time, single_gps);

    /* Multi-thread */
    pthread_t threads[NUM_THREADS];
    ThreadResult results[NUM_THREADS];

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < NUM_THREADS; i++) {
        results[i].thread_id = i;
        pthread_create(&threads[i], NULL, worker, &results[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double wall = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    int tw[4] = {0};
    for (int i = 0; i < NUM_THREADS; i++) {
        for (int c = 0; c < 4; c++) tw[c] += results[i].wins[c];
        printf("  Thread %2d: %.1fs\n", i, results[i].elapsed);
    }

    double gps = TOTAL_GAMES / wall;
    printf("\n==========================================\n");
    printf("  %dk 4-player AB:%d games\n", TOTAL_GAMES/1000, AB_DEPTH);
    printf("==========================================\n");
    printf("  Single thread:  %.0f games/sec\n", single_gps);
    printf("  %d threads:      %.0f games/sec\n", NUM_THREADS, gps);
    printf("  Scaling:         %.1fx\n", gps / single_gps);
    printf("  Wall time:       %.2fs\n", wall);
    printf("  Wins: R=%d B=%d O=%d W=%d\n", tw[0], tw[1], tw[2], tw[3]);
    printf("\n  Per hour:  %.1fM games\n", gps * 3600 / 1e6);
    printf("  Per day:   %.0fM games\n", gps * 86400 / 1e6);

    return 0;
}
