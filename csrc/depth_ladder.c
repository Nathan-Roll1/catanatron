/*
 * 100k 4-player games: one seat each for AB:2, AB:3, AB:4, AB:5.
 * Randomized seating via rotation. Report per-depth win rates.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

#define NT 18
#define GAMES 99972  /* divisible by NT and 4 */
#define GPT (GAMES/NT)

static int g_depths[4]; /* per-seat depth, rotated per game */

static double depth_eval(Game *g, Color c) {
    return base_value_fn(g, c);
}

typedef struct {
    int tid, n;
    uint64_t sb;
    int wins[4]; /* wins by seat index */
} WA;

static void *worker(void *a) {
    WA *w = (WA*)a;
    memset(w->wins, 0, sizeof(w->wins));
    RngState rng;
    SearchCtx *ctx = (SearchCtx*)malloc(sizeof(SearchCtx));
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    for (int gi = 0; gi < w->n; gi++) {
        /* Rotate depth assignment: seat s gets depth g_depths[(gi+s)%4] */
        int depths[4];
        for (int s = 0; s < 4; s++) depths[s] = g_depths[(gi + s) % 4];

        uint64_t seed = w->sb + (uint64_t)w->tid * 100000ULL + gi;
        rng_init(&rng, seed);
        CatanMap map; build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
        Game game; game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);

        Action acts[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, acts, MAX_ACTIONS);
        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Action a;
            if (n == 1) { a = acts[0]; }
            else {
                Color cur = state_current_color(&game.state);
                int seat = -1;
                for (int i = 0; i < 4; i++) if (colors[i] == cur) { seat = i; break; }
                ctx->depth_counter = 0;
                Game cp; game_copy(&cp, &game);
                SearchResult sr = alphabeta_search(ctx, &cp, acts, n,
                    depths[seat], -1e30, 1e30, cur, depth_eval);
                a = (sr.action.type || sr.action.color) ? sr.action : acts[0];
            }
            game_execute(&game, a, acts, &n);
        }
        Color w2 = game_winning_color(&game);
        if (w2 != COLOR_NONE) {
            int ws = game.state.color_to_index[(int)w2];
            /* Map back to original depth: seat ws had depth depths[ws],
               which came from g_depths[(gi+ws)%4]. Find which depth index that is. */
            int depth_idx = (gi + ws) % 4;
            w->wins[depth_idx]++;
        }
    }
    return NULL;
}

int main(void) {
    srand(time(NULL));
    RngState tmp; rng_init(&tmp, 0);
    CatanMap tm; build_map(&tm, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp);
    board_init_static_graph(&tm);

    g_depths[0] = 2; g_depths[1] = 2; g_depths[2] = 3; g_depths[3] = 3;

    printf("=== Depth Ladder: AB:2 vs AB:3 vs AB:4 vs AB:5 ===\n");
    printf("%dk 4-player games, %d threads, rotated seating\n\n", GAMES/1000, NT);

    uint64_t seed = (uint64_t)time(NULL) * 1000003ULL;
    pthread_t th[NT]; WA wa[NT];
    for (int i = 0; i < NT; i++) { wa[i].tid = i; wa[i].n = GPT; wa[i].sb = seed; }

    struct timespec T0, T1;
    clock_gettime(CLOCK_MONOTONIC, &T0);

    for (int i = 0; i < NT; i++) pthread_create(&th[i], NULL, worker, &wa[i]);
    for (int i = 0; i < NT; i++) pthread_join(th[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &T1);
    double el = (T1.tv_sec-T0.tv_sec) + (T1.tv_nsec-T0.tv_nsec) / 1e9;

    int total_wins[4] = {0};
    for (int i = 0; i < NT; i++)
        for (int d = 0; d < 4; d++)
            total_wins[d] += wa[i].wins[d];

    int total = total_wins[0] + total_wins[1] + total_wins[2] + total_wins[3];

    printf("  Depth  Wins    Win%%     vs 25%%\n");
    printf("  -----  ------  ------   ------\n");
    for (int d = 0; d < 4; d++) {
        double wr = 100.0 * total_wins[d] / GAMES;
        double se = sqrt(wr/100*(1-wr/100)/GAMES) * 100;
        printf("  AB:%-2d  %6d  %5.1f%%   %+.1f pp  (+/-%.1f)\n",
               g_depths[d], total_wins[d], wr, wr - 25, 1.96 * se);
    }
    printf("\n  Total: %d wins / %d games\n", total, GAMES);
    printf("  Time: %.0fs (%.0f g/s)\n", el, GAMES / el);

    return 0;
}
