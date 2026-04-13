#include <stdio.h>
#include <time.h>
#include <string.h>
#include "game.h"
#include "search.h"
#include "value.h"
#include "actions.h"
#include "rng.h"

static long copy_ns = 0, value_ns = 0, actions_ns = 0, search_ns = 0;
static int copy_count = 0, value_count = 0, actions_count = 0;

static struct timespec ts_now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t;
}
static long ts_diff_ns(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000000000L + (b.tv_nsec - a.tv_nsec);
}

/* Benchmark: state_copy */
void bench_copy(Game *g, int n) {
    struct timespec t0 = ts_now();
    Game tmp;
    for (int i = 0; i < n; i++) game_copy(&tmp, g);
    struct timespec t1 = ts_now();
    copy_ns = ts_diff_ns(t0, t1);
    copy_count = n;
}

/* Benchmark: value function */
void bench_value(Game *g, int n) {
    struct timespec t0 = ts_now();
    volatile double v = 0;
    for (int i = 0; i < n; i++) v += base_value_fn(g, g->state.colors[0]);
    struct timespec t1 = ts_now();
    value_ns = ts_diff_ns(t0, t1);
    value_count = n;
}

/* Benchmark: action generation */
void bench_actions(Game *g, int n) {
    Action buf[MAX_ACTIONS];
    struct timespec t0 = ts_now();
    volatile int cnt = 0;
    for (int i = 0; i < n; i++) cnt += generate_playable_actions(&g->state, buf, MAX_ACTIONS);
    struct timespec t1 = ts_now();
    actions_ns = ts_diff_ns(t0, t1);
    actions_count = n;
}

int main(void) {
    /* Set up a mid-game state */
    Color colors[] = {COLOR_RED, COLOR_BLUE};
    Game g;
    game_init(&g, 2, colors, 42, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, 7, false, 10);
    /* Play ~30 ticks with random to get to mid-game */
    for (int i = 0; i < 80 && game_winning_color(&g) == COLOR_NONE; i++) {
        Action a = g.playable_actions[rng_choice_index(g.num_playable_actions)];
        game_execute(&g, a);
    }

    int N = 100000;
    bench_copy(&g, N);
    bench_value(&g, N);
    bench_actions(&g, N);

    printf("=== Micro-benchmarks (%d iterations) ===\n", N);
    printf("  game_copy:              %7.1f ns/call  (%d calls)\n", (double)copy_ns/copy_count, copy_count);
    printf("  base_value_fn:          %7.1f ns/call  (%d calls)\n", (double)value_ns/value_count, value_count);
    printf("  generate_playable_actions: %7.1f ns/call  (%d calls)\n", (double)actions_ns/actions_count, actions_count);

    /* Full game benchmark */
    printf("\n=== Full game benchmarks ===\n");

    /* Random */
    struct timespec t0 = ts_now();
    int RN = 10000;
    for (int s = 0; s < RN; s++) {
        Game g2;
        game_init(&g2, 2, colors, s, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, 7, false, 10);
        while (game_winning_color(&g2) == COLOR_NONE && g2.state.num_turns < TURNS_LIMIT) {
            Action a = g2.playable_actions[rng_choice_index(g2.num_playable_actions)];
            game_execute(&g2, a);
        }
    }
    struct timespec t1 = ts_now();
    double r_elapsed = ts_diff_ns(t0, t1) / 1e9;
    printf("  R vs R:   %d games in %.3fs (%.0f games/sec, %.3f ms/game)\n",
           RN, r_elapsed, RN/r_elapsed, r_elapsed/RN*1000);

    /* AB:2 vs AB:2 */
    t0 = ts_now();
    int AN = 100;
    for (int s = 0; s < AN; s++) {
        Game g2;
        game_init(&g2, 2, colors, s, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, 7, false, 10);
        while (game_winning_color(&g2) == COLOR_NONE && g2.state.num_turns < TURNS_LIMIT) {
            if (g2.num_playable_actions == 1) {
                game_execute(&g2, g2.playable_actions[0]);
                continue;
            }
            Color cur = state_current_color(&g2.state);
            Game copy;
            game_copy(&copy, &g2);
            double deadline = (double)clock()/CLOCKS_PER_SEC + 120.0;
            SearchResult sr = alphabeta_search(&copy, 2, -1e30, 1e30, deadline, cur);
            game_execute(&g2, sr.action.type != 0 ? sr.action : g2.playable_actions[0]);
        }
    }
    t1 = ts_now();
    double ab_elapsed = ts_diff_ns(t0, t1) / 1e9;
    printf("  AB:2 vs AB:2: %d games in %.3fs (%.1f games/sec, %.1f ms/game)\n",
           AN, ab_elapsed, AN/ab_elapsed, ab_elapsed/AN*1000);

    printf("\n  sizeof(State): %zu bytes\n", sizeof(State));
    printf("  sizeof(Game):  %zu bytes\n", sizeof(Game));

    return 0;
}
