#include <stdio.h>
#include <time.h>
#include <string.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    int N = 100;
    int wins[6] = {0};
    int total_turns = 0;
    Color all_colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    int base_depths[4] = {2, 3, 4, 5};

    clock_t total_start = clock();

    for (int gi = 0; gi < N; gi++) {
        int depths[4];
        memcpy(depths, base_depths, sizeof(base_depths));
        rng_seed((uint64_t)(20000 + gi));
        rng_shuffle_int(depths, 4);

        CatanMap map;
        rng_seed((uint64_t)(30000 + gi));
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 4, all_colors, (uint64_t)(30000 + gi), 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Color cur = state_current_color(&game.state);
            int seat = -1;
            for (int i = 0; i < 4; i++)
                if (all_colors[i] == cur) { seat = i; break; }

            Action action;
            if (n == 1) {
                action = actions[0];
            } else {
                double deadline = (double)clock()/CLOCKS_PER_SEC + 180.0;
                Game copy;
                game_copy(&copy, &game);
                SearchResult sr = alphabeta_search(&copy, actions, n,
                    depths[seat], -1e30, 1e30, deadline, cur);
                action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
            }
            game_execute(&game, action, actions, &n);
        }

        Color winner = game_winning_color(&game);
        total_turns += game.state.num_turns;
        if (winner != COLOR_NONE) {
            int ws = -1;
            for (int i = 0; i < 4; i++)
                if (all_colors[i] == winner) { ws = i; break; }
            wins[depths[ws]]++;
        }

        if ((gi + 1) % 25 == 0) {
            double elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
            printf("  [%3d/%d] AB2=%d AB3=%d AB4=%d AB5=%d  (%.1fs)\n",
                   gi + 1, N, wins[2], wins[3], wins[4], wins[5], elapsed);
        }
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;

    printf("\n==========================================\n");
    printf("  Ladder: AB2 vs AB3 vs AB4 vs AB5\n");
    printf("  %d games, randomized seating\n", N);
    printf("==========================================\n");
    for (int d = 2; d <= 5; d++) {
        char bar[51] = {0};
        int len = (int)(wins[d] * 50.0 / N * 4);
        if (len > 50) len = 50;
        for (int i = 0; i < len; i++) bar[i] = '#';
        printf("  AB:%d  %3d wins  (%5.1f%%)  %s\n", d, wins[d], 100.0 * wins[d] / N, bar);
    }
    printf("\n  Avg turns: %.1f\n", (double)total_turns / N);
    printf("  Total time: %.1fs (%.1f games/sec)\n", total_elapsed, N / total_elapsed);

    return 0;
}
