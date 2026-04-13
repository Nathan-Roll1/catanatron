#include <stdio.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    int N = 100;
    int wins[5] = {0};
    int games[5] = {0};
    int total_turns = 0;
    clock_t total_start = clock();
    Color all_colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    int combos[6][4] = {
        {4,4,3,3}, {4,3,4,3}, {4,3,3,4},
        {3,4,4,3}, {3,4,3,4}, {3,3,4,4}
    };

    for (int gi = 0; gi < N; gi++) {
        int depths[4];
        for (int i = 0; i < 4; i++) {
            depths[i] = combos[gi % 6][i];
            games[depths[i]]++;
        }

        CatanMap map;
        rng_seed((uint64_t)(9000 + gi));
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 4, all_colors, (uint64_t)(9000 + gi), 7, false, 10);

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
                double deadline = (double)clock()/CLOCKS_PER_SEC + 120.0;
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

        if ((gi + 1) % 10 == 0) {
            double elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
            double r3 = 100.0 * wins[3] / (games[3] > 0 ? games[3] : 1);
            double r4 = 100.0 * wins[4] / (games[4] > 0 ? games[4] : 1);
            printf("  [%3d/%d] AB3=%.1f%% AB4=%.1f%% (%.1fs)\n",
                   gi + 1, N, r3, r4, elapsed);
        }
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;

    printf("\n==========================================\n");
    printf("  4-Player: 2x AB:3 vs 2x AB:4 -- %d games\n", N);
    printf("==========================================\n");
    printf("  Per-player win rate (baseline = 25.0%%):\n");
    printf("    AB:3 player: %d wins / %d seats = %.1f%%\n",
           wins[3], games[3], 100.0 * wins[3] / games[3]);
    printf("    AB:4 player: %d wins / %d seats = %.1f%%\n",
           wins[4], games[4], 100.0 * wins[4] / games[4]);
    printf("\n  Avg turns: %.1f\n", (double)total_turns / N);
    printf("  Total time: %.2fs (%.1f games/sec)\n", total_elapsed, N / total_elapsed);

    return 0;
}
