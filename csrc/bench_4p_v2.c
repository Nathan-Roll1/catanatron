#include <stdio.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    int N = 500;
    int wins_by_depth[4] = {0}; /* wins_by_depth[d] for d=2,3 */
    int games_by_depth[4] = {0};
    int total_turns = 0;

    clock_t total_start = clock();
    Color all_colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    /* 6 combos for placing 2x AB3 among 4 seats */
    int combos[6][4] = {
        {3,3,2,2}, {3,2,3,2}, {3,2,2,3},
        {2,3,3,2}, {2,3,2,3}, {2,2,3,3}
    };

    for (int gi = 0; gi < N; gi++) {
        int depths[4];
        int combo_idx = gi % 6;
        for (int i = 0; i < 4; i++) {
            depths[i] = combos[combo_idx][i];
            games_by_depth[depths[i]]++;
        }

        CatanMap map;
        rng_seed((uint64_t)(5000 + gi));
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 4, all_colors, (uint64_t)(5000 + gi), 7, false, 10);

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
            int winner_seat = -1;
            for (int i = 0; i < 4; i++)
                if (all_colors[i] == winner) { winner_seat = i; break; }
            wins_by_depth[depths[winner_seat]]++;
        }

        if ((gi + 1) % 50 == 0) {
            double elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
            double ab2_wr = 100.0 * wins_by_depth[2] / (games_by_depth[2] > 0 ? games_by_depth[2] : 1);
            double ab3_wr = 100.0 * wins_by_depth[3] / (games_by_depth[3] > 0 ? games_by_depth[3] : 1);
            printf("  [%3d/%d] AB2=%.1f%% AB3=%.1f%% (%.1fs)\n",
                   gi + 1, N, ab2_wr, ab3_wr, elapsed);
        }
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
    double ab2_wr = 100.0 * wins_by_depth[2] / games_by_depth[2];
    double ab3_wr = 100.0 * wins_by_depth[3] / games_by_depth[3];

    printf("\n==========================================\n");
    printf("  4-Player: 2x AB:2 vs 2x AB:3 -- %d games\n", N);
    printf("==========================================\n");
    printf("  Per-player win rate (baseline = 25.0%%):\n");
    printf("    AB:2 player: %d wins / %d seats = %.1f%%\n",
           wins_by_depth[2], games_by_depth[2], ab2_wr);
    printf("    AB:3 player: %d wins / %d seats = %.1f%%\n",
           wins_by_depth[3], games_by_depth[3], ab3_wr);
    printf("\n  AB:3 advantage: +%.1f pp over baseline\n", ab3_wr - 25.0);
    printf("  AB:2 disadvantage: %.1f pp below baseline\n", 25.0 - ab2_wr);
    printf("\n  Avg turns: %.1f\n", (double)total_turns / N);
    printf("  Total time: %.2fs (%.1f games/sec)\n", total_elapsed, N / total_elapsed);

    return 0;
}
