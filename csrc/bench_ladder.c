#include <stdio.h>
#include <time.h>
#include <string.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    int N = 20;
    int wins[6] = {0};  /* wins[d] for depth d */
    int total_turns = 0;
    Color all_colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    const char *cnames[] = {"RED", "BLUE", "ORG", "WHT"};
    int base_depths[4] = {2, 3, 4, 5};

    clock_t total_start = clock();

    for (int gi = 0; gi < N; gi++) {
        /* Shuffle seat assignments for this game */
        int depths[4];
        memcpy(depths, base_depths, sizeof(base_depths));
        /* Use a separate RNG sequence for shuffling seats */
        rng_seed((uint64_t)(20000 + gi));
        rng_shuffle_int(depths, 4);

        printf("Game %2d: seats [AB%d AB%d AB%d AB%d] ", gi + 1,
               depths[0], depths[1], depths[2], depths[3]);
        fflush(stdout);

        /* Build map with game seed */
        CatanMap map;
        rng_seed((uint64_t)(30000 + gi));
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 4, all_colors, (uint64_t)(30000 + gi), 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        clock_t game_start = clock();

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

        double game_time = (double)(clock() - game_start) / CLOCKS_PER_SEC;
        Color winner = game_winning_color(&game);
        total_turns += game.state.num_turns;

        if (winner != COLOR_NONE) {
            int ws = -1;
            for (int i = 0; i < 4; i++)
                if (all_colors[i] == winner) { ws = i; break; }
            wins[depths[ws]]++;
            printf("-> AB%d(%s) wins  t=%d  %.2fs\n",
                   depths[ws], cnames[ws], game.state.num_turns, game_time);
        } else {
            printf("-> DRAW  t=%d  %.2fs\n", game.state.num_turns, game_time);
        }
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;

    printf("\n==========================================\n");
    printf("  Ladder: AB2 vs AB3 vs AB4 vs AB5\n");
    printf("  %d games, randomized seating\n", N);
    printf("==========================================\n");
    for (int d = 2; d <= 5; d++) {
        printf("  AB:%d  %2d / %d wins  (%5.1f%%)\n", d, wins[d], N, 100.0 * wins[d] / N);
    }
    printf("\n  Avg turns: %.1f\n", (double)total_turns / N);
    printf("  Total time: %.2fs (%.1f games/sec)\n", total_elapsed, N / total_elapsed);

    return 0;
}
