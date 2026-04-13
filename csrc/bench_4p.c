#include <stdio.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    int N = 200;
    int ab2_wins = 0, ab3_wins = 0, draws = 0;
    int total_turns = 0;

    /* Track wins by seat position */
    int ab2_wins_by_seat[4] = {0};
    int ab3_wins_by_seat[4] = {0};
    int ab2_games_by_seat[4] = {0};
    int ab3_games_by_seat[4] = {0};

    clock_t total_start = clock();

    Color all_colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    for (int gi = 0; gi < N; gi++) {
        /* Randomize which seats are AB2 vs AB3.
         * Pick 2 of 4 seats for AB3, rest are AB2.
         * Use game index to deterministically vary assignments. */
        int depths[4];
        /* 6 possible combinations of 2-of-4: use gi % 6 */
        int combos[6][4] = {
            {3,3,2,2}, {3,2,3,2}, {3,2,2,3},
            {2,3,3,2}, {2,3,2,3}, {2,2,3,3}
        };
        int combo_idx = gi % 6;
        for (int i = 0; i < 4; i++) depths[i] = combos[combo_idx][i];

        for (int i = 0; i < 4; i++) {
            if (depths[i] == 2) ab2_games_by_seat[i]++;
            else ab3_games_by_seat[i]++;
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
            /* Find seat of current player */
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

            if (depths[winner_seat] == 2) {
                ab2_wins++;
                ab2_wins_by_seat[winner_seat]++;
            } else {
                ab3_wins++;
                ab3_wins_by_seat[winner_seat]++;
            }
        } else {
            draws++;
        }

        if ((gi + 1) % 20 == 0) {
            double elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
            printf("  [%3d/%d] AB2=%d AB3=%d draws=%d (%.1fs)\n",
                   gi + 1, N, ab2_wins, ab3_wins, draws, elapsed);
        }
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;
    const char *names[] = {"RED", "BLUE", "ORANGE", "WHITE"};

    printf("\n==========================================\n");
    printf("  4-Player: 2x AB:2 vs 2x AB:3 -- %d games\n", N);
    printf("==========================================\n");
    printf("  AB:2 wins: %3d / %d  (%.1f%%)  -- baseline 50%%\n",
           ab2_wins, N, 100.0 * ab2_wins / N);
    printf("  AB:3 wins: %3d / %d  (%.1f%%)  -- baseline 50%%\n",
           ab3_wins, N, 100.0 * ab3_wins / N);
    printf("  Draws:     %3d\n", draws);

    /* Per-type win rate (each type has 2 players, so baseline is 25% per player = 50% per type) */
    int ab2_total_seats = 0, ab3_total_seats = 0;
    for (int i = 0; i < 4; i++) {
        ab2_total_seats += ab2_games_by_seat[i];
        ab3_total_seats += ab3_games_by_seat[i];
    }
    printf("\n  Per-player baseline: 25%% win rate\n");
    printf("  AB:2 avg win rate per player: %.1f%%\n",
           100.0 * ab2_wins / ab2_total_seats * 2);
    printf("  AB:3 avg win rate per player: %.1f%%\n",
           100.0 * ab3_wins / ab3_total_seats * 2);

    printf("\n  By seat position:\n");
    for (int i = 0; i < 4; i++) {
        printf("    Seat %d (%s): AB2 %d/%d wins, AB3 %d/%d wins\n",
               i, names[i],
               ab2_wins_by_seat[i], ab2_games_by_seat[i],
               ab3_wins_by_seat[i], ab3_games_by_seat[i]);
    }

    printf("\n  Avg turns: %.1f\n", (double)total_turns / N);
    printf("  Total time: %.2fs (%.1f games/sec)\n", total_elapsed, N / total_elapsed);

    return 0;
}
