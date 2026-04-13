#include <stdio.h>
#include <time.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"

int main(void) {
    int N = 100;
    int ab2_wins = 0, ab3_wins = 0, draws = 0;
    int ab2_as_red = 0, ab3_as_red = 0;
    int ab2_wins_as_red = 0, ab2_wins_as_blue = 0;
    int ab3_wins_as_red = 0, ab3_wins_as_blue = 0;

    clock_t total_start = clock();

    for (int gi = 0; gi < N; gi++) {
        /* Alternate positions: even games AB2=RED, odd games AB3=RED */
        int ab2_depth, ab3_depth;
        Color ab2_color, ab3_color;
        if (gi % 2 == 0) {
            ab2_color = COLOR_RED;  ab3_color = COLOR_BLUE;
            ab2_depth = 2; ab3_depth = 3;
            ab2_as_red++;
        } else {
            ab2_color = COLOR_BLUE; ab3_color = COLOR_RED;
            ab2_depth = 2; ab3_depth = 3;
            ab3_as_red++;
        }

        Color colors[] = {COLOR_RED, COLOR_BLUE};
        CatanMap map;
        rng_seed((uint64_t)(1000 + gi));
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

        Game game;
        game_init_with_map(&game, &map, 2, colors, (uint64_t)(1000 + gi), 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Color cur = state_current_color(&game.state);
            int depth = (cur == ab2_color) ? ab2_depth : ab3_depth;

            Action action;
            if (n == 1) {
                action = actions[0];
            } else {
                double deadline = (double)clock()/CLOCKS_PER_SEC + 120.0;
                Game copy;
                game_copy(&copy, &game);
                SearchResult sr = alphabeta_search(&copy, actions, n,
                    depth, -1e30, 1e30, deadline, cur);
                action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
            }
            game_execute(&game, action, actions, &n);
        }

        Color winner = game_winning_color(&game);
        if (winner == ab2_color) {
            ab2_wins++;
            if (ab2_color == COLOR_RED) ab2_wins_as_red++;
            else ab2_wins_as_blue++;
        } else if (winner == ab3_color) {
            ab3_wins++;
            if (ab3_color == COLOR_RED) ab3_wins_as_red++;
            else ab3_wins_as_blue++;
        } else {
            draws++;
        }

        printf("Game %3d: AB%d(%s) vs AB%d(%s) -> winner=%s turns=%d\n",
               gi + 1,
               (ab2_color == COLOR_RED) ? 2 : 3,
               "RED",
               (ab2_color == COLOR_RED) ? 3 : 2,
               "BLUE",
               winner == COLOR_NONE ? "DRAW" :
               (winner == COLOR_RED ? "RED" : "BLUE"),
               game.state.num_turns);
    }

    double total_elapsed = (double)(clock() - total_start) / CLOCKS_PER_SEC;

    printf("\n==============================\n");
    printf("  AB:2 vs AB:3 -- %d games\n", N);
    printf("==============================\n");
    printf("  AB:2 wins: %d (%.1f%%)\n", ab2_wins, 100.0 * ab2_wins / N);
    printf("  AB:3 wins: %d (%.1f%%)\n", ab3_wins, 100.0 * ab3_wins / N);
    printf("  Draws:     %d\n", draws);
    printf("\n  Position breakdown:\n");
    printf("    AB:2 as RED: %d/%d wins\n", ab2_wins_as_red, ab2_as_red);
    printf("    AB:2 as BLUE: %d/%d wins\n", ab2_wins_as_blue, N - ab2_as_red);
    printf("    AB:3 as RED: %d/%d wins\n", ab3_wins_as_red, ab3_as_red);
    printf("    AB:3 as BLUE: %d/%d wins\n", ab3_wins_as_blue, N - ab3_as_red);
    printf("\n  Total time: %.2fs (%.1f games/sec)\n", total_elapsed, N / total_elapsed);

    return 0;
}
