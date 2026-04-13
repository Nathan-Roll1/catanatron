#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"

typedef enum { PT_RANDOM, PT_AB } PlayerType;
typedef struct { PlayerType type; int depth; } PlayerConfig;

static void parse_players(const char *str, PlayerConfig *configs, int *num_players) {
    *num_players = 0;
    char buf[256]; strncpy(buf, str, sizeof(buf)-1); buf[sizeof(buf)-1] = 0;
    char *token = strtok(buf, ",");
    while (token && *num_players < MAX_PLAYERS) {
        int idx = *num_players;
        if (strncmp(token, "AB", 2) == 0) {
            configs[idx].type = PT_AB;
            configs[idx].depth = 2;
            if (token[2] == ':') configs[idx].depth = atoi(token + 3);
        } else { configs[idx].type = PT_RANDOM; configs[idx].depth = 0; }
        (*num_players)++; token = strtok(NULL, ",");
    }
}

int main(int argc, char **argv) {
    int num_games = 5; char player_str[256] = "R,R"; bool quiet = false;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--num=", 6) == 0) num_games = atoi(argv[i] + 6);
        else if (strncmp(argv[i], "-n", 2) == 0 && argv[i][2] == 0 && i+1 < argc) num_games = atoi(argv[++i]);
        else if (strncmp(argv[i], "--players=", 10) == 0) strncpy(player_str, argv[i] + 10, sizeof(player_str)-1);
        else if (strcmp(argv[i], "--quiet") == 0) quiet = true;
    }
    PlayerConfig configs[MAX_PLAYERS]; int num_players;
    parse_players(player_str, configs, &num_players);
    Color colors[MAX_PLAYERS] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
    const char *color_names[] = {"RED", "BLUE", "ORANGE", "WHITE"};

    int wins[MAX_PLAYERS] = {0}; int total_turns = 0; double total_time = 0;
    SearchCtx ctx;
    RngState map_rng;

    for (int gi = 0; gi < num_games; gi++) {
        rng_init(&map_rng, (uint64_t)gi);
        CatanMap map;
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &map_rng);
        Game game;
        game_init_with_map(&game, &map, num_players, colors, (uint64_t)gi, 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);

        while (game_winning_color(&game) == COLOR_NONE && game.state.num_turns < TURNS_LIMIT) {
            Color cur = state_current_color(&game.state);
            int pi = -1;
            for (int p = 0; p < num_players; p++) if (colors[p] == cur) { pi = p; break; }
            if (pi < 0) pi = 0;

            Action action;
            if (configs[pi].type == PT_AB && n > 1) {
                ctx.depth_counter = 0;
                Game copy; game_copy(&copy, &game);
                SearchResult sr = alphabeta_search(&ctx, &copy, actions, n,
                    configs[pi].depth, -1e30, 1e30, cur, base_value_fn);
                action = (sr.action.type != 0 || sr.action.color != 0) ? sr.action : actions[0];
            } else {
                action = actions[rng_choice_index(&game.rng, n)];
            }
            game_execute(&game, action, actions, &n);
        }

        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
        total_time += elapsed;
        Color winner = game_winning_color(&game);
        total_turns += game.state.num_turns;
        if (winner != COLOR_NONE) wins[(int)winner]++;
        if (!quiet) printf("Game %d: winner=%s turns=%d (%.3fs)\n", gi+1,
               winner==COLOR_NONE?"NONE":color_names[(int)winner], game.state.num_turns, elapsed);
    }
    printf("\n=== Results (%d games) ===\n", num_games);
    for (int i = 0; i < num_players; i++) {
        const char *ts = configs[i].type == PT_AB ? "AB" : "R";
        printf("  %s(%s): %d wins (%.1f%%)\n", color_names[i], ts, wins[i], 100.0*wins[i]/num_games);
    }
    printf("  Avg turns: %.1f\n", (double)total_turns / num_games);
    printf("  Total time: %.3fs (%.1f games/sec)\n", total_time, num_games/(total_time>0?total_time:1e-9));
    return 0;
}
