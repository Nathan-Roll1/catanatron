#include "game.h"
#include "actions.h"
#include "apply_action.h"
#include <string.h>

void game_init_with_map(Game *g, CatanMap *map, int num_players, Color colors[],
                        uint64_t seed, int discard_limit, bool friendly_robber,
                        int vps_to_win) {
    g->seed = seed;
    g->vps_to_win = vps_to_win;
    g->map = map;
    rng_init(&g->rng, seed);
    state_init(&g->state, num_players, colors, map, discard_limit, friendly_robber, vps_to_win, &g->rng);
}

void game_copy(Game *dst, const Game *src) {
    *dst = *src;
}

Color game_winning_color(Game *g) {
    for (int i = 0; i < g->state.num_players; i++) {
        if (g->state.player_state[i][PS_ACTUAL_VICTORY_POINTS] >= g->vps_to_win)
            return g->state.colors[i];
    }
    return COLOR_NONE;
}

void game_execute(Game *g, Action action, Action *action_buf, int *action_count) {
    apply_action(&g->state, action, &g->rng);
    *action_count = generate_playable_actions(&g->state, action_buf, MAX_ACTIONS);
}

Color game_play(Game *g, DecideFn decide_fn) {
    Action actions[MAX_ACTIONS];
    int n = generate_playable_actions(&g->state, actions, MAX_ACTIONS);
    while (game_winning_color(g) == COLOR_NONE && g->state.num_turns < TURNS_LIMIT) {
        Action action = decide_fn(&g->state, actions, n);
        apply_action(&g->state, action, &g->rng);
        n = generate_playable_actions(&g->state, actions, MAX_ACTIONS);
    }
    return game_winning_color(g);
}
