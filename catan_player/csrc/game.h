#ifndef GAME_H
#define GAME_H

#include "state.h"

typedef Action (*DecideFn)(State *state, Action *actions, int num_actions);

typedef struct {
    State    state;
    CatanMap *map;
    RngState rng;       /* per-game RNG, thread-safe */
    const void *eval_ctx; /* opaque pointer for eval function context */
    uint64_t seed;
    int      vps_to_win;
} Game;

void game_init_with_map(Game *g, CatanMap *map, int num_players, Color colors[],
                        uint64_t seed, int discard_limit, bool friendly_robber,
                        int vps_to_win);
void game_copy(Game *dst, const Game *src);
Color game_winning_color(Game *g);
void game_execute(Game *g, Action action, Action *action_buf, int *action_count);
Color game_play(Game *g, DecideFn decide_fn);

#endif
