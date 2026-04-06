#include "state.h"
#include "rng.h"
#include <string.h>

void state_init(State *s, int num_players, Color player_colors[], CatanMap *map,
                int discard_limit, bool friendly_robber, int vps_to_win, RngState *rng) {
    memset(s, 0, sizeof(State));
    s->num_players = num_players;
    s->discard_limit = discard_limit;
    s->friendly_robber = friendly_robber;
    s->vps_to_win = vps_to_win;

    /* Shuffle seating order (Python: random.sample(players, len(players))) */
    int indices[MAX_PLAYERS];
    for (int i = 0; i < num_players; i++) indices[i] = i;
    int shuffled[MAX_PLAYERS];
    rng_sample_int(rng, indices, num_players, shuffled, num_players);
    for (int i = 0; i < num_players; i++) {
        s->colors[i] = player_colors[shuffled[i]];
        s->color_to_index[(int)s->colors[i]] = i;
    }

    /* Initialize board */
    board_init(&s->board, map);

    /* Initialize player state */
    for (int i = 0; i < num_players; i++) {
        for (int f = 0; f < NUM_PLAYER_STATE_FIELDS; f++) {
            s->player_state[i][f] = PLAYER_INIT[f];
        }
    }

    /* Initialize bank */
    for (int i = 0; i < NUM_RESOURCES; i++)
        s->resource_freqdeck[i] = 19;

    /* Initialize dev card deck: 14 knight, 2 yop, 2 rb, 2 mono, 5 vp */
    int deck_idx = 0;
    for (int i = 0; i < 14; i++) s->development_listdeck[deck_idx++] = DEV_KNIGHT;
    for (int i = 0; i < 2; i++)  s->development_listdeck[deck_idx++] = DEV_YEAR_OF_PLENTY;
    for (int i = 0; i < 2; i++)  s->development_listdeck[deck_idx++] = DEV_ROAD_BUILDING;
    for (int i = 0; i < 2; i++)  s->development_listdeck[deck_idx++] = DEV_MONOPOLY;
    for (int i = 0; i < 5; i++)  s->development_listdeck[deck_idx++] = DEV_VICTORY_POINT;
    s->dev_deck_size = DEV_DECK_SIZE;

    /* Shuffle dev deck (Python: random.shuffle(self.development_listdeck)) */
    rng_shuffle_int(rng, s->development_listdeck, s->dev_deck_size);

    /* Building lists start empty */
    memset(s->settlement_count, 0, sizeof(s->settlement_count));
    memset(s->city_count, 0, sizeof(s->city_count));
    memset(s->road_count, 0, sizeof(s->road_count));

    s->num_turns = 0;
    s->current_player_index = 0;
    s->current_turn_index = 0;
    s->current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT;
    s->is_initial_build_phase = true;
    s->num_action_records = 0;

    memset(s->discard_counts, 0, sizeof(s->discard_counts));
    memset(s->acceptees, 0, sizeof(s->acceptees));
    memset(s->current_trade, 0, sizeof(s->current_trade));
}
