#include <stdio.h>
#include "state.h"
#include "rng.h"

int main(void) {
    CatanMap map;
    rng_seed(42);
    build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL);

    Color colors[] = {COLOR_RED, COLOR_BLUE};
    State s;
    state_init(&s, 2, colors, &map, 7, false, 10);

    printf("Num players: %d\n", s.num_players);
    printf("Seating: %d, %d\n", s.colors[0], s.colors[1]);
    printf("P0 roads avail: %d\n", s.player_state[0][PS_ROADS_AVAILABLE]);
    printf("P0 settlements avail: %d\n", s.player_state[0][PS_SETTLEMENTS_AVAILABLE]);
    printf("Bank wood: %d\n", s.resource_freqdeck[RES_WOOD]);
    printf("Dev deck size: %d\n", s.dev_deck_size);
    printf("Current prompt: %d (expect %d)\n", s.current_prompt, PROMPT_BUILD_INITIAL_SETTLEMENT);
    printf("Is initial build: %d\n", s.is_initial_build_phase);

    /* Test state_copy */
    State s2;
    state_copy(&s2, &s);
    printf("\nCopy: P0 roads avail: %d\n", s2.player_state[0][PS_ROADS_AVAILABLE]);
    printf("sizeof(State): %zu bytes\n", sizeof(State));

    printf("\nAll state tests passed.\n");
    return 0;
}
