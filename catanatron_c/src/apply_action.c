/*
 * apply_action: mutate game state according to the given action.
 * This is the core game engine, matching Python's apply_action.py.
 */

#include "apply_action.h"
#include "actions.h"
#include "rng.h"
#include <string.h>

/* ---- Helpers ---- */

static void advance_turn(State *s, int direction) {
    int next = (s->current_player_index + direction + s->num_players) % s->num_players;
    s->current_player_index = next;
    s->current_turn_index = next;
    s->num_turns++;
}

static void player_clean_turn(State *s, Color c) {
    int idx = s->color_to_index[(int)c];
    s->player_state[idx][PS_HAS_ROLLED] = 0;
    s->player_state[idx][PS_HAS_PLAYED_DEV_CARD_IN_TURN] = 0;

    /* Snapshot owned-at-start for dev card play restrictions */
    s->player_state[idx][PS_KNIGHT_OWNED_AT_START] =
        s->player_state[idx][PS_KNIGHT_IN_HAND];
    s->player_state[idx][PS_MONOPOLY_OWNED_AT_START] =
        s->player_state[idx][PS_MONOPOLY_IN_HAND];
    s->player_state[idx][PS_YEAR_OF_PLENTY_OWNED_AT_START] =
        s->player_state[idx][PS_YEAR_OF_PLENTY_IN_HAND];
    s->player_state[idx][PS_ROAD_BUILDING_OWNED_AT_START] =
        s->player_state[idx][PS_ROAD_BUILDING_IN_HAND];
}

static void maintain_longest_road(State *s) {
    for (int i = 0; i < s->num_players; i++)
        s->player_state[i][PS_LONGEST_ROAD_LENGTH] = s->board.road_lengths[i];

    int rc = s->board.road_color;
    /* Find who previously had road */
    int prev_holder = -1;
    for (int i = 0; i < s->num_players; i++) {
        if (s->player_state[i][PS_HAS_ROAD]) {
            prev_holder = i;
            break;
        }
    }

    if (rc == COLOR_NONE) return;
    int winner_idx = s->color_to_index[rc];
    if (prev_holder == winner_idx) return;

    s->player_state[winner_idx][PS_HAS_ROAD] = 1;
    s->player_state[winner_idx][PS_VICTORY_POINTS] += 2;
    s->player_state[winner_idx][PS_ACTUAL_VICTORY_POINTS] += 2;
    if (prev_holder >= 0) {
        s->player_state[prev_holder][PS_HAS_ROAD] = 0;
        s->player_state[prev_holder][PS_VICTORY_POINTS] -= 2;
        s->player_state[prev_holder][PS_ACTUAL_VICTORY_POINTS] -= 2;
    }
}

static void maintain_largest_army(State *s, Color c) {
    int idx = s->color_to_index[(int)c];
    int candidate = s->player_state[idx][PS_PLAYED_KNIGHT];
    if (candidate < 3) return;

    int prev_holder = -1, prev_size = 0;
    for (int i = 0; i < s->num_players; i++) {
        if (s->player_state[i][PS_HAS_ARMY]) {
            prev_holder = i;
            prev_size = s->player_state[i][PS_PLAYED_KNIGHT];
            break;
        }
    }

    if (prev_holder < 0) {
        s->player_state[idx][PS_HAS_ARMY] = 1;
        s->player_state[idx][PS_VICTORY_POINTS] += 2;
        s->player_state[idx][PS_ACTUAL_VICTORY_POINTS] += 2;
    } else if (prev_size < candidate && prev_holder != idx) {
        s->player_state[idx][PS_HAS_ARMY] = 1;
        s->player_state[idx][PS_VICTORY_POINTS] += 2;
        s->player_state[idx][PS_ACTUAL_VICTORY_POINTS] += 2;
        s->player_state[prev_holder][PS_HAS_ARMY] = 0;
        s->player_state[prev_holder][PS_VICTORY_POINTS] -= 2;
        s->player_state[prev_holder][PS_ACTUAL_VICTORY_POINTS] -= 2;
    }
}

static void state_build_settlement(State *s, Color c, int node_id, bool is_free) {
    int idx = s->color_to_index[(int)c];
    s->settlements[idx][s->settlement_count[idx]++] = node_id;
    s->player_state[idx][PS_SETTLEMENTS_AVAILABLE]--;
    s->player_state[idx][PS_VICTORY_POINTS]++;
    s->player_state[idx][PS_ACTUAL_VICTORY_POINTS]++;
    if (!is_free) {
        s->player_state[idx][PS_WOOD_IN_HAND]--;
        s->player_state[idx][PS_BRICK_IN_HAND]--;
        s->player_state[idx][PS_SHEEP_IN_HAND]--;
        s->player_state[idx][PS_WHEAT_IN_HAND]--;
    }
}

static void state_build_road(State *s, Color c, int a, int b, bool is_free) {
    int idx = s->color_to_index[(int)c];
    int ri = s->road_count[idx];
    s->roads[idx][ri][0] = a;
    s->roads[idx][ri][1] = b;
    s->road_count[idx]++;
    s->player_state[idx][PS_ROADS_AVAILABLE]--;
    if (!is_free) {
        s->player_state[idx][PS_WOOD_IN_HAND]--;
        s->player_state[idx][PS_BRICK_IN_HAND]--;
        s->resource_freqdeck[RES_WOOD]++;
        s->resource_freqdeck[RES_BRICK]++;
    }
}

static void state_build_city(State *s, Color c, int node_id) {
    int idx = s->color_to_index[(int)c];
    /* Remove from settlements, add to cities */
    for (int i = 0; i < s->settlement_count[idx]; i++) {
        if (s->settlements[idx][i] == node_id) {
            s->settlements[idx][i] = s->settlements[idx][s->settlement_count[idx]-1];
            s->settlement_count[idx]--;
            break;
        }
    }
    s->cities[idx][s->city_count[idx]++] = node_id;
    s->player_state[idx][PS_CITIES_AVAILABLE]--;
    s->player_state[idx][PS_SETTLEMENTS_AVAILABLE]++;
    s->player_state[idx][PS_VICTORY_POINTS]++;
    s->player_state[idx][PS_ACTUAL_VICTORY_POINTS]++;
    s->player_state[idx][PS_WHEAT_IN_HAND] -= 2;
    s->player_state[idx][PS_ORE_IN_HAND] -= 3;
}

static void play_dev_card(State *s, Color c, DevCardType dev) {
    int idx = s->color_to_index[(int)c];
    s->player_state[idx][PS_DEV_IN_HAND(dev)]--;
    s->player_state[idx][PS_PLAYED_DEV(dev)]++;
    s->player_state[idx][PS_HAS_PLAYED_DEV_CARD_IN_TURN] = 1;
}



static void yield_resources(State *s, int number) {
    Board *b = &s->board;
    CatanMap *map = b->map;

    int payout[MAX_PLAYERS][5];
    memset(payout, 0, sizeof(payout));
    int totals[5] = {0,0,0,0,0};

    for (int t = 0; t < map->num_land_tiles; t++) {
        LandTile *tile = &map->land_tiles[t];
        if (tile->number != number || tile->resource == RES_NONE) continue;
        if (coord_eq(map->land_tile_coords[t], b->robber_coordinate)) continue;

        int res = (int)tile->resource;
        for (int ni = 0; ni < 6; ni++) {
            int node = tile->nodes[ni];
            if (b->buildings[node] < 0) continue;
            Color nc = (Color)(b->buildings[node] >> 2);
            BuildingType bt = (BuildingType)(b->buildings[node] & 3);
            int ci = s->color_to_index[(int)nc];
            int amount = (bt == BLD_CITY) ? 2 : 1;
            payout[ci][res] += amount;
            totals[res] += amount;
        }
    }

    /* Check for depleted resources */
    bool depleted[5] = {false};
    for (int r = 0; r < 5; r++) {
        if (totals[r] > s->resource_freqdeck[r])
            depleted[r] = true;
    }

    for (int p = 0; p < s->num_players; p++) {
        for (int r = 0; r < 5; r++) {
            if (!depleted[r] && payout[p][r] > 0) {
                s->player_state[p][PS_RESOURCE_IN_HAND(r)] += payout[p][r];
                s->resource_freqdeck[r] -= payout[p][r];
            }
        }
    }
}


/* ---- Action handlers ---- */

static void do_end_turn(State *s, Action a) {
    player_clean_turn(s, a.color);
    advance_turn(s, 1);
    s->current_prompt = PROMPT_PLAY_TURN;
}

static void do_build_settlement(State *s, Action a) {
    int node_id = a.value[0];
    int idx = s->color_to_index[(int)a.color];

    if (s->is_initial_build_phase) {
        board_build_settlement(&s->board, a.color, node_id, true);
        state_build_settlement(s, a.color, node_id, true);

        bool is_second = (s->settlement_count[idx] == 2);
        if (is_second) {
            CatanMap *map = s->board.map;
            for (int i = 0; i < map->adjacent_tiles_count[node_id]; i++) {
                int ti = map->adjacent_tiles[node_id][i];
                Resource res = map->land_tiles[ti].resource;
                if (res != RES_NONE) {
                    s->resource_freqdeck[(int)res]--;
                    s->player_state[idx][PS_RESOURCE_IN_HAND((int)res)]++;
                }
            }
        }
        s->current_prompt = PROMPT_BUILD_INITIAL_ROAD;
    } else {
        board_build_settlement(&s->board, a.color, node_id, false);
        state_build_settlement(s, a.color, node_id, false);
        for (int r = 0; r < 5; r++)
            s->resource_freqdeck[r] += SETTLEMENT_COST[r];
        maintain_longest_road(s);
    }
}

static void do_build_road(State *s, Action a) {
    int ea = a.value[0], eb = a.value[1];

    if (s->is_initial_build_phase) {
        board_build_road(&s->board, a.color, ea, eb);
        state_build_road(s, a.color, ea, eb, true);

        int total_settlements = 0;
        for (int i = 0; i < s->num_players; i++)
            total_settlements += s->settlement_count[i];

        bool going_forward = total_settlements < s->num_players;
        bool at_the_end = total_settlements == s->num_players;

        if (going_forward) {
            advance_turn(s, 1);
            s->current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT;
        } else if (at_the_end) {
            s->current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT;
        } else if (total_settlements == 2 * s->num_players) {
            s->is_initial_build_phase = false;
            s->current_prompt = PROMPT_PLAY_TURN;
        } else {
            advance_turn(s, -1);
            s->current_prompt = PROMPT_BUILD_INITIAL_SETTLEMENT;
        }
    } else if (s->is_road_building && s->free_roads_available > 0) {
        board_build_road(&s->board, a.color, ea, eb);
        state_build_road(s, a.color, ea, eb, true);
        maintain_longest_road(s);

        s->free_roads_available--;
        if (s->free_roads_available == 0) {
            s->is_road_building = false;
        } else {
            Action tmp[MAX_ACTIONS];
            int rb_cnt = generate_playable_actions(s, tmp, MAX_ACTIONS);
            /* If road building but no more edges, end it */
            bool has_road_action = false;
            for (int i = 0; i < rb_cnt; i++)
                if (tmp[i].type == AT_BUILD_ROAD) { has_road_action = true; break; }
            if (!has_road_action) {
                s->is_road_building = false;
                s->free_roads_available = 0;
            }
        }
    } else {
        board_build_road(&s->board, a.color, ea, eb);
        state_build_road(s, a.color, ea, eb, false);
        maintain_longest_road(s);
    }
}

static void do_build_city(State *s, Action a) {
    int node_id = a.value[0];
    board_build_city(&s->board, a.color, node_id);
    state_build_city(s, a.color, node_id);
    for (int r = 0; r < 5; r++)
        s->resource_freqdeck[r] += CITY_COST[r];
}

static void do_buy_dev_card(State *s, Action a) {
    int idx = s->color_to_index[(int)a.color];
    int card = s->development_listdeck[s->dev_deck_size - 1];
    s->dev_deck_size--;

    s->player_state[idx][PS_DEV_IN_HAND(card)]++;
    for (int r = 0; r < 5; r++) {
        s->player_state[idx][PS_RESOURCE_IN_HAND(r)] -= DEV_CARD_COST[r];
        s->resource_freqdeck[r] += DEV_CARD_COST[r];
    }
    if (card == DEV_VICTORY_POINT) {
        s->player_state[idx][PS_ACTUAL_VICTORY_POINTS]++;
    }
}

static void do_roll(State *s, Action a, RngState *rng) {
    int idx = s->color_to_index[(int)a.color];
    s->player_state[idx][PS_HAS_ROLLED] = 1;

    int d1 = rng_randint(rng, 1, 6);
    int d2 = rng_randint(rng, 1, 6);
    int number = d1 + d2;

    if (number == 7) {
        int first_discard = -1;
        for (int i = 0; i < s->num_players; i++) {
            int num_cards = player_num_resources(s, i);
            int dc = (num_cards > s->discard_limit) ? num_cards / 2 : 0;
            s->discard_counts[i] = dc;
            if (dc > 0 && first_discard < 0) first_discard = i;
        }

        if (first_discard >= 0) {
            s->current_player_index = first_discard;
            s->current_prompt = PROMPT_DISCARD;
            s->is_discarding = true;
        } else {
            memset(s->discard_counts, 0, sizeof(s->discard_counts));
            s->current_prompt = PROMPT_MOVE_ROBBER;
            s->is_moving_knight = true;
        }
    } else {
        yield_resources(s, number);
        s->current_prompt = PROMPT_PLAY_TURN;
    }
}

static void do_discard(State *s, Action a) {
    int res = a.value[0];
    int pi = s->color_to_index[(int)a.color];
    s->player_state[pi][PS_RESOURCE_IN_HAND(res)]--;
    s->resource_freqdeck[res]++;
    s->discard_counts[pi]--;

    if (s->discard_counts[pi] > 0) return;

    int next = -1;
    for (int i = pi + 1; i < s->num_players; i++) {
        if (s->discard_counts[i] > 0) { next = i; break; }
    }
    if (next >= 0) {
        s->current_player_index = next;
    } else {
        s->current_player_index = s->current_turn_index;
        s->current_prompt = PROMPT_MOVE_ROBBER;
        s->is_discarding = false;
        s->is_moving_knight = true;
        memset(s->discard_counts, 0, sizeof(s->discard_counts));
    }
}

static void do_move_robber(State *s, Action a, RngState *rng) {
    Coordinate coord = {a.value[0], a.value[1], a.value[2]};
    int robbed_color = a.value[3];

    if (robbed_color != COLOR_NONE) {
        int ri = s->color_to_index[robbed_color];
        /* Random steal: build array of resources in hand, pick one */
        int hand[256];
        int hand_count = 0;
        for (int r = 0; r < 5; r++) {
            int amt = s->player_state[ri][PS_RESOURCE_IN_HAND(r)];
            if (amt < 0) amt = 0;
            for (int k = 0; k < amt && hand_count < 255; k++)
                hand[hand_count++] = r;
        }
        if (hand_count > 0) {
            int stolen = hand[rng_choice_index(rng, hand_count)];
            int ai = s->color_to_index[(int)a.color];
            s->player_state[ri][PS_RESOURCE_IN_HAND(stolen)]--;
            s->player_state[ai][PS_RESOURCE_IN_HAND(stolen)]++;
        }
    }
    s->board.robber_coordinate = coord;
    s->current_prompt = PROMPT_PLAY_TURN;
}

static void do_play_knight(State *s, Action a) {
    play_dev_card(s, a.color, DEV_KNIGHT);
    maintain_largest_army(s, a.color);
    s->current_prompt = PROMPT_MOVE_ROBBER;
}

static void do_play_yop(State *s, Action a) {
    int r1 = a.value[0], r2 = a.value[1];
    int idx = s->color_to_index[(int)a.color];
    s->player_state[idx][PS_RESOURCE_IN_HAND(r1)]++;
    s->resource_freqdeck[r1]--;
    if (r2 >= 0) {
        s->player_state[idx][PS_RESOURCE_IN_HAND(r2)]++;
        s->resource_freqdeck[r2]--;
    }
    play_dev_card(s, a.color, DEV_YEAR_OF_PLENTY);
    s->current_prompt = PROMPT_PLAY_TURN;
}

static void do_play_monopoly(State *s, Action a) {
    int res = a.value[0];
    int idx = s->color_to_index[(int)a.color];
    int total_stolen = 0;
    for (int i = 0; i < s->num_players; i++) {
        if (i == idx) continue;
        int amt = s->player_state[i][PS_RESOURCE_IN_HAND(res)];
        s->player_state[i][PS_RESOURCE_IN_HAND(res)] = 0;
        total_stolen += amt;
    }
    s->player_state[idx][PS_RESOURCE_IN_HAND(res)] += total_stolen;
    play_dev_card(s, a.color, DEV_MONOPOLY);
    s->current_prompt = PROMPT_PLAY_TURN;
}

static void do_play_road_building(State *s, Action a) {
    play_dev_card(s, a.color, DEV_ROAD_BUILDING);
    s->is_road_building = true;
    s->free_roads_available = 2;
    s->current_prompt = PROMPT_PLAY_TURN;
}

static void do_maritime_trade(State *s, Action a) {
    int idx = s->color_to_index[(int)a.color];
    /* Offering: values[0..3] are resource indices (or -1 for padding) */
    for (int i = 0; i < 4; i++) {
        if (a.value[i] >= 0) {
            s->player_state[idx][PS_RESOURCE_IN_HAND(a.value[i])]--;
            s->resource_freqdeck[a.value[i]]++;
        }
    }
    /* Receiving: value[4] */
    s->player_state[idx][PS_RESOURCE_IN_HAND(a.value[4])]++;
    s->resource_freqdeck[a.value[4]]--;
    s->current_prompt = PROMPT_PLAY_TURN;
}

/* ---- Main dispatcher ---- */

void apply_action(State *s, Action a, RngState *rng) {
    switch (a.type) {
        case AT_END_TURN:               do_end_turn(s, a); break;
        case AT_BUILD_SETTLEMENT:       do_build_settlement(s, a); break;
        case AT_BUILD_ROAD:             do_build_road(s, a); break;
        case AT_BUILD_CITY:             do_build_city(s, a); break;
        case AT_BUY_DEVELOPMENT_CARD:   do_buy_dev_card(s, a); break;
        case AT_ROLL:                   do_roll(s, a, rng); break;
        case AT_DISCARD_RESOURCE:       do_discard(s, a); break;
        case AT_MOVE_ROBBER:            do_move_robber(s, a, rng); break;
        case AT_PLAY_KNIGHT_CARD:       do_play_knight(s, a); break;
        case AT_PLAY_YEAR_OF_PLENTY:    do_play_yop(s, a); break;
        case AT_PLAY_MONOPOLY:          do_play_monopoly(s, a); break;
        case AT_PLAY_ROAD_BUILDING:     do_play_road_building(s, a); break;
        case AT_MARITIME_TRADE:         do_maritime_trade(s, a); break;
        default: break;
    }

    s->num_action_records++;
}
