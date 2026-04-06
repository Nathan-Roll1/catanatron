/*
 * Action generation: given game state, produce all legal actions.
 * Matches Python's generate_playable_actions exactly.
 */

#include "actions.h"
#include <string.h>

static Action make_action(Color c, ActionType t, int v0, int v1, int v2, int v3, int v4) {
    Action a;
    a.color = c;
    a.type = t;
    a.value[0] = v0; a.value[1] = v1; a.value[2] = v2; a.value[3] = v3; a.value[4] = v4;
    return a;
}

static inline bool player_can_play_dev(State *s, Color c, DevCardType dev) {
    int idx = s->color_to_index[(int)c];
    if (s->player_state[idx][PS_HAS_PLAYED_DEV_CARD_IN_TURN]) return false;
    if (s->player_state[idx][PS_DEV_IN_HAND(dev)] <= 0) return false;

    /* Can't play card bought this turn (owned_at_start check) */
    int owned_at_start_field;
    switch (dev) {
        case DEV_KNIGHT:         owned_at_start_field = PS_KNIGHT_OWNED_AT_START; break;
        case DEV_MONOPOLY:       owned_at_start_field = PS_MONOPOLY_OWNED_AT_START; break;
        case DEV_YEAR_OF_PLENTY: owned_at_start_field = PS_YEAR_OF_PLENTY_OWNED_AT_START; break;
        case DEV_ROAD_BUILDING:  owned_at_start_field = PS_ROAD_BUILDING_OWNED_AT_START; break;
        default: return false;
    }
    return s->player_state[idx][owned_at_start_field] > 0;
}

static inline bool player_has_rolled(State *s, Color c) {
    return s->player_state[s->color_to_index[(int)c]][PS_HAS_ROLLED];
}

static inline bool player_can_afford(State *s, int idx, const int cost[5]) {
    return s->player_state[idx][PS_WOOD_IN_HAND] >= cost[0]
        && s->player_state[idx][PS_BRICK_IN_HAND] >= cost[1]
        && s->player_state[idx][PS_SHEEP_IN_HAND] >= cost[2]
        && s->player_state[idx][PS_WHEAT_IN_HAND] >= cost[3]
        && s->player_state[idx][PS_ORE_IN_HAND] >= cost[4];
}

/* ---- Sub-generators ---- */

static int settlement_possibilities(State *s, Color c, bool initial, Action *out, int max) {
    int nodes[TOTAL_NODES], cnt;
    cnt = board_buildable_node_ids(&s->board, c, initial, nodes, TOTAL_NODES);
    int n = 0;
    for (int i = 0; i < cnt && n < max; i++)
        out[n++] = make_action(c, AT_BUILD_SETTLEMENT, nodes[i], 0, 0, 0, 0);
    return n;
}

static int initial_road_possibilities(State *s, Color c, Action *out, int max) {
    /* Must connect to last settlement built */
    int last_settle = s->settlements[s->color_to_index[(int)c]]
                                    [s->settlement_count[s->color_to_index[(int)c]] - 1];

    int edges[MAX_ROAD_EDGES][2];
    int edge_cnt = board_buildable_edges(&s->board, c, edges, MAX_ROAD_EDGES);
    int n = 0;
    for (int i = 0; i < edge_cnt && n < max; i++) {
        if (edges[i][0] == last_settle || edges[i][1] == last_settle) {
            out[n++] = make_action(c, AT_BUILD_ROAD, edges[i][0], edges[i][1], 0, 0, 0);
        }
    }
    return n;
}

static int road_building_possibilities(State *s, Color c, bool check_money,
                                        Action *out, int max) {
    int idx = s->color_to_index[(int)c];
    if (s->player_state[idx][PS_ROADS_AVAILABLE] <= 0) return 0;
    if (check_money && !player_can_afford(s, idx, ROAD_COST)) return 0;

    int edges[MAX_ROAD_EDGES][2];
    int edge_cnt = board_buildable_edges(&s->board, c, edges, MAX_ROAD_EDGES);
    int n = 0;
    for (int i = 0; i < edge_cnt && n < max; i++)
        out[n++] = make_action(c, AT_BUILD_ROAD, edges[i][0], edges[i][1], 0, 0, 0);
    return n;
}

static int city_possibilities(State *s, Color c, Action *out, int max) {
    int idx = s->color_to_index[(int)c];
    if (!player_can_afford(s, idx, CITY_COST)) return 0;
    if (s->player_state[idx][PS_CITIES_AVAILABLE] <= 0) return 0;

    int n = 0;
    for (int i = 0; i < s->settlement_count[idx] && n < max; i++)
        out[n++] = make_action(c, AT_BUILD_CITY, s->settlements[idx][i], 0, 0, 0, 0);
    return n;
}

static int robber_possibilities(State *s, Color c, Action *out, int max) {
    int n = 0;
    CatanMap *map = s->board.map;

    for (int t = 0; t < map->num_land_tiles && n < max; t++) {
        Coordinate coord = map->land_tile_coords[t];
        if (coord_eq(coord, s->board.robber_coordinate)) continue;

        /* Find stealable colors on this tile */
        Color steal_from[MAX_PLAYERS];
        int steal_count = 0;
        bool seen[MAX_PLAYERS] = {false};

        for (int ni = 0; ni < 6; ni++) {
            int node = map->land_tiles[t].nodes[ni];
            Color nc = board_get_node_color(&s->board, node);
            if (nc == COLOR_NONE || nc == c) continue;
            int nc_idx = s->color_to_index[(int)nc];
            if (player_num_resources(s, nc_idx) >= 1 && !seen[(int)nc]) {
                seen[(int)nc] = true;
                steal_from[steal_count++] = nc;
            }
        }

        if (steal_count == 0) {
            /* coord packed as (x, y, z, -1) meaning no steal */
            out[n++] = make_action(c, AT_MOVE_ROBBER, coord.x, coord.y, coord.z,
                                   COLOR_NONE, 0);
        } else {
            for (int j = 0; j < steal_count && n < max; j++) {
                out[n++] = make_action(c, AT_MOVE_ROBBER, coord.x, coord.y, coord.z,
                                       (int)steal_from[j], 0);
            }
        }
    }

    /* Friendly robber: filter out actions that block low-VP opponents */
    if (s->friendly_robber && n > 0) {
        Action filtered[MAX_ACTIONS];
        int fn = 0;
        for (int i = 0; i < n; i++) {
            bool blocks_low = false;
            Coordinate ac = {out[i].value[0], out[i].value[1], out[i].value[2]};
            /* Find tile index */
            for (int t = 0; t < map->num_land_tiles; t++) {
                if (!coord_eq(map->land_tile_coords[t], ac)) continue;
                for (int ni = 0; ni < 6; ni++) {
                    int node = map->land_tiles[t].nodes[ni];
                    Color nc = board_get_node_color(&s->board, node);
                    if (nc == COLOR_NONE || nc == c) continue;
                    int ci = s->color_to_index[(int)nc];
                    if (s->player_state[ci][PS_ACTUAL_VICTORY_POINTS] < 3) {
                        blocks_low = true;
                        break;
                    }
                }
                break;
            }
            if (!blocks_low) filtered[fn++] = out[i];
        }
        if (fn > 0) {
            memcpy(out, filtered, fn * sizeof(Action));
            n = fn;
        }
    }
    return n;
}

static int year_of_plenty_possibilities(Color c, const int bank[5], Action *out, int max) {
    int n = 0;
    bool single_seen[5] = {false};

    for (int i = 0; i < 5 && n < max; i++) {
        for (int j = i; j < 5 && n < max; j++) {
            int needed[5] = {0,0,0,0,0};
            needed[i]++;
            needed[j]++;
            if (freqdeck_contains(bank, needed)) {
                out[n++] = make_action(c, AT_PLAY_YEAR_OF_PLENTY, i, j, 0, 0, 0);
            } else {
                if (bank[i] >= 1 && !single_seen[i]) {
                    single_seen[i] = true;
                    out[n++] = make_action(c, AT_PLAY_YEAR_OF_PLENTY, i, -1, 0, 0, 0);
                }
                if (bank[j] >= 1 && !single_seen[j]) {
                    single_seen[j] = true;
                    out[n++] = make_action(c, AT_PLAY_YEAR_OF_PLENTY, j, -1, 0, 0, 0);
                }
            }
        }
    }
    return n;
}

static int maritime_trade_possibilities(State *s, Color c, Action *out, int max) {
    int idx = s->color_to_index[(int)c];
    int hand[5];
    player_get_hand(s, idx, hand);

    bool ports[6];
    board_get_player_port_resources(&s->board, c, ports);

    int rates[5] = {4, 4, 4, 4, 4};
    if (ports[5]) /* has 3:1 port */
        for (int i = 0; i < 5; i++) rates[i] = 3;
    for (int i = 0; i < 5; i++)
        if (ports[i]) rates[i] = 2;

    int n = 0;
    for (int i = 0; i < 5 && n < max; i++) {
        if (hand[i] < rates[i]) continue;
        for (int j = 0; j < 5 && n < max; j++) {
            if (j == i) continue;
            if (s->resource_freqdeck[j] <= 0) continue;
            /* value: [giving_resource x rate, padded with -1, receiving_resource] */
            int v[5];
            for (int k = 0; k < 4; k++)
                v[k] = (k < rates[i]) ? i : -1;
            v[4] = j;
            out[n++] = make_action(c, AT_MARITIME_TRADE, v[0], v[1], v[2], v[3], v[4]);
        }
    }
    return n;
}

static int discard_possibilities(State *s, Color c, Action *out, int max) {
    int idx = s->color_to_index[(int)c];
    if (s->discard_counts[idx] <= 0) return 0;

    int n = 0;
    for (int r = 0; r < 5 && n < max; r++) {
        if (s->player_state[idx][PS_RESOURCE_IN_HAND(r)] > 0)
            out[n++] = make_action(c, AT_DISCARD_RESOURCE, r, 0, 0, 0, 0);
    }
    return n;
}

/* ---- Main generator ---- */

int generate_playable_actions(State *s, Action *out, int max_out) {
    Color c = state_current_color(s);
    int n = 0;

    switch (s->current_prompt) {
        case PROMPT_BUILD_INITIAL_SETTLEMENT:
            n = settlement_possibilities(s, c, true, out, max_out);
            break;

        case PROMPT_BUILD_INITIAL_ROAD:
            n = initial_road_possibilities(s, c, out, max_out);
            break;

        case PROMPT_MOVE_ROBBER:
            n = robber_possibilities(s, c, out, max_out);
            break;

        case PROMPT_PLAY_TURN:
            if (s->is_road_building) {
                n = road_building_possibilities(s, c, false, out, max_out);
            } else {
                /* Dev card plays (before or after roll) */
                if (player_can_play_dev(s, c, DEV_YEAR_OF_PLENTY))
                    n += year_of_plenty_possibilities(c, s->resource_freqdeck,
                                                      out + n, max_out - n);
                if (player_can_play_dev(s, c, DEV_MONOPOLY)) {
                    for (int r = 0; r < 5 && n < max_out; r++)
                        out[n++] = make_action(c, AT_PLAY_MONOPOLY, r, 0, 0, 0, 0);
                }
                if (player_can_play_dev(s, c, DEV_KNIGHT))
                    out[n++] = make_action(c, AT_PLAY_KNIGHT_CARD, 0, 0, 0, 0, 0);
                if (player_can_play_dev(s, c, DEV_ROAD_BUILDING)) {
                    Action tmp[MAX_ACTIONS];
                    int rb_count = road_building_possibilities(s, c, false, tmp, MAX_ACTIONS);
                    if (rb_count > 0)
                        out[n++] = make_action(c, AT_PLAY_ROAD_BUILDING, 0, 0, 0, 0, 0);
                }

                if (!player_has_rolled(s, c)) {
                    out[n++] = make_action(c, AT_ROLL, 0, 0, 0, 0, 0);
                } else {
                    out[n++] = make_action(c, AT_END_TURN, 0, 0, 0, 0, 0);
                    n += road_building_possibilities(s, c, true, out + n, max_out - n);
                    n += settlement_possibilities(s, c, false, out + n, max_out - n);
                    n += city_possibilities(s, c, out + n, max_out - n);

                    int idx = s->color_to_index[(int)c];
                    if (player_can_afford(s, idx, DEV_CARD_COST) && s->dev_deck_size > 0)
                        out[n++] = make_action(c, AT_BUY_DEVELOPMENT_CARD, 0, 0, 0, 0, 0);

                    n += maritime_trade_possibilities(s, c, out + n, max_out - n);
                }
            }
            break;

        case PROMPT_DISCARD:
            n = discard_possibilities(s, c, out, max_out);
            break;

        case PROMPT_DECIDE_TRADE:
            out[n++] = make_action(c, AT_REJECT_TRADE,
                s->current_trade[0], s->current_trade[1], s->current_trade[2],
                s->current_trade[3], s->current_trade[4]);
            {
                int hand[5];
                player_get_hand(s, s->color_to_index[(int)c], hand);
                int asked[5] = {s->current_trade[5], s->current_trade[6],
                                s->current_trade[7], s->current_trade[8],
                                s->current_trade[9]};
                if (freqdeck_contains(hand, asked))
                    out[n++] = make_action(c, AT_ACCEPT_TRADE,
                        s->current_trade[0], s->current_trade[1], s->current_trade[2],
                        s->current_trade[3], s->current_trade[4]);
            }
            break;

        case PROMPT_DECIDE_ACCEPTEES:
            out[n++] = make_action(c, AT_CANCEL_TRADE, 0, 0, 0, 0, 0);
            for (int i = 0; i < s->num_players && n < max_out; i++) {
                if (s->acceptees[i]) {
                    out[n++] = make_action(c, AT_CONFIRM_TRADE,
                        s->current_trade[0], s->current_trade[1], s->current_trade[2],
                        s->current_trade[3], (int)s->colors[i]);
                }
            }
            break;

        default:
            break;
    }
    return n;
}
