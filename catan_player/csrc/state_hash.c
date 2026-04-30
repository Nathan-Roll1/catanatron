#include "state_hash.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define HASH_OFFSET 1469598103934665603ULL
#define HASH_PRIME  1099511628211ULL

typedef struct {
    uint64_t h;
} HashState;

static void hash_byte(HashState *hs, uint8_t byte) {
    hs->h ^= (uint64_t)byte;
    hs->h *= HASH_PRIME;
}

static void hash_u64(HashState *hs, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        hash_byte(hs, (uint8_t)(value & 0xffU));
        value >>= 8;
    }
}

static void hash_i64(HashState *hs, int64_t value) {
    hash_u64(hs, (uint64_t)value);
}

static void hash_bool(HashState *hs, bool value) {
    hash_byte(hs, value ? 1U : 0U);
}

static void hash_tag(HashState *hs, uint64_t tag) {
    hash_u64(hs, 0x9e3779b97f4a7c15ULL);
    hash_u64(hs, tag);
}

static int bounded_count(int count, int max_count) {
    if (count < 0) return 0;
    if (count > max_count) return max_count;
    return count;
}

static void hash_coord(HashState *hs, const Coordinate *coord) {
    hash_i64(hs, coord->x);
    hash_i64(hs, coord->y);
    hash_i64(hs, coord->z);
}

static void hash_double(HashState *hs, double value) {
    unsigned char bytes[sizeof(value)];
    memcpy(bytes, &value, sizeof(bytes));
    hash_u64(hs, (uint64_t)sizeof(bytes));
    for (size_t i = 0; i < sizeof(bytes); i++) {
        hash_byte(hs, bytes[i]);
    }
}

static void hash_land_tile(HashState *hs, const LandTile *tile) {
    hash_i64(hs, tile->id);
    hash_i64(hs, tile->resource);
    hash_i64(hs, tile->number);
    for (int i = 0; i < 6; i++) {
        hash_i64(hs, tile->nodes[i]);
    }
    for (int i = 0; i < 6; i++) {
        hash_i64(hs, tile->edges[i][0]);
        hash_i64(hs, tile->edges[i][1]);
    }
}

static void hash_port(HashState *hs, const Port *port) {
    hash_i64(hs, port->id);
    hash_i64(hs, port->resource);
    hash_i64(hs, port->direction);
    for (int i = 0; i < 6; i++) {
        hash_i64(hs, port->nodes[i]);
    }
    for (int i = 0; i < 6; i++) {
        hash_i64(hs, port->edges[i][0]);
        hash_i64(hs, port->edges[i][1]);
    }
}

static void hash_map(HashState *hs, const CatanMap *map, uint64_t tag) {
    hash_tag(hs, tag);
    hash_bool(hs, map != NULL);
    if (map == NULL) return;

    int land_count = bounded_count(map->num_land_tiles, NUM_LAND_TILES);
    int port_count = bounded_count(map->num_ports, NUM_PORTS);
    int land_node_count = bounded_count(map->num_land_nodes, NUM_NODES);

    hash_i64(hs, map->num_land_tiles);
    for (int i = 0; i < land_count; i++) {
        hash_land_tile(hs, &map->land_tiles[i]);
    }

    hash_i64(hs, map->num_ports);
    for (int i = 0; i < port_count; i++) {
        hash_port(hs, &map->ports[i]);
    }

    hash_i64(hs, map->num_land_nodes);
    for (int i = 0; i < land_node_count; i++) {
        hash_i64(hs, map->land_nodes[i]);
    }

    for (int node = 0; node < NUM_NODES; node++) {
        int count = bounded_count(map->adjacent_tiles_count[node], MAX_ADJ_TILES);
        hash_i64(hs, map->adjacent_tiles_count[node]);
        for (int i = 0; i < count; i++) {
            hash_i64(hs, map->adjacent_tiles[node][i]);
        }
    }

    for (int res = 0; res < 6; res++) {
        int count = bounded_count(map->port_nodes_count[res], 10);
        hash_i64(hs, map->port_nodes_count[res]);
        for (int i = 0; i < count; i++) {
            hash_i64(hs, map->port_nodes[res][i]);
        }
    }

    for (int i = 0; i < 13; i++) {
        hash_double(hs, map->dice_probas[i]);
    }

    for (int i = 0; i < land_count; i++) {
        hash_coord(hs, &map->land_tile_coords[i]);
    }
}

static void hash_rng(HashState *hs, const RngState *rng) {
    hash_tag(hs, 0x726e675f73746174ULL);
    for (int i = 0; i < MT_N; i++) {
        hash_u64(hs, rng->mt[i]);
    }
    hash_i64(hs, rng->mti);
}

static void hash_board(HashState *hs, const Board *board) {
    hash_tag(hs, 0x626f617264000001ULL);

    for (int i = 0; i < TOTAL_NODES; i++) {
        hash_i64(hs, board->buildings[i]);
    }

    for (int node = 0; node < TOTAL_NODES; node++) {
        for (int adj = 0; adj < MAX_DEGREE; adj++) {
            hash_i64(hs, board->road_owner[node][adj]);
        }
    }

    for (int color = 0; color < MAX_PLAYERS; color++) {
        hash_i64(hs, board->cc_count[color]);
        for (int comp = 0; comp < MAX_COMPONENTS; comp++) {
            hash_u64(hs, board->cc_sets[color][comp][0]);
            hash_u64(hs, board->cc_sets[color][comp][1]);
        }
    }

    hash_u64(hs, board->buildable[0]);
    hash_u64(hs, board->buildable[1]);

    for (int i = 0; i < MAX_PLAYERS; i++) {
        hash_i64(hs, board->road_lengths[i]);
    }
    hash_i64(hs, board->road_color);
    hash_i64(hs, board->road_length);

    hash_coord(hs, &board->robber_coordinate);
}

static void hash_state(HashState *hs, const State *state) {
    hash_tag(hs, 0x7374617465000001ULL);
    hash_board(hs, &state->board);

    hash_i64(hs, state->num_players);
    for (int i = 0; i < MAX_PLAYERS; i++) {
        hash_i64(hs, state->colors[i]);
    }
    for (int i = 0; i < MAX_PLAYERS; i++) {
        hash_i64(hs, state->color_to_index[i]);
    }

    for (int player = 0; player < MAX_PLAYERS; player++) {
        for (int field = 0; field < NUM_PLAYER_STATE_FIELDS; field++) {
            hash_i64(hs, state->player_state[player][field]);
        }
    }

    for (int i = 0; i < NUM_RESOURCES; i++) {
        hash_i64(hs, state->resource_freqdeck[i]);
    }

    int dev_deck_size = bounded_count(state->dev_deck_size, MAX_DEV_DECK);
    hash_i64(hs, state->dev_deck_size);
    for (int i = 0; i < dev_deck_size; i++) {
        hash_i64(hs, state->development_listdeck[i]);
    }

    for (int player = 0; player < MAX_PLAYERS; player++) {
        int count = bounded_count(state->settlement_count[player], 5);
        hash_i64(hs, state->settlement_count[player]);
        for (int i = 0; i < count; i++) {
            hash_i64(hs, state->settlements[player][i]);
        }
    }

    for (int player = 0; player < MAX_PLAYERS; player++) {
        int count = bounded_count(state->city_count[player], 4);
        hash_i64(hs, state->city_count[player]);
        for (int i = 0; i < count; i++) {
            hash_i64(hs, state->cities[player][i]);
        }
    }

    for (int player = 0; player < MAX_PLAYERS; player++) {
        int count = bounded_count(state->road_count[player], 15);
        hash_i64(hs, state->road_count[player]);
        for (int i = 0; i < count; i++) {
            hash_i64(hs, state->roads[player][i][0]);
            hash_i64(hs, state->roads[player][i][1]);
        }
    }

    hash_i64(hs, state->num_action_records);
    hash_i64(hs, state->num_turns);
    hash_i64(hs, state->current_player_index);
    hash_i64(hs, state->current_turn_index);
    hash_i64(hs, state->current_prompt);

    hash_bool(hs, state->is_initial_build_phase);
    hash_bool(hs, state->is_discarding);
    for (int i = 0; i < MAX_PLAYERS; i++) {
        hash_i64(hs, state->discard_counts[i]);
    }
    hash_bool(hs, state->is_moving_knight);
    hash_bool(hs, state->is_road_building);
    hash_i64(hs, state->free_roads_available);

    hash_bool(hs, state->is_resolving_trade);
    for (int i = 0; i < 11; i++) {
        hash_i64(hs, state->current_trade[i]);
    }
    for (int i = 0; i < MAX_PLAYERS; i++) {
        hash_bool(hs, state->acceptees[i]);
    }

    hash_i64(hs, state->discard_limit);
    hash_bool(hs, state->friendly_robber);
    hash_i64(hs, state->vps_to_win);
}

uint64_t game_full_hash(const Game *g) {
    HashState hs = {HASH_OFFSET};

    hash_tag(&hs, 0x636174616e5f7631ULL);
    hash_bool(&hs, g != NULL);
    if (g == NULL) return hs.h;

    hash_u64(&hs, g->seed);
    hash_i64(&hs, g->vps_to_win);

    hash_map(&hs, g->map, 0x67616d655f6d6170ULL);
    hash_map(&hs, g->state.board.map, 0x626f6172645f6d70ULL);
    hash_rng(&hs, &g->rng);
    hash_state(&hs, &g->state);

    return hs.h ? hs.h : 0x6a09e667f3bcc909ULL;
}

uint64_t game_dynamic_hash(const Game *g) {
    HashState hs = {HASH_OFFSET};

    hash_tag(&hs, 0x636174616e5f6479ULL);
    hash_bool(&hs, g != NULL);
    if (g == NULL) return hs.h;

    hash_u64(&hs, g->seed);
    hash_i64(&hs, g->vps_to_win);
    hash_rng(&hs, &g->rng);
    hash_state(&hs, &g->state);

    return hs.h ? hs.h : 0xbb67ae8584caa73bULL;
}
