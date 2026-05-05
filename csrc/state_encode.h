#ifndef STATE_ENCODE_H
#define STATE_ENCODE_H

#include "game.h"
#include "board.h"
#include "map.h"

/* Mirror of Python state_encoder.py constants. */
#define ENC_NUM_NODES        54
#define ENC_NUM_EDGES       144  /* directed: 72 undirected × 2 */
#define ENC_NODE_FEAT_DIM    18
#define ENC_EDGE_FEAT_DIM     5
#define ENC_FLAT_FEAT_DIM   115

/* Pre-computed lookup tables, one StateEncoderC per game (per map).
 * Built once via state_encoder_init() then used for every encode_state call.
 */
typedef struct {
    int N;                                    /* num land nodes (54) */
    int E;                                    /* num directed edges (144) */
    int land_to_local[TOTAL_NODES];           /* global -> local (-1 if not land) */
    int local_to_global[ENC_NUM_NODES];       /* local -> global */
    int ltiles[NUM_LAND_TILES][6];            /* tile -> local node IDs */
    int tile_coords[NUM_LAND_TILES][3];       /* tile -> cube coords */
    int road_src_global[ENC_NUM_EDGES];       /* directed edge -> global src node */
    int road_adj_idx[ENC_NUM_EDGES];          /* directed edge -> adj index in src */
    float port_oh[ENC_NUM_NODES][7];          /* port one-hot per local node */
    int n_real_players;                       /* num_players (2/3/4) */
} StateEncoderC;

/* Initialize from a fully-built game's map. The game must have been
 * created (so STATIC_ADJ is populated and the map has tiles/ports).
 * num_players sets which seats are "real" (affects flat-feature padding). */
void state_encoder_init(StateEncoderC *enc, const Game *g, int num_players);

/* Encode a game state into nf, ef, flat. Same output as Python encoder. */
void encode_state(const StateEncoderC *enc, const Game *g,
                  float *nf,    /* shape: [N][18] = [54][18] */
                  float *ef,    /* shape: [E][5]  = [144][5] */
                  float *flat); /* shape: [115] */

#endif
