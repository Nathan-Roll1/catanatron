#ifndef POLICY_TOPK_H
#define POLICY_TOPK_H

#include "game.h"
#include "nn.h"
#include "state_encode.h"

/* Encode `action` to a flat policy-space index (0..336). Returns -1 on
 * unsupported action types (e.g. AT_OFFER_TRADE). Mirrors the Python
 * ActionEncoder.encode() function exactly.
 *
 * Uses lookup tables from NNModel (node_to_compact, edge_lut, coord_to_tile,
 * mar_lut, idx_to_edge) that were exported with the model weights. */
int policy_action_encode(const NNModel *m, const Action *action);

/* Run encode + nn_forward + top-K and return up to K legal action indices
 * (positions in the `actions` array) ranked by policy logit (descending).
 * Returns the number of indices written to `out_indices`.
 *
 * Buffers nf, ef, mk, out_buf are caller-supplied scratch space:
 *   nf  : float[ENC_NUM_NODES * ENC_NODE_FEAT_DIM] = float[54*18]
 *   ef  : float[ENC_NUM_EDGES * ENC_EDGE_FEAT_DIM] = float[144*5]
 *   ff  : float[ENC_FLAT_FEAT_DIM] = float[115]
 *   mk  : float[NN_MASK_DIM] = float[397]
 *   out : float[4 + NN_MASK_DIM] = float[401]
 *
 * Why caller-supplied: avoids per-call malloc, matches Python's reuse pattern. */
int policy_top_k(const StateEncoderC *enc, const NNModel *m,
                 const Game *g, const Action *actions, int n_actions,
                 int k, int *out_indices,
                 float *nf, float *ef, float *ff, float *mk, float *out);

int policy_top_k_ex(const StateEncoderC *enc, const NNModel *m,
                    const Game *g, const Action *actions, int n_actions,
                    int k, int *out_indices,
                    float *nf, float *ef, float *ff, float *mk, float *out,
                    int use_algo_policy);

void policy_algo_configure(int flags, int use_value_tiebreak);

#endif
