/* policy_topk.c — Pure C policy callback for deep_search.
 *
 * Replaces the Python policy callback with a single C function that:
 *   1. Encodes game state to nf/ef/ff buffers (state_encode.c)
 *   2. Builds the action mask from legal actions
 *   3. Runs nn_forward (libnn)
 *   4. Sorts legal actions by their policy logit
 *   5. Returns top-K indices (positions in `actions` array)
 *
 * This eliminates ALL Python overhead in the deep_search inner loop.
 */

#include "policy_topk.h"
#include "actions.h"
#include "state_encode.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Action type enum values (must match catan_types.h) */
#define AT_ROLL                  0
#define AT_MOVE_ROBBER           1
#define AT_DISCARD_RESOURCE      2
#define AT_BUILD_ROAD            3
#define AT_BUILD_SETTLEMENT      4
#define AT_BUILD_CITY            5
#define AT_BUY_DEVELOPMENT_CARD  6
#define AT_PLAY_KNIGHT_CARD      7
#define AT_PLAY_YEAR_OF_PLENTY   8
#define AT_PLAY_MONOPOLY         9
#define AT_PLAY_ROAD_BUILDING   10
#define AT_MARITIME_TRADE       11
#define AT_OFFER_TRADE          12
#define AT_ACCEPT_TRADE         13
#define AT_REJECT_TRADE         14
#define AT_CONFIRM_TRADE        15
#define AT_CANCEL_TRADE         16
#define AT_END_TURN             17

/* Policy-space slot starts (must match Python ActionEncoder constants) */
#define SLOT_ROLL          0
#define SLOT_END_TURN      1
#define SLOT_BUY_DEV       2
#define SLOT_KNIGHT        3
#define SLOT_ROAD_BUILDING 4
#define SLOT_SETTLEMENT    5     /* + node_to_compact[v[0]] */
#define SLOT_CITY         59     /* + node_to_compact[v[0]] */
#define SLOT_ROAD        113     /* + edge_lut[min(v[0],v[1])][max(v[0],v[1])] */
#define SLOT_ROBBER      185     /* + tile*5 + steal_idx */
#define SLOT_DISCARD     280     /* + v[0] */
#define SLOT_YOP         285     /* + yop_lut (unused for top-K, see note) */
#define SLOT_MONOPOLY    305     /* + v[0] */
#define SLOT_MARITIME    310     /* + mar_lut[v[0]][v[4]] */
#define SLOT_ACCEPT      330
#define SLOT_REJECT      331
#define SLOT_CANCEL      332
#define SLOT_CONFIRM     333     /* + v[4] (color) */

#define NO_STEAL_IDX 4
#define POLICY_AD 337   /* size of policy logit space (NN AD) */

/* Encode action to flat policy-space index. Returns -1 if unencodable. */
int policy_action_encode(const NNModel *m, const Action *a) {
    int t = a->type;
    const int32_t *v = a->value;

    /* Hot path: most common types first */
    if (t == AT_BUILD_SETTLEMENT) {
        int compact = m->node_to_compact[v[0]];
        if (compact < 0) return -1;
        return SLOT_SETTLEMENT + compact;
    }
    if (t == AT_BUILD_ROAD) {
        int a0 = v[0], a1 = v[1];
        int lo = a0 < a1 ? a0 : a1;
        int hi = a0 < a1 ? a1 : a0;
        if (lo < 0 || hi >= 96) return -1;
        int e = m->edge_lut[lo][hi];
        if (e < 0) return -1;
        return SLOT_ROAD + e;
    }
    if (t == AT_BUILD_CITY) {
        int compact = m->node_to_compact[v[0]];
        if (compact < 0) return -1;
        return SLOT_CITY + compact;
    }
    if (t == AT_ROLL) return SLOT_ROLL;
    if (t == AT_END_TURN) return SLOT_END_TURN;
    if (t == AT_MOVE_ROBBER) {
        int x = v[0], y = v[1], z = v[2];
        /* coord_to_tile is offset by +3 in each axis */
        if (x < -3 || x > 3 || y < -3 || y > 3 || z < -3 || z > 3) return -1;
        int tile = m->coord_to_tile[x + 3][y + 3][z + 3];
        if (tile < 0) return -1;
        int steal = (v[3] >= 0) ? v[3] : NO_STEAL_IDX;
        return SLOT_ROBBER + tile * 5 + steal;
    }
    if (t == AT_BUY_DEVELOPMENT_CARD) return SLOT_BUY_DEV;
    if (t == AT_MARITIME_TRADE) {
        int give = v[0], recv = v[4];
        if (give < 0 || give >= 5 || recv < 0 || recv >= 5) return -1;
        int idx = m->mar_lut[give][recv];
        if (idx < 0) return -1;
        return SLOT_MARITIME + idx;
    }
    if (t == AT_PLAY_KNIGHT_CARD) return SLOT_KNIGHT;
    if (t == AT_PLAY_ROAD_BUILDING) return SLOT_ROAD_BUILDING;
    if (t == AT_DISCARD_RESOURCE) {
        if (v[0] < 0 || v[0] >= 5) return -1;
        return SLOT_DISCARD + v[0];
    }
    if (t == AT_PLAY_MONOPOLY) {
        if (v[0] < 0 || v[0] >= 5) return -1;
        return SLOT_MONOPOLY + v[0];
    }
    if (t == AT_PLAY_YEAR_OF_PLENTY) {
        /* YOP encoding requires a YOP-pairs LUT not in NNModel.
         * This is rare and unlikely to be in top-K. Skip. */
        return -1;
    }
    if (t == AT_ACCEPT_TRADE) return SLOT_ACCEPT;
    if (t == AT_REJECT_TRADE) return SLOT_REJECT;
    if (t == AT_CANCEL_TRADE) return SLOT_CANCEL;
    if (t == AT_CONFIRM_TRADE) {
        if (v[4] < 0 || v[4] >= 4) return -1;
        return SLOT_CONFIRM + v[4];
    }
    return -1;  /* AT_OFFER_TRADE and unknowns */
}

/* nn_forward signature (declared here, defined in nn.c) */
extern void nn_forward(const NNModel *m,
                       const float node_feat[NN_NODES][NN_NODE_FEAT],
                       const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                       const float flat_feat[NN_FLAT_DIM],
                       const float mask[NN_MASK_DIM],
                       NNOutput *out);

typedef struct {
    float logit;
    int idx;
} ScoredAction;

/* Sort descending by logit (qsort comparator) */
static int scored_cmp_desc(const void *a, const void *b) {
    const ScoredAction *sa = (const ScoredAction *)a;
    const ScoredAction *sb = (const ScoredAction *)b;
    if (sa->logit > sb->logit) return -1;
    if (sa->logit < sb->logit) return 1;
    return sa->idx - sb->idx;  /* stable for ties: lower idx first */
}

/* Insertion sort top-K: for small K (<=16) much faster than full qsort.
 * Maintains a max-heap of size k with min at top, replacing it when a
 * better candidate arrives. Then sort the final heap descending. */
static int top_k_select(const ScoredAction *all, int n, int k,
                         ScoredAction *out_sorted) {
    if (n == 0) return 0;
    if (k > n) k = n;

    /* For small k, use a simple heap. Push first k, then for each remaining,
     * if better than min, replace and re-heapify. At end, sort descending. */
    /* Simple approach: copy first k, then for each remaining, check against
     * the current minimum in the heap. Use linear-min for tiny k. */

    /* Copy first k */
    for (int i = 0; i < k; i++) out_sorted[i] = all[i];
    /* Find min in out_sorted[0..k] */
    int min_idx = 0;
    for (int i = 1; i < k; i++) {
        if (out_sorted[i].logit < out_sorted[min_idx].logit) min_idx = i;
    }
    /* For each remaining */
    for (int i = k; i < n; i++) {
        if (all[i].logit > out_sorted[min_idx].logit) {
            out_sorted[min_idx] = all[i];
            /* Recompute min */
            min_idx = 0;
            for (int j = 1; j < k; j++) {
                if (out_sorted[j].logit < out_sorted[min_idx].logit) min_idx = j;
            }
        }
    }
    /* Sort the top-k descending */
    qsort(out_sorted, k, sizeof(ScoredAction), scored_cmp_desc);
    return k;
}

int policy_top_k(const StateEncoderC *enc, const NNModel *m,
                 const Game *g, const Action *actions, int n_actions,
                 int k, int *out_indices,
                 float *nf, float *ef, float *ff, float *mk, float *out) {
    if (n_actions <= 0 || k <= 0) return 0;

    /* 1. Encode state */
    encode_state_full(enc, g, nf, ef, ff);

    /* 2. Build action mask from legal actions, tracking each action's
     *    policy index alongside. We compute scored entries directly. */
    memset(mk, 0, NN_MASK_DIM * sizeof(float));

    ScoredAction scored[256];  /* MAX_ACTIONS = 128 in catanatron */
    int n_scored = 0;
    int policy_idx_per_action[256];  /* policy index per action (-1 if unencodable) */
    for (int i = 0; i < n_actions && i < 256; i++) {
        int pidx = policy_action_encode(m, &actions[i]);
        policy_idx_per_action[i] = pidx;
        if (pidx >= 0 && pidx < NN_MASK_DIM) {
            mk[pidx] = 1.0f;
        }
    }

    /* 3. Run NN forward (uses nf, ef, ff, mk; writes to out as NNOutput) */
    NNOutput *nn_out = (NNOutput *)out;
    nn_forward(m,
               (const float (*)[NN_NODE_FEAT])nf,
               (const float (*)[NN_EDGE_FEAT])ef,
               ff, mk, nn_out);

    /* 4. Score each legal action by its policy logit */
    for (int i = 0; i < n_actions; i++) {
        int pidx = policy_idx_per_action[i];
        if (pidx >= 0 && pidx < POLICY_AD) {
            scored[n_scored].logit = nn_out->policy[pidx];
        } else {
            scored[n_scored].logit = -1e9f;
        }
        scored[n_scored].idx = i;
        n_scored++;
    }

    /* 5. Top-K selection */
    if (k > n_scored) k = n_scored;
    if (k > 64) k = 64;
    ScoredAction selected[64];
    int n_top = top_k_select(scored, n_scored, k, selected);
    for (int i = 0; i < n_top; i++) {
        out_indices[i] = selected[i].idx;
    }
    return n_top;
}
