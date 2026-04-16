#ifndef NN_H
#define NN_H

#include <stdint.h>

/* Architecture constants (must match export) */
#define NN_NODES        54
#define NN_MAX_EDGES    144
#define NN_GNN_HIDDEN   80
#define NN_GNN_OUTPUT   128
#define NN_GNN_LAYERS   4
#define NN_TRUNK_CH     192
#define NN_TRUNK_BLOCKS 6
#define NN_VALUE_HIDDEN 128
#define NN_FLAT_DIM     115
#define NN_MASK_DIM     397
#define NN_NODE_FEAT    18
#define NN_EDGE_FEAT    5
#define NN_TRUNK_INPUT  (NN_GNN_OUTPUT + NN_FLAT_DIM + NN_MASK_DIM)  /* 640 */
#define NN_POLICY_HIDDEN 128
#define NN_SCORER_HIDDEN 48
#define NN_NUM_TYPES    12

/* Fused BN: y = x * scale + shift (precomputed from weight/bias/mean/var) */
typedef struct { float scale[NN_TRUNK_CH]; float shift[NN_TRUNK_CH]; } BN128;
typedef struct { float scale[NN_VALUE_HIDDEN]; float shift[NN_VALUE_HIDDEN]; } BNV;
typedef struct { float scale[NN_POLICY_HIDDEN]; float shift[NN_POLICY_HIDDEN]; } BNP;

typedef struct {
    /* msg_mlp: Linear(192,64) + Mish + Linear(64,64) + Mish */
    float msg_w1[NN_GNN_HIDDEN][3*NN_GNN_HIDDEN];
    float msg_b1[NN_GNN_HIDDEN];
    float msg_w2[NN_GNN_HIDDEN][NN_GNN_HIDDEN];
    float msg_b2[NN_GNN_HIDDEN];
    /* update_mlp: Linear(128,64) + Mish + Linear(64,64) */
    float upd_w1[NN_GNN_HIDDEN][2*NN_GNN_HIDDEN];
    float upd_b1[NN_GNN_HIDDEN];
    float upd_w2[NN_GNN_HIDDEN][NN_GNN_HIDDEN];
    float upd_b2[NN_GNN_HIDDEN];
    /* LayerNorm(64) */
    float ln_w[NN_GNN_HIDDEN];
    float ln_b[NN_GNN_HIDDEN];
} EdgeConvWeights;

typedef struct {
    float fc1_w[NN_TRUNK_CH][NN_TRUNK_CH];
    float fc1_b[NN_TRUNK_CH];
    BN128 bn1;
    float fc2_w[NN_TRUNK_CH][NN_TRUNK_CH];
    float fc2_b[NN_TRUNK_CH];
    BN128 bn2;
} ResBlockWeights;

typedef struct {
    float fc1_w[NN_VALUE_HIDDEN][NN_VALUE_HIDDEN];
    float fc1_b[NN_VALUE_HIDDEN];
    BNV bn1;
    float fc2_w[NN_VALUE_HIDDEN][NN_VALUE_HIDDEN];
    float fc2_b[NN_VALUE_HIDDEN];
    BNV bn2;
} ValResBlockWeights;

typedef struct {
    /* Topology */
    int num_edges;
    int edge_src[NN_MAX_EDGES];
    int edge_dst[NN_MAX_EDGES];
    int road_pairs[72][2];
    int tile_nodes[19][6];
    int land_nodes[NN_NODES];
    int node_to_compact[96];
    int edge_lut[96][96];
    int coord_to_tile[7][7][7];
    int mar_lut[5][5];
    int idx_to_edge[72][2];

    /* GNN encoder */
    float node_proj_w[NN_GNN_HIDDEN][NN_NODE_FEAT];
    float node_proj_b[NN_GNN_HIDDEN];
    float edge_proj_w[NN_GNN_HIDDEN][NN_EDGE_FEAT];
    float edge_proj_b[NN_GNN_HIDDEN];
    EdgeConvWeights gnn_layers[NN_GNN_LAYERS];
    float out_proj_w1[NN_GNN_OUTPUT][2*NN_GNN_HIDDEN];
    float out_proj_b1[NN_GNN_OUTPUT];
    float out_proj_w2[NN_GNN_OUTPUT][NN_GNN_OUTPUT];
    float out_proj_b2[NN_GNN_OUTPUT];

    /* Trunk */
    float trunk_ip_w[NN_TRUNK_CH][NN_TRUNK_INPUT];
    float trunk_ip_b[NN_TRUNK_CH];
    BN128 trunk_ip_bn;
    ResBlockWeights trunk_blocks[NN_TRUNK_BLOCKS];

    /* Value head */
    float val_fc1_w[NN_VALUE_HIDDEN][NN_TRUNK_CH];
    float val_fc1_b[NN_VALUE_HIDDEN];
    BNV val_bn1;
    ValResBlockWeights val_res[2];
    float val_out_w[4][NN_VALUE_HIDDEN];
    float val_out_b[4];

    /* Policy head */
    float pol_trunk_ln_w[NN_TRUNK_CH];
    float pol_trunk_ln_b[NN_TRUNK_CH];
    float pol_node_ln_w[NN_GNN_HIDDEN];
    float pol_node_ln_b[NN_GNN_HIDDEN];
    float pol_type_w1[NN_POLICY_HIDDEN][NN_TRUNK_CH];
    float pol_type_b1[NN_POLICY_HIDDEN];
    BNP   pol_type_bn;
    float pol_type_w2[NN_NUM_TYPES][NN_POLICY_HIDDEN];
    float pol_type_b2[NN_NUM_TYPES];
    /* Sub-action heads: discard_yop_mono(30), maritime(20), trade(67) */
    float pol_dym_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH]; float pol_dym_b1[NN_SCORER_HIDDEN];
    float pol_dym_w2[30][NN_SCORER_HIDDEN]; float pol_dym_b2[30];
    float pol_mar_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH]; float pol_mar_b1[NN_SCORER_HIDDEN];
    float pol_mar_w2[20][NN_SCORER_HIDDEN]; float pol_mar_b2[20];
    float pol_trd_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH]; float pol_trd_b1[NN_SCORER_HIDDEN];
    float pol_trd_w2[67][NN_SCORER_HIDDEN]; float pol_trd_b2[67];
    /* Spatial scorers: sett(192->48->1), city(192->48->1), road(256->48->1), robber(192->48->5) */
    float pol_sett_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH+NN_GNN_HIDDEN]; float pol_sett_b1[NN_SCORER_HIDDEN];
    float pol_sett_w2[1][NN_SCORER_HIDDEN]; float pol_sett_b2[1];
    float pol_city_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH+NN_GNN_HIDDEN]; float pol_city_b1[NN_SCORER_HIDDEN];
    float pol_city_w2[1][NN_SCORER_HIDDEN]; float pol_city_b2[1];
    float pol_road_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH+2*NN_GNN_HIDDEN]; float pol_road_b1[NN_SCORER_HIDDEN];
    float pol_road_w2[1][NN_SCORER_HIDDEN]; float pol_road_b2[1];
    float pol_rob_w1[NN_SCORER_HIDDEN][NN_TRUNK_CH+NN_GNN_HIDDEN]; float pol_rob_b1[NN_SCORER_HIDDEN];
    float pol_rob_w2[5][NN_SCORER_HIDDEN]; float pol_rob_b2[5];

} NNModel;

/* Inference output */
typedef struct {
    float value[4];
    float policy[NN_MASK_DIM];
} NNOutput;

/* Load model from binary file. Returns 0 on success. */
int nn_load(NNModel *m, const char *path);

/* Run forward pass (batch_size=1). */
void nn_forward(const NNModel *m,
                const float node_feat[NN_NODES][NN_NODE_FEAT],
                const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                const float flat_feat[NN_FLAT_DIM],
                const float mask[NN_MASK_DIM],
                NNOutput *out);

/* Value-only forward (skips policy head, faster). */
void nn_value_only(const NNModel *m,
                   const float node_feat[NN_NODES][NN_NODE_FEAT],
                   const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                   const float flat_feat[NN_FLAT_DIM],
                   const float mask[NN_MASK_DIM],
                   float value_out[4]);

#endif
