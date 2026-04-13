/*
 * nn.c — Highly optimized pure C inference for HumanBotNet (~600K params).
 *
 * Apple Accelerate BLAS (AMX coprocessor) for batched GEMMs in GNN.
 * ARM NEON SIMD for vectorized mish, 4-row matvec, fused BN+activation.
 * Scalar fallback for non-ARM architectures.
 */

#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define USE_BLAS 1
#elif defined(HAVE_CBLAS)
#include <cblas.h>
#define USE_BLAS 1
#else
#define USE_BLAS 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#else
#define USE_NEON 0
#endif

/* ================================================================
 * NEON fast exp and mish
 * ================================================================ */

#if USE_NEON

static inline float32x4_t neon_exp(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.33f));
    x = vminq_f32(x, vdupq_n_f32(88.72f));

    const float32x4_t inv_ln2 = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);

    float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_ln2));
    float32x4_t r = vfmsq_f32(x, n, ln2);

    /* Horner: exp(r) = 1 + r + r^2/2! + ... + r^6/6! for |r| <= ln2/2 */
    float32x4_t p = vdupq_n_f32(1.0f / 720.0f);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);

    int32x4_t ni = vcvtq_s32_f32(n);
    ni = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(ni));
}

/* mish(x) = x * tanh(softplus(x)) = x*(u²-1)/(u²+1), u = 1+exp(x) */
static inline float32x4_t neon_mish(float32x4_t x) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t u = vaddq_f32(one, neon_exp(x));
    float32x4_t u2 = vmulq_f32(u, u);
    float32x4_t result = vmulq_f32(x, vdivq_f32(vsubq_f32(u2, one),
                                                  vaddq_f32(u2, one)));
    result = vbslq_f32(vcgtq_f32(x, vdupq_n_f32(6.0f)), x, result);
    result = vbslq_f32(vcltq_f32(x, vdupq_n_f32(-6.0f)),
                       vdupq_n_f32(0.0f), result);
    return result;
}

#endif /* USE_NEON */

/* Scalar mish: single exp instead of log+exp+tanh */
static inline float mish_fast(float x) {
    if (x > 6.0f) return x;
    if (x < -6.0f) return 0.0f;
    float u = 1.0f + expf(x);
    float u2 = u * u;
    return x * (u2 - 1.0f) / (u2 + 1.0f);
}

/* ================================================================
 * SIMD math primitives
 * ================================================================ */

#if USE_NEON

static inline float dot_neon(const float *a, const float *b, int n) {
    float32x4_t acc = vdupq_n_f32(0);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc = vfmaq_f32(acc, vld1q_f32(a+i),    vld1q_f32(b+i));
        acc = vfmaq_f32(acc, vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
        acc = vfmaq_f32(acc, vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
        acc = vfmaq_f32(acc, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    for (; i + 3 < n; i += 4)
        acc = vfmaq_f32(acc, vld1q_f32(a+i), vld1q_f32(b+i));
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* 4-row matvec: shares input vector load across 4 output rows */
static void matvec(float *out, const float *W, const float *x,
                   const float *bias, int rows, int cols) {
    int r = 0;
    for (; r + 3 < rows; r += 4) {
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        const float *w0 = W + r*cols, *w1 = W + (r+1)*cols;
        const float *w2 = W + (r+2)*cols, *w3 = W + (r+3)*cols;
        int c = 0;
        for (; c + 3 < cols; c += 4) {
            float32x4_t v = vld1q_f32(x + c);
            a0 = vfmaq_f32(a0, vld1q_f32(w0+c), v);
            a1 = vfmaq_f32(a1, vld1q_f32(w1+c), v);
            a2 = vfmaq_f32(a2, vld1q_f32(w2+c), v);
            a3 = vfmaq_f32(a3, vld1q_f32(w3+c), v);
        }
        out[r]   = (bias ? bias[r]   : 0.0f) + vaddvq_f32(a0);
        out[r+1] = (bias ? bias[r+1] : 0.0f) + vaddvq_f32(a1);
        out[r+2] = (bias ? bias[r+2] : 0.0f) + vaddvq_f32(a2);
        out[r+3] = (bias ? bias[r+3] : 0.0f) + vaddvq_f32(a3);
        for (; c < cols; c++) {
            float xc = x[c];
            out[r]   += w0[c] * xc;
            out[r+1] += w1[c] * xc;
            out[r+2] += w2[c] * xc;
            out[r+3] += w3[c] * xc;
        }
    }
    for (; r < rows; r++)
        out[r] = (bias ? bias[r] : 0.0f) + dot_neon(W + r*cols, x, cols);
}

static void vec_scale_shift(float *x, const float *scale, const float *shift, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(x+i, vfmaq_f32(vld1q_f32(shift+i), vld1q_f32(x+i), vld1q_f32(scale+i)));
    for (; i < n; i++) x[i] = x[i] * scale[i] + shift[i];
}

static void vec_add(float *dst, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4)
        vst1q_f32(dst+i, vaddq_f32(vld1q_f32(a+i), vld1q_f32(b+i)));
    for (; i < n; i++) dst[i] = a[i] + b[i];
}

static void scatter_add(float *agg, const float *msg, int dim) {
    int i = 0;
    for (; i + 3 < dim; i += 4)
        vst1q_f32(agg+i, vaddq_f32(vld1q_f32(agg+i), vld1q_f32(msg+i)));
    for (; i < dim; i++) agg[i] += msg[i];
}

#else /* scalar fallback */

static void matvec(float *out, const float *W, const float *x,
                   const float *bias, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float s = bias ? bias[r] : 0.0f;
        const float *row = W + r * cols;
        for (int c = 0; c < cols; c++) s += row[c] * x[c];
        out[r] = s;
    }
}

static void vec_scale_shift(float *x, const float *s, const float *sh, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i]*s[i] + sh[i];
}

static void vec_add(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

static void scatter_add(float *agg, const float *msg, int dim) {
    for (int i = 0; i < dim; i++) agg[i] += msg[i];
}

#endif

/* Portable dot product: uses NEON when available, scalar otherwise */
static inline float dot_product(const float *a, const float *b, int n) {
#if USE_NEON
    return dot_neon(a, b, n);
#else
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* ================================================================
 * Activations (NEON-vectorized where possible)
 * ================================================================ */

static void apply_mish(float *x, int n) {
    int i = 0;
#if USE_NEON
    for (; i + 3 < n; i += 4)
        vst1q_f32(x+i, neon_mish(vld1q_f32(x+i)));
#endif
    for (; i < n; i++) x[i] = mish_fast(x[i]);
}

static void apply_bn_mish(float *x, const float *scale, const float *shift, int n) {
    int i = 0;
#if USE_NEON
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vfmaq_f32(vld1q_f32(shift+i), vld1q_f32(x+i),
                                   vld1q_f32(scale+i));
        vst1q_f32(x+i, neon_mish(v));
    }
#endif
    for (; i < n; i++) {
        float v = x[i] * scale[i] + shift[i];
        x[i] = mish_fast(v);
    }
}

static void add_mish(float *out, const float *a, const float *b, int n) {
    int i = 0;
#if USE_NEON
    for (; i + 3 < n; i += 4)
        vst1q_f32(out+i, neon_mish(vaddq_f32(vld1q_f32(a+i), vld1q_f32(b+i))));
#endif
    for (; i < n; i++) out[i] = mish_fast(a[i] + b[i]);
}

static void apply_layernorm(float *x, const float *w, const float *b, int n) {
#if USE_NEON
    float32x4_t sum4 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < n; i += 4) sum4 = vaddq_f32(sum4, vld1q_f32(x+i));
    float mean = vaddvq_f32(sum4);
    for (; i < n; i++) mean += x[i];
    mean /= n;

    float32x4_t vm = vdupq_n_f32(mean);
    float32x4_t var4 = vdupq_n_f32(0);
    for (i = 0; i + 3 < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(x+i), vm);
        var4 = vfmaq_f32(var4, d, d);
    }
    float var = vaddvq_f32(var4);
    for (; i < n; i++) { float d = x[i] - mean; var += d*d; }
    var /= n;

    float inv = 1.0f / sqrtf(var + 1e-5f);
    float32x4_t vinv = vdupq_n_f32(inv);
    for (i = 0; i + 3 < n; i += 4) {
        float32x4_t d = vmulq_f32(vsubq_f32(vld1q_f32(x+i), vm), vinv);
        vst1q_f32(x+i, vfmaq_f32(vld1q_f32(b+i), d, vld1q_f32(w+i)));
    }
    for (; i < n; i++) x[i] = (x[i] - mean) * inv * w[i] + b[i];
#else
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d*d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * inv * w[i] + b[i];
#endif
}

static void log_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
    float lse = mx + logf(sum);
    for (int i = 0; i < n; i++) x[i] -= lse;
}

/* ================================================================
 * Batched FC: C[M,N] = A[M,K] @ B[N,K]^T + bias[N]
 * Uses Accelerate BLAS for large batches, NEON matvec for small.
 * ================================================================ */

static void batch_fc(float *C, const float *A, const float *B,
                     const float *bias, int M, int N, int K) {
#if USE_BLAS
    if (M >= 4) {
        if (bias) {
            for (int m = 0; m < M; m++)
                memcpy(C + m * N, bias, (size_t)N * sizeof(float));
        } else {
            memset(C, 0, (size_t)M * N * sizeof(float));
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    M, N, K, 1.0f, A, K, B, K, 1.0f, C, N);
        return;
    }
#endif
    for (int m = 0; m < M; m++)
        matvec(C + m * N, B, A + m * K, bias, N, K);
}

/* ================================================================
 * GNN forward — batched with BLAS GEMM
 * ================================================================ */

static void gnn_forward(const NNModel *m,
                        const float nf[NN_NODES][NN_NODE_FEAT],
                        const float ef[NN_MAX_EDGES][NN_EDGE_FEAT],
                        float board_emb[NN_GNN_OUTPUT],
                        float node_emb[NN_NODES][NN_GNN_HIDDEN]) {
    const int N = NN_NODES;
    const int E = m->num_edges;
    const int H = NN_GNN_HIDDEN;

    /* node_proj: (N, 18) -> (N, H) + mish */
    batch_fc(&node_emb[0][0], &nf[0][0], (const float *)m->node_proj_w,
             m->node_proj_b, N, H, NN_NODE_FEAT);
    apply_mish(&node_emb[0][0], N * H);

    /* edge_proj: (E, 5) -> (E, H) */
    float edge_emb[NN_MAX_EDGES][NN_GNN_HIDDEN];
    batch_fc(&edge_emb[0][0], &ef[0][0], (const float *)m->edge_proj_w,
             m->edge_proj_b, E, H, NN_EDGE_FEAT);

    float msg_in[NN_MAX_EDGES][3 * NN_GNN_HIDDEN];
    float msg_tmp[NN_MAX_EDGES][NN_GNN_HIDDEN];
    float msg_out[NN_MAX_EDGES][NN_GNN_HIDDEN];
    float agg[NN_NODES][NN_GNN_HIDDEN];
    float upd_in[NN_NODES][2 * NN_GNN_HIDDEN];
    float upd_tmp[NN_NODES][NN_GNN_HIDDEN];
    float upd_out[NN_NODES][NN_GNN_HIDDEN];

    for (int L = 0; L < NN_GNN_LAYERS; L++) {
        const EdgeConvWeights *lw = &m->gnn_layers[L];

        for (int e = 0; e < E; e++) {
            memcpy(msg_in[e], node_emb[m->edge_src[e]], H * sizeof(float));
            memcpy(msg_in[e] + H, node_emb[m->edge_dst[e]], H * sizeof(float));
            memcpy(msg_in[e] + 2*H, edge_emb[e], H * sizeof(float));
        }

        /* msg_mlp layer 1: (E, 3H) -> (E, H) + mish */
        batch_fc(&msg_tmp[0][0], &msg_in[0][0], (const float *)lw->msg_w1,
                 lw->msg_b1, E, H, 3*H);
        apply_mish(&msg_tmp[0][0], E * H);

        /* msg_mlp layer 2: (E, H) -> (E, H) + mish */
        batch_fc(&msg_out[0][0], &msg_tmp[0][0], (const float *)lw->msg_w2,
                 lw->msg_b2, E, H, H);
        apply_mish(&msg_out[0][0], E * H);

        /* Scatter-add messages to destination nodes */
        memset(agg, 0, sizeof(agg));
        for (int e = 0; e < E; e++)
            scatter_add(agg[m->edge_dst[e]], msg_out[e], H);

        for (int n = 0; n < N; n++) {
            memcpy(upd_in[n], node_emb[n], H * sizeof(float));
            memcpy(upd_in[n] + H, agg[n], H * sizeof(float));
        }

        /* update_mlp layer 1: (N, 2H) -> (N, H) + mish */
        batch_fc(&upd_tmp[0][0], &upd_in[0][0], (const float *)lw->upd_w1,
                 lw->upd_b1, N, H, 2*H);
        apply_mish(&upd_tmp[0][0], N * H);

        /* update_mlp layer 2: (N, H) -> (N, H) */
        batch_fc(&upd_out[0][0], &upd_tmp[0][0], (const float *)lw->upd_w2,
                 lw->upd_b2, N, H, H);

        for (int n = 0; n < N; n++) {
            vec_add(node_emb[n], node_emb[n], upd_out[n], H);
            apply_layernorm(node_emb[n], lw->ln_w, lw->ln_b, H);
        }
    }

    /* Global pooling: mean + max */
    float mean_pool[NN_GNN_HIDDEN], max_pool[NN_GNN_HIDDEN];
    memcpy(mean_pool, node_emb[0], H * sizeof(float));
    memcpy(max_pool, node_emb[0], H * sizeof(float));
    for (int n = 1; n < N; n++) {
#if USE_NEON
        for (int i = 0; i + 3 < H; i += 4) {
            float32x4_t v = vld1q_f32(&node_emb[n][i]);
            vst1q_f32(&mean_pool[i], vaddq_f32(vld1q_f32(&mean_pool[i]), v));
            vst1q_f32(&max_pool[i], vmaxq_f32(vld1q_f32(&max_pool[i]), v));
        }
#else
        for (int i = 0; i < H; i++) {
            mean_pool[i] += node_emb[n][i];
            if (node_emb[n][i] > max_pool[i]) max_pool[i] = node_emb[n][i];
        }
#endif
    }
    float inv_n = 1.0f / N;
    for (int i = 0; i < H; i++) mean_pool[i] *= inv_n;

    float cat[2 * NN_GNN_HIDDEN];
    memcpy(cat, mean_pool, H * sizeof(float));
    memcpy(cat + H, max_pool, H * sizeof(float));

    float tmp[NN_GNN_OUTPUT];
    matvec(tmp, (const float *)m->out_proj_w1, cat, m->out_proj_b1,
           NN_GNN_OUTPUT, 2*H);
    apply_mish(tmp, NN_GNN_OUTPUT);
    matvec(board_emb, (const float *)m->out_proj_w2, tmp, m->out_proj_b2,
           NN_GNN_OUTPUT, NN_GNN_OUTPUT);
}

/* ================================================================
 * Trunk forward
 * ================================================================ */

static void trunk_forward(const NNModel *m,
                          const float board_emb[NN_GNN_OUTPUT],
                          const float flat[NN_FLAT_DIM],
                          const float mask[NN_MASK_DIM],
                          float trunk_out[NN_TRUNK_CH]) {
    float combined[NN_TRUNK_INPUT];
    memcpy(combined, board_emb, NN_GNN_OUTPUT * sizeof(float));
    memcpy(combined + NN_GNN_OUTPUT, flat, NN_FLAT_DIM * sizeof(float));
    memcpy(combined + NN_GNN_OUTPUT + NN_FLAT_DIM, mask, NN_MASK_DIM * sizeof(float));

    matvec(trunk_out, (const float *)m->trunk_ip_w, combined, m->trunk_ip_b,
           NN_TRUNK_CH, NN_TRUNK_INPUT);
    apply_bn_mish(trunk_out, m->trunk_ip_bn.scale, m->trunk_ip_bn.shift, NN_TRUNK_CH);

    float h[NN_TRUNK_CH], h2[NN_TRUNK_CH];
    for (int b = 0; b < NN_TRUNK_BLOCKS; b++) {
        const ResBlockWeights *rb = &m->trunk_blocks[b];
        matvec(h, (const float *)rb->fc1_w, trunk_out, rb->fc1_b,
               NN_TRUNK_CH, NN_TRUNK_CH);
        apply_bn_mish(h, rb->bn1.scale, rb->bn1.shift, NN_TRUNK_CH);
        matvec(h2, (const float *)rb->fc2_w, h, rb->fc2_b,
               NN_TRUNK_CH, NN_TRUNK_CH);
        vec_scale_shift(h2, rb->bn2.scale, rb->bn2.shift, NN_TRUNK_CH);
        add_mish(trunk_out, h2, trunk_out, NN_TRUNK_CH);
    }
}

/* ================================================================
 * Value head
 * ================================================================ */

static void value_head(const NNModel *m, const float trunk[NN_TRUNK_CH],
                       float out[4]) {
    float h[NN_VALUE_HIDDEN];
    matvec(h, (const float *)m->val_fc1_w, trunk, m->val_fc1_b,
           NN_VALUE_HIDDEN, NN_TRUNK_CH);
    apply_bn_mish(h, m->val_bn1.scale, m->val_bn1.shift, NN_VALUE_HIDDEN);

    float t1[NN_VALUE_HIDDEN], t2[NN_VALUE_HIDDEN];
    for (int r = 0; r < 2; r++) {
        const ValResBlockWeights *rb = &m->val_res[r];
        matvec(t1, (const float *)rb->fc1_w, h, rb->fc1_b,
               NN_VALUE_HIDDEN, NN_VALUE_HIDDEN);
        apply_bn_mish(t1, rb->bn1.scale, rb->bn1.shift, NN_VALUE_HIDDEN);
        matvec(t2, (const float *)rb->fc2_w, t1, rb->fc2_b,
               NN_VALUE_HIDDEN, NN_VALUE_HIDDEN);
        vec_scale_shift(t2, rb->bn2.scale, rb->bn2.shift, NN_VALUE_HIDDEN);
        add_mish(h, t2, h, NN_VALUE_HIDDEN);
    }

    matvec(out, (const float *)m->val_out_w, h, m->val_out_b, 4, NN_VALUE_HIDDEN);
}

/* ================================================================
 * Policy head — batched spatial scorers via GEMM
 * ================================================================ */

static void policy_head(const NNModel *m,
                        const float trunk[NN_TRUNK_CH],
                        const float node_emb[NN_NODES][NN_GNN_HIDDEN],
                        float policy[NN_MASK_DIM]) {
    const int TC = NN_TRUNK_CH, H = NN_GNN_HIDDEN, SH = NN_SCORER_HIDDEN;

    float tn[NN_TRUNK_CH];
    memcpy(tn, trunk, TC * sizeof(float));
    apply_layernorm(tn, m->pol_trunk_ln_w, m->pol_trunk_ln_b, TC);

    float nn_[NN_NODES][NN_GNN_HIDDEN];
    for (int n = 0; n < NN_NODES; n++) {
        memcpy(nn_[n], node_emb[n], H * sizeof(float));
        apply_layernorm(nn_[n], m->pol_node_ln_w, m->pol_node_ln_b, H);
    }

    /* Type logits */
    float type_h[NN_POLICY_HIDDEN], type_logits[NN_NUM_TYPES];
    matvec(type_h, (const float *)m->pol_type_w1, tn, m->pol_type_b1,
           NN_POLICY_HIDDEN, TC);
    apply_bn_mish(type_h, m->pol_type_bn.scale, m->pol_type_bn.shift,
                  NN_POLICY_HIDDEN);
    matvec(type_logits, (const float *)m->pol_type_w2, type_h, m->pol_type_b2,
           NN_NUM_TYPES, NN_POLICY_HIDDEN);

    float log_type[NN_NUM_TYPES];
    memcpy(log_type, type_logits, sizeof(log_type));
    log_softmax(log_type, NN_NUM_TYPES);

    policy[0] = log_type[0]; policy[1] = log_type[1]; policy[2] = log_type[2];
    policy[3] = log_type[3]; policy[4] = log_type[4];

    /* Spatial scorer helper: compute trunk contribution from strided weight */
    #define TRUNK_CONTRIB(out, W, stride, bias, tn, sh_dim, tc_dim) do { \
        for (int _s = 0; _s < (sh_dim); _s++) \
            (out)[_s] = (bias)[_s] + dot_product((W) + _s*(stride), tn, tc_dim); \
    } while(0)

    /* Spatial scorer helper: add tn_part to each row + mish */
    #define ADD_TRUNK_MISH(sh, tn_part, rows, sh_dim) do { \
        for (int _r = 0; _r < (rows); _r++) { \
            int _i = 0; \
            IF_NEON(for (; _i + 3 < (sh_dim); _i += 4) \
                vst1q_f32(&(sh)[_r][_i], neon_mish(vaddq_f32( \
                    vld1q_f32(&(sh)[_r][_i]), vld1q_f32(&(tn_part)[_i]))));) \
            for (; _i < (sh_dim); _i++) \
                (sh)[_r][_i] = mish_fast((sh)[_r][_i] + (tn_part)[_i]); \
        } \
    } while(0)

#if USE_NEON
    #define IF_NEON(code) code
#else
    #define IF_NEON(code)
#endif

    /* Settlement (54 nodes) */
    {
        const float *W = (const float *)m->pol_sett_w1;
        const int S = TC + H;
        float tn_part[NN_SCORER_HIDDEN];
        TRUNK_CONTRIB(tn_part, W, S, m->pol_sett_b1, tn, SH, TC);

        float sh[NN_NODES][NN_SCORER_HIDDEN];
#if USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    NN_NODES, SH, H, 1.0f, &nn_[0][0], H,
                    W + TC, S, 0.0f, &sh[0][0], SH);
        ADD_TRUNK_MISH(sh, tn_part, NN_NODES, SH);
#else
        {
            float ctx[NN_NODES][NN_TRUNK_CH + NN_GNN_HIDDEN];
            for (int n = 0; n < NN_NODES; n++) {
                memcpy(ctx[n], tn, TC * sizeof(float));
                memcpy(ctx[n] + TC, nn_[n], H * sizeof(float));
            }
            batch_fc(&sh[0][0], &ctx[0][0], W, m->pol_sett_b1, NN_NODES, SH, S);
            apply_mish(&sh[0][0], NN_NODES * SH);
        }
#endif
        float raw[NN_NODES];
        for (int n = 0; n < NN_NODES; n++)
            raw[n] = dot_product((const float *)m->pol_sett_w2, sh[n], SH)
                     + m->pol_sett_b2[0];
        log_softmax(raw, NN_NODES);
        for (int n = 0; n < NN_NODES; n++) policy[5+n] = log_type[5] + raw[n];
    }

    /* City (54 nodes) */
    {
        const float *W = (const float *)m->pol_city_w1;
        const int S = TC + H;
        float tn_part[NN_SCORER_HIDDEN];
        TRUNK_CONTRIB(tn_part, W, S, m->pol_city_b1, tn, SH, TC);

        float sh[NN_NODES][NN_SCORER_HIDDEN];
#if USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    NN_NODES, SH, H, 1.0f, &nn_[0][0], H,
                    W + TC, S, 0.0f, &sh[0][0], SH);
        ADD_TRUNK_MISH(sh, tn_part, NN_NODES, SH);
#else
        {
            float ctx[NN_NODES][NN_TRUNK_CH + NN_GNN_HIDDEN];
            for (int n = 0; n < NN_NODES; n++) {
                memcpy(ctx[n], tn, TC * sizeof(float));
                memcpy(ctx[n] + TC, nn_[n], H * sizeof(float));
            }
            batch_fc(&sh[0][0], &ctx[0][0], W, m->pol_city_b1, NN_NODES, SH, S);
            apply_mish(&sh[0][0], NN_NODES * SH);
        }
#endif
        float raw[NN_NODES];
        for (int n = 0; n < NN_NODES; n++)
            raw[n] = dot_product((const float *)m->pol_city_w2, sh[n], SH)
                     + m->pol_city_b2[0];
        log_softmax(raw, NN_NODES);
        for (int n = 0; n < NN_NODES; n++) policy[59+n] = log_type[6] + raw[n];
    }

    /* Road (72 edges) */
    {
        const float *W = (const float *)m->pol_road_w1;
        const int S = TC + 2*H;

        float road_src[72][NN_GNN_HIDDEN], road_dst[72][NN_GNN_HIDDEN];
        for (int r = 0; r < 72; r++) {
            memcpy(road_src[r], nn_[m->road_pairs[r][0]], H * sizeof(float));
            memcpy(road_dst[r], nn_[m->road_pairs[r][1]], H * sizeof(float));
        }
        float sh[72][NN_SCORER_HIDDEN];
#if USE_BLAS
        {
            float tn_part[NN_SCORER_HIDDEN];
            TRUNK_CONTRIB(tn_part, W, S, m->pol_road_b1, tn, SH, TC);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        72, SH, H, 1.0f, &road_src[0][0], H,
                        W + TC, S, 0.0f, &sh[0][0], SH);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        72, SH, H, 1.0f, &road_dst[0][0], H,
                        W + TC + H, S, 1.0f, &sh[0][0], SH);
            ADD_TRUNK_MISH(sh, tn_part, 72, SH);
        }
#else
        {
            float ctx[72][NN_TRUNK_CH + 2*NN_GNN_HIDDEN];
            for (int r = 0; r < 72; r++) {
                memcpy(ctx[r], tn, TC * sizeof(float));
                memcpy(ctx[r]+TC, road_src[r], H * sizeof(float));
                memcpy(ctx[r]+TC+H, road_dst[r], H * sizeof(float));
            }
            batch_fc(&sh[0][0], &ctx[0][0], W, m->pol_road_b1, 72, SH, S);
            apply_mish(&sh[0][0], 72 * SH);
        }
#endif
        float raw[72];
        for (int r = 0; r < 72; r++)
            raw[r] = dot_product((const float *)m->pol_road_w2, sh[r], SH)
                     + m->pol_road_b2[0];
        log_softmax(raw, 72);
        for (int r = 0; r < 72; r++) policy[113+r] = log_type[7] + raw[r];
    }

    /* Robber (19 tiles x 5) */
    {
        const float *W = (const float *)m->pol_rob_w1;
        const int S = TC + H;

        float tile_emb[19][NN_GNN_HIDDEN];
        for (int t = 0; t < 19; t++) {
            memset(tile_emb[t], 0, H * sizeof(float));
            for (int k = 0; k < 6; k++)
                scatter_add(tile_emb[t], nn_[m->tile_nodes[t][k]], H);
            for (int i = 0; i < H; i++) tile_emb[t][i] *= (1.0f / 6.0f);
        }
        float sh[19][NN_SCORER_HIDDEN];
#if USE_BLAS
        {
            float tn_part[NN_SCORER_HIDDEN];
            TRUNK_CONTRIB(tn_part, W, S, m->pol_rob_b1, tn, SH, TC);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        19, SH, H, 1.0f, &tile_emb[0][0], H,
                        W + TC, S, 0.0f, &sh[0][0], SH);
            ADD_TRUNK_MISH(sh, tn_part, 19, SH);
        }
#else
        {
            float tile_ctx[19][NN_TRUNK_CH + NN_GNN_HIDDEN];
            for (int t = 0; t < 19; t++) {
                memcpy(tile_ctx[t], tn, TC * sizeof(float));
                memcpy(tile_ctx[t] + TC, tile_emb[t], H * sizeof(float));
            }
            batch_fc(&sh[0][0], &tile_ctx[0][0], W, m->pol_rob_b1, 19, SH, S);
            apply_mish(&sh[0][0], 19 * SH);
        }
#endif
        float raw[95];
        batch_fc(raw, &sh[0][0], (const float *)m->pol_rob_w2,
                 m->pol_rob_b2, 19, 5, SH);
        log_softmax(raw, 95);
        for (int i = 0; i < 95; i++) policy[185+i] = log_type[8] + raw[i];
    }

#undef TRUNK_CONTRIB
#undef ADD_TRUNK_MISH
#undef IF_NEON

    /* Non-spatial grouped heads */
    {
        float sh[NN_SCORER_HIDDEN], raw[30];
        matvec(sh, (const float *)m->pol_dym_w1, tn, m->pol_dym_b1, SH, TC);
        apply_mish(sh, SH);
        matvec(raw, (const float *)m->pol_dym_w2, sh, m->pol_dym_b2, 30, SH);
        log_softmax(raw, 30);
        for (int i = 0; i < 30; i++) policy[280+i] = log_type[9] + raw[i];
    }
    {
        float sh[NN_SCORER_HIDDEN], raw[20];
        matvec(sh, (const float *)m->pol_mar_w1, tn, m->pol_mar_b1, SH, TC);
        apply_mish(sh, SH);
        matvec(raw, (const float *)m->pol_mar_w2, sh, m->pol_mar_b2, 20, SH);
        log_softmax(raw, 20);
        for (int i = 0; i < 20; i++) policy[310+i] = log_type[10] + raw[i];
    }
    {
        float sh[NN_SCORER_HIDDEN], raw[67];
        matvec(sh, (const float *)m->pol_trd_w1, tn, m->pol_trd_b1, SH, TC);
        apply_mish(sh, SH);
        matvec(raw, (const float *)m->pol_trd_w2, sh, m->pol_trd_b2, 67, SH);
        log_softmax(raw, 67);
        for (int i = 0; i < 67; i++) policy[330+i] = log_type[11] + raw[i];
    }
}

/* ================================================================
 * Public API
 * ================================================================ */

static int read_exact(FILE *f, void *buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

int nn_load(NNModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nn_load: cannot open %s\n", path); return -1; }

    char magic[4];
    uint32_t hdr[15];
    if (read_exact(f, magic, 4) || memcmp(magic, "HBOT", 4) != 0) goto fail;
    if (read_exact(f, hdr, sizeof(hdr))) goto fail;

    m->num_edges = (int)hdr[2];

    /* Topology */
    {
        int pairs[NN_MAX_EDGES][2];
        if (read_exact(f, pairs, m->num_edges * 2 * sizeof(int))) goto fail;
        for (int i = 0; i < m->num_edges; i++) {
            m->edge_src[i] = pairs[i][0];
            m->edge_dst[i] = pairs[i][1];
        }
    }
    if (read_exact(f, m->road_pairs, sizeof(m->road_pairs))) goto fail;
    if (read_exact(f, m->tile_nodes, sizeof(m->tile_nodes))) goto fail;
    if (read_exact(f, m->land_nodes, sizeof(m->land_nodes))) goto fail;
    if (read_exact(f, m->node_to_compact, sizeof(m->node_to_compact))) goto fail;
    if (read_exact(f, m->edge_lut, sizeof(m->edge_lut))) goto fail;
    if (read_exact(f, m->coord_to_tile, sizeof(m->coord_to_tile))) goto fail;
    if (read_exact(f, m->mar_lut, sizeof(m->mar_lut))) goto fail;
    if (read_exact(f, m->idx_to_edge, sizeof(m->idx_to_edge))) goto fail;

    /* GNN */
    if (read_exact(f, m->node_proj_w, sizeof(m->node_proj_w))) goto fail;
    if (read_exact(f, m->node_proj_b, sizeof(m->node_proj_b))) goto fail;
    if (read_exact(f, m->edge_proj_w, sizeof(m->edge_proj_w))) goto fail;
    if (read_exact(f, m->edge_proj_b, sizeof(m->edge_proj_b))) goto fail;
    for (int i = 0; i < NN_GNN_LAYERS; i++)
        if (read_exact(f, &m->gnn_layers[i], sizeof(EdgeConvWeights))) goto fail;
    if (read_exact(f, m->out_proj_w1, sizeof(m->out_proj_w1))) goto fail;
    if (read_exact(f, m->out_proj_b1, sizeof(m->out_proj_b1))) goto fail;
    if (read_exact(f, m->out_proj_w2, sizeof(m->out_proj_w2))) goto fail;
    if (read_exact(f, m->out_proj_b2, sizeof(m->out_proj_b2))) goto fail;

    /* Trunk */
    if (read_exact(f, m->trunk_ip_w, sizeof(m->trunk_ip_w))) goto fail;
    if (read_exact(f, m->trunk_ip_b, sizeof(m->trunk_ip_b))) goto fail;
    if (read_exact(f, &m->trunk_ip_bn, sizeof(BN128))) goto fail;
    for (int i = 0; i < NN_TRUNK_BLOCKS; i++)
        if (read_exact(f, &m->trunk_blocks[i], sizeof(ResBlockWeights))) goto fail;

    /* Value head */
    if (read_exact(f, m->val_fc1_w, sizeof(m->val_fc1_w))) goto fail;
    if (read_exact(f, m->val_fc1_b, sizeof(m->val_fc1_b))) goto fail;
    if (read_exact(f, &m->val_bn1, sizeof(BNV))) goto fail;
    for (int i = 0; i < 2; i++)
        if (read_exact(f, &m->val_res[i], sizeof(ValResBlockWeights))) goto fail;
    if (read_exact(f, m->val_out_w, sizeof(m->val_out_w))) goto fail;
    if (read_exact(f, m->val_out_b, sizeof(m->val_out_b))) goto fail;

    /* Policy head */
    if (read_exact(f, m->pol_trunk_ln_w, sizeof(m->pol_trunk_ln_w))) goto fail;
    if (read_exact(f, m->pol_trunk_ln_b, sizeof(m->pol_trunk_ln_b))) goto fail;
    if (read_exact(f, m->pol_node_ln_w, sizeof(m->pol_node_ln_w))) goto fail;
    if (read_exact(f, m->pol_node_ln_b, sizeof(m->pol_node_ln_b))) goto fail;
    if (read_exact(f, m->pol_type_w1, sizeof(m->pol_type_w1))) goto fail;
    if (read_exact(f, m->pol_type_b1, sizeof(m->pol_type_b1))) goto fail;
    if (read_exact(f, &m->pol_type_bn, sizeof(BNP))) goto fail;
    if (read_exact(f, m->pol_type_w2, sizeof(m->pol_type_w2))) goto fail;
    if (read_exact(f, m->pol_type_b2, sizeof(m->pol_type_b2))) goto fail;

    if (read_exact(f, m->pol_dym_w1, sizeof(m->pol_dym_w1))) goto fail;
    if (read_exact(f, m->pol_dym_b1, sizeof(m->pol_dym_b1))) goto fail;
    if (read_exact(f, m->pol_dym_w2, sizeof(m->pol_dym_w2))) goto fail;
    if (read_exact(f, m->pol_dym_b2, sizeof(m->pol_dym_b2))) goto fail;
    if (read_exact(f, m->pol_mar_w1, sizeof(m->pol_mar_w1))) goto fail;
    if (read_exact(f, m->pol_mar_b1, sizeof(m->pol_mar_b1))) goto fail;
    if (read_exact(f, m->pol_mar_w2, sizeof(m->pol_mar_w2))) goto fail;
    if (read_exact(f, m->pol_mar_b2, sizeof(m->pol_mar_b2))) goto fail;
    if (read_exact(f, m->pol_trd_w1, sizeof(m->pol_trd_w1))) goto fail;
    if (read_exact(f, m->pol_trd_b1, sizeof(m->pol_trd_b1))) goto fail;
    if (read_exact(f, m->pol_trd_w2, sizeof(m->pol_trd_w2))) goto fail;
    if (read_exact(f, m->pol_trd_b2, sizeof(m->pol_trd_b2))) goto fail;

    if (read_exact(f, m->pol_sett_w1, sizeof(m->pol_sett_w1))) goto fail;
    if (read_exact(f, m->pol_sett_b1, sizeof(m->pol_sett_b1))) goto fail;
    if (read_exact(f, m->pol_sett_w2, sizeof(m->pol_sett_w2))) goto fail;
    if (read_exact(f, m->pol_sett_b2, sizeof(m->pol_sett_b2))) goto fail;
    if (read_exact(f, m->pol_city_w1, sizeof(m->pol_city_w1))) goto fail;
    if (read_exact(f, m->pol_city_b1, sizeof(m->pol_city_b1))) goto fail;
    if (read_exact(f, m->pol_city_w2, sizeof(m->pol_city_w2))) goto fail;
    if (read_exact(f, m->pol_city_b2, sizeof(m->pol_city_b2))) goto fail;
    if (read_exact(f, m->pol_road_w1, sizeof(m->pol_road_w1))) goto fail;
    if (read_exact(f, m->pol_road_b1, sizeof(m->pol_road_b1))) goto fail;
    if (read_exact(f, m->pol_road_w2, sizeof(m->pol_road_w2))) goto fail;
    if (read_exact(f, m->pol_road_b2, sizeof(m->pol_road_b2))) goto fail;
    if (read_exact(f, m->pol_rob_w1, sizeof(m->pol_rob_w1))) goto fail;
    if (read_exact(f, m->pol_rob_b1, sizeof(m->pol_rob_b1))) goto fail;
    if (read_exact(f, m->pol_rob_w2, sizeof(m->pol_rob_w2))) goto fail;
    if (read_exact(f, m->pol_rob_b2, sizeof(m->pol_rob_b2))) goto fail;

    fclose(f);
    return 0;

fail:
    fprintf(stderr, "nn_load: read error in %s at offset %ld\n", path, ftell(f));
    fclose(f);
    return -1;
}

void nn_forward(const NNModel *m,
                const float node_feat[NN_NODES][NN_NODE_FEAT],
                const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                const float flat_feat[NN_FLAT_DIM],
                const float mask[NN_MASK_DIM],
                NNOutput *out) {
    float board_emb[NN_GNN_OUTPUT];
    float node_emb[NN_NODES][NN_GNN_HIDDEN];
    float trunk_out[NN_TRUNK_CH];

    gnn_forward(m, node_feat, edge_feat, board_emb, node_emb);
    trunk_forward(m, board_emb, flat_feat, mask, trunk_out);
    value_head(m, trunk_out, out->value);
    policy_head(m, trunk_out, node_emb, out->policy);

    for (int i = 0; i < NN_MASK_DIM; i++)
        if (mask[i] < 0.5f) out->policy[i] = -1e9f;
}

void nn_value_only(const NNModel *m,
                   const float node_feat[NN_NODES][NN_NODE_FEAT],
                   const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                   const float flat_feat[NN_FLAT_DIM],
                   const float mask[NN_MASK_DIM],
                   float value_out[4]) {
    float board_emb[NN_GNN_OUTPUT];
    float node_emb[NN_NODES][NN_GNN_HIDDEN];
    float trunk_out[NN_TRUNK_CH];

    gnn_forward(m, node_feat, edge_feat, board_emb, node_emb);
    trunk_forward(m, board_emb, flat_feat, mask, trunk_out);
    value_head(m, trunk_out, value_out);
}
