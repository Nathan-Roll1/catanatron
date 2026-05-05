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
#include <stdint.h>

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

#if USE_NEON && defined(__ARM_FEATURE_DOTPROD)
#define USE_NEON_DOT 1
#else
#define USE_NEON_DOT 0
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
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

/* ================================================================
 * AVX2+FMA fast exp and mish (8-wide, float32)
 * ================================================================ */

#if USE_AVX2

static inline __m256 avx2_exp(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.33f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.72f));

    const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);

    __m256 n = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);

    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    __m256i ni = _mm256_cvtps_epi32(n);
    ni = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ni));
}

static inline __m256 avx2_mish(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 six = _mm256_set1_ps(6.0f);
    const __m256 neg_six = _mm256_set1_ps(-6.0f);
    const __m256 zero = _mm256_setzero_ps();

    __m256 u = _mm256_add_ps(one, avx2_exp(x));
    __m256 u2 = _mm256_mul_ps(u, u);
    __m256 result = _mm256_mul_ps(x,
        _mm256_div_ps(_mm256_sub_ps(u2, one), _mm256_add_ps(u2, one)));

    result = _mm256_blendv_ps(result, x, _mm256_cmp_ps(x, six, _CMP_GT_OQ));
    result = _mm256_blendv_ps(result, zero, _mm256_cmp_ps(x, neg_six, _CMP_LT_OQ));
    return result;
}

#endif /* USE_AVX2 */

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

/* 4-row matvec: shares input vector load across 4 output rows.
 * Uses 8-wide unrolling (32 floats per iteration) to maximize NEON
 * throughput and L1 cache line utilization on Apple Silicon. */
static void matvec(float *out, const float *W, const float *x,
                   const float *bias, int rows, int cols) {
    int r = 0;
    for (; r + 3 < rows; r += 4) {
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        float32x4_t b0 = vdupq_n_f32(0.0f);
        float32x4_t b1 = vdupq_n_f32(0.0f);
        float32x4_t b2 = vdupq_n_f32(0.0f);
        float32x4_t b3 = vdupq_n_f32(0.0f);
        const float *w0 = W + r*cols, *w1 = W + (r+1)*cols;
        const float *w2 = W + (r+2)*cols, *w3 = W + (r+3)*cols;
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            float32x4_t v0 = vld1q_f32(x + c);
            float32x4_t v1 = vld1q_f32(x + c + 4);
            a0 = vfmaq_f32(a0, vld1q_f32(w0+c), v0);
            b0 = vfmaq_f32(b0, vld1q_f32(w0+c+4), v1);
            a1 = vfmaq_f32(a1, vld1q_f32(w1+c), v0);
            b1 = vfmaq_f32(b1, vld1q_f32(w1+c+4), v1);
            a2 = vfmaq_f32(a2, vld1q_f32(w2+c), v0);
            b2 = vfmaq_f32(b2, vld1q_f32(w2+c+4), v1);
            a3 = vfmaq_f32(a3, vld1q_f32(w3+c), v0);
            b3 = vfmaq_f32(b3, vld1q_f32(w3+c+4), v1);
        }
        a0 = vaddq_f32(a0, b0);
        a1 = vaddq_f32(a1, b1);
        a2 = vaddq_f32(a2, b2);
        a3 = vaddq_f32(a3, b3);
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

#elif USE_AVX2

static inline float dot_avx2(const float *a, const float *b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),   _mm256_loadu_ps(b+i),   acc);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8), _mm256_loadu_ps(b+i+8), acc);
    }
    for (; i + 7 < n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    /* horizontal sum of 8 floats */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static void matvec(float *out, const float *W, const float *x,
                   const float *bias, int rows, int cols) {
    int r = 0;
    for (; r + 3 < rows; r += 4) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        const float *w0 = W + r*cols, *w1 = W + (r+1)*cols;
        const float *w2 = W + (r+2)*cols, *w3 = W + (r+3)*cols;
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            __m256 v = _mm256_loadu_ps(x + c);
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w0+c), v, a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1+c), v, a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w2+c), v, a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w3+c), v, a3);
        }
        /* Reduce each __m256 to scalar */
        __m128 t;
        t = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        out[r] = (bias ? bias[r] : 0.0f) + _mm_cvtss_f32(t);

        t = _mm_add_ps(_mm256_castps256_ps128(a1), _mm256_extractf128_ps(a1, 1));
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        out[r+1] = (bias ? bias[r+1] : 0.0f) + _mm_cvtss_f32(t);

        t = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps(a2, 1));
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        out[r+2] = (bias ? bias[r+2] : 0.0f) + _mm_cvtss_f32(t);

        t = _mm_add_ps(_mm256_castps256_ps128(a3), _mm256_extractf128_ps(a3, 1));
        t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t);
        out[r+3] = (bias ? bias[r+3] : 0.0f) + _mm_cvtss_f32(t);

        for (; c < cols; c++) {
            float xc = x[c];
            out[r]   += w0[c] * xc;
            out[r+1] += w1[c] * xc;
            out[r+2] += w2[c] * xc;
            out[r+3] += w3[c] * xc;
        }
    }
    for (; r < rows; r++)
        out[r] = (bias ? bias[r] : 0.0f) + dot_avx2(W + r*cols, x, cols);
}

static void vec_scale_shift(float *x, const float *s, const float *sh, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x+i, _mm256_fmadd_ps(_mm256_loadu_ps(x+i),
                         _mm256_loadu_ps(s+i), _mm256_loadu_ps(sh+i)));
    for (; i < n; i++) x[i] = x[i]*s[i] + sh[i];
}

static void vec_add(float *dst, const float *a, const float *b, int n) {
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst+i, _mm256_add_ps(_mm256_loadu_ps(a+i),
                         _mm256_loadu_ps(b+i)));
    for (; i < n; i++) dst[i] = a[i] + b[i];
}

static void scatter_add(float *agg, const float *msg, int dim) {
    int i = 0;
    for (; i + 7 < dim; i += 8)
        _mm256_storeu_ps(agg+i, _mm256_add_ps(_mm256_loadu_ps(agg+i),
                         _mm256_loadu_ps(msg+i)));
    for (; i < dim; i++) agg[i] += msg[i];
}

#else /* pure scalar fallback */

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

static inline float dot_product(const float *a, const float *b, int n) {
#if USE_NEON
    return dot_neon(a, b, n);
#elif USE_AVX2
    return dot_avx2(a, b, n);
#else
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

/* ================================================================
 * Dynamic-activation INT8 primitives
 * ================================================================ */

static float quantize_i8_block(int8_t *dst, const float *src, int n) {
#if USE_NEON
    float32x4_t vmaxv = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4)
        vmaxv = vmaxq_f32(vmaxv, vabsq_f32(vld1q_f32(src + i)));
    float amax = vmaxvq_f32(vmaxv);
    for (; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float scale = (amax > 1e-20f) ? (amax / 127.0f) : 1.0f;
    float inv = 1.0f / scale;
    float32x4_t invv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 7 < n; i += 8) {
        int32x4_t q0 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(src + i), invv)));
        int32x4_t q1 = vcvtq_s32_f32(vrndnq_f32(vmulq_f32(vld1q_f32(src + i + 4), invv)));
        int16x8_t q16 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        vst1_s8(dst + i, vqmovn_s16(q16));
    }
    for (; i < n; i++) {
        int q = (int)lrintf(src[i] * inv);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        dst[i] = (int8_t)q;
    }
    return scale;
#else
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float scale = (amax > 1e-20f) ? (amax / 127.0f) : 1.0f;
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i++) {
        int q = (int)lrintf(src[i] * inv);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        dst[i] = (int8_t)q;
    }
    return scale;
#endif
}

static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
#if USE_NEON_DOT
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 15 < n; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
    int32_t s = vaddvq_s32(acc);
    for (; i < n; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i] * (int32_t)b[i];
    return s;
#endif
}

static inline void dot_i8_4rows(const int8_t *w0, const int8_t *w1,
                                const int8_t *w2, const int8_t *w3,
                                const int8_t *x, int n,
                                int32_t *s0, int32_t *s1,
                                int32_t *s2, int32_t *s3) {
#if USE_NEON_DOT
    int32x4_t a0 = vdupq_n_s32(0);
    int32x4_t a1 = vdupq_n_s32(0);
    int32x4_t a2 = vdupq_n_s32(0);
    int32x4_t a3 = vdupq_n_s32(0);
    int i = 0;
    for (; i + 15 < n; i += 16) {
        int8x16_t xv = vld1q_s8(x + i);
        a0 = vdotq_s32(a0, vld1q_s8(w0 + i), xv);
        a1 = vdotq_s32(a1, vld1q_s8(w1 + i), xv);
        a2 = vdotq_s32(a2, vld1q_s8(w2 + i), xv);
        a3 = vdotq_s32(a3, vld1q_s8(w3 + i), xv);
    }
    int32_t t0 = vaddvq_s32(a0);
    int32_t t1 = vaddvq_s32(a1);
    int32_t t2 = vaddvq_s32(a2);
    int32_t t3 = vaddvq_s32(a3);
    for (; i < n; i++) {
        int32_t xi = (int32_t)x[i];
        t0 += (int32_t)w0[i] * xi;
        t1 += (int32_t)w1[i] * xi;
        t2 += (int32_t)w2[i] * xi;
        t3 += (int32_t)w3[i] * xi;
    }
    *s0 = t0; *s1 = t1; *s2 = t2; *s3 = t3;
#else
    int32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (int i = 0; i < n; i++) {
        int32_t xi = (int32_t)x[i];
        t0 += (int32_t)w0[i] * xi;
        t1 += (int32_t)w1[i] * xi;
        t2 += (int32_t)w2[i] * xi;
        t3 += (int32_t)w3[i] * xi;
    }
    *s0 = t0; *s1 = t1; *s2 = t2; *s3 = t3;
#endif
}

static void matvec_i8(float *out, const int8_t *W, const float *w_scales,
                      const float *x, const float *bias, int rows, int cols) {
    int8_t qx[NN_TRUNK_INPUT];
    float x_scale = quantize_i8_block(qx, x, cols);
    int r = 0;
    for (; r + 3 < rows; r += 4) {
        int32_t s0, s1, s2, s3;
        dot_i8_4rows(W + r * cols, W + (r + 1) * cols,
                     W + (r + 2) * cols, W + (r + 3) * cols,
                     qx, cols, &s0, &s1, &s2, &s3);
        out[r] = (bias ? bias[r] : 0.0f) + (float)s0 * (x_scale * w_scales[r]);
        out[r + 1] = (bias ? bias[r + 1] : 0.0f) + (float)s1 * (x_scale * w_scales[r + 1]);
        out[r + 2] = (bias ? bias[r + 2] : 0.0f) + (float)s2 * (x_scale * w_scales[r + 2]);
        out[r + 3] = (bias ? bias[r + 3] : 0.0f) + (float)s3 * (x_scale * w_scales[r + 3]);
    }
    for (; r < rows; r++) {
        int32_t acc = dot_i8(W + r * cols, qx, cols);
        out[r] = (bias ? bias[r] : 0.0f) + (float)acc * (x_scale * w_scales[r]);
    }
}

/* ================================================================
 * Activations (NEON-vectorized where possible)
 * ================================================================ */

static void apply_mish(float *x, int n) {
    int i = 0;
#if USE_NEON
    for (; i + 3 < n; i += 4)
        vst1q_f32(x+i, neon_mish(vld1q_f32(x+i)));
#elif USE_AVX2
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x+i, avx2_mish(_mm256_loadu_ps(x+i)));
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
#elif USE_AVX2
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_fmadd_ps(_mm256_loadu_ps(x+i),
                                    _mm256_loadu_ps(scale+i),
                                    _mm256_loadu_ps(shift+i));
        _mm256_storeu_ps(x+i, avx2_mish(v));
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
#elif USE_AVX2
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(out+i, avx2_mish(_mm256_add_ps(
            _mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i))));
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
#elif USE_AVX2
    __m256 sum8 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) sum8 = _mm256_add_ps(sum8, _mm256_loadu_ps(x+i));
    __m128 lo = _mm256_castps256_ps128(sum8);
    __m128 hi = _mm256_extractf128_ps(sum8, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    float mean = _mm_cvtss_f32(lo);
    for (; i < n; i++) mean += x[i];
    mean /= n;

    __m256 vm = _mm256_set1_ps(mean);
    __m256 var8 = _mm256_setzero_ps();
    for (i = 0; i + 7 < n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(x+i), vm);
        var8 = _mm256_fmadd_ps(d, d, var8);
    }
    lo = _mm256_castps256_ps128(var8);
    hi = _mm256_extractf128_ps(var8, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    float var = _mm_cvtss_f32(lo);
    for (; i < n; i++) { float d = x[i] - mean; var += d*d; }
    var /= n;

    float inv = 1.0f / sqrtf(var + 1e-5f);
    __m256 vinv = _mm256_set1_ps(inv);
    for (i = 0; i + 7 < n; i += 8) {
        __m256 d = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x+i), vm), vinv);
        _mm256_storeu_ps(x+i, _mm256_fmadd_ps(d, _mm256_loadu_ps(w+i),
                         _mm256_loadu_ps(b+i)));
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

static int mask_any_range(const float *mask, int start, int end) {
    for (int i = start; i < end; i++)
        if (mask[i] > 0.5f) return 1;
    return 0;
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

#define NN_I8_MAX_BATCH_ELEMS (NN_MAX_EDGES * 3 * NN_GNN_HIDDEN)

static int batch_fc_i8(float *C, const float *A, const int8_t *Bq, const float *w_scales,
                       const float *bias, int M, int N, int K) {
    if (M * K > NN_I8_MAX_BATCH_ELEMS) return -1;
    int8_t qA[NN_I8_MAX_BATCH_ELEMS];
    float a_scales[NN_MAX_EDGES > NN_NODES ? NN_MAX_EDGES : NN_NODES];
    for (int m = 0; m < M; m++)
        a_scales[m] = quantize_i8_block(qA + m * K, A + m * K, K);

    for (int m = 0; m < M; m++) {
        int n = 0;
        for (; n + 3 < N; n += 4) {
            int32_t s0, s1, s2, s3;
            dot_i8_4rows(Bq + n * K, Bq + (n + 1) * K,
                         Bq + (n + 2) * K, Bq + (n + 3) * K,
                         qA + m * K, K, &s0, &s1, &s2, &s3);
            C[m * N + n] = (bias ? bias[n] : 0.0f) + (float)s0 * (a_scales[m] * w_scales[n]);
            C[m * N + n + 1] = (bias ? bias[n + 1] : 0.0f) + (float)s1 * (a_scales[m] * w_scales[n + 1]);
            C[m * N + n + 2] = (bias ? bias[n + 2] : 0.0f) + (float)s2 * (a_scales[m] * w_scales[n + 2]);
            C[m * N + n + 3] = (bias ? bias[n + 3] : 0.0f) + (float)s3 * (a_scales[m] * w_scales[n + 3]);
        }
        for (; n < N; n++) {
            int32_t acc = dot_i8(Bq + n * K, qA + m * K, K);
            C[m * N + n] = (bias ? bias[n] : 0.0f) + (float)acc * (a_scales[m] * w_scales[n]);
        }
    }
    return 0;
}

static void batch_fc_compute(const NNModel *m, float *C, const float *A,
                             const float *B, const int8_t *Bq, const float *w_scales,
                             const float *bias, int M, int N, int K) {
    if (m->compute_i8_batch && Bq && batch_fc_i8(C, A, Bq, w_scales, bias, M, N, K) == 0)
        return;
    batch_fc(C, A, B, bias, M, N, K);
}

static void matvec_compute(const NNModel *m, float *out, const float *W,
                           const int8_t *Wq, const float *w_scales, const float *x,
                           const float *bias, int rows, int cols) {
    if (m->compute_i8 && Wq) {
        matvec_i8(out, Wq, w_scales, x, bias, rows, cols);
        return;
    }
    matvec(out, W, x, bias, rows, cols);
}

/* ================================================================
 * GNN forward — batched with BLAS GEMM
 * ================================================================ */

/* Pre-computed scatter matrix: scatter_mat[N][E] where scatter_mat[dst][e]=1
 * if edge e points to node dst. Turns scatter_add loop into a BLAS SGEMM:
 *   agg[N,H] = scatter_mat[N,E] @ msg_out[E,H]
 * Built once in nn_load, reused every forward. */
static float _scatter_mat[NN_NODES][NN_MAX_EDGES];
static int   _scatter_mat_built = 0;

static void build_scatter_mat(const NNModel *m) {
    if (_scatter_mat_built) return;
    memset(_scatter_mat, 0, sizeof(_scatter_mat));
    for (int e = 0; e < m->num_edges; e++)
        _scatter_mat[m->edge_dst[e]][e] = 1.0f;
    _scatter_mat_built = 1;
}

static void gnn_forward(const NNModel *m,
                        const float nf[NN_NODES][NN_NODE_FEAT],
                        const float ef[NN_MAX_EDGES][NN_EDGE_FEAT],
                        float board_emb[NN_GNN_OUTPUT],
                        float node_emb[NN_NODES][NN_GNN_HIDDEN]) {
    const int N = NN_NODES;
    const int E = m->num_edges;
    const int H = NN_GNN_HIDDEN;

    build_scatter_mat(m);

    /* node_proj: (N, 18) -> (N, H) + mish */
    batch_fc_compute(m, &node_emb[0][0], &nf[0][0], (const float *)m->node_proj_w,
             (const int8_t *)m->node_proj_w_i8, m->node_proj_w_s,
             m->node_proj_b, N, H, NN_NODE_FEAT);
    apply_mish(&node_emb[0][0], N * H);

    /* edge_proj: (E, 5) -> (E, H) */
    float edge_emb[NN_MAX_EDGES][NN_GNN_HIDDEN];
    batch_fc_compute(m, &edge_emb[0][0], &ef[0][0], (const float *)m->edge_proj_w,
             (const int8_t *)m->edge_proj_w_i8, m->edge_proj_w_s,
             m->edge_proj_b, E, H, NN_EDGE_FEAT);

    float msg_tmp[NN_MAX_EDGES][NN_GNN_HIDDEN];
    float msg_out[NN_MAX_EDGES][NN_GNN_HIDDEN];
    float agg[NN_NODES][NN_GNN_HIDDEN];
    float upd_tmp[NN_NODES][NN_GNN_HIDDEN];
    float upd_out[NN_NODES][NN_GNN_HIDDEN];

    float msg_in[NN_MAX_EDGES][3 * NN_GNN_HIDDEN];

    for (int L = 0; L < NN_GNN_LAYERS; L++) {
        const EdgeConvWeights *lw = &m->gnn_layers[L];

        /* Gather src/dst node embeddings + edge features */
        for (int e = 0; e < E; e++) {
            memcpy(msg_in[e], node_emb[m->edge_src[e]], H * sizeof(float));
            memcpy(msg_in[e] + H, node_emb[m->edge_dst[e]], H * sizeof(float));
            memcpy(msg_in[e] + 2*H, edge_emb[e], H * sizeof(float));
        }

        /* msg_mlp layer 1: (E, 3H) -> (E, H) + mish */
        const EdgeConvWeightsI8 *lwi8 = &m->gnn_layers_i8[L];
        batch_fc_compute(m, &msg_tmp[0][0], &msg_in[0][0], (const float *)lw->msg_w1,
                 (const int8_t *)lwi8->msg_w1, lwi8->msg_w1_s,
                 lw->msg_b1, E, H, 3*H);
        apply_mish(&msg_tmp[0][0], E * H);

        /* msg_mlp layer 2: (E, H) -> (E, H) + mish */
        batch_fc_compute(m, &msg_out[0][0], &msg_tmp[0][0], (const float *)lw->msg_w2,
                 (const int8_t *)lwi8->msg_w2, lwi8->msg_w2_s,
                 lw->msg_b2, E, H, H);
        apply_mish(&msg_out[0][0], E * H);

        /* Scatter-add messages to destination nodes */
        memset(agg, 0, sizeof(agg));
        for (int e = 0; e < E; e++)
            scatter_add(agg[m->edge_dst[e]], msg_out[e], H);

        {
            float upd_in[NN_NODES][2 * NN_GNN_HIDDEN];
            for (int n = 0; n < N; n++) {
                memcpy(upd_in[n], node_emb[n], H * sizeof(float));
                memcpy(upd_in[n] + H, agg[n], H * sizeof(float));
            }
            batch_fc_compute(m, &upd_tmp[0][0], &upd_in[0][0], (const float *)lw->upd_w1,
                     (const int8_t *)lwi8->upd_w1, lwi8->upd_w1_s,
                     lw->upd_b1, N, H, 2*H);
            apply_mish(&upd_tmp[0][0], N * H);
        }

        /* update_mlp layer 2: (N, H) -> (N, H) */
        batch_fc_compute(m, &upd_out[0][0], &upd_tmp[0][0], (const float *)lw->upd_w2,
                 (const int8_t *)lwi8->upd_w2, lwi8->upd_w2_s,
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
#elif USE_AVX2
        for (int i = 0; i + 7 < H; i += 8) {
            __m256 v = _mm256_loadu_ps(&node_emb[n][i]);
            _mm256_storeu_ps(&mean_pool[i], _mm256_add_ps(
                _mm256_loadu_ps(&mean_pool[i]), v));
            _mm256_storeu_ps(&max_pool[i], _mm256_max_ps(
                _mm256_loadu_ps(&max_pool[i]), v));
        }
        for (int i = (H / 8) * 8; i < H; i++) {
            mean_pool[i] += node_emb[n][i];
            if (node_emb[n][i] > max_pool[i]) max_pool[i] = node_emb[n][i];
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
    matvec_compute(m, tmp, (const float *)m->out_proj_w1,
           (const int8_t *)m->out_proj_w1_i8, m->out_proj_w1_s,
           cat, m->out_proj_b1, NN_GNN_OUTPUT, 2*H);
    apply_mish(tmp, NN_GNN_OUTPUT);
    matvec_compute(m, board_emb, (const float *)m->out_proj_w2,
           (const int8_t *)m->out_proj_w2_i8, m->out_proj_w2_s,
           tmp, m->out_proj_b2, NN_GNN_OUTPUT, NN_GNN_OUTPUT);
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

    matvec_compute(m, trunk_out, (const float *)m->trunk_ip_w,
           (const int8_t *)m->trunk_ip_w_i8, m->trunk_ip_w_s,
           combined, m->trunk_ip_b, NN_TRUNK_CH, NN_TRUNK_INPUT);
    apply_bn_mish(trunk_out, m->trunk_ip_bn.scale, m->trunk_ip_bn.shift, NN_TRUNK_CH);

    float h[NN_TRUNK_CH], h2[NN_TRUNK_CH];
    for (int b = 0; b < NN_TRUNK_BLOCKS; b++) {
        const ResBlockWeights *rb = &m->trunk_blocks[b];
        const ResBlockWeightsI8 *rbi8 = &m->trunk_blocks_i8[b];
        matvec_compute(m, h, (const float *)rb->fc1_w,
               (const int8_t *)rbi8->fc1_w, rbi8->fc1_s,
               trunk_out, rb->fc1_b, NN_TRUNK_CH, NN_TRUNK_CH);
        apply_bn_mish(h, rb->bn1.scale, rb->bn1.shift, NN_TRUNK_CH);
        matvec_compute(m, h2, (const float *)rb->fc2_w,
               (const int8_t *)rbi8->fc2_w, rbi8->fc2_s,
               h, rb->fc2_b, NN_TRUNK_CH, NN_TRUNK_CH);
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
    matvec_compute(m, h, (const float *)m->val_fc1_w,
           (const int8_t *)m->val_fc1_w_i8, m->val_fc1_w_s,
           trunk, m->val_fc1_b, NN_VALUE_HIDDEN, NN_TRUNK_CH);
    apply_bn_mish(h, m->val_bn1.scale, m->val_bn1.shift, NN_VALUE_HIDDEN);

    float t1[NN_VALUE_HIDDEN], t2[NN_VALUE_HIDDEN];
    for (int r = 0; r < 2; r++) {
        const ValResBlockWeights *rb = &m->val_res[r];
        const ValResBlockWeightsI8 *rbi8 = &m->val_res_i8[r];
        matvec_compute(m, t1, (const float *)rb->fc1_w,
               (const int8_t *)rbi8->fc1_w, rbi8->fc1_s,
               h, rb->fc1_b, NN_VALUE_HIDDEN, NN_VALUE_HIDDEN);
        apply_bn_mish(t1, rb->bn1.scale, rb->bn1.shift, NN_VALUE_HIDDEN);
        matvec_compute(m, t2, (const float *)rb->fc2_w,
               (const int8_t *)rbi8->fc2_w, rbi8->fc2_s,
               t1, rb->fc2_b, NN_VALUE_HIDDEN, NN_VALUE_HIDDEN);
        vec_scale_shift(t2, rb->bn2.scale, rb->bn2.shift, NN_VALUE_HIDDEN);
        add_mish(h, t2, h, NN_VALUE_HIDDEN);
    }

    matvec_compute(m, out, (const float *)m->val_out_w,
           (const int8_t *)m->val_out_w_i8, m->val_out_w_s,
           h, m->val_out_b, 4, NN_VALUE_HIDDEN);
}

/* ================================================================
 * Policy head — batched spatial scorers via GEMM
 * ================================================================ */

static void policy_head(const NNModel *m,
                        const float trunk[NN_TRUNK_CH],
                        const float node_emb[NN_NODES][NN_GNN_HIDDEN],
                        const float mask[NN_MASK_DIM],
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
    matvec_compute(m, type_h, (const float *)m->pol_type_w1,
           (const int8_t *)m->pol_type_w1_i8, m->pol_type_w1_s,
           tn, m->pol_type_b1, NN_POLICY_HIDDEN, TC);
    apply_bn_mish(type_h, m->pol_type_bn.scale, m->pol_type_bn.shift,
                  NN_POLICY_HIDDEN);
    matvec_compute(m, type_logits, (const float *)m->pol_type_w2,
           (const int8_t *)m->pol_type_w2_i8, m->pol_type_w2_s,
           type_h, m->pol_type_b2, NN_NUM_TYPES, NN_POLICY_HIDDEN);

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
    if (mask_any_range(mask, 5, 59)) {
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
    if (mask_any_range(mask, 59, 113)) {
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
    if (mask_any_range(mask, 113, 185)) {
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
    if (mask_any_range(mask, 185, 280)) {
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
    if (mask_any_range(mask, 280, 310)) {
        float sh[NN_SCORER_HIDDEN], raw[30];
        matvec_compute(m, sh, (const float *)m->pol_dym_w1,
                       (const int8_t *)m->pol_dym_w1_i8, m->pol_dym_w1_s,
                       tn, m->pol_dym_b1, SH, TC);
        apply_mish(sh, SH);
        matvec_compute(m, raw, (const float *)m->pol_dym_w2,
                       (const int8_t *)m->pol_dym_w2_i8, m->pol_dym_w2_s,
                       sh, m->pol_dym_b2, 30, SH);
        log_softmax(raw, 30);
        for (int i = 0; i < 30; i++) policy[280+i] = log_type[9] + raw[i];
    }
    if (mask_any_range(mask, 310, 330)) {
        float sh[NN_SCORER_HIDDEN], raw[20];
        matvec_compute(m, sh, (const float *)m->pol_mar_w1,
                       (const int8_t *)m->pol_mar_w1_i8, m->pol_mar_w1_s,
                       tn, m->pol_mar_b1, SH, TC);
        apply_mish(sh, SH);
        matvec_compute(m, raw, (const float *)m->pol_mar_w2,
                       (const int8_t *)m->pol_mar_w2_i8, m->pol_mar_w2_s,
                       sh, m->pol_mar_b2, 20, SH);
        log_softmax(raw, 20);
        for (int i = 0; i < 20; i++) policy[310+i] = log_type[10] + raw[i];
    }
    if (mask_any_range(mask, 330, 397)) {
        float sh[NN_SCORER_HIDDEN], raw[67];
        matvec_compute(m, sh, (const float *)m->pol_trd_w1,
                       (const int8_t *)m->pol_trd_w1_i8, m->pol_trd_w1_s,
                       tn, m->pol_trd_b1, SH, TC);
        apply_mish(sh, SH);
        matvec_compute(m, raw, (const float *)m->pol_trd_w2,
                       (const int8_t *)m->pol_trd_w2_i8, m->pol_trd_w2_s,
                       sh, m->pol_trd_b2, 67, SH);
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

/* IEEE-754 half to float (weights file version 2). */
static float f16_bits_to_f32(uint16_t h) {
    int s = (h >> 15) & 1;
    int e = (h >> 10) & 0x1f;
    int m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) return s ? -0.0f : 0.0f;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e--;
        }
        e++;
        m &= 0x3ff;
    } else if (e == 31) {
        return m ? nanf("0x400000") : (s ? (float)-INFINITY : (float)INFINITY);
    }
    e = e - 15 + 127;
    {
        uint32_t u = ((uint32_t)s << 31) | ((uint32_t)e << 23) | ((uint32_t)m << 13);
        float o;
        memcpy(&o, &u, 4);
        return o;
    }
}

/* Version 1: fp32 blobs. Version 2: fp16 per scalar. Version 3: int8 + per-block f32 scale. */
static int read_tensor_block(FILE *f, void *dst, size_t nbytes, unsigned ver) {
    if (ver == 1) return read_exact(f, dst, nbytes);
    if (ver == 2) {
        size_t nf = nbytes / sizeof(float);
        if (nbytes != nf * (size_t)sizeof(float)) return -1;
        for (size_t i = 0; i < nf; i++) {
            uint16_t h;
            if (fread(&h, 2, 1, f) != 1) return -1;
            ((float *)dst)[i] = f16_bits_to_f32(h);
        }
        return 0;
    }
    if (ver == 3) {
        size_t nf = nbytes / sizeof(float);
        if (nbytes != nf * (size_t)sizeof(float)) return -1;
        int8_t *tmp = (int8_t *)malloc(nf);
        if (!tmp) return -1;
        if (read_exact(f, tmp, nf)) { free(tmp); return -1; }
        float sc;
        if (read_exact(f, &sc, sizeof(sc))) { free(tmp); return -1; }
        float *d = (float *)dst;
        for (size_t i = 0; i < nf; i++) d[i] = (float)tmp[i] * sc;
        free(tmp);
        return 0;
    }
    return -1;
}

static void pack_i8_rows(int8_t *dst, float *scales, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        scales[r] = quantize_i8_block(dst + r * cols, src + r * cols, cols);
}

static void pack_model_i8(NNModel *m) {
    pack_i8_rows((int8_t *)m->node_proj_w_i8, m->node_proj_w_s,
        (const float *)m->node_proj_w, NN_GNN_HIDDEN, NN_NODE_FEAT);
    pack_i8_rows((int8_t *)m->edge_proj_w_i8, m->edge_proj_w_s,
        (const float *)m->edge_proj_w, NN_GNN_HIDDEN, NN_EDGE_FEAT);
    for (int i = 0; i < NN_GNN_LAYERS; i++) {
        const EdgeConvWeights *src = &m->gnn_layers[i];
        EdgeConvWeightsI8 *dst = &m->gnn_layers_i8[i];
        pack_i8_rows((int8_t *)dst->msg_w1, dst->msg_w1_s,
            (const float *)src->msg_w1, NN_GNN_HIDDEN, 3 * NN_GNN_HIDDEN);
        pack_i8_rows((int8_t *)dst->msg_w2, dst->msg_w2_s,
            (const float *)src->msg_w2, NN_GNN_HIDDEN, NN_GNN_HIDDEN);
        pack_i8_rows((int8_t *)dst->upd_w1, dst->upd_w1_s,
            (const float *)src->upd_w1, NN_GNN_HIDDEN, 2 * NN_GNN_HIDDEN);
        pack_i8_rows((int8_t *)dst->upd_w2, dst->upd_w2_s,
            (const float *)src->upd_w2, NN_GNN_HIDDEN, NN_GNN_HIDDEN);
    }
    pack_i8_rows((int8_t *)m->out_proj_w1_i8, m->out_proj_w1_s,
        (const float *)m->out_proj_w1, NN_GNN_OUTPUT, 2 * NN_GNN_HIDDEN);
    pack_i8_rows((int8_t *)m->out_proj_w2_i8, m->out_proj_w2_s,
        (const float *)m->out_proj_w2, NN_GNN_OUTPUT, NN_GNN_OUTPUT);

    pack_i8_rows((int8_t *)m->trunk_ip_w_i8, m->trunk_ip_w_s,
        (const float *)m->trunk_ip_w, NN_TRUNK_CH, NN_TRUNK_INPUT);
    for (int i = 0; i < NN_TRUNK_BLOCKS; i++) {
        const ResBlockWeights *src = &m->trunk_blocks[i];
        ResBlockWeightsI8 *dst = &m->trunk_blocks_i8[i];
        pack_i8_rows((int8_t *)dst->fc1_w, dst->fc1_s,
            (const float *)src->fc1_w, NN_TRUNK_CH, NN_TRUNK_CH);
        pack_i8_rows((int8_t *)dst->fc2_w, dst->fc2_s,
            (const float *)src->fc2_w, NN_TRUNK_CH, NN_TRUNK_CH);
    }

    pack_i8_rows((int8_t *)m->val_fc1_w_i8, m->val_fc1_w_s,
        (const float *)m->val_fc1_w, NN_VALUE_HIDDEN, NN_TRUNK_CH);
    for (int i = 0; i < 2; i++) {
        const ValResBlockWeights *src = &m->val_res[i];
        ValResBlockWeightsI8 *dst = &m->val_res_i8[i];
        pack_i8_rows((int8_t *)dst->fc1_w, dst->fc1_s,
            (const float *)src->fc1_w, NN_VALUE_HIDDEN, NN_VALUE_HIDDEN);
        pack_i8_rows((int8_t *)dst->fc2_w, dst->fc2_s,
            (const float *)src->fc2_w, NN_VALUE_HIDDEN, NN_VALUE_HIDDEN);
    }
    pack_i8_rows((int8_t *)m->val_out_w_i8, m->val_out_w_s,
        (const float *)m->val_out_w, 4, NN_VALUE_HIDDEN);

    pack_i8_rows((int8_t *)m->pol_type_w1_i8, m->pol_type_w1_s,
        (const float *)m->pol_type_w1, NN_POLICY_HIDDEN, NN_TRUNK_CH);
    pack_i8_rows((int8_t *)m->pol_type_w2_i8, m->pol_type_w2_s,
        (const float *)m->pol_type_w2, NN_NUM_TYPES, NN_POLICY_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_dym_w1_i8, m->pol_dym_w1_s,
        (const float *)m->pol_dym_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH);
    pack_i8_rows((int8_t *)m->pol_dym_w2_i8, m->pol_dym_w2_s,
        (const float *)m->pol_dym_w2, 30, NN_SCORER_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_mar_w1_i8, m->pol_mar_w1_s,
        (const float *)m->pol_mar_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH);
    pack_i8_rows((int8_t *)m->pol_mar_w2_i8, m->pol_mar_w2_s,
        (const float *)m->pol_mar_w2, 20, NN_SCORER_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_trd_w1_i8, m->pol_trd_w1_s,
        (const float *)m->pol_trd_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH);
    pack_i8_rows((int8_t *)m->pol_trd_w2_i8, m->pol_trd_w2_s,
        (const float *)m->pol_trd_w2, 67, NN_SCORER_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_sett_w1_i8, m->pol_sett_w1_s,
        (const float *)m->pol_sett_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH + NN_GNN_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_sett_w2_i8, m->pol_sett_w2_s,
        (const float *)m->pol_sett_w2, 1, NN_SCORER_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_city_w1_i8, m->pol_city_w1_s,
        (const float *)m->pol_city_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH + NN_GNN_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_city_w2_i8, m->pol_city_w2_s,
        (const float *)m->pol_city_w2, 1, NN_SCORER_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_road_w1_i8, m->pol_road_w1_s,
        (const float *)m->pol_road_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH + 2 * NN_GNN_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_road_w2_i8, m->pol_road_w2_s,
        (const float *)m->pol_road_w2, 1, NN_SCORER_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_rob_w1_i8, m->pol_rob_w1_s,
        (const float *)m->pol_rob_w1, NN_SCORER_HIDDEN, NN_TRUNK_CH + NN_GNN_HIDDEN);
    pack_i8_rows((int8_t *)m->pol_rob_w2_i8, m->pol_rob_w2_s,
        (const float *)m->pol_rob_w2, 5, NN_SCORER_HIDDEN);
}

int nn_load(NNModel *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "nn_load: cannot open %s\n", path); return -1; }

    char magic[4];
    uint32_t hdr[15];
    if (read_exact(f, magic, 4) || memcmp(magic, "HBOT", 4) != 0) goto fail;
    if (read_exact(f, hdr, sizeof(hdr))) goto fail;
    unsigned wver = (unsigned)hdr[0];
    if (wver != 1U && wver != 2U && wver != 3U) {
        fprintf(stderr, "nn_load: unsupported file version %u in %s\n", wver, path);
        goto fail;
    }
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

    /* GNN + trunk + value + policy weights (v1=fp32, v2=fp16) */
    if (read_tensor_block(f, m->node_proj_w, sizeof(m->node_proj_w), wver)) goto fail;
    if (read_tensor_block(f, m->node_proj_b, sizeof(m->node_proj_b), wver)) goto fail;
    if (read_tensor_block(f, m->edge_proj_w, sizeof(m->edge_proj_w), wver)) goto fail;
    if (read_tensor_block(f, m->edge_proj_b, sizeof(m->edge_proj_b), wver)) goto fail;
    for (int i = 0; i < NN_GNN_LAYERS; i++)
        if (read_tensor_block(f, &m->gnn_layers[i], sizeof(EdgeConvWeights), wver)) goto fail;
    if (read_tensor_block(f, m->out_proj_w1, sizeof(m->out_proj_w1), wver)) goto fail;
    if (read_tensor_block(f, m->out_proj_b1, sizeof(m->out_proj_b1), wver)) goto fail;
    if (read_tensor_block(f, m->out_proj_w2, sizeof(m->out_proj_w2), wver)) goto fail;
    if (read_tensor_block(f, m->out_proj_b2, sizeof(m->out_proj_b2), wver)) goto fail;

    /* Trunk */
    if (read_tensor_block(f, m->trunk_ip_w, sizeof(m->trunk_ip_w), wver)) goto fail;
    if (read_tensor_block(f, m->trunk_ip_b, sizeof(m->trunk_ip_b), wver)) goto fail;
    if (read_tensor_block(f, &m->trunk_ip_bn, sizeof(BN128), wver)) goto fail;
    for (int i = 0; i < NN_TRUNK_BLOCKS; i++)
        if (read_tensor_block(f, &m->trunk_blocks[i], sizeof(ResBlockWeights), wver)) goto fail;

    /* Value head */
    if (read_tensor_block(f, m->val_fc1_w, sizeof(m->val_fc1_w), wver)) goto fail;
    if (read_tensor_block(f, m->val_fc1_b, sizeof(m->val_fc1_b), wver)) goto fail;
    if (read_tensor_block(f, &m->val_bn1, sizeof(BNV), wver)) goto fail;
    for (int i = 0; i < 2; i++)
        if (read_tensor_block(f, &m->val_res[i], sizeof(ValResBlockWeights), wver)) goto fail;
    if (read_tensor_block(f, m->val_out_w, sizeof(m->val_out_w), wver)) goto fail;
    if (read_tensor_block(f, m->val_out_b, sizeof(m->val_out_b), wver)) goto fail;

    /* Policy head */
    if (read_tensor_block(f, m->pol_trunk_ln_w, sizeof(m->pol_trunk_ln_w), wver)) goto fail;
    if (read_tensor_block(f, m->pol_trunk_ln_b, sizeof(m->pol_trunk_ln_b), wver)) goto fail;
    if (read_tensor_block(f, m->pol_node_ln_w, sizeof(m->pol_node_ln_w), wver)) goto fail;
    if (read_tensor_block(f, m->pol_node_ln_b, sizeof(m->pol_node_ln_b), wver)) goto fail;
    if (read_tensor_block(f, m->pol_type_w1, sizeof(m->pol_type_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_type_b1, sizeof(m->pol_type_b1), wver)) goto fail;
    if (read_tensor_block(f, &m->pol_type_bn, sizeof(BNP), wver)) goto fail;
    if (read_tensor_block(f, m->pol_type_w2, sizeof(m->pol_type_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_type_b2, sizeof(m->pol_type_b2), wver)) goto fail;

    if (read_tensor_block(f, m->pol_dym_w1, sizeof(m->pol_dym_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_dym_b1, sizeof(m->pol_dym_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_dym_w2, sizeof(m->pol_dym_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_dym_b2, sizeof(m->pol_dym_b2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_mar_w1, sizeof(m->pol_mar_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_mar_b1, sizeof(m->pol_mar_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_mar_w2, sizeof(m->pol_mar_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_mar_b2, sizeof(m->pol_mar_b2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_trd_w1, sizeof(m->pol_trd_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_trd_b1, sizeof(m->pol_trd_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_trd_w2, sizeof(m->pol_trd_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_trd_b2, sizeof(m->pol_trd_b2), wver)) goto fail;

    if (read_tensor_block(f, m->pol_sett_w1, sizeof(m->pol_sett_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_sett_b1, sizeof(m->pol_sett_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_sett_w2, sizeof(m->pol_sett_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_sett_b2, sizeof(m->pol_sett_b2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_city_w1, sizeof(m->pol_city_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_city_b1, sizeof(m->pol_city_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_city_w2, sizeof(m->pol_city_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_city_b2, sizeof(m->pol_city_b2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_road_w1, sizeof(m->pol_road_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_road_b1, sizeof(m->pol_road_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_road_w2, sizeof(m->pol_road_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_road_b2, sizeof(m->pol_road_b2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_rob_w1, sizeof(m->pol_rob_w1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_rob_b1, sizeof(m->pol_rob_b1), wver)) goto fail;
    if (read_tensor_block(f, m->pol_rob_w2, sizeof(m->pol_rob_w2), wver)) goto fail;
    if (read_tensor_block(f, m->pol_rob_b2, sizeof(m->pol_rob_b2), wver)) goto fail;

    pack_model_i8(m);
    {
        const char *mode = getenv("CATAN_NN_COMPUTE");
        const char *batch_mode = getenv("CATAN_NN_INT8_BATCH");
        m->compute_i8 = 0;
        m->compute_i8_batch = 0;
        if (mode && strcmp(mode, "int8") == 0) m->compute_i8 = 1;
        if (mode && strcmp(mode, "fp32") == 0) m->compute_i8 = 0;
        if (batch_mode && strcmp(batch_mode, "1") == 0) m->compute_i8_batch = m->compute_i8;
#if !USE_NEON_DOT
        if (m->compute_i8) {
            fprintf(stderr, "nn_load: CATAN_NN_COMPUTE=int8 requested, but NEON dot-product is unavailable; using scalar int8 fallback\n");
        }
#endif
    }

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
    policy_head(m, trunk_out, node_emb, mask, out->policy);

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

void nn_policy_only(const NNModel *m,
                    const float node_feat[NN_NODES][NN_NODE_FEAT],
                    const float edge_feat[NN_MAX_EDGES][NN_EDGE_FEAT],
                    const float flat_feat[NN_FLAT_DIM],
                    const float mask[NN_MASK_DIM],
                    float policy_out[NN_MASK_DIM]) {
    float board_emb[NN_GNN_OUTPUT];
    float node_emb[NN_NODES][NN_GNN_HIDDEN];
    float trunk_out[NN_TRUNK_CH];

    gnn_forward(m, node_feat, edge_feat, board_emb, node_emb);
    trunk_forward(m, board_emb, flat_feat, mask, trunk_out);
    policy_head(m, trunk_out, node_emb, mask, policy_out);

    for (int i = 0; i < NN_MASK_DIM; i++)
        if (mask[i] < 0.5f) policy_out[i] = -1e9f;
}
