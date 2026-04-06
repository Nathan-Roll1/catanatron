/*
 * MT19937 RNG: both instance-based (thread-safe) and global (legacy) APIs.
 */

#include "rng.h"

#define MT_M 397
#define MATRIX_A   0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

static void init_genrand(RngState *r, uint32_t s) {
    r->mt[0] = s & 0xffffffffUL;
    for (r->mti = 1; r->mti < MT_N; r->mti++) {
        r->mt[r->mti] = (1812433253UL * (r->mt[r->mti-1] ^ (r->mt[r->mti-1] >> 30)) + r->mti);
        r->mt[r->mti] &= 0xffffffffUL;
    }
}

static void init_by_array(RngState *r, const uint32_t init_key[], int key_length) {
    int i, j, k;
    init_genrand(r, 19650218UL);
    i = 1; j = 0;
    k = (MT_N > key_length ? MT_N : key_length);
    for (; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i-1] ^ (r->mt[i-1] >> 30)) * 1664525UL)) + init_key[j] + j;
        r->mt[i] &= 0xffffffffUL;
        i++; j++;
        if (i >= MT_N) { r->mt[0] = r->mt[MT_N-1]; i = 1; }
        if (j >= key_length) j = 0;
    }
    for (k = MT_N - 1; k; k--) {
        r->mt[i] = (r->mt[i] ^ ((r->mt[i-1] ^ (r->mt[i-1] >> 30)) * 1566083941UL)) - i;
        r->mt[i] &= 0xffffffffUL;
        i++;
        if (i >= MT_N) { r->mt[0] = r->mt[MT_N-1]; i = 1; }
    }
    r->mt[0] = 0x80000000UL;
}

void rng_init(RngState *r, uint64_t seed) {
    if (seed <= 0xffffffffULL) {
        uint32_t key[1] = {(uint32_t)seed};
        init_by_array(r, key, 1);
    } else {
        uint32_t key[2] = {(uint32_t)(seed & 0xffffffff), (uint32_t)(seed >> 32)};
        init_by_array(r, key, 2);
    }
}

uint32_t rng_genrand(RngState *r) {
    uint32_t y;
    static const uint32_t mag01[2] = {0x0UL, MATRIX_A};
    if (r->mti >= MT_N) {
        int kk;
        for (kk = 0; kk < MT_N - MT_M; kk++) {
            y = (r->mt[kk] & UPPER_MASK) | (r->mt[kk+1] & LOWER_MASK);
            r->mt[kk] = r->mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 1];
        }
        for (; kk < MT_N - 1; kk++) {
            y = (r->mt[kk] & UPPER_MASK) | (r->mt[kk+1] & LOWER_MASK);
            r->mt[kk] = r->mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 1];
        }
        y = (r->mt[MT_N-1] & UPPER_MASK) | (r->mt[0] & LOWER_MASK);
        r->mt[MT_N-1] = r->mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 1];
        r->mti = 0;
    }
    y = r->mt[r->mti++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);
    return y;
}

int rng_randbelow(RngState *r, int n) {
    if (n <= 0) return 0;
    int k = 0; unsigned tmp = (unsigned)n; while (tmp) { k++; tmp >>= 1; }
    int v; do { v = (int)(rng_genrand(r) >> (32 - k)); } while (v >= n);
    return v;
}

int rng_randint(RngState *r, int a, int b) { return a + rng_randbelow(r, b - a + 1); }
int rng_choice_index(RngState *r, int n) { return rng_randbelow(r, n); }

void rng_shuffle_int(RngState *r, int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rng_randbelow(r, i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

void rng_sample_int(RngState *r, const int *src, int src_n, int *dst, int k) {
    int pool[512];
    for (int i = 0; i < src_n && i < 512; i++) pool[i] = src[i];
    for (int i = 0; i < k; i++) {
        int j = rng_randbelow(r, src_n - i);
        dst[i] = pool[j];
        pool[j] = pool[src_n - 1 - i];
    }
}

/* ---- Global RNG (legacy, not thread-safe) ---- */
static RngState g_rng;

void     rng_seed(uint64_t seed) { rng_init(&g_rng, seed); }
uint32_t rng_genrand_uint32(void) { return rng_genrand(&g_rng); }
double   rng_random(void) {
    uint32_t a = rng_genrand(&g_rng) >> 5;
    uint32_t b = rng_genrand(&g_rng) >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}
int  rng_randbelow_g(int n) { return rng_randbelow(&g_rng, n); }
int  rng_randint_g(int a, int b) { return rng_randint(&g_rng, a, b); }
int  rng_choice_index_g(int n) { return rng_choice_index(&g_rng, n); }
void rng_shuffle_int_g(int *arr, int n) { rng_shuffle_int(&g_rng, arr, n); }
void rng_sample_int_g(const int *src, int n, int *dst, int k) { rng_sample_int(&g_rng, src, n, dst, k); }
void rng_save_state(void *buf) { memcpy(buf, &g_rng, sizeof(g_rng)); }
void rng_restore_state(const void *buf) { memcpy(&g_rng, buf, sizeof(g_rng)); }
