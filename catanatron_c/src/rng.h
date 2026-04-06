#ifndef RNG_H
#define RNG_H

#include <stdint.h>
#include <string.h>

#define MT_N 624

typedef struct {
    uint32_t mt[MT_N];
    int      mti;
} RngState;

void     rng_init(RngState *r, uint64_t seed);
uint32_t rng_genrand(RngState *r);
int      rng_randbelow(RngState *r, int n);
int      rng_randint(RngState *r, int a, int b);
int      rng_choice_index(RngState *r, int n);
void     rng_shuffle_int(RngState *r, int *arr, int n);
void     rng_sample_int(RngState *r, const int *src, int src_n, int *dst, int k);

/* Legacy global API (uses a process-wide RNG -- NOT thread safe) */
void     rng_seed(uint64_t seed);
uint32_t rng_genrand_uint32(void);
double   rng_random(void);
int      rng_randbelow_g(int n);
int      rng_randint_g(int a, int b);
int      rng_choice_index_g(int n);
void     rng_shuffle_int_g(int *arr, int n);
void     rng_sample_int_g(const int *src, int src_n, int *dst, int k);
void     rng_save_state(void *buf);
void     rng_restore_state(const void *buf);

#endif
