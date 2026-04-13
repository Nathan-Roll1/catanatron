/* Verify C NN forward pass matches Python output. */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv) {
    const char *weights = argc > 1 ? argv[1] : "csrc/nn_weights.bin";
    const char *test    = argc > 2 ? argv[2] : "csrc/nn_weights_test.bin";

    NNModel *m = calloc(1, sizeof(NNModel));
    if (!m) { fprintf(stderr, "OOM\n"); return 1; }

    printf("Loading weights from %s ...\n", weights);
    if (nn_load(m, weights) != 0) return 1;
    printf("  Loaded. num_edges=%d\n", m->num_edges);

    /* Load test vectors */
    FILE *f = fopen(test, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", test); return 1; }

    float nf[NN_NODES][NN_NODE_FEAT];
    float ef[NN_MAX_EDGES][NN_EDGE_FEAT];
    float ff[NN_FLAT_DIM];
    float mask[NN_MASK_DIM];
    float expected_value[4];
    float expected_policy[NN_MASK_DIM];

    fread(nf, sizeof(nf), 1, f);
    fread(ef, sizeof(ef), 1, f);
    fread(ff, sizeof(ff), 1, f);
    fread(mask, sizeof(mask), 1, f);
    fread(expected_value, sizeof(expected_value), 1, f);
    fread(expected_policy, sizeof(expected_policy), 1, f);
    fclose(f);

    printf("\nExpected value: [%.6f, %.6f, %.6f, %.6f]\n",
           expected_value[0], expected_value[1], expected_value[2], expected_value[3]);

    /* Run forward pass */
    NNOutput out;
    nn_forward(m, nf, ef, ff, mask, &out);

    printf("C value:        [%.6f, %.6f, %.6f, %.6f]\n",
           out.value[0], out.value[1], out.value[2], out.value[3]);

    /* Check value accuracy */
    float max_err = 0;
    for (int i = 0; i < 4; i++) {
        float err = fabsf(out.value[i] - expected_value[i]);
        if (err > max_err) max_err = err;
    }
    printf("Value max error: %.6f  %s\n", max_err, max_err < 0.01f ? "PASS" : "FAIL");

    /* Check policy accuracy (only for legal actions) */
    float pol_max_err = 0;
    int legal_count = 0;
    for (int i = 0; i < NN_MASK_DIM; i++) {
        if (mask[i] > 0.5f) {
            float err = fabsf(out.policy[i] - expected_policy[i]);
            if (err > pol_max_err) pol_max_err = err;
            legal_count++;
        }
    }
    printf("Policy max error (legal): %.6f over %d actions  %s\n",
           pol_max_err, legal_count, pol_max_err < 0.05f ? "PASS" : "FAIL");

    /* Benchmark: time 1000 value-only forward passes */
    printf("\nBenchmark: 1000 value-only forward passes...\n");
    float val[4];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 1000; i++)
        nn_value_only(m, nf, ef, ff, mask, val);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("  1000 calls in %.3f s  (%.1f us/call, %.0f calls/sec)\n",
           elapsed, elapsed * 1000.0, 1000.0 / elapsed);

    /* Benchmark: 1000 full forward passes */
    printf("Benchmark: 1000 full forward passes...\n");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 1000; i++)
        nn_forward(m, nf, ef, ff, mask, &out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("  1000 calls in %.3f s  (%.1f us/call, %.0f calls/sec)\n",
           elapsed, elapsed * 1000.0, 1000.0 / elapsed);

    free(m);
    return 0;
}
