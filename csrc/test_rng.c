#include <stdio.h>
#include "rng.h"

int main(void) {
    /* Test 1: getrandbits(32) -- raw MT output */
    printf("=== GETRANDBITS(32) x10 ===\n");
    rng_seed(42);
    for (int i = 0; i < 10; i++) {
        printf("%d: %u\n", i, rng_genrand_uint32());
    }

    /* Test 2: random() floats */
    printf("\n=== RANDOM FLOATS ===\n");
    rng_seed(42);
    for (int i = 0; i < 10; i++) {
        printf("%d: %.17g\n", i, rng_random());
    }

    /* Test 3: randrange(0, 1000) */
    printf("\n=== RANDRANGE(0, 1000) ===\n");
    rng_seed(42);
    for (int i = 0; i < 10; i++) {
        printf("%d: %d\n", i, rng_randrange(0, 1000));
    }

    /* Test 4: shuffle */
    printf("\n=== SHUFFLE [0,1,2,3] ===\n");
    rng_seed(42);
    int arr[] = {0, 1, 2, 3};
    rng_shuffle_int(arr, 4);
    printf("[%d, %d, %d, %d]\n", arr[0], arr[1], arr[2], arr[3]);

    /* Test 5: sample */
    printf("\n=== SAMPLE 3 from [0,1,2,3,4,5,6,7] ===\n");
    rng_seed(42);
    int src[] = {0,1,2,3,4,5,6,7};
    int dst[3];
    rng_sample_int(src, 8, dst, 3);
    printf("[%d, %d, %d]\n", dst[0], dst[1], dst[2]);

    /* Test 6: choice */
    printf("\n=== CHOICE from [10,20,30,40,50] ===\n");
    rng_seed(42);
    int choices[] = {10,20,30,40,50};
    for (int i = 0; i < 10; i++) {
        int idx = rng_choice_index(5);
        printf("%d: %d\n", i, choices[idx]);
    }

    /* Test 7: randint(0, 99) */
    printf("\n=== RANDOM INTS (0-99) ===\n");
    rng_seed(42);
    for (int i = 0; i < 20; i++) {
        printf("%d: %d\n", i, rng_randint(0, 99));
    }

    return 0;
}
