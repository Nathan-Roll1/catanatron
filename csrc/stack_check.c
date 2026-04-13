#include <stdio.h>
#include <pthread.h>
#include <sys/resource.h>

static void *check_stack(void *arg) {
    struct rlimit rl;
    getrlimit(RLIMIT_STACK, &rl);
    printf("Thread stack limit: %llu bytes (%llu KB)\n",
           (unsigned long long)rl.rlim_cur, (unsigned long long)rl.rlim_cur/1024);
    /* Check actual remaining stack */
    char dummy;
    printf("Stack var addr: %p\n", &dummy);
    return NULL;
}

int main(void) {
    struct rlimit rl;
    getrlimit(RLIMIT_STACK, &rl);
    printf("Main stack limit: %llu bytes (%llu KB)\n",
           (unsigned long long)rl.rlim_cur, (unsigned long long)rl.rlim_cur/1024);

    pthread_t t;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    size_t stacksize;
    pthread_attr_getstacksize(&attr, &stacksize);
    printf("Default pthread stack size: %zu bytes (%zu KB)\n", stacksize, stacksize/1024);

    pthread_create(&t, NULL, check_stack, NULL);
    pthread_join(t, NULL);
    return 0;
}
