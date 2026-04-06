# Multi-threaded Deployment

This guide shows how to run catanatron_c at maximum throughput using all available CPU cores. On an Apple M5 Max (6 performance + 12 efficiency cores), this achieves **3,500+ AB:2 4-player games/sec** and **9,900+ AB:2 2-player games/sec**.

## Requirements

- POSIX threads (`-lpthread`)
- Any modern C compiler with `-O3 -march=native -flto`
- Multi-core CPU (scales ~linearly up to core count)

## How It Works

The game engine is fully thread-safe when each thread owns its own:

1. **`RngState`** -- per-game RNG (not the global one)
2. **`SearchCtx`** -- heap-allocated search pool (via `malloc`)
3. **`CatanMap`** -- stack-local map per game
4. **`Game`** -- independent game instance

The only shared state is `STATIC_ADJ[]` (the board graph adjacency), which is read-only after initialization. Call `board_init_static_graph()` once before spawning threads.

## Example: Parallel AB:2 Tournament

```c
#include <pthread.h>
#include "game.h"
#include "search.h"
#include "value.h"

#define NUM_THREADS 18  // match your core count

typedef struct {
    int thread_id, num_games;
    int wins[4];
} WorkerArgs;

void *worker(void *arg) {
    WorkerArgs *wa = (WorkerArgs *)arg;
    memset(wa->wins, 0, sizeof(wa->wins));

    // Each thread gets its own RNG, search context, maps
    RngState rng;
    SearchCtx *ctx = (SearchCtx *)malloc(sizeof(SearchCtx));
    Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};

    for (int gi = 0; gi < wa->num_games; gi++) {
        uint64_t seed = (uint64_t)wa->thread_id * 1000000 + gi;

        rng_init(&rng, seed);
        CatanMap map;
        build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);

        Game game;
        game_init_with_map(&game, &map, 4, colors, seed, 7, false, 10);

        Action actions[MAX_ACTIONS];
        int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

        while (game_winning_color(&game) == COLOR_NONE &&
               game.state.num_turns < TURNS_LIMIT) {
            Action action;
            if (n == 1) {
                action = actions[0];
            } else {
                Color cur = state_current_color(&game.state);
                ctx->depth_counter = 0;
                Game cp;
                game_copy(&cp, &game);
                SearchResult sr = alphabeta_search(
                    ctx, &cp, actions, n, 2,
                    -1e30, 1e30, cur, base_value_fn);
                action = sr.action;
            }
            game_execute(&game, action, actions, &n);
        }

        Color w = game_winning_color(&game);
        if (w != COLOR_NONE) {
            wa->wins[game.state.color_to_index[(int)w]]++;
        }
    }
    free(ctx);
    return NULL;
}

int main(void) {
    // Initialize static graph ONCE before any threads
    RngState tmp;
    rng_init(&tmp, 0);
    CatanMap tmp_map;
    build_map(&tmp_map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &tmp);
    board_init_static_graph(&tmp_map);

    // Launch threads
    pthread_t threads[NUM_THREADS];
    WorkerArgs args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].num_games = 625;  // total = 625 * 18 = 11,250
        pthread_create(&threads[i], NULL, worker, &args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    // Aggregate results
    int total[4] = {0};
    for (int i = 0; i < NUM_THREADS; i++)
        for (int c = 0; c < 4; c++)
            total[c] += args[i].wins[c];

    printf("Results: R=%d B=%d O=%d W=%d\n",
           total[0], total[1], total[2], total[3]);
    return 0;
}
```

Build with:

```bash
cc -O3 -march=native -flto -o parallel_example parallel_example.c \
   src/game.c src/apply_action.c src/actions.c src/state.c \
   src/board.c src/map.c src/rng.c src/value.c src/search.c \
   -lm -lpthread
```

## Thread Safety Checklist

| Component | Thread-safe? | Notes |
|-----------|-------------|-------|
| `STATIC_ADJ[]` | Yes (read-only) | Initialize once before threads |
| `RngState` | Yes | Use per-game instance, not global `rng_seed()` |
| `SearchCtx` | Yes | Heap-allocate per thread (`malloc`) |
| `CatanMap` | Yes | Stack-local per game |
| `Game` / `State` | Yes | Each thread owns its games |
| `base_value_fn()` | Yes | Pure function, no global state |
| Global RNG (`rng_seed()`) | **No** | Use `rng_init()` + `RngState` instead |

## Scaling Results (Apple M5 Max)

Measured with 18 threads (6P + 12E cores):

```
Config              1 thread    18 threads   Scaling
─────────────────   ─────────   ──────────   ───────
R vs R (2p)           20,480      ~275,000     13.4x
AB:2 vs AB:2 (2p)       781        ~9,900     12.7x
AB:2 4-player            282        ~3,654     13.0x
```

The ~13x scaling on 18 cores (vs theoretical 18x) is due to the efficiency cores running at ~60% of performance core speed.

## Tips

- Use **`-O3 -march=native -flto`** for maximum performance. LTO enables cross-file inlining of the value function.
- **`SearchCtx` is large** (~50-150 KB depending on `MAX_SEARCH_DEPTH`). Always heap-allocate it, never put it on the stack in a thread.
- For AB:2, `MAX_SEARCH_DEPTH = 8` is sufficient. Higher depths (AB:3+) may need `MAX_SEARCH_DEPTH = 16+`.
- Each thread needs ~200 KB of heap (SearchCtx + game states). 18 threads = ~3.6 MB total.
