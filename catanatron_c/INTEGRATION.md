# Integrating catanatron_c Into Your Catan Project

## What This Is

[catanatron_c](https://github.com/Nathan-Roll1/catanatron/tree/main/catanatron_c) is a high-performance Settlers of Catan game engine and AI written in pure C. It can simulate **20,000+ random games/sec** or **780+ AlphaBeta AI games/sec** single-threaded, with **10,000+ games/sec** across multiple cores. Zero dependencies, 66KB binary, 2,800 lines of C.

It's a C port of [Catanatron](https://github.com/bcollazo/catanatron) (Python).

**Repo:** https://github.com/Nathan-Roll1/catanatron  
**Path:** `catanatron_c/`

## Quick Start

```bash
git clone https://github.com/Nathan-Roll1/catanatron.git
cd catanatron/catanatron_c
make
./catanatron --players=AB:2,AB:2 --num=100 --quiet
```

## How to Use It In Your Project

### 1. As a CLI tool (simplest)

Run games and parse the output:

```bash
./catanatron --players=AB:2,R,R,R --num=1000 --quiet
```

Output:
```
=== Results (1000 games) ===
  RED(AB): 612 wins (61.2%)
  BLUE(R): 131 wins (13.1%)
  ORANGE(R): 128 wins (12.8%)
  WHITE(R): 129 wins (12.9%)
  Avg turns: 71.4
  Total time: 2.891s (345.9 games/sec)
```

### 2. As a C library (embed in your code)

Link the source files into your project:

```c
#include "game.h"
#include "search.h"
#include "actions.h"
#include "rng.h"
#include "value.h"
```

#### Minimal game loop:

```c
// 1. Initialize
RngState rng;
rng_init(&rng, 42);  // seed for reproducibility
CatanMap map;
build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
board_init_static_graph(&map);  // call once globally

Color colors[4] = {COLOR_RED, COLOR_BLUE, COLOR_ORANGE, COLOR_WHITE};
Game game;
game_init_with_map(&game, &map, 4, colors, 42, 7, false, 10);

// 2. Get legal actions
Action actions[MAX_ACTIONS];
int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);

// 3. Pick an action (AI or your own logic)
SearchCtx *ctx = malloc(sizeof(SearchCtx));
ctx->depth_counter = 0;
Game copy;
game_copy(&copy, &game);
SearchResult sr = alphabeta_search(ctx, &copy, actions, n,
    2,          // depth (2 = strongest stable setting)
    -1e30, 1e30,
    state_current_color(&game.state),
    base_value_fn);
Action best_action = sr.action;

// 4. Execute the action
game_execute(&game, best_action, actions, &n);
// `actions` and `n` are now updated with the new legal moves

// 5. Check for winner
Color winner = game_winning_color(&game);  // COLOR_NONE if game continues
```

#### Play a full game automatically:

```c
Color winner = game_play(&game, random_player_decide);
// Or write your own DecideFn:
// Action my_decide(State *s, Action *actions, int n) { ... }
```

## Key Types

### Action

```c
typedef struct {
    Color    color;      // who is acting (COLOR_RED, etc.)
    ActionType type;     // what they're doing (AT_BUILD_SETTLEMENT, etc.)
    int32_t  value[5];   // parameters (node_id, edge endpoints, resource, etc.)
} Action;
```

Action types and what `value[]` means:

| ActionType | value[0] | value[1] | value[2..4] |
|------------|----------|----------|-------------|
| `AT_ROLL` | - | - | - |
| `AT_END_TURN` | - | - | - |
| `AT_BUILD_SETTLEMENT` | node_id | - | - |
| `AT_BUILD_CITY` | node_id | - | - |
| `AT_BUILD_ROAD` | node_a | node_b | - |
| `AT_BUY_DEVELOPMENT_CARD` | - | - | - |
| `AT_MOVE_ROBBER` | coord.x | coord.y | coord.z, victim_color |
| `AT_DISCARD_RESOURCE` | resource | - | - |
| `AT_MARITIME_TRADE` | give[0] | give[1] | give[2], give[3], receive |
| `AT_PLAY_KNIGHT_CARD` | - | - | - |
| `AT_PLAY_YEAR_OF_PLENTY` | res1 | res2 (-1 if one) | - |
| `AT_PLAY_MONOPOLY` | resource | - | - |
| `AT_PLAY_ROAD_BUILDING` | - | - | - |

### Game State

```c
Game game;
game.state.num_turns;                    // current turn number
game.state.current_prompt;               // what kind of action is expected
game.state.player_state[idx][field];     // player stats (see PS_* constants)
game.state.board.buildings[node_id];     // who built what where
game.state.board.robber_coordinate;      // robber position
game.state.resource_freqdeck[5];         // bank resources
state_current_color(&game.state);        // whose turn it is
game_winning_color(&game);               // winner or COLOR_NONE
```

### Player state fields (indexed by `PS_*` constants):

```c
PS_VICTORY_POINTS          PS_ROADS_AVAILABLE
PS_SETTLEMENTS_AVAILABLE   PS_CITIES_AVAILABLE
PS_WOOD_IN_HAND            PS_BRICK_IN_HAND
PS_SHEEP_IN_HAND           PS_WHEAT_IN_HAND
PS_ORE_IN_HAND             PS_KNIGHT_IN_HAND
PS_PLAYED_KNIGHT           PS_LONGEST_ROAD_LENGTH
PS_HAS_ROAD                PS_HAS_ARMY
```

## Common Integration Patterns

### Use AB:2 as an opponent for your agent

```c
// Your agent plays RED, AB:2 plays the rest
while (game_winning_color(&game) == COLOR_NONE) {
    Action actions[MAX_ACTIONS];
    int n = generate_playable_actions(&game.state, actions, MAX_ACTIONS);
    Action action;
    
    if (state_current_color(&game.state) == COLOR_RED) {
        action = your_agent_decide(&game, actions, n);
    } else {
        // AB:2 opponent
        ctx->depth_counter = 0;
        Game cp; game_copy(&cp, &game);
        SearchResult sr = alphabeta_search(ctx, &cp, actions, n,
            2, -1e30, 1e30, state_current_color(&game.state), base_value_fn);
        action = sr.action;
    }
    game_execute(&game, action, actions, &n);
}
```

### Evaluate a position (get a score)

```c
double score = base_value_fn(&game, COLOR_RED);
// Higher = better for RED. Dominated by VP count (3e14 per VP).
```

### Run many games for statistics

```c
int wins = 0;
for (int i = 0; i < 10000; i++) {
    rng_init(&rng, i);
    build_map(&map, MAP_BASE, NPLACE_OFFICIAL_SPIRAL, &rng);
    game_init_with_map(&game, &map, 2, colors, i, 7, false, 10);
    Color w = game_play(&game, random_player_decide);
    if (w == COLOR_RED) wins++;
}
printf("RED win rate: %.1f%%\n", 100.0 * wins / 10000);
```

## Build Integration

Add these files to your build:

```
src/rng.c src/map.c src/board.c src/state.c src/actions.c
src/apply_action.c src/game.c src/value.c src/search.c
```

All headers are in `src/`. Compile with:

```bash
cc -O3 -march=native -flto -Isrc -c src/*.c
cc -O3 -flto -o your_program your_main.c src/*.o -lm
```

No external dependencies. C99 standard. Works on Linux, macOS, and Windows (with MinGW).
