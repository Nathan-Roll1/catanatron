# catanatron_c

A high-performance C implementation of the [Catanatron](https://github.com/bcollazo/catanatron) Settlers of Catan simulator and AI. This is a ground-up rewrite of the Python engine in pure C, optimized for maximum simulation throughput.

## Performance

### Single-threaded

| Metric | Python (catanatron) | C (catanatron_c) | Speedup |
|--------|-------------------|-------------------|---------|
| Random vs Random | 119 games/sec | **20,480 games/sec** | **172x** |
| AB:2 vs AB:2 | 1.15 games/sec | **781 games/sec** | **679x** |
| Binary size | ~50 MB (Python + deps) | **66 KB** | 775x smaller |
| Dependencies | networkx, click, rich | **none** | - |
| Lines of code | 5,729 (Python) | **2,813 (C)** | 2x smaller |

### Multi-threaded (Apple M5 Max, 18 cores)

| Metric | Single thread | 18 threads | Scaling |
|--------|-------------|------------|---------|
| R vs R (2p) | 20,480/sec | ~275,000/sec | 13x |
| AB:2 vs AB:2 (2p) | 781/sec | ~9,900/sec | 13x |
| AB:2 4-player | 282/sec | **3,654/sec** | 13x |

See [PARALLEL.md](PARALLEL.md) for multi-threaded deployment instructions.

AB:2 is the AlphaBeta search player at depth 2, the strongest bot configuration. It uses a hand-crafted 11-parameter linear evaluation function with alpha-beta pruning, move ordering, and free-ROLL depth semantics.

## Build

Requires a C compiler (gcc, clang) with C99 support. No external dependencies.

```bash
make
```

## Usage

```bash
# 1000 games, AlphaBeta depth 2 vs itself
./catanatron --players=AB:2,AB:2 --num=1000 --quiet

# Random vs Random, 10k games
./catanatron --players=R,R --num=10000 --quiet

# AlphaBeta vs Random with output
./catanatron --players=AB:2,R --num=20

# 4-player game
./catanatron --players=AB:2,AB:2,AB:2,AB:2 --num=100 --quiet
```

### Player Types

| Code | Player | Description |
|------|--------|-------------|
| `R` | Random | Chooses actions uniformly at random |
| `AB:N` | AlphaBeta | Alpha-beta search at depth N (recommended: 2) |

### Options

| Flag | Description |
|------|-------------|
| `--players=X,Y` | Comma-separated player codes (2-4 players) |
| `--num=N` | Number of games to play |
| `--quiet` | Suppress per-game output, show only summary |

## Architecture

```
src/
├── catan_types.h      # Enums, constants, Action struct
├── rng.c/h            # MT19937 RNG (matches CPython's random module)
├── map.c/h            # Map generation, tiles, ports, coordinates
├── board.c/h          # Board state, buildings, roads, connected components
├── state.c/h          # Game state, player state arrays, decks
├── actions.c/h        # Legal move generation
├── apply_action.c/h   # Action application (game rules engine)
├── game.c/h           # Game loop
├── value.c/h          # Heuristic evaluation function
├── search.c/h         # Alpha-beta search with move ordering
└── main.c             # CLI entry point
```

### Key Design Decisions

- **Flat arrays** instead of Python dicts: `player_state[4][29]` replaces `{"P0_WOOD_IN_HAND": 3, ...}`
- **Bitsets** for node sets: 128-bit bitsets replace Python `set()` for connected components
- **`memcpy` state cloning**: `State` is a flat 2.9 KB struct copied in one operation
- **Inline evaluation**: value function reads arrays directly, no string hashing
- **MT19937 RNG**: bit-identical to CPython's `random` module for reproducibility

## Credits

This is a C port of [Catanatron](https://github.com/bcollazo/catanatron) by Bryan Collazo. The original Python implementation provides the game rules, AI algorithms, and evaluation function weights that this C version faithfully reproduces. All credit for the game engine design, heuristic weights, and AlphaBeta search architecture goes to the original project.
