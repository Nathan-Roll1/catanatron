# Catan Player

Pure C neural network Catan player. No Python, no external dependencies
beyond a C compiler. Plays 4-player Catan using a learned policy for move
selection and a hand-crafted value function for position evaluation.

## Quick Start

```bash
./build.sh            # compile (~2s)
./catan_player        # play a game (ABt30, seed 42)
```

## Usage

```bash
./catan_player                         # ABt30, seed 42
./catan_player --seed 777              # different board
./catan_player --depth 0               # policy only (no search)
./catan_player --depth 20              # shallower search
./catan_player --games 100             # multi-game benchmark
./catan_player --verbose               # print every action
./catan_player --vs-ab2 --games 100    # 2v2: NN vs AB2 heuristic
./catan_player --1v3 --games 100       # 1v3: NN vs AB2 heuristic
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed for board generation |
| `--depth` | 30 | Search depth in plies (~7 full turns) |
| `--top-k` | 5 | Candidate moves at root |
| `--games` | 1 | Number of games to play |
| `--verbose` | off | Print all actions including rolls |
| `--vs-ab2` | off | 2v2: 2 NN seats vs 2 AB2 (2-ply) seats |
| `--1v3` | off | 1v3: 1 NN seat vs 3 AB2 (2-ply) seats |

## How It Works

Each decision:

1. The neural network's **policy head** scores all legal moves and selects the **top 5**
2. Each candidate is rolled forward **30 plies** using policy argmax
3. At the leaf, `base_value_fn` evaluates VP, production, and board position
4. The candidate with the highest leaf value is played

At `--depth 0`, the policy head's top move is played directly (no search).

The state is encoded directly from C game structs into GNN input tensors
(54 nodes x 18 features, 144 edges x 5 features, 115 global features).
The NN forward pass runs in pure C with ARM NEON SIMD.

## Evaluation Modes

**Self-play** (default): All 4 seats use the same NN. Useful for benchmarking speed.

**2v2** (`--vs-ab2`): NN plays seats 0+2, AB2 (2-ply greedy heuristic) plays seats 1+3.
Seat assignments rotate across games to eliminate positional bias.

**1v3** (`--1v3`): NN plays 1 seat, AB2 plays the other 3.
The NN seat rotates across games.

## Model

- **Architecture**: ~1M parameter GNN (4 EdgeConv layers, 6-block ResNet trunk, hierarchical policy head)
- **Training**: Supervised on 2-ply alpha-beta games, refined via self-play
- **Inference**: ~1.3ms per forward pass on Apple Silicon

## Strength

Over 500+ games each:

| Setup | NN Win Rate | Baseline |
|-------|------------|----------|
| 2v2 vs AB2 (0-ply) | 56% | 50% |
| 2v2 vs AB2 (ABt30) | 60% | 50% |
| 1v3 vs AB2 (0-ply) | 35% | 25% |

## Building

Requires only a C compiler and math library:

```bash
./build.sh          # auto-detects platform and BLAS
./build.sh clean    # remove binary
```

| Platform | SIMD | BLAS | Speed |
|----------|------|------|-------|
| macOS Apple Silicon | ARM NEON | Accelerate | ~1.3 ms/inference |
| Linux ARM | ARM NEON | OpenBLAS | ~3 ms/inference |
| Linux x86 | scalar | OpenBLAS | ~5 ms/inference |

## Files

```
catan_player/
  build.sh              Build script
  weights/model.bin     Neural network weights (4.0 MB)
  csrc/
    fast_player.c       Main: state encoding, action encoding, search, game loop
    nn.c, nn.h          Neural network forward pass
    game.c, state.c     Game engine
    value.c             Hand-crafted position evaluator
    actions.c           Legal action generation
    ...                 Supporting modules
```

## Reproducing Results

```bash
# Build
./build.sh

# Self-play benchmark
./catan_player --games 100 --depth 0 --seed 1000

# 2v2 evaluation (500 games, ~11s)
./catan_player --games 500 --depth 0 --vs-ab2 --seed 1000

# 1v3 evaluation (500 games, ~7s)
./catan_player --games 500 --depth 0 --1v3 --seed 1000

# 2v2 with search (slower but stronger, ~50 games)
./catan_player --games 50 --depth 30 --vs-ab2 --seed 1000
```
