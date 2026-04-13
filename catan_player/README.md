# Catan Player

Pure C neural network Catan player. No Python, no external dependencies
beyond a C compiler. Plays 4-player Catan using a learned policy for move
selection and a hand-crafted value function for position evaluation.

## Quick Start

```bash
./build.sh            # compile (~2s)
./catan_player        # play a game
```

## Usage

```bash
./catan_player                         # ABt30, seed 42
./catan_player --seed 777              # different board
./catan_player --depth 20              # faster search
./catan_player --games 10              # multi-game benchmark
./catan_player --verbose               # print every action
./catan_player --seed 42 --depth 0     # policy only (no search)
```

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed for board generation |
| `--depth` | 30 | Search depth in plies (~7 full turns) |
| `--top-k` | 5 | Candidate moves at root |
| `--games` | 1 | Number of games to play |
| `--verbose` | off | Print all actions including rolls |

## How It Works

Each decision:

1. The neural network's **policy head** scores all legal moves and selects the **top 5**
2. Each candidate is rolled forward **30 plies** using policy argmax
3. At the leaf, `base_value_fn` evaluates VP, production, and board position
4. The candidate with the highest leaf value is played

The state is encoded directly from C game structs into GNN input tensors
(54 nodes x 18 features, 144 edges x 5 features, 115 global features).
The NN forward pass runs in pure C with ARM NEON SIMD.

## Model

- **Architecture**: 602k parameter GNN (4 EdgeConv layers, 6-block ResNet trunk, hierarchical policy head)
- **Training**: Supervised on 100k 2-ply alpha-beta games, refined via Expert Iteration
- **Inference**: ~1.8ms per forward pass on Apple Silicon

## Strength

Elo 1606 over 2000 games (95% CI: 1577-1635), ~180 Elo above the hand-crafted AB2 heuristic.

## Building

Requires only a C compiler and math library:

```bash
./build.sh          # auto-detects platform and BLAS
./build.sh clean    # remove binary
```

| Platform | SIMD | BLAS | Speed |
|----------|------|------|-------|
| macOS Apple Silicon | ARM NEON | Accelerate | ~1.8 ms/inference |
| Linux ARM | ARM NEON | OpenBLAS | ~3 ms/inference |
| Linux x86 | scalar | OpenBLAS | ~5 ms/inference |

## Files

```
catan_player/
  build.sh              Build script
  weights/model.bin     Neural network weights (2.4 MB)
  csrc/
    fast_player.c       Main: state encoding, action encoding, search, game loop
    nn.c, nn.h          Neural network forward pass
    game.c, state.c     Game engine
    value.c             Hand-crafted position evaluator
    actions.c           Legal action generation
    ...                 Supporting modules
```
