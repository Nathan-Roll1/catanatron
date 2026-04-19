# Catan Player

Pure C neural network Catan player. Plays 4-player Catan using a learned
policy and hand-crafted value function. No Python, no external dependencies.

## Build & Run

```bash
./build.sh
./catan_player
```

## NN vs AB2 Evaluation

The neural network (0-ply policy argmax) vs AB2 (2-ply alpha-beta with full
chance-node expansion — algorithmically equivalent to catanatron's
`AlphaBetaPlayer(depth=2)`):

```bash
# 2v2: 2 NN seats vs 2 AB2 seats (200 games, ~6s)
./catan_player --games 200 --depth 0 --vs-ab2 --seed 1000
# Expected: NN ~55% WR

# 1v3: 1 NN seat vs 3 AB2 seats (200 games, ~3s)
./catan_player --games 200 --depth 0 --1v3 --seed 1000
# Expected: NN ~32% WR  (25% = random baseline)

# Adjust opponent strength: --ab-depth 1 (greedy) | 2 (default) | 3 (stronger)
./catan_player --games 200 --depth 0 --vs-ab2 --ab-depth 3 --seed 1000
# Expected: NN ~50% WR vs AB3
```

NN seat assignments rotate across games to eliminate positional bias.

## AB2 Implementation

The AB2 opponent is a proper alpha-beta minimax search with **expectimax over
all chance nodes**, matching Python catanatron's `tree_search_utils.execute_spectrum`:

- `ROLL` — expanded into 11 dice outcomes weighted by 2d6 probability
- `BUY_DEVELOPMENT_CARD` — expanded over deck composition, including
  opponent face-down cards (current deck + opponent hidden devs)
- `MOVE_ROBBER` (with steal) — expanded over 5 possible stolen resources

The static evaluator (`base_value_fn` in `csrc/value.c`) mirrors
catanatron's `base_fn`: weights victory points, production, hand synergy,
buildable nodes, longest road, dev cards, and army size.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--depth` | 30 | NN search depth (0 = policy argmax only) |
| `--top-k` | 5 | NN candidates at root for search |
| `--games` | 1 | Number of games |
| `--vs-ab2` | off | 2v2: NN seats vs AB2 seats with rotation |
| `--1v3` | off | 1v3: 1 NN seat vs 3 AB2 seats with rotation |
| `--ab2-only` | off | 4xAB self-play (no NN) |
| `--ab-depth` | 2 | Alpha-beta depth for AB opponents |
| `--verbose` | off | Print all actions |
