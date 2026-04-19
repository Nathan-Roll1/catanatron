# Catan Player

Pure C neural network Catan player. Plays 4-player Catan using a learned
policy and hand-crafted value function. No Python, no external dependencies.

## Build & Run

```bash
./build.sh
./catan_player
```

## NN vs AB2 Evaluation

The neural network (0-ply policy argmax) vs AB2 (2-ply greedy heuristic):

```bash
# 2v2: 2 NN seats vs 2 AB2 seats (500 games, ~12s)
./catan_player --games 500 --depth 0 --vs-ab2 --seed 1000
# Expected: NN ~56% WR

# 1v3: 1 NN seat vs 3 AB2 seats (500 games, ~8s)
./catan_player --games 500 --depth 0 --1v3 --seed 1000
# Expected: NN ~35% WR (25% = random baseline)
```

NN seat assignments rotate across games to eliminate positional bias.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--depth` | 30 | Search depth (0 = policy only) |
| `--top-k` | 5 | Candidates at root for search |
| `--games` | 1 | Number of games |
| `--vs-ab2` | off | 2v2: NN vs AB2 with seat rotation |
| `--1v3` | off | 1v3: NN vs AB2 with seat rotation |
| `--verbose` | off | Print all actions |
