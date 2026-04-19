# Catan Player

Pure C neural network Catan player. Plays 4-player Catan using a learned
policy and hand-crafted value function. No Python, no external dependencies.

## Build & Run

```bash
./build.sh
./catan_player
```

## Evaluation

```bash
# 2v2: 2 NN seats vs 2 AB2 (2-ply heuristic) seats
./catan_player --games 500 --depth 0 --vs-ab2 --seed 1000
# Expected: NN ~56% WR

# 1v3: 1 NN seat vs 3 AB2 seats
./catan_player --games 500 --depth 0 --1v3 --seed 1000
# Expected: NN ~35% WR (25% = random baseline)

# With search (ABt30): stronger but slower
./catan_player --games 50 --depth 30 --vs-ab2 --seed 1000
# Expected: NN ~60% WR

# Self-play (4x same NN)
./catan_player --games 100 --depth 0 --seed 1000
```

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
