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
# Expected: NN ~56% WR vs AB3
```

NN seat assignments rotate across games to eliminate positional bias.

## Benchmark Results

Comprehensive evaluation against the strengthened AB2 (full chance-node
expectimax matching Python catanatron). All numbers below from **8,700+ games**.

### NN 0-ply policy vs AB2

| Setup | Games | NN Wins | NN WR | 95% CI | Random Baseline |
|---|---|---|---|---|---|
| **2v2** (4 × 500 seeds) | 2000 | 1135 | **56.75%** | 54.6 – 58.9% | 50% |
| **1v3** (4 × 500 seeds) | 2000 | 634 | **31.70%** | 29.7 – 33.7% | 25% |

NN-2v2 is >3σ above 50%; NN-1v3 is >3σ above the 25% random baseline.

### Depth Ladder — NN-0ply vs AB at varying depths (2v2, 1000 games each)

| Opponent | NN Wins | NN WR | 95% CI |
|---|---|---|---|
| AB1 | 869 / 1000 | 86.9% | ±2.1% |
| AB2 | 558 / 1000 | 55.8% | ±3.1% |
| AB3 | 555 / 1000 | 55.5% | ±3.1% |

AB3 plateaus relative to AB2 — the underlying `base_value_fn` heuristic
appears to saturate at depth 2 on this position class.

### NN with deep search (ABt30) vs AB2 (200 games, 2v2)

| Configuration | NN Wins | NN WR | 95% CI |
|---|---|---|---|
| ABt30 (NN policy + depth-30 top-5 search) | 120 / 200 | **60.0%** | ±6.8% |

Adding deep search on top of the policy adds ~3pp over pure 0-ply.
Most of the strength is already in the policy.

### 4xAB2 Self-Play — Heuristic Bias Sanity (1500 games)

| Seat | Wins | % | 95% CI |
|---|---|---|---|
| P0 | 170 / 1500 | 11.3% | ±1.6% |
| P1 | 370 / 1500 | 24.7% | ±2.2% |
| P2 | 469 / 1500 | 31.3% | ±2.4% |
| P3 | 491 / 1500 | 32.7% | ±2.4% |

The asymmetry is structural to `base_fn`'s single-enemy heuristic
(`enemy = colors[1] if colors[0] == self else colors[0]`), which causes
seats 1, 2, 3 to all "gang up on" seat 0 in their search. The same bug
exists in Python catanatron but is typically masked by 100-game variance.
NN evals correctly handle it via seat rotation, so headline numbers are
unaffected.

### Reproducing

```bash
./build.sh

# 2v2 headline (~30s)
./catan_player --games 2000 --depth 0 --vs-ab2 --seed 1000
# 1v3 headline (~15s)
./catan_player --games 2000 --depth 0 --1v3 --seed 1000

# Depth ladder
for d in 1 2 3; do ./catan_player --games 1000 --depth 0 --vs-ab2 --ab-depth $d --seed 1000; done

# Deep search variant (~10 min)
./catan_player --games 200 --depth 30 --vs-ab2 --seed 1000

# 4xAB2 self-play sanity
./catan_player --games 1500 --ab2-only --ab-depth 2 --seed 1000
```

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
