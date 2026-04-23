# Catan Player

Pure C neural network Catan player. Plays 4-player Catan using a learned
policy and the catanatron `base_value_fn` heuristic. No Python, no
external dependencies — single binary, single weights file.

## Quick start

```bash
./build.sh
./catan_player --games 5 --1v3 --super-m2 --seed 95000
```

Expected output: NN wins 4-5 of 5 games against 3 AB2 opponents (random
baseline = 25%).

---

## 4p_super_m2 — recommended setup

Deep recursive minimax search with neural-network policy pruning at every
one of our turns, AB2 opponent simulation in rollouts, and AB-leaf
evaluation.

| Mode | Wins | Win Rate | Avg Rank (NN) | Avg VP (NN/opp) |
|------|------|----------|---------------|-----------------|
| **1v3 vs 3× AB2** (10 games, parallel benchmark) | **8/10** | **80%** | 1.60/4 | 8.9 / 5.0 |
| **1v3 vs 3× AB2** (10 games, this binary, single core) | **10/10** | **100%** | 1.00/4 | 9.4 / 4.5 |

Random baseline in 1v3 is 25%. Pure 0-ply NN baseline is 32% (see below).

### Run it

```bash
# 1v3: 1 NN seat (super_m2) vs 3 AB2 seats — recommended benchmark
./catan_player --games 10 --1v3 --super-m2 --seed 95000

# 2v2: 2 NN seats (super_m2) vs 2 AB2 seats
./catan_player --games 10 --vs-ab2 --super-m2 --seed 95000

# Verbose play-by-play (1 game)
./catan_player --games 1 --1v3 --super-m2 --verbose --seed 95000
```

Per-game wall time: **~75-85 seconds** on Apple M5 Max single core. For
parallel benchmarking (8 games at once on 8 cores), use the Python
runner in the parent catanatron repo (`human_bot/superbot_v3_c2.py`).

### Configuration (hardcoded for super_m2)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `our_depth` | 6 | Recursion depth (number of "our turns" expanded) |
| `k_schedule` | `[12, 8, 6, 5, 4, 3]` | Policy top-K at depths 0..5 |
| `opp_ab_depth` | 2 | Opponents simulated as AB2 with chance nodes |
| `time_budget_sec` | 4.0 | Per-decision wall budget |
| `leaf_cache_bits` | 20 | 1M-entry leaf eval cache (Zobrist-hashed) |

### How it works

At each of our turns during the search:

1. **NN policy ranks legal actions** → keep top-K by policy logit
2. **For each top-K candidate, recurse:**
   - Apply candidate, simulate opponents with deterministic AB2 until our next turn
   - At our next turn, branch top-K again (with smaller K from schedule)
   - Continue to depth-6
3. **At leaves** evaluate with `base_value_fn` (catanatron's hand-crafted heuristic)
4. **Alpha-beta pruning** prunes branches dominated by an already-found win
5. **Terminal-win shortcut**: if any move wins immediately, take it
6. **Leaf cache** with full state hash (board + hands + dev cards + robber)

The pure-C inner loop is critical — see `csrc/deep_search.c`,
`csrc/state_encode.c`, and `csrc/policy_topk.c`. Going through Python
adds ~2x overhead per decision.

### NN ablation

Replacing the NN policy with an action-type-priority heuristic (city >
settle > road > buy_dev > ...) and keeping everything else identical:

| Setup | 1v3 WR (10 games) | Avg Rank |
|-------|------------------|----------|
| **super_m2 with NN** | **80%** | **1.60** |
| super_m2 with action-type heuristic (no NN) | 40% | 2.50 |

The NN contributes +40 percentage points. The deep search amplifies the
policy — bad moves can't recover from search.

---

## All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--super-m2` | off | **Recommended.** Deep recursive search with NN policy pruning |
| `--seed` | 42 | Random seed |
| `--games` | 1 | Number of games |
| `--vs-ab2` | off | 2v2: NN seats vs AB2 seats with rotation |
| `--1v3` | off | 1v3: 1 NN seat vs 3 AB2 seats with rotation |
| `--ab2-only` | off | 4×AB self-play (no NN) |
| `--ab-depth` | 2 | Alpha-beta depth for AB opponents |
| `--depth` | 30 | NN search depth (0 = policy only). Ignored with `--super-m2` |
| `--top-k` | 5 | Root candidates for `--depth` mode. Ignored with `--super-m2` |
| `--verbose` | off | Print all actions |

---

## Reference: weaker baselines

These older modes are kept for benchmarking and reproducing prior
results.

### NN 0-ply policy vs AB2

Pure NN policy argmax (no search), 8,700+ games:

| Setup | Games | NN Wins | NN WR | 95% CI | Random |
|---|---|---|---|---|---|
| **2v2** | 2000 | 1135 | 56.75% | 54.6 – 58.9% | 50% |
| **1v3** | 2000 | 634 | 31.70% | 29.7 – 33.7% | 25% |

```bash
./catan_player --games 2000 --depth 0 --vs-ab2 --seed 1000   # ~30s
./catan_player --games 2000 --depth 0 --1v3 --seed 1000      # ~15s
```

### Depth ladder — NN-0ply vs AB at varying depths (2v2, 1000 games)

| Opponent | NN Wins | NN WR |
|---|---|---|
| AB1 | 869 / 1000 | 86.9% |
| AB2 | 558 / 1000 | 55.8% |
| AB3 | 555 / 1000 | 55.5% |

AB3 plateaus relative to AB2 — `base_value_fn` saturates at depth 2.

```bash
for d in 1 2 3; do
    ./catan_player --games 1000 --depth 0 --vs-ab2 --ab-depth $d --seed 1000
done
```

### NN ABt30 (rollout-style) vs AB2 (2v2, 200 games)

| Configuration | NN Wins | NN WR |
|---|---|---|
| ABt30 (NN policy + depth-30 top-5 single-path rollout) | 120 / 200 | 60.0% |

```bash
./catan_player --games 200 --depth 30 --vs-ab2 --seed 1000   # ~10 min
```

ABt30 adds ~3pp over 0-ply. Compare to **super_m2 which adds ~25pp**.
The branching at every "our turn" (rather than single-path rollout) is
the key difference.

### 4×AB2 self-play sanity (1500 games)

| Seat | Wins | % |
|---|---|---|
| P0 | 170 / 1500 | 11.3% |
| P1 | 370 / 1500 | 24.7% |
| P2 | 469 / 1500 | 31.3% |
| P3 | 491 / 1500 | 32.7% |

Asymmetry is structural to `base_fn`'s single-enemy heuristic
(`enemy = first non-self color`), causing seats 1, 2, 3 to all "gang up
on" seat 0 in their search. Same bug exists in Python catanatron, masked
by 100-game variance. NN evals handle it via seat rotation.

```bash
./catan_player --games 1500 --ab2-only --ab-depth 2 --seed 1000
```

---

## AB2 Implementation

The AB2 opponent (used for benchmarks AND inside super_m2's rollouts) is
a proper alpha-beta minimax search with **expectimax over all chance
nodes**, matching Python catanatron's `tree_search_utils.execute_spectrum`:

- `ROLL` — expanded into 11 dice outcomes weighted by 2d6 probability
- `BUY_DEVELOPMENT_CARD` — expanded over deck composition, including
  opponent face-down cards (current deck + opponent hidden devs)
- `MOVE_ROBBER` (with steal) — expanded over 5 possible stolen resources

The static evaluator (`base_value_fn` in `csrc/value.c`) mirrors
catanatron's `base_fn`: weights victory points, production, hand
synergy, buildable nodes, longest road, dev cards, and army size.

---

## Project layout

```
.
├── README.md             (this file)
├── build.sh              (single-step build script)
├── catan_player          (compiled binary, .gitignored)
├── csrc/                 (all sources, no external deps)
│   ├── actions.{c,h}     game engine: legal action generation
│   ├── apply_action.{c,h}game engine: state transitions
│   ├── board.{c,h}       game engine: board topology
│   ├── catan_types.h     enums and constants
│   ├── deep_search.{c,h} super_m2 recursive search
│   ├── fast_player.c     main, mode dispatch
│   ├── game.{c,h}        game engine: top-level
│   ├── map.{c,h}         game engine: map generation
│   ├── nn.{c,h}          NN inference (NEON / Accelerate / OpenBLAS)
│   ├── nn_topology.h     NN architecture constants
│   ├── policy_topk.{c,h} encode + nn_forward + top-K
│   ├── rng.{c,h}         Mersenne Twister
│   ├── search.{c,h}      alpha-beta with expectimax
│   ├── state.{c,h}       game state struct
│   ├── state_encode.{c,h}NN input encoding
│   └── value.{c,h}       base_value_fn heuristic
└── weights/
    └── model.bin         (3.9 MB M2 NN weights, ~1M params)
```

Builds with `cc -O3 -march=native -flto`. macOS uses Accelerate; Linux
optionally uses OpenBLAS (auto-detected). Runs single-threaded.
