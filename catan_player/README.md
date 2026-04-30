# catan_player

Hyper-lightweight, self-contained Catan agents in raw C. No Python runtime,
no training code, no evaluation harnesses outside this folder.

This repo exposes exactly four agents:

- `H-S`: the strongest validated no-ML heuristic search bot.
- `H-S+`: the stronger known-future variant with parallel root evaluation.
- `AB2`: the full strong alpha-beta baseline used inside H-S rollouts.
- `m2_0ply`: M2 neural policy argmax, no search.

## Build

```bash
./build.sh
```

The build produces `./catan_player`. `H-S` and `AB2` do not use neural network
inference. The only runtime data file is `weights/model.bin`, needed by
`m2_0ply`.

## Run

```bash
# H-S no-ML heuristic search self-play
./catan_player --agent h-s --games 10 --seed 810000

# H-S+ parallel single-game optimized search
./catan_player --agent h-s+ --games 10 --seed 810000

# M2 0-ply neural policy self-play
./catan_player --agent m2_0ply --games 10 --seed 810000

# Full strong AB2 self-play
./catan_player --agent ab2 --games 10 --seed 810000

# 2v2 head-to-head, alternating seats each game
./catan_player --h2h --agent h-s --opponent ab2 --games 20 --seed 810000

# FFA: one rotating primary seat against three field seats
./catan_player --ffa --agent h-s+ --opponent ab2 --games 20 --seed 810000
```

Useful flags:

- `--agent h-s|h-s+|ab2|m2_0ply`
- `--opponent h-s|h-s+|ab2|m2_0ply` with `--h2h` or `--ffa`
- `--games N`
- `--seed S`
- `--plus-workers N`, `--plus-k K[,K...]`, `--plus-depth N` for H-S+
- `--plus-leaf-mode N`, `--plus-policy-profile N` for H-S+ tuning
- `--plus-opp-model ab2|det-ab2|det-kf-ab2|det-maxn|det-kf-maxn|hs|hs-leaf` for H-S+ rollout modeling
- `--plus-tt-bits N`, `--plus-pvs 1`, `--plus-lmr 1`, `--plus-id 1` for experimental search modes
- `--plus-root-ensemble N`, `--plus-root-rollout 1`, `--plus-rescue 1`, `--plus-leaf-extend 1` for experimental root/leaf modes
- `--weights PATH` for alternate M2 weights
- `--verbose` for action-level logging

## H-S

`H-S` is the strongest no-ML setup we have validated so far. It is designed
for users who want the best heuristic bot without shipping a model or running
any neural-network inference.

- root/deep move ordering: hand-coded C policy in `csrc/policy_topk.c`
- search depth: `6`
- K schedule: `6,4,2,2,2,2`
- opponent model inside search: AB2
- leaf evaluator: opponent-aware Leaf4 (`base_value_fn` minus 0.1x all-opponent full pressure)
- cache: 1M leaf buckets

Why these settings:

- The hand-coded policy was faster than M2 policy inference and strong enough
  to support a narrow search tree.
- The `6,4,2,2,2,2` schedule keeps the tree narrow while adding one more
  search ply over the original fast H-S baseline.
- Leaf4 plus depth 6 beat the old Leaf0/depth-5 H-S baseline in 1v3 AB2 tests
  and in direct 2v2 H-S H2H validation, so it is now the public heuristic bot.

The old internal name for this setup was `leaf0_search`; `h-s`, `hs`,
`heuristic`, and `leaf0_search` are accepted as aliases, but the bot is called
`H-S` in output and docs.

## H-S+

`H-S+` is the single-game optimized variant for machines with many cores. Its
default config keeps the same depth and K schedule as `H-S`, evaluates root
candidates in parallel, uses the FFA leader-heavy known-future Leaf7 evaluator
with opening-aware policy profile 2, and models opponents with deterministic AB2
so copied `Game.rng` drives exact future rolls, development-card draws, and
robber steals instead of expectimax chance nodes.

On an 18-core Apple Silicon machine, seed `810000` self-play dropped from
about `6.65s` with `H-S` to about `2.0s` with `H-S+`, with the same winner and
final VP vector.

Current FFA default validation:

- Promoted default (`Leaf7 + policy2`) vs old default (`Leaf5 + policy1`),
  seeds `980000,981000,982000,983000,984000,985000,986000,987000`, four games
  per seed block:
  - vs `H-S`: new default `15-17`, old default `12-20`
  - vs `AB2`: new default `27-5`, old default `25-7`
- The FFA sweep harness is `bench/ffa_sweep.py`; use variant `old-default` to
  compare against the previous H-S+ default after this promotion.
- The mixed-seat Elo arena is `bench/elo_arena.py`. It runs four distinct
  variants per game, randomizes seats/seeds, and updates ratings from pairwise
  finish-order comparisons. Use `--variants core` for the viable no-NN pool or
  `--variants all` to include slower/rejected experimental knobs.

For a much faster but less H-S-faithful mode, use the cheap heuristic opponent
model inside search:

```bash
./catan_player --agent h-s+ --games 10 --seed 810000 \
  --plus-depth 7 --plus-k 12,8,4,3,2,2,2 --plus-opp-model hs --plus-time-ms 1000
```

## AB2

`AB2` is the full strong alpha-beta baseline, not a random or greedy player.
It is also the opponent model used inside `H-S` search.

- search depth: `2`
- search type: alpha-beta minimax
- chance handling: expectimax expansion for dice rolls, development-card
  draws, and robber steal outcomes
- evaluator: original `base_value_fn`
- aliases: `ab2`, `AB2`, `strong_ab2`, `full_ab2`

Run it directly:

```bash
./catan_player --agent ab2 --games 10 --seed 810000
```

Run H-S against AB2 in 2v2:

```bash
./catan_player --h2h --agent h-s --opponent ab2 --games 100 --seed 810000
```

## Layout

```text
.
├── build.sh
├── csrc/
│   ├── fast_player.c      CLI and agent dispatch
│   ├── policy_topk.c      M2 action encoding and no-ML policy ordering
│   ├── deep_search.c      H-S recursive search
│   ├── search.c           AB2 expectimax/alpha-beta
│   ├── value.c            base_value_fn
│   ├── nn.c               M2 inference for m2_0ply
│   └── ...                minimal game engine, map, state, actions
└── weights/
    └── model.bin
```
