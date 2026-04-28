# catan_player

Hyper-lightweight, self-contained Catan agents in raw C. No Python runtime,
no training code, no evaluation harnesses outside this folder.

This repo exposes exactly two agents:

- `H-S`: the strongest validated no-ML heuristic search bot.
- `m2_0ply`: M2 neural policy argmax, no search.

## Build

```bash
./build.sh
```

The build produces `./catan_player`. `H-S` does not use neural network
inference. The only runtime data file is `weights/model.bin`, needed by
`m2_0ply`.

## Run

```bash
# H-S no-ML heuristic search self-play
./catan_player --agent h-s --games 10 --seed 810000

# M2 0-ply neural policy self-play
./catan_player --agent m2_0ply --games 10 --seed 810000

# 2v2 head-to-head, alternating seats each game
./catan_player --h2h --agent h-s --opponent m2_0ply --games 20 --seed 810000
```

Useful flags:

- `--agent h-s|m2_0ply`
- `--opponent h-s|m2_0ply` with `--h2h`
- `--games N`
- `--seed S`
- `--weights PATH` for alternate M2 weights
- `--verbose` for action-level logging

## H-S

`H-S` is the strongest no-ML setup we have validated so far. It is designed
for users who want the best heuristic bot without shipping a model or running
any neural-network inference.

- root/deep move ordering: hand-coded C policy in `csrc/policy_topk.c`
- search depth: `5`
- K schedule: `6,4,2,2,2`
- opponent model inside search: AB2
- leaf evaluator: original `base_value_fn` only, also called Leaf0
- cache: 1M leaf/policy buckets

Why these settings:

- The hand-coded policy was faster than M2 policy inference and strong enough
  to support a narrow search tree.
- The `6,4,2,2,2` schedule gave the best speed/strength balance in our no-ML
  tests.
- Leaf0 beat or matched the opponent-aware leaf variants in 2v2 H2H, so it is
  the default for the public heuristic bot.

The old internal name for this setup was `leaf0_search`; `h-s`, `hs`,
`heuristic`, and `leaf0_search` are accepted as aliases, but the bot is called
`H-S` in output and docs.

## Layout

```text
.
├── build.sh
├── csrc/
│   ├── fast_player.c      CLI and two-agent dispatch
│   ├── policy_topk.c      M2 action encoding and no-ML policy ordering
│   ├── deep_search.c      H-S recursive search
│   ├── search.c           AB2 expectimax/alpha-beta
│   ├── value.c            base_value_fn
│   ├── nn.c               M2 inference for m2_0ply
│   └── ...                minimal game engine, map, state, actions
└── weights/
    └── model.bin
```
