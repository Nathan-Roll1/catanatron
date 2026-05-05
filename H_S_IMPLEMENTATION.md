# Current Best H-S Implementation

`H-S` is our strongest validated no-ML Catan bot. It is a pure heuristic
search agent: no neural-network policy, no training code, and no Python
runtime in the standalone build.

## Where It Lives

The implementation users should run is the standalone repo:

- GitHub remote: `catan-player`
- Local path in this workspace: `catan_player/`
- Main entry point: `catan_player/csrc/fast_player.c`
- Heuristic move ordering: `catan_player/csrc/policy_topk.c`
- Recursive search: `catan_player/csrc/deep_search.c`
- Leaf evaluator: `catan_player/csrc/value.c`
- Full AB2 baseline/opponent model: `catan_player/csrc/search.c`

The parent repo still contains research code, training scripts, old
experiments, and Python evaluation harnesses. Treat `catan_player/` as the
clean product repo for H-S.

## Recommended Build

```bash
cd catan_player
./build.sh
```

This produces:

```bash
./catan_player
```

On macOS the binary links only system libraries (`Accelerate` and
`libSystem`). `H-S` itself does not load model weights.

## Recommended H-S Commands

Run H-S self-play:

```bash
./catan_player --agent h-s --games 10 --seed 810000
```

Run H-S against the full AB2 baseline in 2v2:

```bash
./catan_player --h2h --agent h-s --opponent ab2 --games 100 --seed 810000
```

Compare H-S against M2 0-ply:

```bash
./catan_player --h2h --agent h-s --opponent m2_0ply --games 100 --seed 810000
```

Run AB2 directly:

```bash
./catan_player --agent ab2 --games 10 --seed 810000
```

## H-S Settings

These are the current best no-ML settings:

- Agent name: `H-S`
- Accepted CLI aliases: `h-s`, `hs`, `heuristic`, `leaf0_search`, `leaf0`, `search`
- Search depth: `6`
- Top-K schedule: `6,4,2,2,2,2`
- Root/deep action ordering: hand-coded heuristic policy in `policy_topk.c`
- Opponent model inside search: full AB2
- Leaf evaluator: Leaf4, `base_value_fn` minus 0.1x all-opponent full pressure
- Cache: `1 << 20` leaf/policy buckets
- Immediate-win shortcut: enabled

Why this setup:

- The no-ML policy ordering is much cheaper than M2 policy inference.
- The narrow `6,4,2,2,2,2` schedule adds one ply over the original fast H-S
  while keeping per-decision cost modest.
- The depth-6 Leaf4 setup beat the old depth-5 Leaf0 H-S baseline in both
  AB2 1v3 evaluation and direct 2v2 H-S H2H validation, so it is the current
  recommended no-ML bot.

## Important Performance Note

The standalone `catan_player` binary is single-process and self-contained.
It is slower wall-clock than the parent repo's Python parallel harness because
that harness can distribute root candidates across many worker processes.

Use `catan_player` for the clean public implementation. Use the parent repo's
research harness only when you need parallel evaluation or experiments.
