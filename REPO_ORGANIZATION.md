# Repository Organization

This repo now has two clear roles:

1. `catan_player/` is the standalone product repo for H-S, AB2, and M2 0-ply.
2. The parent repo is the research workspace for training, evaluation, exports,
   and experiments.

## Core Product Path

Use this path for the current best no-ML bot:

- `catan_player/`
- `H_S_IMPLEMENTATION.md`

`catan_player/` is the clean, self-contained C implementation. It should stay
small and only contain files needed to build and run the public agents.

## Research Source

These directories are source and should stay in the parent repo:

- `catanatron/`: Python game package.
- `hexzero/`: bindings, encoder, scripts, and training/runtime glue.
- `human_bot/`: research bots, training scripts, evaluation harnesses.
- `csrc/`: parent C research implementation and export/runtime experiments.
- `tests/`: Python and integration tests.

## Generated Artifacts

Generated outputs should not be added to git:

- compiled objects: `*.o`
- local dynamic libraries: `*.dylib`
- debugger bundles: `*.dSYM/`
- exported weight variants: `csrc/nn_weights_*.bin`
- policy heuristic binaries: `csrc/policy_heuristic_*.bin`
- self-play datasets: `csrc/data_super_m2/`
- local dashboards/logs: `csrc/*.html`, `csrc/*.jsonl`, `wandb/`

These are now ignored in `.gitignore`. Existing tracked historical artifacts
should be removed in a dedicated cleanup commit only after deciding which
weights/data must remain reproducible.

## Suggested Future Cleanup

The safest larger cleanup is:

1. Keep `catan_player/` as the only public H-S distribution.
2. Move old experimental Python scripts in `human_bot/` into grouped folders:
   `human_bot/eval/`, `human_bot/training/`, `human_bot/diagnostics/`, and
   `human_bot/legacy/`.
3. Move old standalone C experiment files in `csrc/` into `csrc/experiments/`.
4. Remove tracked build outputs and obsolete weights from git in one explicit
   artifact-pruning commit.
5. Keep only canonical model exports required by current workflows.

Do not mix source reshuffling with algorithm changes; it makes H-S performance
regressions harder to track.

## Current M2 Fine-Tuning Path

M2 is not trained inside `catan_player/`. The clean product artifact is the
flat HBOT fp32 runtime file at `catan_player/weights/model.bin`; the training
artifact is a PyTorch `HumanBotNet` checkpoint in this parent research repo.

The current canonical seed is:

```bash
checkpoints/sp_latest2.pt
```

It uses the product-compatible large shape:

- GNN hidden: `80`
- trunk channels: `192`
- action mask: `397`

The direct offline fine-tune path is:

```bash
python3 -m human_bot.export_nn \
  --checkpoint checkpoints/sp_latest2.pt \
  --output csrc/nn_weights_m2.bin

python3 -m human_bot.collect_super_m2_dataset \
  --games 100 \
  --workers 8 \
  --all-seats \
  --dense \
  --depth 6 \
  --k-schedule 12,8,6,5,4,3 \
  --time-ms 4000 \
  --weights csrc/nn_weights_m2.bin \
  --out csrc/data_super_m2 \
  --seed-base 20000000 \
  --shard-id super_m2_4way_resume_100g

python3 -m human_bot.train_m3_local \
  --seed-checkpoint checkpoints/sp_latest2.pt \
  --shard-dir csrc/data_super_m2 \
  --shard-glob 'super_m2_4way_*g_chunk*.pt' \
  --out-pt checkpoints/m5.pt \
  --out-bin csrc/nn_weights_m5.bin \
  --policy-mode mixed \
  --hard-weight 0.85 \
  --search-only \
  --epochs 2 \
  --batch-size 4096 \
  --lr 5e-6 \
  --label-smoothing 0.02 \
  --search-value-weight 0.5 \
  --val-fraction 0.05
```

Export a kept model back into the product with:

```bash
python3 -m human_bot.export_nn \
  --checkpoint checkpoints/m5.pt \
  --output catan_player/weights/model.bin
```

Use default fp32 export for `catan_player`. The parent research loader has
fp16/int8 support, but the standalone product loader is product-tested against
fp32 HBOT files.

Important current state:

- `csrc/data_super_m2/` is the direct M2 distillation dataset location, but it
  is not present after cleanup.
- Historical logs recorded a complete 300-game super-M2 set
  (`43,346` examples, `453.5 MB`) and a partial 1000-game run
  (`119,789` total examples when combined), but the shards themselves are gone.
- Prior M3/M4 fine-tunes tied or slightly lost to M2 at raw 0-ply; their likely
  value is as better leaf/value heads inside search, not as pure argmax policy.

## Current Cleanup State

The May cleanup keeps source, current research docs, product runtime weights,
product code, useful M2 checkpoints, and human training data. It removes
generated artifacts and old local data that are either rebuildable or not part
of the current 397-action M2 path.

Kept:

- `catan_player/weights/model.bin`
- `checkpoints/sp_latest2.pt`
- recent M2/M3/M4/fine-tune checkpoints
- `data/human_v2_fixed/`
- `data/colonist_raw/games/`

Pruned:

- local virtualenvs, caches, logs, W&B runs, and native build products
- generated C executables, `.o`, `.dylib`, `.dSYM`, dashboard, JSONL, and
  exported old `csrc/*.bin` artifacts
- old 337-action data: `data/abv3/` and `data/ab2_stream/`
- extracted duplicate archive: `data/colonist_raw/games.tar.gz`
- legacy checkpoint families: `v2*`, `abv3*`, `cl*.pt`, old `exit*`, and old
  human-bot pipeline directories
