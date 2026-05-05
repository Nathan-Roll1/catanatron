# Human Bot: models, configs, and how to run them

## Current 0-Ply Policy Zoo (2026-05-05)

The active objective is **pure 0-ply neural policy strength**.  Runtime play is
one `nn_forward` argmax per move.  AB2 remains a fixed 2-ply Catanatron search
anchor for evaluation only and is pinned to Elo 1000.

Current active candidate pointer:

```bash
csrc/nn_weights_candidate.pt
csrc/nn_weights_candidate.bin
```

These currently mirror `search_distill_m2/kept_iter_0054`.  The binary is a
local export and is ignored by git; the `.pt` checkpoint is the portable copy.

Important bots:

| Label | Runtime | Artifact | Current interpretation |
|-------|---------|----------|------------------------|
| `ab2` | AB2 2-ply search | built-in C value/search | Elo anchor fixed at 1000 |
| `m2` | 0-ply NN | `csrc/nn_weights_m2.bin` | original M2 baseline, around 1060 Elo in the current 4-way zoo |
| `eg0143` | 0-ply NN | `autoresearch-results/eggroll_m2_hillclimb/kept_iter_0143.*` | older Eggroll high-water, still a useful style anchor |
| `eg0150` | 0-ply NN | `autoresearch-results/eggroll_m2_hillclimb/kept_iter_0150.*` | strongest pre-distillation Eggroll policy, around 1106 Elo |
| `sd0029` | 0-ply NN | `autoresearch-results/search_distill_m2/kept_iter_0029.*` | H-S winner-trajectory keep, useful non-transitive anchor |
| `sd0034` | 0-ply NN | `autoresearch-results/search_distill_m2/kept_iter_0034.*` | dense H-S type-head keep, strong in several no-search tables |
| `sd0054` | 0-ply NN | `autoresearch-results/search_distill_m2/kept_iter_0054.*` | current best broadly re-anchored policy |
| `sd0072` | 0-ply NN | `autoresearch-results/search_distill_m2/kept_iter_0072.*` | compact-gate keep that did not beat `sd0054` on broad re-anchor |

The best validated broad table so far, after the `sd0072` audit:

| Model | Elo | 95% CI | Win rate |
|-------|----:|--------|---------:|
| `sd0054` | 1112.0 | [1103.6, 1121.6] | 28.0% |
| `sd0072` | 1110.9 | [1102.2, 1119.7] | 27.9% |
| `sd0034` | 1104.3 | [1096.0, 1113.5] | 27.1% |
| `eg0150` | 1101.8 | [1093.1, 1112.0] | 26.7% |
| `sd0029` | 1099.9 | [1090.6, 1109.6] | 26.5% |
| `eg0143` | 1099.3 | [1090.1, 1108.8] | 26.4% |
| `m2` | 1055.8 | [1046.1, 1064.2] | 21.4% |
| `ab2` | 1000.0 | fixed | 16.0% |

Promotion rule now used for 0-ply research: a candidate must beat the **best
non-candidate** in the same fixed zoo table, not merely the named incumbent.
This fixed an earlier bad keep where a candidate beat the current incumbent but
lost to another historical policy in the same table.

## Checkpoints

All checkpoints are PyTorch `.pt` files loadable with:

```python
from human_bot.model import HumanBotNet
net = HumanBotNet.load_checkpoint("checkpoints/<name>.pt", device="cpu")
```

Each has a corresponding C inference binary, exported with:

```bash
python -m human_bot.export_nn --checkpoint checkpoints/<name>.pt --output csrc/nn_weights_<name>.bin
```

Current M2 baseline:

```bash
checkpoints/sp_latest2.pt
```

This is the preferred fine-tuning seed for product-compatible M2 work. It uses
the large `80/192/397` shape expected by `catan_player`. Use
`checkpoints/imported_m2_from_bin.pt` only as a reference/fallback when exact
binary reconstruction matters.

## Architecture

Two model sizes:

### Small (602k params, default)
- GNN encoder: 4 EdgeConvLayers, hidden=64, output=128
- ResNet trunk: 6 blocks, channels=128, input=640 (128 board + 115 flat + 397 mask)
- Hierarchical policy head: 12-way type classification then sub-action scorers
- Value head: fc(128) + BN + 2 ResBlocks(128) + fc(4)

### Large (~1M params)
- GNN encoder: 4 EdgeConvLayers, hidden=80, output=128
- ResNet trunk: 6 blocks, channels=192, input=640
- Same policy and value head structure, wider layers
- Set via `GNN_HIDDEN=80 TRUNK_CHANNELS=192`

The action mask is fed as a direct input feature (`mask_as_input=True`), giving the model information about what actions are legal.

## Search configurations

### Value function

| Value fn | Description |
|----------|-------------|
| AB2 (`base_value_fn`) | Hand-crafted: VP * 3e14 + production * 1e8 + synergy + buildable nodes. ~150 lines of C |
| NN value head | Learned from game outcomes. 4-way output (per-seat win probability) |

### Response simulation

| Response model | Description |
|---------------|-------------|
| AB2 greedy | `base_value_fn` 1-ply greedy, C function |
| NN policy argmax | NN forward pass per step |

### Depth

With AB2 value + AB2 responses, depth saturates at ~5 ply. With NN responses, depth helps up to ~10 ply because the trajectories are more realistic.

## Training pipeline (v4)

### 1. AB2 pretraining data

100k games where all 4 seats use AB2 with 2-ply search (real opponent response modeling). Generated with:

```bash
python -m hexzero.scripts.collect_ab2_games --output-dir data/ab2_v4 --num-games 100000 --depth 2
```

### 2. Human game data

~44k Colonist.io games converted with fixed port encoding:

```bash
python -m human_bot.colonist_converter --input-dir data/colonist_raw/games --output-dir data/human_v4
```

### 3. Pretraining

AB2 supervised pretraining + human finetuning:

```bash
python -u human_bot/cluster_train_inner.py --ab2-dir data/ab2_v4 --human-dir data/human_v4 --ckpt-dir checkpoints/pretrain_v4
```

### 4. Self-play (ExIt)

Distributed GPU self-play with depth-5 batched search:

```bash
python -u human_bot/selfplay.py --checkpoint checkpoints/pretrain_v4/final.pt --role all --num-actor-gpus 99 --actors-per-gpu 4 --search-depth 1 --deep-search-depth 5
```

Features:
- Graded rewards: winner gets 1.0 + speed bonus, losers get VP/20
- Speed-weighted step weights: shorter games = stronger signal
- Dirichlet noise exploration at search root
- Random setup exploration (20% of games)
- Temperature annealing: 1.0 → 0.2 over 200 rounds
- Multi-node support: actor-only nodes write shards to shared NFS

### C-inference self-play (alternative)

For CPU-only actors with C inference (best for ARM/Apple Silicon):

```bash
python -u human_bot/c_selfplay.py --checkpoint checkpoints/pretrain_v4/final.pt --num-actors 90 --search-depth 10
```

## Current M2 Fine-Tuning Loop

For local super-M2 distillation, export the seed, collect dense search-labeled
shards, fine-tune conservatively, then export back to the product:

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

python3 -m human_bot.export_nn \
  --checkpoint checkpoints/m5.pt \
  --output catan_player/weights/model.bin
```

Use `--policy-mode mixed --hard-weight 0.85 --search-only` as the default
starting point. Earlier pure soft-target fine-tunes improved dense validation
signals but hurt raw 0-ply argmax play, so any kept model should be evaluated
both as `m2_0ply` and as a value/policy component inside search.

Do not export `fp16` or `int8` for the standalone `catan_player` product unless
its loader is updated and retested; keep product `model.bin` as fp32 HBOT.

## C inference

Pure C implementation in `csrc/nn.c` with:
- ARM NEON SIMD (Apple Silicon)
- AVX2+FMA SIMD (x86)
- OpenBLAS for batched GEMMs

Compile:

```bash
# macOS (ARM)
cc -shared -O3 -march=native -flto -fPIC -o csrc/libnn.dylib csrc/nn.c -lm -framework Accelerate

# Linux (x86, with OpenBLAS)
cc -shared -O3 -march=native -flto -fPIC -DHAVE_CBLAS -o csrc/libnn.so csrc/nn.c -lm -lopenblas
```

## Files

```
human_bot/
  model.py              - HumanBotNet architecture
  train.py              - Training loop
  loss.py               - Policy and value losses
  selfplay.py           - GPU self-play (distributed actors + learner)
  c_selfplay.py         - CPU self-play with C inference
  eval_search.py        - Benchmark vs AB2
  export_nn.py          - Export weights to C binary format
  search_heuristics.py  - Search-time bonuses (city, settlement, robber steal)
  colonist_converter.py - Convert Colonist.io JSON to training data
  dataset.py            - Data loading and mask fixing
  cluster_train_inner.py - Streaming pretrain on cluster

csrc/
  nn.h, nn.c            - Pure C NN inference (NEON + AVX2 SIMD)
  value.c               - AB2 hand-crafted value function
  nn_weights_*.bin      - Exported model weights
```
