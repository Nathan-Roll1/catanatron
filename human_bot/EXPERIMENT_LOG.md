# Human Bot Experiment Log

## Available Checkpoints

| File | What it is | Policy Acc | Value Acc |
|------|-----------|-----------|-----------|
| `checkpoints/human_bot/best.pt` | Original model (old architecture, old loss) | 88.0% | 81.9% |
| `checkpoints/human_bot_experiment/epoch1.pt` | Early experiment (VP-proportional targets, smooth_l1) | 67.9% | 35.8% |
| `checkpoints/human_bot_experiment/latest.pt` | 100k AB2 pretrain + 1 epoch human finetune | 66.2% | 32.3% |
| `checkpoints/human_bot_experiment/ab2_pretrained.pt` | AB2-only pretrained (being rebuilt) | 95.4% | 69.3% |

**The checkpoint that achieved ~42% win rate on 50-game tests was overwritten.** It was the model after 2 full epochs of human data with binary winner targets and no AB2 pretraining.

## How to Evaluate Any Checkpoint

```bash
# 50-game eval at 0-ply, 1-ply, 2-ply (fast, ~2 min)
python3 -m human_bot.eval_search \
  --checkpoint checkpoints/human_bot_experiment/latest.pt \
  --num-games 50 --search-depth 0 1 2

# 500-game eval for stable numbers (~20 min)
python3 -m human_bot.eval_search \
  --checkpoint checkpoints/human_bot_experiment/latest.pt \
  --num-games 500 --search-depth 2
```

## How to Reproduce the Best Results

The best configuration was:
1. No AB2 pretraining (or light 16k pretrain)
2. Binary winner value targets with turn-based loss weighting
3. 2 full epochs over all ~38.5k human games
4. Anti-circular maritime trade filter in eval

```bash
# Step 1: Optional light AB2 pretrain (revert pretrain_ab2.py to use data/ab2_pretrain)
python3 -u -m human_bot.pretrain_ab2

# Step 2: Full epoch on human data
python3 -u -m human_bot.run_experiment --games 38500 --eval-games 50 --reset \
  --pretrained checkpoints/human_bot_experiment/ab2_pretrained.pt

# Step 3: Second epoch (reset cursor, keep model)
python3 -c "
import json
with open('checkpoints/human_bot_experiment/cursor.json') as f:
    c = json.load(f)
c['examples_seen'] = 0
with open('checkpoints/human_bot_experiment/cursor.json', 'w') as f:
    json.dump(c, f)
"
python3 -u -m human_bot.run_experiment --games 38500 --eval-games 50

# Step 4: Evaluate at scale
python3 -m human_bot.eval_search \
  --checkpoint checkpoints/human_bot_experiment/latest.pt \
  --num-games 500 --search-depth 0 1 2
```

## Result History (Reliable Numbers)

All results below use the anti-circular maritime trade filter in `eval_search.py`.

### 500-game evaluations (most reliable)

| Model | 0-ply | 1-ply | 2-ply |
|-------|-------|-------|-------|
| 100k AB2 pretrain + 1 human epoch | 7.9% (39/452) | 2.4% (12/487) | **28.7% (143/356)** |

### 200-game evaluations

| Model | 1-ply | 2-ply |
|-------|-------|-------|
| No pretrain, 2 epochs human (VP targets) | 23.0% (46/154) | 24.0% (47/149) |
| 16k AB2 pretrain + 1 human epoch (VP targets) | 29.3% (58/140) | 31.5% (63/137) |

### 50-game evaluations (high variance, +/- 10%)

| Model | 0-ply | 1-ply | 2-ply |
|-------|-------|-------|-------|
| Before all fixes (baseline) | 0% | 0% | 0% |
| After fixes, no pretrain, 1 epoch 5k games | 0% | 0% | 4% |
| After fixes, no pretrain, 2 epochs 15k games | 6.7% | 12.2% | 22.0% |
| After fixes, no pretrain, 3 epochs 35k games | 9.3% | 6.0% | 8.0% |
| Binary targets, 1 epoch 20k games | 2.6% | 8.0% | **30.0%** |
| Binary targets, 2 epochs 38.5k (best lost ckpt) | 21.4% | **38.8%** | **42.0%** |
| Binary targets, 3 epochs 77k (second pass) | 2.3% | 26.0% | 36.0% |
| 16k AB2 pretrain + 1 human epoch | 8.3% | **38.0%** | 36.0% |
| 16k AB2 pretrain + 1 human (w/ anti-circular) | -- | **42.0%** | 38.0% |

**Note:** The 42% and 38% figures on 50 games are inflated by variance. The true win rate for the best model was likely 28-32% based on 200/500-game runs.

## Fixes Applied (All Active)

1. **Robber steal data bug** (`dataset.py`): Prefers steal over no-steal in action matching
2. **Value loss -> KL divergence** (`loss.py`): Cross-entropy against softmax predictions
3. **Binary winner targets** (`dataset.py`): One-hot winner instead of VP proportions
4. **Turn-based value loss weighting** (`loss.py`): Late-game positions weighted 5x more
5. **Action-type loss weighting** (`loss.py`): Building/trade 3x, ROLL 0.2x
6. **Deeper value head** (`model.py`): +2 ResidualBlocks (510k -> 577k params)
7. **Argmax placement + low temperature** (`eval_search.py`): Deterministic initial placement
8. **Anti-circular trade filter** (`eval_search.py`): Blocks trading back received resources
9. **Vectorized value targets** (`dataset.py`): 100x faster data loading

## Key Findings

### What the NN does well
- Dev card rush strategy (11+ dev cards, 6+ knights for largest army)
- Robber steals (~84% steal rate after fix)
- Initial settlement placement (policy head picks reasonable spots)

### What still needs work
- **Cities**: NN builds 0.3 cities/game in losses vs AB2's 5.2
- **Resource accumulation**: Trades away Ore/Grain needed for cities
- **Build gaps**: 20-34 turn stretches with no building
- **Value head consistency**: Evaluates trades in isolation, can be fooled by state noise

### Architecture notes
- Model: 577k params, GNN encoder (4 layers, 64-dim) + ResNet trunk (6 blocks, 128-ch) + spatial policy (397 actions) + value head (2 ResBlocks)
- Training data: ~10.9M examples from ~40.5k Colonist.io human games
- AB2 pretraining data: ~10.9M (16k games) or ~67.8M (100k games)
