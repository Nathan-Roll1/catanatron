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

## 2026-05-03 Base M2 No-Search Autoresearch

Scope was narrowed to exported base M2 with **no search teacher and no runtime
tree search**. Deployment remains one C `nn_forward` call.

Current kept candidate:

```bash
autoresearch-results/m2_type_bias_road_penalty_x3.pt
csrc/nn_weights_candidate.bin
```

It is `checkpoints/sp_latest2.pt` plus a small action-type bias patch on
`policy_head.type_fc.3.bias`:

| Type | Bias |
|------|------|
| END_TURN | -0.15 |
| BUY_DEV | +0.45 |
| BUILD_SETTLEMENT | +0.45 |
| BUILD_CITY | +0.30 |
| BUILD_ROAD | -1.20 |
| MARITIME | +0.15 |
| TRADE | -0.15 |

Promoted no-search 2v2 checks versus frozen `sp_latest2`:

| Check | Candidate wins |
|-------|----------------|
| 256+256 paired | 259/512 |
| 512+512 paired | 517/1024 |
| 1024+1024 paired | 1039/2048 |
| Aggregate | 1815/3584 = 50.64% |

Interpretation: small positive sign, wide uncertainty, zero inference cost.
`bench_nn_compute` guard passed with full forward around 226-229 us and no
format/parity issue.

Rejected branches:

- Random low-rank Eggroll perturbations of policy/value heads: mostly changed
  logits without stable top-1 action improvements.
- Existing checkpoint sweep: `finetune_dense100_policyonly_lr1e5_freezebn.pt`
  looked promising at 64 games but failed 256+256 paired validation.
- Naive human fine-tuning from `sp_latest2`: meaningful drift but hurt no-search
  game strength.
- KL-anchored human fine-tuning and base-top-k filtered imitation: healthier
  validation metrics, but no robust game improvement.
- Final-layer scope bug fixed: final type classifier is
  `policy_head.type_fc.3`, not `.2`.
- Neighbor hill-climb around the kept road penalty vector did not beat the kept
  candidate on proxy seeds.

## 2026-05-04 No-Search Eggroll / FFA Follow-Up

The deployed no-search candidate is still the `road_penalty_x3` type-bias patch
above. A larger foreground loop added faster population evaluators and tested
several zero-runtime policy surfaces. None robustly replaced the incumbent.

New evaluation harnesses:

- `human_bot/eval_2v2_many_nn_fast.py`: batch many candidate bins against one
  opponent for 2v2 proxy evolution.
- `human_bot/eval_2v2_many_nn_paired_fast.py`: candidate-as-A and candidate-as-B
  paired proxy to reduce seat bias.
- `human_bot/eval_1v3_many_nn_fast.py`: full-seat-sweep free-for-all evaluator.
  Each seed runs the candidate from all four seats against three opponent
  copies, so identical weights give exactly 25%.

Main rejected no-search variants:

- Policy-final bias ES over 134 dims: proxy 20/32, but direct incumbent
  validation was only 1562/3072 = 50.85%.
- Spatial scorer ES on the 2v2 metric: promoted candidates failed direct gates,
  including 1529/3072 and 508/1024 against the incumbent.
- FFA type-bias beam from base M2: mild tall/road and aggressive type-bias
  sweeps stayed at about 25% and often had 0% fixed top-action drift.
- FFA spatial scorer ES from base M2 produced a sibling that beat frozen base
  M2 in role-reversal gates, but it lost the more relevant FFA head-to-head
  versus the retained incumbent: sibling 1517/6144, incumbent 1572/6144.
- FFA spatial scorer ES directly against the incumbent had large proxy leaders
  but collapsed in promotion: best 130/512.
- Compact head-calibration ES (12 type biases plus 8 final-head temperature
  scales) showed a first-gate edge, but the larger fresh gate flattened:
  aggregate candidate 1557/6144 vs incumbent reverse 1521/6144, and 2v2 was
  only 514/1024. This is under the keep threshold for a more complex weight
  patch.

Current interpretation: for a single plain M2 forward pass, `road_penalty_x3`
is still the best retained deployment candidate. The useful next directions are
not more tiny global biases; they are either larger supervised/outcome training
loops, a runtime heuristic policy wrapper, or allowing shallow search.

## 2026-05-04 Eggroll Zoo-Elo Pivot

The Eggroll loop now treats the incumbent as the highest observed policy-zoo
Elo model, not the candidate that clears a fixed 51% league score or original-M2
confirmation gate. The original-M2 1600-game check still runs, but only as a
diagnostic anchor to detect drift.

Current retained incumbent:

- Weights: `csrc/nn_weights_candidate.pt` and `csrc/nn_weights_candidate.bin`
- SHA256: `54c32d952ec769173220063727906b10d86a9b7def4de2d8227380f02f3c9baa`
- Retained metric: 1545.58 policy-zoo Elo
- Kept source: iteration 23, `wide_prev_spatial_p768_g6_lr035_s022`
- Original-M2 anchor at keep time: 0.4919

Post-pivot checks verified the new keep rule:

| Iteration | Decision | Zoo Elo | Delta vs retained | Original-M2 anchor | Setup |
|-----------|----------|--------:|------------------:|-------------------:|-------|
| 28 | discard | 1529.73 | -15.85 | 0.4906 | `wide_prev_spatial_p128_g32_lr030_s020` |
| 29 | discard | 1537.37 | -8.22 | 0.5069 | `wide_prev_final_p256_g12_lr035_s020` |
| 30 | discard | 1514.51 | -31.07 | 0.5181 | `wide_prev_headrel_p192_g12_lr010_s005` |

The bookkeeping metric in `autoresearch-results/state.json` has been renamed to
`policy_zoo_elo` so future status rows do not confuse Elo with paired winrate.
The population exporter now skips test-vector files for intermediate Eggroll
candidates and deletes non-survivor generation artifacts after proxy scoring.

## 2026-05-05 Winner/Search Distillation Loop

New loop file: `human_bot/search_distill_m2.py`.

Goal: improve the fast pure M2 0-ply policy through supervised distillation
without hard-coding AB2 as teacher. The first phase used winner-as-teacher
mixed league games: M2, AB2, incumbent, and rotating past keepers. Later phases
added an optional H-S search player and dense H-S root search targets.

Important retained artifacts:

- Current search-distilled high-water: `autoresearch-results/search_distill_m2/kept_iter_0034.bin`
- Previous robust general baseline: `autoresearch-results/search_distill_m2/kept_iter_0012.bin`
- Search-winner keep: `autoresearch-results/search_distill_m2/kept_iter_0029.bin`

What held up:

- `kept_iter_0012`: best pure winner-BC keeper before adding H-S. It remained
  the robust general baseline after fresh checks.
- `kept_iter_0029`: trained from H-S winner trajectories using full policy-head
  BC on the top winner teacher. Confirmed stronger in include-HS league
  (`candidate 1074.2 / 25.3%` vs `incumbent 1070.2 / 24.7%`) and neutral in
  standard no-HS (`candidate 1070.3 / 28.1%` vs `incumbent 1070.5 / 28.1%`).
- `kept_iter_0034`: dense H-S search-only type-head distillation. Confirmed
  stronger in include-HS league (`candidate 1087.8 / 26.2%` vs
  `incumbent 1083.1 / 25.7%`). Standard no-HS was slightly down but inside
  noise (`candidate 1058.6 / 27.6%` vs `incumbent 1060.6 / 27.9%`).

What did not hold up:

- Strategic-only winner filtering cut too much stabilizing context.
- Doubling winner replay to 8192 games did not produce a robust keep by itself.
- Strict teacher balancing, true-frequency replay, capped replay, and
  disagreement-boosting all produced quick-gate mirages that failed larger
  confirmation.
- Opening-only full policy-head and opening-only settlement/road scorer updates
  did not beat the incumbent.
- Full policy-head dense H-S distillation and full-model dense H-S training
  hurt standard no-HS performance.

Implementation notes:

- AB2 is still only a league member, not a privileged BC teacher.
- H-S support is optional via `--include-hs`; when present, the table remains
  capped at six players by replacing one rotating historical keeper.
- Dense H-S data collected with
  `human_bot/collect_super_m2_dataset.py --all-seats --dense`.
- The useful transfer so far is broad action-type bias from search, not direct
  spatial/action imitation.

## 2026-05-05 0-Ply AB2-Pinned Autoresearch Pivot

The active objective is now stricter and simpler:

- Runtime policy candidates are **0-ply NN argmax only**.
- AB2 remains the only searched player and is fixed as an Elo anchor at 1000.
- H-S/search output may be used only as offline training data, never as the
  runtime policy being scored.
- Promotion uses a fixed zoo table and a Bradley-Terry fit.  A candidate must
  beat the **best non-candidate** in the same table, not merely the named
  incumbent.

New tooling:

- `human_bot/eval_policy_zoo_ab2.py`: fixed mixed-zoo evaluation with AB2
  pinned at Elo 1000 plus bootstrap CIs.
- `human_bot/run_search_distill_0ply_autoresearch.py`: foreground
  winner-trajectory BC loop with fixed no-H-S gates.
- `human_bot/eval_candidate_sweep_ab2.py`: batch candidate sweeps against the
  fixed AB2-pinned zoo.
- `human_bot/make_policy_soups.py`: zero-runtime checkpoint soups / weight
  interpolation between strong policies.
- `human_bot/search_distill_m2.py`: now supports `teacher_allowlist` so BC can
  learn only from selected winner teachers while still seating weaker anchors as
  opponents.

Current bot zoo:

| Label | Runtime | Notes |
|-------|---------|-------|
| `ab2` | AB2 depth-2 search | Fixed Elo anchor at 1000 |
| `m2` | 0-ply NN | Original M2, around 1060 Elo in current zoo tables |
| `eg0143` | 0-ply NN | Older Eggroll high-water, still a useful distinct anchor |
| `eg0150` | 0-ply NN | Strongest pre-distillation Eggroll policy; broad table 1106.5 Elo |
| `sd0029` | 0-ply NN | H-S winner-trajectory keep; non-transitive, often rises on some seeds |
| `sd0034` | 0-ply NN | Dense H-S type-head keep; strong no-search anchor |
| `sd0054` | 0-ply NN | Best broadly re-anchored policy so far |
| `sd0072` | 0-ply NN | Latest active compact-gate keep; needs broad re-anchor |

Stable broad re-anchor with `sd0054` included, gpc=120, AB2 pinned to 1000:

| Rank | Model | Elo | 95% CI | WR |
|-----:|-------|----:|--------|---:|
| 1 | `sd0054` | 1110.0 | [1098.7, 1120.7] | 28.1% |
| 2 | `eg0150` | 1106.5 | [1094.6, 1117.4] | 27.6% |
| 3 | `sd0029` | 1103.2 | [1091.1, 1113.7] | 27.1% |
| 4 | `sd0034` | 1101.5 | [1090.1, 1112.6] | 26.9% |
| 5 | `eg0143` | 1099.5 | [1088.5, 1109.8] | 26.7% |
| 6 | `m2` | 1066.1 | [1054.2, 1077.9] | 22.7% |
| 7 | `ab2` | 1000.0 | fixed | 16.0% |

Retained or important iterations:

| Iteration | Decision | Result | Interpretation |
|-----------|----------|--------|----------------|
| fixed baseline | baseline | `eg0150` led the first broad pure 0-ply zoo at 1106.1 Elo | Old Eggroll was stronger than the first H-S distills under the stricter no-search metric |
| `0041` | keep under old gate | Confirmed +4.2 Elo vs active incumbent, but later broad table put it behind `eg0150`/`sd0034` | Useful but not current high-water |
| `0045` | rejected after audit | Beat the active incumbent but lost to another historical model in the same table | Triggered corrected best-other gate |
| `0054` | keep | Tiny `policy_trunk` update from `eg0150`; confirm 1117.4 vs best other 1113.8, broad re-anchor 1110.0 | First real corrected-gate improvement |
| `0072` | keep | Top-NN-filtered `policy_head` BC from `sd0054`; confirm 1117.3 vs best other `eg0150` 1112.8 | Latest active candidate; next step is broad gpc=120 re-anchor |

What has helped:

- Fixed AB2-pinned zoo evaluation; it exposed non-transitivity and prevented
  several false keeps.
- Tiny trunk unfreezing from diverse rotating winner data (`sd0054`).
- Filtering BC targets to top NN winner teachers, then applying a very small
  policy-head update (`sd0072`).
- Keeping AB2 and M2 in the table as anchors/opponents without forcing them to
  be privileged teachers.

What has not held up:

- Candidate-vs-active-incumbent gates.  These were too easy to exploit because
  another past keeper could be stronger in the same table.
- Type-head-only updates.  Several looked good on quick gates and failed
  confirmation.
- Strategic-only winner filtering.  It removed stabilizing ordinary moves.
- Elite-only league winner data by itself.  It made imitation cleaner but did
  not consistently improve game Elo.
- Checkpoint soups / weight interpolation.  Some soups looked positive at
  gpc=8, but all failed gpc=32 confirmation.

Current best practical export:

```bash
csrc/nn_weights_candidate.pt   # mirrors kept_iter_0072.pt
csrc/nn_weights_candidate.bin  # mirrors kept_iter_0072.bin, ignored by git
```

Immediate next autoresearch step after this commit: broad re-anchor `sd0072`
against `ab2`, `m2`, `eg0143`, `eg0150`, `sd0029`, `sd0034`, and `sd0054` at
gpc=120 with CIs.  If `sd0072` holds, continue with top-NN-filtered head/trunk
micro-updates; if it collapses, revert the active incumbent to `sd0054`.
