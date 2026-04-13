# Human Bot: models, configs, and how to run them

## Checkpoints

All checkpoints are PyTorch `.pt` files loadable with:

```python
from human_bot.model import HumanBotNet
net = HumanBotNet.load_checkpoint("checkpoints/<name>.pt", device="cpu")
```

Each has a corresponding C inference binary at `csrc/nn_weights_<name>.bin`, exported with:

```bash
python -m human_bot.export_nn --checkpoint checkpoints/<name>.pt --output csrc/nn_weights_<name>.bin
```

### Checkpoint inventory

| Name | File | Training data | Method |
|------|------|--------------|--------|
| clb | `cluster_run/final.pt` | 100k AB2 games + 44k human games | Supervised pretraining + finetune |
| cl1k | `exit_cluster_r1.pt` | clb + 1k self-play (ABt10 vs AB2) | ExIt round 1, winner-weighted policy |
| cl2k | `cl2k.pt` | cl1k + 1k self-play (all seats ABt10) | ExIt round 2, winner-weighted policy |
| cl3k | `cl3k.pt` | cl2k + 500 self-play (all seats ABt10) | ExIt round 3, winner-weighted policy |
| cl4k | `cl4kv2.pt` | cl3k + 1k self-play (all seats ABt10) | ExIt round 4, separate policy then value training |
| cl5k | `cl5k.pt` | cl4k + 1k self-play (all seats NNt10) | ExIt round 5, first pure-NN self-play round |

### C inference binaries

| File | Model |
|------|-------|
| `csrc/nn_weights_cluster.bin` | clb |
| `csrc/nn_weights_exit_cluster.bin` | cl1k |
| `csrc/nn_weights_cl2k.bin` | cl2k |
| `csrc/nn_weights_cl3k.bin` | cl3k |
| `csrc/nn_weights_cl4kv2.bin` | cl4k |
| `csrc/nn_weights_cl5k.bin` | cl5k |

## Strongest configurations

The best-performing setup we found is **ABt** (NN policy for move ordering + AB2 value function for evaluation + AB2 greedy for response simulation). At ABt5 or ABt10, these models beat pure AB2 ~88% of the time.

**cl3k and cl4k are the strongest models.** They're roughly equal in head-to-head play. cl4k has a separately-trained value head that beats AB2's value function when paired with NN responses (62% over 40 games).

### Quick reference: how to play a game

```python
# Load C NN
nn_lib = ctypes.CDLL("csrc/libnn.dylib")
# ... (set up argtypes)
model_buf = (ctypes.c_char * (8*1024*1024))()
model_ptr = ctypes.cast(model_buf, ctypes.c_void_p)
nn_lib.nn_load(model_ptr, b"csrc/nn_weights_cl4kv2.bin")

# At each decision point:
# 1. Get top-5 candidates from NN policy
candidates = c_top_k(game, legal_actions, k=5)

# 2. For each candidate, simulate forward with AB2 responses
# 3. Evaluate leaf with AB2 value function (base_value_fn)
# 4. Pick the candidate with highest value
```

See `human_bot/eval_search.py` for the full implementation.

## Search configurations

We tested many combinations. Here's what matters:

### Value function

| Value fn | Description | Strength |
|----------|-------------|----------|
| AB2 (`base_value_fn`) | Hand-crafted: VP * 3e14 + production * 1e8 + synergy + buildable nodes | Strong, fast, ~150 lines of C |
| NN value head | Learned from game outcomes | Weaker than AB2 until cl4k; cl4k's value head beats AB2 at depth 10 with NN responses |

### Response simulation

| Response model | Description | Best for |
|---------------|-------------|----------|
| AB2 greedy | `base_value_fn` 1-ply greedy, C function | Fast, accurate when opponent is AB2 |
| NN policy argmax | NN forward pass per step | Slower, better for deep search (15+ ply) and self-play |

### Depth

With AB2 value + AB2 responses, depth saturates at ~5 ply. Going deeper doesn't help because the value function captures most position information in one evaluation. With NN responses, depth helps up to ~10 ply because the trajectories are more realistic.

### Move ordering

The NN policy's top-5 pruning adds ~57% win rate vs AB2's exhaustive search (at same depth, same value function). The policy learned from human games knows which moves are worth considering.

## Expert Iteration (ExIt) training process

This is the process that produced the cl1k through cl5k series. Each round:

### 1. Self-play data collection

Play 500-1000 games where all 4 seats use the current model at depth 10 (ABt10 or NNt10). Record at every multi-choice decision point:
- Game state (node features, edge features, flat features, action mask)
- The search-selected action
- Whether this player won the game

Use C inference (`csrc/libnn.dylib`) for the NN policy during collection. This keeps it fast (~0.3-7s per game depending on whether AB2 or NN responses are used).

Data is saved in 200-game shards to `/tmp/cl<N>k_data/` to avoid OOM.

### 2a. Train policy (value head frozen)

Load the previous checkpoint. Freeze the value head. Train for 3 epochs on the search-selected actions with:
- Winner-weighted cross-entropy: 2.0x weight for winning player actions, 0.5x for losers
- Label smoothing 0.02
- AdamW lr=3e-3, weight_decay=1e-4
- Batch size 4096
- Entropy regularization 0.01
- Gradient clipping 1.0

### 2b. Train value head (everything else frozen)

Freeze encoder, trunk, and policy head. Unfreeze only the value head. Train for 3 epochs on real game outcomes:
- Value target: one-hot winner vector, rotated to current player's perspective
- KL divergence loss with turn-progress weighting
- AdamW lr=1e-3

### 3. Export and evaluate

Export to C binary, then run:
- 50 games ABt5 vs pure AB2 (target: >85%)
- 50 games ABt5 vs previous model (target: >50%)

### Key details

- Different random seeds for each round (300000, 400000, 500000, 600000, 700000, 800000)
- cl1k-cl3k used ABt10 for self-play (AB2 value + AB2 responses)
- cl4k used ABt10 with separate policy/value training
- cl5k switched to NNt10 (NN value + NN responses) for self-play

## Performance summary

### vs pure AB2 (ABt5, 100 games)

| Model | Win rate |
|-------|---------|
| clb (base) | 74% |
| cl1k | 90% |
| cl2k | 87% |
| cl3k | 88% |
| cl4k | 82-86% |

### Head-to-head (ABt10, 150 games)

cl3k and cl4k are roughly tied at the top. cl2k is close behind. All ExIt models beat the base model (clb).

### NN value vs AB2 value (cl4k, NNt10 vs ABt10, 40 games)

cl4k's NN value head wins 62%. This is the first model where the learned value function surpasses the hand-crafted heuristic.

## Architecture

602,253 parameters total:
- GNN encoder: 4 EdgeConvLayers, hidden=64, output=128
- ResNet trunk: 6 blocks, channels=128, input=640 (128 board + 115 flat + 397 mask)
- Hierarchical policy head: 12-way type classification then sub-action scorers
- Value head: fc(128) + BN + 2 ResBlocks(128) + fc(4)

The action mask is fed as a direct input feature (`mask_as_input=True`), giving the model information about what actions are legal.

## C inference

The pure C implementation in `csrc/nn.c` runs the full forward pass (GNN + trunk + value + policy) with ARM NEON SIMD. Compile with:

```bash
cc -shared -O3 -march=native -flto -fPIC -o csrc/libnn.dylib csrc/nn.c -lm
```

Performance on Apple M5 Max: ~1.7ms per value-only call, ~1.8ms per full forward pass. Verified to match PyTorch output within float32 tolerance.

## Files

```
human_bot/
  model.py              - HumanBotNet architecture
  train.py              - Training loop
  loss.py               - Policy and value losses
  expert_iteration.py   - ExIt training script
  eval_search.py        - Benchmark vs AB2
  export_nn.py          - Export weights to C binary format
  search_heuristics.py  - Search-time bonuses (city, settlement, robber steal)
  colonist_converter.py - Convert Colonist.io JSON to training data
  dataset.py            - Data loading and mask fixing

csrc/
  nn.h, nn.c            - Pure C NN inference (NEON SIMD)
  test_nn.c             - Verification and benchmarks
  value.c               - AB2 hand-crafted value function
  libnn.dylib           - Compiled shared library
  nn_weights_*.bin      - Exported model weights
```
