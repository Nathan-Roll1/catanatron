# HexaZero: Jag GPU Deployment Plan

## 1. Storage Layout

```
/nlp/scr/USERNAME/hexazero/          # Main project dir (within 300GB scratch)
├── code/                             # Git clone of catanatron repo
├── envs/                             # Conda environment
├── checkpoints/                      # Model checkpoints (~50MB each)
├── replay_buffer/                    # Serialized replay buffers (~4GB each)
├── elo_history/                      # ELO tracking JSON files
└── logs/                             # Training logs, slurm output

/scr-ssd/hexazero/                    # LOCAL SSD on each jag (fast I/O)
├── replay_buffer/                    # Local copy for training reads
└── selfplay_cache/                   # Temp self-play game data
```

**Rationale**: Self-play writes thousands of small game records. Writing
directly to juice servers will hammer NFS. Instead, write to local
`/scr-ssd/` on the compute node, then rsync snapshots to
`/nlp/scr/USERNAME/hexazero/` periodically.

---

## 2. Environment Setup

Run these from `sc` (login node):

```bash
# 1. Install miniconda at /nlp/scr/USERNAME if not done
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /nlp/scr/$USER/miniconda3

# 2. Create hexazero conda env
conda create -n hexazero python=3.11 -y
conda activate hexazero

# 3. Install PyTorch (match CUDA on jags -- check /usr/local/cuda* on a jag)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install other deps
pip install numpy

# 5. Clone and set up
cd /nlp/scr/$USER/hexazero/code
git clone <repo-url> catanatron
cd catanatron

# 6. Build the C shared library
python -m hexzero.bindings.build_lib
```

---

## 3. GPU Selection Strategy

The jagupards have mixed GPU generations. Check the cluster dashboard to
identify which jags have which GPUs. General guidance:

| GPU Class         | Typical Jags     | VRAM  | Use For               |
|-------------------|------------------|-------|-----------------------|
| RTX 3090 / A5000  | varies           | 24GB  | Training + Self-play  |
| RTX 2080 Ti       | varies           | 11GB  | Self-play only        |
| GTX 1080 Ti       | varies           | 11GB  | Self-play only        |
| Titan X / older   | varies           | 12GB  | Self-play only        |

The HexaZero model is only **3.8M parameters** (~15MB), so it fits on
any GPU. The constraint is batch size during training:

- **11GB VRAM**: batch_size=1024, mixed precision
- **24GB VRAM**: batch_size=2048, mixed precision (default config)

**Exclude weak jags for training** (check dashboard, adjust these):
```bash
# Example: exclude jags with GPUs too old for CUDA 12
nlprun -q jag -g 1 -r 60G -c 16 -p standard -x jagupard[18-25]
```

---

## 4. Job Architecture

HexaZero has three distinct workloads that run as separate Slurm jobs:

```
┌─────────────────────────────────────────────────────┐
│                   COORDINATOR                        │
│  (runs on sc or a john node, no GPU)                │
│  Launches self-play and training jobs.              │
│  Monitors progress. Triggers evaluation.            │
└──────────┬──────────────────────┬───────────────────┘
           │                      │
    ┌──────▼──────┐       ┌──────▼──────┐
    │ SELF-PLAY   │       │  TRAINING   │
    │ jag-lo/std  │       │ jag-std/imp │
    │ 1 GPU each  │       │ 1 GPU       │
    │ N parallel  │       │ 1 job       │
    │ workers     │       │             │
    └──────┬──────┘       └──────┬──────┘
           │   writes              │ reads
           ▼                       ▼
    ┌──────────────────────────────────┐
    │     REPLAY BUFFER (on disk)      │
    │  /nlp/scr/USER/hexazero/replay/  │
    └──────────────────────────────────┘
```

---

## 5. Self-Play Jobs

Self-play is the workhorse. Each worker plays games using MCTS + the
current neural network checkpoint. This is where you want the most
parallelism.

### Script: `scripts/selfplay.sh`

```bash
#!/bin/bash
# Self-play worker -- launched via nlprun from sc

PROJ=/nlp/scr/$USER/hexazero
CKPT=$PROJ/checkpoints/latest.pt
BUFFER=$PROJ/replay_buffer/buffer.pt
LOCAL=/scr-ssd/hexazero_sp_$$

# Set up local scratch
mkdir -p $LOCAL

conda activate hexazero
cd $PROJ/code/catanatron

python -m hexzero.scripts.selfplay_worker \
    --checkpoint $CKPT \
    --output-dir $LOCAL \
    --games-per-batch 25 \
    --mcts-sims 200 \
    --device cuda \
    --seed $SLURM_JOB_ID

# Merge local results into shared replay buffer
python -m hexzero.scripts.merge_buffer \
    --source $LOCAL \
    --target $BUFFER

# Clean up local scratch
rm -rf $LOCAL
```

### Launching Self-Play (from sc)

```bash
# Launch 6 parallel self-play workers on jag-lo (cheap, preemptable)
for i in $(seq 1 6); do
    nlprun -q jag -g 1 -r 30G -c 8 -p low \
        "bash /nlp/scr/$USER/hexazero/code/catanatron/scripts/selfplay.sh"
done

# Or for more important runs, use jag-standard
for i in $(seq 1 4); do
    nlprun -q jag -g 1 -r 30G -c 8 -p standard \
        "bash /nlp/scr/$USER/hexazero/code/catanatron/scripts/selfplay.sh"
done
```

### Self-Play Config Tuning for Jags

Start conservative and scale up:

| Phase     | MCTS Sims | Games/Worker | Workers | GPU Hours/Iter |
|-----------|-----------|--------------|---------|----------------|
| Phase 1   | 100       | 50           | 4       | ~2h            |
| Phase 2   | 400       | 50           | 6       | ~8h            |
| Phase 3   | 800       | 50           | 8       | ~24h           |

With 100 MCTS sims and the 3.8M param model, expect ~2-3 games/minute
per worker on a modern jag GPU.

---

## 6. Training Jobs

Training reads from the replay buffer and updates the network. This is a
single-GPU job that runs periodically after self-play generates enough
data.

### Script: `scripts/train.sh`

```bash
#!/bin/bash
# Training job -- single GPU

PROJ=/nlp/scr/$USER/hexazero
BUFFER=$PROJ/replay_buffer/buffer.pt
CKPT_DIR=$PROJ/checkpoints
LOCAL=/scr-ssd/hexazero_train_$$

mkdir -p $LOCAL $CKPT_DIR

# Copy replay buffer to local SSD for fast reads
cp $BUFFER $LOCAL/buffer.pt

conda activate hexazero
cd $PROJ/code/catanatron

python -m hexzero.scripts.train \
    --replay-buffer $LOCAL/buffer.pt \
    --checkpoint-dir $CKPT_DIR \
    --epochs 10 \
    --batch-size 2048 \
    --lr 0.001 \
    --device cuda \
    --amp

# Copy latest checkpoint to shared storage
cp $CKPT_DIR/latest.pt $PROJ/checkpoints/latest.pt

rm -rf $LOCAL
```

### Launching Training (from sc)

```bash
# Single GPU training job -- use jag-important so it won't be preempted
nlprun -q jag -g 1 -r 60G -c 16 -p important \
    "bash /nlp/scr/$USER/hexazero/code/catanatron/scripts/train.sh"
```

---

## 7. Evaluation Jobs

Evaluation pits the new checkpoint against AB2 (baseline at 100 ELO).
This is lightweight -- mostly C engine computation with occasional
GPU inference.

### Script: `scripts/evaluate.sh`

```bash
#!/bin/bash
# Evaluation job -- measure ELO against AB2 baseline

PROJ=/nlp/scr/$USER/hexazero

conda activate hexazero
cd $PROJ/code/catanatron

python -m hexzero.scripts.evaluate \
    --checkpoint $PROJ/checkpoints/latest.pt \
    --num-games 100 \
    --elo-file $PROJ/elo_history/ratings.json \
    --device cuda
```

### Launching Evaluation (from sc)

```bash
# Evaluation is fast, use jag-lo
nlprun -q jag -g 1 -r 20G -c 8 -p low \
    "bash /nlp/scr/$USER/hexazero/code/catanatron/scripts/evaluate.sh"
```

---

## 8. Entry-Point Scripts to Create

These Python entry points need to be written in `hexzero/scripts/`:

```
hexzero/scripts/
├── __init__.py
├── selfplay_worker.py    # Loads checkpoint, plays N games, saves examples
├── merge_buffer.py       # Merges local game data into shared replay buffer
├── train.py              # Loads buffer, trains for N epochs, saves checkpoint
├── evaluate.py           # Runs arena evaluation, updates ELO file
└── run_iteration.py      # Orchestrates one full self-play -> train -> eval cycle
```

---

## 9. Iteration Loop (Manual or Automated)

### Manual (Recommended Initially)

```bash
# From sc, run one full iteration:

# Step 1: Self-play (parallel, ~2-8 hours depending on config)
for i in $(seq 1 4); do
    nlprun -q jag -g 1 -r 30G -c 8 -p low \
        "bash scripts/selfplay.sh" &
done
wait  # Wait for all self-play jobs to finish

# Step 2: Train (single GPU, ~30 min)
nlprun -q jag -g 1 -r 60G -c 16 -p important \
    "bash scripts/train.sh"

# Step 3: Evaluate (single GPU, ~15 min)
nlprun -q jag -g 1 -r 20G -c 8 -p low \
    "bash scripts/evaluate.sh"

# Step 4: Check ELO progress
cat /nlp/scr/$USER/hexazero/elo_history/ratings.json | python -m json.tool
```

### Automated (After Debugging)

Create a coordinator script that submits Slurm jobs with dependencies:

```bash
# Submit self-play, training depends on self-play, eval depends on training
SP_JOB=$(sbatch --parsable scripts/selfplay_slurm.sh)
TR_JOB=$(sbatch --parsable --dependency=afterok:$SP_JOB scripts/train_slurm.sh)
EV_JOB=$(sbatch --parsable --dependency=afterok:$TR_JOB scripts/evaluate_slurm.sh)
echo "Pipeline: selfplay=$SP_JOB -> train=$TR_JOB -> eval=$EV_JOB"
```

---

## 10. Queue Strategy

| Job Type    | Queue           | Priority    | GPUs | Why                                      |
|-------------|-----------------|-------------|------|------------------------------------------|
| Self-play   | jag             | low         | 1    | Bulk work, OK to be preempted            |
| Self-play   | jag             | standard    | 1    | When approaching a deadline              |
| Training    | jag             | important   | 1    | Must not be interrupted mid-epoch        |
| Evaluation  | jag             | low         | 1    | Fast, cheap, can retry                   |
| Data merge  | john            | low         | 0    | CPU-only buffer merging                  |

**GPU budget per iteration** (Phase 1 settings):
- Self-play: 4 workers x 1 GPU x 2h = **8 GPU-hours**
- Training: 1 GPU x 0.5h = **0.5 GPU-hours**
- Evaluation: 1 GPU x 0.25h = **0.25 GPU-hours**
- **Total: ~9 GPU-hours per iteration**

Target: 10 iterations/day on jag-lo = **~90 GPU-hours/day**

---

## 11. Checkpointing & Fault Tolerance

Since `jag-lo` jobs can be preempted at any time:

1. **Self-play workers** save completed games to disk after each game
   (not at the end of the batch). A preempted worker loses at most 1
   in-progress game.

2. **Training** checkpoints after every epoch. On restart, resume from
   the last epoch checkpoint.

3. **Replay buffer** is append-only on shared storage. Multiple workers
   can write concurrently (file-level locking via `fcntl`).

4. **ELO history** is a JSON append log. Never overwritten, only
   appended.

---

## 12. Monitoring

```bash
# Check your running jobs
squeue -u $USER

# Check GPU utilization on a jag (from sc)
ssh jagupard10 nvidia-smi

# Watch training loss (tail the slurm output)
tail -f /nlp/scr/$USER/hexazero/logs/slurm-*.out

# Check ELO progress
python -c "
import json
data = json.load(open('/nlp/scr/$USER/hexazero/elo_history/ratings.json'))
for row in sorted(data.get('ratings', {}).items(), key=lambda x: -x[1]):
    print(f'{row[0]:20s} {row[1]:8.1f}')
"

# Check replay buffer size
python -c "
import torch
buf = torch.load('/nlp/scr/$USER/hexazero/replay_buffer/buffer.pt', weights_only=False)
print(f'Buffer size: {buf[\"size\"]:,} positions')
"
```

---

## 13. Phase Plan

### Phase 1: Validate Pipeline (Days 1-3)
- **Goal**: End-to-end pipeline works, ELO tracking functional
- 100 MCTS sims, 4 self-play workers, 50 games/iter
- 10 training epochs, batch_size=1024
- Run on `jag-lo`, no need for priority
- Verify: checkpointing, buffer merge, ELO updates

### Phase 2: Initial Training (Days 4-14)
- **Goal**: HexaZero beats random player, approaches AB2
- 400 MCTS sims, 6 workers, 100 games/iter
- 20 iterations total (~180 GPU-hours)
- Track ELO: Random=~0, AB2=100, HexaZero=?

### Phase 3: Scaling Up (Days 15-30)
- **Goal**: HexaZero consistently beats AB2
- 800 MCTS sims, 8 workers, 100 games/iter
- 50+ iterations (~900 GPU-hours)
- Switch training to `jag-important` for stability
- Tune hyperparameters based on loss curves

### Phase 4: Deep Training (Days 30+)
- **Goal**: HexaZero significantly outperforms AB2
- Increase replay buffer to 2M+
- Consider sphinx access if available
- Full 800-sim MCTS, extensive self-play

---

## 14. First Commands to Run

```bash
# 1. SSH to sc
ssh $USER@sc.stanford.edu

# 2. Set up project directory
mkdir -p /nlp/scr/$USER/hexazero/{code,checkpoints,replay_buffer,elo_history,logs}

# 3. Clone repo
cd /nlp/scr/$USER/hexazero/code
git clone <repo> catanatron && cd catanatron

# 4. Set up conda env
conda activate hexazero  # (or create it first)
pip install torch numpy

# 5. Build C library
python -m hexzero.bindings.build_lib

# 6. Smoke test (interactive session on a jag)
nlprun -q jag -g 1 -r 20G -c 8 -p low

# Inside the interactive session:
python -c "
from hexzero.game.interface import CatanGame
from hexzero.model.network import HexaZeroNet
from hexzero.config import get_default_config
import torch

game = CatanGame(seed=42)
game.reset()
cfg = get_default_config()
net = HexaZeroNet(cfg.network).cuda().eval()
print(f'Model: {net.num_parameters:,} params on {next(net.parameters()).device}')

encoder = game.make_state_encoder()
sv = game.get_state_view()
enc = encoder.encode(sv)
batch = {k: v.unsqueeze(0).cuda() for k, v in enc.items()}
out = net.predict(batch)
print(f'Policy shape: {out[\"policy_probs\"].shape}, Value: {out[\"value\"]}')
print('Jag GPU smoke test passed!')
"
```
