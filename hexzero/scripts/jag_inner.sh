#!/bin/bash
# =============================================================================
# Inner script: runs INSIDE the allocated jag GPU node.
# Called by run_jag.sh via nlprun, or run manually inside an interactive session.
#
# Does: conda setup → build C lib → self-play → train w/ W&B → evaluate
# =============================================================================
set -euo pipefail

USER_ID="nroll"
PROJECT="/nlp/scr/${USER_ID}/catanatron"
HEXZERO="${PROJECT}/hexzero"
WANDB_KEY="wandb_v1_5Wm7tx6uj1GvNyXjt5ogWR8WJyO"

# Directories
CKPT_DIR="${PROJECT}/checkpoints"
BUFFER_DIR="${PROJECT}/replay_buffer"
SP_DIR="${PROJECT}/selfplay_data"
ELO_DIR="${PROJECT}/elo_history"
LOG_DIR="${PROJECT}/logs"

mkdir -p "$CKPT_DIR" "$BUFFER_DIR" "$SP_DIR" "$ELO_DIR" "$LOG_DIR"

echo "============================================"
echo " HexaZero Jag Inner Pipeline"
echo " Host: $(hostname)"
echo " Project: ${PROJECT}"
echo "============================================"

# ── GPU info ─────────────────────────────────────────────────────────
echo ""
echo "[GPU] nvidia-smi:"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""

# ── Conda ────────────────────────────────────────────────────────────
echo "[1/6] Setting up conda environment..."

# Source conda
CONDA_BASE="/nlp/scr/${USER_ID}/miniconda3"
if [ ! -d "$CONDA_BASE" ]; then
    echo "  Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_BASE"
    rm /tmp/miniconda.sh
fi

eval "$($CONDA_BASE/bin/conda shell.bash hook)"

if ! conda env list | grep -q "hexazero"; then
    echo "  Creating hexazero conda env..."
    conda create -n hexazero python=3.11 -y -q
fi

conda activate hexazero

# Install deps if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "  Installing PyTorch + deps..."
    # Detect CUDA version on this machine
    CUDA_VER=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 | grep -oP '\d+\.\d+' || echo "12.1")
    CUDA_SHORT=$(echo "$CUDA_VER" | tr -d '.')
    pip install -q torch numpy wandb --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" 2>/dev/null \
        || pip install -q torch numpy wandb
fi

# Ensure wandb is installed
pip install -q wandb 2>/dev/null || true

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "  VRAM: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")')"
fi

# ── Build C library ──────────────────────────────────────────────────
echo ""
echo "[2/6] Building C shared library..."
cd "$PROJECT"

if [ ! -f "${HEXZERO}/bindings/lib/libcatan.so" ] && [ ! -f "${HEXZERO}/bindings/lib/libcatan.dylib" ]; then
    python -m hexzero.bindings.build_lib
    echo "  Built successfully."
else
    echo "  Already built, skipping."
fi

# Quick smoke test
python -c "
from hexzero.game.interface import CatanGame
g = CatanGame(seed=1); g.reset()
print(f'  Smoke test: {g}')
" || { echo "FATAL: smoke test failed"; exit 1; }

# ── W&B login ────────────────────────────────────────────────────────
echo ""
echo "[3/6] Logging into Weights & Biases..."
export WANDB_API_KEY="$WANDB_KEY"
python -c "import wandb; wandb.login(key='${WANDB_KEY}', relogin=True)" 2>/dev/null
echo "  W&B authenticated."

# ── Self-play data generation ────────────────────────────────────────
echo ""
echo "[4/6] Generating self-play data..."

ITERATION=0
NUM_GAMES=50
MCTS_SIMS=50  # start low for speed; increase later

# Check if we already have enough data
BUFFER_FILE="${BUFFER_DIR}/buffer.pt"
NEED_SELFPLAY=true

if [ -f "$BUFFER_FILE" ]; then
    BUF_SIZE=$(python -c "
import torch
d = torch.load('${BUFFER_FILE}', weights_only=False, map_location='cpu')
print(d.get('size', 0))
" 2>/dev/null || echo "0")
    echo "  Existing buffer: ${BUF_SIZE} positions"
    if [ "$BUF_SIZE" -ge 2048 ]; then
        NEED_SELFPLAY=false
        echo "  Enough data for training, skipping self-play."
    fi
fi

if $NEED_SELFPLAY; then
    echo "  Playing ${NUM_GAMES} games with ${MCTS_SIMS} MCTS sims..."
    echo "  (this generates training data from scratch with a random network)"

    python -m hexzero.scripts.selfplay_worker \
        --output-dir "$SP_DIR" \
        --games "$NUM_GAMES" \
        --mcts-sims "$MCTS_SIMS" \
        --device cuda \
        --seed 42

    echo "  Merging into replay buffer..."
    python -m hexzero.scripts.merge_buffer \
        --source "$SP_DIR" \
        --target "$BUFFER_FILE"
fi

# ── Training with W&B ────────────────────────────────────────────────
echo ""
echo "[5/6] Training with W&B logging..."

python -m hexzero.scripts.train_wandb \
    --replay-buffer "$BUFFER_FILE" \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs 20 \
    --batch-size 1024 \
    --lr 0.002 \
    --device cuda \
    --iteration "$ITERATION" \
    --wandb-project hexazero \
    --wandb-name "iter${ITERATION}-$(hostname)"

# ── Evaluation ───────────────────────────────────────────────────────
echo ""
echo "[6/6] Evaluating against AB2 baseline..."

python -m hexzero.scripts.evaluate \
    --checkpoint "${CKPT_DIR}/latest.pt" \
    --num-games 20 \
    --mcts-sims 50 \
    --elo-file "${ELO_DIR}/ratings.json" \
    --device cuda

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Checkpoint: ${CKPT_DIR}/latest.pt"
echo " ELO file:   ${ELO_DIR}/ratings.json"
echo " W&B:        https://wandb.ai (project: hexazero)"
echo "============================================"
