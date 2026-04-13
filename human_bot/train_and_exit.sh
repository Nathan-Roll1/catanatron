#!/bin/bash
# ============================================================================
# Train on existing AB2 (depth-2) data + human data, then self-play ExIt.
# Assumes data already exists in data/ab2_v2 and data/human_v2_fixed.
#
#   nlprun -q jag -g 4 -r 120G -c 48 -p standard -n train-exit \
#     'cd /nlp/scr/nroll/catan_training_big && bash human_bot/train_and_exit.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ~1M param model
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

DATA_ROOT="/nlp/scr/nroll/catan_training"
AB2_DIR="${DATA_ROOT}/data/ab2_v2"
HUMAN_DIR="${DATA_ROOT}/data/human_v2_fixed"
CKPT_DIR="${PROJECT_DIR}/checkpoints/pretrain_ab2d2"
SELFPLAY_SHARD_DIR="${PROJECT_DIR}/data/selfplay_ab2d2"
SELFPLAY_CKPT_DIR="${PROJECT_DIR}/checkpoints/selfplay_ab2d2"

NUM_CPUS=$(nproc 2>/dev/null || echo 8)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

mkdir -p "${CKPT_DIR}" \
         "${SELFPLAY_SHARD_DIR}/pending" "${SELFPLAY_SHARD_DIR}/consumed" \
         "${SELFPLAY_CKPT_DIR}"

AB2_COUNT=$(find "${AB2_DIR}" -name '*.pt' ! -name 'metadata.pt' 2>/dev/null | wc -l | tr -d ' ')
HUMAN_COUNT=$(find "${HUMAN_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')

echo "============================================================"
echo "  AB2(d2) Pretrain + Human Finetune → Self-play ExIt"
echo "  Node: $(hostname)"
echo "  CPUs: ${NUM_CPUS}  GPUs: ${NUM_GPUS}"
echo "  AB2 shards:   ${AB2_COUNT} (${AB2_DIR})"
echo "  Human shards: ${HUMAN_COUNT} (${HUMAN_DIR})"
echo "============================================================"

if [ "${AB2_COUNT}" -lt 100 ]; then
    echo "ERROR: Not enough AB2 data (${AB2_COUNT} shards). Run regen_ab2_data.sh first."
    exit 1
fi

# ── Build C library ──────────────────────────────────────────────
echo ""
echo "Building C library..."
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1
echo "  Done."

# ── Phase 1: Pretrain ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 1: AB2 pretrain + Human finetune"
echo "============================================================"

TRAIN_ARGS="--ab2-dir ${AB2_DIR} --ckpt-dir ${CKPT_DIR} --batch-size 8192 --shards-per-group 20 --eval-games 50"
if [ "${HUMAN_COUNT}" -gt 5 ]; then
    TRAIN_ARGS="${TRAIN_ARGS} --human-dir ${HUMAN_DIR}"
    echo "  Including human data (${HUMAN_COUNT} shards)"
else
    echo "  No human data available, AB2-only pretrain"
fi

python3 -u human_bot/cluster_train_inner.py ${TRAIN_ARGS}

PRETRAINED="${CKPT_DIR}/final.pt"
if [ ! -f "${PRETRAINED}" ]; then
    echo "ERROR: Pretrained checkpoint not found at ${PRETRAINED}"
    exit 1
fi
echo "  Pretrained: ${PRETRAINED}"

# ── Phase 2: Self-play ExIt ──────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 2: Self-play Expert Iteration"
echo "============================================================"

NUM_ACTOR_GPUS=$(( NUM_GPUS > 1 ? NUM_GPUS - 1 : 1 ))
ACTORS_PER_GPU=$(( (NUM_CPUS - 1) / NUM_ACTOR_GPUS ))
ACTORS_PER_GPU=$(( ACTORS_PER_GPU > 1 ? ACTORS_PER_GPU : 1 ))
TOTAL_ACTORS=$(( ACTORS_PER_GPU * NUM_ACTOR_GPUS ))

echo "  Checkpoint:  ${PRETRAINED}"
echo "  GPUs:        ${NUM_GPUS} (1 learner + ${NUM_ACTOR_GPUS} actor)"
echo "  Actors:      ${TOTAL_ACTORS} (${ACTORS_PER_GPU}/gpu)"
echo "  Search:      1-ply + 5-ply deep on important positions"

python3 -u human_bot/selfplay.py \
    --checkpoint "${PRETRAINED}" \
    --role all \
    --num-actor-gpus "${NUM_ACTOR_GPUS}" \
    --actors-per-gpu "${ACTORS_PER_GPU}" \
    --shard-dir "${SELFPLAY_SHARD_DIR}" \
    --ckpt-dir "${SELFPLAY_CKPT_DIR}" \
    --batch-size 8192 \
    --shards-per-train 20 \
    --search-depth 1 \
    --deep-search-depth 5 \
    --eval-games 50 \
    --eval-interval 4 \
    --reload-interval 100 \
    --wandb-name "exit-ab2d2-$(date +%m%d-%H%M)"
