#!/bin/bash
# ============================================================================
# Pipeline v3: AB2 pretrain → Self-play (large model, no human games)
#
# Submit with:
#   nlprun -q jag -g 4 -r 120G -c 48 -p standard -n sp-large-v3 \
#     'bash /nlp/scr/nroll/catan_training_big/human_bot/full_pipeline_v3.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="/nlp/scr/nroll/catan_training_big"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Model config
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# W&B
export WANDB_API_KEY="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"

# AB2 data from the original training dir
AB2_DIR="/nlp/scr/nroll/catan_training/data/ab2_v2"
PRETRAIN_CKPT_DIR="${PROJECT_DIR}/checkpoints/pretrain_v3"
SELFPLAY_SHARD_DIR="${PROJECT_DIR}/data/selfplay_v3"
SELFPLAY_CKPT_DIR="${PROJECT_DIR}/checkpoints/selfplay_v3"

NUM_CPUS=$(nproc 2>/dev/null || echo 8)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

mkdir -p "${PRETRAIN_CKPT_DIR}" \
         "${SELFPLAY_SHARD_DIR}/pending" "${SELFPLAY_SHARD_DIR}/consumed" \
         "${SELFPLAY_CKPT_DIR}"

echo "============================================================"
echo "  Pipeline v3: ~1M param model, AB2 pretrain → Self-play"
echo "  Node: $(hostname)"
echo "  Model: gnn_hidden=${GNN_HIDDEN}, trunk_channels=${TRUNK_CHANNELS}"
echo "  AB2 data: ${AB2_DIR}"
echo "  CPUs: ${NUM_CPUS}  GPUs: ${NUM_GPUS}"
echo "============================================================"

# ── Build C library ──────────────────────────────────────────────
echo ""
echo "Building libcatan.so..."
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1
echo "  Done."

# ── Verify AB2 data ──────────────────────────────────────────────
AB2_COUNT=$(find "${AB2_DIR}" -name '*.pt' ! -name 'metadata.pt' | wc -l)
if [ "${AB2_COUNT}" -lt 100 ]; then
    echo "ERROR: AB2 data not found or too few shards (${AB2_COUNT}) in ${AB2_DIR}"
    exit 1
fi
echo ""
echo "Phase 1: AB2 data verified: ${AB2_COUNT} shards"

# ── Phase 1: AB2 pretrain ────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 1: AB2 pretrain → final.pt"
echo "============================================================"

python3 -u human_bot/cluster_train_inner.py \
    --ab2-dir "${AB2_DIR}" \
    --ckpt-dir "${PRETRAIN_CKPT_DIR}" \
    --batch-size 8192 \
    --shards-per-group 20 \
    --eval-games 50

PRETRAINED="${PRETRAIN_CKPT_DIR}/final.pt"
if [ ! -f "${PRETRAINED}" ]; then
    echo "ERROR: Pretrained checkpoint not found at ${PRETRAINED}"
    exit 1
fi
echo "  Pretrained model: ${PRETRAINED}"

# ── Phase 2: Self-play ExIt ──────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 2: Self-play Expert Iteration"
echo "============================================================"

NUM_ACTOR_GPUS=$(( NUM_GPUS > 1 ? NUM_GPUS - 1 : 1 ))
ACTORS_PER_GPU=$(( (NUM_CPUS - 1) / NUM_ACTOR_GPUS ))
ACTORS_PER_GPU=$(( ACTORS_PER_GPU > 1 ? ACTORS_PER_GPU : 1 ))
TOTAL_ACTORS=$(( ACTORS_PER_GPU * NUM_ACTOR_GPUS ))

echo "  Checkpoint:    ${PRETRAINED}"
echo "  GPUs:          ${NUM_GPUS} (1 learner + ${NUM_ACTOR_GPUS} actor GPUs)"
echo "  Actors:        ${TOTAL_ACTORS} (${ACTORS_PER_GPU}/gpu)"
echo "  Search:        5-ply tapering"
echo "  Temperature:   1.0 → 0.2 over 200 rounds"

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
    --wandb-name "sp-large-$(date +%m%d-%H%M)"
