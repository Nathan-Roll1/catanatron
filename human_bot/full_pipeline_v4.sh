#!/bin/bash
# ============================================================================
# Full Pipeline v4: Generate data → Pretrain → Self-play ExIt
#
# Phase 1: Generate 100k AB2 games (2-ply search, all CPUs)
# Phase 2: Convert ~44k human games from Colonist JSON (all CPUs)
# Phase 3: Pretrain on AB2 + finetune on human → final.pt (GPU 0)
# Phase 4: Self-play ExIt with 5-ply tapering search (all GPUs)
#
# Data is generated fresh (old data cleared).
# Code runs from catan_training_big; data reads/writes to catan_training.
#
# Submit from sc:
#   nlprun -q jag -g 4 -r 120G -c 48 -p standard -n pipeline-v4 \
#     'cd /nlp/scr/nroll/catan_training_big && bash human_bot/full_pipeline_v4.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# ~1M param model
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# W&B — load from environment or ~/.wandb_key
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
fi

# Data lives under catan_training; code lives under catan_training_big
DATA_ROOT="/nlp/scr/nroll/catan_training"
COLONIST_RAW="${DATA_ROOT}/data/colonist_raw/games"
AB2_DIR="${DATA_ROOT}/data/ab2_v4"
HUMAN_DIR="${DATA_ROOT}/data/human_v4"
CKPT_DIR="${PROJECT_DIR}/checkpoints/pretrain_v4"
SELFPLAY_SHARD_DIR="${PROJECT_DIR}/data/selfplay_v4"
SELFPLAY_CKPT_DIR="${PROJECT_DIR}/checkpoints/selfplay_v4"

NUM_CPUS=$(nproc 2>/dev/null || echo 8)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
WORKERS=$(( NUM_CPUS > 36 ? 36 : NUM_CPUS ))

T_START=$(date +%s)

echo "============================================================"
echo "  Full Pipeline v4"
echo "  Node:    $(hostname)"
echo "  CPUs:    ${NUM_CPUS}  Workers: ${WORKERS}  GPUs: ${NUM_GPUS}"
echo "  Model:   GNN_HIDDEN=${GNN_HIDDEN} TRUNK_CHANNELS=${TRUNK_CHANNELS}"
echo "  AB2:     ${AB2_DIR}"
echo "  Human:   ${HUMAN_DIR}"
echo "  Colonist: ${COLONIST_RAW}"
echo "  Ckpts:   ${CKPT_DIR}"
echo "  Selfplay: ${SELFPLAY_CKPT_DIR}"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Setup directories ────────────────────────────────────────────
mkdir -p "${AB2_DIR}" "${HUMAN_DIR}" "${CKPT_DIR}" \
         "${SELFPLAY_SHARD_DIR}/pending" "${SELFPLAY_SHARD_DIR}/consumed" \
         "${SELFPLAY_CKPT_DIR}"

# ── Build C library ──────────────────────────────────────────────
echo "[setup] Building C library..."
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1
echo "[setup] Done."
echo ""

# ── Phase 1: Generate AB2 games (2-ply search) ──────────────────
echo "============================================================"
echo "  Phase 1: Generate 100k AB2 games (depth-2)"
echo "============================================================"

# Clear old AB2 data
OLD_AB2=$(find "${AB2_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
if [ "${OLD_AB2}" -gt 0 ]; then
    echo "[phase1] Clearing ${OLD_AB2} old AB2 shards..."
    find "${AB2_DIR}" -name '*.pt' -delete
fi

python3 -u -m hexzero.scripts.collect_ab2_games \
    --output-dir "${AB2_DIR}" \
    --num-games 100000 \
    --num-workers "${WORKERS}" \
    --games-per-file 25 \
    --depth 2

AB2_COUNT=$(find "${AB2_DIR}" -name '*.pt' ! -name 'metadata.pt' | wc -l | tr -d ' ')
echo "[phase1] AB2 shards: ${AB2_COUNT}"

if [ "${AB2_COUNT}" -lt 100 ]; then
    echo "ERROR: AB2 generation failed (only ${AB2_COUNT} shards)"
    exit 1
fi
echo ""

# ── Phase 2: Convert human games ────────────────────────────────
echo "============================================================"
echo "  Phase 2: Convert human games from Colonist JSON"
echo "============================================================"

if [ ! -d "${COLONIST_RAW}" ]; then
    echo "[phase2] WARNING: Colonist JSON dir not found at ${COLONIST_RAW}"
    echo "[phase2] Skipping human game conversion."
    HUMAN_COUNT=0
else
    COLONIST_COUNT=$(find "${COLONIST_RAW}" -name "*.json" -type f | wc -l | tr -d ' ')
    echo "[phase2] Colonist JSON files: ${COLONIST_COUNT}"

    # Clear old human data
    OLD_HUMAN=$(find "${HUMAN_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
    if [ "${OLD_HUMAN}" -gt 0 ]; then
        echo "[phase2] Clearing ${OLD_HUMAN} old human shards..."
        find "${HUMAN_DIR}" -name '*.pt' -delete
    fi

    if [ "${COLONIST_COUNT}" -gt 0 ]; then
        python3 -u -m human_bot.colonist_converter \
            --input-dir "${COLONIST_RAW}" \
            --output-dir "${HUMAN_DIR}" \
            --num-workers "${WORKERS}" \
            --games-per-shard 200
    fi

    HUMAN_COUNT=$(find "${HUMAN_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
    echo "[phase2] Human shards: ${HUMAN_COUNT}"
fi
echo ""

# ── Phase 3: Pretrain ────────────────────────────────────────────
echo "============================================================"
echo "  Phase 3: AB2 pretrain + Human finetune"
echo "============================================================"

TRAIN_CMD="python3 -u human_bot/cluster_train_inner.py \
    --ab2-dir ${AB2_DIR} \
    --ckpt-dir ${CKPT_DIR} \
    --batch-size 16384 \
    --shards-per-group 80 \
    --eval-games 50"

if [ "${HUMAN_COUNT}" -gt 5 ]; then
    TRAIN_CMD="${TRAIN_CMD} --human-dir ${HUMAN_DIR}"
    echo "[phase3] AB2 pretrain + Human finetune (${HUMAN_COUNT} human shards)"
else
    echo "[phase3] AB2-only pretrain (no human data)"
fi

eval ${TRAIN_CMD}

PRETRAINED="${CKPT_DIR}/final.pt"
if [ ! -f "${PRETRAINED}" ]; then
    echo "ERROR: Pretrained checkpoint not found at ${PRETRAINED}"
    exit 1
fi
echo "[phase3] Pretrained: ${PRETRAINED}"
echo ""

# ── Phase 4: Self-play ExIt ──────────────────────────────────────
echo "============================================================"
echo "  Phase 4: Self-play Expert Iteration"
echo "============================================================"

NUM_ACTOR_GPUS=$(( NUM_GPUS > 1 ? NUM_GPUS - 1 : 1 ))
ACTORS_PER_GPU=$(( (NUM_CPUS - 1) / NUM_ACTOR_GPUS ))
ACTORS_PER_GPU=$(( ACTORS_PER_GPU > 1 ? ACTORS_PER_GPU : 1 ))
TOTAL_ACTORS=$(( ACTORS_PER_GPU * NUM_ACTOR_GPUS ))

echo "[phase4] Checkpoint:  ${PRETRAINED}"
echo "[phase4] GPUs:        ${NUM_GPUS} (1 learner + ${NUM_ACTOR_GPUS} actor)"
echo "[phase4] Actors:      ${TOTAL_ACTORS} (${ACTORS_PER_GPU}/gpu)"
echo "[phase4] Search:      1-ply + 5-ply deep on important positions"

python3 -u human_bot/selfplay.py \
    --checkpoint "${PRETRAINED}" \
    --role all \
    --num-actor-gpus "${NUM_ACTOR_GPUS}" \
    --actors-per-gpu "${ACTORS_PER_GPU}" \
    --shard-dir "${SELFPLAY_SHARD_DIR}" \
    --ckpt-dir "${SELFPLAY_CKPT_DIR}" \
    --batch-size 16384 \
    --shards-per-train 40 \
    --search-depth 1 \
    --deep-search-depth 5 \
    --eval-games 50 \
    --eval-interval 4 \
    --reload-interval 100 \
    --wandb-name "v4-1M-$(date +%m%d-%H%M)"

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "  Pipeline v4 complete in $(( (T_END - T_START) / 60 )) minutes"
echo "  Pretrained:  ${PRETRAINED}"
echo "  Self-play:   ${SELFPLAY_CKPT_DIR}/latest.pt"
echo "  Ended: $(date)"
echo "============================================================"
