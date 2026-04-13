#!/bin/bash
# ============================================================================
# Human Bot: Full Pipeline on Stanford NLP Cluster (1 GPU + 40 CPU)
#
# 1. Generate 60k AB2 games + convert human games (parallel, all CPUs)
# 2. Train: AB2 pretrain -> Human finetune (GPU 0)
# 3. Benchmark: 0/1/2-ply vs AB2
#
# All data on /scr-ssd (local fast disk), cleaned up on exit.
#
# Usage from sc:
#   nlprun -q jag -g 1 -r 60G -c 40 -p standard 'bash human_bot/cluster_train.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

LOCAL_SCRATCH="${PROJECT_DIR}/tmp_train_$$"
mkdir -p "${LOCAL_SCRATCH}"
# No auto-cleanup: data is reusable across restarts
# Clean up manually with: rm -rf /nlp/scr/nroll/catan_training/tmp_train_*

CKPT_DIR="${LOCAL_SCRATCH}/checkpoints"
AB2_DIR="${PROJECT_DIR}/data/ab2_games"
HUMAN_DIR="${LOCAL_SCRATCH}/human_data"
COLONIST_RAW="${PROJECT_DIR}/data/colonist_raw/games"
FINAL_CKPT_DIR="${PROJECT_DIR}/checkpoints/cluster_run"

mkdir -p "${CKPT_DIR}" "${AB2_DIR}" "${HUMAN_DIR}" "${FINAL_CKPT_DIR}"

NUM_CPUS=$(nproc 2>/dev/null || echo 8)

# Pre-build the C library before any workers spawn
echo "Building libcatan.so..."
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1
echo "  Done."

echo "============================================"
echo "  Human Bot: Full Cluster Pipeline"
echo "  Project:       ${PROJECT_DIR}"
echo "  Local scratch: ${LOCAL_SCRATCH}"
echo "  CPUs: ${NUM_CPUS}"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "============================================"

# ── Step 1a: Generate AB2 games (resumes from existing shards) ──
AB2_TARGET=60000
WORKERS=$(( NUM_CPUS > 36 ? 36 : NUM_CPUS ))
echo ""
echo "Step 1a: AB2 games (target ${AB2_TARGET}, resumes if data exists)..."
python3 -u -m hexzero.scripts.collect_ab2_games \
    --output-dir "${AB2_DIR}" \
    --num-games "${AB2_TARGET}" \
    --num-workers "${WORKERS}" \
    --games-per-file 25 \
    --depth 2
AB2_COUNT=$(ls "${AB2_DIR}"/*.pt 2>/dev/null | grep -v metadata | wc -l)
echo "  AB2 data ready: ${AB2_COUNT} shards in ${AB2_DIR}"

# ── Step 1b: Convert human games ─────────────────────────────
HUMAN_DIR="${PROJECT_DIR}/data/human_games_fixed"
HUMAN_SHARD_COUNT=$(ls "${HUMAN_DIR}"/*.pt 2>/dev/null | wc -l)
MIN_HUMAN_SHARDS="${MIN_HUMAN_SHARDS:-100}"

if [ "${HUMAN_SHARD_COUNT}" -ge "${MIN_HUMAN_SHARDS}" ]; then
    echo ""
    echo "Step 1b: Using pre-converted human data: ${HUMAN_DIR} (${HUMAN_SHARD_COUNT} shards)"
elif [ -d "${COLONIST_RAW}" ]; then
    echo ""
    echo "Step 1b: Wiping stale data and reconverting..."
    rm -f "${HUMAN_DIR}"/*.pt
    mkdir -p "${HUMAN_DIR}"
    WORKERS=$(( NUM_CPUS > 36 ? 36 : NUM_CPUS ))
    COLONIST_COUNT=$(find "${COLONIST_RAW}" -name "*.json" -type f | wc -l)
    echo "  Converting ${COLONIST_COUNT} human games (${WORKERS} workers)..."
    python3 -u -m human_bot.colonist_converter \
        --input-dir "${COLONIST_RAW}" \
        --output-dir "${HUMAN_DIR}" \
        --num-workers "${WORKERS}" \
        --games-per-shard 200
else
    echo "  ERROR: No human data available."
    exit 1
fi
HUMAN_COUNT=$(ls "${HUMAN_DIR}"/*.pt 2>/dev/null | wc -l)
echo "  Human data ready: ${HUMAN_COUNT} shards in ${HUMAN_DIR}"

# ── Step 2+3: Train + Benchmark ──────────────────────────────
echo ""
echo "Step 2: Training + Benchmark..."
python3 -u human_bot/cluster_train_inner.py \
    --ab2-dir "${AB2_DIR}" \
    --human-dir "${HUMAN_DIR}" \
    --ckpt-dir "${CKPT_DIR}" \
    --batch-size 8192 \
    --shards-per-group 20 \
    --eval-games 100

# ── Copy checkpoints to persistent storage ────────────────────
echo ""
echo "Copying checkpoints to ${FINAL_CKPT_DIR}/"
cp -v "${CKPT_DIR}"/*.pt "${FINAL_CKPT_DIR}/" 2>/dev/null || true

echo ""
echo "Pipeline complete."
echo "  Checkpoints: ${FINAL_CKPT_DIR}/"
