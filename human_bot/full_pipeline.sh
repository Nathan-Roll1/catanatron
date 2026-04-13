#!/bin/bash
# ============================================================================
# Full Pipeline: AB2 data → Human data → Pretrain → Self-play (4 GPUs + 48 CPUs)
#
# Phase 1: Generate 100k AB2 games (all CPUs)
# Phase 2: Convert 44k human games (all CPUs)
# Phase 3: Train: AB2 pretrain → Human finetune → finalv2.pt (GPU 0)
# Phase 4: Self-play ExIt with 5-ply tapering search (all 4 GPUs + all CPUs)
#
# Usage from sc:
#   nlprun -q jag -g 4 -r 120G -c 48 -p standard -n full_pipeline -m jagupard28 \
#     'export PROJECT_DIR=/nlp/scr/nroll/catan_training && cd $PROJECT_DIR && bash human_bot/full_pipeline.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

AB2_DIR="${PROJECT_DIR}/data/ab2_v2"
HUMAN_DIR="${PROJECT_DIR}/data/human_v2"
COLONIST_RAW="${PROJECT_DIR}/data/colonist_raw/games"
PRETRAIN_CKPT_DIR="${PROJECT_DIR}/checkpoints/pretrain_v2"
SELFPLAY_SHARD_DIR="${PROJECT_DIR}/data/selfplay_v2"
SELFPLAY_CKPT_DIR="${PROJECT_DIR}/checkpoints/selfplay_v2"

NUM_CPUS=$(nproc 2>/dev/null || echo 8)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
WORKERS=$(( NUM_CPUS > 36 ? 36 : NUM_CPUS ))

mkdir -p "${AB2_DIR}" "${HUMAN_DIR}" "${PRETRAIN_CKPT_DIR}" \
         "${SELFPLAY_SHARD_DIR}/pending" "${SELFPLAY_SHARD_DIR}/consumed" \
         "${SELFPLAY_CKPT_DIR}"

# ── Wipe old data from broken game engine ────────────────────────
echo "Cleaning old (broken engine) data..."
rm -rf "${PROJECT_DIR}/data/ab2_games" \
       "${PROJECT_DIR}/data/human_games_fixed" \
       "${PROJECT_DIR}/data/selfplay" \
       "${PROJECT_DIR}/checkpoints/cluster_run" \
       "${PROJECT_DIR}/checkpoints/selfplay" \
       "${PROJECT_DIR}/tmp_train_"*
echo "  Done."

# ── Rebuild C library (fixed engine) ─────────────────────────────
echo "Building libcatan.so (fixed engine)..."
rm -f "${PROJECT_DIR}/hexzero/bindings/lib/libcatan.so" \
      "${PROJECT_DIR}/hexzero/bindings/lib/libcatan.dylib"
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1
echo "  Done."

echo ""
echo "============================================================"
echo "  Full Pipeline v2 (fixed game engine — no negative resources)"
echo "  Project:       ${PROJECT_DIR}"
echo "  CPUs:          ${NUM_CPUS}"
echo "  GPUs:          ${NUM_GPUS}"
echo "============================================================"

T_START=$(date +%s)

# ── Phase 1: Generate 100k AB2 games ─────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 1: Generate 100k AB2 games"
echo "============================================================"
python3 -u -m hexzero.scripts.collect_ab2_games \
    --output-dir "${AB2_DIR}" \
    --num-games 100000 \
    --num-workers "${WORKERS}" \
    --games-per-file 25 \
    --depth 2
AB2_COUNT=$(ls "${AB2_DIR}"/*.pt 2>/dev/null | grep -v metadata | wc -l)
echo "  AB2 data: ${AB2_COUNT} shards"

# ── Phase 2: Convert 44k human games ────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 2: Convert human games (fresh with fixed engine)"
echo "============================================================"
rm -f "${HUMAN_DIR}"/*.pt
COLONIST_COUNT=$(find "${COLONIST_RAW}" -name "*.json" -type f | wc -l)
echo "  Converting ${COLONIST_COUNT} human games (${WORKERS} workers)..."
python3 -u -m human_bot.colonist_converter \
    --input-dir "${COLONIST_RAW}" \
    --output-dir "${HUMAN_DIR}" \
    --num-workers "${WORKERS}" \
    --games-per-shard 200
HUMAN_SHARD_COUNT=$(ls "${HUMAN_DIR}"/*.pt 2>/dev/null | wc -l)
echo "  Human data: ${HUMAN_SHARD_COUNT} shards"

# ── Phase 3: Pretrain (AB2 + Human → finalv2.pt) ────────────────
echo ""
echo "============================================================"
echo "  Phase 3: AB2 pretrain + Human finetune → finalv2.pt"
echo "============================================================"
python3 -u human_bot/cluster_train_inner.py \
    --ab2-dir "${AB2_DIR}" \
    --human-dir "${HUMAN_DIR}" \
    --ckpt-dir "${PRETRAIN_CKPT_DIR}" \
    --batch-size 8192 \
    --shards-per-group 20 \
    --eval-games 50

FINALV2="${PRETRAIN_CKPT_DIR}/final.pt"
echo "  Pretrained model: ${FINALV2}"

# ── Phase 4: Self-play ExIt ──────────────────────────────────────
echo ""
echo "============================================================"
echo "  Phase 4: Self-play Expert Iteration"
echo "============================================================"

NUM_ACTOR_GPUS=$(( NUM_GPUS > 1 ? NUM_GPUS - 1 : 1 ))
ACTORS_PER_GPU=$(( (NUM_CPUS - 1) / NUM_ACTOR_GPUS ))
ACTORS_PER_GPU=$(( ACTORS_PER_GPU > 1 ? ACTORS_PER_GPU : 1 ))
TOTAL_ACTORS=$(( ACTORS_PER_GPU * NUM_ACTOR_GPUS ))

echo "  Checkpoint:    ${FINALV2}"
echo "  GPUs:          ${NUM_GPUS} (1 learner + ${NUM_ACTOR_GPUS} actor GPUs)"
echo "  Actors:        ${TOTAL_ACTORS} (${ACTORS_PER_GPU}/gpu)"

python3 -u human_bot/selfplay.py \
    --checkpoint "${FINALV2}" \
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
    --reload-interval 100

T_END=$(date +%s)
echo ""
echo "============================================================"
echo "  Pipeline complete in $(( (T_END - T_START) / 60 )) minutes"
echo "  Pretrained:  ${FINALV2}"
echo "  Self-play:   ${SELFPLAY_CKPT_DIR}/best.pt"
echo "============================================================"
