#!/bin/bash
# ============================================================================
# Self-play improvement via Expert Iteration
#
# 4 GPUs: 1 learner (GPU 0) + 3 actors (GPUs 1-3)
# 48 CPUs: game simulation for actors
#
# Usage from sc:
#   nlprun -q jag -g 4 -r 120G -c 48 -p standard -n selfplay \
#     'cd /nlp/scr/nroll/catan_training && bash human_bot/selfplay_cluster.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

if [ -f "${PROJECT_DIR}/checkpoints/selfplay/latest.pt" ]; then
    CHECKPOINT="${PROJECT_DIR}/checkpoints/selfplay/latest.pt"
else
    CHECKPOINT="${1:-checkpoints/cluster_run/final.pt}"
fi
SHARD_DIR="${PROJECT_DIR}/data/selfplay"
CKPT_DIR="${PROJECT_DIR}/checkpoints/selfplay"

mkdir -p "${SHARD_DIR}/pending" "${SHARD_DIR}/consumed" "${CKPT_DIR}"

echo "Building libcatan.so..."
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1
echo "  Done."

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_ACTOR_GPUS=$(( NUM_GPUS > 1 ? NUM_GPUS - 1 : 1 ))
NUM_CPUS=$(nproc 2>/dev/null || echo 8)
ACTORS_PER_GPU=$(( (NUM_CPUS - 1) / NUM_ACTOR_GPUS ))
ACTORS_PER_GPU=$(( ACTORS_PER_GPU > 1 ? ACTORS_PER_GPU : 1 ))
TOTAL_ACTORS=$(( ACTORS_PER_GPU * NUM_ACTOR_GPUS ))

echo "============================================"
echo "  Self-Play: Expert Iteration"
echo "  Project:       ${PROJECT_DIR}"
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  Shard dir:     ${SHARD_DIR}"
echo "  Checkpoint dir: ${CKPT_DIR}"
echo "  GPUs:          ${NUM_GPUS} (1 learner + ${NUM_ACTOR_GPUS} actor GPUs)"
echo "  CPUs:          ${NUM_CPUS} (1 learner + ${TOTAL_ACTORS} actors)"
echo "  Actors/GPU:    ${ACTORS_PER_GPU}"
echo "============================================"

python3 -u human_bot/selfplay.py \
    --checkpoint "${CHECKPOINT}" \
    --role all \
    --num-actor-gpus "${NUM_ACTOR_GPUS}" \
    --actors-per-gpu "${ACTORS_PER_GPU}" \
    --shard-dir "${SHARD_DIR}" \
    --ckpt-dir "${CKPT_DIR}" \
    --batch-size 8192 \
    --shards-per-train 20 \
    --search-depth 1 \
    --deep-search-depth 5 \
    --eval-games 50 \
    --eval-interval 4 \
    --reload-interval 100
