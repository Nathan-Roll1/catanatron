#!/usr/bin/env bash
# ExIt vs AB2 actors: 1 NN seat (search) + (N-1) AB2 seats per game.
#
# Reads from environment:
#   NUM_ACTORS    : actor processes (default 4)
#   NUM_GPUS      : GPUs available (default 1)
#   ACTOR_OFFSET  : first actor id (set distinct per node)
#   SEARCH_DEPTH  : NN argmax-rollout depth (default 15)
#   TOP_K         : NN policy top-k (default 2)
#   MAX_PENDING   : pending shard cap (default 200)
#   PLAYER_COUNTS : comma-list (default 2,3,4)
#   CKPT          : path to current learner checkpoint
#                   (default checkpoints/exit_gpu_v1/latest.pt)
#   SHARD_DIR     : (default data/exit_gpu_v1) — same as ExIt self-play so
#                   shards feed the same learner
#   CKPT_DIR      : (default checkpoints/exit_gpu_v1)
#
# Launch via:
#   ACTOR_OFFSET=20 NUM_ACTORS=4 NUM_GPUS=1 \
#     nlprun -q jag -g 1 -c 16 -r 60G -p standard -m jagupard31 \
#       -n exit-vs-ab2-A bash human_bot/exit_vs_ab2_actors.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# PyTorch / MKL stability on jag nodes
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

: "${NUM_ACTORS:=4}"
: "${NUM_GPUS:=1}"
: "${ACTOR_OFFSET:=0}"
: "${SEARCH_DEPTH:=15}"
: "${TOP_K:=2}"
: "${MAX_PENDING:=200}"
: "${PLAYER_COUNTS:=2,3,4}"
: "${CKPT:=checkpoints/exit_gpu_v1/latest.pt}"
: "${SHARD_DIR:=data/exit_gpu_v1}"
: "${CKPT_DIR:=checkpoints/exit_gpu_v1}"

echo "[launcher] num_actors=$NUM_ACTORS num_gpus=$NUM_GPUS offset=$ACTOR_OFFSET"
echo "[launcher] depth=$SEARCH_DEPTH top_k=$TOP_K max_pending=$MAX_PENDING"
echo "[launcher] ckpt=$CKPT (live from learner)"
echo "[launcher] shard_dir=$SHARD_DIR  ckpt_dir=$CKPT_DIR"

mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"

echo "[launcher] Waiting up to 120s for $CKPT to exist..."
for _ in $(seq 1 120); do
    [ -f "$CKPT" ] && break
    sleep 1
done
if [ ! -f "$CKPT" ]; then
    echo "[launcher] ERROR: $CKPT never appeared. Is the learner running?" >&2
    exit 1
fi
echo "[launcher] Found $CKPT, proceeding."

python3 -u human_bot/exit_vs_ab2_actors.py \
    --checkpoint "$CKPT" \
    --shard-dir "$SHARD_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --num-actors "$NUM_ACTORS" \
    --num-gpus "$NUM_GPUS" \
    --actor-id-offset "$ACTOR_OFFSET" \
    --search-depth "$SEARCH_DEPTH" \
    --top-k "$TOP_K" \
    --max-pending "$MAX_PENDING" \
    --player-counts "$PLAYER_COUNTS"
