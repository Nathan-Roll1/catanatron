#!/usr/bin/env bash
# GPU actors for ExIt self-play (NN policy + ABt30 k=2 + AB-value leaf).
#
# Reads from environment:
#   NUM_ACTORS    : actor processes per node (default 4)
#   NUM_GPUS      : GPUs available on this node (default 1)
#   ACTOR_OFFSET  : first actor id (set distinct per node)
#   SEARCH_DEPTH  : argmax-rollout depth (default 30)
#   TOP_K         : policy top-k pruning (default 2)
#   MAX_PENDING   : pending shard cap (default 200)
#   PLAYER_COUNTS : comma-list (default 2,3,4)
#   CKPT          : path to PyTorch checkpoint
#                   (default checkpoints/ab2_imit_v1/latest.pt)
#   SHARD_DIR     : (default data/exit_gpu_v1)
#   CKPT_DIR      : (default checkpoints/exit_gpu_v1)
#
# Launch via:
#   ACTOR_OFFSET=0 NUM_ACTORS=4 NUM_GPUS=1 \
#     nlprun -q jag -g 1 -c 16 -r 80G -p standard -m jagupard28 \
#       -n exit-act-A bash human_bot/exit_gpu_actors.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

: "${NUM_ACTORS:=4}"
: "${NUM_GPUS:=1}"
: "${ACTOR_OFFSET:=0}"
: "${SEARCH_DEPTH:=30}"
: "${TOP_K:=2}"
: "${MAX_PENDING:=200}"
: "${PLAYER_COUNTS:=2,3,4}"
: "${CKPT:=checkpoints/ab2_imit_v1/latest.pt}"
: "${SHARD_DIR:=data/exit_gpu_v1}"
: "${CKPT_DIR:=checkpoints/exit_gpu_v1}"

echo "[launcher] num_actors=$NUM_ACTORS num_gpus=$NUM_GPUS offset=$ACTOR_OFFSET"
echo "[launcher] depth=$SEARCH_DEPTH top_k=$TOP_K max_pending=$MAX_PENDING"
echo "[launcher] ckpt=$CKPT"
echo "[launcher] shard_dir=$SHARD_DIR  ckpt_dir=$CKPT_DIR"

mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"

python3 -u human_bot/exit_gpu_actors.py \
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
