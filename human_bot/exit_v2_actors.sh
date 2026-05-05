#!/usr/bin/env bash
# Actor launcher for the robust improvement run (exit_v2).
#
# Routes each actor type to the right shard_dir (shared with the v2 learner)
# and reads the live checkpoint the v2 learner exports. Pick which actor
# type to run with ACTOR_TYPE={ab2,exit,exit_vs_ab2}.
#
# Reads from environment (all optional with sensible defaults):
#   ACTOR_TYPE    : ab2 | exit | exit_vs_ab2 (REQUIRED)
#   NUM_ACTORS    : actor processes (default 4)
#   NUM_GPUS      : GPUs on this node (default 1; ab2 ignores)
#   ACTOR_OFFSET  : first actor id (MUST be distinct per node)
#   SEARCH_DEPTH  : ExIt search depth (default 15; ab2 ignores)
#   TOP_K         : ExIt top-k (default 2; ab2 ignores)
#   AB_DEPTH      : AB2 search depth (default 2; non-AB2 ignores)
#   MAX_PENDING   : pending shard cap (default 200)
#   PLAYER_COUNTS : comma-list (default 2,3,4)
#   CKPT          : live checkpoint (default checkpoints/exit_v2/latest.pt)
#   SHARD_DIR     : output (default data/exit_v2)
#   CKPT_DIR      : ckpt dir (default checkpoints/exit_v2)
#
# Launch via:
#   ACTOR_TYPE=exit_vs_ab2 ACTOR_OFFSET=20 NUM_ACTORS=4 NUM_GPUS=1 \
#     nlprun -q jag -g 1 -c 16 -r 60G -p standard -m jagupard31 \
#       -n exit-v2-va2 bash human_bot/exit_v2_actors.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# PyTorch / MKL stability
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

: "${ACTOR_TYPE:?Must set ACTOR_TYPE=ab2|exit|exit_vs_ab2}"
: "${NUM_ACTORS:=4}"
: "${NUM_GPUS:=1}"
: "${ACTOR_OFFSET:=0}"
: "${SEARCH_DEPTH:=15}"
: "${TOP_K:=2}"
: "${AB_DEPTH:=2}"
: "${MAX_PENDING:=200}"
: "${PLAYER_COUNTS:=2,3,4}"
: "${CKPT:=checkpoints/exit_v2/latest.pt}"
: "${SHARD_DIR:=data/exit_v2}"
: "${CKPT_DIR:=checkpoints/exit_v2}"

mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"

echo "[launcher] ACTOR_TYPE=$ACTOR_TYPE NUM_ACTORS=$NUM_ACTORS OFFSET=$ACTOR_OFFSET"
echo "[launcher] CKPT=$CKPT SHARD_DIR=$SHARD_DIR"

case "$ACTOR_TYPE" in
  ab2)
    # Pure AB2-vs-AB2 imitation; no GPU/CKPT needed.
    python3 -u human_bot/ab2_stream.py \
        --shard-dir "$SHARD_DIR" \
        --num-workers "$NUM_ACTORS" \
        --depth "$AB_DEPTH" \
        --max-pending "$MAX_PENDING" \
        --player-counts "$PLAYER_COUNTS" \
        --actor-id-offset "$ACTOR_OFFSET"
    ;;
  exit)
    # NN-vs-NN ExIt self-play (all 4 seats use NN + search).
    echo "[launcher] Waiting up to 120s for $CKPT..."
    for _ in $(seq 1 120); do [ -f "$CKPT" ] && break; sleep 1; done
    if [ ! -f "$CKPT" ]; then
        echo "[launcher] ERROR: $CKPT not found" >&2; exit 1
    fi
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
    ;;
  exit_vs_ab2)
    # 1 NN seat (ExIt search) vs (N-1) AB2 seats.
    echo "[launcher] Waiting up to 120s for $CKPT..."
    for _ in $(seq 1 120); do [ -f "$CKPT" ] && break; sleep 1; done
    if [ ! -f "$CKPT" ]; then
        echo "[launcher] ERROR: $CKPT not found" >&2; exit 1
    fi
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
    ;;
  *)
    echo "[launcher] Unknown ACTOR_TYPE=$ACTOR_TYPE" >&2
    exit 1
    ;;
esac
