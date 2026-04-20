#!/usr/bin/env bash
# GPU learner for the ExIt run (consumes shards from GPU actors).
#
# Reads from environment:
#   CKPT       : starting checkpoint (default checkpoints/ab2_imit_v1/latest.pt)
#   SHARD_DIR  : (default data/exit_gpu_v1)
#   CKPT_DIR   : (default checkpoints/exit_gpu_v1)
#   WANDB_NAME : (default exit_gpu_v1)
#
# Launch via:
#   nlprun -q jag -g 1 -c 8 -r 80G -p standard -m jagupard28 \
#       -n exit-learn bash human_bot/exit_gpu_learn.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# W&B (same key as ab2_imit_learn.sh)
: "${WANDB_API_KEY:=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}"
export WANDB_API_KEY

echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

: "${CKPT:=checkpoints/ab2_imit_v1/latest.pt}"
: "${SHARD_DIR:=data/exit_gpu_v1}"
: "${CKPT_DIR:=checkpoints/exit_gpu_v1}"
: "${WANDB_NAME:=exit_gpu_v1}"

mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"
# Seed initial checkpoint if not already present.
# Also seed latest.pt so actors can pick up the bootstrap weights on startup
# before the learner's first training round has produced a new checkpoint.
if [ ! -f "$CKPT_DIR/init.pt" ]; then
    cp "$CKPT" "$CKPT_DIR/init.pt"
    echo "[launcher] Seeded $CKPT_DIR/init.pt from $CKPT"
fi
if [ ! -f "$CKPT_DIR/latest.pt" ]; then
    cp "$CKPT" "$CKPT_DIR/latest.pt"
    echo "[launcher] Seeded $CKPT_DIR/latest.pt from $CKPT (actors will start from this)"
fi

python3 -u human_bot/c_selfplay.py \
    --role learner \
    --checkpoint "$CKPT_DIR/init.pt" \
    --shard-dir "$SHARD_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --player-counts 2,3,4 \
    --batch-size 8192 \
    --shards-per-train 20 \
    --eval-games 50 \
    --eval-interval 4 \
    --wandb-name "$WANDB_NAME"
