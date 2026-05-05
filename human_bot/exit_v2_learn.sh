#!/usr/bin/env bash
# GPU learner for the ROBUST improvement run (exit_v2).
#
# Key differences from exit_gpu_learn.sh:
#   - Consumes shards with source-aware mixing (60% exit_vs_ab2,
#     25% exit, 15% ab2) instead of FIFO-by-filename.
#   - Evals are AB2-leaf search (reliable, not NN-value dependent).
#   - Resumes round/total_examples/best_wr from the seed checkpoint.
#   - Logs train/lr and per-source shard counts.
#
# Reads from environment:
#   CKPT       : starting checkpoint (default checkpoints/ab2_imit_v1/latest.pt)
#   SHARD_DIR  : (default data/exit_v2)
#   CKPT_DIR   : (default checkpoints/exit_v2)
#   WANDB_NAME : (default exit_v2)
#   SOURCE_MIX : comma-separated source weights
#                 (default exit_vs_ab2:0.6,exit:0.25,ab2:0.15)
#   EVAL_GAMES : per-eval (default 200; AB-leaf eval halves this internally)
#
# Launch via:
#   nlprun -q jag -g 1 -c 8 -r 80G -p standard -m jagupard28 \
#       -n exit-v2-learn bash human_bot/exit_v2_learn.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# PyTorch / MKL stability on jag nodes: avoid AVX-mismatch SIGILL crashes
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# W&B — load from environment or ~/.wandb_key (never hardcode keys)
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
fi

echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

: "${CKPT:=checkpoints/ab2_imit_v1/latest.pt}"
: "${SHARD_DIR:=data/exit_v2}"
: "${CKPT_DIR:=checkpoints/exit_v2}"
: "${WANDB_NAME:=exit_v2}"
: "${SOURCE_MIX:=exit_vs_ab2:0.6,exit:0.25,ab2:0.15}"
: "${EVAL_GAMES:=200}"

mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"
if [ ! -f "$CKPT_DIR/init.pt" ]; then
    cp "$CKPT" "$CKPT_DIR/init.pt"
    echo "[launcher] Seeded $CKPT_DIR/init.pt from $CKPT"
fi
if [ ! -f "$CKPT_DIR/latest.pt" ]; then
    cp "$CKPT" "$CKPT_DIR/latest.pt"
    echo "[launcher] Seeded $CKPT_DIR/latest.pt from $CKPT"
fi

# Run the pre-flight gates. Bail loudly if any gate fails.
echo "[launcher] Running pre-flight gates..."
python3 -m human_bot.run_preflight \
    --ckpt "$CKPT_DIR/init.pt" \
    --eval-games 60

python3 -u human_bot/c_selfplay.py \
    --role learner \
    --checkpoint "$CKPT_DIR/latest.pt" \
    --shard-dir "$SHARD_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --player-counts 2,3,4 \
    --batch-size 8192 \
    --shards-per-train 20 \
    --eval-games "$EVAL_GAMES" \
    --eval-interval 4 \
    --source-mix "$SOURCE_MIX" \
    --wandb-name "$WANDB_NAME"
