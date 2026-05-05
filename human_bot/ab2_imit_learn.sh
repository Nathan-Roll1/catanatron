#!/usr/bin/env bash
# GPU learner for the AB2 imitation run (1M-param M2-arch).
# Launch via:
#   nlprun -q jag -g 1 -c 8 -r 80G -p standard -n ab2-imit-learn \
#       bash human_bot/ab2_imit_learn.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# W&B: same key that's hard-coded in human_bot/cluster_train_inner.py.
# W&B — load from environment or ~/.wandb_key (never hardcode keys)
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
fi

# Rebuild libcatan for this node's CPU (-march=native was built elsewhere
# → "Illegal instruction" on jag nodes with different SIMD support).
# flock serializes concurrent rebuilds across jobs sharing the repo.
echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

python3 -u human_bot/c_selfplay.py \
    --role learner \
    --checkpoint checkpoints/ab2_imit_v1/init.pt \
    --shard-dir data/ab2_imit_v1 \
    --ckpt-dir checkpoints/ab2_imit_v1 \
    --player-counts 2,3,4 \
    --batch-size 8192 \
    --shards-per-train 20 \
    --eval-games 50 \
    --eval-interval 4 \
    --wandb-name ab2_imit_v1
