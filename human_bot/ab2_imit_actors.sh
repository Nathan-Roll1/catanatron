#!/usr/bin/env bash
# Streaming AB2 actor node for the AB2 imitation run.
#
# Picks up ACTOR_OFFSET and NUM_WORKERS from the environment so you can
# run multiple actor jobs without editing the script.
#
# Launch via:
#   ACTOR_OFFSET=0   nlprun -q jag -g 0 -c 40 -r 60G -p standard \
#       -n ab2-act-A bash human_bot/ab2_imit_actors.sh
#   ACTOR_OFFSET=100 nlprun -q jag -g 0 -c 40 -r 60G -p standard \
#       -n ab2-act-B bash human_bot/ab2_imit_actors.sh
set -euo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD:$PYTHONPATH

# Rebuild libcatan for this node's CPU (-march=native was built elsewhere
# → "Illegal instruction" on jag nodes with different SIMD support).
# flock serializes concurrent rebuilds across jobs sharing the repo.
echo "[launcher] Rebuilding libcatan on $(hostname)..."
flock /nlp/scr/nroll/catan_training_big/.libcatan.lock \
    python3 -m hexzero.bindings.build_lib

: "${NUM_WORKERS:=40}"
: "${ACTOR_OFFSET:=0}"
: "${MAX_PENDING:=200}"
: "${PLAYER_COUNTS:=2,3,4}"

echo "[ab2_imit_actors] num_workers=$NUM_WORKERS offset=$ACTOR_OFFSET " \
     "max_pending=$MAX_PENDING player_counts=$PLAYER_COUNTS"

python3 -u human_bot/ab2_stream.py \
    --shard-dir data/ab2_imit_v1 \
    --num-workers "$NUM_WORKERS" \
    --depth 2 \
    --max-pending "$MAX_PENDING" \
    --player-counts "$PLAYER_COUNTS" \
    --actor-id-offset "$ACTOR_OFFSET"
