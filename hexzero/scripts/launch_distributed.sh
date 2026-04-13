#!/bin/bash
# =============================================================================
# Launch distributed HexaZero training on the Stanford NLP jag cluster.
#
# Submits:
#   1 trainer job   (jag-important, 1 GPU) -- trains + evals + saves checkpoints
#   N worker jobs   (jag-lo, 1 GPU each)   -- MCTS self-play, write game files
#
# Usage from sc:
#   bash /nlp/scr/nroll/catanatron/hexzero/scripts/launch_distributed.sh
#   bash /nlp/scr/nroll/catanatron/hexzero/scripts/launch_distributed.sh --workers 6
# =============================================================================

set -euo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
GAMES_DIR="${PROJECT}/games"
CKPT_DIR="${PROJECT}/checkpoints"
CONDA_HOOK='eval "$(/nlp/scr/nroll/miniconda3/bin/conda shell.bash hook)" && conda activate hexazero'
WANDB_KEY="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"

NUM_WORKERS=${1:-4}
MCTS_SIMS=${2:-50}
GAMES_PER_WORKER=${3:-200}
CONCURRENT=${4:-8}

mkdir -p "$GAMES_DIR" "$CKPT_DIR"

echo "============================================"
echo " HexaZero Distributed Launch"
echo " Workers: ${NUM_WORKERS} (jag-standard)"
echo " Trainer: 1 (jag-standard)"
echo " MCTS sims: ${MCTS_SIMS}"
echo " Games/worker: ${GAMES_PER_WORKER}"
echo " Concurrent/worker: ${CONCURRENT}"
echo "============================================"
echo ""

# --- Submit trainer (non-preemptable) ---
echo "Submitting trainer..."
nlprun -q jag -g 1 -r 40G -c 8 -p standard -n hz-trainer \
    "${CONDA_HOOK} && cd ${PROJECT} && python -m hexzero.scripts.train_loop \
    --games-dir ${GAMES_DIR} \
    --checkpoint-dir ${CKPT_DIR} \
    --batch-size 2048 \
    --lr 0.001 \
    --epochs-per-cycle 5 \
    --min-new-games 10 \
    --eval-every 3 \
    --eval-games 24 \
    --max-cycles 50 \
    --poll-interval 15 \
    --wandb-key ${WANDB_KEY}"
echo "  Trainer submitted."
echo ""

# Give trainer a head start to create initial checkpoint
sleep 5

# --- Submit self-play workers (preemptable) ---
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    echo "Submitting worker ${i}..."
    nlprun -q jag -g 1 -r 30G -c 8 -p standard -n "hz-worker-${i}" \
        "${CONDA_HOOK} && cd ${PROJECT} && python -m hexzero.scripts.selfplay_loop \
        --games-dir ${GAMES_DIR} \
        --checkpoint-dir ${CKPT_DIR} \
        --worker-id ${i} \
        --concurrent ${CONCURRENT} \
        --mcts-sims ${MCTS_SIMS} \
        --total-games ${GAMES_PER_WORKER} \
        --reload-every 10"
    echo "  Worker ${i} submitted."
done

echo ""
echo "============================================"
echo " All jobs submitted!"
echo " Monitor: squeue -u nroll"
echo " Trainer: tail -f ~/hz-trainer.out"
echo " Worker:  tail -f ~/hz-worker-0.out"
echo " W&B:     https://wandb.ai (project: hexazero)"
echo "============================================"
