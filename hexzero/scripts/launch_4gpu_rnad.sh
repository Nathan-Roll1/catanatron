#!/bin/bash
# =============================================================================
# 4-GPU Population-Based R-NaD Training
#
# Launches 4 independent R-NaD agents on separate jag GPUs with staggered
# seeds and periodic weight sharing via a coordinator.
#
# Usage from sc:
#   bash /nlp/scr/nroll/catanatron/hexzero/scripts/launch_4gpu_rnad.sh
# =============================================================================
set -euo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
CKPT_BASE="${PROJECT}/checkpoints"
CONDA_HOOK='eval "$(/nlp/scr/nroll/miniconda3/bin/conda shell.bash hook)" && conda activate hexazero'
WANDB_KEY="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"

NUM_AGENTS=4
CONCURRENT=8
OUTER_STEPS=50
INNER_STEPS=50
EVAL_EVERY=1
LR=0.001

echo "============================================"
echo " 4-GPU R-NaD Population Training"
echo " Agents: ${NUM_AGENTS}"
echo " Concurrent/agent: ${CONCURRENT}"
echo " Outer steps: ${OUTER_STEPS}"
echo " Inner steps: ${INNER_STEPS}"
echo "============================================"
echo ""

# Create checkpoint directories
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    mkdir -p "${CKPT_BASE}/agent${i}"
done
mkdir -p "${CKPT_BASE}/shared"

# Submit agents
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    SEED_OFFSET=$((i * 1000000))
    AGENT_CKPT="${CKPT_BASE}/agent${i}"
    SHARED_BEST="${CKPT_BASE}/shared/best.pt"

    echo "Submitting agent ${i} (seed_offset=${SEED_OFFSET})..."
    nlprun -q jag -g 1 -r 40G -c 8 -p standard -n "rnad-agent${i}" \
        "${CONDA_HOOK} && cd ${PROJECT} && python -m hexzero.scripts.rnad_train \
        --concurrent ${CONCURRENT} \
        --outer-steps ${OUTER_STEPS} \
        --inner-steps ${INNER_STEPS} \
        --eval-games 24 \
        --eval-every ${EVAL_EVERY} \
        --lr ${LR} \
        --seed-offset ${SEED_OFFSET} \
        --agent-id ${i} \
        --checkpoint-dir ${AGENT_CKPT} \
        --auto-resume ${SHARED_BEST} \
        --wandb-key ${WANDB_KEY} \
        --wandb-name rnad-agent${i}"
    echo "  Agent ${i} submitted."
done

echo ""
echo "All ${NUM_AGENTS} agents submitted."
echo ""

# Start coordinator in background
echo "Starting coordinator..."
nohup bash "${PROJECT}/hexzero/scripts/coordinate_pbt.sh" \
    > "${PROJECT}/coordinator.log" 2>&1 &
COORD_PID=$!
echo "  Coordinator PID: ${COORD_PID}"
echo "  Log: ${PROJECT}/coordinator.log"

echo ""
echo "============================================"
echo " All launched!"
echo " Monitor agents:  squeue -u nroll"
echo " Agent 0 output:  tail -f ~/rnad-agent0.out"
echo " Agent 1 output:  tail -f ~/rnad-agent1.out"
echo " Agent 2 output:  tail -f ~/rnad-agent2.out"
echo " Agent 3 output:  tail -f ~/rnad-agent3.out"
echo " Coordinator:     tail -f ${PROJECT}/coordinator.log"
echo " W&B:             https://wandb.ai (project: hexazero)"
echo "============================================"
