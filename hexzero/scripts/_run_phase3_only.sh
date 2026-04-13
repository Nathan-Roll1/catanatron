#!/bin/bash
# Phase 3: Expert Iteration self-play.
# All 4 seats are HZ. Each move uses 1-ply NN search (top-K) to produce
# improved action targets. Learner trains via cross-entropy (no REINFORCE).
set -uo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
WANDB_KEY="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"

LOCAL="/scr/nroll/hexazero"
CKPT_DIR="${LOCAL}/checkpoints"
TRAJ_DIR="${LOCAL}/trajectories"
NFS_BACKUP="${PROJECT}/checkpoints"
mkdir -p "$CKPT_DIR" "$TRAJ_DIR" "${TRAJ_DIR}/processed" "$NFS_BACKUP"

SEED_CKPT="${NFS_BACKUP}/supervised_best.pt"
if [ ! -f "$SEED_CKPT" ]; then
    SEED_CKPT="${CKPT_DIR}/best.pt"
fi
if [ ! -f "$SEED_CKPT" ]; then
    echo "ERROR: No checkpoint found. Aborting."
    exit 1
fi

rm -f "${TRAJ_DIR}"/actor*.pt "${TRAJ_DIR}"/processed/*.pt 2>/dev/null

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
TOTAL_CPUS=$(nproc 2>/dev/null || echo 8)
ACTOR_GPUS=$((NUM_GPUS - 1))
[ $ACTOR_GPUS -lt 1 ] && ACTOR_GPUS=1
ACTORS_PER_GPU=$(( (TOTAL_CPUS - 4) / ACTOR_GPUS ))
[ $ACTORS_PER_GPU -gt 50 ] && ACTORS_PER_GPU=50
TOTAL_ACTORS=$((ACTORS_PER_GPU * ACTOR_GPUS))

TOP_K=5
ACTOR_TEMP=0.3

echo "============================================"
echo " Expert Iteration on $(hostname)"
echo " Seed ckpt   : ${SEED_CKPT}"
echo " Learner GPU : 0 (BS=4096, cross-entropy)"
echo " Actor GPUs  : 1-${ACTOR_GPUS} (${ACTORS_PER_GPU} each)"
echo " Total actors: ${TOTAL_ACTORS}"
echo " Top-K       : ${TOP_K}"
echo " Temperature : ${ACTOR_TEMP}"
echo " Games       : $((TOTAL_ACTORS * 8)) concurrent"
echo "============================================"
echo ""

# ── Learner on GPU 0 ──────────────────────────────────────────────────
echo "Starting learner ..."
CUDA_VISIBLE_DEVICES=0 python -m hexzero.scripts.selfplay_learner \
    --checkpoint "$SEED_CKPT" \
    --checkpoint-dir "$CKPT_DIR" \
    --trajectory-dir "$TRAJ_DIR" \
    --batch-size 4096 --lr 0.0003 \
    --entropy-weight 0.01 \
    --max-files-per-step 50 \
    --eval-every 25 --eval-games 10 \
    --poll-interval 2 \
    --wandb-key "$WANDB_KEY" \
    --wandb-project hexazero-exit \
    > "${PROJECT}/learner.log" 2>&1 &
LEARNER_PID=$!
echo "  Learner PID=$LEARNER_PID"

echo "  Waiting for initial checkpoint..."
for i in $(seq 1 120); do
    [ -f "${CKPT_DIR}/latest.pt" ] && break
    sleep 1
done
echo "  Checkpoint ready"

# ── Actors on GPUs 1..N ───────────────────────────────────────────────
AID=0
for gpu in $(seq 1 $ACTOR_GPUS); do
    echo "GPU ${gpu}: ${ACTORS_PER_GPU} actors"
    for _ in $(seq 1 $ACTORS_PER_GPU); do
        CUDA_VISIBLE_DEVICES=$gpu python -m hexzero.scripts.selfplay_actor \
            --actor-id $AID \
            --checkpoint-dir "$CKPT_DIR" \
            --trajectory-dir "$TRAJ_DIR" \
            --games-per-batch 8 \
            --top-k $TOP_K \
            --temperature $ACTOR_TEMP \
            --reload-every 5 \
            > "${PROJECT}/sp_actor${AID}.log" 2>&1 &
        AID=$((AID + 1))
    done
done

echo ""
echo "Launched: 1 learner (GPU 0) + ${AID} actors (GPUs 1-${ACTOR_GPUS})"
echo ""

# ── Background backup + cleanup ──────────────────────────────────────
(
    while kill -0 $LEARNER_PID 2>/dev/null; do
        sleep 120
        cp "${CKPT_DIR}/latest.pt" "${NFS_BACKUP}/exit_latest.pt" 2>/dev/null
        cp "${CKPT_DIR}/best.pt" "${NFS_BACKUP}/exit_best.pt" 2>/dev/null
        ls -t "${TRAJ_DIR}/processed"/*.pt 2>/dev/null | tail -n +101 | xargs rm -f 2>/dev/null
        ls -t "${CKPT_DIR}"/step_*.pt 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null
    done
) &

echo "Monitor:"
echo "  tail -f ${PROJECT}/learner.log"
echo "  tail -1 ${PROJECT}/sp_actor{0,5,10}.log"
echo ""
echo "Waiting for learner (PID=$LEARNER_PID)..."
wait $LEARNER_PID

cp "${CKPT_DIR}/latest.pt" "${NFS_BACKUP}/exit_latest.pt" 2>/dev/null || true
cp "${CKPT_DIR}/best.pt" "${NFS_BACKUP}/exit_best.pt" 2>/dev/null || true
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true
echo "Done."
