#!/bin/bash
# =============================================================================
# Optimal layout: local SSD for IO, scale actors to available CPU cores.
#
# Key optimizations:
#   1. Trajectories + checkpoints on /scr-ssd/ (local NVMe), not NFS
#   2. Scale actors to fill available CPU cores (up to 40)
#   3. Periodic rsync of best checkpoint back to NFS for persistence
# =============================================================================
set -uo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
WANDB_KEY="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"

# ── Local SSD paths (100x faster than NFS for small files) ────────────
# Try /scr-ssd/nroll/, fall back to /tmp/ if permission denied
LOCAL=""
for CANDIDATE in "/scr-ssd/nroll/hexazero_$$" "/scr/nroll/hexazero_$$" "/tmp/hexazero_$$"; do
    PARENT=$(dirname "$CANDIDATE")
    if mkdir -p "$PARENT" 2>/dev/null && [ -w "$PARENT" ]; then
        LOCAL="$CANDIDATE"
        break
    fi
done
if [ -z "$LOCAL" ]; then
    LOCAL="/tmp/hexazero_$$"
fi
CKPT_DIR="${LOCAL}/checkpoints"
TRAJ_DIR="${LOCAL}/trajectories"
mkdir -p "$CKPT_DIR" "$TRAJ_DIR" "${TRAJ_DIR}/processed"
echo "Local IO dir: ${LOCAL}"

# NFS backup dir for persistence across jobs
NFS_BACKUP="${PROJECT}/checkpoints"
mkdir -p "$NFS_BACKUP"

CONCURRENT=8
RELOAD_EVERY=50

# Detect GPUs
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
else
    GPUS=(0 1 2 3)
fi
NUM_GPUS=${#GPUS[@]}
unset CUDA_VISIBLE_DEVICES

# Scale actors to available CPU cores (leave 2 for OS + learner)
TOTAL_CPUS=$(nproc)
MAX_ACTORS=$((TOTAL_CPUS - 2))
if [ $MAX_ACTORS -gt 46 ]; then
    MAX_ACTORS=46
fi
if [ $MAX_ACTORS -lt $NUM_GPUS ]; then
    MAX_ACTORS=$NUM_GPUS
fi

# Distribute actors evenly across GPUs
ACTORS_PER_GPU=$((MAX_ACTORS / NUM_GPUS))

echo "============================================"
echo " R-NaD Actor-Learner on $(hostname)"
echo " GPUs: ${GPUS[*]} (${NUM_GPUS}x)"
echo " CPUs: ${TOTAL_CPUS} available, using $((MAX_ACTORS + 1))"
echo " Actors: ${MAX_ACTORS} (${ACTORS_PER_GPU}/GPU)"
echo " IO: local SSD at ${LOCAL}"
echo " Concurrent: ${CONCURRENT} games/actor"
echo " Total concurrent games: $((MAX_ACTORS * CONCURRENT))"
echo "============================================"
echo ""

ACTOR_ARGS="--trajectory-dir ${TRAJ_DIR} --checkpoint-dir ${CKPT_DIR} --concurrent ${CONCURRENT} --reload-every ${RELOAD_EVERY}"

# ── Learner on GPU 0 ─────────────────────────────────────────────────
G=${GPUS[0]}
echo "Starting learner on GPU ${G}..."

CUDA_VISIBLE_DEVICES=$G python -m hexzero.scripts.rnad_learner \
    --trajectory-dir "$TRAJ_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --lr 0.003 --batch-size 2048 --eta 2.0 \
    --anchor-interval 25 --eval-every 25 --eval-games 24 \
    --poll-interval 3 --min-trajectories 2 \
    --wandb-key "$WANDB_KEY" \
    > "${PROJECT}/learner.log" 2>&1 &
LEARNER_PID=$!
echo "  Learner PID=$LEARNER_PID"

echo "  Waiting for initial checkpoint..."
for i in $(seq 1 60); do
    [ -f "${CKPT_DIR}/latest.pt" ] && break
    sleep 1
done
echo "  Checkpoint ready"

# ── Actors across all GPUs ────────────────────────────────────────────
AID=0
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    G=${GPUS[$gpu_idx]}

    # GPU 0 gets fewer actors (learner is there too)
    if [ $gpu_idx -eq 0 ]; then
        N=$((ACTORS_PER_GPU - 1))
        [ $N -lt 1 ] && N=1
    else
        N=$ACTORS_PER_GPU
    fi

    echo "GPU ${G}: ${N} actors (${AID}-$((AID + N - 1)))"
    for _ in $(seq 1 $N); do
        CUDA_VISIBLE_DEVICES=$G python -m hexzero.scripts.rnad_actor --actor-id $AID $ACTOR_ARGS \
            > "${PROJECT}/actor${AID}.log" 2>&1 &
        AID=$((AID + 1))
    done
done

TOTAL_ACTORS=$AID
echo ""
echo "Launched: 1 learner + ${TOTAL_ACTORS} actors = $((TOTAL_ACTORS + 1)) processes"
echo "Total concurrent games: $((TOTAL_ACTORS * CONCURRENT))"
echo ""

# ── Background: checkpoint backup + disk cleanup ─────────────────────
(
    while kill -0 $LEARNER_PID 2>/dev/null; do
        sleep 120
        # Backup checkpoint to NFS
        if [ -f "${CKPT_DIR}/latest.pt" ]; then
            cp "${CKPT_DIR}/latest.pt" "${NFS_BACKUP}/latest.pt" 2>/dev/null
        fi
        # Purge processed trajectories to prevent disk full
        PROCESSED="${TRAJ_DIR}/processed"
        if [ -d "$PROCESSED" ]; then
            NPROC=$(ls "$PROCESSED" 2>/dev/null | wc -l)
            if [ "$NPROC" -gt 200 ]; then
                ls -t "$PROCESSED"/*.pt 2>/dev/null | tail -n +101 | xargs rm -f 2>/dev/null
            fi
        fi
    done
) &
BACKUP_PID=$!
echo "Checkpoint backup + disk cleanup every 2min (PID=$BACKUP_PID)"

echo ""
echo "Monitor:"
echo "  tail -f ${PROJECT}/learner.log"
echo "  tail -n 3 ${PROJECT}/actor{0,1,2,3}.log"
echo "  ls ${TRAJ_DIR}/ | wc -l"
echo "  ssh $(hostname) nvidia-smi"
echo ""
echo "Waiting for learner (PID=$LEARNER_PID)..."
wait $LEARNER_PID
EXIT_CODE=$?

echo "Learner exited ($EXIT_CODE)"
echo "Final checkpoint backup to NFS..."
cp "${CKPT_DIR}/latest.pt" "${NFS_BACKUP}/latest.pt" 2>/dev/null || true
echo "Cleaning up..."
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true

echo "Local SSD usage: $(du -sh ${LOCAL} 2>/dev/null | cut -f1)"
rm -rf "${LOCAL}"
echo "Done."
