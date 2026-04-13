#!/bin/bash
# =============================================================================
# Full no-lookahead pipeline on jag28 (8 GPUs, 92 CPUs, shared SSD)
#
# Phase 1: Collect 2000 AB2-vs-AB2 games (CPU-only, ~90 workers)
# Phase 2: Supervised BC on collected data (8-GPU DataParallel)
# Phase 3: Self-play actor-learner (8 GPUs: 1 learner + 7*N actors on SSD)
# =============================================================================
set -uo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
WANDB_KEY="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"

# ── Local SSD (shared across all processes on this node) ──────────────
LOCAL=""
for CANDIDATE in "/scr-ssd/nroll/hexazero" "/scr/nroll/hexazero" "/tmp/hexazero"; do
    PARENT=$(dirname "$CANDIDATE")
    if mkdir -p "$PARENT" 2>/dev/null && [ -w "$PARENT" ]; then
        LOCAL="$CANDIDATE"
        break
    fi
done
[ -z "$LOCAL" ] && LOCAL="/tmp/hexazero"

DATA_DIR="${LOCAL}/ab2_data"
CKPT_DIR="${LOCAL}/checkpoints"
TRAJ_DIR="${LOCAL}/trajectories"
mkdir -p "$DATA_DIR" "$CKPT_DIR" "$TRAJ_DIR" "${TRAJ_DIR}/processed"

NFS_BACKUP="${PROJECT}/checkpoints"
mkdir -p "$NFS_BACKUP"

echo "============================================"
echo " No-Lookahead Pipeline on $(hostname)"
echo " Local SSD : ${LOCAL}"
echo " NFS backup: ${NFS_BACKUP}"
echo "============================================"
echo ""

# Detect GPUs
if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
    NUM_GPUS=1
fi
TOTAL_CPUS=$(nproc 2>/dev/null || echo 8)
echo "Hardware: ${NUM_GPUS} GPUs, ${TOTAL_CPUS} CPUs"
echo ""

# =====================================================================
# PHASE 1: Collect AB2 games (CPU-only)
# =====================================================================
PHASE1_GAMES=2000
PHASE1_WORKERS=$((TOTAL_CPUS - 2))
[ $PHASE1_WORKERS -lt 4 ] && PHASE1_WORKERS=4
[ $PHASE1_WORKERS -gt 90 ] && PHASE1_WORKERS=90

echo "========== PHASE 1: Collecting ${PHASE1_GAMES} AB2 games =========="
echo "  Workers: ${PHASE1_WORKERS}"
echo "  Output:  ${DATA_DIR}"

T1_START=$(date +%s)
python -m hexzero.scripts.collect_ab2_games \
    --num-games $PHASE1_GAMES \
    --num-workers $PHASE1_WORKERS \
    --output-dir "$DATA_DIR" \
    --games-per-file 50
T1_END=$(date +%s)
echo "Phase 1 done: $((T1_END - T1_START))s"

NFILES=$(ls "$DATA_DIR"/*.pt 2>/dev/null | wc -l)
if [ "$NFILES" -lt 1 ]; then
    echo "ERROR: No data files generated. Aborting."
    exit 1
fi
echo "  ${NFILES} data files written"
echo ""

# =====================================================================
# PHASE 2: Supervised training (single GPU)
# =====================================================================
PHASE2_EPOCHS=10
PHASE2_BS=2048
PHASE2_LR=0.001
PHASE2_EVAL=25

echo "========== PHASE 2: Supervised BC (${PHASE2_EPOCHS} epochs) =========="
echo "  GPU: 0 (single, edge_index topology not scatterable)"
echo "  BS:  ${PHASE2_BS}"

T2_START=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python -m hexzero.scripts.supervised_train \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs $PHASE2_EPOCHS \
    --batch-size $PHASE2_BS \
    --lr $PHASE2_LR \
    --eval-games $PHASE2_EVAL \
    --entropy-weight 0.01 \
    --wandb-key "$WANDB_KEY" \
    --wandb-project hexazero
T2_END=$(date +%s)
echo "Phase 2 done: $((T2_END - T2_START))s"

# Copy best checkpoint to NFS for persistence
if [ -f "${CKPT_DIR}/best.pt" ]; then
    cp "${CKPT_DIR}/best.pt" "${NFS_BACKUP}/supervised_best.pt"
    echo "Supervised checkpoint backed up to NFS"
fi

# Pick seed for phase 3
SEED_CKPT="${CKPT_DIR}/best.pt"
if [ ! -f "$SEED_CKPT" ]; then
    SEED_CKPT=$(ls -t "${CKPT_DIR}"/epoch_*.pt 2>/dev/null | head -1)
fi
if [ -z "$SEED_CKPT" ] || [ ! -f "$SEED_CKPT" ]; then
    echo "ERROR: No supervised checkpoint found. Aborting."
    exit 1
fi
echo "Phase 3 seed: ${SEED_CKPT}"
echo ""

# =====================================================================
# PHASE 3: Self-play actor-learner (8 GPUs)
#
# Each actor = 1 Python process, mostly CPU-bound (AB2 1-ply lookahead).
# GPU forward passes are tiny (batch=8, ~1ms), so many actors share a GPU.
# Scale actor count to saturate CPU cores, not GPUs.
#
# ~400MB VRAM per actor → 24GB GPU fits ~50 actors. CPU is the bottleneck.
# With 92 CPUs: reserve 2 for OS + 2 for learner → 88 actor processes.
# 88 / 8 GPUs = 11 per GPU. GPU 0 gets 10 (learner takes one slot).
# =====================================================================

# GPU 0 = learner (optimized: tensorized data, BS=8192, single bulk H2D).
# GPUs 1-7 = actors. Scale to available CPUs; backpressure caps disk usage.
ACTOR_GPUS=$((NUM_GPUS - 1))  # GPUs 1..7
[ $ACTOR_GPUS -lt 1 ] && ACTOR_GPUS=1
AVAILABLE_CPUS=$((TOTAL_CPUS - 4))  # reserve for OS + learner
[ $AVAILABLE_CPUS -lt $ACTOR_GPUS ] && AVAILABLE_CPUS=$ACTOR_GPUS
ACTORS_PER_GPU=$((AVAILABLE_CPUS / ACTOR_GPUS))
[ $ACTORS_PER_GPU -gt 50 ] && ACTORS_PER_GPU=50
TOTAL_ACTORS=$((ACTORS_PER_GPU * ACTOR_GPUS))

GAMES_PER_ACTOR=8
ACTOR_TEMP=1.0
ACTOR_RELOAD=5

echo "========== PHASE 3: Self-play Actor-Learner =========="
echo "  Learner GPU    : 0 (tensorized, BS=4096)"
echo "  Actor GPUs     : 1-$((ACTOR_GPUS)) (${ACTORS_PER_GPU} actors each)"
echo "  Total actors   : ${TOTAL_ACTORS}"
echo "  Games/actor    : ${GAMES_PER_ACTOR}"
echo "  Concurrent games: $((TOTAL_ACTORS * GAMES_PER_ACTOR))"

# ── Start learner on GPU 0 ────────────────────────────────────────────
echo "Starting learner on GPU 0 ..."
CUDA_VISIBLE_DEVICES=0 python -m hexzero.scripts.selfplay_learner \
    --checkpoint "$SEED_CKPT" \
    --checkpoint-dir "$CKPT_DIR" \
    --trajectory-dir "$TRAJ_DIR" \
    --batch-size 4096 --lr 0.0003 \
    --entropy-weight 0.1 \
    --ab2-weight-start 1.0 --ab2-weight-end 0.0 --ab2-anneal-steps 500 \
    --max-files-per-step 50 \
    --eval-every 25 --eval-games 25 \
    --poll-interval 2 \
    --wandb-key "$WANDB_KEY" \
    --wandb-project hexazero-selfplay \
    > "${PROJECT}/learner.log" 2>&1 &
LEARNER_PID=$!
echo "  Learner PID=$LEARNER_PID"

echo "  Waiting for initial checkpoint..."
for i in $(seq 1 120); do
    [ -f "${CKPT_DIR}/latest.pt" ] && break
    sleep 1
done
if [ ! -f "${CKPT_DIR}/latest.pt" ]; then
    echo "WARNING: Timed out waiting for learner checkpoint, continuing anyway"
fi
echo "  Checkpoint ready"

# ── Start actors across GPUs 1..N ─────────────────────────────────────
AID=0
for gpu in $(seq 1 $ACTOR_GPUS); do
    echo "GPU ${gpu}: ${ACTORS_PER_GPU} actors"
    for _ in $(seq 1 $ACTORS_PER_GPU); do
        CUDA_VISIBLE_DEVICES=$gpu python -m hexzero.scripts.selfplay_actor \
            --actor-id $AID \
            --checkpoint-dir "$CKPT_DIR" \
            --trajectory-dir "$TRAJ_DIR" \
            --games-per-batch $GAMES_PER_ACTOR \
            --temperature $ACTOR_TEMP \
            --reload-every $ACTOR_RELOAD \
            > "${PROJECT}/sp_actor${AID}.log" 2>&1 &
        AID=$((AID + 1))
    done
done

echo ""
echo "Launched: 1 learner (GPU 0) + ${AID} actors (GPUs 1-${ACTOR_GPUS})"
echo "Concurrent games: $((AID * GAMES_PER_ACTOR))"
echo ""

# ── Background: NFS backup + disk cleanup ─────────────────────────────
(
    while kill -0 $LEARNER_PID 2>/dev/null; do
        sleep 120
        if [ -f "${CKPT_DIR}/latest.pt" ]; then
            cp "${CKPT_DIR}/latest.pt" "${NFS_BACKUP}/selfplay_latest.pt" 2>/dev/null
        fi
        if [ -f "${CKPT_DIR}/best.pt" ]; then
            cp "${CKPT_DIR}/best.pt" "${NFS_BACKUP}/selfplay_best.pt" 2>/dev/null
        fi
        # Prune old processed trajectories
        PROCESSED="${TRAJ_DIR}/processed"
        if [ -d "$PROCESSED" ]; then
            NPROC=$(ls "$PROCESSED" 2>/dev/null | wc -l)
            if [ "$NPROC" -gt 200 ]; then
                ls -t "$PROCESSED"/*.pt 2>/dev/null | tail -n +101 | xargs rm -f 2>/dev/null
            fi
        fi
        # Keep 30 most recent step checkpoints (eval references step-25 back)
        ls -t "${CKPT_DIR}"/step_*.pt 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null
    done
) &
BACKUP_PID=$!
echo "Background backup + cleanup (PID=$BACKUP_PID)"

echo ""
echo "Monitor:"
echo "  tail -f ${PROJECT}/learner.log"
echo "  tail -n 5 ${PROJECT}/sp_actor{0,1,2,3}.log"
echo "  ls ${TRAJ_DIR}/ | wc -l"
echo "  nvidia-smi"
echo ""
echo "Waiting for learner (PID=$LEARNER_PID)..."
wait $LEARNER_PID
EXIT_CODE=$?

echo "Learner exited ($EXIT_CODE)"
echo "Final checkpoint backup..."
cp "${CKPT_DIR}/latest.pt" "${NFS_BACKUP}/selfplay_latest.pt" 2>/dev/null || true
cp "${CKPT_DIR}/best.pt" "${NFS_BACKUP}/selfplay_best.pt" 2>/dev/null || true

echo "Cleaning up..."
kill $(jobs -p) 2>/dev/null || true
wait 2>/dev/null || true

echo "Local SSD usage: $(du -sh ${LOCAL} 2>/dev/null | cut -f1)"
echo "Done."
