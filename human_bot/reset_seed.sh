#!/usr/bin/env bash
# Atomically reset the exit_v2 run from a known-good seed checkpoint.
#
# Why this exists:
#   `scancel -n exit-v2-learn` is async — it sends SIGTERM and returns.
#   If you immediately mv/cp checkpoints, the still-running learner
#   finishes its current round and overwrites your fresh seed with the
#   stale weights. This script kills, polls until squeue shows zero
#   learner jobs, THEN swaps checkpoints, then verifies the swap.
#
# Usage:
#   bash human_bot/reset_seed.sh [SEED_CKPT]
#
# Defaults to checkpoints/c_selfplay_v4/latest.pt (M2). Pass a different
# path as $1 to override.
set -euo pipefail

cd /nlp/scr/nroll/catan_training_big

SEED_CKPT="${1:-checkpoints/c_selfplay_v4/latest.pt}"
RUN_DIR="checkpoints/exit_v2"
SHARD_DIR="data/exit_v2"

if [ ! -f "$SEED_CKPT" ]; then
    echo "ERROR: seed checkpoint $SEED_CKPT does not exist" >&2
    exit 1
fi

echo "=== reset_seed.sh ==="
echo "  seed = $SEED_CKPT"
echo "  size = $(du -h "$SEED_CKPT" | cut -f1)"
echo "  md5  = $(md5sum "$SEED_CKPT" | cut -d' ' -f1)"

# 1. Kill ALL nroll jobs (we are doing a hard reset; expect to relaunch
#    actors after this).
echo
echo "=== killing existing jobs ==="
scancel -u nroll 2>/dev/null || true

# 2. Poll squeue until no exit-v2-* jobs remain. SIGTERM can take 10-60s
#    for the learner to finish its round and exit cleanly. We wait up to
#    180s, then SIGKILL anything left.
echo "=== polling squeue until clear (up to 180s) ==="
for i in $(seq 1 36); do
    pending=$(squeue -u nroll -h | grep -c "exit-v2-" || true)
    if [ "$pending" -eq 0 ]; then
        echo "  cleared after $((i * 5))s"
        break
    fi
    sleep 5
done

# Force-kill anything still hanging on
remaining=$(squeue -u nroll -h | awk '/exit-v2-/ {print $1}')
if [ -n "$remaining" ]; then
    echo "=== SIGKILL stragglers ==="
    for jid in $remaining; do
        echo "  scancel --signal=KILL $jid"
        scancel --signal=KILL "$jid" 2>/dev/null || true
    done
    sleep 10
fi

squeue -u nroll | grep -E "exit-v2-" && {
    echo "ERROR: jobs still present after kill" >&2
    squeue -u nroll
    exit 1
} || echo "  squeue is clean"

# 3. Backup whatever was in exit_v2/ (so we can recover if the seed
#    turns out to be bad).
echo
echo "=== backing up old exit_v2 checkpoints ==="
ts=$(date +%Y%m%d_%H%M%S)
for f in init.pt latest.pt best.pt; do
    if [ -f "$RUN_DIR/$f" ]; then
        mv "$RUN_DIR/$f" "$RUN_DIR/${f%.pt}_pre_${ts}.pt"
        echo "  $RUN_DIR/$f -> $RUN_DIR/${f%.pt}_pre_${ts}.pt"
    fi
done

# 4. Seed init.pt + latest.pt from the chosen checkpoint
echo
echo "=== seeding fresh checkpoints ==="
mkdir -p "$RUN_DIR"
cp "$SEED_CKPT" "$RUN_DIR/init.pt"
cp "$SEED_CKPT" "$RUN_DIR/latest.pt"

# 5. Verify the writes actually landed and match the seed
src_md5=$(md5sum "$SEED_CKPT" | cut -d' ' -f1)
init_md5=$(md5sum "$RUN_DIR/init.pt" | cut -d' ' -f1)
latest_md5=$(md5sum "$RUN_DIR/latest.pt" | cut -d' ' -f1)
echo "  src    md5 = $src_md5"
echo "  init   md5 = $init_md5"
echo "  latest md5 = $latest_md5"
if [ "$src_md5" != "$init_md5" ] || [ "$src_md5" != "$latest_md5" ]; then
    echo "ERROR: md5 mismatch — checkpoint copy didn't land cleanly" >&2
    exit 1
fi
echo "  md5s match — seed is in place"

# 6. Clear stale shards from the previous run + W&B id so the next learner
#    launch starts a fresh W&B run.
echo
echo "=== clearing stale state ==="
mkdir -p "$SHARD_DIR/pending"
rm -f "$SHARD_DIR/pending/"*.pt 2>/dev/null || true
rm -f "$RUN_DIR/.wandb_id" 2>/dev/null || true
rm -f "$RUN_DIR/.actor_stats/"*.json 2>/dev/null || true
echo "  pending shards: $(ls "$SHARD_DIR/pending/" | wc -l)"

echo
echo "=== reset complete — safe to launch learner now ==="
