#!/bin/bash
# ============================================================================
# Regenerate AB2 pretraining data with ACTUAL 2-ply search (not 1-ply greedy).
#
# Wipes existing ab2_v2 shards and regenerates 100k games with --depth 2.
# Run on cluster with plenty of CPUs:
#
#   nlprun -q jag -g 0 -r 30G -c 48 -p standard -n regen-ab2 \
#     'cd /nlp/scr/nroll/catan_training && bash human_bot/regen_ab2_data.sh'
#
# Or from local machine:
#   bash human_bot/launch_regen_ab2.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

AB2_DIR="${PROJECT_DIR}/data/ab2_v2"
NUM_GAMES=100000
NUM_CPUS=$(nproc 2>/dev/null || echo 8)
WORKERS=$(( NUM_CPUS > 36 ? 36 : NUM_CPUS ))

mkdir -p "${AB2_DIR}"

echo "============================================================"
echo "  Regenerating AB2 data with 2-ply search"
echo "  Node: $(hostname)"
echo "  CPUs: ${NUM_CPUS}  Workers: ${WORKERS}"
echo "  Games: ${NUM_GAMES}"
echo "  Output: ${AB2_DIR}"
echo "============================================================"
echo ""

# Build C library
python3 -c "from hexzero.bindings.lib_loader import load_library; load_library()" 2>&1

# Count existing (1-ply) shards
OLD_COUNT=$(find "${AB2_DIR}" -name '*.pt' ! -name 'metadata.pt' 2>/dev/null | wc -l | tr -d ' ')
echo "Existing shards (1-ply): ${OLD_COUNT}"

if [ "${OLD_COUNT}" -gt 0 ]; then
    BACKUP="${AB2_DIR}_1ply_backup"
    echo "Backing up old data to ${BACKUP}..."
    mkdir -p "${BACKUP}"
    find "${AB2_DIR}" -name '*.pt' -exec mv {} "${BACKUP}/" \;
    echo "  Moved ${OLD_COUNT} shards to backup."
fi

echo ""
echo "Generating ${NUM_GAMES} games with --depth 2..."
echo ""

python3 -u -m hexzero.scripts.collect_ab2_games \
    --output-dir "${AB2_DIR}" \
    --num-games "${NUM_GAMES}" \
    --num-workers "${WORKERS}" \
    --games-per-file 25 \
    --depth 2

echo ""
NEW_COUNT=$(find "${AB2_DIR}" -name '*.pt' ! -name 'metadata.pt' | wc -l | tr -d ' ')
echo "Done. New 2-ply shards: ${NEW_COUNT}"
