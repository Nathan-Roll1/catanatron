#!/bin/bash
# ============================================================================
# Regenerate human game training data from Colonist JSON with fixed port encoding.
#
# Clears existing converted shards and re-converts from the raw JSON archive.
#
#   nlprun -q jag -g 0 -r 30G -c 48 -p standard -n regen-human \
#     'cd /nlp/scr/nroll/catan_training && bash human_bot/regen_human_data.sh'
# ============================================================================

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

COLONIST_RAW="${PROJECT_DIR}/data/colonist_raw/games"
HUMAN_DIR="${PROJECT_DIR}/data/human_v2_fixed"
NUM_CPUS=$(nproc 2>/dev/null || echo 8)
WORKERS=$(( NUM_CPUS > 36 ? 36 : NUM_CPUS ))

mkdir -p "${HUMAN_DIR}"

echo "============================================================"
echo "  Regenerating human game data (fixed port encoding)"
echo "  Node: $(hostname)"
echo "  CPUs: ${NUM_CPUS}  Workers: ${WORKERS}"
echo "  Input:  ${COLONIST_RAW}"
echo "  Output: ${HUMAN_DIR}"
echo "============================================================"
echo ""

COLONIST_COUNT=$(find "${COLONIST_RAW}" -name "*.json" -type f | wc -l | tr -d ' ')
echo "Colonist JSON files: ${COLONIST_COUNT}"

OLD_COUNT=$(find "${HUMAN_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')
if [ "${OLD_COUNT}" -gt 0 ]; then
    echo "Clearing ${OLD_COUNT} old shards..."
    find "${HUMAN_DIR}" -name '*.pt' -delete
fi

echo ""
echo "Converting with fixed port encoder..."
echo ""

python3 -u -m human_bot.colonist_converter \
    --input-dir "${COLONIST_RAW}" \
    --output-dir "${HUMAN_DIR}" \
    --num-workers "${WORKERS}" \
    --games-per-shard 200

echo ""
NEW_COUNT=$(find "${HUMAN_DIR}" -name '*.pt' | wc -l | tr -d ' ')
echo "Done. New shards: ${NEW_COUNT}"
