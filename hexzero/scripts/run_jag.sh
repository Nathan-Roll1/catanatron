#!/bin/bash
# =============================================================================
# HexaZero Jag GPU Bootstrap & Launch
# Run from sc: bash /nlp/scr/nroll/catanatron/hexzero/scripts/run_jag.sh
# =============================================================================
set -euo pipefail

USER_ID="nroll"
PROJECT="/nlp/scr/${USER_ID}/catanatron"
WANDB_KEY="wandb_v1_5Wm7tx6uj1GvNyXjt5ogWR8WJyO"

echo "============================================"
echo " HexaZero - Jag GPU Pipeline"
echo "============================================"

# ── Step 1: Find a free jag GPU ──────────────────────────────────────
echo ""
echo "[1/5] Requesting jag-standard interactive GPU session..."
echo "      (this will block until a GPU is allocated)"
echo ""

# Use nlprun to get an interactive session with 1 GPU
# Exclude older jags with small VRAM (<11GB) if known
# Request 60GB RAM and 16 cores for self-play + training in one session
exec nlprun -q jag -g 1 -r 60G -c 16 -p standard -a "$PROJECT/hexzero/scripts/jag_inner.sh"
