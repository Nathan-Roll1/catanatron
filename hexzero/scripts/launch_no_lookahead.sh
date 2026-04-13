#!/bin/bash
# =============================================================================
# Submit the no-lookahead pipeline to jag28 (8 GPUs, 92 CPUs)
#
# Usage from sc:
#   bash /nlp/scr/nroll/catanatron/hexzero/scripts/launch_no_lookahead.sh
# =============================================================================
set -euo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
CONDA_HOOK='eval "$(/nlp/scr/nroll/miniconda3/bin/conda shell.bash hook)" && conda activate hexazero'

echo "============================================"
echo " No-Lookahead Pipeline (8 GPUs, shared SSD)"
echo "============================================"
echo ""

# Clear old logs
rm -f "${PROJECT}"/learner.log "${PROJECT}"/sp_actor*.log

echo "Submitting ..."
nlprun -q jag -g 3 -r 100G -c 30 -p standard -n hz-nolookahead \
    "${CONDA_HOOK} && cd ${PROJECT} && bash hexzero/scripts/_run_no_lookahead.sh"

echo ""
echo "Monitor:"
echo "  squeue -u nroll"
echo "  tail -f ${PROJECT}/hz-nolookahead.out"
echo "  tail -f ${PROJECT}/learner.log"
echo "  tail -n 5 ${PROJECT}/sp_actor{0,1,2,3}.log"
echo ""
