#!/bin/bash
# =============================================================================
# Actor-Learner R-NaD Training (single node, 4 GPUs)
#
# Requests 4 GPUs on one jag node, then launches:
#   GPU 0: Learner + 3 actors
#   GPU 1: 4 actors
#   GPU 2: 3 actors
#   GPU 3: 3 actors
#   = 13 actors + 1 learner = 14 processes on 14 CPU cores
#
# Usage from sc:
#   bash /nlp/scr/nroll/catanatron/hexzero/scripts/launch_actor_learner.sh
# =============================================================================
set -euo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
CONDA_HOOK='eval "$(/nlp/scr/nroll/miniconda3/bin/conda shell.bash hook)" && conda activate hexazero'

echo "============================================"
echo " Actor-Learner R-NaD (4 GPUs, 13 actors)"
echo "============================================"

echo "Submitting single-node 4-GPU job..."
nlprun -q jag -g 8 -r 200G -c 92 -p standard -m jagupard28 -n rnad-cluster \
    "${CONDA_HOOK} && cd ${PROJECT} && bash hexzero/scripts/_run_actor_learner.sh"
echo "  Job submitted."

echo ""
echo "Monitor:"
echo "  squeue -u nroll"
echo "  tail -f ${PROJECT}/rnad-cluster.out"
echo "  tail -f ${PROJECT}/learner.log"
echo "  tail -f ${PROJECT}/actor{0,1,2}.log"
echo "  ls ${PROJECT}/trajectories/ | wc -l"
echo ""
