#!/usr/bin/env bash
# Run _diag_m2.py against every M2 candidate so we can pick the strongest
# seed for the next exit_v2 launch. Meant to be invoked under nlprun on a
# jag node (do NOT run on the sc head).
set -uo pipefail
cd /nlp/scr/nroll/catan_training_big
export PYTHONPATH=$PWD

CANDIDATES=(
    "checkpoints/c_selfplay_v4/latest.pt"
    "checkpoints/c_selfplay_v4/best.pt"
    "checkpoints/ab2_imit_v1/best.pt"
    "checkpoints/ab2_imit_v1/latest.pt"
)

for p in "${CANDIDATES[@]}"; do
    if [ ! -f "$p" ]; then
        echo "###### $p ###### (MISSING)"
        continue
    fi
    echo
    echo "###### $p ######"
    python3 -u human_bot/_diag_m2.py "$p" 2>&1 | \
        grep -E "size:|md5:|cp=|ef channel|flat per|top-5 action|value logits|WR ="
done
