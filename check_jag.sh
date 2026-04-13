#!/bin/bash
# Quick cluster status check
# Usage: bash check_jag.sh [node]  (default: jagupard28)

NODE="${1:-jagupard28}"

echo "=== ${NODE} ==="
ssh "nroll@${NODE}.stanford.edu" '
echo "--- GPU ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while IFS="," read idx util mem_used mem_total temp power; do
    pct=$((mem_used * 100 / mem_total))
    printf "  GPU %s: %3s%% util  %5s/%5s MiB (%2d%%)  %s°C  %sW\n" "$idx" "$util" "$mem_used" "$mem_total" "$pct" "$temp" "$power"
done
echo ""
echo "--- CPU ---"
top -bn1 | head -3 | tail -2
echo ""
echo "--- Memory ---"
free -h | grep -E "Mem|Swap"
echo ""
echo "--- Jobs ---"
ps aux | grep -E "selfplay|cluster_train|full_pipeline" | grep -v grep | wc -l | xargs -I{} echo "  Running processes: {}"
echo ""
echo "--- Disk ---"
du -sh /nlp/scr/nroll/catan_training/data/selfplay_v2/pending/ 2>/dev/null | sed "s/^/  pending: /"
du -sh /nlp/scr/nroll/catan_training/data/selfplay_v2/consumed/ 2>/dev/null | sed "s/^/  consumed: /"
du -sh /nlp/scr/nroll/catan_training/checkpoints/selfplay_v2/ 2>/dev/null | sed "s/^/  checkpoints: /"
'
