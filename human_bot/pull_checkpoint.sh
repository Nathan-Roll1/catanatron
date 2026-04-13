#!/bin/bash
# Pull the sp-0412-0459 self-play checkpoint (small ~602k model) from cluster.
#
# Usage: bash human_bot/pull_checkpoint.sh
set -euo pipefail

rsync -avz --progress \
    nroll@sc:/nlp/scr/nroll/catan_training/checkpoints/selfplay_v2/latest.pt \
    checkpoints/sp_0412.pt
