#!/bin/bash
# ============================================================================
# Launch script for Pipeline v3 (large model, AB2-only pretrain → self-play)
#
# Run this locally to submit the job to a jag node.
# ============================================================================

set -euo pipefail

PROJECT_DIR="/nlp/scr/nroll/catan_training"

# ── Sync code to cluster ─────────────────────────────────────────
echo "Syncing code to ${PROJECT_DIR}..."
rsync -avz --delete \
    --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='.git' --exclude='csrc/*.o' --exclude='csrc/*.so' \
    --exclude='csrc/*.dylib' --exclude='csrc/*_asan*' \
    --exclude='data/' --exclude='checkpoints/' \
    --exclude='.benchmarks/' --exclude='node_modules/' \
    --exclude='*.dSYM' \
    ~/Documents/catanatron/ \
    nroll@jagupard10.stanford.edu:${PROJECT_DIR}/

echo "Done syncing."
echo ""

# ── Submit job ───────────────────────────────────────────────────
WANDB_KEY="${WANDB_API_KEY:-$(cat "$HOME/.wandb_key" 2>/dev/null || echo "")}"

cat << 'SCRIPT_EOF'
Submitting pipeline v3 job:
  - 4 GPUs, 48 CPUs, 120G RAM
  - Model: gnn_hidden=80, trunk_channels=192 (~1M params)
  - AB2 pretrain (100k games from data/ab2_v2) → Self-play ExIt
  - Temperature: 1.0 → 0.2 over 200 rounds
  - Eval: vs AB2 + h2h vs previous checkpoint
  - W&B: sp-large-MMDD-HHMM
SCRIPT_EOF

ssh nroll@jagupard10.stanford.edu "nlprun -q jag -g 4 -r 120G -c 48 -p standard -n sp-large-v3 \
  'export PROJECT_DIR=${PROJECT_DIR} && \
   export WANDB_API_KEY=${WANDB_KEY} && \
   export GNN_HIDDEN=80 && \
   export TRUNK_CHANNELS=192 && \
   cd \${PROJECT_DIR} && \
   export PYTHONPATH=\${PROJECT_DIR}:\${PYTHONPATH:-} && \

   echo \"============================================================\" && \
   echo \"  Pipeline v3: Large model AB2 pretrain → Self-play\" && \
   echo \"  Node: \$(hostname)\" && \
   echo \"  GPUs: \$(nvidia-smi -L 2>/dev/null | wc -l)\" && \
   echo \"  CPUs: \$(nproc)\" && \
   echo \"============================================================\" && \

   # ── Build C library ──
   python3 -c \"from hexzero.bindings.lib_loader import load_library; load_library()\" 2>&1 && \

   # ── Verify AB2 data ──
   AB2_COUNT=\$(ls \${PROJECT_DIR}/data/ab2_v2/*.pt 2>/dev/null | grep -vc metadata) && \
   echo \"AB2 shards: \${AB2_COUNT}\" && \

   # ── Dirs ──
   mkdir -p \${PROJECT_DIR}/checkpoints/pretrain_v3 \
            \${PROJECT_DIR}/data/selfplay_v3/pending \
            \${PROJECT_DIR}/data/selfplay_v3/consumed \
            \${PROJECT_DIR}/checkpoints/selfplay_v3 && \

   # ── Phase 1: AB2 Pretrain ──
   echo \"\" && \
   echo \"Phase 1: AB2 pretrain...\" && \
   python3 -u human_bot/cluster_train_inner.py \
     --ab2-dir \${PROJECT_DIR}/data/ab2_v2 \
     --ckpt-dir \${PROJECT_DIR}/checkpoints/pretrain_v3 \
     --batch-size 8192 \
     --shards-per-group 20 \
     --eval-games 50 && \

   PRETRAINED=\${PROJECT_DIR}/checkpoints/pretrain_v3/final.pt && \
   echo \"Pretrained: \${PRETRAINED}\" && \

   # ── Phase 2: Self-play ──
   NUM_GPUS=\$(nvidia-smi -L 2>/dev/null | wc -l) && \
   NUM_CPUS=\$(nproc) && \
   NUM_ACTOR_GPUS=\$(( NUM_GPUS > 1 ? NUM_GPUS - 1 : 1 )) && \
   ACTORS_PER_GPU=\$(( (NUM_CPUS - 1) / NUM_ACTOR_GPUS )) && \
   ACTORS_PER_GPU=\$(( ACTORS_PER_GPU > 1 ? ACTORS_PER_GPU : 1 )) && \
   echo \"\" && \
   echo \"Phase 2: Self-play ExIt\" && \
   echo \"  Learner GPU: cuda:0\" && \
   echo \"  Actor GPUs: \${NUM_ACTOR_GPUS}  Actors/GPU: \${ACTORS_PER_GPU}\" && \
   echo \"  Temperature: 1.0 -> 0.2 over 200 rounds\" && \

   python3 -u human_bot/selfplay.py \
     --checkpoint \${PRETRAINED} \
     --role all \
     --num-actor-gpus \${NUM_ACTOR_GPUS} \
     --actors-per-gpu \${ACTORS_PER_GPU} \
     --shard-dir \${PROJECT_DIR}/data/selfplay_v3 \
     --ckpt-dir \${PROJECT_DIR}/checkpoints/selfplay_v3 \
     --batch-size 8192 \
     --shards-per-train 20 \
     --search-depth 1 \
     --deep-search-depth 5 \
     --eval-games 50 \
     --eval-interval 4 \
     --reload-interval 100 \
     --wandb-name sp-large-\$(date +%m%d-%H%M)
  '"

echo ""
echo "Job submitted. Monitor with:"
echo "  ssh nroll@jagupard10.stanford.edu 'nlprun -o'"
echo "  ssh nroll@jagupard10.stanford.edu 'tail -f /nlp/scr/nroll/catan_training/nohup.out'"
