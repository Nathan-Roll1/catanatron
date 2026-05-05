#!/usr/bin/env bash
# remote_super_exit.sh — drive super_m2 ExIt deployment from your laptop.
#
# All ssh/rsync go through jamie.stanford.edu (off-VPN bastion). The cluster
# project lives at /nlp/scr/nroll/catan_super_exit/. Subcommands wrap the
# in-cluster `human_bot/deploy_super_exit.sh` so you can fully drive things
# from your laptop.
#
# Subcommands:
#   sync             rsync local repo -> cluster (excludes data/, logs/, dylibs)
#   seed             Pick a seed M2 .pt from /nlp/scr/nroll/catan_training_big
#                    and copy it to $CLUSTER_ROOT/checkpoints/super_exit/init.pt
#   launch           Sync + (optionally seed) + launch-all on cluster
#   stop             Touch .stop on cluster (graceful shutdown)
#   reset-stop       Remove .stop on cluster (before re-launch)
#   status           Print status from cluster
#   tail-learn       Tail the learner output file on cluster
#   tail-actor N     Tail actor N's output file (0..3)
#   pull-eval        rsync eval/*.json back to local for inspection
#   ssh              Open an interactive ssh shell to sc via jamie
#   help             This message
#
# Required env (defaults given):
#   CLUSTER_USER=nroll
#   CLUSTER_ROOT=/nlp/scr/nroll/catan_super_exit
#   LOCAL_REPO=$HOME/Documents/catanatron
#   SEED_DONOR=/nlp/scr/nroll/catan_training_big/checkpoints/exit_v2/latest.pt
#
# One-shot usage:
#   bash human_bot/remote_super_exit.sh launch
#   bash human_bot/remote_super_exit.sh tail-learn
#   bash human_bot/remote_super_exit.sh stop

set -euo pipefail

CLUSTER_USER="${CLUSTER_USER:-nroll}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/nlp/scr/nroll/catan_super_exit}"
LOCAL_REPO="${LOCAL_REPO:-$HOME/Documents/catanatron}"
SEED_DONOR="${SEED_DONOR:-/nlp/scr/nroll/catan_training_big/checkpoints/exit_v2/latest.pt}"
JAMIE="${JAMIE_HOST:-jamie.stanford.edu}"
SC="${SC_HOST:-sc}"

# All ssh and rsync go through jamie via OpenSSH ProxyJump (-J).
# Requires modern openssh (>= 7.3); ssh-add your key locally.
SSH_OPTS=(-J "${CLUSTER_USER}@${JAMIE}")
SSH_TARGET="${CLUSTER_USER}@${SC}"
RSYNC_RSH="ssh -J ${CLUSTER_USER}@${JAMIE}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Run a command on sc via jamie. Quoted properly so spaces/quotes survive.
ssh_sc() {
    ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "$@"
}

cmd_sync() {
    log "rsync $LOCAL_REPO/ -> ${SSH_TARGET}:${CLUSTER_ROOT}/"
    # First make sure the cluster dir exists.
    ssh_sc "mkdir -p ${CLUSTER_ROOT}/{checkpoints/super_exit,data/super_exit/pending,logs}"
    rsync -azP --delete \
        --exclude='.git/' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='*.dylib' \
        --exclude='*.so' \
        --exclude='catan_player/catan_player' \
        --exclude='catan_player/weights/' \
        --exclude='data/colonist_raw/' \
        --exclude='data/human_v2_fixed/' \
        --exclude='data/super_m2/' \
        --exclude='checkpoints/' \
        --exclude='logs/' \
        --exclude='terminals/' \
        --exclude='*.tmp' \
        --exclude='/csrc/nn_weights_*test*.bin' \
        -e "$RSYNC_RSH" \
        "${LOCAL_REPO}/" \
        "${SSH_TARGET}:${CLUSTER_ROOT}/"
    log "Sync complete."
}

cmd_seed() {
    log "Seeding initial M2 checkpoint from $SEED_DONOR ..."
    ssh_sc "
        if [[ ! -f $SEED_DONOR ]]; then
            echo 'FATAL: $SEED_DONOR not found on cluster'
            ls /nlp/scr/nroll/catan_training_big/checkpoints/ 2>/dev/null || true
            exit 1
        fi
        mkdir -p ${CLUSTER_ROOT}/checkpoints/super_exit
        target=${CLUSTER_ROOT}/checkpoints/super_exit/init.pt
        if [[ -f \$target ]]; then
            echo \"Existing init.pt found at \$target — keeping it (use --force to overwrite).\"
        else
            cp $SEED_DONOR \$target
            echo \"Seeded \$target from $SEED_DONOR\"
            ls -lh \$target
        fi
    "
}

cmd_launch() {
    cmd_sync
    cmd_seed
    log "Launching on cluster (1 learner + 4 actors)..."
    # All env vars are evaluated on the cluster
    ssh_sc "
        cd ${CLUSTER_ROOT} && \
        bash human_bot/deploy_super_exit.sh reset-stop && \
        bash human_bot/deploy_super_exit.sh launch-all
    "
}

cmd_stop() {
    log "Touching .stop on cluster..."
    ssh_sc "cd ${CLUSTER_ROOT} && bash human_bot/deploy_super_exit.sh stop"
}

cmd_reset_stop() {
    ssh_sc "cd ${CLUSTER_ROOT} && bash human_bot/deploy_super_exit.sh reset-stop"
}

cmd_status() {
    ssh_sc "cd ${CLUSTER_ROOT} && bash human_bot/deploy_super_exit.sh status"
}

cmd_tail_learn() {
    ssh_sc "tail -F ~/super-learn.out 2>/dev/null || tail -F ~/super-learn.err"
}

cmd_tail_actor() {
    local i="${1:-0}"
    ssh_sc "tail -F ~/super-actor-${i}.out 2>/dev/null || tail -F ~/super-actor-${i}.err"
}

cmd_pull_eval() {
    log "Pulling eval results -> ${LOCAL_REPO}/cluster_eval/"
    mkdir -p "${LOCAL_REPO}/cluster_eval"
    rsync -azP \
        -e "$RSYNC_RSH" \
        "${SSH_TARGET}:${CLUSTER_ROOT}/checkpoints/super_exit/eval/" \
        "${LOCAL_REPO}/cluster_eval/"
    log "Done."
}

cmd_ssh() {
    log "Opening interactive ssh -J ${CLUSTER_USER}@${JAMIE} ${SSH_TARGET}"
    exec ssh -t "${SSH_OPTS[@]}" "$SSH_TARGET"
}

cmd_help() {
    sed -n '1,/^# Required env/p' "$0" | sed 's/^# \?//'
    echo
    echo "Subcommands:"
    echo "  sync         rsync local repo -> cluster (via jamie)"
    echo "  seed         Copy seed M2 .pt to cluster init.pt"
    echo "  launch       sync + seed + launch-all on cluster"
    echo "  stop         Touch .stop on cluster (graceful)"
    echo "  reset-stop   Remove .stop on cluster"
    echo "  status       Print cluster-side status"
    echo "  tail-learn   tail -F learner output on cluster"
    echo "  tail-actor N tail -F actor N output on cluster"
    echo "  pull-eval    rsync eval JSONs back to local"
    echo "  ssh          Open interactive ssh sc (via jamie)"
    echo "  help         This message"
}

case "${1:-help}" in
    sync)        cmd_sync ;;
    seed)        cmd_seed ;;
    launch)      cmd_launch ;;
    stop)        cmd_stop ;;
    reset-stop)  cmd_reset_stop ;;
    status)      cmd_status ;;
    tail-learn)  cmd_tail_learn ;;
    tail-actor)  cmd_tail_actor "${2:-0}" ;;
    pull-eval)   cmd_pull_eval ;;
    ssh)         cmd_ssh ;;
    help|--help|-h|"") cmd_help ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
