#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# deploy.sh — Unified ExIt deployment for Stanford NLP cluster (JAG)
#
# M2 model (~1M params) + search with catanatron AB2 value function
# for expert guidance. Source-aware shard mixing across 3 actor types.
#
# Architecture:
#   1 GPU learner  — trains on shards, exports C weights, evals vs AB2
#   N GPU actors   — NN policy + ABt search + AB-leaf eval (strongest signal)
#   M CPU actors   — pure AB2 imitation (cheap, curriculum foundation)
#
# Usage (run FROM sc, the login node):
#
#   # ── STEP 1: Launch learner (1 jag GPU) ─────────────────────────
#   nlprun -q jag -g 1 -c 8 -r 80G -p standard \
#       -n exit-learn bash human_bot/deploy.sh learn
#
#   # ── STEP 2: Launch GPU actors (1 GPU each, 4 actors/node) ──────
#   # Node 1:
#   ACTOR_OFFSET=0  NUM_ACTORS=4 \
#   nlprun -q jag -g 1 -c 16 -r 60G -p standard \
#       -n exit-gpu-0 bash human_bot/deploy.sh actors-gpu
#
#   # Node 2:
#   ACTOR_OFFSET=10 NUM_ACTORS=4 \
#   nlprun -q jag -g 1 -c 16 -r 60G -p standard \
#       -n exit-gpu-1 bash human_bot/deploy.sh actors-gpu
#
#   # ── STEP 3: Launch CPU actors (no GPU, 40 workers on jag) ──────
#   # Node 1:
#   ACTOR_OFFSET=100 NUM_WORKERS=40 \
#   nlprun -q jag -g 0 -r 40G -c 48 -p standard \
#       -n exit-ab2-0 bash human_bot/deploy.sh actors-cpu
#
#   # Node 2 (optional):
#   ACTOR_OFFSET=200 NUM_WORKERS=40 \
#   nlprun -q jag -g 0 -r 40G -c 48 -p standard \
#       -n exit-ab2-1 bash human_bot/deploy.sh actors-cpu
#
#   # ── STOP everything gracefully ─────────────────────────────────
#   touch /nlp/scr/nroll/catan_training_big/checkpoints/exit_v2/.stop
#
#   # ── CHECK status ───────────────────────────────────────────────
#   bash human_bot/deploy.sh status
#
# Environment overrides (all optional):
#   CKPT           — seed checkpoint (default: checkpoints/ab2_imit_v1/latest.pt)
#   SHARD_DIR      — shard directory (default: data/exit_v2)
#   CKPT_DIR       — checkpoint directory (default: checkpoints/exit_v2)
#   WANDB_NAME     — W&B run name (default: exit_v2)
#   WANDB_API_KEY  — W&B API key (reads ~/.wandb_key if unset)
#   SEARCH_DEPTH   — ExIt search depth (default: 15)
#   TOP_K          — policy top-k pruning (default: 2)
#   PLAYER_COUNTS  — comma-separated (default: 2,3,4)
#   SOURCE_MIX     — learner shard mix (default: exit_vs_ab2:0.6,exit:0.25,ab2:0.15)
#   EVAL_GAMES     — eval games per checkpoint (default: 200)
#   NUM_ACTORS     — GPU actor processes per node (default: 4)
#   NUM_WORKERS    — CPU AB2 workers per node (default: auto)
#   ACTOR_OFFSET   — first actor ID (MUST be unique per node)
#   NUM_GPUS       — GPUs on this node (default: 1)
#   MAX_PENDING    — backpressure cap (default: 200)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT="/nlp/scr/nroll/catan_training_big"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ── Model config (M2 = ~1M params) ────────────────────────────────
export GNN_HIDDEN=80
export TRUNK_CHANNELS=192

# ── Stability flags for JAG nodes ─────────────────────────────────
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── W&B key: env → file → disabled (NEVER hardcoded) ─────────────
if [ -z "${WANDB_API_KEY:-}" ]; then
    if [ -f "$HOME/.wandb_key" ]; then
        export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
    else
        echo "[deploy] WARNING: No WANDB_API_KEY set and ~/.wandb_key not found."
        echo "[deploy]          W&B logging will be disabled."
    fi
fi

# ── Defaults ───────────────────────────────────────────────────────
: "${CKPT:=checkpoints/ab2_imit_v1/latest.pt}"
: "${SHARD_DIR:=data/exit_v2}"
: "${CKPT_DIR:=checkpoints/exit_v2}"
: "${WANDB_NAME:=exit_v2}"
: "${SEARCH_DEPTH:=15}"
: "${TOP_K:=2}"
: "${AB_DEPTH:=2}"
: "${MAX_PENDING:=200}"
: "${PLAYER_COUNTS:=2,3,4}"
: "${SOURCE_MIX:=exit_vs_ab2:0.6,exit:0.25,ab2:0.15}"
: "${EVAL_GAMES:=200}"
: "${NUM_ACTORS:=4}"
: "${NUM_WORKERS:=0}"
: "${NUM_GPUS:=1}"
: "${ACTOR_OFFSET:=0}"

STOP_FILE="$CKPT_DIR/.stop"

# ── Helpers ────────────────────────────────────────────────────────
build_libcatan() {
    echo "[deploy] Rebuilding libcatan on $(hostname)..."
    flock "$PROJECT_ROOT/.libcatan.lock" \
        python3 -m hexzero.bindings.build_lib
}

seed_checkpoint() {
    mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"
    if [ ! -f "$CKPT_DIR/latest.pt" ]; then
        if [ ! -f "$CKPT" ]; then
            echo "[deploy] ERROR: Seed checkpoint $CKPT not found" >&2
            exit 1
        fi
        cp "$CKPT" "$CKPT_DIR/latest.pt"
        echo "[deploy] Seeded $CKPT_DIR/latest.pt from $CKPT"
    fi
    if [ ! -f "$CKPT_DIR/init.pt" ]; then
        cp "$CKPT" "$CKPT_DIR/init.pt" 2>/dev/null || true
    fi
    rm -f "$STOP_FILE"
}

wait_for_file() {
    local path="$1" timeout="${2:-300}"
    echo "[deploy] Waiting up to ${timeout}s for $path..."
    for _ in $(seq 1 "$timeout"); do
        [ -f "$path" ] && return 0
        sleep 1
    done
    echo "[deploy] ERROR: $path not found after ${timeout}s" >&2
    return 1
}

print_banner() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ExIt deployment: $1"
    echo "  Host:        $(hostname)"
    echo "  Checkpoint:  $CKPT_DIR/latest.pt"
    echo "  Shards:      $SHARD_DIR"
    echo "  Player cnts: $PLAYER_COUNTS"
    echo "  Stop file:   $STOP_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Subcommands ────────────────────────────────────────────────────
cmd_learn() {
    print_banner "GPU LEARNER"
    echo "  Source mix:  $SOURCE_MIX"
    echo "  Eval games:  $EVAL_GAMES"
    echo "  W&B:         ${WANDB_NAME} (key=${WANDB_API_KEY:+set}${WANDB_API_KEY:-unset})"
    echo ""

    export OMP_NUM_THREADS=4
    build_libcatan
    seed_checkpoint

    echo "[deploy] Running pre-flight gates..."
    python3 -m human_bot.run_preflight \
        --ckpt "$CKPT_DIR/init.pt" \
        --eval-games 60

    exec python3 -u human_bot/c_selfplay.py \
        --role learner \
        --checkpoint "$CKPT_DIR/latest.pt" \
        --shard-dir "$SHARD_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        --player-counts "$PLAYER_COUNTS" \
        --batch-size 4096 \
        --shards-per-train 10 \
        --eval-games "$EVAL_GAMES" \
        --eval-interval 4 \
        --source-mix "$SOURCE_MIX" \
        --wandb-name "$WANDB_NAME"
}

cmd_actors_gpu() {
    print_banner "GPU ACTORS (exit_vs_ab2)"
    echo "  Actors:      $NUM_ACTORS on $NUM_GPUS GPU(s)"
    echo "  Offset:      $ACTOR_OFFSET"
    echo "  Search:      depth=$SEARCH_DEPTH top_k=$TOP_K (AB-leaf)"
    echo ""

    export OMP_NUM_THREADS=2
    build_libcatan

    mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"
    wait_for_file "$CKPT_DIR/latest.pt" 300

    exec python3 -u human_bot/exit_vs_ab2_actors.py \
        --checkpoint "$CKPT_DIR/latest.pt" \
        --shard-dir "$SHARD_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        --num-actors "$NUM_ACTORS" \
        --num-gpus "$NUM_GPUS" \
        --actor-id-offset "$ACTOR_OFFSET" \
        --search-depth "$SEARCH_DEPTH" \
        --top-k "$TOP_K" \
        --max-pending "$MAX_PENDING" \
        --player-counts "$PLAYER_COUNTS"
}

cmd_actors_gpu_selfplay() {
    print_banner "GPU ACTORS (exit self-play)"
    echo "  Actors:      $NUM_ACTORS on $NUM_GPUS GPU(s)"
    echo "  Offset:      $ACTOR_OFFSET"
    echo "  Search:      depth=$SEARCH_DEPTH top_k=$TOP_K (AB-leaf)"
    echo ""

    export OMP_NUM_THREADS=2
    build_libcatan

    mkdir -p "$CKPT_DIR" "$SHARD_DIR/pending"
    wait_for_file "$CKPT_DIR/latest.pt" 300

    exec python3 -u human_bot/exit_gpu_actors.py \
        --checkpoint "$CKPT_DIR/latest.pt" \
        --shard-dir "$SHARD_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        --num-actors "$NUM_ACTORS" \
        --num-gpus "$NUM_GPUS" \
        --actor-id-offset "$ACTOR_OFFSET" \
        --search-depth "$SEARCH_DEPTH" \
        --top-k "$TOP_K" \
        --max-pending "$MAX_PENDING" \
        --player-counts "$PLAYER_COUNTS"
}

cmd_actors_cpu() {
    print_banner "CPU ACTORS (AB2 imitation)"
    if [ "$NUM_WORKERS" = "0" ]; then
        NUM_WORKERS=$(($(nproc) - 2))
        [ "$NUM_WORKERS" -lt 1 ] && NUM_WORKERS=1
    fi
    echo "  Workers:     $NUM_WORKERS (CPU)"
    echo "  Offset:      $ACTOR_OFFSET"
    echo "  AB depth:    $AB_DEPTH"
    echo ""

    export OMP_NUM_THREADS=1
    build_libcatan

    mkdir -p "$SHARD_DIR/pending"

    exec python3 -u human_bot/ab2_stream.py \
        --shard-dir "$SHARD_DIR" \
        --num-workers "$NUM_WORKERS" \
        --depth "$AB_DEPTH" \
        --max-pending "$MAX_PENDING" \
        --player-counts "$PLAYER_COUNTS" \
        --actor-id-offset "$ACTOR_OFFSET"
}

cmd_status() {
    echo "━━━ ExIt Deployment Status ━━━"
    echo ""

    if [ -f "$STOP_FILE" ]; then
        echo "STOP FILE: EXISTS ($STOP_FILE) — graceful shutdown requested"
    else
        echo "STOP FILE: not present (system running)"
    fi
    echo ""

    echo "── Pending shards ──"
    if [ -d "$SHARD_DIR/pending" ]; then
        local total=$(find "$SHARD_DIR/pending" -name "*.pt" ! -name "*.tmp" | wc -l)
        local ab2=$(find "$SHARD_DIR/pending" -name "ab2_*.pt" | wc -l)
        local exit_va2=$(find "$SHARD_DIR/pending" -name "exit_vs_ab2_*.pt" | wc -l)
        local exit_sp=$(find "$SHARD_DIR/pending" -name "exit_a*.pt" | wc -l)
        echo "  Total: $total  (ab2=$ab2  exit_vs_ab2=$exit_va2  exit=$exit_sp)"
    else
        echo "  (pending dir does not exist)"
    fi
    echo ""

    echo "── Checkpoint ──"
    if [ -f "$CKPT_DIR/latest.pt" ]; then
        ls -lh "$CKPT_DIR/latest.pt"
    else
        echo "  (no latest.pt)"
    fi
    if [ -f "$CKPT_DIR/.round" ]; then
        echo "  Round: $(cat "$CKPT_DIR/.round")"
    fi
    echo ""

    echo "── Actor stats ──"
    if [ -d "$CKPT_DIR/.actor_stats" ]; then
        python3 -c "
import json, os, glob
stats_dir = '$CKPT_DIR/.actor_stats'
total_gps = 0
for f in sorted(glob.glob(os.path.join(stats_dir, '*.json'))):
    try:
        s = json.load(open(f))
        gps = s.get('gps', 0)
        total_gps += gps
        name = os.path.basename(f).replace('.json','')
        print(f'  {name}: {s.get(\"games\",0)} games, {gps:.2f} g/s')
    except: pass
print(f'  ── Total: {total_gps:.1f} games/sec across all actors')
" 2>/dev/null || echo "  (could not read actor stats)"
    else
        echo "  (no actor stats yet)"
    fi
    echo ""

    echo "── SLURM jobs ──"
    squeue -u "$(whoami)" -o "%.8i %.20j %.8T %.10M %.4C %.4D %R" 2>/dev/null || \
        echo "  (squeue not available — run from sc)"
}

cmd_stop() {
    echo "[deploy] Creating stop file: $STOP_FILE"
    mkdir -p "$CKPT_DIR"
    touch "$STOP_FILE"
    echo "[deploy] All workers will stop after finishing their current game."
    echo "[deploy] To resume, delete the stop file:"
    echo "         rm $STOP_FILE"
}

# ── Quick-launch helpers (run these from sc) ──────────────────────
cmd_launch_all() {
    echo "━━━ Quick Launch: Full ExIt Pipeline ━━━"
    echo ""
    echo "Submitting 4 SLURM jobs..."
    echo ""

    rm -f "$STOP_FILE"

    echo "1/4  Learner (1 GPU, jag-standard)..."
    nlprun -q jag -g 1 -c 8 -r 80G -p standard \
        -n exit-learn bash human_bot/deploy.sh learn &
    sleep 2

    echo "2/4  GPU actors node 0 (1 GPU, 4 actors)..."
    ACTOR_OFFSET=0 NUM_ACTORS=4 \
    nlprun -q jag -g 1 -c 16 -r 60G -p standard \
        -n exit-gpu-0 bash human_bot/deploy.sh actors-gpu &
    sleep 1

    echo "3/4  GPU actors node 1 (1 GPU, 4 actors)..."
    ACTOR_OFFSET=10 NUM_ACTORS=4 \
    nlprun -q jag -g 1 -c 16 -r 60G -p standard \
        -n exit-gpu-1 bash human_bot/deploy.sh actors-gpu &
    sleep 1

    echo "4/4  CPU actors (AB2, 40 workers on jag)..."
    ACTOR_OFFSET=100 NUM_WORKERS=40 \
    nlprun -q jag -g 0 -r 40G -c 48 -p standard \
        -n exit-ab2-0 bash human_bot/deploy.sh actors-cpu &

    echo ""
    echo "All jobs submitted. Monitor with:"
    echo "  bash human_bot/deploy.sh status"
    echo "  squeue -u $(whoami)"
    echo ""
    echo "Stop gracefully:"
    echo "  bash human_bot/deploy.sh stop"
}

# ── Dispatch ───────────────────────────────────────────────────────
case "${1:-help}" in
    learn)             cmd_learn ;;
    actors-gpu)        cmd_actors_gpu ;;
    actors-gpu-sp)     cmd_actors_gpu_selfplay ;;
    actors-cpu)        cmd_actors_cpu ;;
    status)            cmd_status ;;
    stop)              cmd_stop ;;
    launch-all)        cmd_launch_all ;;
    help|--help|-h)
        echo "Usage: bash human_bot/deploy.sh <command>"
        echo ""
        echo "Commands (run on compute nodes via nlprun):"
        echo "  learn           GPU learner (1 GPU)"
        echo "  actors-gpu      GPU ExIt actors: NN+search vs AB2 opponents"
        echo "  actors-gpu-sp   GPU ExIt actors: NN+search self-play (all seats)"
        echo "  actors-cpu      CPU AB2 imitation actors (no GPU)"
        echo ""
        echo "Commands (run from sc):"
        echo "  launch-all      Submit learner + 2 GPU actor nodes + 1 CPU actor node"
        echo "  status          Show pipeline status"
        echo "  stop            Create stop file for graceful shutdown"
        echo ""
        echo "See top of script for full environment variable documentation."
        ;;
    *)
        echo "[deploy] Unknown command: $1" >&2
        echo "Run 'bash human_bot/deploy.sh help' for usage." >&2
        exit 1
        ;;
esac
