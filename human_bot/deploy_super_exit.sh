#!/usr/bin/env bash
# deploy_super_exit.sh — unified launcher for super_m2 ExIt on Stanford NLP cluster.
#
# Architecture:
#   1 learner job  : 1 GPU, 8 CPUs, 80G RAM (jagupard)
#                    — picks up actor shards, trains M2 policy + value head,
#                      exports to nn_weights_latest.bin, runs eval every 10 rounds.
#   4 actor jobs   : 1 GPU + 16 CPUs + 60G RAM each (jagupard)
#                    — 4-way super_m2 self-play (all 4 seats), records every
#                      decision, writes shards atomically to data/super_exit/pending/.
#                    GPU is allocated only because jagupard machines have many
#                    CPU cores; super_m2's inner loop is C+BLAS, not CUDA.
#
# Subcommands:
#   launch-all   Submit learner + 4 actors via nlprun (one shell call).
#   launch-learn Submit only the learner.
#   launch-actor Submit one actor (uses ACTOR_ID env var; defaults to 0).
#   learn        Run learner inline (called from inside nlprun).
#   actor        Run one actor inline (uses ACTOR_ID env var).
#   stop         Touch the .stop file. Workers exit gracefully.
#   reset-stop   Remove the .stop file (use before re-launching).
#   status       Print pending shards, latest round, W&B id, recent log.
#   help         This message.
#
# Required env (or sane defaults):
#   PROJECT_ROOT        Path on cluster scratch  (default /nlp/scr/nroll/catan_super_exit)
#   WANDB_API_KEY       W&B key (else read from ~/.wandb_key, else W&B disabled)
#   WANDB_PROJECT       (default human-bot-super-exit)
#   WANDB_NAME          (default super-exit-MMDD-HHMM)
#   SEED_CKPT           Initial M2 .pt (default $PROJECT_ROOT/checkpoints/super_exit/init.pt)
#   N_ACTORS            Number of actor nodes (default 4)
#
# Usage on the cluster:
#   sc$ cd /nlp/scr/nroll/catan_super_exit
#   sc$ bash human_bot/deploy_super_exit.sh launch-all
#   sc$ bash human_bot/deploy_super_exit.sh status
#   sc$ bash human_bot/deploy_super_exit.sh stop          # graceful
#   sc$ bash human_bot/deploy_super_exit.sh reset-stop    # before re-launch

set -euo pipefail

# ──────────────── paths & env ───────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/nlp/scr/nroll/catan_super_exit}"
CKPT_DIR="${CKPT_DIR:-$PROJECT_ROOT/checkpoints/super_exit}"
SHARD_DIR="${SHARD_DIR:-$PROJECT_ROOT/data/super_exit}"
WEIGHTS_BIN="${WEIGHTS_BIN:-$CKPT_DIR/nn_weights_latest.bin}"
BASELINE_BIN="${BASELINE_BIN:-$CKPT_DIR/nn_weights_baseline.bin}"
SEED_CKPT="${SEED_CKPT:-$CKPT_DIR/init.pt}"
STOP_FILE="$CKPT_DIR/.stop"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

WANDB_PROJECT="${WANDB_PROJECT:-human-bot-super-exit}"
WANDB_NAME="${WANDB_NAME:-super-exit-$(date +%m%d-%H%M)}"

N_ACTORS="${N_ACTORS:-4}"

# Actor allocation: jagupard nodes for everything (jag queue), with
# GPU optional. super_m2's inner loop is C+BLAS — CUDA isn't used —
# so default ACTOR_GPU=0 to leave GPUs for the learner / other users.
ACTOR_QUEUE="${ACTOR_QUEUE:-jag}"
ACTOR_GPU="${ACTOR_GPU:-0}"           # 0 = CPU-only on jagupard
ACTOR_CORES="${ACTOR_CORES:-32}"      # CPU cores per actor node
ACTOR_RAM="${ACTOR_RAM:-60G}"

# Learner allocation
LEARNER_QUEUE="${LEARNER_QUEUE:-jag}"
LEARNER_GPU="${LEARNER_GPU:-1}"
LEARNER_CORES="${LEARNER_CORES:-8}"
LEARNER_RAM="${LEARNER_RAM:-80G}"

# Super_m2 + training defaults
SUPER_DEPTH="${SUPER_DEPTH:-6}"
SUPER_K_SCHEDULE="${SUPER_K_SCHEDULE:-12,8,6,5,4,3}"
SUPER_TIME_MS="${SUPER_TIME_MS:-4000}"
# parallel_games: how many concurrent games per actor process. Each game
# uses 4 super_m2 bot instances but only 1 is active per turn, so with
# OPENBLAS_NUM_THREADS=1 we can pack as many games as cores. Default
# leaves a few cores headroom for OS + Python overhead.
PARALLEL_GAMES="${PARALLEL_GAMES:-$((ACTOR_CORES > 4 ? ACTOR_CORES - 4 : ACTOR_CORES))}"
GAMES_PER_SHARD="${GAMES_PER_SHARD:-8}"
MAX_PENDING="${MAX_PENDING:-64}"
DENSE_SIGNAL="${DENSE_SIGNAL:-1}"      # 1 = use dense soft-target shards
SHARDS_PER_TRAIN="${SHARDS_PER_TRAIN:-4}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-1e-5}"
SEARCH_VALUE_WEIGHT="${SEARCH_VALUE_WEIGHT:-0.5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
EVAL_GAMES="${EVAL_GAMES:-20}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
AB_DEPTH="${AB_DEPTH:-2}"

# M2 model dimensions (must match the seed checkpoint)
GNN_HIDDEN="${GNN_HIDDEN:-80}"
TRUNK_CHANNELS="${TRUNK_CHANNELS:-192}"

# ──────────────── helpers ───────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

ensure_dirs() {
    mkdir -p "$PROJECT_ROOT" "$CKPT_DIR" "$SHARD_DIR/pending" "$LOG_DIR"
}

setup_python_env() {
    cd "$PROJECT_ROOT"
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export GNN_HIDDEN
    export TRUNK_CHANNELS
    export MKL_SERVICE_FORCE_INTEL="${MKL_SERVICE_FORCE_INTEL:-1}"
    export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"

    # W&B key resolution
    if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.wandb_key" ]]; then
        export WANDB_API_KEY="$(cat "$HOME/.wandb_key")"
        log "Loaded WANDB_API_KEY from ~/.wandb_key"
    fi
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        log "WARNING: WANDB_API_KEY not set; W&B logging disabled."
    fi
}

build_libcatan() {
    # Rebuild libcatan if needed (for hexzero.bindings + alphabeta_search).
    # flock so multiple jobs don't race on the build.
    cd "$PROJECT_ROOT"
    flock -n "$PROJECT_ROOT/.libcatan.lock" \
        python3 -m hexzero.bindings.build_lib \
        || log "libcatan build skipped (lock held or already built)"
}

build_libnn() {
    # Build libnn.so + libdeep.so for the C inference path used by actors and eval.
    # macOS variants (.dylib) are committed to git; Linux .so files are not, so
    # we build them here on the cluster. flock so multiple jobs don't race.
    cd "$PROJECT_ROOT"
    local host
    host="$(hostname -s)"
    local libnn_path="$PROJECT_ROOT/csrc/libnn_${host}.so"
    local libnn_generic="$PROJECT_ROOT/csrc/libnn.so"
    local libdeep_path="$PROJECT_ROOT/csrc/libdeep.so"

    # CC + CFLAGS (Linux jagupard)
    local CC_BIN="${CC:-cc}"
    local CFLAGS="-O3 -march=native -flto -ffast-math -funroll-loops -fPIC -shared -I$PROJECT_ROOT/csrc"
    CFLAGS="$CFLAGS -DNN_GNN_HIDDEN=$GNN_HIDDEN -DNN_TRUNK_CH=$TRUNK_CHANNELS"
    local LDFLAGS="-lm"
    if pkg-config --exists openblas 2>/dev/null; then
        CFLAGS="$CFLAGS -DHAVE_CBLAS $(pkg-config --cflags openblas)"
        LDFLAGS="$LDFLAGS $(pkg-config --libs openblas)"
    elif [ -f /usr/include/cblas.h ] || [ -f /usr/include/openblas/cblas.h ]; then
        CFLAGS="$CFLAGS -DHAVE_CBLAS"
        LDFLAGS="$LDFLAGS -lopenblas"
    fi

    (
        flock -n 9 || { log "libnn build lock held; skipping (another job is building)"; exit 0; }
        if [[ ! -f "$libnn_path" && ! -f "$libnn_generic" ]]; then
            log "Building $libnn_generic (host $host, GNN_HIDDEN=$GNN_HIDDEN TRUNK_CH=$TRUNK_CHANNELS)..."
            $CC_BIN $CFLAGS $PROJECT_ROOT/csrc/nn.c -o "$libnn_generic" $LDFLAGS
            log "  -> $(ls -lh "$libnn_generic" | awk '{print $5}')"
        fi
        if [[ ! -f "$libdeep_path" ]]; then
            log "Building $libdeep_path..."
            $CC_BIN $CFLAGS \
                $PROJECT_ROOT/csrc/nn.c \
                $PROJECT_ROOT/csrc/state_encode.c \
                $PROJECT_ROOT/csrc/policy_topk.c \
                $PROJECT_ROOT/csrc/deep_search.c \
                $PROJECT_ROOT/csrc/board.c \
                $PROJECT_ROOT/csrc/value.c \
                $PROJECT_ROOT/csrc/search.c \
                $PROJECT_ROOT/csrc/state.c \
                $PROJECT_ROOT/csrc/actions.c \
                $PROJECT_ROOT/csrc/apply_action.c \
                $PROJECT_ROOT/csrc/game.c \
                $PROJECT_ROOT/csrc/map.c \
                $PROJECT_ROOT/csrc/rng.c \
                -o "$libdeep_path" $LDFLAGS
            log "  -> $(ls -lh "$libdeep_path" | awk '{print $5}')"
        fi
    ) 9>"$PROJECT_ROOT/.libnn.lock"
}

# ──────────────── inline runners ────────────────────────────────────────
cmd_learn() {
    ensure_dirs
    setup_python_env
    build_libcatan
    build_libnn

    if [[ ! -f "$SEED_CKPT" && ! -f "$CKPT_DIR/latest.pt" ]]; then
        log "FATAL: no seed checkpoint at $SEED_CKPT and no resume at $CKPT_DIR/latest.pt"
        exit 1
    fi

    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

    log "Launching learner (round file: $CKPT_DIR/.round)"
    cd "$PROJECT_ROOT"
    exec python3 -u -m human_bot.super_learner \
        --checkpoint "$SEED_CKPT" \
        --weights-bin "$WEIGHTS_BIN" \
        --baseline-bin "$BASELINE_BIN" \
        --shard-dir "$SHARD_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        --shards-per-train "$SHARDS_PER_TRAIN" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --search-value-weight "$SEARCH_VALUE_WEIGHT" \
        --eval-interval "$EVAL_INTERVAL" \
        --eval-games "$EVAL_GAMES" \
        --eval-workers "$EVAL_WORKERS" \
        --ab-depth "$AB_DEPTH" \
        --wandb-project "$WANDB_PROJECT" \
        --wandb-name "$WANDB_NAME"
}

cmd_actor() {
    ensure_dirs
    setup_python_env
    build_libcatan
    build_libnn

    local actor_id="${ACTOR_ID:-0}"
    # Force single-threaded BLAS so we can pack many games per actor
    # without thread-pool contention. mp.Pool gives us the parallelism.
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

    local dense_flag=""
    if [[ "$DENSE_SIGNAL" == "1" ]]; then
        dense_flag="--dense"
    fi

    log "Launching actor $actor_id (parallel_games=$PARALLEL_GAMES, dense=$DENSE_SIGNAL)"
    cd "$PROJECT_ROOT"
    exec python3 -u -m human_bot.super_actor \
        --actor-id "$actor_id" \
        --weights-bin "$WEIGHTS_BIN" \
        --shard-dir "$SHARD_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        --parallel-games "$PARALLEL_GAMES" \
        --games-per-shard "$GAMES_PER_SHARD" \
        --depth "$SUPER_DEPTH" \
        --k-schedule "$SUPER_K_SCHEDULE" \
        --time-ms "$SUPER_TIME_MS" \
        --max-pending "$MAX_PENDING" \
        $dense_flag \
        --seed-base "$((400000 + actor_id * 100000))"
}

# ──────────────── nlprun submitters ─────────────────────────────────────
submit_learn() {
    ensure_dirs
    log "Submitting learner ($LEARNER_QUEUE queue, $LEARNER_GPU GPU, $LEARNER_CORES CPUs, $LEARNER_RAM)..."
    nlprun -q "$LEARNER_QUEUE" -g "$LEARNER_GPU" \
        -c "$LEARNER_CORES" -r "$LEARNER_RAM" -p standard \
        -n super-learn \
        "bash $PROJECT_ROOT/human_bot/deploy_super_exit.sh learn"
}

submit_actor() {
    ensure_dirs
    local actor_id="${1:-0}"
    log "Submitting actor $actor_id ($ACTOR_QUEUE queue, $ACTOR_GPU GPU, $ACTOR_CORES CPUs, $ACTOR_RAM)..."
    ACTOR_ID="$actor_id" nlprun -q "$ACTOR_QUEUE" -g "$ACTOR_GPU" \
        -c "$ACTOR_CORES" -r "$ACTOR_RAM" -p standard \
        -n "super-actor-$actor_id" \
        "ACTOR_ID=$actor_id bash $PROJECT_ROOT/human_bot/deploy_super_exit.sh actor"
}

cmd_launch_all() {
    ensure_dirs
    setup_python_env

    if [[ -f "$STOP_FILE" ]]; then
        log "Removing stale stop file $STOP_FILE"
        rm -f "$STOP_FILE"
    fi

    if [[ ! -f "$SEED_CKPT" ]]; then
        log "FATAL: SEED_CKPT not found: $SEED_CKPT"
        log "  Place an initial M2 .pt there, e.g.:"
        log "    cp /nlp/scr/nroll/catan_training_big/checkpoints/.../latest.pt $SEED_CKPT"
        exit 1
    fi

    # Throughput estimate
    # Each 4-way game ≈ 300s of single-core CPU. With OPENBLAS_NUM_THREADS=1
    # and PARALLEL_GAMES games per actor, throughput per actor is roughly
    # PARALLEL_GAMES games / 300s = PARALLEL_GAMES * 12 games/hour.
    local total_parallel=$((PARALLEL_GAMES * N_ACTORS))
    local games_per_hour=$((total_parallel * 12))
    local steps_per_hour_est=$((games_per_hour * 140))  # ~140 super_m2 decisions per game
    local rounds_per_hour=$((games_per_hour / (SHARDS_PER_TRAIN * GAMES_PER_SHARD)))
    if [[ "$rounds_per_hour" -lt 1 ]]; then rounds_per_hour=1; fi

    log "============================================================"
    log "Launching super_m2 ExIt deployment"
    log "  PROJECT_ROOT:   $PROJECT_ROOT"
    log "  SEED_CKPT:      $SEED_CKPT"
    log "  CKPT_DIR:       $CKPT_DIR"
    log "  SHARD_DIR:      $SHARD_DIR"
    log "  WEIGHTS_BIN:    $WEIGHTS_BIN"
    log "  BASELINE_BIN:   $BASELINE_BIN"
    log "  W&B:            project=$WANDB_PROJECT name=$WANDB_NAME"
    log "  Actors:         $N_ACTORS x ($ACTOR_QUEUE queue, $ACTOR_GPU GPU, $ACTOR_CORES CPU, $ACTOR_RAM)"
    log "                  $PARALLEL_GAMES games/actor concurrently = $total_parallel total"
    log "                  dense signal: $DENSE_SIGNAL"
    log "  Learner:        $LEARNER_QUEUE queue, $LEARNER_GPU GPU, $LEARNER_CORES CPU, $LEARNER_RAM"
    log "  Training:       shards_per_train=$SHARDS_PER_TRAIN, batch=$BATCH_SIZE, lr=$LR"
    log "                  search_value_weight=$SEARCH_VALUE_WEIGHT (auxiliary V(s) loss)"
    log "  Eval:           every $EVAL_INTERVAL rounds, $EVAL_GAMES games each"
    log "                  (vs AB$AB_DEPTH and vs frozen-baseline 0-ply)"
    log "  Throughput est: ~$games_per_hour games/hr,"
    log "                  ~$steps_per_hour_est super_m2 decisions/hr,"
    log "                  ~$rounds_per_hour train rounds/hr"
    log "                  -> eval roughly every $((60 * EVAL_INTERVAL / rounds_per_hour)) min"
    log "============================================================"

    submit_learn
    log "Sleeping 10s to give learner head start..."
    sleep 10

    for i in $(seq 0 $((N_ACTORS - 1))); do
        submit_actor "$i"
        sleep 2
    done

    log "============================================================"
    log "All jobs submitted."
    log "Monitor: bash $PROJECT_ROOT/human_bot/deploy_super_exit.sh status"
    log "         squeue -u \$USER"
    log "         tail -f ~/super-learn.out  ~/super-actor-*.out"
    log "Stop:    bash $PROJECT_ROOT/human_bot/deploy_super_exit.sh stop"
    log "         (workers exit gracefully after the current shard)"
    log "============================================================"
}

cmd_stop() {
    ensure_dirs
    log "Touching $STOP_FILE — workers will exit after current game/shard"
    touch "$STOP_FILE"
}

cmd_reset_stop() {
    if [[ -f "$STOP_FILE" ]]; then
        rm -f "$STOP_FILE"
        log "Removed $STOP_FILE"
    else
        log "No stop file present at $STOP_FILE"
    fi
}

cmd_status() {
    ensure_dirs
    setup_python_env
    echo "--- super_m2 ExIt status ---"
    echo "  PROJECT_ROOT:    $PROJECT_ROOT"
    echo "  CKPT_DIR:        $CKPT_DIR"
    echo "  SHARD_DIR:       $SHARD_DIR"
    if [[ -f "$CKPT_DIR/.round" ]]; then
        echo "  Latest round:    $(cat "$CKPT_DIR/.round")"
    else
        echo "  Latest round:    (not started)"
    fi
    if [[ -f "$CKPT_DIR/.wandb_id" ]]; then
        echo "  W&B run id:      $(cat "$CKPT_DIR/.wandb_id")"
    fi
    if [[ -f "$STOP_FILE" ]]; then
        echo "  STOP FILE:       PRESENT — workers will exit"
    else
        echo "  STOP FILE:       absent"
    fi
    if [[ -f "$WEIGHTS_BIN" ]]; then
        echo "  weights mtime:   $(date -r "$WEIGHTS_BIN" '+%Y-%m-%d %H:%M:%S')"
    fi
    if [[ -d "$SHARD_DIR/pending" ]]; then
        local n
        n=$(ls "$SHARD_DIR/pending"/*.pt 2>/dev/null | wc -l | tr -d ' ')
        echo "  pending shards:  $n"
    fi
    echo
    echo "--- squeue -u $USER ---"
    squeue -u "$USER" 2>/dev/null || echo "  (squeue not available here)"
    echo
    echo "--- recent eval results ---"
    if [[ -d "$CKPT_DIR/eval" ]]; then
        ls -t "$CKPT_DIR/eval"/*.json 2>/dev/null | head -3 | while read -r f; do
            echo "  $f"
            python3 -c "
import json, sys
d = json.load(open('$f'))
ab = d['vs_ab2']; nn = d['vs_0ply']
print(f'    round={d[\"round\"]}'
      f'  vs_AB2={ab[\"wins\"]}/{ab[\"n_games\"]} ({100*ab[\"winrate\"]:.1f}%) rank={ab[\"avg_rank\"]:.2f}'
      f'  vs_0ply={nn[\"wins\"]}/{nn[\"n_games\"]} ({100*nn[\"winrate\"]:.1f}%) rank={nn[\"avg_rank\"]:.2f}')" \
                2>/dev/null || true
        done
    else
        echo "  (none yet)"
    fi
}

cmd_help() {
    sed -n '1,/^# Required env/p' "$0" | sed 's/^# \?//'
    echo
    echo "Subcommands:"
    echo "  launch-all     Submit learner + $N_ACTORS actors via nlprun"
    echo "  launch-learn   Submit only the learner"
    echo "  launch-actor   Submit one actor (uses ACTOR_ID env var)"
    echo "  learn          Run learner inline (called from inside nlprun)"
    echo "  actor          Run one actor inline (uses ACTOR_ID env var)"
    echo "  stop           Touch .stop file (graceful shutdown)"
    echo "  reset-stop     Remove .stop file (before re-launch)"
    echo "  status         Print pending shards, latest round, recent evals"
    echo "  help           This message"
}

# ──────────────── dispatch ──────────────────────────────────────────────
case "${1:-help}" in
    launch-all)        cmd_launch_all ;;
    launch-learn)      submit_learn ;;
    launch-actor)      submit_actor "${2:-${ACTOR_ID:-0}}" ;;
    learn)             cmd_learn ;;
    actor)             cmd_actor ;;
    stop)              cmd_stop ;;
    reset-stop)        cmd_reset_stop ;;
    status)            cmd_status ;;
    help|--help|-h|"") cmd_help ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
