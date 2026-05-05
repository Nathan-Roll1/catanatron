#!/usr/bin/env bash
set -euo pipefail

# Local single-game Super M2 deployment.
# Uses multiple worker processes for one move's root candidates, never multiple games.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

WEIGHTS="${WEIGHTS:-$PROJECT_ROOT/csrc/nn_weights_m2.bin}"
MODE="${MODE:-1v3}"
DEPTH="${DEPTH:-6}"
K_SCHEDULE="${K_SCHEDULE:-12,8,6,5,4,3}"
TIME_MS="${TIME_MS:-4000}"
SEED_BASE="${SEED_BASE:-95000}"
WORKERS="${WORKERS:-0}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-1}"
BUILD_NATIVE="${BUILD_NATIVE:-1}"
REBUILD_NATIVE="${REBUILD_NATIVE:-0}"
PROFILE="${PROFILE:-0}"
BACKEND="${BACKEND:-c2}"

for arg in "$@"; do
  case "$arg" in
    --games|--games=*)
      echo "deploy_local_super_m2.sh always runs exactly one game; do not pass --games." >&2
      exit 2
      ;;
  esac
done

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The game-level parallelism is fixed at one; keep native math from
# oversubscribing each root-candidate worker.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_WORKER}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$THREADS_PER_WORKER}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights file: $WEIGHTS" >&2
  echo "Export or copy the M2 weights to csrc/nn_weights_m2.bin, or set WEIGHTS=/path/to/model.bin." >&2
  exit 1
fi

lib_ext="so"
nn_lib="$PROJECT_ROOT/csrc/libnn.so"
deep_lib="$PROJECT_ROOT/csrc/libdeep.so"
if [[ "$(uname -s)" == "Darwin" ]]; then
  lib_ext="dylib"
  nn_lib="$PROJECT_ROOT/csrc/libnn.dylib"
  deep_lib="$PROJECT_ROOT/csrc/libdeep.dylib"
fi

if [[ "$BUILD_NATIVE" == "1" ]]; then
  "$PYTHON_BIN" -m hexzero.bindings.build_lib >/dev/null

  if [[ "$REBUILD_NATIVE" == "1" || ! -f "$nn_lib" ]]; then
    if [[ "$lib_ext" == "dylib" ]]; then
      cc -shared -O3 -march=native -flto -fPIC \
        -o "$nn_lib" "$PROJECT_ROOT/csrc/nn.c" -lm -framework Accelerate
    else
      cc -shared -O3 -march=native -flto -fPIC -DHAVE_CBLAS \
        -o "$nn_lib" "$PROJECT_ROOT/csrc/nn.c" -lm -lopenblas
    fi
  fi

  if [[ "$REBUILD_NATIVE" == "1" || ! -f "$deep_lib" ]]; then
    if [[ "$lib_ext" == "dylib" ]]; then
      cc -shared -O3 -march=native -flto -ffast-math -funroll-loops -fPIC \
        -I"$PROJECT_ROOT/csrc" \
        -o "$deep_lib" \
        "$PROJECT_ROOT/csrc/nn.c" \
        "$PROJECT_ROOT/csrc/state_encode.c" \
        "$PROJECT_ROOT/csrc/policy_topk.c" \
        "$PROJECT_ROOT/csrc/deep_search.c" \
        "$PROJECT_ROOT/csrc/board.c" \
        "$PROJECT_ROOT/csrc/value.c" \
        "$PROJECT_ROOT/csrc/search.c" \
        "$PROJECT_ROOT/csrc/state.c" \
        "$PROJECT_ROOT/csrc/actions.c" \
        "$PROJECT_ROOT/csrc/apply_action.c" \
        "$PROJECT_ROOT/csrc/game.c" \
        "$PROJECT_ROOT/csrc/map.c" \
        "$PROJECT_ROOT/csrc/rng.c" \
        -lm -framework Accelerate
    else
      cc -shared -O3 -march=native -flto -ffast-math -funroll-loops -fPIC -DHAVE_CBLAS \
        -I"$PROJECT_ROOT/csrc" \
        -o "$deep_lib" \
        "$PROJECT_ROOT/csrc/nn.c" \
        "$PROJECT_ROOT/csrc/state_encode.c" \
        "$PROJECT_ROOT/csrc/policy_topk.c" \
        "$PROJECT_ROOT/csrc/deep_search.c" \
        "$PROJECT_ROOT/csrc/board.c" \
        "$PROJECT_ROOT/csrc/value.c" \
        "$PROJECT_ROOT/csrc/search.c" \
        "$PROJECT_ROOT/csrc/state.c" \
        "$PROJECT_ROOT/csrc/actions.c" \
        "$PROJECT_ROOT/csrc/apply_action.c" \
        "$PROJECT_ROOT/csrc/game.c" \
        "$PROJECT_ROOT/csrc/map.c" \
        "$PROJECT_ROOT/csrc/rng.c" \
        -lm -lopenblas
    fi
  fi
fi

echo "Super M2 local deployment"
echo "  one game: yes"
echo "  weights:  $WEIGHTS"
echo "  workers:  $WORKERS (0 = auto min(CPUs, root K))"
echo "  depth:    $DEPTH"
echo "  schedule: $K_SCHEDULE"
echo "  budget:   ${TIME_MS}ms/decision"
echo "  mode:     $MODE"
echo "  profile:  $PROFILE"
echo "  backend:  $BACKEND"
echo "  C stats env:   ${CATAN_MEASURE_C_STATS:-0} (1 adds --measure-c-stats)"
echo

cmd=("$PYTHON_BIN" "$PROJECT_ROOT/human_bot/superbot_v3_parallel.py"
  --single-game \
  --weights "$WEIGHTS" \
  --games 1 \
  --workers "$WORKERS" \
  --depth "$DEPTH" \
  --k-schedule "$K_SCHEDULE" \
  --time-ms "$TIME_MS" \
  --mode "$MODE" \
  --seed-base "$SEED_BASE" \
  --backend "$BACKEND")

if [[ "$PROFILE" == "1" ]]; then
  cmd+=(--profile)
fi
if [[ "${CATAN_MEASURE_C_STATS:-0}" == "1" ]]; then
  cmd+=(--measure-c-stats)
fi

exec "${cmd[@]}" "$@"
