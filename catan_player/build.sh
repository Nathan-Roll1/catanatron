#!/usr/bin/env bash
#
# Build the Catan player binary from C source.
#
# Usage:
#   ./build.sh          # build
#   ./build.sh clean    # remove binary
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CSRC="$SCRIPT_DIR/csrc"
OUT="$SCRIPT_DIR/catan_player"

CC="${CC:-cc}"
CFLAGS="-O3 -march=native -flto -ffast-math -funroll-loops -I${CSRC}"
LDFLAGS="-lm"

case "$(uname -s)" in
    Darwin) CFLAGS="$CFLAGS -framework Accelerate" ;;
    *)
        if pkg-config --exists openblas 2>/dev/null; then
            CFLAGS="$CFLAGS -DHAVE_CBLAS $(pkg-config --cflags openblas)"
            LDFLAGS="$LDFLAGS $(pkg-config --libs openblas)"
        elif [ -f /usr/include/cblas.h ] || [ -f /usr/include/openblas/cblas.h ]; then
            CFLAGS="$CFLAGS -DHAVE_CBLAS"
            LDFLAGS="$LDFLAGS -lopenblas"
        fi ;;
esac

if [ "${1:-}" = "clean" ]; then
    rm -f "$OUT"
    echo "Cleaned."
    exit 0
fi

SRCS="fast_player.c nn.c rng.c map.c board.c state.c actions.c"
SRCS="$SRCS apply_action.c game.c value.c search.c"
SRCS="$SRCS state_encode.c policy_topk.c deep_search.c"

SRC_PATHS=""
for s in $SRCS; do SRC_PATHS="$SRC_PATHS $CSRC/$s"; done

echo "Building catan_player ..."
$CC $CFLAGS $SRC_PATHS -o "$OUT" $LDFLAGS

SIZE=$(wc -c < "$OUT" | tr -d ' ')
echo "Built $OUT ($((SIZE / 1024)) KB)"
