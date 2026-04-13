#!/bin/bash
# =============================================================================
# PBT Coordinator: periodically selects the best agent and shares its weights.
#
# Runs on sc (no GPU needed). Every POLL_INTERVAL seconds:
#   1. Checks which agent checkpoints have been updated
#   2. Quick-evaluates each new checkpoint vs AB2 (CPU-only, C engine greedy)
#   3. Copies the best to shared/best.pt
#   4. Agents auto-reload on their next outer step
#
# Usage:
#   bash coordinate_pbt.sh                   (foreground)
#   nohup bash coordinate_pbt.sh > coord.log 2>&1 &  (background)
# =============================================================================
set -uo pipefail

PROJECT="/nlp/scr/nroll/catanatron"
CKPT_BASE="${PROJECT}/checkpoints"
SHARED_BEST="${CKPT_BASE}/shared/best.pt"
NUM_AGENTS=4
POLL_INTERVAL=300  # 5 minutes
EVAL_GAMES=8

CONDA_HOOK='eval "$(/nlp/scr/nroll/miniconda3/bin/conda shell.bash hook)" && conda activate hexazero'

# Track last-seen mtime for each agent
declare -A LAST_MTIME
for i in $(seq 0 $((NUM_AGENTS - 1))); do
    LAST_MTIME[$i]=0
done

echo "[coordinator] Starting PBT coordinator"
echo "[coordinator] Agents: ${NUM_AGENTS}"
echo "[coordinator] Poll interval: ${POLL_INTERVAL}s"
echo "[coordinator] Eval games: ${EVAL_GAMES}"
echo ""

ROUND=0

while true; do
    ROUND=$((ROUND + 1))
    echo "[coordinator] === Round ${ROUND} ($(date)) ==="

    # Check if any agents are still running
    RUNNING=$(squeue -u nroll -h -n "rnad-agent0,rnad-agent1,rnad-agent2,rnad-agent3" 2>/dev/null | wc -l)
    if [ "$RUNNING" -eq 0 ]; then
        echo "[coordinator] No agents running, exiting."
        break
    fi
    echo "[coordinator] ${RUNNING} agents running"

    # Find updated checkpoints
    UPDATED_AGENTS=""
    for i in $(seq 0 $((NUM_AGENTS - 1))); do
        CKPT="${CKPT_BASE}/agent${i}/latest.pt"
        if [ -f "$CKPT" ]; then
            MTIME=$(stat -c %Y "$CKPT" 2>/dev/null || stat -f %m "$CKPT" 2>/dev/null || echo "0")
            if [ "$MTIME" -gt "${LAST_MTIME[$i]}" ]; then
                UPDATED_AGENTS="${UPDATED_AGENTS} ${i}"
                LAST_MTIME[$i]=$MTIME
                echo "[coordinator] Agent ${i} has new checkpoint (mtime=${MTIME})"
            fi
        fi
    done

    if [ -z "$UPDATED_AGENTS" ]; then
        echo "[coordinator] No new checkpoints, sleeping ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Evaluate each updated agent vs AB2 (CPU-only, fast)
    BEST_AGENT=-1
    BEST_WINS=-1

    for i in $UPDATED_AGENTS; do
        CKPT="${CKPT_BASE}/agent${i}/latest.pt"
        echo "[coordinator] Evaluating agent ${i}..."

        WINS=$(eval "$CONDA_HOOK" && cd "$PROJECT" && python -c "
import os, sys, random, math, ctypes, torch
os.environ['PYTHONUNBUFFERED']='1'
from hexzero.model.network import HexaZeroNet
from hexzero.encoder.action_encoder import ActionEncoder
from hexzero.game.interface import CatanGame
from hexzero.bindings.lib_loader import load_library
from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS

lib = load_library()
ae = ActionEncoder()
net = HexaZeroNet.load_checkpoint('${CKPT}', device='cpu')
net.eval()
g = CatanGame(seed=0); g.reset()
se = g.make_state_encoder()

hz_w = 0
for gi in range(${EVAL_GAMES}):
    g = CatanGame(seed=90000+gi); g.reset()
    hz_s, ab2_s = gi%4, (gi+1)%4
    while not g.is_terminal() and g.turn_number < 1000:
        cp = g.current_player(); le = g.get_legal_actions()
        if not le: break
        if cp == hz_s:
            bi, bv = 0, -1e9
            for i in range(len(le)):
                c = g.clone(); c.step(i)
                if c.is_terminal():
                    v = 10.0 if c.winner() == hz_s else -10.0
                else:
                    enc = se.encode(c.get_state_view())
                    bb = {k: v.unsqueeze(0) for k, v in enc.items()}
                    cl = c.get_legal_actions()
                    if cl: bb['action_mask'] = ae.get_action_mask(cl).unsqueeze(0)
                    with torch.no_grad(): v = net(bb)['value'][0,0].item()
                if v > bv: bv = v; bi = i
            g.step(bi)
        elif cp == ab2_s:
            cg = g._game; bc = cg.state.colors[cg.state.current_player_index]
            bi, bv = 0, -math.inf
            ch = CGame(); ca = (Action*MAX_ACTIONS)(); cn = ctypes.c_int(0)
            for i, act in enumerate(le):
                lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
                lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
                v = lib.base_value_fn(ctypes.byref(ch), bc)
                if v > bv: bv = v; bi = i
            g.step(bi)
        else:
            g.step(random.randrange(len(le)))
    w = g.winner()
    if w == hz_s: hz_w += 1
print(hz_w)
" 2>/dev/null || echo "0")

        echo "[coordinator] Agent ${i}: ${WINS}/${EVAL_GAMES} wins vs AB2"

        if [ "$WINS" -gt "$BEST_WINS" ]; then
            BEST_WINS=$WINS
            BEST_AGENT=$i
        fi
    done

    # Copy best agent's checkpoint to shared location
    if [ "$BEST_AGENT" -ge 0 ]; then
        SRC="${CKPT_BASE}/agent${BEST_AGENT}/latest.pt"
        echo "[coordinator] Best: agent ${BEST_AGENT} (${BEST_WINS} wins)"
        echo "[coordinator] Copying ${SRC} -> ${SHARED_BEST}"
        cp "$SRC" "${SHARED_BEST}.tmp" && mv "${SHARED_BEST}.tmp" "$SHARED_BEST"
        echo "[coordinator] Shared best.pt updated"
    else
        echo "[coordinator] No improvement found"
    fi

    echo "[coordinator] Sleeping ${POLL_INTERVAL}s..."
    echo ""
    sleep "$POLL_INTERVAL"
done

echo "[coordinator] Finished."
