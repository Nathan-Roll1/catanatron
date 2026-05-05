"""N-ply tree search evaluation using the NN value head vs AB2.

Supports three evaluation modes:
  0-ply: Sample from policy head (temperature=0.1)
  1-ply: Greedy on NN value head after each candidate move
  2-ply: Greedy after our move + opponent's best response (base_value_fn)

Usage:
    python -m human_bot.eval_search --checkpoint checkpoints/human_bot/latest.pt \
        --num-games 50 --search-depth 1
"""

from __future__ import annotations

import argparse
import ctypes
import time

import numpy as np
import torch
import torch.nn.functional as F

from human_bot.model import HumanBotNet
from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal


def evaluate_search_vs_ab2(
    net: torch.nn.Module,
    state_enc,
    action_enc,
    device: str,
    lib,
    num_games: int = 25,
    search_depth: int = 1,
    temperature: float = 0.1,
    seed_offset: int = 0,
    nn_opponent: bool = False,
    ab_value_leaf: bool = False,
) -> dict[str, int | float]:
    """Play games: 2 NN seats vs 2 AB2 seats with configurable search depth.

    search_depth=0: policy sampling, search_depth=1: 1-ply NN value greedy,
    search_depth=2: our move + opponent response, then value eval.

    nn_opponent: use NN value head for opponent model instead of base_value_fn.
    ab_value_leaf: evaluate search leaves with AB2 base_value_fn instead of
        the NN value head. Recommended for reliable external-style eval
        while the NN value head is still being trained.
    """
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import (
        Game as CGame, Action, MAX_ACTIONS, SearchCtx, ValueFn,
    )

    AD = 337
    # SearchCtx + wrapped eval_fn for proper expectimax AB2 (matches Python
    # catanatron's AlphaBetaPlayer(depth=2)). Reused across all decisions.
    _ab_ctx = SearchCtx()
    _ab_buf = (Action * MAX_ACTIONS)()
    _ab_eval = ValueFn(lib.base_value_fn)

    def _proper_ab2_choose(g, le, depth=2):
        n = len(le)
        if n == 0: return 0
        if n == 1: return 0
        cg = g._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, act in enumerate(le):
            _ab_buf[i] = act
        res = lib.alphabeta_search(
            ctypes.byref(_ab_ctx), ctypes.byref(cg), _ab_buf,
            ctypes.c_int(n), ctypes.c_int(depth),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), _ab_eval,
        )
        chosen_bytes = ctypes.string_at(ctypes.byref(res.action),
                                         ctypes.sizeof(res.action))
        for i, act in enumerate(le):
            if ctypes.string_at(ctypes.byref(act),
                                ctypes.sizeof(act)) == chosen_bytes:
                return i
        return 0
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF = state_enc.NODE_FEATURE_DIM
    EF = state_enc.EDGE_FEATURE_DIM
    FF = state_enc.FLAT_FEATURE_DIM

    edge_index_dev = state_enc._edge_index.to(device)
    net.eval()

    games = [CatanGame(seed=80000 + seed_offset * 1000 + i) for i in range(num_games)]
    for g in games:
        g.reset()

    hz_seats = [{i % 4, (i + 2) % 4} for i in range(num_games)]
    ab2_seats = [{(i + 1) % 4, (i + 3) % 4} for i in range(num_games)]

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)

    total_nn_calls = 0
    active = list(range(num_games))
    # Track last maritime-trade receive per (game, player) to block circular trades.
    # Reset when the player does a non-maritime action or turn ends.
    _mar_received: dict[tuple[int, int], set[int]] = {}

    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # --- AB2 seats: proper alpha-beta minimax with chance-node
        #     expectimax (matches Python catanatron AlphaBetaPlayer(depth=2)) ---
        progress = True
        while progress:
            progress = False
            for idx in active:
                g = games[idx]
                if g.is_terminal() or g.turn_number >= 1000:
                    continue
                cp = g.current_player()
                if cp not in ab2_seats[idx]:
                    continue
                le = g.get_legal_actions()
                if not le:
                    continue
                bi = _proper_ab2_choose(g, le, depth=2)
                g.step(bi)
                progress = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 1000]
        if not active:
            break

        # --- NN seats ---
        any_moved = False
        for idx in active:
            g = games[idx]
            if g.is_terminal() or g.turn_number >= 1000:
                continue
            cp = g.current_player()
            if cp not in hz_seats[idx]:
                continue
            le = g.get_legal_actions()
            if not le:
                continue

            if len(le) == 1:
                act = le[0]
                if act.type == 11:  # AT_MARITIME_TRADE
                    _mar_received.setdefault((idx, cp), set()).add(act.value[4])
                elif act.type != 11:
                    _mar_received.pop((idx, cp), None)
                g.step(0)
                any_moved = True
                continue

            # Filter out circular maritime trades
            received = _mar_received.get((idx, cp), set())
            if received:
                filtered_le = [a for a in le
                               if a.type != 11 or a.value[0] not in received]
                if not filtered_le:
                    filtered_le = le
                le_to_use = filtered_le
                idx_map = [le.index(a) for a in le_to_use]
            else:
                le_to_use = le
                idx_map = list(range(len(le)))

            if search_depth == 0:
                chosen = _policy_sample(
                    g, le_to_use, net, state_enc, action_enc, device,
                    edge_index_dev, temperature, AD, N, E, NF, EF, FF,
                )
                chosen = idx_map[chosen]
                total_nn_calls += 1
            else:
                chosen, nn_calls = _value_search(
                    g, le_to_use, hz_seats[idx], ab2_seats[idx], net, state_enc,
                    device, edge_index_dev, lib, search_depth, N, E, NF, EF, FF,
                    ch, ca, cn, nn_opponent=nn_opponent,
                    ab_value_leaf=ab_value_leaf,
                )
                chosen = idx_map[chosen]
                total_nn_calls += nn_calls

            act = le[chosen]
            if act.type == 11:  # AT_MARITIME_TRADE
                _mar_received.setdefault((idx, cp), set()).add(act.value[4])
            else:
                _mar_received.pop((idx, cp), None)

            g.step(chosen)
            any_moved = True

        if not any_moved:
            break

    hz_wins = ab2_wins = 0
    nn_rank_sum = 0.0
    nn_rank_count = 0
    for idx in range(num_games):
        g = games[idx]
        w = g.winner()
        if w is not None:
            if w in hz_seats[idx]:
                hz_wins += 1
            elif w in ab2_seats[idx]:
                ab2_wins += 1

        vps = [g._game.state.player_state[p][0] for p in range(4)]
        ranked = sorted(range(4), key=lambda p: vps[p], reverse=True)
        rank_of = {p: r + 1 for r, p in enumerate(ranked)}
        for seat in hz_seats[idx]:
            nn_rank_sum += rank_of[seat]
            nn_rank_count += 1

    total = hz_wins + ab2_wins
    avg_rank = nn_rank_sum / max(nn_rank_count, 1)
    return {
        "hz_wins": hz_wins,
        "ab2_wins": ab2_wins,
        "draws": num_games - total,
        "win_rate": hz_wins / max(total, 1),
        "avg_rank": avg_rank,
        "nn_fwd_calls": total_nn_calls,
    }


def _policy_sample(g, le, net, state_enc, action_enc, device,
                   edge_index_dev, temperature, AD, N, E, NF, EF, FF):
    """0-ply: sample from masked policy with temperature.

    Uses argmax during initial build phase (turns 0-7) where one-shot
    placement quality matters most.  Mid-game uses low temperature (0.05).
    """
    is_initial = g.turn_number <= 7
    effective_temp = 0.01 if is_initial else max(temperature, 1e-6)

    nf = np.zeros((1, N, NF), dtype=np.float32)
    ef = np.zeros((1, E, EF), dtype=np.float32)
    ff = np.zeros((1, FF), dtype=np.float32)
    state_enc.encode_into(g.get_state_view(), nf[0], ef[0], ff[0])
    mask_np = action_enc.get_action_mask(le).numpy()

    use_argmax = (temperature < 0.001)

    with torch.no_grad():
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        pad = torch.zeros(1, 397 - AD, device=device)
        mask_397 = torch.cat([mask_t, pad], dim=1)
        batch = {
            "node_features": torch.from_numpy(nf.copy()).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(ef.copy()).to(device),
            "flat_features": torch.from_numpy(ff.copy()).to(device),
            "action_mask": mask_397,
        }
        out = net(batch)
        lo = out["policy_logits"][:, :AD]
        lo = lo.masked_fill(mask_t == 0, -1e9)

        if is_initial or use_argmax:
            aidx = int(lo.argmax(dim=-1).item())
        else:
            lo = lo / effective_temp
            pr = F.softmax(lo, dim=-1).cpu().numpy()[0]
            if pr.sum() < 1e-6:
                pr = mask_np / max(mask_np.sum(), 1e-8)
            pr = pr / pr.sum()
            aidx = int(np.random.choice(AD, p=pr))

    return next((i for i, a in enumerate(le) if action_enc.encode(a) == aidx), 0)


def _value_search(g, le, hz_seats_set, ab2_seats_set, net, state_enc,
                  device, edge_index_dev, lib, search_depth, N, E, NF, EF, FF,
                  ch, ca, cn, top_k: int = 5, nn_opponent: bool = False,
                  ab_value_leaf: bool = False):
    """N-ply search: pick action maximising value for our seat.

    For depth >= 2, restricts branching to top_k moves by policy score
    to keep computation tractable.  If nn_opponent=True, uses the NN
    value head for opponent responses instead of base_value_fn. If
    ab_value_leaf=True, evaluates leaves with AB2 base_value_fn (from
    our seat's perspective) instead of the NN value head — the NN value
    head is often unreliable during training, so this keeps eval
    external and stable.
    """
    our_seat = g.current_player()

    # Use policy head to rank moves and restrict to top_k for deeper search
    candidates = list(range(len(le)))
    if len(le) > top_k and search_depth >= 2:
        candidates = _policy_top_k(
            g, le, net, state_enc, device, edge_index_dev, top_k, N, E, NF, EF, FF,
        )

    from hexzero.encoder.action_encoder import ActionEncoder
    _ae = getattr(_value_search, "_ae", None)
    if _ae is None:
        _ae = ActionEncoder()
        _value_search._ae = _ae

    B = len(candidates)
    nf_buf = np.zeros((B, N, NF), dtype=np.float32)
    ef_buf = np.zeros((B, E, EF), dtype=np.float32)
    ff_buf = np.zeros((B, FF), dtype=np.float32)
    mask_buf = np.zeros((B, 397), dtype=np.float32)
    terminal = np.zeros(B, dtype=np.float32)
    terminal_val = np.zeros(B, dtype=np.float32)
    child_current = np.zeros(B, dtype=np.int32)
    non_terminal_count = 0

    for bi, ai in enumerate(candidates):
        gc = g.clone()
        gc.step(ai)

        if search_depth >= 2 and not gc.is_terminal():
            cp = gc.current_player()
            if cp in ab2_seats_set:
                if nn_opponent:
                    _nn_respond_any(gc, net, state_enc, device,
                                   edge_index_dev, N, E, NF, EF, FF)
                else:
                    _ab2_respond(gc, lib, ch, ca, cn)

        if search_depth >= 3 and not gc.is_terminal():
            _nn_respond(gc, hz_seats_set, net, state_enc, device,
                        edge_index_dev, N, E, NF, EF, FF, top_k)

        if gc.is_terminal():
            terminal[bi] = 1.0
            w = gc.winner()
            if ab_value_leaf:
                # Use base_value_fn for terminals too so the scale matches
                # the non-terminal AB-leaf values (~1e14). Otherwise the
                # search would never prefer a terminal win over a "good"
                # non-terminal position. base_value_fn naturally captures
                # win/loss because the winner has VP=10 and losers <10.
                cg = gc._game
                bot_color = cg.state.colors[our_seat]
                terminal_val[bi] = float(lib.base_value_fn(
                    ctypes.byref(cg), bot_color))
            else:
                # NN-value path stays on the [-10, 10] scale
                if w is not None and w == our_seat:
                    terminal_val[bi] = 10.0
                elif w is not None:
                    terminal_val[bi] = -10.0
                else:
                    terminal_val[bi] = 0.0
        else:
            sv = gc.get_state_view()
            state_enc.encode_into(sv, nf_buf[non_terminal_count],
                                  ef_buf[non_terminal_count], ff_buf[non_terminal_count])
            child_le = gc.get_legal_actions()
            child_mask = _ae.get_action_mask(child_le).numpy()
            mask_buf[non_terminal_count, :len(child_mask)] = child_mask
            child_current[bi] = gc.current_player()
            non_terminal_count += 1

    nn_calls = 0
    values = np.zeros((B, 4), dtype=np.float32)
    ab_leaf_values = np.zeros(B, dtype=np.float64)  # scalar per leaf, from our seat

    if non_terminal_count > 0:
        if ab_value_leaf:
            # Evaluate each non-terminal leaf with AB2 base_value_fn from
            # our seat's perspective. One C call per leaf; no GPU forward.
            # We still need to track which non-terminal position was which
            # original candidate.
            vi = 0
            for bi in range(B):
                if terminal[bi] > 0:
                    continue
                # The candidate gc was rolled forward in-place earlier into
                # the `gc` temp; we need to reconstruct it here. Cheapest:
                # re-apply the candidate step + any responses into a fresh clone.
                gc = g.clone()
                gc.step(candidates[bi])
                if search_depth >= 2 and not gc.is_terminal():
                    cp = gc.current_player()
                    if cp in ab2_seats_set:
                        if nn_opponent:
                            _nn_respond_any(gc, net, state_enc, device,
                                           edge_index_dev, N, E, NF, EF, FF)
                        else:
                            _ab2_respond(gc, lib, ch, ca, cn)
                if search_depth >= 3 and not gc.is_terminal():
                    _nn_respond(gc, hz_seats_set, net, state_enc, device,
                                edge_index_dev, N, E, NF, EF, FF, top_k)
                cg = gc._game
                bot_color = cg.state.colors[our_seat]
                ab_leaf_values[bi] = float(lib.base_value_fn(
                    ctypes.byref(cg), bot_color))
                vi += 1
        else:
            with torch.no_grad():
                batch = {
                    "node_features": torch.from_numpy(nf_buf[:non_terminal_count].copy()).to(device),
                    "edge_index": edge_index_dev,
                    "edge_features": torch.from_numpy(ef_buf[:non_terminal_count].copy()).to(device),
                    "flat_features": torch.from_numpy(ff_buf[:non_terminal_count].copy()).to(device),
                    "action_mask": torch.from_numpy(mask_buf[:non_terminal_count].copy()).to(device),
                }
                out = net(batch)
                raw_values = out["value"].cpu().numpy()
            nn_calls = 1

            vi = 0
            for bi in range(B):
                if terminal[bi] == 0:
                    values[bi] = raw_values[vi]
                    vi += 1

    best_bi = 0
    best_val = -1e30
    for bi in range(B):
        if terminal[bi] > 0:
            v = terminal_val[bi]
        elif ab_value_leaf:
            v = float(ab_leaf_values[bi])
        else:
            new_current = child_current[bi]
            offset = (our_seat - new_current) % 4
            v = float(values[bi, offset])
        v = apply_action_bonus(v, le[candidates[bi]])
        if v > best_val:
            best_val = v
            best_bi = bi

    chosen = candidates[best_bi]
    chosen = fix_robber_steal(chosen, le)
    return chosen, nn_calls


AD = 337


def _policy_top_k(g, le, net, state_enc, device, edge_index_dev, k, N, E, NF, EF, FF):
    """Return indices of the top-k legal moves ranked by policy head."""
    from hexzero.encoder.action_encoder import ActionEncoder
    _ae = getattr(_policy_top_k, "_ae", None)
    if _ae is None:
        _ae = ActionEncoder()
        _policy_top_k._ae = _ae

    nf = np.zeros((1, N, NF), dtype=np.float32)
    ef = np.zeros((1, E, EF), dtype=np.float32)
    ff = np.zeros((1, FF), dtype=np.float32)
    state_enc.encode_into(g.get_state_view(), nf[0], ef[0], ff[0])
    mask_np = _ae.get_action_mask(le).numpy()

    with torch.no_grad():
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        pad = torch.zeros(1, 397 - AD, device=device)
        mask_397 = torch.cat([mask_t, pad], dim=1)
        batch = {
            "node_features": torch.from_numpy(nf.copy()).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(ef.copy()).to(device),
            "flat_features": torch.from_numpy(ff.copy()).to(device),
            "action_mask": mask_397,
        }
        out = net(batch)
        logits = out["policy_logits"][0, :AD].cpu().numpy()

    action_to_le = {}
    for i, act in enumerate(le):
        enc = _ae.encode(act)
        action_to_le[enc] = i

    scored = [(logits[enc], le_idx) for enc, le_idx in action_to_le.items()]
    scored.sort(reverse=True)
    return [le_idx for _, le_idx in scored[:k]]


def _nn_respond(gc, hz_seats_set, net, state_enc, device, edge_index_dev,
                N, E, NF, EF, FF, top_k):
    """Let the current NN-allied player take one greedy step using policy argmax."""
    if gc.is_terminal():
        return
    cp = gc.current_player()
    if cp not in hz_seats_set:
        return
    le = gc.get_legal_actions()
    if not le:
        return
    if len(le) == 1:
        gc.step(0)
        return

    from hexzero.encoder.action_encoder import ActionEncoder
    _ae = getattr(_nn_respond, "_ae", None)
    if _ae is None:
        _ae = ActionEncoder()
        _nn_respond._ae = _ae

    nf = np.zeros((1, N, NF), dtype=np.float32)
    ef = np.zeros((1, E, EF), dtype=np.float32)
    ff = np.zeros((1, FF), dtype=np.float32)
    state_enc.encode_into(gc.get_state_view(), nf[0], ef[0], ff[0])
    mask_np = _ae.get_action_mask(le).numpy()

    with torch.no_grad():
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
        pad = torch.zeros(1, 397 - AD, device=device)
        mask_397 = torch.cat([mask_t, pad], dim=1)
        batch = {
            "node_features": torch.from_numpy(nf.copy()).to(device),
            "edge_index": edge_index_dev,
            "edge_features": torch.from_numpy(ef.copy()).to(device),
            "flat_features": torch.from_numpy(ff.copy()).to(device),
            "action_mask": mask_397,
        }
        out = net(batch)
        logits = out["policy_logits"][0, :AD]
        logits = logits.masked_fill(mask_t[0] == 0, -1e9)
        best_aidx = int(logits.argmax().item())

    chosen = next((i for i, a in enumerate(le) if _ae.encode(a) == best_aidx), 0)
    gc.step(chosen)


def _nn_respond_any(gc, net, state_enc, device, edge_index_dev, N, E, NF, EF, FF):
    """Let the current player take one greedy step using NN value head."""
    if gc.is_terminal():
        return
    le = gc.get_legal_actions()
    if not le:
        return
    if len(le) == 1:
        gc.step(0)
        return

    from hexzero.encoder.action_encoder import ActionEncoder
    _ae = getattr(_nn_respond_any, "_ae", None)
    if _ae is None:
        _ae = ActionEncoder()
        _nn_respond_any._ae = _ae

    our_seat = gc.current_player()
    B = len(le)
    nf_buf = np.zeros((B, N, NF), dtype=np.float32)
    ef_buf = np.zeros((B, E, EF), dtype=np.float32)
    ff_buf = np.zeros((B, FF), dtype=np.float32)
    mask_buf = np.zeros((B, 397), dtype=np.float32)
    child_current = np.zeros(B, dtype=np.int32)
    terminal = np.zeros(B, dtype=bool)
    term_val = np.zeros(B, dtype=np.float32)
    nt = 0

    for ai in range(B):
        gc2 = gc.clone()
        gc2.step(ai)
        if gc2.is_terminal():
            terminal[ai] = True
            w = gc2.winner()
            term_val[ai] = 10.0 if w == our_seat else (-10.0 if w is not None else 0.0)
        else:
            state_enc.encode_into(gc2.get_state_view(), nf_buf[nt], ef_buf[nt], ff_buf[nt])
            child_le = gc2.get_legal_actions()
            child_mask = _ae.get_action_mask(child_le).numpy()
            mask_buf[nt, :len(child_mask)] = child_mask
            child_current[ai] = gc2.current_player()
            nt += 1

    values = np.zeros((B, 4), dtype=np.float32)
    if nt > 0:
        with torch.no_grad():
            batch = {
                "node_features": torch.from_numpy(nf_buf[:nt].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:nt].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:nt].copy()).to(device),
                "action_mask": torch.from_numpy(mask_buf[:nt].copy()).to(device),
            }
            rv = net(batch)["value"].cpu().numpy()
        vi = 0
        for ai in range(B):
            if not terminal[ai]:
                values[ai] = rv[vi]
                vi += 1

    best_ai, best_val = 0, -1e30
    for ai in range(B):
        if terminal[ai]:
            v = float(term_val[ai])
        else:
            offset = (our_seat - child_current[ai]) % 4
            v = float(values[ai, offset])
        v = apply_action_bonus(v, le[ai])
        if v > best_val:
            best_val = v
            best_ai = ai

    best_ai = fix_robber_steal(best_ai, le)
    gc.step(best_ai)


def _ab2_respond(gc, lib, ch, ca, cn):
    """Let the current AB2 player take one greedy step."""
    le = gc.get_legal_actions()
    if not le:
        return
    if len(le) == 1:
        gc.step(0)
        return
    cg = gc._game
    bc = cg.state.colors[cg.state.current_player_index]
    bi, bv = 0, -1e30
    for i, act in enumerate(le):
        lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
        lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
        v = lib.base_value_fn(ctypes.byref(ch), bc)
        if v > bv:
            bv = v
            bi = i
    gc.step(bi)


def main():
    parser = argparse.ArgumentParser(description="N-ply NN search eval vs AB2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--search-depth", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--nn-opponent", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    action_enc = ActionEncoder()
    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()

    net = HumanBotNet.load_checkpoint(args.checkpoint, device=device)
    print(f"Loaded {args.checkpoint} ({net.num_parameters:,} params) on {device}")

    for depth in args.search_depth:
        print(f"\n{'='*50}")
        print(f"  {depth}-ply search  vs  AB2  ({args.num_games} games, 2v2)")
        print(f"{'='*50}")
        t0 = time.perf_counter()
        result = evaluate_search_vs_ab2(
            net, state_enc, action_enc, device, lib,
            num_games=args.num_games,
            search_depth=depth,
            seed_offset=args.seed_offset + depth * 100,
            nn_opponent=args.nn_opponent,
        )
        elapsed = time.perf_counter() - t0
        print(f"  NN wins: {result['hz_wins']}   AB2 wins: {result['ab2_wins']}   "
              f"draws: {result['draws']}")
        print(f"  Win rate: {result['win_rate']:.1%}   Avg rank: {result['avg_rank']:.2f}")
        print(f"  NN forward calls: {result['nn_fwd_calls']:,}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/max(args.num_games,1):.2f}s/game)")


if __name__ == "__main__":
    main()
