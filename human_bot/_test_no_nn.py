"""Ablation: super_m2 with NN policy REPLACED by heuristic action-type ordering.

If WR stays ~80%, the NN isn't contributing much.
If WR drops to ~30-50%, the NN is doing real work.

Heuristic ordering (mirrors csrc/search.c's action_order):
  CITY > SETTLEMENT > BUY_DEV > ROAD > KNIGHT > MONOPOLY > YOP >
  ROAD_BUILDING > MARITIME > ROBBER > END_TURN > ROLL > rest

We pick top-K by this ordering instead of NN logits. Everything else
identical to super_m2.
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import sys
import time

import numpy as np


# Action-type ordering: lower number = higher priority (matches csrc/search.c)
# AT enum: ROLL=0 ROBBER=1 DISCARD=2 ROAD=3 SETTLEMENT=4 CITY=5 BUY_DEV=6
#          KNIGHT=7 YOP=8 MONOPOLY=9 ROAD_BUILDING=10 MARITIME=11
#          OFFER=12 ACCEPT=13 REJECT=14 CONFIRM=15 CANCEL=16 END_TURN=17
TYPE_ORDER = {
    5: 0,  # CITY
    4: 1,  # SETTLEMENT
    6: 2,  # BUY_DEV
    3: 3,  # ROAD
    7: 4,  # KNIGHT
    9: 5,  # MONOPOLY
    8: 6,  # YOP
    10: 7,  # ROAD_BUILDING
    11: 8,  # MARITIME
    1: 9,  # ROBBER
    17: 10,  # END_TURN
    0: 11,  # ROLL
    2: 12,  # DISCARD
}


def _heuristic_top_k(userdata, game_ptr, actions_ptr, n, k, out_ptr):
    """Replacement policy: rank by action-type priority instead of NN."""
    from hexzero.bindings.structs import Action as CAction
    actions_array = (CAction * n).from_address(actions_ptr)
    # Score each action by type priority (lower = better)
    scored = []
    for i in range(n):
        act_type = actions_array[i].type
        priority = TYPE_ORDER.get(act_type, 20)
        scored.append((priority, i))
    scored.sort()  # ascending = highest priority first
    kk = min(k, len(scored))
    for j in range(kk):
        out_ptr[j] = scored[j][1]
    return kk


def _play_one_no_nn(args):
    """Worker: play one game with super_m2 deep_search but heuristic top-K."""
    (game_idx, seed, nn_seat, weights_path, our_depth, k_schedule,
     time_budget_ms) = args

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import ctypes as C
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.game.interface import CatanGame
    from human_bot.search_heuristics import fix_robber_steal
    from human_bot._test_c_encoder import StateEncoderC
    from human_bot.superbot_v3_c import (
        DeepSearchStats, PolicyTopKFn, _LIB_PATH as _DEEP_LIB,
    )

    lib = load_library()
    libdeep = ctypes.CDLL(_DEEP_LIB)

    FP = ctypes.POINTER(ctypes.c_float)
    libdeep.nn_load.restype = ctypes.c_int
    libdeep.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libdeep.state_encoder_init.restype = None
    libdeep.state_encoder_init.argtypes = [
        ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_int]
    libdeep.deep_search_create.restype = ctypes.c_void_p
    libdeep.deep_search_create.argtypes = [
        ctypes.c_int, ctypes.c_void_p, PolicyTopKFn]
    libdeep.deep_search_destroy.restype = None
    libdeep.deep_search_destroy.argtypes = [ctypes.c_void_p]
    libdeep.deep_search_configure.restype = None
    libdeep.deep_search_configure.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_double]
    libdeep.deep_search_root.restype = ctypes.c_double
    libdeep.deep_search_root.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int)]
    libdeep.deep_search_get_stats.restype = None
    libdeep.deep_search_get_stats.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(DeepSearchStats)]

    # We DON'T use the C-direct path. Use the python-callback path with
    # our heuristic instead of the NN.
    policy_cb = PolicyTopKFn(_heuristic_top_k)
    ds_ctx = libdeep.deep_search_create(20, None, policy_cb)
    schedule_arr = (ctypes.c_int * len(k_schedule))(*k_schedule)
    libdeep.deep_search_configure(
        ds_ctx, our_depth, schedule_arr, len(k_schedule), 2,
        time_budget_ms / 1000)

    # AB2 opponent
    ab_ctx = SearchCtx()
    ab_buf = (CAction * MAX_ACTIONS)()
    ab_eval = ValueFn(lib.base_value_fn)

    def ab2_choose(game, le):
        n = len(le)
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le): ab_buf[i] = a
        res = lib.alphabeta_search(
            C.byref(ab_ctx), C.byref(cg), ab_buf,
            C.c_int(n), C.c_int(2),
            C.c_double(-1e30), C.c_double(1e30),
            C.c_int(bc), ab_eval)
        cb = C.string_at(C.byref(res.action), C.sizeof(res.action))
        for i, a in enumerate(le):
            if C.string_at(C.byref(a), C.sizeof(a)) == cb: return i
        return 0

    def super_m2_no_nn_pick(game):
        le = game.get_legal_actions()
        if not le: return -1
        if len(le) == 1: return 0
        seat = game.current_player()
        # Terminal-win shortcut
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                return i
        # Get top-K root candidates via heuristic (no NN)
        n = len(le)
        actions_arr = (CAction * MAX_ACTIONS)()
        for i, a in enumerate(le):
            actions_arr[i] = a
        K_root = min(k_schedule[0], n)
        out_indices = (ctypes.c_int * 64)()
        n_top = _heuristic_top_k(
            None, ctypes.addressof(game._game),
            ctypes.addressof(actions_arr), n, K_root, out_indices)
        candidates = [out_indices[i] for i in range(n_top)]
        # Recursive search in C (uses heuristic at each level via callback)
        our_color = int(game._game.state.colors[seat])
        cand_arr = (ctypes.c_int * n_top)(*candidates)
        best_idx_out = ctypes.c_int(-1)
        libdeep.deep_search_root(
            ds_ctx, ctypes.addressof(game._game), our_color,
            cand_arr, n_top, ctypes.byref(best_idx_out))
        best_pi = max(0, best_idx_out.value)
        chosen = fix_robber_steal(candidates[best_pi], le)
        return chosen

    t0 = time.time()
    game = CatanGame(seed=seed); game.reset()
    nn_seats = {nn_seat}
    while not game.is_terminal() and game.turn_number < 500:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        if cp in nn_seats:
            game.step(super_m2_no_nn_pick(game))
        else:
            game.step(ab2_choose(game, le))

    libdeep.deep_search_destroy(ds_ctx)
    dt = time.time() - t0
    w = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    return game_idx, nn_seat, w, vps, game.turn_number, dt


def main():
    weights_path = os.path.abspath("csrc/nn_weights_m2.bin")
    our_depth = 6
    k_schedule = (12, 8, 6, 5, 4, 3)
    time_budget_ms = 4000
    num_games = 10
    num_workers = 8
    seed_base = 95000

    jobs = [(gi, seed_base + gi, gi % 4, weights_path, our_depth,
             k_schedule, time_budget_ms) for gi in range(num_games)]

    print(f"=== ABLATION: super_m2 with NN REPLACED by action-type heuristic ===")
    print(f"  Same depth=6, k=12,8,6,5,4,3, AB-leaf, AB2 opps, 4s budget")
    print(f"  Top-K selection: by action-type priority (CITY > SETTLE > BUY_DEV...)")
    print(f"  10 games (1v3, random seat), 8 workers")
    print()

    ctx = mp.get_context("spawn")
    nn_wins = ab_wins = 0
    rank_sum = 0
    nn_vp_sum = ab_vp_sum = 0
    completed = 0
    t_start = time.time()

    with ctx.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_play_one_no_nn, jobs):
            gi, nn_seat, w, vps, turns, dt = result
            completed += 1
            nn_vp = vps[nn_seat]
            opp_avg = sum(v for s, v in enumerate(vps) if s != nn_seat) / 3
            nn_vp_sum += nn_vp
            ab_vp_sum += opp_avg
            rank = sorted(range(4), key=lambda s: -vps[s]).index(nn_seat) + 1
            rank_sum += rank
            if w == nn_seat: nn_wins += 1
            elif w is not None: ab_wins += 1
            elapsed = time.time() - t_start
            wr = nn_wins / max(nn_wins + ab_wins, 1)
            print(f"  [{completed:>2d}/{num_games}] g{gi} seat={nn_seat} "
                  f"W={w} VP={nn_vp}/{int(opp_avg)} rank={rank} "
                  f"({turns}t {dt:.0f}s) | WR={wr:.0%} avg_rank={rank_sum/completed:.2f} "
                  f"[{elapsed:.0f}s wall]", flush=True)

    elapsed = time.time() - t_start
    total = nn_wins + ab_wins
    wr = nn_wins / max(total, 1)
    print(f"\n===== RESULTS =====")
    print(f"  Wins:      {nn_wins}/{total} ({wr:.1%})")
    print(f"  Avg rank:  {rank_sum/num_games:.2f} / 4 (random=2.50)")
    print(f"  Avg VP:    NN={nn_vp_sum/num_games:.2f}  opp={ab_vp_sum/num_games:.2f}")
    print(f"  Wall time: {elapsed:.1f}s ({num_games/elapsed*60:.1f} g/min)")
    print()
    print(f"  Compare to super_m2 WITH NN: 8/10 (80%) avg_rank=1.60 "
          f"NN_VP=8.9 opp=5.0")


if __name__ == "__main__":
    main()
