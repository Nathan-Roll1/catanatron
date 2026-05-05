"""Single 2v2 game: 2 C-direct players vs 2 Python-wrapped players.

Both use the same libdeep.dylib (same algorithm, same config). The only
difference:
  - C-direct: calls libdeep functions directly, no Python wrapper overhead
  - Python-wrapped: uses SuperBotV3C2 (forced-move check, terminal-win
    check, low-entropy fast path, then libdeep)

Tracks per-decision: wall time, C_leaves, C_calls.
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np

from human_bot._test_c_encoder import StateEncoderC
from human_bot.superbot_v3_c import DeepSearchStats, _LIB_PATH as _DEEP_LIB
from human_bot.search_heuristics import fix_robber_steal


def _setup_libdeep_bindings(libdeep):
    FP = ctypes.POINTER(ctypes.c_float)
    libdeep.nn_load.restype = ctypes.c_int
    libdeep.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    libdeep.state_encoder_init.restype = None
    libdeep.state_encoder_init.argtypes = [
        ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_int]
    libdeep.policy_top_k.restype = ctypes.c_int
    libdeep.policy_top_k.argtypes = [
        ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        FP, FP, FP, FP, FP]
    libdeep.deep_search_create_c.restype = ctypes.c_void_p
    libdeep.deep_search_create_c.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(StateEncoderC)]
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
    libdeep.deep_search_reset_stats.restype = None
    libdeep.deep_search_reset_stats.argtypes = [ctypes.c_void_p]
    return FP


class CDirectBot:
    """Mimics catan_player's super_m2_choose — no Python in inner loop.
    Skips the Python wrapper's entropy check / forced-move shortcuts."""
    def __init__(self, weights_path, our_depth=6,
                 k_schedule=(12, 8, 6, 5, 4, 3),
                 time_budget_ms=4000):
        from hexzero.bindings.lib_loader import load_library
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()

        self._libdeep = ctypes.CDLL(_DEEP_LIB)
        FP = _setup_libdeep_bindings(self._libdeep)
        self._FP = FP

        # Load NN
        mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        rc = self._libdeep.nn_load(mptr, weights_path.encode())
        assert rc == 0
        self._mbuf = mbuf
        self._mptr = mptr

        # State encoder
        self._enc = StateEncoderC()
        self._libdeep.state_encoder_init(
            ctypes.byref(self._enc), ctypes.addressof(g0._game), 4)

        # Deep search context
        self._ds_ctx = self._libdeep.deep_search_create_c(
            20, mptr, ctypes.byref(self._enc))
        schedule = (ctypes.c_int * len(k_schedule))(*k_schedule)
        self._libdeep.deep_search_configure(
            self._ds_ctx, our_depth, schedule, len(k_schedule), 2,
            time_budget_ms / 1000)

        self._top_k_root = k_schedule[0]

        # Scratch
        self._nf = np.zeros((54, 18), dtype=np.float32)
        self._ef = np.zeros((144, 5), dtype=np.float32)
        self._ff = np.zeros(115, dtype=np.float32)
        self._mk = np.zeros(397, dtype=np.float32)
        self._out = np.zeros(4 + 397, dtype=np.float32)
        self._nfp = self._nf.ctypes.data_as(FP)
        self._efp = self._ef.ctypes.data_as(FP)
        self._ffp = self._ff.ctypes.data_as(FP)
        self._mkp = self._mk.ctypes.data_as(FP)
        self._outp = self._out.ctypes.data_as(FP)

        # Per-decision stats
        self.decisions = []  # list of dicts

    def pick(self, game):
        from hexzero.bindings.structs import Action as CAction, MAX_ACTIONS
        t0 = time.perf_counter()

        le = game.get_legal_actions()
        if not le:
            return -1
        if len(le) == 1:
            self.decisions.append({"type": "forced", "ms": 0.0,
                                    "C_leaves": 0, "C_calls": 0})
            return 0

        seat = game.current_player()

        # Terminal-winning move check
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.decisions.append({"type": "term_win",
                                        "ms": (time.perf_counter() - t0) * 1000,
                                        "C_leaves": 0, "C_calls": 0})
                return i

        # Reset stats so we measure JUST this decision
        self._libdeep.deep_search_reset_stats(self._ds_ctx)

        # Get top-K candidates via C policy
        actions_arr = (CAction * MAX_ACTIONS)()
        for i, a in enumerate(le):
            actions_arr[i] = a
        out_indices = (ctypes.c_int * 64)()
        K_root = min(self._top_k_root, len(le))
        n_top = self._libdeep.policy_top_k(
            ctypes.byref(self._enc), self._mptr,
            ctypes.addressof(game._game),
            ctypes.cast(actions_arr, ctypes.c_void_p),
            len(le), K_root, out_indices,
            self._nfp, self._efp, self._ffp, self._mkp, self._outp)
        candidates = [out_indices[i] for i in range(n_top)]

        # Recursive search
        our_color = int(game._game.state.colors[seat])
        cand_arr = (ctypes.c_int * n_top)(*candidates)
        best_idx_out = ctypes.c_int(-1)
        self._libdeep.deep_search_root(
            self._ds_ctx, ctypes.addressof(game._game), our_color,
            cand_arr, n_top, ctypes.byref(best_idx_out))

        # Read stats
        stats = DeepSearchStats()
        self._libdeep.deep_search_get_stats(self._ds_ctx, ctypes.byref(stats))

        best_pi = max(0, best_idx_out.value)
        chosen = fix_robber_steal(candidates[best_pi], le)

        ms = (time.perf_counter() - t0) * 1000
        self.decisions.append({
            "type": "search",
            "ms": ms,
            "C_leaves": stats.n_leaves,
            "C_calls": stats.n_calls,
        })
        return chosen


def _instrumented_super_v3_c2(weights_path):
    """Wraps SuperBotV3C2 to track per-decision stats (same way as CDirectBot)."""
    from human_bot.superbot_v3_c2 import SuperBotV3C2

    bot = SuperBotV3C2(weights_path,
                        our_depth=6,
                        top_k_schedule=(12, 8, 6, 5, 4, 3),
                        entropy_fast_thresh=0.15,
                        time_budget_ms=4000,
                        leaf_cache_bits=20)
    bot.decisions = []

    orig_pick = bot.pick

    def pick_instrumented(game):
        # Snapshot stats before
        s_before = DeepSearchStats()
        bot._libdeep.deep_search_get_stats(bot._ds_ctx, ctypes.byref(s_before))
        t0 = time.perf_counter()
        chosen = orig_pick(game)
        ms = (time.perf_counter() - t0) * 1000

        s_after = DeepSearchStats()
        bot._libdeep.deep_search_get_stats(bot._ds_ctx, ctypes.byref(s_after))

        n_leaves_dec = s_after.n_leaves - s_before.n_leaves
        n_calls_dec = s_after.n_calls - s_before.n_calls

        # Classify: was the search actually run? (delta_calls > 0)
        if n_calls_dec == 0:
            # Either forced, term_win, or low_entropy
            if ms < 1.0:
                kind = "forced"
            elif ms < 5.0:
                kind = "term_win_or_lowH"
            else:
                kind = "fast_path"
        else:
            kind = "search"

        bot.decisions.append({
            "type": kind, "ms": ms,
            "C_leaves": n_leaves_dec, "C_calls": n_calls_dec,
        })
        return chosen

    bot.pick = pick_instrumented
    return bot


def main():
    from hexzero.bindings.lib_loader import load_library
    from hexzero.game.interface import CatanGame

    weights = os.path.abspath("csrc/nn_weights_m2.bin")
    seed = 95001  # different seed than the 100% benchmark for fair test

    load_library()
    game = CatanGame(seed=seed); game.reset()

    # Seats 0, 2 = Python-wrapped. Seats 1, 3 = C-direct.
    py_bot = _instrumented_super_v3_c2(weights)
    c_bot = CDirectBot(weights)

    py_seats = {0, 2}
    c_seats = {1, 3}

    print(f"=== 1 game, 2v2: 2 Python-wrapped vs 2 C-direct ===")
    print(f"Seed: {seed}")
    print(f"Both bots: super_m2 (depth=6, k=12,8,6,5,4,3, 4s budget)")
    print(f"Python seats: {sorted(py_seats)}  C-direct seats: {sorted(c_seats)}")
    print()

    move_log = []
    t_start = time.time()
    while not game.is_terminal() and game.turn_number < 500:
        le = game.get_legal_actions()
        if not le: break
        if len(le) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        if cp in py_seats:
            chosen = py_bot.pick(game)
            who = f"PY{cp}"
        else:
            chosen = c_bot.pick(game)
            who = f"C{cp}"
        game.step(chosen)
        move_log.append(who)

    wall = time.time() - t_start
    w = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]

    print(f"=== Game result ===")
    print(f"Winner: P{w}  VPs: {vps}  Turns: {game.turn_number}  Wall: {wall:.0f}s")
    print()

    # Per-bot stats
    def summarize(name, decisions):
        n = len(decisions)
        searches = [d for d in decisions if d["type"] == "search"]
        if not searches:
            print(f"{name}: {n} decisions (no search decisions)")
            return
        ms_sum = sum(d["ms"] for d in searches)
        leaves_sum = sum(d["C_leaves"] for d in searches)
        calls_sum = sum(d["C_calls"] for d in searches)
        ms_max = max(d["ms"] for d in searches)
        leaves_max = max(d["C_leaves"] for d in searches)
        n_forced = sum(1 for d in decisions if d["type"] in ("forced",))
        n_termwin = sum(1 for d in decisions if "term_win" in d["type"])
        n_fast = sum(1 for d in decisions if d["type"] == "fast_path")
        n_lowH = sum(1 for d in decisions if "lowH" in d["type"])

        print(f"{name}: {n} decisions = {len(searches)} search "
              f"+ {n_forced} forced + {n_termwin} term-win + {n_fast} fast")
        print(f"  Search totals: {ms_sum:.0f}ms wall, {leaves_sum:,} leaves, {calls_sum:,} C-calls")
        print(f"  Per-search avg: {ms_sum/len(searches):.0f}ms, "
              f"{leaves_sum//len(searches):,} leaves, "
              f"{calls_sum//len(searches):,} C-calls")
        print(f"  Per-search max: {ms_max:.0f}ms, {leaves_max:,} leaves")
        print(f"  Leaves/sec: {leaves_sum/(ms_sum/1000):,.0f}")

    summarize("PY (seats 0,2)", py_bot.decisions)
    print()
    summarize(" C (seats 1,3)", c_bot.decisions)

    # Save raw data
    with open("/tmp/c_vs_py_decisions.txt", "w") as f:
        f.write("# bot, type, ms, C_leaves, C_calls\n")
        for d in py_bot.decisions:
            f.write(f"PY,{d['type']},{d['ms']:.2f},{d['C_leaves']},{d['C_calls']}\n")
        for d in c_bot.decisions:
            f.write(f"C ,{d['type']},{d['ms']:.2f},{d['C_leaves']},{d['C_calls']}\n")
    print(f"\n(raw per-decision data: /tmp/c_vs_py_decisions.txt)")


if __name__ == "__main__":
    main()
