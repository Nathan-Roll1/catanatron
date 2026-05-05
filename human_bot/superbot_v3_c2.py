"""SuperBotV3C2: pure-C deep search, NO Python in the inner loop.

Differences from SuperBotV3C:
  - Uses libdeep.dylib's `deep_search_create_c` (C-built-in policy_top_k).
  - Loads the NN model into libdeep's address space unless CATAN_POLICY_ALGO=1
    selects the pure algorithmic policy_top_k path.
  - Builds a StateEncoderC from the game's map.

The Python pick() still does:
  - Forced-move check
  - Terminal-win check
  - Low-entropy fast path (NN policy mode only)
  - Top-K root candidates (calls libdeep.policy_top_k once)

But the recursive deep_search runs entirely in C.

Verification: should produce identical moves to SuperBotV3C on the same seed.
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import numpy as np

from human_bot._test_c_encoder import StateEncoderC
from human_bot.superbot_v3_c import (
    DeepSearchStats, PolicyTopKFn, _LIB_PATH as _DEEP_LIB_PATH,
)

AD = 337
MASK_DIM = 397


class SuperBotV3C2:
    def __init__(self,
                 weights_path: str,
                 our_depth: int = 5,
                 top_k_schedule: tuple[int, ...] = (10, 7, 5, 4, 3),
                 entropy_fast_thresh: float = 0.15,
                 leaf_cache_bits: int = 20,
                 time_budget_ms: float = 5000,
                 opponent_ab_depth: int = 2,
                 leaf_mode: int = 0,
                 algo_policy: bool | None = None,
                 opponent_model: str = "ab2",
                 algo_flags: int | None = None,
                 algo_value_tiebreak: bool | None = None,
                 robust_opponent_model: str | None = None,
                 robust_penalty_weight: float = 0.5,
                 leaf_pressure_weight: float | None = None,
                 leaf_threat_bonus: float | None = None,
                 leaf_threat_vp: int = 8,
                 endgame_extra_depth: int = 0,
                 endgame_vp_threshold: int = 8,
                 threat_extra_depth: int = 0,
                 threat_vp_threshold: int = 8,
                 threat_opp_ab_depth: int | None = None,
                 iterative_deepening: bool = False,
                 iter_start_depth: int = 2,
                 critical_vp_threshold: int = 100,
                 critical_extra_depth: int = 0):
        from hexzero.bindings.lib_loader import load_library
        from hexzero.bindings.structs import (
            Action as CAction, MAX_ACTIONS,
        )
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        self._catan_lib = load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        self._CAction = CAction

        # Load libdeep (which contains nn.c, state_encode.c, policy_topk.c)
        self._libdeep = ctypes.CDLL(_DEEP_LIB_PATH)
        self._setup_libdeep_bindings()

        self._algo_policy = (
            os.environ.get("CATAN_POLICY_ALGO", "") not in ("", "0")
            if algo_policy is None else bool(algo_policy)
        )
        if algo_flags is None:
            flags_env = os.environ.get("CATAN_POLICY_ALGO_FLAGS", "")
            if flags_env:
                self._algo_flags = 0
                for part in flags_env.split(","):
                    try:
                        self._algo_flags |= 1 << int(part)
                    except ValueError:
                        pass
            else:
                variant = int(os.environ.get("CATAN_POLICY_ALGO_VARIANT", "0") or 0)
                self._algo_flags = (1 << variant) if 0 < variant <= 5 else 0
        else:
            self._algo_flags = int(algo_flags)
        self._algo_value_tiebreak = (
            os.environ.get("CATAN_POLICY_ALGO_VALUE", "") not in ("", "0")
            if algo_value_tiebreak is None else bool(algo_value_tiebreak)
        )
        # Keep a valid NNModel buffer even for CATAN_POLICY_ALGO=1. Some
        # shared C plumbing still carries this pointer, but the algorithmic
        # policy path does not use NN logits for decisions.
        mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        if not self._algo_policy:
            rc = self._libdeep.nn_load(mptr, weights_path.encode())
            if rc != 0:
                raise RuntimeError(f"nn_load failed: {rc}")
        self._mbuf = mbuf
        self._mptr = mptr

        # Initialize StateEncoderC from a fresh game's map
        self._enc = StateEncoderC()
        self._libdeep.state_encoder_init(
            ctypes.byref(self._enc), ctypes.addressof(g0._game), 4)

        # Create deep_search context using the C-policy constructor
        self._ds_ctx = self._libdeep.deep_search_create_c(
            leaf_cache_bits, mptr, ctypes.byref(self._enc))
        if not self._ds_ctx:
            raise RuntimeError("deep_search_create_c failed")

        schedule_arr = (ctypes.c_int * len(top_k_schedule))(*top_k_schedule)
        self._libdeep.deep_search_configure(
            self._ds_ctx, our_depth, schedule_arr, len(top_k_schedule),
            opponent_ab_depth, time_budget_ms / 1000)
        if hasattr(self._libdeep, "deep_search_set_leaf_mode"):
            self._libdeep.deep_search_set_leaf_mode(self._ds_ctx, leaf_mode)
        if hasattr(self._libdeep, "deep_search_set_algo_policy"):
            self._libdeep.deep_search_set_algo_policy(self._ds_ctx, int(self._algo_policy))
        if opponent_model in ("hs1", "hs-leaf", "h-s-leaf", "algo-leaf"):
            self._opponent_model = 2
        elif opponent_model in ("hs", "h-s", "algo"):
            self._opponent_model = 1
        else:
            self._opponent_model = 0
        if robust_opponent_model in ("hs1", "hs-leaf", "h-s-leaf", "algo-leaf"):
            self._robust_opponent_model = 2
        elif robust_opponent_model in ("hs", "h-s", "algo"):
            self._robust_opponent_model = 1
        else:
            self._robust_opponent_model = 0
        self._robust_penalty_weight = float(robust_penalty_weight)
        if hasattr(self._libdeep, "deep_search_set_opponent_model"):
            self._libdeep.deep_search_set_opponent_model(self._ds_ctx, self._opponent_model)
        self._configure_algo_policy()
        self._endgame_extra_depth = int(endgame_extra_depth)
        self._endgame_vp_threshold = int(endgame_vp_threshold)
        self._threat_extra_depth = int(threat_extra_depth)
        self._threat_vp_threshold = int(threat_vp_threshold)
        self._base_opp_ab_depth = int(opponent_ab_depth)
        self._threat_opp_ab_depth = (
            int(threat_opp_ab_depth) if threat_opp_ab_depth is not None
            else self._base_opp_ab_depth
        )
        self._top_k_schedule = tuple(int(x) for x in top_k_schedule)
        self._time_budget_s = float(time_budget_ms) / 1000.0
        self._leaf_pressure_weight = leaf_pressure_weight
        self._leaf_threat_bonus = leaf_threat_bonus
        self._leaf_threat_vp = int(leaf_threat_vp)
        self._reapply_value_knobs()
        self._iterative = bool(iterative_deepening)
        self._iter_start_depth = int(iter_start_depth)
        self._critical_vp_threshold = int(critical_vp_threshold)
        self._critical_extra_depth = int(critical_extra_depth)
        if hasattr(self._libdeep, "deep_search_set_iterative"):
            self._libdeep.deep_search_set_iterative(
                self._ds_ctx, int(self._iterative), int(self._iter_start_depth))
        if hasattr(self._libdeep, "deep_search_set_critical_extension"):
            self._libdeep.deep_search_set_critical_extension(
                self._ds_ctx, int(self._critical_vp_threshold),
                int(self._critical_extra_depth))

        self._our_depth = our_depth
        self._top_k_root = top_k_schedule[0]
        self._entropy_thresh = entropy_fast_thresh
        self._leaf_mode = leaf_mode

        # Scratch buffers for the root-level Python operations
        FP = ctypes.POINTER(ctypes.c_float)
        self._FP = FP
        self._nf = np.zeros((54, 18), dtype=np.float32)
        self._ef = np.zeros((144, 5), dtype=np.float32)
        self._ff = np.zeros(115, dtype=np.float32)
        self._mk = np.zeros(MASK_DIM, dtype=np.float32)
        self._out = np.zeros(4 + MASK_DIM, dtype=np.float32)
        self._nfp = self._nf.ctypes.data_as(FP)
        self._efp = self._ef.ctypes.data_as(FP)
        self._ffp = self._ff.ctypes.data_as(FP)
        self._mkp = self._mk.ctypes.data_as(FP)
        self._outp = self._out.ctypes.data_as(FP)

        # Stats
        self.n_decisions = 0
        self.n_forced = 0
        self.n_lowH = 0
        self.n_search = 0
        self.n_terminal_win = 0
        self.n_robust = 0
        self.t_total = 0.0

    def _setup_libdeep_bindings(self):
        FP = ctypes.POINTER(ctypes.c_float)

        # nn_load
        self._libdeep.nn_load.restype = ctypes.c_int
        self._libdeep.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        # nn_forward (for our root-level entropy check)
        self._libdeep.nn_forward.restype = None
        self._libdeep.nn_forward.argtypes = [
            ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]

        # state_encoder_init
        self._libdeep.state_encoder_init.restype = None
        self._libdeep.state_encoder_init.argtypes = [
            ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_int]

        # policy_top_k
        self._libdeep.policy_top_k.restype = ctypes.c_int
        self._libdeep.policy_top_k.argtypes = [
            ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            FP, FP, FP, FP, FP]
        if hasattr(self._libdeep, "policy_top_k_ex"):
            self._libdeep.policy_top_k_ex.restype = ctypes.c_int
            self._libdeep.policy_top_k_ex.argtypes = [
                ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                FP, FP, FP, FP, FP, ctypes.c_int]
        if hasattr(self._libdeep, "policy_algo_configure"):
            self._libdeep.policy_algo_configure.restype = None
            self._libdeep.policy_algo_configure.argtypes = [
                ctypes.c_int, ctypes.c_int]

        # deep_search_create_c
        self._libdeep.deep_search_create_c.restype = ctypes.c_void_p
        self._libdeep.deep_search_create_c.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(StateEncoderC)]
        # deep_search_destroy
        self._libdeep.deep_search_destroy.restype = None
        self._libdeep.deep_search_destroy.argtypes = [ctypes.c_void_p]
        # deep_search_configure
        self._libdeep.deep_search_configure.restype = None
        self._libdeep.deep_search_configure.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
            ctypes.c_int, ctypes.c_int, ctypes.c_double]
        if hasattr(self._libdeep, "deep_search_set_leaf_mode"):
            self._libdeep.deep_search_set_leaf_mode.restype = None
            self._libdeep.deep_search_set_leaf_mode.argtypes = [
                ctypes.c_void_p, ctypes.c_int]
        if hasattr(self._libdeep, "deep_search_set_algo_policy"):
            self._libdeep.deep_search_set_algo_policy.restype = None
            self._libdeep.deep_search_set_algo_policy.argtypes = [
                ctypes.c_void_p, ctypes.c_int]
        if hasattr(self._libdeep, "deep_search_set_opponent_model"):
            self._libdeep.deep_search_set_opponent_model.restype = None
            self._libdeep.deep_search_set_opponent_model.argtypes = [
                ctypes.c_void_p, ctypes.c_int]
        if hasattr(self._libdeep, "value_set_pressure_weight"):
            self._libdeep.value_set_pressure_weight.restype = None
            self._libdeep.value_set_pressure_weight.argtypes = [ctypes.c_double]
        if hasattr(self._libdeep, "value_set_threat_bonus"):
            self._libdeep.value_set_threat_bonus.restype = None
            self._libdeep.value_set_threat_bonus.argtypes = [
                ctypes.c_double, ctypes.c_int]
        if hasattr(self._libdeep, "deep_search_set_iterative"):
            self._libdeep.deep_search_set_iterative.restype = None
            self._libdeep.deep_search_set_iterative.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        if hasattr(self._libdeep, "deep_search_set_critical_extension"):
            self._libdeep.deep_search_set_critical_extension.restype = None
            self._libdeep.deep_search_set_critical_extension.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        # deep_search_root
        self._libdeep.deep_search_root.restype = ctypes.c_double
        self._libdeep.deep_search_root.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int)]
        # deep_search_root_full (per-candidate exact values, no root AB pruning)
        if hasattr(self._libdeep, "deep_search_root_full"):
            self._libdeep.deep_search_root_full.restype = ctypes.c_double
            self._libdeep.deep_search_root_full.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double)]
        # stats
        self._libdeep.deep_search_get_stats.restype = None
        self._libdeep.deep_search_get_stats.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(DeepSearchStats)]

    def __del__(self):
        try:
            if hasattr(self, '_ds_ctx') and self._ds_ctx:
                self._libdeep.deep_search_destroy(self._ds_ctx)
        except Exception:
            pass

    def _configure_algo_policy(self):
        if self._algo_policy and hasattr(self._libdeep, "policy_algo_configure"):
            self._libdeep.policy_algo_configure(
                self._algo_flags, int(self._algo_value_tiebreak))

    def _set_opponent_model(self, model: int):
        if hasattr(self._libdeep, "deep_search_set_opponent_model"):
            self._libdeep.deep_search_set_opponent_model(self._ds_ctx, model)

    def _reapply_value_knobs(self):
        if (
            self._leaf_pressure_weight is not None
            and hasattr(self._libdeep, "value_set_pressure_weight")
        ):
            self._libdeep.value_set_pressure_weight(
                float(self._leaf_pressure_weight))
        if (
            self._leaf_threat_bonus is not None
            and hasattr(self._libdeep, "value_set_threat_bonus")
        ):
            self._libdeep.value_set_threat_bonus(
                float(self._leaf_threat_bonus), int(self._leaf_threat_vp))

    def _max_enemy_vp(self, game, seat):
        return max(
            (
                int(game._game.state.player_state[s][0])
                for s in range(4)
                if s != seat
            ),
            default=0,
        )

    def _own_vp(self, game, seat):
        return int(game._game.state.player_state[seat][0])

    def _apply_adaptive_search(self, game, seat):
        depth = self._our_depth
        opp_ab = self._base_opp_ab_depth
        max_enemy_vp = self._max_enemy_vp(game, seat)
        own_vp = self._own_vp(game, seat)
        if (
            self._endgame_extra_depth
            and own_vp >= self._endgame_vp_threshold
        ):
            depth += self._endgame_extra_depth
        if (
            self._threat_extra_depth
            and max_enemy_vp >= self._threat_vp_threshold
        ):
            depth += self._threat_extra_depth
        if (
            self._threat_opp_ab_depth != self._base_opp_ab_depth
            and max_enemy_vp >= self._threat_vp_threshold
        ):
            opp_ab = self._threat_opp_ab_depth
        if depth == self._our_depth and opp_ab == self._base_opp_ab_depth:
            return False
        schedule_arr = (ctypes.c_int * len(self._top_k_schedule))(*self._top_k_schedule)
        self._libdeep.deep_search_configure(
            self._ds_ctx,
            int(depth),
            schedule_arr,
            len(self._top_k_schedule),
            int(opp_ab),
            self._time_budget_s,
        )
        return True

    def _restore_search(self):
        schedule_arr = (ctypes.c_int * len(self._top_k_schedule))(*self._top_k_schedule)
        self._libdeep.deep_search_configure(
            self._ds_ctx,
            int(self._our_depth),
            schedule_arr,
            len(self._top_k_schedule),
            int(self._base_opp_ab_depth),
            self._time_budget_s,
        )

    def _nn_forward_raw(self, game, le):
        """Run NN via libdeep's nn_forward (for entropy check at root)."""
        self._se.encode_into(game.get_state_view(), self._nf, self._ef, self._ff)
        mn = self._ae.get_action_mask(le).numpy()
        self._mk[:] = 0
        self._mk[:len(mn)] = mn
        self._libdeep.nn_forward(
            self._mptr, self._nfp, self._efp, self._ffp, self._mkp,
            ctypes.cast(self._outp, ctypes.c_void_p))
        return self._out[4:4 + AD], mn

    def _c_policy_top_k(self, game, le, k):
        """Call libdeep's policy_top_k for the root-level top-K."""
        from hexzero.bindings.structs import Action as CAction, MAX_ACTIONS
        n = len(le)
        actions_arr = (CAction * MAX_ACTIONS)()
        for i, a in enumerate(le):
            actions_arr[i] = a
        out_indices = (ctypes.c_int * 64)()
        if hasattr(self._libdeep, "policy_top_k_ex"):
            self._configure_algo_policy()
            n_top = self._libdeep.policy_top_k_ex(
                ctypes.byref(self._enc), self._mptr,
                ctypes.addressof(game._game),
                ctypes.cast(actions_arr, ctypes.c_void_p),
                n, k, out_indices,
                self._nfp, self._efp, self._ffp, self._mkp, self._outp,
                int(self._algo_policy))
        else:
            n_top = self._libdeep.policy_top_k(
                ctypes.byref(self._enc), self._mptr,
                ctypes.addressof(game._game),
                ctypes.cast(actions_arr, ctypes.c_void_p),
                n, k, out_indices,
                self._nfp, self._efp, self._ffp, self._mkp, self._outp)
        return [out_indices[i] for i in range(n_top)]

    def pick(self, game) -> int:
        from human_bot.search_heuristics import fix_robber_steal
        t_start = time.perf_counter()
        self.n_decisions += 1

        le = game.get_legal_actions()
        if not le:
            return -1
        if len(le) == 1:
            self.n_forced += 1
            self.t_total += time.perf_counter() - t_start
            return 0

        seat = game.current_player()

        # Terminal-winning move
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.n_terminal_win += 1
                self.t_total += time.perf_counter() - t_start
                return i

        # Low-entropy fast path. Disabled for CATAN_POLICY_ALGO=1 so the
        # algorithmic player has no NN-dependent decision path.
        is_critical = any(a.type in {1, 3, 4, 5, 6} for a in le)
        if not self._algo_policy and not is_critical:
            lo, mn = self._nn_forward_raw(game, le)
            lo_masked = lo.copy()
            lo_masked[mn[:AD] < 0.5] = -1e9
            a2i = {}
            for i, a in enumerate(le):
                try: a2i[self._ae.encode(a)] = i
                except ValueError: continue
            if a2i:
                encs = np.array(list(a2i.keys()), dtype=np.int64)
                scores = lo_masked[encs]
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum() + 1e-12
                ent = -float((probs * np.log(probs + 1e-12)).sum())
                norm_ent = ent / max(np.log(len(encs)), 1e-9)
                if norm_ent < self._entropy_thresh:
                    self.n_lowH += 1
                    chosen = a2i[int(encs[int(np.argmax(probs))])]
                    self.t_total += time.perf_counter() - t_start
                    return chosen

        # Root: get top-K candidates via C policy
        self.n_search += 1
        self._configure_algo_policy()
        self._reapply_value_knobs()
        candidates = self._c_policy_top_k(game, le, self._top_k_root)
        K = len(candidates)
        if K == 0:
            return 0

        # Recursive search in C (no Python crossings)
        our_color = int(game._game.state.colors[seat])
        cand_arr = (ctypes.c_int * K)(*candidates)
        best_idx_out = ctypes.c_int(-1)
        game_addr = ctypes.addressof(game._game)
        adaptive_applied = self._apply_adaptive_search(game, seat)

        if self._robust_opponent_model and hasattr(self._libdeep, "deep_search_root_full"):
            self.n_robust += 1
            base_values = (ctypes.c_double * K)(*([0.0] * K))
            robust_values = (ctypes.c_double * K)(*([0.0] * K))
            base_best = ctypes.c_int(-1)
            robust_best = ctypes.c_int(-1)
            self._set_opponent_model(self._opponent_model)
            self._configure_algo_policy()
            self._libdeep.deep_search_root_full(
                self._ds_ctx, game_addr, our_color,
                cand_arr, K, ctypes.byref(base_best), base_values)
            self._set_opponent_model(self._robust_opponent_model)
            self._configure_algo_policy()
            self._libdeep.deep_search_root_full(
                self._ds_ctx, game_addr, our_color,
                cand_arr, K, ctypes.byref(robust_best), robust_values)
            self._set_opponent_model(self._opponent_model)

            best_score = -3.0
            best_pi = 0
            for i in range(K):
                base_v = float(base_values[i])
                robust_v = float(robust_values[i])
                if base_v < -1.5:
                    continue
                collapse = min(0.0, robust_v - base_v) if robust_v > -1.5 else -1.0
                score = base_v + self._robust_penalty_weight * collapse
                if score > best_score:
                    best_score = score
                    best_pi = i
            best_idx_out.value = best_pi
        else:
            self._set_opponent_model(self._opponent_model)
            self._libdeep.deep_search_root(
                self._ds_ctx, game_addr, our_color,
                cand_arr, K, ctypes.byref(best_idx_out))

        best_pi = max(0, best_idx_out.value)
        chosen = fix_robber_steal(candidates[best_pi], le)
        if adaptive_applied:
            self._restore_search()
        self.t_total += time.perf_counter() - t_start
        return chosen

    def pick_full(self, game):
        """Same as pick(), but also returns per-candidate search values for
        dense training signal.

        Returns a dict:
            {
                "chosen": int,                     # index into le
                "kind": str,                       # 'forced'|'terminal'|'lowH'|'search'
                "candidates": list[int] | None,    # indices into le (search only)
                "values": list[float] | None,      # per-candidate values (search only)
                "policy_logits": np.ndarray | None # NN policy logits (lowH only)
                "mask_full": np.ndarray            # 397-dim legal mask
            }

        For 'search' kind, candidates[i] is an index in `le` and values[i]
        is the exact minimax value of taking le[candidates[i]] from this
        state. The chosen action is candidates[argmax(values)].

        For 'lowH' (low-entropy fast path), the policy_logits ARE the soft
        target — we trust the NN's confident pick.

        For 'terminal' (immediate-winning move available), candidates contains
        just the winning move with value = +1.0.

        For 'forced' (only one legal action), nothing useful to learn.
        """
        from human_bot.search_heuristics import fix_robber_steal
        from hexzero.bindings.structs import Action as CAction, MAX_ACTIONS

        t_start = time.time()
        self.n_decisions += 1

        le = game.get_legal_actions()
        n_le = len(le)
        mn_full = self._ae.get_action_mask(le).numpy()
        mask_full = np.zeros(MASK_DIM, dtype=np.float32)
        mask_full[:len(mn_full)] = mn_full

        if n_le == 0:
            return {"chosen": -1, "kind": "forced", "candidates": None,
                    "values": None, "policy_logits": None,
                    "mask_full": mask_full}
        if n_le == 1:
            self.n_forced += 1
            self.t_total += time.time() - t_start
            return {"chosen": 0, "kind": "forced", "candidates": None,
                    "values": None, "policy_logits": None,
                    "mask_full": mask_full}

        seat = game.current_player()

        # Terminal-winning move
        for i in range(n_le):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.n_terminal_win += 1
                self.t_total += time.time() - t_start
                return {"chosen": i, "kind": "terminal",
                        "candidates": [i], "values": [1.0],
                        "policy_logits": None, "mask_full": mask_full}

        # Low-entropy fast path. Disabled for CATAN_POLICY_ALGO=1.
        is_critical = any(a.type in {1, 3, 4, 5, 6} for a in le)
        if not self._algo_policy and not is_critical:
            lo, mn = self._nn_forward_raw(game, le)
            lo_masked = lo.copy()
            lo_masked[mn[:AD] < 0.5] = -1e9
            a2i = {}
            for i, a in enumerate(le):
                try: a2i[self._ae.encode(a)] = i
                except ValueError: continue
            if a2i:
                encs = np.array(list(a2i.keys()), dtype=np.int64)
                scores = lo_masked[encs]
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum() + 1e-12
                ent = -float((probs * np.log(probs + 1e-12)).sum())
                norm_ent = ent / max(np.log(len(encs)), 1e-9)
                if norm_ent < self._entropy_thresh:
                    self.n_lowH += 1
                    chosen = a2i[int(encs[int(np.argmax(probs))])]
                    self.t_total += time.time() - t_start
                    # Return the NN policy distribution as the soft target
                    return {"chosen": chosen, "kind": "lowH",
                            "candidates": None, "values": None,
                            "policy_logits": lo_masked,  # full 337-dim
                            "mask_full": mask_full}

        # Deep search path
        self.n_search += 1
        self._configure_algo_policy()
        candidates = self._c_policy_top_k(game, le, self._top_k_root)
        K = len(candidates)
        if K == 0:
            return {"chosen": 0, "kind": "forced", "candidates": None,
                    "values": None, "policy_logits": None,
                    "mask_full": mask_full}

        our_color = int(game._game.state.colors[seat])
        cand_arr = (ctypes.c_int * K)(*candidates)
        best_idx_out = ctypes.c_int(-1)
        values_arr = (ctypes.c_double * K)(*([0.0] * K))
        game_addr = ctypes.addressof(game._game)

        if hasattr(self._libdeep, "deep_search_root_full"):
            self._libdeep.deep_search_root_full(
                self._ds_ctx, game_addr, our_color,
                cand_arr, K, ctypes.byref(best_idx_out), values_arr)
            values = [float(values_arr[i]) for i in range(K)]
        else:
            # Fallback: use deep_search_root (no per-candidate values exposed)
            self._libdeep.deep_search_root(
                self._ds_ctx, game_addr, our_color,
                cand_arr, K, ctypes.byref(best_idx_out))
            values = None

        best_pi = max(0, best_idx_out.value)
        chosen = fix_robber_steal(candidates[best_pi], le)
        self.t_total += time.time() - t_start
        return {"chosen": chosen, "kind": "search",
                "candidates": list(candidates), "values": values,
                "policy_logits": None, "mask_full": mask_full}

    def stats_summary(self):
        d = max(self.n_decisions, 1)
        s = DeepSearchStats()
        self._libdeep.deep_search_get_stats(self._ds_ctx, ctypes.byref(s))
        ch = s.n_cache_hits + s.n_cache_misses
        return (f"d={self.n_decisions} forced={self.n_forced} "
                f"lowH={self.n_lowH} search={self.n_search} "
                f"win_short={self.n_terminal_win} robust={self.n_robust} "
                f"C_calls={s.n_calls} C_leaves={s.n_leaves} "
                f"C_pruned={s.n_pruned} C_termshort={s.n_terminal_short} "
                f"leaf_cache={100*s.n_cache_hits/max(ch,1):.0f}%({ch}) "
                f"ms/dec={1000*self.t_total/d:.1f}")


def _play_one_game(args):
    """Worker: play one game with SuperBotV3C2."""
    (game_idx, seed, nn_seat, weights_path, our_depth, k_schedule,
     entropy_thresh, time_budget_ms) = args

    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import ctypes as C
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.game.interface import CatanGame
    from human_bot.superbot_v3_c2 import SuperBotV3C2

    lib = load_library()
    bot = SuperBotV3C2(weights_path,
                       our_depth=our_depth,
                       top_k_schedule=k_schedule,
                       entropy_fast_thresh=entropy_thresh,
                       time_budget_ms=time_budget_ms,
                       leaf_cache_bits=20)

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
            game.step(bot.pick(game))
        else:
            game.step(ab2_choose(game, le))

    dt = time.time() - t0
    w = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    return game_idx, nn_seat, w, vps, game.turn_number, dt, bot.stats_summary()


def benchmark_arena(num_games=5, num_workers=5, our_depth=5,
                    k_schedule=(10, 7, 5, 4, 3), time_budget_ms=2000,
                    seed_base=95000, weights="csrc/nn_weights_m2.bin"):
    import multiprocessing as mp

    weights_path = os.path.abspath(weights)
    jobs = []
    for gi in range(num_games):
        nn_seat = gi % 4
        jobs.append((gi, seed_base + gi, nn_seat, weights_path, our_depth,
                     k_schedule, 0.15, time_budget_ms))

    print(f"=== SuperBotV3C2 arena (PURE C, no Python in inner loop) ===")
    print(f"  Games: {num_games} (1v3, random seat)")
    print(f"  Workers: {num_workers}")
    print(f"  Depth: {our_depth} k={k_schedule}")
    print(f"  Time/decision: {time_budget_ms}ms cap")
    print()

    ctx = mp.get_context("spawn")
    nn_wins = ab2_wins = 0
    rank_sum = 0
    nn_vp_sum = ab_vp_sum = 0
    completed = 0
    t_start = time.time()
    last_stats = ""

    with ctx.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(_play_one_game, jobs):
            gi, nn_seat, w, vps, turns, dt, stats = result
            completed += 1
            last_stats = stats
            nn_vp = vps[nn_seat]
            opp_avg = sum(v for s, v in enumerate(vps) if s != nn_seat) / 3
            nn_vp_sum += nn_vp
            ab_vp_sum += opp_avg
            rank = sorted(range(4), key=lambda s: -vps[s]).index(nn_seat) + 1
            rank_sum += rank
            if w == nn_seat: nn_wins += 1
            elif w is not None: ab2_wins += 1

            elapsed = time.time() - t_start
            wr = nn_wins / max(nn_wins + ab2_wins, 1)
            print(f"  [{completed:>3d}/{num_games}] g{gi} seat={nn_seat} "
                  f"W={w} VP={nn_vp}/{int(opp_avg)} rank={rank} "
                  f"({turns}t {dt:.0f}s) | WR={wr:.0%} avg_rank={rank_sum/completed:.2f} "
                  f"[{elapsed:.0f}s wall]", flush=True)

    elapsed = time.time() - t_start
    total = nn_wins + ab2_wins
    wr = nn_wins / max(total, 1)
    print(f"\n===== RESULTS =====")
    print(f"  Wins:      {nn_wins}/{total} ({wr:.1%})")
    print(f"  Avg rank:  {rank_sum/num_games:.2f} / 4")
    print(f"  Avg VP:    NN={nn_vp_sum/num_games:.2f}  opp={ab_vp_sum/num_games:.2f}")
    print(f"  Wall time: {elapsed:.1f}s ({num_games/elapsed*60:.1f} g/min)")
    print(f"  Last stats: {last_stats}")
    return wr, elapsed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--k-schedule", type=str, default="10,7,5,4,3")
    p.add_argument("--time-ms", type=int, default=2000)
    p.add_argument("--seed-base", type=int, default=95000)
    args = p.parse_args()
    schedule = tuple(int(x) for x in args.k_schedule.split(","))
    benchmark_arena(args.games, args.workers, args.depth,
                    schedule, args.time_ms, args.seed_base)
