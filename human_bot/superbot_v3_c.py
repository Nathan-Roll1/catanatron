"""SuperBotV3 with the inner deep_search loop in C.

The Python policy callback is the only thing that crosses back into Python:
when the C search needs top-K legal actions ranked by NN policy, it calls
us, we run the NN forward (which is itself C via libnn), and return indices.

Everything else — recursion, fast-forward, leaf eval, AB2 opponents,
state hashing, cache — runs in pure C with no FFI per game tree node.

Expected speedup: ~5-15x over the pure-Python SuperBotV3 because:
  - 4,200 leaves/decision × ~10 FFI calls per leaf = 42,000 FFI calls
  - Each FFI call costs ~10-50us round-trip
  - Total FFI overhead: ~1-2 seconds per decision
  - C version: ~K_root FFI calls per decision (only for policy_topk)
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import numpy as np

AD = 337
MASK_DIM = 397


# ── ctypes bindings for libdeep.dylib ────────────────────────────
_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "csrc", "libdeep.dylib")

# Policy callback signature: int (*)(void* ud, Game* g, Action* acts, int n,
#                                     int k, int* out)
PolicyTopKFn = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))


class DeepSearchStats(ctypes.Structure):
    _fields_ = [
        ("n_calls", ctypes.c_long),
        ("n_leaves", ctypes.c_long),
        ("n_pruned", ctypes.c_long),
        ("n_terminal_short", ctypes.c_long),
        ("n_cache_hits", ctypes.c_long),
        ("n_cache_misses", ctypes.c_long),
        ("n_pcache_hits", ctypes.c_long),
        ("n_pcache_misses", ctypes.c_long),
        ("n_root_early_exits", ctypes.c_long),
    ]


def _load_libdeep():
    lib = ctypes.CDLL(_LIB_PATH)
    lib.deep_search_create.restype = ctypes.c_void_p
    lib.deep_search_create.argtypes = [ctypes.c_int, ctypes.c_void_p, PolicyTopKFn]

    lib.deep_search_destroy.restype = None
    lib.deep_search_destroy.argtypes = [ctypes.c_void_p]

    lib.deep_search_configure.restype = None
    lib.deep_search_configure.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ]

    lib.deep_search_root.restype = ctypes.c_double
    lib.deep_search_root.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]

    lib.deep_search_get_stats.restype = None
    lib.deep_search_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(DeepSearchStats)]

    lib.deep_search_reset_stats.restype = None
    lib.deep_search_reset_stats.argtypes = [ctypes.c_void_p]

    return lib


class SuperBotV3C:
    """SuperBotV3 with the recursion in C. Same algorithm, ~10x faster."""

    def __init__(self,
                 weights_path: str,
                 our_depth: int = 5,
                 top_k_schedule: tuple[int, ...] = (10, 7, 5, 4, 3),
                 entropy_fast_thresh: float = 0.15,
                 leaf_cache_bits: int = 20,
                 time_budget_ms: float = 5000,
                 opponent_ab_depth: int = 2):
        from hexzero.bindings.lib_loader import load_library
        from hexzero.bindings.structs import Action as CAction
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        self._catan_lib = load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        self._CAction = CAction

        # NN inference (libnn.dylib)
        FP = ctypes.POINTER(ctypes.c_float)
        nn_lib = ctypes.CDLL(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "csrc", "libnn.dylib"))
        nn_lib.nn_load.restype = ctypes.c_int
        nn_lib.nn_forward.restype = None
        nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
        mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        rc = nn_lib.nn_load(mptr, weights_path.encode())
        if rc != 0:
            raise RuntimeError(f"nn_load failed: {rc}")
        self._nn_lib = nn_lib
        self._mbuf = mbuf
        self._mptr = mptr

        # Scratch buffers for NN forward
        self._nf = np.zeros((54, 18), dtype=np.float32)
        self._ef = np.zeros((144, 5), dtype=np.float32)
        self._ff = np.zeros(115, dtype=np.float32)
        self._mk = np.zeros(MASK_DIM, dtype=np.float32)
        self._out = np.zeros(4 + MASK_DIM, dtype=np.float32)
        self._nfp = self._nf.ctypes.data_as(FP)
        self._efp = self._ef.ctypes.data_as(FP)
        self._ffp = self._ff.ctypes.data_as(FP)
        self._mkp = self._mk.ctypes.data_as(FP)
        self._outp = self._out.ctypes.data_as(ctypes.c_void_p)

        # libdeep.dylib
        self._libdeep = _load_libdeep()

        # The policy callback. Must keep a reference to prevent GC.
        self._policy_cb = PolicyTopKFn(self._policy_topk_callback)

        self._ds_ctx = self._libdeep.deep_search_create(
            leaf_cache_bits, None, self._policy_cb)
        if not self._ds_ctx:
            raise RuntimeError("deep_search_create failed")

        schedule_arr = (ctypes.c_int * len(top_k_schedule))(*top_k_schedule)
        self._libdeep.deep_search_configure(
            self._ds_ctx, our_depth, schedule_arr, len(top_k_schedule),
            opponent_ab_depth, time_budget_ms / 1000)

        self._our_depth = our_depth
        self._top_k_root = top_k_schedule[0]
        self._entropy_thresh = entropy_fast_thresh
        self._time_budget_ms = time_budget_ms

        # Stats
        self.n_decisions = 0
        self.n_forced = 0
        self.n_lowH = 0
        self.n_search = 0
        self.n_terminal_win = 0
        self.t_total = 0.0

    def __del__(self):
        try:
            if hasattr(self, '_ds_ctx') and self._ds_ctx:
                self._libdeep.deep_search_destroy(self._ds_ctx)
        except Exception:
            pass

    def _nn_forward_raw(self, game, le):
        """Run NN forward, return (logits[AD], legal_mask[MASK_DIM])."""
        self._se.encode_into(game.get_state_view(), self._nf, self._ef, self._ff)
        mn = self._ae.get_action_mask(le).numpy()
        self._mk[:] = 0
        self._mk[:len(mn)] = mn
        self._nn_lib.nn_forward(
            self._mptr, self._nfp, self._efp, self._ffp, self._mkp, self._outp)
        return self._out[4:4 + AD], mn

    def _policy_topk_callback(self, userdata, game_ptr, actions_ptr, n, k, out_ptr):
        """C calls this to get top-K legal action indices ranked by policy."""
        # Construct a CatanGame view over the C Game pointer.
        # We have to reconstruct the high-level wrapper from the C state pointer,
        # because the encoder needs CatanGame.get_state_view().
        # The CatanGame constructor that wraps an existing C struct isn't
        # exposed; instead we'll do the encoding ourselves via the underlying
        # state pointer.
        try:
            from hexzero.bindings.structs import Game as CGame, Action as CAction
            cgame = CGame.from_address(game_ptr)
            actions_array = (CAction * n).from_address(actions_ptr)

            # Use the encoder via a temporary CatanGame-like view.
            # We need a state_view: CatanGame uses self._game which is a CGame.
            # The encoder's encode_into takes a state_view which is the CGame's state.
            # Build a minimal view object that the encoder accepts.
            view = _CGameStateView(cgame)
            self._se.encode_into(view, self._nf, self._ef, self._ff)

            # Build mask from legal actions
            self._mk[:] = 0
            le = [actions_array[i] for i in range(n)]
            mn = self._ae.get_action_mask(le).numpy()
            self._mk[:len(mn)] = mn

            self._nn_lib.nn_forward(
                self._mptr, self._nfp, self._efp, self._ffp, self._mkp, self._outp)

            # Score legal actions by their encoded position in the policy
            # Using the same logic as Python _nn_topk
            logits = self._out[4:4 + AD]
            scored = []
            for i in range(n):
                try:
                    enc = self._ae.encode(actions_array[i])
                    scored.append((logits[enc], i))
                except ValueError:
                    scored.append((-1e9, i))
            scored.sort(reverse=True)
            kk = min(k, len(scored))
            for j in range(kk):
                out_ptr[j] = scored[j][1]
            return kk
        except Exception as e:
            # Fall back to first-k if anything goes wrong
            kk = min(k, n)
            for j in range(kk):
                out_ptr[j] = j
            return kk

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

        # Low-entropy fast path
        is_critical = any(a.type in {1, 3, 4, 5, 6} for a in le)
        if not is_critical:
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

        # Top-K root candidates
        self.n_search += 1
        lo, mn = self._nn_forward_raw(game, le)
        lo_masked = lo.copy()
        lo_masked[mn[:AD] < 0.5] = -1e9
        a2i = {}
        for i, a in enumerate(le):
            try: a2i[self._ae.encode(a)] = i
            except ValueError: continue
        scored = sorted([(lo_masked[e], i) for e, i in a2i.items()], reverse=True)
        K = min(self._top_k_root, len(scored))
        candidates = [scored[j][1] for j in range(K)]

        # Call into C: provide candidate indices, get back best.
        # Note: C wants the COLOR, not the seat index. The colors[] array
        # maps seat index → color enum (set at game init, possibly shuffled).
        our_color = int(game._game.state.colors[seat])
        cand_arr = (ctypes.c_int * K)(*candidates)
        best_idx_out = ctypes.c_int(-1)
        game_addr = ctypes.addressof(game._game)
        self._libdeep.deep_search_root(
            self._ds_ctx, game_addr, our_color,
            cand_arr, K, ctypes.byref(best_idx_out))

        best_pi = max(0, best_idx_out.value)
        chosen = fix_robber_steal(candidates[best_pi], le)
        self.t_total += time.perf_counter() - t_start
        return chosen

    def stats_summary(self):
        d = max(self.n_decisions, 1)
        s = DeepSearchStats()
        self._libdeep.deep_search_get_stats(self._ds_ctx, ctypes.byref(s))
        ch = s.n_cache_hits + s.n_cache_misses
        ph = s.n_pcache_hits + s.n_pcache_misses
        return (f"d={self.n_decisions} forced={self.n_forced} "
                f"lowH={self.n_lowH} search={self.n_search} "
                f"win_short={self.n_terminal_win} "
                f"C_calls={s.n_calls} C_leaves={s.n_leaves} "
                f"C_pruned={s.n_pruned} C_termshort={s.n_terminal_short} "
                f"leaf_cache={100*s.n_cache_hits/max(ch,1):.0f}%({ch}) "
                f"pol_cache={100*s.n_pcache_hits/max(ph,1):.0f}%({ph}) "
                f"ms/dec={1000*self.t_total/d:.1f}")


# ── Helper: Pretend a Game* is a state_view for the encoder ────────
class _CGameStateView:
    """Wraps a C Game pointer to satisfy state_encoder.encode_into.

    The encoder reads attributes directly from the structs.Game wrapper.
    We expose them as properties on this class.
    """
    __slots__ = ("_g",)

    def __init__(self, cgame):
        self._g = cgame

    @property
    def current_player(self): return self._g.state.current_player_index
    @property
    def num_turns(self): return self._g.state.num_turns
    @property
    def is_initial_build_phase(self): return self._g.state.is_initial_build_phase
    @property
    def current_prompt(self): return self._g.state.current_prompt
    @property
    def num_players(self): return self._g.state.num_players
    @property
    def player_state(self):
        ps = self._g.state.player_state
        return np.frombuffer(
            (ctypes.c_int * (4 * 29)).from_address(ctypes.addressof(ps)),
            dtype=np.int32).reshape(4, 29)
    @property
    def buildings(self):
        return np.frombuffer(
            (ctypes.c_int8 * 96).from_address(ctypes.addressof(self._g.state.board.buildings)),
            dtype=np.int8)
    @property
    def road_owners(self):
        return np.frombuffer(
            (ctypes.c_int8 * (96 * 3)).from_address(
                ctypes.addressof(self._g.state.board.road_owner)),
            dtype=np.int8).reshape(96, 3)
    @property
    def color_to_index(self):
        return np.frombuffer(
            (ctypes.c_int * 4).from_address(ctypes.addressof(self._g.state.color_to_index)),
            dtype=np.int32)
    @property
    def robber_coord(self):
        rc = self._g.state.board.robber_coordinate
        return (rc.x, rc.y, rc.z)
    @property
    def resource_bank(self):
        return np.frombuffer(
            (ctypes.c_int * 5).from_address(
                ctypes.addressof(self._g.state.resource_freqdeck)),
            dtype=np.int32)
    @property
    def dev_deck_size(self): return self._g.state.dev_deck_size
    @property
    def tile_resources(self):
        # tiles are on board.map; map has land_tiles[].resource
        m = self._g.state.board.map
        if not m: return np.zeros(19, dtype=np.int8)
        out = np.zeros(19, dtype=np.int8)
        for i in range(19):
            out[i] = m.contents.land_tiles[i].resource
        return out
    @property
    def tile_numbers(self):
        m = self._g.state.board.map
        if not m: return np.zeros(19, dtype=np.int8)
        out = np.zeros(19, dtype=np.int8)
        for i in range(19):
            out[i] = m.contents.land_tiles[i].number
        return out
    @property
    def tile_nodes(self):
        m = self._g.state.board.map
        if not m: return np.zeros((19, 6), dtype=np.int32)
        out = np.zeros((19, 6), dtype=np.int32)
        for i in range(19):
            for j in range(6):
                out[i, j] = m.contents.land_tiles[i].nodes[j]
        return out
    @property
    def is_discarding(self): return self._g.state.is_discarding
    @property
    def is_road_building(self): return self._g.state.is_road_building
    @property
    def is_moving_knight(self): return self._g.state.is_moving_knight
    @property
    def is_resolving_trade(self): return self._g.state.is_resolving_trade


def _play_one_game(args):
    """Worker: play one game, return result."""
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
    from human_bot.superbot_v3_c import SuperBotV3C

    lib = load_library()
    bot = SuperBotV3C(weights_path,
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

    print(f"=== SuperBotV3C arena (C deep_search) ===")
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
            print(f"  [{completed:>2d}/{num_games}] g{gi} seat={nn_seat} "
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
