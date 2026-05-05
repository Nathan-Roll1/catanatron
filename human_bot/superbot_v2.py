"""SuperBot v2: deeper search modeling opponents as AB2 explicitly.

Key insight from research: opponents in deployment ARE AB2. Modeling them
as AB2 in our search rollouts is exactly correct (Sturtevant 2008: wrong
opponent model = worse than random; right model = real Elo gain).

Algorithm:
  1. Policy top-K at root (k=5).
  2. For each candidate, fast-forward the game:
     - Our seat: take NN argmax (best-effort policy continuation).
     - Opponent seats: take AB2 1-ply (matches deployment exactly,
       50-100x cheaper than full AB2 2-ply but correlated).
  3. After D plies, evaluate leaf with AB2 base_value_fn.
  4. MCTS-Solver: short-circuit on terminal wins.
  5. Forced-move + low-entropy fast paths.
  6. Leaf cache (Zobrist-style).

This trades the recursive same-turn AB (which only deepens within our
own turn, often only 1-3 plies before END_TURN) for an opponent-modeled
rollout that simulates D full game plies including opponents.
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import numpy as np

AD = 337
MASK_DIM = 397
WIN_VAL = 1e18
LOSS_VAL = -1e18


def _state_hash_fast(game) -> int:
    """Fast position hash for the leaf cache."""
    st = game._game.state
    h = 0xcbf29ce484222325
    bld = ctypes.string_at(ctypes.addressof(st.board.buildings), 96)
    for b in bld:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    rd = ctypes.string_at(ctypes.addressof(st.board.road_owner), 288)
    for b in rd[::4]:  # subsample for speed
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    for p in range(4):
        vp = int(st.player_state[p][0])
        h ^= (vp << 8) | p
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    h ^= (int(st.current_player_index) << 16) | int(st.current_prompt) | (int(st.num_turns) << 32)
    return h & 0xFFFFFFFFFFFFFFFF


class SuperBotV2:
    def __init__(self,
                 weights_path: str,
                 search_depth: int = 12,
                 top_k: int = 5,
                 entropy_fast_thresh: float = 0.15,
                 use_leaf_cache: bool = True,
                 leaf_cache_bits: int = 18):
        from hexzero.bindings.lib_loader import load_library
        from hexzero.bindings.structs import (
            Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
            Game as CGame,
        )
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        self._lib = load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        self._CAction = CAction
        self._CGame = CGame
        self._MAX_ACTIONS = MAX_ACTIONS

        # NN
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

        # AB2 search context for opponent modeling
        self._ab_ctx = SearchCtx()
        self._ab_buf = (CAction * MAX_ACTIONS)()
        self._ab_eval = ValueFn(self._lib.base_value_fn)
        # 1-ply AB scratch
        self._ch = CGame()
        self._ca = (CAction * MAX_ACTIONS)()
        self._cn = ctypes.c_int(0)

        self._depth = search_depth
        self._k = top_k
        self._entropy_thresh = entropy_fast_thresh

        # Cache
        self._cache_size = 1 << leaf_cache_bits if use_leaf_cache else 0
        self._cache_mask = self._cache_size - 1 if use_leaf_cache else 0
        if use_leaf_cache:
            self._cache_keys = np.zeros(self._cache_size, dtype=np.uint64)
            self._cache_vals = np.zeros(self._cache_size, dtype=np.float64)

        # Scratch buffers
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

        # Stats
        self.n_decisions = 0
        self.n_forced = 0
        self.n_lowH = 0
        self.n_search = 0
        self.n_terminal_win_short = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.t_total = 0.0

    def _nn_forward(self, game, le):
        self._se.encode_into(game.get_state_view(), self._nf, self._ef, self._ff)
        mn = self._ae.get_action_mask(le).numpy()
        self._mk[:] = 0
        self._mk[:len(mn)] = mn
        self._nn_lib.nn_forward(
            self._mptr, self._nfp, self._efp, self._ffp, self._mkp, self._outp)
        return self._out[4:4 + AD].copy(), mn

    def _nn_argmax(self, game, le):
        if len(le) == 1:
            return 0
        lo, mn = self._nn_forward(game, le)
        lo[mn[:AD] < 0.5] = -1e9
        best_enc = int(np.argmax(lo))
        for i, a in enumerate(le):
            try:
                if self._ae.encode(a) == best_enc:
                    return i
            except ValueError:
                continue
        return 0

    def _nn_topk(self, game, le, k):
        if len(le) <= k:
            return list(range(len(le)))
        lo, mn = self._nn_forward(game, le)
        a2i = {}
        for i, a in enumerate(le):
            try:
                a2i[self._ae.encode(a)] = i
            except ValueError:
                continue
        scored = sorted([(lo[e], i) for e, i in a2i.items()], reverse=True)
        return [i for _, i in scored[:k]]

    def _ab1_choose(self, game, le):
        """Cheap 1-ply AB2 (opponent model). 50x faster than 2-ply."""
        n = len(le)
        if n == 1:
            return 0
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            self._lib.game_copy(ctypes.byref(self._ch), ctypes.byref(cg))
            self._lib.game_execute(
                ctypes.byref(self._ch), act, self._ca, ctypes.byref(self._cn))
            v = float(self._lib.base_value_fn(ctypes.byref(self._ch), bc))
            if v > bv:
                bv = v
                bi = i
        return bi

    def _leaf_eval(self, game, seat: int) -> float:
        if self._cache_size:
            h = _state_hash_fast(game)
            idx = h & self._cache_mask
            if self._cache_keys[idx] == h and h != 0:
                self.cache_hits += 1
                return float(self._cache_vals[idx])
            self.cache_misses += 1

        cg = game._game
        bot_color = cg.state.colors[seat]
        v = float(self._lib.base_value_fn(ctypes.byref(cg), bot_color))

        if self._cache_size:
            self._cache_keys[idx] = h
            self._cache_vals[idx] = v
        return v

    def _rollout(self, game, depth: int, seat: int) -> float:
        """Roll forward `depth` plies. Our seat plays NN argmax, opponents
        play 1-ply AB. Evaluate leaf with AB2 heuristic."""
        for _ in range(depth):
            if game.is_terminal():
                w = game.winner()
                if w == seat:
                    return WIN_VAL
                elif w is not None:
                    return LOSS_VAL
            le = game.get_legal_actions()
            if not le:
                break
            cp = game.current_player()
            if cp == seat:
                idx = self._nn_argmax(game, le)
            else:
                idx = self._ab1_choose(game, le)
            game.step(idx)
        return self._leaf_eval(game, seat)

    def pick(self, game) -> int:
        from human_bot.search_heuristics import fix_robber_steal
        from hexzero.bindings.structs import Action

        t0 = time.perf_counter()
        self.n_decisions += 1

        le = game.get_legal_actions()
        if not le:
            return -1

        # 1. Forced move
        if len(le) == 1:
            self.n_forced += 1
            self.t_total += time.perf_counter() - t0
            return 0

        seat = game.current_player()

        # 2. Terminal-winning move (MCTS-Solver lite)
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.n_terminal_win_short += 1
                self.t_total += time.perf_counter() - t0
                return i

        # 3. Low-entropy fast path on non-critical moves
        is_critical = any(a.type in {1, 3, 4, 5, 6} for a in le)
        if not is_critical:
            lo, mn = self._nn_forward(game, le)
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
                    self.t_total += time.perf_counter() - t0
                    return chosen

        # 4. Top-K + opponent-modeled rollout
        self.n_search += 1
        candidates = self._nn_topk(game, le, self._k)
        K = len(candidates)

        values = np.zeros(K, dtype=np.float64)
        for pi, ci in enumerate(candidates):
            gc = game.clone()
            gc.step(ci)
            values[pi] = self._rollout(gc, self._depth - 1, seat)

        best_pi = int(np.argmax(values))
        chosen = fix_robber_steal(candidates[best_pi], le)
        self.t_total += time.perf_counter() - t0
        return chosen

    def stats_summary(self):
        d = max(self.n_decisions, 1)
        return (f"d={self.n_decisions} forced={self.n_forced} "
                f"win_short={self.n_terminal_win_short} "
                f"lowH={self.n_lowH} search={self.n_search} "
                f"cache={100*self.cache_hits/max(self.cache_hits+self.cache_misses,1):.0f}% "
                f"ms/dec={1000*self.t_total/d:.1f}")


def benchmark_v2(weights="csrc/nn_weights_m2.bin", num_games=100,
                 depth=12, top_k=5, seed_base=80000, verbose=True):
    import ctypes as C
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    bot = SuperBotV2(weights, search_depth=depth, top_k=top_k,
                     entropy_fast_thresh=0.15, use_leaf_cache=True)

    # Proper AB2 opponent
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

    nn_wins = ab2_wins = draws = 0
    nn_vp_sum = ab2_vp_sum = 0.0
    t0 = time.time()
    print(f"SuperBotV2 (d{depth} k{top_k} NN-self + AB1-opp + AB-leaf) "
          f"vs proper AB2 2-ply, {num_games} games\n")

    for gi in range(num_games):
        game = CatanGame(seed=seed_base + gi); game.reset()
        nn_seats = {gi % 4, (gi + 2) % 4}
        ab_seats = {(gi + 1) % 4, (gi + 3) % 4}
        while not game.is_terminal() and game.turn_number < 1000:
            le = game.get_legal_actions()
            if not le: break
            if len(le) == 1: game.step(0); continue
            cp = game.current_player()
            if cp in nn_seats: game.step(bot.pick(game))
            else: game.step(ab2_choose(game, le))

        w = game.winner()
        vps = [game._game.state.player_state[s][0] for s in range(4)]
        nn_vp_sum += sum(vps[s] for s in nn_seats) / 2
        ab2_vp_sum += sum(vps[s] for s in ab_seats) / 2
        if w is not None and w in nn_seats: nn_wins += 1
        elif w is not None: ab2_wins += 1
        else: draws += 1

        if verbose and (gi+1) % 10 == 0:
            elapsed = time.time() - t0
            wr = nn_wins / max(nn_wins + ab2_wins, 1)
            print(f"  {gi+1}/{num_games}: NN={nn_wins} AB2={ab2_wins} "
                  f"WR={wr:.0%} VP(NN={nn_vp_sum/(gi+1):.1f} "
                  f"AB2={ab2_vp_sum/(gi+1):.1f}) ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    total = nn_wins + ab2_wins
    wr = nn_wins / max(total, 1)
    print(f"\n===== RESULTS (d{depth} k{top_k}) =====")
    print(f"  SuperBotV2:   {nn_wins} wins ({wr:.1%})")
    print(f"  AB2 2-ply:    {ab2_wins} wins ({1-wr:.1%})")
    print(f"  Draws:        {draws}")
    print(f"  Avg VP:       NN={nn_vp_sum/num_games:.1f} AB2={ab2_vp_sum/num_games:.1f}")
    print(f"  Speed:        {num_games/elapsed:.2f} g/s ({elapsed:.0f}s)")
    print(f"  Stats:        {bot.stats_summary()}")
    return wr, nn_wins, ab2_wins, elapsed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    benchmark_v2(num_games=args.games, depth=args.depth, top_k=args.top_k)
