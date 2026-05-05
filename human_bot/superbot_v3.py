"""SuperBotV3: Deep recursive tree search with policy-pruned branching.

Key insight: since opponents are deterministic AB2 and dice are known,
the game is effectively a DETERMINISTIC MULTI-AGENT TREE where only our
own decisions create branching. This lets us do proper minimax with NN
policy pruning at every one of our decision points.

Algorithm:
    deep_search(game, seat, our_depth_left, top_k):
        # Fast-forward past opponent turns and forced moves
        while not game.is_terminal() and current_player != seat:
            game.step(ab2_choose(game))

        if our_depth_left == 0 or terminal:
            return leaf_value(game, seat)

        # Branch on top-K of our moves by policy
        candidates = nn_top_k(game, legal_actions, top_k)
        return max(
            deep_search(clone_and_step(game, ci), seat, our_depth - 1, top_k)
            for ci in candidates
        )

This explores k^our_depth branches (default 5^4 = 625) with each leaf
~5ms of compute = ~3s per decision. Huge jump from the 24ms of v2.

Enhancements:
  1. Terminal-win early exit (MCTS-Solver).
  2. Leaf cache for transposition hits (49% hit rate observed).
  3. Forced-move fast path (skip search when len(le)==1).
  4. Low-entropy fast path for non-critical turns.
  5. Alpha-beta pruning: track best-sibling-so-far, skip branches that
     can't improve.
  6. Per-depth top-k tapering: k=8 at root, k=5 at depth 1, k=3 deeper
     — matches the "top candidates are more trustworthy than nested ones"
     intuition and halves effective compute.
  7. Time budget: break recursion if wall-clock exceeds deadline.
"""
from __future__ import annotations

import ctypes
import os
import time
from typing import Optional

import numpy as np

AD = 337
MASK_DIM = 397
WIN_VAL = 1.0
LOSS_VAL = -1.0

# Value function normalization: base_value_fn ~ 3e14 * VP + ... ~ 3e15 for VP=10
VALUE_SCALE = 3e15


def _state_hash_fast(game) -> int:
    """Hash of all state inputs to base_value_fn.

    Must include EVERY field that affects base_value_fn or the cache will
    return wrong values for hash-collision states. Verified against value.c:
      - VP (PS_VICTORY_POINTS)
      - settlements + cities (via board.buildings)
      - roads (full)
      - resource counts (PS_*_IN_HAND for 5 resources)
      - dev cards (PS_*_IN_HAND for 5 dev types)
      - played knights (PS_PLAYED_KNIGHT)
      - longest road length (PS_LONGEST_ROAD_LENGTH)
      - robber position
      - current player index + prompt
    """
    st = game._game.state
    h = 0xcbf29ce484222325
    MASK = 0xFFFFFFFFFFFFFFFF
    PRIME = 0x100000001b3

    # Buildings (96 bytes — settlements/cities by node)
    bld = ctypes.string_at(ctypes.addressof(st.board.buildings), 96)
    for b in bld:
        h = ((h ^ b) * PRIME) & MASK

    # Roads (full 288 bytes — ownership at every edge)
    rd = ctypes.string_at(ctypes.addressof(st.board.road_owner), 288)
    for b in rd:
        h = ((h ^ b) * PRIME) & MASK

    # Robber coordinate
    rc = st.board.robber_coordinate
    h = ((h ^ ((rc.x & 0xff) | ((rc.y & 0xff) << 8) | ((rc.z & 0xff) << 16))) * PRIME) & MASK

    # Per-player state: VP, hand resources, hand dev cards, played knight, longest road
    # Indices from catan_types.h: VP=0, RES=14..18, DEV=19..23, PLAYED_KNIGHT=24, LONGEST_ROAD=9
    for p in range(4):
        ps = st.player_state[p]
        # VP (0)
        h = ((h ^ ((int(ps[0]) & 0xff) | (p << 8))) * PRIME) & MASK
        # Longest road length (9)
        h = ((h ^ (int(ps[9]) & 0xff)) * PRIME) & MASK
        # Played knights (24)
        h = ((h ^ (int(ps[24]) & 0xff)) * PRIME) & MASK
        # Resources (14..18) and dev cards in hand (19..23) — 10 fields
        for f in range(14, 24):
            h = ((h ^ (int(ps[f]) & 0xff)) * PRIME) & MASK

    # Current player + prompt + turn number
    h = ((h ^ ((int(st.current_player_index) & 0xff) |
               ((int(st.current_prompt) & 0xff) << 8) |
               ((int(st.num_turns) & 0xffff) << 16))) * PRIME) & MASK

    return h & MASK


class SuperBotV3:
    def __init__(self,
                 weights_path: str,
                 our_depth: int = 4,
                 top_k_schedule: tuple[int, ...] = (8, 5, 4, 3),
                 entropy_fast_thresh: float = 0.15,
                 use_leaf_cache: bool = True,
                 leaf_cache_bits: int = 20,  # 1M entries
                 time_budget_ms: Optional[float] = 3000,
                 use_alpha_beta: bool = True):
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

        self._ab_ctx = SearchCtx()
        self._ab_buf = (CAction * MAX_ACTIONS)()
        self._ab_eval = ValueFn(self._lib.base_value_fn)
        self._ch = CGame()
        self._ca = (CAction * MAX_ACTIONS)()
        self._cn = ctypes.c_int(0)

        self._our_depth = our_depth
        self._top_k_schedule = top_k_schedule
        self._entropy_thresh = entropy_fast_thresh
        self._time_budget_s = time_budget_ms / 1000 if time_budget_ms else float('inf')
        self._use_ab = use_alpha_beta

        self._cache_size = 1 << leaf_cache_bits if use_leaf_cache else 0
        self._cache_mask = self._cache_size - 1 if use_leaf_cache else 0
        if use_leaf_cache:
            self._cache_keys = np.zeros(self._cache_size, dtype=np.uint64)
            self._cache_vals = np.zeros(self._cache_size, dtype=np.float64)

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

        self._deadline = float('inf')

        self.n_decisions = 0
        self.n_forced = 0
        self.n_lowH = 0
        self.n_search = 0
        self.n_terminal_win = 0
        self.n_leaves = 0
        self.n_pruned = 0
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

    def _nn_topk(self, game, le, k):
        if len(le) <= k:
            return list(range(len(le)))
        lo, mn = self._nn_forward(game, le)
        a2i = {}
        for i, a in enumerate(le):
            try: a2i[self._ae.encode(a)] = i
            except ValueError: continue
        scored = sorted([(lo[e], i) for e, i in a2i.items()], reverse=True)
        return [i for _, i in scored[:k]]

    def _ab2_choose(self, game, le):
        """Full AB2 2-ply with expectimax — matches deployment opponent."""
        n = len(le)
        if n == 1: return 0
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le): self._ab_buf[i] = a
        res = self._lib.alphabeta_search(
            ctypes.byref(self._ab_ctx), ctypes.byref(cg), self._ab_buf,
            ctypes.c_int(n), ctypes.c_int(2),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), self._ab_eval)
        cb = ctypes.string_at(ctypes.byref(res.action), ctypes.sizeof(res.action))
        for i, a in enumerate(le):
            if ctypes.string_at(ctypes.byref(a), ctypes.sizeof(a)) == cb:
                return i
        return 0

    def _normalize_value(self, game, seat: int) -> float:
        """Map position value to [-1, 1] for clean minimax math."""
        if game.is_terminal():
            w = game.winner()
            if w == seat: return WIN_VAL
            if w is not None: return LOSS_VAL
            return 0.0

        if self._cache_size:
            h = _state_hash_fast(game)
            idx = h & self._cache_mask
            if self._cache_keys[idx] == h and h != 0:
                self.cache_hits += 1
                return float(self._cache_vals[idx])
            self.cache_misses += 1

        cg = game._game
        bot_color = cg.state.colors[seat]
        raw = float(self._lib.base_value_fn(ctypes.byref(cg), bot_color))
        v = max(-0.99, min(0.99, raw / VALUE_SCALE))

        if self._cache_size:
            self._cache_keys[idx] = h
            self._cache_vals[idx] = v
        return v

    def _fast_forward_to_our_turn(self, game, seat: int) -> bool:
        """Play opponent turns + forced moves until our turn. Return
        True if we reach our turn (False if terminal).
        """
        while not game.is_terminal() and game.turn_number < 500:
            le = game.get_legal_actions()
            if not le:
                return False
            cp = game.current_player()
            if cp == seat and len(le) > 1:
                return True
            if len(le) == 1:
                game.step(0)
                continue
            if cp == seat:
                return True  # len==1 handled above
            game.step(self._ab2_choose(game, le))
        return False

    def _deep_search(self, game, seat: int, our_depth_left: int,
                     alpha: float, beta: float, depth_idx: int) -> float:
        """Recursive minimax-style search, branching only at our turns.
        Returns value in [-1, 1] from seat's perspective.
        """
        if time.perf_counter() >= self._deadline:
            return self._normalize_value(game, seat)

        if not self._fast_forward_to_our_turn(game, seat):
            self.n_leaves += 1
            return self._normalize_value(game, seat)

        if our_depth_left == 0:
            self.n_leaves += 1
            return self._normalize_value(game, seat)

        le = game.get_legal_actions()
        if not le:
            self.n_leaves += 1
            return self._normalize_value(game, seat)

        # Terminal-win shortcut (MCTS-Solver)
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.n_terminal_win += 1
                return WIN_VAL

        # Select k from schedule
        k = self._top_k_schedule[min(depth_idx, len(self._top_k_schedule) - 1)]
        candidates = self._nn_topk(game, le, k)

        best_v = -2.0
        for ci in candidates:
            if time.perf_counter() >= self._deadline:
                break
            gc = game.clone()
            gc.step(ci)
            v = self._deep_search(gc, seat, our_depth_left - 1,
                                   alpha, beta, depth_idx + 1)
            if v > best_v:
                best_v = v
            if self._use_ab:
                alpha = max(alpha, best_v)
                if alpha >= beta:
                    self.n_pruned += (len(candidates) - candidates.index(ci) - 1)
                    break
            if best_v >= WIN_VAL - 1e-6:
                break  # found guaranteed win

        return best_v

    def pick(self, game) -> int:
        from human_bot.search_heuristics import fix_robber_steal
        t_start = time.perf_counter()
        self._deadline = t_start + self._time_budget_s
        self.n_decisions += 1

        le = game.get_legal_actions()
        if not le:
            return -1

        if len(le) == 1:
            self.n_forced += 1
            self.t_total += time.perf_counter() - t_start
            return 0

        seat = game.current_player()

        # Terminal-winning move at root
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
                    self.t_total += time.perf_counter() - t_start
                    return chosen

        # Deep search with branching at our turns
        self.n_search += 1
        k_root = self._top_k_schedule[0]
        candidates = self._nn_topk(game, le, k_root)
        K = len(candidates)

        values = np.full(K, -2.0, dtype=np.float64)
        alpha = -2.0
        for pi, ci in enumerate(candidates):
            if time.perf_counter() >= self._deadline:
                break
            gc = game.clone()
            gc.step(ci)
            v = self._deep_search(gc, seat, self._our_depth - 1,
                                   alpha, 2.0, 1)
            values[pi] = v
            if self._use_ab and v > alpha:
                alpha = v

        best_pi = int(np.argmax(values))
        chosen = fix_robber_steal(candidates[best_pi], le)
        self.t_total += time.perf_counter() - t_start
        return chosen

    def stats_summary(self):
        d = max(self.n_decisions, 1)
        ch = self.cache_hits + self.cache_misses
        return (f"d={self.n_decisions} forced={self.n_forced} "
                f"win_short={self.n_terminal_win} "
                f"lowH={self.n_lowH} search={self.n_search} "
                f"leaves={self.n_leaves} pruned={self.n_pruned} "
                f"cache={100*self.cache_hits/max(ch,1):.0f}% "
                f"ms/dec={1000*self.t_total/d:.1f}")


def benchmark_v3(weights="csrc/nn_weights_m2.bin",
                 num_games=20, our_depth=4,
                 top_k_schedule=(8, 5, 4, 3),
                 time_budget_ms=3000,
                 seed_base=80000, mode="2v2", verbose=True):
    import ctypes as C
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    bot = SuperBotV3(weights,
                     our_depth=our_depth,
                     top_k_schedule=top_k_schedule,
                     entropy_fast_thresh=0.15,
                     time_budget_ms=time_budget_ms,
                     use_leaf_cache=True)

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
    nn_rank_sum = 0
    t0 = time.time()

    schedule_str = ",".join(str(k) for k in top_k_schedule)
    print(f"SuperBotV3 (our_depth={our_depth} k_schedule=[{schedule_str}] "
          f"budget={time_budget_ms}ms) vs AB2 2-ply, {num_games} games ({mode})\n")

    for gi in range(num_games):
        game = CatanGame(seed=seed_base + gi); game.reset()
        if mode == "1v3":
            nn_seats = {gi % 4}
            ab_seats = {s for s in range(4) if s != gi % 4}
        else:
            nn_seats = {gi % 4, (gi + 2) % 4}
            ab_seats = {(gi + 1) % 4, (gi + 3) % 4}

        while not game.is_terminal() and game.turn_number < 500:
            le = game.get_legal_actions()
            if not le: break
            if len(le) == 1: game.step(0); continue
            cp = game.current_player()
            if cp in nn_seats: game.step(bot.pick(game))
            else: game.step(ab2_choose(game, le))

        w = game.winner()
        vps = [game._game.state.player_state[s][0] for s in range(4)]

        # Per-seat stats
        for s in nn_seats:
            rank = sorted(range(4), key=lambda p: -vps[p]).index(s) + 1
            nn_rank_sum += rank
        nn_vp_sum += sum(vps[s] for s in nn_seats) / len(nn_seats)
        ab2_vp_sum += sum(vps[s] for s in ab_seats) / len(ab_seats)

        if w is not None and w in nn_seats: nn_wins += 1
        elif w is not None: ab2_wins += 1
        else: draws += 1

        if verbose:
            elapsed = time.time() - t0
            wr = nn_wins / max(nn_wins + ab2_wins, 1)
            print(f"  {gi+1}/{num_games}: W={w} VPs={vps} | "
                  f"NN_WR={wr:.0%} ({nn_wins}/{nn_wins+ab2_wins}) "
                  f"avg_rank={nn_rank_sum/((gi+1)*len(nn_seats)):.2f} "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    total = nn_wins + ab2_wins
    wr = nn_wins / max(total, 1)
    n_nn_seats = num_games * (2 if mode == "2v2" else 1)

    print(f"\n===== RESULTS (mode={mode}, depth={our_depth}, "
          f"k={top_k_schedule}) =====")
    print(f"  SuperBotV3:   {nn_wins} wins ({wr:.1%})")
    print(f"  AB2 2-ply:    {ab2_wins} wins ({1-wr:.1%})")
    print(f"  Draws:        {draws}")
    print(f"  Avg VP:       NN={nn_vp_sum/num_games:.2f} AB2={ab2_vp_sum/num_games:.2f}")
    print(f"  Avg NN rank:  {nn_rank_sum/n_nn_seats:.2f}/4 "
          f"(random={2.5:.2f}, best={1.00:.2f})")
    print(f"  Speed:        {num_games/elapsed:.2f} g/s ({elapsed:.0f}s total)")
    print(f"  Stats:        {bot.stats_summary()}")

    return wr, nn_wins, ab2_wins, elapsed


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--k-schedule", type=str, default="8,5,4,3")
    p.add_argument("--time-ms", type=int, default=3000)
    p.add_argument("--mode", choices=["2v2", "1v3"], default="1v3")
    p.add_argument("--seed-base", type=int, default=80000)
    args = p.parse_args()
    schedule = tuple(int(x) for x in args.k_schedule.split(","))
    benchmark_v3(num_games=args.games, our_depth=args.depth,
                 top_k_schedule=schedule,
                 time_budget_ms=args.time_ms,
                 mode=args.mode, seed_base=args.seed_base)
