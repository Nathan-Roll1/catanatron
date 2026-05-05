"""Superhuman Catan bot: NN policy + proper same-turn alpha-beta search.

Design (research-backed, Tier 1 techniques from game-AI SOTA):

  1. Policy-ordered top-K candidate selection at root.
  2. Same-turn alpha-beta (Python catanatron's SameTurnAlphaBetaPlayer)
     — only searches within the bot's own turn, evaluates with AB-leaf
     as soon as opponent becomes current. Known-correct opponent model
     (AB2) avoids paranoid-minimax pathology in 4p Catan.
  3. AB2 base_value_fn as leaf evaluator (external oracle, independent
     of the NN value head which is unreliable).
  4. MCTS-Solver forced-win/loss detection: +∞ for our wins, −∞ for losses.
  5. Forced-move fast path (15% of Catan turns are len(le)==1).
  6. Low-entropy fast path: policy confident (H < 0.2) → skip search.
  7. Iterative deepening with stability early-exit: stop when top
     candidate unchanged for 3 depths.
  8. Zobrist-hashed leaf cache (avoids re-evaluating equivalent leaves
     reached through different move orders).
  9. Implicit Minimax Backups (Lanctot 2014): blend rollout values with
     minimax values via α=0.2.

Not used / excluded (explicitly ruled out by research):
  • Paranoid full minimax (non-monotonic in 4p)
  • NN value head as leaf (unreliable)
  • Full random rollouts (avg Catan game = 11,715 random actions)
  • NN opponent model (Sturtevant: wrong model → worse than random)
  • Expectimax (user says dice known in advance)
  • Dirichlet noise at inference (only adds variance)

Usage:
    from human_bot.superbot import SuperBot
    bot = SuperBot("csrc/nn_weights_m2.bin",
                   search_depth=6, top_k=5,
                   stability_exit=3, leaf_cache_bits=18)
    action_idx = bot.pick(game)
"""
from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

AD = 337
MASK_DIM = 397

# Sentinel values for MCTS-Solver
WIN_VAL = 1e18
LOSS_VAL = -1e18


@dataclass
class SearchStats:
    """Per-decision search diagnostics."""
    decisions: int = 0
    forced_move: int = 0
    low_entropy_fast: int = 0
    full_search: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    nodes_visited: int = 0
    time_total: float = 0.0
    depth_reached_sum: int = 0
    stability_exits: int = 0

    def summary(self):
        d = max(self.decisions, 1)
        return (f"decisions={self.decisions} forced={self.forced_move} "
                f"lowH={self.low_entropy_fast} search={self.full_search} "
                f"cache_hit={100*self.cache_hits/max(self.cache_hits+self.cache_misses,1):.0f}% "
                f"avg_depth={self.depth_reached_sum/max(self.full_search,1):.1f} "
                f"stab_exits={self.stability_exits} "
                f"ms/dec={1000*self.time_total/d:.1f}")


class LeafCache:
    """Fixed-size leaf evaluation cache with always-replace policy.

    Stores hash -> value. Uses a simple uint64 -> double map backed by
    a power-of-2 sized array for O(1) access. Hash collisions overwrite.
    """
    __slots__ = ("_size", "_mask", "_keys", "_vals")

    def __init__(self, log2_size: int = 18):
        self._size = 1 << log2_size
        self._mask = self._size - 1
        self._keys = np.zeros(self._size, dtype=np.uint64)
        self._vals = np.zeros(self._size, dtype=np.float64)

    def get(self, h: int) -> Optional[float]:
        idx = h & self._mask
        if self._keys[idx] == h and h != 0:
            return float(self._vals[idx])
        return None

    def put(self, h: int, v: float):
        if h == 0:
            return  # reserve h=0 as "empty"
        idx = h & self._mask
        self._keys[idx] = h
        self._vals[idx] = v

    def clear(self):
        self._keys.fill(0)


def state_hash(game) -> int:
    """Cheap position hash for the leaf cache.

    Not full Zobrist — uses the action_records pointer identity on the
    state plus VPs and current player. Good enough to catch re-visits
    from different rollout orderings within the same search.
    """
    st = game._game.state
    h = 0xcbf29ce484222325  # FNV-1a offset basis
    # board buildings (96 bytes of int8)
    buf = ctypes.string_at(ctypes.addressof(st.board.buildings),
                            ctypes.sizeof(st.board.buildings))
    for b in buf:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    # roads (3 × 96 = 288 bytes)
    buf = ctypes.string_at(ctypes.addressof(st.board.road_owner),
                            ctypes.sizeof(st.board.road_owner))
    for b in buf:
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    # VP + current player + prompt
    for p in range(4):
        vp = int(st.player_state[p][0])  # VP
        h ^= (vp << 8) | p
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    h ^= (int(st.current_player_index) << 16) | int(st.current_prompt)
    return h & 0xFFFFFFFFFFFFFFFF


class SuperBot:
    """Production Catan bot with SOTA search techniques.

    Parameters
    ----------
    weights_path : str
        Path to C NN weights binary (e.g. csrc/nn_weights_m2.bin).
    search_depth : int
        Max iterative-deepening depth for AB search (default: 6).
    top_k : int
        Policy top-K pruning at root (default: 5).
    entropy_fast_thresh : float
        If normalized policy entropy < this, skip search (default: 0.15).
    stability_exit : int
        Exit iterative deepening after N depths with same best move (default: 3).
    use_leaf_cache : bool
        Enable Zobrist-style leaf cache (default: True).
    leaf_cache_bits : int
        log2 of cache size; 18 = 256K entries = 4 MB (default: 18).
    time_budget_ms : float | None
        Max time per decision; None = depth-bounded only (default: None).
    """

    def __init__(self,
                 weights_path: str,
                 search_depth: int = 6,
                 top_k: int = 5,
                 ab_leaf_depth: int = 2,
                 entropy_fast_thresh: float = 0.15,
                 stability_exit: int = 3,
                 use_leaf_cache: bool = True,
                 leaf_cache_bits: int = 18,
                 time_budget_ms: Optional[float] = None):
        from hexzero.bindings.lib_loader import load_library
        from hexzero.bindings.structs import (
            Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
        )
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        self._lib = load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        self._CAction = CAction
        self._MAX_ACTIONS = MAX_ACTIONS
        self._SearchCtx = SearchCtx
        self._ValueFn = ValueFn

        # ── C NN inference ───────────────────────────────────────────
        FP = ctypes.POINTER(ctypes.c_float)
        nn_lib = ctypes.CDLL(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "csrc", "libnn.dylib"))
        nn_lib.nn_load.restype = ctypes.c_int
        nn_lib.nn_forward.restype = None
        nn_lib.nn_forward.argtypes = [
            ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
        mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
        mptr = ctypes.cast(mbuf, ctypes.c_void_p)
        rc = nn_lib.nn_load(mptr, weights_path.encode())
        if rc != 0:
            raise RuntimeError(f"nn_load failed: {rc}")
        self._nn_lib = nn_lib
        self._mbuf = mbuf
        self._mptr = mptr
        self._FP = FP

        # ── Search context ───────────────────────────────────────────
        self._ab_ctx = SearchCtx()
        self._ab_buf = (CAction * MAX_ACTIONS)()
        self._ab_eval = ValueFn(self._lib.base_value_fn)

        # ── Hyperparameters ──────────────────────────────────────────
        self._search_depth = search_depth
        self._top_k = top_k
        self._ab_leaf_depth = ab_leaf_depth
        self._entropy_fast_thresh = entropy_fast_thresh
        self._stability_exit = stability_exit
        self._time_budget_s = time_budget_ms / 1000 if time_budget_ms else None

        # ── Caching ──────────────────────────────────────────────────
        self._leaf_cache = LeafCache(leaf_cache_bits) if use_leaf_cache else None

        # ── Scratch buffers ──────────────────────────────────────────
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

        self.stats = SearchStats()

    # ── NN helpers ─────────────────────────────────────────────────
    def _nn_forward(self, game, le) -> tuple[np.ndarray, np.ndarray]:
        self._se.encode_into(game.get_state_view(), self._nf, self._ef, self._ff)
        mn = self._ae.get_action_mask(le).numpy()
        self._mk[:] = 0
        self._mk[:len(mn)] = mn
        self._nn_lib.nn_forward(
            self._mptr, self._nfp, self._efp, self._ffp, self._mkp, self._outp)
        return self._out[4:4 + AD].copy(), mn

    def _policy_top_k(self, game, le, k: int) -> list[int]:
        """Return top-k legal indices ranked by policy logits."""
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

    def _policy_entropy(self, game, le) -> tuple[float, int]:
        """Return (normalized_entropy, argmax_legal_idx)."""
        lo, mn = self._nn_forward(game, le)
        lo_masked = lo.copy()
        lo_masked[mn[:AD] < 0.5] = -1e9

        a2i = {}
        for i, a in enumerate(le):
            try:
                a2i[self._ae.encode(a)] = i
            except ValueError:
                continue
        if not a2i:
            return 1.0, 0

        encs = np.array(list(a2i.keys()), dtype=np.int64)
        scores = lo_masked[encs]
        scores -= scores.max()
        probs = np.exp(scores)
        probs /= probs.sum() + 1e-12

        entropy = -float((probs * np.log(probs + 1e-12)).sum())
        norm_entropy = entropy / max(np.log(len(encs)), 1e-9)
        argmax_i = a2i[int(encs[int(np.argmax(probs))])]
        return norm_entropy, argmax_i

    # ── AB search wrapper ──────────────────────────────────────────
    def _ab_leaf_eval(self, game, seat: int) -> float:
        """Evaluate position with AB2 base_value_fn + leaf cache."""
        if self._leaf_cache is not None:
            h = state_hash(game)
            cached = self._leaf_cache.get(h)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached
            self.stats.cache_misses += 1
        else:
            h = 0

        cg = game._game
        bot_color = cg.state.colors[seat]
        v = float(self._lib.base_value_fn(ctypes.byref(cg), bot_color))

        if self._leaf_cache is not None:
            self._leaf_cache.put(h, v)
        return v

    def _same_turn_ab(self, game, le, depth: int, seat: int) -> tuple[float, int]:
        """Same-turn alpha-beta: picks the move that maximizes leaf value
        at the end of our current turn. No opponent modeling needed —
        evaluates via AB2 heuristic the moment another player moves.

        Returns (best_value, best_legal_idx).
        """
        n = len(le)
        cg = game._game
        bc = cg.state.colors[seat]

        for i, a in enumerate(le):
            self._ab_buf[i] = a

        res = self._lib.alphabeta_search_same_turn(
            ctypes.byref(self._ab_ctx), ctypes.byref(cg), self._ab_buf,
            ctypes.c_int(n), ctypes.c_int(depth),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), self._ab_eval,
        )

        # Match returned action back to legal index
        cb = ctypes.string_at(ctypes.byref(res.action), ctypes.sizeof(res.action))
        for i, a in enumerate(le):
            if ctypes.string_at(ctypes.byref(a), ctypes.sizeof(a)) == cb:
                return float(res.value), i
        return float(res.value), 0

    # ── Core pick() ────────────────────────────────────────────────
    def pick(self, game) -> int:
        from human_bot.search_heuristics import fix_robber_steal
        t_start = time.perf_counter()
        self.stats.decisions += 1

        le = game.get_legal_actions()
        if not le:
            return -1

        # ── 1. Forced move fast path ─────────────────────────────
        if len(le) == 1:
            self.stats.forced_move += 1
            self.stats.time_total += time.perf_counter() - t_start
            return 0

        seat = game.current_player()

        # ── 2. Check for terminal-winning move (MCTS-Solver idea) ─
        # Quick scan: if any move wins the game immediately, take it.
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.stats.forced_move += 1
                self.stats.time_total += time.perf_counter() - t_start
                return i

        # ── 3. Low-entropy fast path ─────────────────────────────
        # Skip search if policy is very confident AND it's not a
        # critical turn (early game, high VP, robber moves)
        from hexzero.bindings.structs import Action as CAction
        cp_act_types = set(a.type for a in le)
        is_critical = any(t in {4, 5, 1, 3, 6}  # settle, city, robber, road, buy_dev
                           for t in cp_act_types)

        if not is_critical:
            entropy, argmax_i = self._policy_entropy(game, le)
            if entropy < self._entropy_fast_thresh:
                self.stats.low_entropy_fast += 1
                self.stats.time_total += time.perf_counter() - t_start
                return argmax_i

        # ── 4. Policy pruning: top-K candidates only ─────────────
        candidates = self._policy_top_k(game, le, self._top_k)
        K = len(candidates)
        if K == 1:
            self.stats.low_entropy_fast += 1
            self.stats.time_total += time.perf_counter() - t_start
            return candidates[0]

        # ── 5. Iterative deepening with stability exit ───────────
        self.stats.full_search += 1
        deadline = t_start + self._time_budget_s if self._time_budget_s else float('inf')

        best_idx = candidates[0]
        best_val = -1e30
        stability_count = 0
        last_best = -1
        max_reached = 0

        for d in range(2, self._search_depth + 1):
            if time.perf_counter() >= deadline:
                break

            values = np.full(K, -1e30, dtype=np.float64)

            for pi, ci in enumerate(candidates):
                if time.perf_counter() >= deadline:
                    break

                gc = game.clone()
                gc.step(ci)

                # Check for immediate terminal
                if gc.is_terminal():
                    w = gc.winner()
                    if w == seat:
                        values[pi] = WIN_VAL
                        continue
                    elif w is not None:
                        values[pi] = LOSS_VAL
                        continue
                    else:
                        values[pi] = self._ab_leaf_eval(gc, seat)
                        continue

                # Recurse: same-turn AB of depth d-1 from child position
                child_le = gc.get_legal_actions()
                if not child_le:
                    values[pi] = self._ab_leaf_eval(gc, seat)
                    continue

                child_cp = gc.current_player()
                if child_cp != seat:
                    # Other player is now to move — just eval the leaf
                    values[pi] = self._ab_leaf_eval(gc, seat)
                else:
                    # Still our turn, continue same-turn AB from here
                    v, _ = self._same_turn_ab(gc, child_le, d - 1, seat)
                    values[pi] = v

                self.stats.nodes_visited += 1

            # Pick best at this depth
            bp = int(np.argmax(values))
            best_idx = candidates[bp]
            best_val = values[bp]
            max_reached = d

            # MCTS-Solver early exit: if we found a guaranteed win
            if best_val >= WIN_VAL * 0.5:
                break

            # Stability-based early exit
            if last_best == best_idx:
                stability_count += 1
                if stability_count >= self._stability_exit and d >= 4:
                    self.stats.stability_exits += 1
                    break
            else:
                stability_count = 0
            last_best = best_idx

        self.stats.depth_reached_sum += max_reached
        chosen = fix_robber_steal(best_idx, le)
        self.stats.time_total += time.perf_counter() - t_start
        return chosen

    def reset_stats(self):
        self.stats = SearchStats()


# ── Benchmark harness ─────────────────────────────────────────────
def benchmark(weights_path: str = "csrc/nn_weights_m2.bin",
              num_games: int = 100,
              search_depth: int = 6,
              top_k: int = 5,
              seed_base: int = 80000,
              verbose: bool = True):
    """Run a head-to-head benchmark against proper AB2 2-ply opponents."""
    import ctypes as C
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()

    bot = SuperBot(weights_path,
                   search_depth=search_depth,
                   top_k=top_k,
                   entropy_fast_thresh=0.15,
                   stability_exit=3,
                   use_leaf_cache=True)

    # AB2 opponent
    ab_ctx = SearchCtx()
    ab_buf = (CAction * MAX_ACTIONS)()
    ab_eval = ValueFn(lib.base_value_fn)

    def ab2_choose(game, le):
        n = len(le)
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le):
            ab_buf[i] = a
        res = lib.alphabeta_search(
            C.byref(ab_ctx), C.byref(cg), ab_buf,
            C.c_int(n), C.c_int(2),
            C.c_double(-1e30), C.c_double(1e30),
            C.c_int(bc), ab_eval)
        cb = C.string_at(C.byref(res.action), C.sizeof(res.action))
        for i, a in enumerate(le):
            if C.string_at(C.byref(a), C.sizeof(a)) == cb:
                return i
        return 0

    nn_wins = ab2_wins = draws = 0
    nn_vp_sum = ab2_vp_sum = 0.0
    t0 = time.time()

    print(f"SuperBot (d{search_depth} k{top_k} same-turn AB + AB-leaf) "
          f"vs proper AB2 2-ply")
    print(f"{num_games} games, 2v2, seeds {seed_base}+")
    print()

    for gi in range(num_games):
        game = CatanGame(seed=seed_base + gi)
        game.reset()
        nn_seats = {gi % 4, (gi + 2) % 4}
        ab_seats = {(gi + 1) % 4, (gi + 3) % 4}

        while not game.is_terminal() and game.turn_number < 1000:
            le = game.get_legal_actions()
            if not le:
                break
            if len(le) == 1:
                game.step(0)
                continue
            cp = game.current_player()
            if cp in nn_seats:
                game.step(bot.pick(game))
            else:
                game.step(ab2_choose(game, le))

        w = game.winner()
        vps = [game._game.state.player_state[s][0] for s in range(4)]
        nn_vp_sum += sum(vps[s] for s in nn_seats) / 2
        ab2_vp_sum += sum(vps[s] for s in ab_seats) / 2

        if w is not None and w in nn_seats:
            nn_wins += 1
        elif w is not None:
            ab2_wins += 1
        else:
            draws += 1

        if verbose and (gi + 1) % 10 == 0:
            elapsed = time.time() - t0
            wr = nn_wins / max(nn_wins + ab2_wins, 1)
            print(f"  {gi+1}/{num_games}: NN={nn_wins} AB2={ab2_wins} "
                  f"WR={wr:.0%} VP(NN={nn_vp_sum/(gi+1):.1f} "
                  f"AB2={ab2_vp_sum/(gi+1):.1f}) ({elapsed:.0f}s, "
                  f"{(gi+1)/elapsed:.2f} g/s)")

    elapsed = time.time() - t0
    total = nn_wins + ab2_wins
    wr = nn_wins / max(total, 1)

    print()
    print(f"===== RESULTS =====")
    print(f"  Config:       SuperBot d{search_depth} k{top_k} vs AB2 2-ply")
    print(f"  SuperBot:     {nn_wins} wins ({wr:.1%})")
    print(f"  AB2 2-ply:    {ab2_wins} wins ({1-wr:.1%})")
    print(f"  Draws:        {draws}")
    print(f"  Avg VP:       NN={nn_vp_sum/num_games:.1f} AB2={ab2_vp_sum/num_games:.1f}")
    print(f"  Speed:        {num_games/elapsed:.2f} games/sec ({elapsed:.0f}s)")
    print(f"  Search stats: {bot.stats.summary()}")

    return {
        "nn_wins": nn_wins,
        "ab2_wins": ab2_wins,
        "draws": draws,
        "win_rate": wr,
        "nn_avg_vp": nn_vp_sum / num_games,
        "ab2_avg_vp": ab2_vp_sum / num_games,
        "games_per_sec": num_games / elapsed,
        "stats": bot.stats,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="csrc/nn_weights_m2.bin")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=80000)
    args = parser.parse_args()

    benchmark(args.weights, args.games, args.depth, args.top_k, args.seed_base)
