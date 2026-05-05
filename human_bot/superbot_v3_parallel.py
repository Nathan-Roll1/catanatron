"""SuperBotV3 parallelized across M5 Max cores.

Each root candidate is evaluated in its own worker process. Workers load
the NN and catanatron library independently (needed because ctypes handles
are not picklable across processes).

Usage:
    from human_bot.superbot_v3_parallel import ParallelSuperBot
    bot = ParallelSuperBot("csrc/nn_weights_m2.bin",
                            num_workers=12, our_depth=5, ...)
    action = bot.pick(game)

Architecture:
    - Main process: get top-K candidates from policy, scatter across workers
    - Worker k: apply candidate_k, run deep_search, return value
    - Main: gather values, argmax, return

Because each root candidate explores an independent branch of the game
tree, there's no shared state — perfect for parallelism. Cache is per-worker
(separate address space) but that's OK; each candidate explores different
positions anyway.
"""
from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import pickle
import time
from typing import Optional

import numpy as np

AD = 337
MASK_DIM = 397


def _default_worker_count(root_k: int) -> int:
    """Use only workers that can receive root jobs for a single decision."""
    cpus = os.cpu_count() or 1
    return max(1, min(cpus, max(1, root_k)))


def _worker_process(job_q, result_q, weights_path, our_depth, k_schedule,
                    entropy_thresh, leaf_cache_bits, time_budget_ms,
                    leaf_mode=0, incremental_replay=True):
    """Worker: replay action list from scratch to reconstruct state, then search."""
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from human_bot.superbot_v3 import SuperBotV3
    from hexzero.game.interface import CatanGame
    from hexzero.config import GameConfig

    bot = SuperBotV3(weights_path,
                     our_depth=our_depth,
                     top_k_schedule=k_schedule,
                     entropy_fast_thresh=entropy_thresh,
                     use_leaf_cache=True,
                     leaf_cache_bits=leaf_cache_bits,
                     time_budget_ms=time_budget_ms)

    # Cache replayed games by (seed, num_players, action_sequence_len) so we
    # don't redo the whole replay for every candidate at the same decision.
    _last_state = {"key": None, "game": None}

    while True:
        job = job_q.get()
        if job is None:
            break
        decision_id, seat, candidate_idx, seed, num_players, action_seq, deadline_wall = job

        t_job_start = time.perf_counter()
        t_now = time.time()
        time_left_ms = max(1, (deadline_wall - t_now) * 1000)
        bot._deadline = time.perf_counter() + time_left_ms / 1000

        key = (seed, num_players, len(action_seq))
        replayed = False
        if _last_state["key"] == key and _last_state["game"] is not None:
            game = _last_state["game"].clone()
        elif (incremental_replay and _last_state["key"] is not None
              and _last_state["game"] is not None
              and _last_state["key"][0] == seed
              and _last_state["key"][1] == num_players
              and _last_state["key"][2] <= len(action_seq)):
            replayed = True
            old_len = _last_state["key"][2]
            game = _last_state["game"].clone()
            for ai in action_seq[old_len:]:
                game.step(ai)
            _last_state["key"] = key
            _last_state["game"] = game.clone()
        else:
            replayed = True
            cfg = GameConfig(num_players=num_players)
            game = CatanGame(seed=seed, config=cfg)
            game.reset()
            for ai in action_seq:
                game.step(ai)
            _last_state["key"] = key
            _last_state["game"] = game.clone()
        t_replay_done = time.perf_counter()

        game.step(candidate_idx)
        v = bot._deep_search(game, seat, our_depth - 1,
                              alpha=-2.0, beta=2.0, depth_idx=1)
        t_done = time.perf_counter()
        result_q.put((decision_id, candidate_idx, v,
                      t_replay_done - t_job_start,
                      t_done - t_replay_done,
                      t_done - t_job_start,
                      len(action_seq), replayed))


def _worker_process_c2(job_q, result_q, weights_path, our_depth, k_schedule,
                       entropy_thresh, leaf_cache_bits, time_budget_ms,
                       leaf_mode=0, incremental_replay=True,
                       algo_policy: bool = False,
                       opponent_model: str = "ab2",
                       algo_flags: int | None = None,
                       algo_value_tiebreak: bool | None = None):
    """Worker: replay state, then evaluate one root candidate in pure C."""
    import sys
    import ctypes
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from human_bot.superbot_v3_c2 import SuperBotV3C2
    from hexzero.game.interface import CatanGame
    from hexzero.config import GameConfig

    bot = SuperBotV3C2(weights_path,
                       our_depth=our_depth,
                       top_k_schedule=k_schedule,
                       entropy_fast_thresh=entropy_thresh,
                       leaf_cache_bits=leaf_cache_bits,
                       time_budget_ms=time_budget_ms,
                       leaf_mode=leaf_mode,
                       algo_policy=algo_policy,
                       opponent_model=opponent_model,
                       algo_flags=algo_flags,
                       algo_value_tiebreak=algo_value_tiebreak)
    schedule_arr = (ctypes.c_int * len(k_schedule))(*k_schedule)
    _last_state = {"key": None, "game": None}

    while True:
        job = job_q.get()
        if job is None:
            break
        decision_id, seat, candidate_idx, seed, num_players, action_seq, deadline_wall = job

        t_job_start = time.perf_counter()
        time_left_s = max(0.001, deadline_wall - time.time())
        bot._libdeep.deep_search_configure(
            bot._ds_ctx, our_depth, schedule_arr, len(k_schedule),
            2, time_left_s)
        bot._configure_algo_policy()

        key = (seed, num_players, len(action_seq))
        replayed = False
        if _last_state["key"] == key and _last_state["game"] is not None:
            game = _last_state["game"].clone()
        elif (incremental_replay and _last_state["key"] is not None
              and _last_state["game"] is not None
              and _last_state["key"][0] == seed
              and _last_state["key"][1] == num_players
              and _last_state["key"][2] <= len(action_seq)):
            replayed = True
            old_len = _last_state["key"][2]
            game = _last_state["game"].clone()
            for ai in action_seq[old_len:]:
                game.step(ai)
            _last_state["key"] = key
            _last_state["game"] = game.clone()
        else:
            replayed = True
            cfg = GameConfig(num_players=num_players)
            game = CatanGame(seed=seed, config=cfg)
            game.reset()
            for ai in action_seq:
                game.step(ai)
            _last_state["key"] = key
            _last_state["game"] = game.clone()
        t_replay_done = time.perf_counter()

        our_color = int(game._game.state.colors[seat])
        cand_arr = (ctypes.c_int * 1)(candidate_idx)
        best_idx_out = ctypes.c_int(-1)
        v = bot._libdeep.deep_search_root(
            bot._ds_ctx, ctypes.addressof(game._game), our_color,
            cand_arr, 1, ctypes.byref(best_idx_out))
        t_done = time.perf_counter()
        out = (decision_id, candidate_idx, v,
               t_replay_done - t_job_start,
               t_done - t_replay_done,
               t_done - t_job_start,
               len(action_seq), replayed)
        if os.environ.get("CATAN_MEASURE_C_STATS", "0") == "1":
            from human_bot.superbot_v3_c import DeepSearchStats
            s = DeepSearchStats()
            bot._libdeep.deep_search_get_stats(bot._ds_ctx, ctypes.byref(s))
            out = out + (
                s.n_pcache_hits, s.n_pcache_misses,
                s.n_calls, s.n_leaves,
                s.n_cache_hits, s.n_cache_misses,
                s.n_root_early_exits,
            )
        result_q.put(out)


class ParallelSuperBot:
    def __init__(self,
                 weights_path: str,
                 num_workers: int = 8,
                 our_depth: int = 5,
                 top_k_schedule: tuple[int, ...] = (10, 7, 5, 4, 3),
                 entropy_fast_thresh: float = 0.15,
                 leaf_cache_bits: int = 20,
                 time_budget_ms: float = 5000,
                 profile: bool = False,
                 backend: str = "python",
                 measure_c_stats: bool | None = None,
                 leaf_mode: int = 0,
                 incremental_replay: bool = True,
                 root_c_policy: bool = False,
                 algo_policy: bool | None = None,
                 opponent_model: str = "ab2",
                 algo_flags: int | None = None,
                 algo_value_tiebreak: bool | None = None):
        from hexzero.bindings.lib_loader import load_library
        from hexzero.bindings.structs import (
            Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
        )
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        self._algo_policy = (
            os.environ.get("CATAN_POLICY_ALGO", "") not in ("", "0")
            if algo_policy is None else bool(algo_policy)
        )

        self._serial_bot = None
        if not (self._algo_policy and backend == "c2" and root_c_policy):
            # Main-process NN for legacy root top-K / entropy shortcuts.
            from human_bot.superbot_v3 import SuperBotV3
            self._serial_bot = SuperBotV3(weights_path,
                                          our_depth=our_depth,
                                          top_k_schedule=top_k_schedule,
                                          entropy_fast_thresh=entropy_fast_thresh,
                                          use_leaf_cache=True,
                                          leaf_cache_bits=leaf_cache_bits,
                                          time_budget_ms=time_budget_ms)
        self._root_c2 = None
        if backend == "c2" and root_c_policy:
            from human_bot.superbot_v3_c2 import SuperBotV3C2
            self._root_c2 = SuperBotV3C2(weights_path,
                                         our_depth=our_depth,
                                         top_k_schedule=top_k_schedule,
                                         entropy_fast_thresh=entropy_fast_thresh,
                                         leaf_cache_bits=leaf_cache_bits,
                                         time_budget_ms=time_budget_ms,
                                         leaf_mode=leaf_mode,
                                         algo_policy=self._algo_policy,
                                         opponent_model=opponent_model,
                                         algo_flags=algo_flags,
                                         algo_value_tiebreak=algo_value_tiebreak)

        self._our_depth = our_depth
        self._k_schedule = top_k_schedule
        self._time_budget_ms = time_budget_ms
        self._entropy_thresh = entropy_fast_thresh
        self._leaf_cache_bits = leaf_cache_bits
        self._weights_path = weights_path
        self._num_workers = num_workers
        self._profile = profile
        self._backend = backend
        self._leaf_mode = leaf_mode
        self._incremental_replay = incremental_replay
        self._root_c_policy = root_c_policy
        self._opponent_model = opponent_model
        self._algo_flags = algo_flags
        self._algo_value_tiebreak = algo_value_tiebreak
        if measure_c_stats is None:
            self._measure_c_stats = os.environ.get(
                "CATAN_MEASURE_C_STATS", "0") == "1"
        else:
            self._measure_c_stats = measure_c_stats
        self.c_stats = {
            "pcache_hits": 0,
            "pcache_misses": 0,
            "n_calls": 0,
            "n_leaves": 0,
            "leaf_hits": 0,
            "leaf_misses": 0,
            "root_early_exits": 0,
        }

        # Spawn workers
        ctx = mp.get_context("spawn")
        self._job_q = ctx.Queue()
        self._result_q = ctx.Queue()
        self._workers = []
        if self._measure_c_stats:
            os.environ["CATAN_MEASURE_C_STATS"] = "1"
        else:
            os.environ.pop("CATAN_MEASURE_C_STATS", None)
        worker_target = _worker_process_c2 if backend == "c2" else _worker_process
        worker_args = (self._job_q, self._result_q, weights_path,
                       our_depth, top_k_schedule, entropy_fast_thresh,
                       leaf_cache_bits, time_budget_ms)
        if backend == "c2":
            worker_args = worker_args + (
                leaf_mode, incremental_replay, self._algo_policy, opponent_model,
                self._algo_flags, self._algo_value_tiebreak)
        else:
            worker_args = worker_args + (leaf_mode, incremental_replay)
        for _ in range(num_workers):
            p = ctx.Process(
                target=worker_target,
                args=worker_args,
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        self._decision_counter = 0

        # Action sequence tracking: for each external game pointer we've seen,
        # record the (seed, num_players, action_history). Workers replay this
        # to reconstruct identical state.
        self._action_history = []
        self._last_seed = None
        self._last_num_players = None

        self.n_decisions = 0
        self.n_forced = 0
        self.n_lowH = 0
        self.n_search = 0
        self.t_total = 0.0
        self._profile_rows = []

    def reset_game(self, seed: int, num_players: int = 4):
        """Tell bot that a fresh game started. Caller must call this before
        the first pick() of each game."""
        self._last_seed = seed
        self._last_num_players = num_players
        self._action_history = []

    def record_action(self, action_idx: int):
        """Caller must call this after each step() (including non-NN seats)
        so we can faithfully replay state in workers."""
        self._action_history.append(action_idx)

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

        # Terminal win check
        for i in range(len(le)):
            gc = game.clone()
            gc.step(i)
            if gc.is_terminal() and gc.winner() == seat:
                self.t_total += time.perf_counter() - t_start
                return i

        # Low-entropy fast path
        is_critical = any(a.type in {1, 3, 4, 5, 6} for a in le)
        if not self._algo_policy and not is_critical:
            if self._root_c2 is not None:
                lo, mn = self._root_c2._nn_forward_raw(game, le)
            else:
                lo, mn = self._serial_bot._nn_forward(game, le)
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

        # Parallel deep search over top-K root candidates
        self.n_search += 1
        t_topk_start = time.perf_counter()
        k_root = self._k_schedule[0]
        if self._root_c2 is not None:
            candidates = self._root_c2._c_policy_top_k(game, le, k_root)
        else:
            candidates = self._serial_bot._nn_topk(game, le, k_root)
        t_topk_done = time.perf_counter()
        K = len(candidates)

        decision_id = self._decision_counter
        self._decision_counter += 1

        deadline_wall = time.time() + self._time_budget_ms / 1000
        if self._last_seed is None:
            raise RuntimeError("reset_game() must be called before first pick()")
        action_seq = tuple(self._action_history)

        values = np.full(K, -2.0, dtype=np.float64)
        cand_to_pos = {candidates[i]: i for i in range(K)}
        replay_time = 0.0
        worker_search_time = 0.0
        worker_total_time = 0.0
        replay_jobs = 0
        max_worker_total = 0.0

        def _consume_row(row):
            nonlocal replay_time, worker_search_time, worker_total_time, replay_jobs, max_worker_total
            if len(row) >= 8:
                replay_time += row[3]
                worker_search_time += row[4]
                worker_total_time += row[5]
                max_worker_total = max(max_worker_total, row[5])
                replay_jobs += int(bool(row[7]))
            if self._measure_c_stats and len(row) >= 15:
                self.c_stats["pcache_hits"] += int(row[8])
                self.c_stats["pcache_misses"] += int(row[9])
                self.c_stats["n_calls"] += int(row[10])
                self.c_stats["n_leaves"] += int(row[11])
                self.c_stats["leaf_hits"] += int(row[12])
                self.c_stats["leaf_misses"] += int(row[13])
                self.c_stats["root_early_exits"] += int(row[14])

        t_scatter_start = time.perf_counter()
        for ci in candidates:
            self._job_q.put((decision_id, seat, ci, self._last_seed,
                             self._last_num_players, action_seq, deadline_wall))
        t_scatter_done = time.perf_counter()
        received = 0
        while received < K:
            row = self._result_q.get()
            if row[0] != decision_id:
                continue
            _consume_row(row)
            values[cand_to_pos[row[1]]] = row[2]
            received += 1
        t_gather_done = time.perf_counter()

        best_pi = int(np.argmax(values))
        chosen = fix_robber_steal(candidates[best_pi], le)
        elapsed = time.perf_counter() - t_start
        self.t_total += elapsed
        if self._profile:
            self._profile_rows.append({
                "decision": decision_id,
                "legal": len(le),
                "candidates": K,
                "action_history": len(action_seq),
                "topk_s": t_topk_done - t_topk_start,
                "scatter_s": t_scatter_done - t_scatter_start,
                "wall_s": elapsed,
                "gather_s": t_gather_done - t_scatter_done,
                "worker_replay_s_sum": replay_time,
                "worker_search_s_sum": worker_search_time,
                "worker_total_s_sum": worker_total_time,
                "worker_total_s_max": max_worker_total,
                "replay_jobs": replay_jobs,
            })
        return chosen

    def shutdown(self):
        for _ in self._workers:
            self._job_q.put(None)
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

    def stats_summary(self):
        d = max(self.n_decisions, 1)
        summary = (f"d={self.n_decisions} forced={self.n_forced} "
                   f"lowH={self.n_lowH} search={self.n_search} "
                   f"ms/dec={1000*self.t_total/d:.1f} "
                   f"workers={self._num_workers} backend={self._backend} "
                   f"incr_replay={int(self._incremental_replay)} "
                   f"root_c={int(self._root_c_policy)}")
        if self._profile_rows:
            rows = self._profile_rows
            n = len(rows)
            wall = sum(r["wall_s"] for r in rows)
            topk = sum(r["topk_s"] for r in rows)
            replay = sum(r["worker_replay_s_sum"] for r in rows)
            wsearch = sum(r["worker_search_s_sum"] for r in rows)
            wtotal = sum(r["worker_total_s_sum"] for r in rows)
            max_worker = max(r["worker_total_s_max"] for r in rows)
            replay_jobs = sum(r["replay_jobs"] for r in rows)
            avg_k = sum(r["candidates"] for r in rows) / n
            summary += (f" | profile search_dec={n} avgK={avg_k:.1f} "
                        f"topk={topk:.2f}s wall={wall:.2f}s "
                        f"worker_sum={wtotal:.2f}s search_sum={wsearch:.2f}s "
                        f"replay_sum={replay:.2f}s replay_jobs={replay_jobs} "
                        f"max_worker={max_worker:.2f}s")
        if self._measure_c_stats:
            c = self.c_stats
            ph = c["pcache_hits"] + c["pcache_misses"]
            lh = c["leaf_hits"] + c["leaf_misses"]
            summary += (
                f" | C_policy pcache_hit%={100.0 * c['pcache_hits'] / max(ph, 1):.1f} "
                f"nn_forw={c['pcache_misses']} "
                f"leaf_hit%={100.0 * c['leaf_hits'] / max(lh, 1):.1f} "
                f"root_early={c['root_early_exits']}"
            )
        return summary


def benchmark(weights="csrc/nn_weights_m2.bin",
              num_games=20, num_workers=8, our_depth=5,
              k_schedule=(10, 7, 5, 4, 3), time_budget_ms=5000,
              seed_base=80000, mode="1v3", profile=False,
              backend="python", measure_c_stats: bool | None = None,
              leaf_mode: int = 0,
              incremental_replay: bool = True,
              root_c_policy: bool = True):
    import ctypes as C
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import (
        Action as CAction, MAX_ACTIONS, SearchCtx, ValueFn,
    )
    from hexzero.bindings.lib_loader import load_library

    lib = load_library()
    bot = ParallelSuperBot(weights,
                            num_workers=num_workers,
                            our_depth=our_depth,
                            top_k_schedule=k_schedule,
                            time_budget_ms=time_budget_ms,
                            profile=profile,
                            backend=backend,
                            measure_c_stats=measure_c_stats,
                            leaf_mode=leaf_mode,
                            incremental_replay=incremental_replay,
                            root_c_policy=root_c_policy)

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
    ab2_time = 0.0
    ab2_calls = 0
    t0 = time.time()

    print(f"ParallelSuperBotV3 (workers={num_workers}, backend={backend}, depth={our_depth}, "
          f"k={k_schedule}, budget={time_budget_ms}ms, leaf_mode={leaf_mode}) "
          f"incr_replay={int(incremental_replay)} root_c={int(root_c_policy)} "
          f"vs AB2 {num_games} games ({mode})\n", flush=True)

    try:
        for gi in range(num_games):
            seed = seed_base + gi
            game = CatanGame(seed=seed); game.reset()
            bot.reset_game(seed, 4)
            if mode == "1v3":
                nn_seats = {gi % 4}
                ab_seats = {s for s in range(4) if s != gi % 4}
            else:
                nn_seats = {gi % 4, (gi + 2) % 4}
                ab_seats = {(gi + 1) % 4, (gi + 3) % 4}

            while not game.is_terminal() and game.turn_number < 500:
                le = game.get_legal_actions()
                if not le: break
                if len(le) == 1:
                    game.step(0)
                    bot.record_action(0)
                    continue
                cp = game.current_player()
                if cp in nn_seats:
                    chosen = bot.pick(game)
                else:
                    t_ab = time.perf_counter()
                    chosen = ab2_choose(game, le)
                    ab2_time += time.perf_counter() - t_ab
                    ab2_calls += 1
                game.step(chosen)
                bot.record_action(chosen)

            w = game.winner()
            vps = [game._game.state.player_state[s][0] for s in range(4)]
            for s in nn_seats:
                rank = sorted(range(4), key=lambda p: -vps[p]).index(s) + 1
                nn_rank_sum += rank
            nn_vp_sum += sum(vps[s] for s in nn_seats) / len(nn_seats)
            ab2_vp_sum += sum(vps[s] for s in ab_seats) / len(ab_seats)
            if w is not None and w in nn_seats: nn_wins += 1
            elif w is not None: ab2_wins += 1
            else: draws += 1

            elapsed = time.time() - t0
            wr = nn_wins / max(nn_wins + ab2_wins, 1)
            print(f"  {gi+1}/{num_games}: W={w} VPs={vps} | "
                  f"WR={wr:.0%} avg_rank={nn_rank_sum/((gi+1)*len(nn_seats)):.2f} "
                  f"({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        total = nn_wins + ab2_wins
        wr = nn_wins / max(total, 1)
        n_nn = num_games * (2 if mode == "2v2" else 1)
        print(f"\n===== RESULTS =====", flush=True)
        print(f"  Wins:    {nn_wins} ({wr:.1%})", flush=True)
        print(f"  Losses:  {ab2_wins}  Draws: {draws}", flush=True)
        print(f"  Avg VP:  NN={nn_vp_sum/num_games:.2f} AB2={ab2_vp_sum/num_games:.2f}", flush=True)
        print(f"  Avg rank: {nn_rank_sum/n_nn:.2f}/4", flush=True)
        print(f"  Speed:   {num_games/elapsed:.3f} g/s ({elapsed:.0f}s)", flush=True)
        if profile:
            print(f"  Opp AB2:  {ab2_calls} calls, {ab2_time:.2f}s total "
                  f"({1000*ab2_time/max(ab2_calls, 1):.1f}ms/call)", flush=True)
        print(f"  Stats:   {bot.stats_summary()}", flush=True)
    finally:
        bot.shutdown()

    return wr


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="csrc/nn_weights_m2.bin")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--workers", type=int, default=8,
                   help="Worker processes for one decision. Use 0 to auto-size to min(CPUs, root K).")
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--k-schedule", type=str, default="10,7,5,4,3")
    p.add_argument("--time-ms", type=int, default=5000)
    p.add_argument("--mode", choices=["1v3", "2v2"], default="1v3")
    p.add_argument("--seed-base", type=int, default=80000)
    p.add_argument("--single-game", action="store_true",
                   help="Require exactly one game; intended for local multi-core deployment.")
    p.add_argument("--profile", action="store_true",
                   help="Print coarse timing totals for one-game bottleneck analysis.")
    p.add_argument("--backend", choices=["python", "c2"], default="python",
                   help="Worker search backend. c2 uses pure-C deep_search per root candidate.")
    p.add_argument("--measure-c-stats", action="store_true",
                   help="Aggregate libdeep DeepSearchStats from workers (adds overhead).")
    p.add_argument("--leaf-mode", type=int, default=0,
                   help="C deep_search leaf mode: 0=original, 1=top-2 enemy VP penalty.")
    p.add_argument("--no-incremental-replay", action="store_true",
                   help="Disable worker incremental replay; use old replay-from-start behavior.")
    p.add_argument("--root-c-policy", action="store_true",
                   help="Use pure-C root top-k instead of old Python root top-k path.")
    p.add_argument("--algo-policy", action="store_true",
                   help="Use the pure algorithmic C policy_top_k path instead of NN logits.")
    args = p.parse_args()
    if args.algo_policy:
        os.environ["CATAN_POLICY_ALGO"] = "1"
        if args.k_schedule == "10,7,5,4,3":
            args.k_schedule = "6,4,2,2,2"
    schedule = tuple(int(x) for x in args.k_schedule.split(","))
    if args.single_game and args.games != 1:
        p.error("--single-game requires --games 1")
    workers = args.workers if args.workers > 0 else _default_worker_count(schedule[0])
    mc = True if args.measure_c_stats else None
    benchmark(weights=args.weights, num_games=args.games, num_workers=workers,
              our_depth=args.depth, k_schedule=schedule,
              time_budget_ms=args.time_ms, mode=args.mode,
              seed_base=args.seed_base, profile=args.profile,
              backend=args.backend, measure_c_stats=mc,
              leaf_mode=args.leaf_mode,
              incremental_replay=not args.no_incremental_replay,
              root_c_policy=args.root_c_policy)
