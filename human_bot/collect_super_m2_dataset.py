"""Collect a finetuning dataset where super_m2 (teacher) plays against
3 copies of the 0-ply NN (student / opponent), recording the teacher's
moves as supervised targets.

Output is one .pt shard compatible with `human_bot.dataset.HumanGameDataset`
and the existing M2 trainer. Each step records:

    node_features, edge_features, flat_features, action_mask,
    action_idx (super_m2's chosen action),
    player (acting seat = super_m2 seat),
    reward_vec (1.0 at winner seat, 0 otherwise),
    step_weight (1.0 default; teacher labels are uniformly trusted),
    num_players (=4)

Usage:
    python -m human_bot.collect_super_m2_dataset \
        --games 20 --workers 8 --out csrc/data_super_m2 \
        --weights csrc/nn_weights_m2.bin --seed-base 100000
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch


MASK_DIM = 397
AD = 337
CAND_MAX = 32


def _empty_root_table():
    return {
        "candidate_legal_idx": np.full(CAND_MAX, -1, dtype=np.int16),
        "candidate_action_idx": np.full(CAND_MAX, -1, dtype=np.int16),
        "candidate_value": np.full(CAND_MAX, np.nan, dtype=np.float32),
        "candidate_count": 0,
    }


def _root_table(res, le_actions, ae):
    """Padded root search table for dense offline analysis."""
    table = _empty_root_table()
    candidates = res.get("candidates") or []
    values = res.get("values") or []
    n = min(len(candidates), CAND_MAX)
    table["candidate_count"] = n
    for j in range(n):
        le_idx = int(candidates[j])
        table["candidate_legal_idx"][j] = le_idx
        try:
            table["candidate_action_idx"][j] = ae.encode(le_actions[le_idx])
        except (ValueError, IndexError):
            table["candidate_action_idx"][j] = -1
        if j < len(values):
            table["candidate_value"][j] = float(values[j])
    return table


def _build_policy_target(le_indices, le_actions, ae, mask_full,
                          kind, search_candidates, search_values,
                          lowh_logits, chosen_idx, temperature=0.1):
    """Build a soft policy target over the 397-dim action space.

    Args:
        le_indices: list of action_idx for each legal action (or None for
            unencodable actions)
        le_actions: list of legal actions (le)
        ae: ActionEncoder
        mask_full: (397,) legal mask
        kind: 'search'|'lowH'|'terminal'
        search_candidates: list of legal-indices (only used if kind=='search')
        search_values: list of values per candidate (only if kind=='search')
        lowh_logits: (337,) NN policy logits (only if kind=='lowH')
        chosen_idx: legal-index of the chosen action (always)
        temperature: softmax temperature for search values (default 0.1)

    Returns:
        (policy_target, action_idx) where:
            policy_target: (397,) float32 distribution
            action_idx: int — action_idx of chosen action (for fallback)
    """
    target = np.zeros(MASK_DIM, dtype=np.float32)
    chosen_action_idx = -1

    if kind == "search" and search_values is not None:
        # Build distribution over the K search candidates
        # filter out timed-out (-2.0) values
        valid = [(le_idx, v) for le_idx, v in zip(search_candidates, search_values)
                 if v > -1.5]
        if not valid:
            valid = list(zip(search_candidates, search_values))

        # Map each candidate's le-index -> action_idx
        cand_act_idx = []
        cand_values = []
        for le_idx, v in valid:
            try:
                aidx = ae.encode(le_actions[le_idx])
                if 0 <= aidx < AD:
                    cand_act_idx.append(aidx)
                    cand_values.append(v)
            except (ValueError, IndexError):
                continue

        if cand_act_idx:
            # softmax(values / temperature) over candidates
            vals = np.asarray(cand_values, dtype=np.float64)
            vals = (vals - vals.max()) / max(temperature, 1e-9)
            probs = np.exp(vals)
            probs /= probs.sum()
            for aidx, p in zip(cand_act_idx, probs):
                target[aidx] = float(p)

        # Set chosen_action_idx for fallback
        try:
            chosen_action_idx = ae.encode(le_actions[chosen_idx])
        except (ValueError, IndexError):
            chosen_action_idx = -1

    elif kind == "lowH" and lowh_logits is not None:
        # Use NN policy softmax (over legal mask) as soft target
        logits = lowh_logits.copy()
        # Mask using the 337-dim portion of mask_full
        mask337 = mask_full[:AD] > 0.5
        logits[~mask337] = -1e9
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum() + 1e-12
        target[:AD] = probs
        try:
            chosen_action_idx = ae.encode(le_actions[chosen_idx])
        except (ValueError, IndexError):
            chosen_action_idx = -1

    elif kind == "terminal":
        # One-hot on the winning move
        try:
            chosen_action_idx = ae.encode(le_actions[chosen_idx])
            if 0 <= chosen_action_idx < AD:
                target[chosen_action_idx] = 1.0
        except (ValueError, IndexError):
            chosen_action_idx = -1

    # Sanity: make sure target is normalized + at least chosen action is set
    s = target.sum()
    if s > 1e-9:
        target /= s
    elif chosen_action_idx >= 0:
        target[chosen_action_idx] = 1.0

    return target, int(chosen_action_idx)


def _play_one_game(args):
    """Worker: play one game.

    Modes (selected by super_seat):
      super_seat in {0,1,2,3}: that seat is super_m2, other 3 are NN-0ply.
                               Record only super_m2 decisions.
      super_seat == -1:        ALL 4 seats are super_m2 (4× bot instances
                               per worker, each with its own leaf cache).
                               Record EVERY decision.

    Args tuple may include an optional `dense=True` flag as the 8th element.
    When dense, each step also records `policy_target` (soft distribution
    from search values) and `signal_kind` (for diagnostics).

    Returns a dict with steps + game metadata.
    """
    if len(args) == 8:
        (game_idx, seed, super_seat, weights_path, our_depth,
         k_schedule, time_budget_ms, dense) = args
    else:
        (game_idx, seed, super_seat, weights_path, our_depth,
         k_schedule, time_budget_ms) = args
        dense = False

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from human_bot.superbot_v3_c2 import SuperBotV3C2

    all_seats = (super_seat == -1)

    t_start = time.time()
    if all_seats:
        # One bot per seat — separate leaf caches, shared libdeep & weights.
        bots = [
            SuperBotV3C2(
                weights_path,
                our_depth=our_depth,
                top_k_schedule=k_schedule,
                entropy_fast_thresh=0.15,
                time_budget_ms=time_budget_ms,
                leaf_cache_bits=20,
            )
            for _ in range(4)
        ]
        bot = bots[0]  # shared encoder buffers below
    else:
        bot = SuperBotV3C2(
            weights_path,
            our_depth=our_depth,
            top_k_schedule=k_schedule,
            entropy_fast_thresh=0.15,
            time_budget_ms=time_budget_ms,
            leaf_cache_bits=20,
        )
        bots = None

    ae = ActionEncoder()
    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    nf_buf = np.zeros((N, NF), dtype=np.float32)
    ef_buf = np.zeros((E, EF), dtype=np.float32)
    ff_buf = np.zeros(FFD, dtype=np.float32)

    FP = ctypes.POINTER(ctypes.c_float)
    nfp = nf_buf.ctypes.data_as(FP)
    efp = ef_buf.ctypes.data_as(FP)
    ffp = ff_buf.ctypes.data_as(FP)
    mk_buf = np.zeros(MASK_DIM, dtype=np.float32)
    mkp = mk_buf.ctypes.data_as(FP)
    out_buf = np.zeros(4 + MASK_DIM, dtype=np.float32)

    def policy_argmax(g, le):
        """0-ply NN argmax using bot's already-loaded libdeep + weights."""
        se.encode_into(g.get_state_view(), nf_buf, ef_buf, ff_buf)
        mn = ae.get_action_mask(le).numpy()
        mk_buf[:] = 0
        mk_buf[:len(mn)] = mn
        bot._libdeep.nn_forward(
            bot._mptr, nfp, efp, ffp, mkp,
            out_buf.ctypes.data_as(ctypes.c_void_p))
        logits = out_buf[4:4 + AD].copy()
        logits[mn[:AD] < 0.5] = -1e9
        a_idx = int(np.argmax(logits))
        for i, a in enumerate(le):
            try:
                if ae.encode(a) == a_idx:
                    return i
            except ValueError:
                continue
        return 0

    steps: list[dict] = []
    seat_decisions = [0, 0, 0, 0]
    kind_counts = {"search": 0, "lowH": 0, "terminal": 0, "forced": 0}
    decision_index = 0
    while not game.is_terminal() and game.turn_number < 500:
        le = game.get_legal_actions()
        if not le:
            break
        n_le = len(le)
        if len(le) == 1:
            game.step(0)
            continue

        cp = game.current_player()
        record_this = all_seats or (cp == super_seat)

        if record_this:
            sv = game.get_state_view()
            se.encode_into(sv, nf_buf, ef_buf, ff_buf)
            mn_full = ae.get_action_mask(le).numpy()
            mask = np.zeros(MASK_DIM, dtype=np.float32)
            mask[:len(mn_full)] = mn_full

            active_bot = bots[cp] if all_seats else bot

            if dense:
                res = active_bot.pick_full(game)
                chosen_local = res["chosen"]
                kind = res["kind"]
                if kind == "forced":
                    # Should not happen here since len(le) > 1, but guard
                    game.step(chosen_local)
                    continue

                kind_counts[kind] = kind_counts.get(kind, 0) + 1

                policy_target, act_idx = _build_policy_target(
                    le_indices=None, le_actions=le, ae=ae,
                    mask_full=mask,
                    kind=kind,
                    search_candidates=res.get("candidates"),
                    search_values=res.get("values"),
                    lowh_logits=res.get("policy_logits"),
                    chosen_idx=chosen_local,
                    temperature=0.1)

                if act_idx < 0:
                    # Could not encode chosen action; skip
                    game.step(chosen_local)
                    continue

                mask[act_idx] = 1.0  # ensure ground-truth label is legal

                step_w = {
                    "search": 1.0,    # gold standard
                    "terminal": 1.5,  # high-value win signal
                    "lowH": 0.3,      # self-distillation, less informative
                }.get(kind, 1.0)
                kind_id = {"search": 0, "lowH": 1, "terminal": 2}.get(kind, 0)

                # search_value = V(s) from acting player's perspective
                # in [-1, 1]. For 'search': max over top-K candidate values
                # (= deep_search_root's return value). For 'terminal': +1.0
                # (we win immediately). For 'lowH': 0.0 (no search done; will
                # be masked out of value loss via signal_kind).
                if kind == "search" and res.get("values"):
                    valid_vals = [v for v in res["values"] if v > -1.5]
                    search_value = float(max(valid_vals)) if valid_vals else 0.0
                elif kind == "terminal":
                    search_value = 1.0
                else:
                    search_value = 0.0

                root_table = _root_table(res, le, ae)
                steps.append({
                    "nf": nf_buf.copy(),
                    "ef": ef_buf.copy(),
                    "ff": ff_buf.copy(),
                    "mask": mask,
                    "action_idx": int(act_idx),
                    "chosen_legal_idx": int(chosen_local),
                    "legal_count": int(n_le),
                    "policy_target": policy_target,
                    "signal_kind": kind_id,
                    "search_value": search_value,
                    "step_weight_per_step": float(step_w),
                    "player": int(cp),
                    "game_idx": int(game_idx),
                    "seed": int(seed),
                    "turn_number": int(game.turn_number),
                    "decision_index": int(decision_index),
                    **root_table,
                })
                decision_index += 1
                seat_decisions[cp] += 1
                game.step(chosen_local)
            else:
                # Sparse mode (one-hot only): original behavior
                chosen_local = active_bot.pick(game)
                chosen_action = le[chosen_local]
                try:
                    act_idx = ae.encode(chosen_action)
                except ValueError:
                    game.step(chosen_local)
                    continue

                mask[act_idx] = 1.0
                steps.append({
                    "nf": nf_buf.copy(),
                    "ef": ef_buf.copy(),
                    "ff": ff_buf.copy(),
                    "mask": mask,
                    "action_idx": int(act_idx),
                    "chosen_legal_idx": int(chosen_local),
                    "legal_count": int(n_le),
                    "player": int(cp),
                    "game_idx": int(game_idx),
                    "seed": int(seed),
                    "turn_number": int(game.turn_number),
                    "decision_index": int(decision_index),
                    **_empty_root_table(),
                })
                decision_index += 1
                seat_decisions[cp] += 1
                game.step(chosen_local)
        else:
            chosen = policy_argmax(game, le)
            game.step(chosen)

    elapsed = time.time() - t_start
    winner = game.winner()
    vps = [int(game._game.state.player_state[s][0]) for s in range(4)]
    ranks = [
        sorted(range(4), key=lambda s: (-vps[s], s)).index(seat) + 1
        for seat in range(4)
    ]

    reward_vec = np.zeros(4, dtype=np.float32)
    if winner is not None:
        reward_vec[winner] = 1.0

    # Per-step weights (use kind-based weight if dense, else uniform)
    if dense:
        step_weights = np.array(
            [s.get("step_weight_per_step", 1.0) for s in steps],
            dtype=np.float32)
    else:
        step_weights = np.ones(len(steps), dtype=np.float32)

    if all_seats:
        stats = " | ".join(b.stats_summary() for b in bots)
    else:
        stats = bot.stats_summary()

    return {
        "game_idx": game_idx,
        "seed": seed,
        "super_seat": super_seat,
        "winner": winner,
        "vps": vps,
        "ranks": ranks,
        "n_steps": len(steps),
        "seat_decisions": seat_decisions,
        "kind_counts": kind_counts,
        "n_turns": game.turn_number,
        "elapsed": elapsed,
        "steps": steps,
        "reward_vec": reward_vec,
        "step_weights": step_weights,
        "stats": stats,
    }


def save_shard(games, output_dir, shard_id):
    """Aggregate per-game results into one .pt shard.

    Backward compatible: existing fields preserved. Adds optional
    `policy_target` (S, 397) and `signal_kind` (S,) when steps include
    them (i.e., when collected with `dense=True`).
    """
    os.makedirs(output_dir, exist_ok=True)

    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player = [], [], []
    all_reward, all_sw, all_np = [], [], []
    all_game_idx, all_seed, all_turn, all_decision = [], [], [], []
    all_legal_count, all_chosen_legal = [], []
    all_winner, all_final_vps, all_final_rank = [], [], []
    all_cand_le, all_cand_act, all_cand_val, all_cand_count = [], [], [], []
    all_pt = []     # policy_target (only if dense)
    all_kind = []   # signal_kind (only if dense)
    all_sv = []     # search_value (only if dense)
    has_dense = False

    for g in games:
        rv = g["reward_vec"]
        sw = g["step_weights"]
        final_vps = np.asarray(g["vps"], dtype=np.int16)
        final_rank = np.asarray(g["ranks"], dtype=np.int16)
        for i, s in enumerate(g["steps"]):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_sw.append(sw[i])
            all_np.append(4)
            all_game_idx.append(s.get("game_idx", g["game_idx"]))
            all_seed.append(s.get("seed", g["seed"]))
            all_turn.append(s.get("turn_number", -1))
            all_decision.append(s.get("decision_index", i))
            all_legal_count.append(s.get("legal_count", -1))
            all_chosen_legal.append(s.get("chosen_legal_idx", -1))
            all_winner.append(-1 if g["winner"] is None else g["winner"])
            all_final_vps.append(final_vps)
            all_final_rank.append(final_rank)
            all_cand_le.append(s.get("candidate_legal_idx",
                                     _empty_root_table()["candidate_legal_idx"]))
            all_cand_act.append(s.get("candidate_action_idx",
                                      _empty_root_table()["candidate_action_idx"]))
            all_cand_val.append(s.get("candidate_value",
                                      _empty_root_table()["candidate_value"]))
            all_cand_count.append(s.get("candidate_count", 0))
            if "policy_target" in s:
                has_dense = True
                all_pt.append(s["policy_target"])
                all_kind.append(s.get("signal_kind", 0))
                all_sv.append(s.get("search_value", 0.0))

    if not all_nf:
        print("  ! No steps to save.", flush=True)
        return 0

    data = {
        "node_features": torch.from_numpy(np.stack(all_nf)),
        "edge_features": torch.from_numpy(np.stack(all_ef)),
        "flat_features": torch.from_numpy(np.stack(all_ff)),
        "action_mask": torch.from_numpy(np.stack(all_mask)),
        "action_idx": torch.tensor(all_act, dtype=torch.int64),
        "player": torch.tensor(all_player, dtype=torch.int64),
        "reward_vec": torch.from_numpy(np.stack(all_reward)),
        "step_weight": torch.tensor(all_sw, dtype=torch.float32),
        "num_players": torch.tensor(all_np, dtype=torch.int64),
        "game_idx": torch.tensor(all_game_idx, dtype=torch.int64),
        "seed": torch.tensor(all_seed, dtype=torch.int64),
        "turn_number": torch.tensor(all_turn, dtype=torch.int64),
        "decision_index": torch.tensor(all_decision, dtype=torch.int64),
        "legal_count": torch.tensor(all_legal_count, dtype=torch.int64),
        "chosen_legal_idx": torch.tensor(all_chosen_legal, dtype=torch.int64),
        "winner": torch.tensor(all_winner, dtype=torch.int64),
        "final_vps": torch.from_numpy(np.stack(all_final_vps)),
        "final_rank": torch.from_numpy(np.stack(all_final_rank)),
        "candidate_legal_idx": torch.from_numpy(np.stack(all_cand_le)),
        "candidate_action_idx": torch.from_numpy(np.stack(all_cand_act)),
        "candidate_value": torch.from_numpy(np.stack(all_cand_val)),
        "candidate_count": torch.tensor(all_cand_count, dtype=torch.int64),
    }
    if has_dense and len(all_pt) == len(all_nf):
        data["policy_target"] = torch.from_numpy(np.stack(all_pt))
        data["signal_kind"] = torch.tensor(all_kind, dtype=torch.int64)
        data["search_value"] = torch.tensor(all_sv, dtype=torch.float32)

    path = os.path.join(output_dir, f"{shard_id}.pt")
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.rename(tmp, path)
    return len(all_nf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=str, default="csrc/data_super_m2")
    p.add_argument("--weights", type=str, default="csrc/nn_weights_m2.bin")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--k-schedule", type=str, default="12,8,6,5,4,3")
    p.add_argument("--time-ms", type=int, default=4000)
    p.add_argument("--seed-base", type=int, default=100000)
    p.add_argument("--shard-id", type=str, default=None)
    p.add_argument("--all-seats", action="store_true",
                   help="All 4 seats are super_m2; record every decision.")
    p.add_argument("--dense", action="store_true",
                   help="Record dense soft policy_target from per-candidate "
                        "search values (vs one-hot on argmax).")
    args = p.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.abspath(args.weights)
    if not os.path.isabs(args.weights) and not os.path.exists(weights_path):
        weights_path = os.path.join(project_root, args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")

    output_dir = args.out if os.path.isabs(args.out) \
        else os.path.join(project_root, args.out)

    schedule = tuple(int(x) for x in args.k_schedule.split(","))
    if args.shard_id is not None:
        shard_id = args.shard_id
    elif args.all_seats:
        shard_id = f"super_m2_4way_seed{args.seed_base}_{args.games}g"
    else:
        shard_id = f"super_m2_seed{args.seed_base}_{args.games}g"

    jobs = []
    for gi in range(args.games):
        seed = args.seed_base + gi * 7919  # spread out, not contiguous
        super_seat = -1 if args.all_seats else (gi % 4)
        jobs.append((gi, seed, super_seat, weights_path, args.depth,
                     schedule, args.time_ms, args.dense))

    if args.all_seats:
        header = "=== super_m2 4-way self-play (all seats record) ==="
    else:
        header = "=== super_m2 (teacher) vs 3x NN-0ply (opponents) ==="
    print(header)
    print(f"  Games: {args.games}")
    print(f"  Workers: {args.workers}")
    print(f"  Depth: {args.depth} k={schedule} time={args.time_ms}ms")
    print(f"  Weights: {weights_path}")
    print(f"  Output: {output_dir}/{shard_id}.pt")
    print(f"  Seeds: {args.seed_base} + i*7919 for i=0..{args.games - 1}")
    print(f"  Dense: {args.dense} "
          f"({'soft policy_target from search values' if args.dense else 'one-hot only'})")
    print(flush=True)

    ctx = mp.get_context("spawn")
    nn_wins = 0
    seat_wins = [0, 0, 0, 0]
    total_steps = 0
    total_turns = 0
    total_super_time = 0.0
    rank_sum = 0
    games_collected = []

    t_start = time.time()
    with ctx.Pool(processes=args.workers) as pool:
        for r in pool.imap_unordered(_play_one_game, jobs):
            games_collected.append(r)
            total_steps += r["n_steps"]
            total_turns += r["n_turns"]
            total_super_time += r["elapsed"]
            elapsed = time.time() - t_start
            done = len(games_collected)

            if args.all_seats:
                if r["winner"] is not None:
                    seat_wins[r["winner"]] += 1
                seat_dec = r["seat_decisions"]
                w = r["winner"]
                w_str = f"P{w}" if w is not None else "—"
                print(f"  [{done:>3d}/{args.games}] g{r['game_idx']:>3d} "
                      f"seed={r['seed']:>7d}  winner={w_str}  "
                      f"VP=[{r['vps'][0]} {r['vps'][1]} {r['vps'][2]} {r['vps'][3]}]  "
                      f"per-seat-decisions=[{seat_dec[0]} {seat_dec[1]} "
                      f"{seat_dec[2]} {seat_dec[3]}]  "
                      f"steps={r['n_steps']:>3d} turns={r['n_turns']:>3d} "
                      f"({r['elapsed']:.0f}s)  | wall={elapsed:.0f}s",
                      flush=True)
            else:
                rank = sorted(range(4),
                              key=lambda s: -r["vps"][s]).index(r["super_seat"]) + 1
                rank_sum += rank
                tag = "WIN " if r["winner"] == r["super_seat"] else "loss"
                if r["winner"] == r["super_seat"]:
                    nn_wins += 1
                print(f"  [{done:>3d}/{args.games}] g{r['game_idx']:>3d} "
                      f"seed={r['seed']:>7d} super=P{r['super_seat']} {tag} "
                      f"VP=[{r['vps'][0]} {r['vps'][1]} {r['vps'][2]} {r['vps'][3]}] "
                      f"rank={rank} steps={r['n_steps']:>3d} turns={r['n_turns']:>3d} "
                      f"({r['elapsed']:.0f}s) "
                      f"| WR={nn_wins/done:.0%} elapsed={elapsed:.0f}s",
                      flush=True)

    elapsed = time.time() - t_start
    games_collected.sort(key=lambda g: g["game_idx"])

    n_saved = save_shard(games_collected, output_dir, shard_id)
    out_path = os.path.join(output_dir, f"{shard_id}.pt")

    print()
    print(f"===== RESULTS =====")
    print(f"  Games:        {args.games}")
    if args.all_seats:
        print(f"  Seat win counts: P0={seat_wins[0]} P1={seat_wins[1]} "
              f"P2={seat_wins[2]} P3={seat_wins[3]}  "
              f"(self-play, expected ~25% each)")
    else:
        print(f"  super_m2 WR:  {nn_wins}/{args.games} "
              f"({100 * nn_wins / args.games:.1f}%)")
        print(f"  Avg rank:     {rank_sum / args.games:.2f} / 4")
    print(f"  Steps saved:  {n_saved:,} (avg {n_saved / args.games:.1f} per game)")
    print(f"  Avg turns:    {total_turns / args.games:.0f}")
    print(f"  Wall time:    {elapsed:.1f}s "
          f"({args.games / max(elapsed, 1e-9) * 60:.1f} games/min)")
    print(f"  Super CPU:    {total_super_time:.1f}s "
          f"({total_super_time / max(args.games, 1):.1f}s avg per super_m2 game-CPU-time)")
    print(f"  Speedup:      {total_super_time / max(elapsed, 1e-9):.1f}x "
          f"(vs single-core)")
    print(f"  Saved to:     {out_path}")
    print(f"  Shard size:   {os.path.getsize(out_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
