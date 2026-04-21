#!/usr/bin/env python3
"""ExIt vs AB2 actors: 1 NN seat with ExIt search + AB2 seats for the rest.

Asymmetric self-play: in each game, ONE seat plays the live NN policy with
ABt15 k=2 search and AB-value leaf (the "high-quality" config we tested);
the OTHER seats play proper alpha-beta minimax (depth 2 with full
chance-node expectimax — Python catanatron AB2 equivalent).

Only the NN seat's decisions are recorded into shards. The model thus
learns to imitate its own search-improved play against the strongest
heuristic opponent we have.

Mixes nicely with `exit_gpu_actors.py` (NN vs NN self-play) and
`ab2_stream.py` (pure AB2 self-play) — all three write the same shard
format into the shared pending dir; the learner doesn't care which actor
type produced a given shard.

Resources:
  - 1 GPU for NN inference
  - A few CPU cores for AB2 opponents (each `alphabeta_search` call is
    single-threaded C; per actor ~1-2 cores)

Launch via human_bot/exit_vs_ab2_actors.sh.
"""
from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
import time
import traceback

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch

GAMES_PER_SHARD = 25
MAX_TURNS = 1000
MAX_STEPS_PER_GAME = 2000
MASK_DIM = 397
AD = 337

_ACT_MOD = np.ones(MASK_DIM, dtype=np.float32)
_ACT_MOD[0] = 0.2
_ACT_MOD[1] = 0.5
_ACT_MOD[2:5] = 1.5
_ACT_MOD[5:113] = 1.5
_ACT_MOD[113:185] = 1.5
_ACT_MOD[185:280] = 1.5
_ACT_MOD[280:285] = 0.3
_ACT_MOD[285:310] = 1.5
_ACT_MOD[310:397] = 1.3


def compute_step_weights(steps, reward_vec):
    """M2 step-weight formula (matches exit_gpu_actors / ab2_stream)."""
    winner = int(np.argmax(reward_vec)) if reward_vec.max() > 0 else -1
    S = len(steps)
    weights = np.ones(S, dtype=np.float32)
    speed_mult = 1.0 + max(0.0, min(0.5, (600 - S) / 600.0))
    for i, s in enumerate(steps):
        progress = i / max(S - 1, 1)
        if s["player"] == winner:
            base = (1.0 + progress) * speed_mult
        else:
            base = max(0.2, 0.6 - 0.4 * progress)
        weights[i] = base * _ACT_MOD[min(s["action_idx"], MASK_DIM - 1)]
    return weights


def temperature_for_round(round_num, start=1.0, end=0.2, anneal=200):
    if round_num >= anneal:
        return end
    return start + (end - start) * (round_num / anneal)


def atomic_torch_save(data, path):
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.rename(tmp, path)


def save_shard(games_data, output_dir, shard_id):
    """Same shard format as c_selfplay / ab2_stream / exit_gpu_actors."""
    all_nf, all_ef, all_ff = [], [], []
    all_mask, all_act, all_player, all_reward, all_sw = [], [], [], [], []
    all_np = []
    for steps, rv, sw, n_players in games_data:
        for i, s in enumerate(steps):
            all_nf.append(s["nf"])
            all_ef.append(s["ef"])
            all_ff.append(s["ff"])
            all_mask.append(s["mask"])
            all_act.append(s["action_idx"])
            all_player.append(s["player"])
            all_reward.append(rv)
            all_sw.append(sw[i])
            all_np.append(n_players)
    if not all_nf:
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
    }
    atomic_torch_save(data, os.path.join(output_dir, f"{shard_id}.pt"))
    return len(all_nf)


def run_actor(actor_id, gpu_id, ckpt_path, shard_dir, ckpt_dir,
              search_depth, top_k, max_pending, player_counts,
              reload_interval):
    try:
        _run_actor(actor_id, gpu_id, ckpt_path, shard_dir, ckpt_dir,
                   search_depth, top_k, max_pending, player_counts,
                   reload_interval)
    except Exception:
        print(f"!!! [exit_vs_ab2 actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_actor(actor_id, gpu_id, ckpt_path, shard_dir, ckpt_dir,
               search_depth, top_k, max_pending, player_counts,
               reload_interval):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import (
        Action as CAction, SearchCtx, ValueFn, MAX_ACTIONS,
    )
    from human_bot.model import HumanBotNet
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    lib = load_library()
    ae = ActionEncoder()
    eval_fn = ValueFn(lib.base_value_fn)
    ab_ctx = SearchCtx()
    ab_buf = (CAction * MAX_ACTIONS)()

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    net = HumanBotNet.load_checkpoint(ckpt_path, device=device)
    net.eval()
    weights_mtime = os.path.getmtime(ckpt_path)

    g0 = CatanGame(seed=0); g0.reset()
    se = g0.make_state_encoder()
    edge_index = se._edge_index.to(device)
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM

    pc_arr = np.asarray(player_counts, dtype=np.int64)
    rng = np.random.default_rng((actor_id + 1) * 999_983)

    pending_dir = os.path.join(shard_dir, "pending")
    os.makedirs(pending_dir, exist_ok=True)
    round_file = os.path.join(ckpt_dir, ".round")

    def get_round():
        if os.path.exists(round_file):
            try:
                return int(open(round_file).read().strip())
            except Exception:
                return 0
        return 0

    # ── Proper AB2 (depth 2, full expectimax) for opponent seats ────
    def ab2_pick(g):
        le = g.get_legal_actions()
        n = len(le)
        if n == 0: return None, le
        if n == 1: return 0, le
        cg = g._game
        bc = cg.state.colors[cg.state.current_player_index]
        for i, a in enumerate(le):
            ab_buf[i] = a
        res = lib.alphabeta_search(
            ctypes.byref(ab_ctx), ctypes.byref(cg), ab_buf,
            ctypes.c_int(n), ctypes.c_int(2),
            ctypes.c_double(-1e30), ctypes.c_double(1e30),
            ctypes.c_int(bc), eval_fn,
        )
        cb = ctypes.string_at(ctypes.byref(res.action),
                               ctypes.sizeof(res.action))
        for i, a in enumerate(le):
            if ctypes.string_at(ctypes.byref(a),
                                ctypes.sizeof(a)) == cb:
                return i, le
        return 0, le

    # ── NN inference helpers ────────────────────────────────────────
    def nn_forward_batch(games_subset):
        B = len(games_subset)
        if B == 0: return None
        nf = np.zeros((B, N, NF), dtype=np.float32)
        ef = np.zeros((B, E, EF), dtype=np.float32)
        ff = np.zeros((B, FFD), dtype=np.float32)
        mk = np.zeros((B, MASK_DIM), dtype=np.float32)
        for i, g in enumerate(games_subset):
            se.encode_into(g.get_state_view(), nf[i], ef[i], ff[i])
            le = g.get_legal_actions()
            m = ae.get_action_mask(le).numpy()
            mk[i, :len(m)] = m
        with torch.no_grad():
            out = net({
                "node_features": torch.from_numpy(nf).to(device, non_blocking=True),
                "edge_index": edge_index,
                "edge_features": torch.from_numpy(ef).to(device, non_blocking=True),
                "flat_features": torch.from_numpy(ff).to(device, non_blocking=True),
                "action_mask": torch.from_numpy(mk).to(device, non_blocking=True),
            })
        return out["policy_logits"][:, :AD].cpu().numpy()

    def nn_argmax_one(g):
        le = g.get_legal_actions()
        if not le: return None
        if len(le) == 1: return 0
        logits = nn_forward_batch([g])[0]
        a2i = {}
        for i, a in enumerate(le):
            try: a2i[ae.encode(a)] = i
            except ValueError: continue
        if not a2i: return 0
        scored = [(logits[e], i) for e, i in a2i.items()]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def nn_argmax_batched(games_list):
        B = len(games_list)
        if B == 0: return []
        idx_with_choice = []
        chosen = [None] * B
        single_le = [None] * B
        for i, g in enumerate(games_list):
            le = g.get_legal_actions()
            single_le[i] = le
            if not le: chosen[i] = None
            elif len(le) == 1: chosen[i] = 0
            else: idx_with_choice.append(i)
        if idx_with_choice:
            sub = [games_list[i] for i in idx_with_choice]
            logits = nn_forward_batch(sub)
            for j, i in enumerate(idx_with_choice):
                le = single_le[i]
                a2i = {}
                for k, a in enumerate(le):
                    try: a2i[ae.encode(a)] = k
                    except ValueError: continue
                if not a2i: chosen[i] = 0; continue
                scored = [(logits[j][e], k) for e, k in a2i.items()]
                scored.sort(key=lambda x: -x[0])
                chosen[i] = scored[0][1]
        return chosen

    def policy_top_k(g, le, k):
        if len(le) <= k: return list(range(len(le)))
        logits = nn_forward_batch([g])[0]
        a2i = {}
        for i, a in enumerate(le):
            try: a2i[ae.encode(a)] = i
            except ValueError: continue
        scored = [(logits[e], i) for e, i in a2i.items()]
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:k]]

    def ab_leaf_eval(g, our_seat):
        cg = g._game
        bot_color = cg.state.colors[our_seat]
        return float(lib.base_value_fn(ctypes.byref(cg), bot_color))

    # ── ExIt search (NN policy + batched argmax rollout + AB2 leaf) ──
    def abt_search(game, le, depth, temperature):
        """Returns (target_action, played_action). Target = search argmax
        (consistent training signal). Played = temperature-sampled (game
        exploration). See exit_gpu_actors.py for rationale."""
        seat = game.current_player()
        cands = policy_top_k(game, le, top_k)
        K = len(cands)

        clones = [game.clone() for _ in range(K)]
        alive = [True] * K
        for p, ci in enumerate(cands):
            try:
                clones[p].step(ci)
                if clones[p].is_terminal() or clones[p].turn_number >= MAX_TURNS:
                    alive[p] = False
            except Exception:
                alive[p] = False

        for ply in range(2, depth + 1):
            live = [p for p, a in enumerate(alive) if a]
            if not live: break
            sub = [clones[p] for p in live]
            chosen_idxs = nn_argmax_batched(sub)
            for j, p in enumerate(live):
                gc = clones[p]; idx = chosen_idxs[j]
                if idx is None: alive[p] = False; continue
                try: gc.step(idx)
                except Exception: alive[p] = False; continue
                if gc.is_terminal() or gc.turn_number >= MAX_TURNS:
                    alive[p] = False

        values = np.zeros(K, dtype=np.float32)
        for p in range(K):
            gc = clones[p]
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (
                    -10.0 if w is not None else 0.0)
            else:
                v = ab_leaf_eval(gc, seat)
            v = apply_action_bonus(v, le[cands[p]])
            values[p] = v

        argmax_p = int(np.argmax(values))
        target_action = fix_robber_steal(cands[argmax_p], le)

        if K == 1 or temperature < 0.01:
            played_p = argmax_p
        else:
            shifted = values - values.max()
            probs = np.exp(shifted / temperature)
            probs /= probs.sum()
            played_p = int(np.random.choice(K, p=probs))
        played_action = fix_robber_steal(cands[played_p], le)

        return target_action, played_action

    # ── Play one asymmetric game; record NN's decisions only ─────────
    def play_one_game(seed, n_players, nn_seat, temperature):
        cfg = GameConfig(num_players=n_players)
        game = CatanGame(seed=seed, config=cfg, random_board=True)
        game.reset()
        steps = []  # NN's decisions only

        while (not game.is_terminal()
               and game.turn_number < MAX_TURNS
               and len(steps) < MAX_STEPS_PER_GAME):
            le = game.get_legal_actions()
            if not le: break
            cp = game.current_player()

            if cp != nn_seat:
                # AB2 plays this seat
                idx, _ = ab2_pick(game)
                if idx is None: break
                game.step(idx)
                continue

            # NN's turn — record snapshot, run search
            sv = game.get_state_view()
            nf = np.zeros((N, NF), dtype=np.float32)
            ef = np.zeros((E, EF), dtype=np.float32)
            ff = np.zeros(FFD, dtype=np.float32)
            se.encode_into(sv, nf, ef, ff)
            mask = ae.get_action_mask(le).numpy()

            target_action = None
            if len(le) == 1:
                played = 0
            elif game.turn_number <= 7:
                played = nn_argmax_one(game)
                if played is None: played = 0
                target_action = played
            else:
                target_action, played = abt_search(game, le, search_depth, temperature)

            recorded = target_action if target_action is not None else played
            try:
                enc_action = ae.encode(le[recorded])
            except ValueError:
                game.step(played)
                continue

            if len(le) > 1:
                steps.append({
                    "nf": nf, "ef": ef, "ff": ff,
                    "mask": mask,
                    "action_idx": enc_action,
                    "player": cp,
                })
            game.step(played)

        winner = game.winner()
        reward_vec = np.zeros(4, dtype=np.float32)
        if winner is not None:
            speed_bonus = max(0.0, min(0.5, (300 - game.turn_number) / 300.0))
            reward_vec[winner] = 1.0 + speed_bonus
            for s in range(n_players):
                if s == winner: continue
                vp = game._game.state.player_state[s][0]
                reward_vec[s] = vp / 20.0
        sw = compute_step_weights(steps, reward_vec)
        return steps, reward_vec, sw, winner

    # ── Actor loop ──────────────────────────────────────────────────
    game_batch = []
    shard_idx = 0
    total_games = 0
    total_steps = 0
    nn_wins = 0
    games_by_pc = {int(pc): 0 for pc in pc_arr}
    t_start = time.time()

    print(f"[exit_vs_ab2 actor {actor_id}] Started on {device}, "
          f"depth={search_depth} top-k={top_k} pcs={list(pc_arr)} "
          f"max_pending={max_pending}", flush=True)

    while True:
        cur_round = get_round()
        temp = temperature_for_round(cur_round)
        seed = (actor_id + 1) * 1_000_000 + total_games
        n_players = int(rng.choice(pc_arr))
        nn_seat = int(rng.integers(0, n_players))  # rotate NN seat each game
        games_by_pc[n_players] += 1
        steps, rv, sw, winner = play_one_game(seed, n_players, nn_seat, temp)
        if winner is not None and winner == nn_seat:
            nn_wins += 1

        if not steps:
            total_games += 1
            continue

        game_batch.append((steps, rv, sw, n_players))
        total_games += 1
        total_steps += len(steps)

        if len(game_batch) >= GAMES_PER_SHARD:
            sid = f"exit_vs_ab2_a{actor_id:03d}_{shard_idx:06d}"
            save_shard(game_batch, pending_dir, sid)
            game_batch = []
            shard_idx += 1
            while True:
                try:
                    n_pending = len([f for f in os.listdir(pending_dir)
                                     if f.endswith(".pt")
                                     and not f.endswith(".tmp")])
                except FileNotFoundError:
                    n_pending = 0
                if n_pending <= max_pending: break
                time.sleep(2)

        if total_games % reload_interval == 0:
            try:
                mt = os.path.getmtime(ckpt_path)
                if mt > weights_mtime:
                    net = HumanBotNet.load_checkpoint(ckpt_path, device=device)
                    net.eval()
                    weights_mtime = mt
                    print(f"[exit_vs_ab2 actor {actor_id}] Reloaded weights, "
                          f"round={cur_round} t={temp:.2f}", flush=True)
            except Exception as e:
                print(f"[exit_vs_ab2 actor {actor_id}] Reload failed: {e}",
                      flush=True)

        if total_games % 5 == 0:
            elapsed = time.time() - t_start
            gps = total_games / elapsed if elapsed > 0 else 0
            avg_s = total_steps / max(total_games, 1)
            wr = nn_wins / max(total_games, 1)
            pc_summary = " ".join(f"{k}p={v}" for k, v in sorted(games_by_pc.items()))
            print(f"[exit_vs_ab2 actor {actor_id}] {total_games} games, "
                  f"{shard_idx} shards, {gps:.2f} g/s, "
                  f"~{avg_s:.0f} NN-steps/g, t={temp:.2f}, "
                  f"NN-WR={wr:.0%} | {pc_summary}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="ExIt vs AB2: 1 NN seat (search) + (N-1) AB2 seats")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--actor-id-offset", type=int, default=0)
    parser.add_argument("--search-depth", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--max-pending", type=int, default=200)
    parser.add_argument("--reload-interval", type=int, default=20)
    parser.add_argument("--player-counts", type=str, default="2,3,4")
    args = parser.parse_args()

    try:
        args.player_counts = tuple(int(x) for x in args.player_counts.split(",")
                                    if x.strip())
    except ValueError:
        parser.error(f"--player-counts must be comma-separated ints")
    for n in args.player_counts:
        if n not in (2, 3, 4):
            parser.error(f"--player-counts values must be 2, 3 or 4 (got {n})")

    os.makedirs(os.path.join(args.shard_dir, "pending"), exist_ok=True)

    print(f"ExIt vs AB2 actors")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Shard dir:   {args.shard_dir}")
    print(f"  Ckpt dir:    {args.ckpt_dir}")
    print(f"  Actors:      {args.num_actors} across {args.num_gpus} GPU(s)")
    print(f"  NN search:   ABt{args.search_depth} top-k={args.top_k} (AB-leaf)")
    print(f"  Opponents:   AB2 (proper alphabeta, depth 2, expectimax)")
    print(f"  Player cnts: {list(args.player_counts)}")
    print(f"  Max pending: {args.max_pending}")
    print(f"  Offset:      {args.actor_id_offset}")
    print(flush=True)

    ctx = mp.get_context("spawn")
    procs = []
    for i in range(args.num_actors):
        aid = args.actor_id_offset + i
        gpu_id = i % args.num_gpus
        p = ctx.Process(
            target=run_actor,
            args=(aid, gpu_id, args.checkpoint, args.shard_dir, args.ckpt_dir,
                  args.search_depth, args.top_k, args.max_pending,
                  args.player_counts, args.reload_interval),
            daemon=True,
        )
        p.start()
        procs.append(p)

    print(f"[main] {len(procs)} actors started", flush=True)

    try:
        while True:
            time.sleep(60)
            alive = sum(1 for p in procs if p.is_alive())
            if alive == 0:
                print("[main] All actors died.", flush=True); break
            if alive < len(procs):
                print(f"[main] {len(procs) - alive}/{len(procs)} died, "
                      f"{alive} running.", flush=True)
    except KeyboardInterrupt:
        print("[main] Interrupted.", flush=True)

    for p in procs:
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
