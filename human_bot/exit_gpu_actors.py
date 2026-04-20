#!/usr/bin/env python3
"""ExIt actors: GPU NN policy + ABt30 k=2 search w/ AB-leaf, M2 reward signal.

Each actor:
  - runs a batch of B parallel games (so a single forward pass batches the
    NN inference across them)
  - at each decision, uses NN policy to pick top-k candidates, then runs
    an N-ply argmax rollout per candidate using NN argmax for non-cp seats,
    and evaluates the leaf with AB2 base_value_fn (avoids the broken NN
    value head)
  - samples the chosen action with temperature over the search-scored
    candidates (for exploration during training data generation)
  - records every decision's snapshot in c_selfplay shard format with
    M2-style step weights (winner moves upweighted) and num_players field

Pairs with c_selfplay.py --role learner consuming the same shard dir.

Each actor process needs one GPU. Use --batch-games-per-actor to control
how many parallel games are co-scheduled inside one process for batching.

Launch via the launcher script human_bot/exit_gpu_actors.sh which handles
the libcatan rebuild + checkpoint reload.
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
    """M2 step-weight formula: winner moves get bigger weight + speed bonus."""
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
    """Same format as c_selfplay.save_shard: includes num_players + step_weight."""
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


# =====================================================================
# Actor: one GPU, B parallel games, NN+ABt search per decision
# =====================================================================

def run_actor(actor_id, gpu_id, ckpt_path, shard_dir, ckpt_dir,
              search_depth, top_k, batch_games, max_pending,
              player_counts, reload_interval):
    try:
        _run_actor(actor_id, gpu_id, ckpt_path, shard_dir, ckpt_dir,
                   search_depth, top_k, batch_games, max_pending,
                   player_counts, reload_interval)
    except Exception:
        print(f"!!! [exit actor {actor_id}] CRASHED !!!", flush=True)
        traceback.print_exc()


def _run_actor(actor_id, gpu_id, ckpt_path, shard_dir, ckpt_dir,
               search_depth, top_k, batch_games, max_pending,
               player_counts, reload_interval):
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from human_bot.model import HumanBotNet
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    lib = load_library()
    ae = ActionEncoder()

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

    # ── NN inference helpers ────────────────────────────────────────
    def nn_forward_batch(games_subset):
        """Forward all games in subset; return (logits[B,AD], values[B,4])."""
        B = len(games_subset)
        if B == 0:
            return None, None
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
        logits = out["policy_logits"][:, :AD].cpu().numpy()
        return logits

    def nn_argmax_one(g):
        """Single-game NN argmax for use in argmax rollouts."""
        le = g.get_legal_actions()
        if not le:
            return None
        if len(le) == 1:
            return 0
        logits = nn_forward_batch([g])[0]
        # Mask illegal
        a2i = {}
        for i, a in enumerate(le):
            try:
                a2i[ae.encode(a)] = i
            except ValueError:
                continue
        if not a2i:
            return 0
        scored = [(logits[e], i) for e, i in a2i.items()]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def nn_argmax_batched(games_list):
        """Batched argmax — one forward over all games_list. Returns list of indices."""
        B = len(games_list)
        if B == 0:
            return []
        # Filter to those with multiple choices
        idx_with_choice = []
        chosen = [None] * B
        single_le = [None] * B
        for i, g in enumerate(games_list):
            le = g.get_legal_actions()
            single_le[i] = le
            if not le:
                chosen[i] = None
            elif len(le) == 1:
                chosen[i] = 0
            else:
                idx_with_choice.append(i)
        if idx_with_choice:
            sub = [games_list[i] for i in idx_with_choice]
            logits = nn_forward_batch(sub)
            for j, i in enumerate(idx_with_choice):
                le = single_le[i]
                a2i = {}
                for k, a in enumerate(le):
                    try:
                        a2i[ae.encode(a)] = k
                    except ValueError:
                        continue
                if not a2i:
                    chosen[i] = 0; continue
                scored = [(logits[j][e], k) for e, k in a2i.items()]
                scored.sort(key=lambda x: -x[0])
                chosen[i] = scored[0][1]
        return chosen

    def policy_top_k(g, le, k):
        """NN policy → top-k legal action indices."""
        if len(le) <= k:
            return list(range(len(le)))
        logits = nn_forward_batch([g])[0]
        a2i = {}
        for i, a in enumerate(le):
            try:
                a2i[ae.encode(a)] = i
            except ValueError:
                continue
        scored = [(logits[e], i) for e, i in a2i.items()]
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:k]]

    def ab_leaf_eval(g, our_seat):
        """AB2 base_value_fn evaluated from our_seat's perspective."""
        cg = g._game
        bot_color = cg.state.colors[our_seat]
        return float(lib.base_value_fn(ctypes.byref(cg), bot_color))

    # ── ABt search (NN policy + argmax rollout + AB2 leaf) ──────────
    def abt_search(game, le, depth, temperature):
        """For each top-k candidate, run depth-N argmax rollout, score with
        AB2 leaf, sample with temperature.

        Candidates' rollouts run in lockstep — one batched NN forward per
        ply across all live candidate clones. Cuts per-decision NN calls
        from `top_k * (depth-1)` sequential to `(depth-1)` batched.
        """
        seat = game.current_player()
        cands = policy_top_k(game, le, top_k)
        K = len(cands)

        # Initialise candidate clones
        clones = [game.clone() for _ in range(K)]
        alive = [True] * K
        for p, ci in enumerate(cands):
            try:
                clones[p].step(ci)
                if clones[p].is_terminal() or clones[p].turn_number >= MAX_TURNS:
                    alive[p] = False
            except Exception:
                alive[p] = False

        # Lockstep argmax rollout, batched across alive clones
        for ply in range(2, depth + 1):
            live = [p for p, a in enumerate(alive) if a]
            if not live:
                break
            sub = [clones[p] for p in live]
            chosen_idxs = nn_argmax_batched(sub)
            for j, p in enumerate(live):
                gc = clones[p]
                idx = chosen_idxs[j]
                if idx is None:
                    alive[p] = False
                    continue
                try:
                    gc.step(idx)
                except Exception:
                    alive[p] = False
                    continue
                if gc.is_terminal() or gc.turn_number >= MAX_TURNS:
                    alive[p] = False

        # Leaf evaluation
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

        # Temperature-sampled selection over top candidates
        if K == 1 or temperature < 0.01:
            best_p = int(np.argmax(values))
        else:
            shifted = values - values.max()
            probs = np.exp(shifted / temperature)
            probs /= probs.sum()
            best_p = int(np.random.choice(K, p=probs))
        chosen = cands[best_p]
        return fix_robber_steal(chosen, le)

    # ── Per-game state container ────────────────────────────────────
    class GameSlot:
        __slots__ = ("game", "n_players", "steps", "seed")
        def __init__(self, seed, n_players):
            cfg = GameConfig(num_players=n_players)
            self.game = CatanGame(seed=seed, config=cfg, random_board=True)
            self.game.reset()
            self.n_players = n_players
            self.steps = []
            self.seed = seed

    def play_one_game_full(seed, n_players, temperature):
        """Play a full game using NN+ABt search; return (steps, rv, sw, winner)."""
        slot = GameSlot(seed, n_players)
        g = slot.game
        steps = slot.steps

        while (not g.is_terminal()
               and g.turn_number < MAX_TURNS
               and len(steps) < MAX_STEPS_PER_GAME):
            le = g.get_legal_actions()
            if not le: break

            # Save snapshot BEFORE choosing
            sv = g.get_state_view()
            nf = np.zeros((N, NF), dtype=np.float32)
            ef = np.zeros((E, EF), dtype=np.float32)
            ff = np.zeros(FFD, dtype=np.float32)
            se.encode_into(sv, nf, ef, ff)
            mask = ae.get_action_mask(le).numpy()

            if len(le) == 1:
                chosen = 0
            elif g.turn_number <= 7:
                # Early turns (initial placement, first roll): policy argmax,
                # search adds little here and is wasteful
                chosen = nn_argmax_one(g)
                if chosen is None:
                    chosen = 0
            else:
                chosen = abt_search(g, le, search_depth, temperature)

            try:
                enc_action = ae.encode(le[chosen])
            except ValueError:
                g.step(chosen)
                continue

            if len(le) > 1:
                steps.append({
                    "nf": nf, "ef": ef, "ff": ff,
                    "mask": mask,
                    "action_idx": enc_action,
                    "player": g.current_player(),
                })
            g.step(chosen)

        winner = g.winner()
        reward_vec = np.zeros(4, dtype=np.float32)
        if winner is not None:
            speed_bonus = max(0.0, min(0.5, (300 - g.turn_number) / 300.0))
            reward_vec[winner] = 1.0 + speed_bonus
            for seat in range(n_players):
                if seat == winner: continue
                vp = g._game.state.player_state[seat][0]
                reward_vec[seat] = vp / 20.0
        sw = compute_step_weights(steps, reward_vec)
        return steps, reward_vec, sw, winner

    # ── Actor loop ──────────────────────────────────────────────────
    game_batch = []
    shard_idx = 0
    total_games = 0
    total_steps = 0
    wins = np.zeros(4, dtype=np.int64)
    games_by_pc = {int(pc): 0 for pc in pc_arr}
    t_start = time.time()

    print(f"[exit_gpu actor {actor_id}] Started on {device}, "
          f"depth={search_depth} top-k={top_k} pcs={list(pc_arr)} "
          f"max_pending={max_pending}", flush=True)

    while True:
        cur_round = get_round()
        temp = temperature_for_round(cur_round)
        seed = (actor_id + 1) * 1_000_000 + total_games
        n_players = int(rng.choice(pc_arr))
        games_by_pc[n_players] += 1
        steps, rv, sw, winner = play_one_game_full(seed, n_players, temp)

        if not steps:
            total_games += 1
            continue

        game_batch.append((steps, rv, sw, n_players))
        total_games += 1
        total_steps += len(steps)
        if winner is not None:
            wins[winner] += 1

        if len(game_batch) >= GAMES_PER_SHARD:
            sid = f"exit_a{actor_id:03d}_{shard_idx:06d}"
            save_shard(game_batch, pending_dir, sid)
            game_batch = []
            shard_idx += 1

            # Backpressure
            while True:
                try:
                    n_pending = len([f for f in os.listdir(pending_dir)
                                     if f.endswith(".pt")
                                     and not f.endswith(".tmp")])
                except FileNotFoundError:
                    n_pending = 0
                if n_pending <= max_pending:
                    break
                time.sleep(2)

        # Reload checkpoint periodically
        if total_games % reload_interval == 0:
            try:
                mt = os.path.getmtime(ckpt_path)
                if mt > weights_mtime:
                    net = HumanBotNet.load_checkpoint(ckpt_path, device=device)
                    net.eval()
                    weights_mtime = mt
                    print(f"[exit_gpu actor {actor_id}] Reloaded weights, "
                          f"round={cur_round} t={temp:.2f}", flush=True)
            except Exception as e:
                print(f"[exit_gpu actor {actor_id}] Reload failed: {e}", flush=True)

        if total_games % 5 == 0:
            elapsed = time.time() - t_start
            gps = total_games / elapsed if elapsed > 0 else 0
            avg_s = total_steps / total_games if total_games else 0
            pc_summary = " ".join(f"{k}p={v}" for k, v in sorted(games_by_pc.items()))
            print(f"[exit_gpu actor {actor_id}] {total_games} games, "
                  f"{shard_idx} shards, {gps:.2f} g/s, "
                  f"~{avg_s:.0f} steps/g, t={temp:.2f}, "
                  f"wins={wins.tolist()} | {pc_summary}", flush=True)


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU-actor ExIt self-play: NN + ABt search w/ AB-leaf.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--ckpt-dir", type=str, required=True)
    parser.add_argument("--num-actors", type=int, default=4,
                        help="Total actor processes across all GPUs")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs to spread actors across")
    parser.add_argument("--actor-id-offset", type=int, default=0)
    parser.add_argument("--search-depth", type=int, default=30,
                        help="ABt search depth (best: 30)")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Policy top-k pruning at root (best: 2)")
    parser.add_argument("--batch-games", type=int, default=1,
                        help="(Reserved for future cross-game batching)")
    parser.add_argument("--max-pending", type=int, default=200)
    parser.add_argument("--reload-interval", type=int, default=20,
                        help="Check for new checkpoint every N games")
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

    print(f"ExIt GPU actors")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Shard dir:   {args.shard_dir}")
    print(f"  Ckpt dir:    {args.ckpt_dir}")
    print(f"  Actors:      {args.num_actors} across {args.num_gpus} GPU(s)")
    print(f"  Search:      ABt{args.search_depth} top-k={args.top_k} (AB-value leaf)")
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
                  args.search_depth, args.top_k, args.batch_games,
                  args.max_pending, args.player_counts, args.reload_interval),
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
