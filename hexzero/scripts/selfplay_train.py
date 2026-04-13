#!/usr/bin/env python3
"""Phase 2: Self-play training with REINFORCE + behavioral cloning.

Refines a pretrained HexaZero policy through self-play. Each game uses
2 HexaZero seats + 2 AB2 seats with randomized seating.

No 1-ply lookahead anywhere. HZ selects actions directly from the policy
head (temperature-scaled sampling). AB2 uses the C engine's base_value_fn.

Training signal:
  HZ moves  -> REINFORCE: -log pi(a|s) * (outcome - V(s))
  AB2 moves -> Behavioral cloning: -log pi(a_ab2|s)
  All moves -> Value cross-entropy (4-player win distribution)
              + Entropy regularization

The pretrained policy (from Phase 1 supervised learning on AB2 games)
starts strong enough that no lookahead is needed. Self-play then refines
it beyond AB2 level, with the value head providing the REINFORCE baseline.

Usage:
    python -m hexzero.scripts.selfplay_train \\
        --checkpoint checkpoints/best.pt \\
        --output-dir selfplay_checkpoints \\
        --games-per-step 32 --batch-size 2048 --lr 0.0003 \\
        --eval-every 10 --eval-games 25 \\
        --wandb-key KEY
"""

from __future__ import annotations

import argparse
import ctypes
import os
import random
import time

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


# ── AB2 action selection ──────────────────────────────────────────────


_ab2_ch = None
_ab2_ca = None
_ab2_cn = None


def _ab2_pick(lib, game, legal_actions):
    """AB2 greedy 1-ply: pick legal-action index maximising base_value_fn."""
    from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS

    global _ab2_ch, _ab2_ca, _ab2_cn
    if _ab2_ch is None:
        _ab2_ch = CGame()
        _ab2_ca = (CAction * MAX_ACTIONS)()
        _ab2_cn = ctypes.c_int(0)

    cg = game._game
    bc = cg.state.colors[cg.state.current_player_index]
    best_i, best_v = 0, -1e30
    for i, act in enumerate(legal_actions):
        lib.game_copy(ctypes.byref(_ab2_ch), ctypes.byref(cg))
        lib.game_execute(ctypes.byref(_ab2_ch), act, _ab2_ca, ctypes.byref(_ab2_cn))
        v = lib.base_value_fn(ctypes.byref(_ab2_ch), bc)
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def _assign_rewards(games, num_games):
    """Compute graded reward vectors for finished games.

    Returns list of (4,) float32 arrays, one per game.
    Winner gets 1.0; others get VP-ranked partial credit.
    Timed-out games get small rewards.
    """
    reward_vecs = []
    for idx in range(num_games):
        g = games[idx]
        winner = g.winner()
        vps = [g._game.state.player_state[p][0] for p in range(4)]
        ranked = sorted(range(4), key=lambda p: vps[p], reverse=True)
        if winner is None:
            grade = {ranked[0]: 0.10, ranked[1]: 0.05,
                     ranked[2]: 0.02, ranked[3]: 0.00}
        else:
            grade = {ranked[0]: 1.0, ranked[1]: 0.3,
                     ranked[2]: 0.1, ranked[3]: 0.0}
            grade[winner] = 1.0
        reward_vecs.append(
            np.array([grade.get(p, 0.0) for p in range(4)], dtype=np.float32))
    return reward_vecs


# ── Play a batch of concurrent games ──────────────────────────────────


def play_batch(
    net, state_enc, action_enc, lib, device, edge_index_dev,
    num_games: int, seed_base: int, temperature: float = 1.0,
) -> tuple[list[dict], int, int]:
    """Play *num_games* concurrent games (2 HZ + 2 AB2 each).

    Returns
    -------
    steps : list[dict]
        Per-move records with keys: nf, ef, ff, mask, action_idx, player,
        is_hz, game_idx, reward, value_target.
    hz_wins, ab2_wins : int
    """
    from hexzero.game.interface import CatanGame

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF, EF, FF = (state_enc.NODE_FEATURE_DIM,
                  state_enc.EDGE_FEATURE_DIM,
                  state_enc.FLAT_FEATURE_DIM)

    nf_buf = np.zeros((num_games + 1, N, NF), dtype=np.float32)
    ef_buf = np.zeros((num_games + 1, E, EF), dtype=np.float32)
    ff_buf = np.zeros((num_games + 1, FF), dtype=np.float32)
    mask_buf = np.zeros((num_games + 1, AD), dtype=np.float32)

    games = [CatanGame(seed=seed_base + i) for i in range(num_games)]
    for g in games:
        g.reset()

    hz_seats: list[set[int]] = []
    ab2_seats: list[set[int]] = []
    for _ in range(num_games):
        s = list(range(4))
        random.shuffle(s)
        hz_seats.append(set(s[:2]))
        ab2_seats.append(set(s[2:]))

    all_steps: list[dict] = []
    active = list(range(num_games))
    net.eval()

    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 750]
        if not active:
            break

        # ── AB2 turns (sequential, C engine) ──────────────────────────
        progressed = True
        while progressed:
            progressed = False
            for idx in active:
                g = games[idx]
                if g.is_terminal() or g.turn_number >= 750:
                    continue
                cp = g.current_player()
                if cp not in ab2_seats[idx]:
                    continue
                le = g.get_legal_actions()
                if not le:
                    continue

                state_enc.encode_into(
                    g.get_state_view(), nf_buf[0], ef_buf[0], ff_buf[0])
                mask = action_enc.get_action_mask(le).numpy()
                chosen = _ab2_pick(lib, g, le)
                aidx = action_enc.encode(le[chosen])

                all_steps.append({
                    "nf": nf_buf[0].copy(), "ef": ef_buf[0].copy(),
                    "ff": ff_buf[0].copy(), "mask": mask.copy(),
                    "action_idx": aidx, "player": cp,
                    "is_hz": False, "game_idx": idx,
                })
                g.step(chosen)
                progressed = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 750]
        if not active:
            break

        # ── HZ turns (batched GPU forward pass) ──────────────────────
        B = 0
        index_map: list[tuple[int, list]] = []
        for idx in active:
            g = games[idx]
            if g.current_player() not in hz_seats[idx]:
                continue
            le = g.get_legal_actions()
            if not le:
                continue
            state_enc.encode_into(
                g.get_state_view(), nf_buf[B], ef_buf[B], ff_buf[B])
            mask_buf[B] = action_enc.get_action_mask(le).numpy()
            index_map.append((idx, le))
            B += 1

        if B == 0:
            continue

        with torch.no_grad():
            batch = {
                "node_features": torch.from_numpy(nf_buf[:B].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B].copy()).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B].copy()).to(device),
            }
            out = net(batch)
            if temperature != 1.0:
                tempered = out["policy_logits"] / temperature
                tempered = tempered.masked_fill(batch["action_mask"] == 0, -1e9)
                probs = F.softmax(tempered, dim=-1)
            else:
                probs = out["policy_probs"]
            probs_np = probs.cpu().numpy()

        for b, (idx, le) in enumerate(index_map):
            g = games[idx]
            p = probs_np[b]
            if p.sum() < 1e-6:
                p = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
            p = p / p.sum()

            aidx = int(np.random.choice(AD, p=p))
            all_steps.append({
                "nf": nf_buf[b].copy(), "ef": ef_buf[b].copy(),
                "ff": ff_buf[b].copy(), "mask": mask_buf[b].copy(),
                "action_idx": aidx, "player": g.current_player(),
                "is_hz": True, "game_idx": idx,
            })
            chosen = next((i for i, a in enumerate(le)
                           if action_enc.encode(a) == aidx), 0)
            g.step(chosen)

    # ── Terminal rewards ──────────────────────────────────────────────
    reward_vecs = _assign_rewards(games, num_games)

    for step in all_steps:
        gi = step["game_idx"]
        pid = step["player"]
        rv = reward_vecs[gi]
        step["reward"] = float(rv[pid])
        rot = np.roll(rv, -pid).copy()
        rsum = rot.sum()
        step["value_target"] = (rot / rsum if rsum > 1e-8
                                else np.ones(4, dtype=np.float32) * 0.25)

    hz_wins = ab2_wins = 0
    for idx in range(num_games):
        w = games[idx].winner()
        if w is None:
            continue
        if w in hz_seats[idx]:
            hz_wins += 1
        else:
            ab2_wins += 1

    return all_steps, hz_wins, ab2_wins


# ── Expert Iteration: play with 1-ply NN search ──────────────────────


_exit_ch = None
_exit_ca = None
_exit_cn = None


def play_batch_exit(
    net, state_enc, action_enc, lib, device, edge_index_dev,
    num_games: int, seed_base: int, top_k: int = 5,
    temperature: float = 0.3,
) -> list[dict]:
    """Play *num_games* of pure self-play with 1-ply NN search (Expert Iteration).

    All 4 seats are HZ. At each move:
      1. Forward pass to get policy probs
      2. Take top-K actions by policy probability
      3. For each candidate: game_copy, game_execute, encode child state
      4. Batch forward pass to get V(s') for each child
      5. Pick action with highest V(s')[current_player] as improved target
      6. Step the game (with temperature sampling for exploration)
      7. Record state + improved_action as training target

    Returns list[dict] with keys: nf, ef, ff, mask, action_idx, player, game_idx
    """
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.structs import Game as CGame, Action as CAction, MAX_ACTIONS

    global _exit_ch, _exit_ca, _exit_cn
    if _exit_ch is None:
        _exit_ch = CGame()
        _exit_ca = (CAction * MAX_ACTIONS)()
        _exit_cn = ctypes.c_int(0)

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF, EF, FF = (state_enc.NODE_FEATURE_DIM,
                  state_enc.EDGE_FEATURE_DIM,
                  state_enc.FLAT_FEATURE_DIM)

    # Buffers for parent states (one per game)
    nf_buf = np.zeros((num_games + 1, N, NF), dtype=np.float32)
    ef_buf = np.zeros((num_games + 1, E, EF), dtype=np.float32)
    ff_buf = np.zeros((num_games + 1, FF), dtype=np.float32)
    mask_buf = np.zeros((num_games + 1, AD), dtype=np.float32)

    # Buffers for child states (top_k per game)
    max_children = num_games * top_k
    child_nf = np.zeros((max_children, N, NF), dtype=np.float32)
    child_ef = np.zeros((max_children, E, EF), dtype=np.float32)
    child_ff = np.zeros((max_children, FF), dtype=np.float32)

    games = [CatanGame(seed=seed_base + i) for i in range(num_games)]
    for g in games:
        g.reset()

    all_steps: list[dict] = []
    active = list(range(num_games))
    net.eval()
    total_actions = 0
    max_total = num_games * 2000

    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 750]
        if not active or total_actions >= max_total:
            break

        # ── Encode all active parent states ───────────────────────
        B = 0
        index_map: list[tuple[int, list]] = []
        for idx in active:
            g = games[idx]
            le = g.get_legal_actions()
            if not le:
                continue
            state_enc.encode_into(
                g.get_state_view(), nf_buf[B], ef_buf[B], ff_buf[B])
            mask_buf[B] = action_enc.get_action_mask(le).numpy()
            index_map.append((idx, le))
            B += 1

        if B == 0:
            continue

        # ── Forward pass on parent states to get policy ───────────
        with torch.no_grad():
            parent_batch = {
                "node_features": torch.from_numpy(nf_buf[:B].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B].copy()).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B].copy()).to(device),
            }
            parent_out = net(parent_batch)
            policy_probs = parent_out["policy_probs"].cpu().numpy()

        # ── For each game, pick top-K and evaluate children ───────
        # Collect all children for a single batched forward pass
        child_count = 0
        # Maps: for each parent b, list of (action_index_in_le, action_enc_idx, child_buf_idx, child_current_player)
        child_info: list[list[tuple[int, int, int, int]]] = []

        for b, (idx, le) in enumerate(index_map):
            g = games[idx]
            cg = g._game
            probs = policy_probs[b]

            # Top-K actions by policy probability
            K = min(top_k, len(le))
            # Get top-K from the 337-dim policy (only legal ones matter)
            le_indices = []
            le_probs = []
            for li, act in enumerate(le):
                aidx = action_enc.encode(act)
                le_indices.append((li, aidx))
                le_probs.append(probs[aidx])
            le_probs = np.array(le_probs, dtype=np.float32)
            top_k_le = np.argsort(le_probs)[-K:][::-1]

            children_for_b = []
            for rank in top_k_le:
                li, aidx = le_indices[rank]
                # Copy game, apply action, encode child
                lib.game_copy(ctypes.byref(_exit_ch), ctypes.byref(cg))
                lib.game_execute(
                    ctypes.byref(_exit_ch), le[li],
                    _exit_ca, ctypes.byref(_exit_cn))
                child_cp = _exit_ch.state.current_player_index
                child_view = g.state_view_from_struct(_exit_ch)
                state_enc.encode_into(
                    child_view,
                    child_nf[child_count], child_ef[child_count],
                    child_ff[child_count])
                children_for_b.append((li, aidx, child_count, child_cp))
                child_count += 1

            child_info.append(children_for_b)

        # ── Batch forward pass on ALL child states ────────────────
        if child_count > 0:
            with torch.no_grad():
                child_batch = {
                    "node_features": torch.from_numpy(
                        child_nf[:child_count].copy()).to(device),
                    "edge_index": edge_index_dev,
                    "edge_features": torch.from_numpy(
                        child_ef[:child_count].copy()).to(device),
                    "flat_features": torch.from_numpy(
                        child_ff[:child_count].copy()).to(device),
                    "action_mask": torch.ones(
                        child_count, AD, device=device),
                }
                child_out = net(child_batch)
                # V(s') as 4-player win probabilities
                child_values = F.softmax(
                    child_out["value"], dim=-1).cpu().numpy()

        # ── Pick best child per game, record step, advance ────────
        for b, (idx, le) in enumerate(index_map):
            g = games[idx]
            cp = g.current_player()
            children = child_info[b]

            if not children:
                continue

            # Find action with highest V(s')[parent_current_player].
            # Value head output is rotated so index 0 = child's current_player.
            # Parent's cp maps to index (cp - child_cp) % 4.
            best_li, best_aidx, best_v = 0, 0, -1e30
            for li, aidx, ci, child_cp in children:
                seat_idx = (cp - child_cp) % 4
                v = float(child_values[ci, seat_idx])
                if v > best_v:
                    best_v = v
                    best_li, best_aidx = li, aidx

            # Record training example
            all_steps.append({
                "nf": nf_buf[b].copy(), "ef": ef_buf[b].copy(),
                "ff": ff_buf[b].copy(), "mask": mask_buf[b].copy(),
                "action_idx": best_aidx, "player": cp,
                "game_idx": idx,
            })

            # Step the game: use search pick with some temperature exploration
            if temperature > 0 and random.random() < temperature:
                # Occasionally sample from policy for exploration
                probs = policy_probs[b]
                probs_safe = probs.copy()
                probs_safe[probs_safe < 0] = 0
                s = probs_safe.sum()
                if s > 1e-8:
                    probs_safe /= s
                    sampled_aidx = int(np.random.choice(AD, p=probs_safe))
                    chosen = next((i for i, a in enumerate(le)
                                   if action_enc.encode(a) == sampled_aidx), best_li)
                else:
                    chosen = best_li
            else:
                chosen = best_li

            g.step(chosen)
            total_actions += 1

    # ── Terminal value targets ────────────────────────────────────
    reward_vecs = _assign_rewards(games, num_games)
    for step in all_steps:
        gi = step["game_idx"]
        pid = step["player"]
        rv = reward_vecs[gi]
        rot = np.roll(rv, -pid).copy()
        rsum = rot.sum()
        step["value_target"] = (rot / rsum if rsum > 1e-8
                                else np.ones(4, dtype=np.float32) * 0.25)

    return all_steps


# ── Training step (Expert Iteration) ─────────────────────────────────


def train_step_exit(
    net, optimizer, data: dict[str, torch.Tensor],
    edge_index_dev, device: str,
    batch_size: int, entropy_weight: float = 0.01,
) -> tuple[dict[str, float], int]:
    """Train on ExIt data: cross-entropy toward search-improved actions.

    ``data`` has keys: nf, ef, ff, mask, action_idx, value_target.
    No reward, no hz_flag, no REINFORCE.
    """
    S = data["nf"].shape[0]
    if S < 16:
        return {}, 0

    perm = torch.randperm(S, device=data["nf"].device)
    net.train()
    accum: dict[str, float] = {}
    n_mb = 0

    for start in range(0, S, batch_size):
        idx = perm[start : start + batch_size]
        if len(idx) < 16:
            continue

        nf = data["nf"][idx]
        ef = data["ef"][idx]
        ff = data["ff"][idx]
        mask = data["mask"][idx]
        actions = data["action_idx"][idx]
        vt = data["value_target"][idx]

        optimizer.zero_grad(set_to_none=True)
        out = net({
            "node_features": nf, "edge_index": edge_index_dev,
            "edge_features": ef, "flat_features": ff, "action_mask": mask,
        })

        logits = out["policy_logits"]
        value_logits = out["value"]

        # Policy: cross-entropy toward search-improved action
        policy_loss = F.cross_entropy(logits, actions)
        policy_loss = torch.nan_to_num(policy_loss, nan=0.0)

        # Value: cross-entropy on terminal win distribution
        vt_norm = vt.detach().clamp(min=0.0)
        vt_sum = vt_norm.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vt_dist = vt_norm / vt_sum
        v_log_probs = F.log_softmax(value_logits, dim=-1)
        value_loss = -(vt_dist * v_log_probs).sum(dim=-1).mean()
        value_loss = torch.nan_to_num(value_loss, nan=0.0)

        # Entropy bonus
        log_pi = F.log_softmax(logits, dim=-1)
        pi = log_pi.exp()
        entropy = -(pi * log_pi * mask).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0).mean()

        total = policy_loss + value_loss - entropy_weight * entropy

        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            pacc = (pred == actions).float().mean().item()
            v_probs = F.softmax(value_logits, dim=-1)
            vacc = (v_probs.argmax(-1) == vt_dist.argmax(-1)).float().mean()

        m = {
            "total_loss": total.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "policy_accuracy": pacc,
            "value_accuracy": vacc.item(),
        }
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
        n_mb += 1

    if n_mb > 0:
        return {k: v / n_mb for k, v in accum.items()}, n_mb
    return {}, 0


# ── Training step (REINFORCE, legacy) ────────────────────────────────


def train_step(
    net, optimizer, data: dict[str, torch.Tensor],
    edge_index_dev, device: str,
    batch_size: int, entropy_weight: float,
    ab2_weight: float = 1.0,
) -> tuple[dict[str, float], int]:
    """Run gradient updates on pre-tensorized game data.

    ``data`` is a dict of GPU-resident tensors with keys:
    nf, ef, ff, mask, action_idx, reward, hz_flag, value_target.

    Returns (averaged_metrics, num_mini_batches).
    """
    S = data["nf"].shape[0]
    if S < 16:
        return {}, 0

    perm = torch.randperm(S, device=data["nf"].device)
    net.train()
    accum: dict[str, float] = {}
    n_mb = 0

    for start in range(0, S, batch_size):
        idx = perm[start : start + batch_size]
        if len(idx) < 16:
            continue

        nf = data["nf"][idx]
        ef = data["ef"][idx]
        ff = data["ff"][idx]
        mask = data["mask"][idx]
        actions = data["action_idx"][idx]
        rewards = data["reward"][idx]
        hz_flag = data["hz_flag"][idx]
        vt = data["value_target"][idx]

        optimizer.zero_grad(set_to_none=True)
        out = net({
            "node_features": nf, "edge_index": edge_index_dev,
            "edge_features": ef, "flat_features": ff, "action_mask": mask,
        })

        logits = out["policy_logits"]
        value_logits = out["value"]

        log_pi = F.log_softmax(logits, dim=-1)
        taken_lp = log_pi.gather(1, actions.unsqueeze(1)).squeeze(1)

        v_probs = F.softmax(value_logits, dim=-1)
        v_self = v_probs[:, 0]  # win prob for acting player (seat 0 after rotation)

        # ── HZ policy: REINFORCE with value baseline ──────────────────
        n_hz = hz_flag.sum().clamp(min=1.0)
        raw_adv = (rewards - v_self.detach()) * hz_flag
        # Normalize advantages over HZ moves to zero-mean unit-variance
        hz_mask = hz_flag.bool()
        if hz_mask.any():
            hz_adv = raw_adv[hz_mask]
            adv_mean = hz_adv.mean()
            adv_std = hz_adv.std().clamp(min=1e-6)
            advantage = torch.where(hz_mask, (raw_adv - adv_mean) / adv_std, raw_adv)
        else:
            advantage = raw_adv
        hz_loss = -(taken_lp * advantage).sum() / n_hz
        hz_loss = torch.nan_to_num(hz_loss, nan=0.0)
        # Scale REINFORCE to a gentle nudge -- game-level rewards over ~500
        # moves have ~1:250 signal-to-noise; unscaled it destroys the policy.
        hz_loss = hz_loss * 0.01

        # ── AB2 policy: behavioral cloning (NLL of expert action) ─────
        ab2_flag = 1.0 - hz_flag
        n_ab2 = ab2_flag.sum().clamp(min=1.0)
        ab2_loss = -(taken_lp * ab2_flag).sum() / n_ab2
        ab2_loss = torch.nan_to_num(ab2_loss, nan=0.0)

        policy_loss = hz_loss + ab2_weight * ab2_loss

        # ── Value: cross-entropy with 4-player win distribution ───────
        vt_norm = vt.detach().clamp(min=0.0)
        vt_sum = vt_norm.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vt_dist = vt_norm / vt_sum
        v_log_probs = F.log_softmax(value_logits, dim=-1)
        value_loss = -(vt_dist * v_log_probs).sum(dim=-1).mean()
        value_loss = torch.nan_to_num(value_loss, nan=0.0)

        # ── Entropy bonus ─────────────────────────────────────────────
        pi = F.softmax(logits, dim=-1)
        entropy = -(pi * log_pi * mask).sum(dim=-1)
        entropy = torch.nan_to_num(entropy, nan=0.0).mean()

        total = policy_loss + value_loss - entropy_weight * entropy

        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            vacc = (v_probs.argmax(-1) == vt_dist.argmax(-1)).float().mean()

        m = {
            "total_loss": total.item(),
            "hz_policy_loss": hz_loss.item(),
            "ab2_policy_loss": ab2_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "value_accuracy": vacc.item(),
            "mean_advantage": advantage.abs().mean().item(),
            "ab2_weight": ab2_weight,
        }
        for k, v in m.items():
            accum[k] = accum.get(k, 0.0) + v
        n_mb += 1

    if n_mb > 0:
        return {k: v / n_mb for k, v in accum.items()}, n_mb
    return {}, 0


# ── Evaluation ────────────────────────────────────────────────────────


def eval_batch(
    net, state_enc, action_enc, lib, device, edge_index_dev,
    num_games: int, seed_offset: int, temperature: float = 0.01,
) -> tuple[int, int]:
    """Run concurrent eval games (2 HZ near-greedy + 2 AB2).

    Returns (hz_wins, ab2_wins).
    """
    from hexzero.game.interface import CatanGame

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF, EF, FF = (state_enc.NODE_FEATURE_DIM,
                  state_enc.EDGE_FEATURE_DIM,
                  state_enc.FLAT_FEATURE_DIM)

    nf_buf = np.zeros((num_games + 1, N, NF), dtype=np.float32)
    ef_buf = np.zeros((num_games + 1, E, EF), dtype=np.float32)
    ff_buf = np.zeros((num_games + 1, FF), dtype=np.float32)
    mask_buf = np.zeros((num_games + 1, AD), dtype=np.float32)

    games = [CatanGame(seed=80000 + seed_offset * 1000 + i)
             for i in range(num_games)]
    for g in games:
        g.reset()

    hz_seats: list[set[int]] = []
    ab2_seats: list[set[int]] = []
    for _ in range(num_games):
        s = list(range(4))
        random.shuffle(s)
        hz_seats.append(set(s[:2]))
        ab2_seats.append(set(s[2:]))

    active = list(range(num_games))
    net.eval()

    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 750]
        if not active:
            break

        # AB2 turns
        prog = True
        while prog:
            prog = False
            for idx in active:
                g = games[idx]
                if g.is_terminal() or g.turn_number >= 750:
                    continue
                if g.current_player() not in ab2_seats[idx]:
                    continue
                le = g.get_legal_actions()
                if not le:
                    continue
                g.step(_ab2_pick(lib, g, le))
                prog = True

        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 750]
        if not active:
            break

        # HZ turns (batched, near-greedy)
        B = 0
        imap: list[tuple[int, list]] = []
        for idx in active:
            g = games[idx]
            if g.current_player() not in hz_seats[idx]:
                continue
            le = g.get_legal_actions()
            if not le:
                continue
            state_enc.encode_into(
                g.get_state_view(), nf_buf[B], ef_buf[B], ff_buf[B])
            mask_buf[B] = action_enc.get_action_mask(le).numpy()
            imap.append((idx, le))
            B += 1

        if B == 0:
            continue

        with torch.no_grad():
            batch = {
                "node_features": torch.from_numpy(nf_buf[:B].copy()).to(device),
                "edge_index": edge_index_dev,
                "edge_features": torch.from_numpy(ef_buf[:B].copy()).to(device),
                "flat_features": torch.from_numpy(ff_buf[:B].copy()).to(device),
                "action_mask": torch.from_numpy(mask_buf[:B].copy()).to(device),
            }
            out = net(batch)
            lo = out["policy_logits"] / temperature
            lo = lo.masked_fill(batch["action_mask"] == 0, -1e9)
            pr = F.softmax(lo, dim=-1).cpu().numpy()

        for b, (idx, le) in enumerate(imap):
            p = pr[b]
            if p.sum() < 1e-6:
                p = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
            p = p / p.sum()
            aidx = int(np.random.choice(AD, p=p))
            chosen = next((i for i, a in enumerate(le)
                           if action_enc.encode(a) == aidx), 0)
            games[idx].step(chosen)

    hz_w = ab2_w = 0
    for idx in range(num_games):
        w = games[idx].winner()
        if w is None:
            continue
        if w in hz_seats[idx]:
            hz_w += 1
        else:
            ab2_w += 1
    return hz_w, ab2_w


def eval_vs_model(
    net_current, net_opponent, state_enc, action_enc, device, edge_index_dev,
    num_games: int, seed_offset: int, temperature: float = 0.01,
) -> tuple[int, int]:
    """Play concurrent games: 2 current HZ vs 2 opponent HZ (past checkpoint).

    Both sides use batched GPU inference. Returns (current_wins, opponent_wins).
    """
    from hexzero.game.interface import CatanGame

    AD = 337
    N, E = state_enc.num_nodes, state_enc.num_edges
    NF, EF, FF = (state_enc.NODE_FEATURE_DIM,
                  state_enc.EDGE_FEATURE_DIM,
                  state_enc.FLAT_FEATURE_DIM)

    nf_buf = np.zeros((num_games + 1, N, NF), dtype=np.float32)
    ef_buf = np.zeros((num_games + 1, E, EF), dtype=np.float32)
    ff_buf = np.zeros((num_games + 1, FF), dtype=np.float32)
    mask_buf = np.zeros((num_games + 1, AD), dtype=np.float32)

    games = [CatanGame(seed=70000 + seed_offset * 1000 + i)
             for i in range(num_games)]
    for g in games:
        g.reset()

    cur_seats: list[set[int]] = []
    opp_seats: list[set[int]] = []
    for i in range(num_games):
        cur_seats.append({i % 4, (i + 2) % 4})
        opp_seats.append({(i + 1) % 4, (i + 3) % 4})

    active = list(range(num_games))
    net_current.eval()
    net_opponent.eval()
    total_actions = 0
    max_total_actions = num_games * 2000

    while True:
        active = [i for i in active
                  if not games[i].is_terminal() and games[i].turn_number < 750]
        if not active or total_actions >= max_total_actions:
            break

        for role_net, role_seats in [(net_opponent, opp_seats),
                                     (net_current, cur_seats)]:
            need = []
            for idx in active:
                g = games[idx]
                if g.is_terminal() or g.turn_number >= 750:
                    continue
                if g.current_player() not in role_seats[idx]:
                    continue
                le = g.get_legal_actions()
                if not le:
                    continue
                need.append((idx, le))

            if not need:
                continue

            B = 0
            for idx, le in need:
                state_enc.encode_into(
                    games[idx].get_state_view(), nf_buf[B], ef_buf[B], ff_buf[B])
                mask_buf[B] = action_enc.get_action_mask(le).numpy()
                B += 1

            with torch.no_grad():
                batch = {
                    "node_features": torch.from_numpy(nf_buf[:B].copy()).to(device),
                    "edge_index": edge_index_dev,
                    "edge_features": torch.from_numpy(ef_buf[:B].copy()).to(device),
                    "flat_features": torch.from_numpy(ff_buf[:B].copy()).to(device),
                    "action_mask": torch.from_numpy(mask_buf[:B].copy()).to(device),
                }
                out = role_net(batch)
                lo = out["policy_logits"] / temperature
                lo = lo.masked_fill(batch["action_mask"] == 0, -1e9)
                pr = F.softmax(lo, dim=-1).cpu().numpy()

            for b, (idx, le) in enumerate(need):
                p = pr[b]
                if p.sum() < 1e-6:
                    p = mask_buf[b] / max(mask_buf[b].sum(), 1e-8)
                p = p / p.sum()
                aidx = int(np.random.choice(AD, p=p))
                chosen = next((i for i, a in enumerate(le)
                               if action_enc.encode(a) == aidx), 0)
                games[idx].step(chosen)
                total_actions += 1

    cur_w = opp_w = 0
    for idx in range(num_games):
        w = games[idx].winner()
        if w is None:
            continue
        if w in cur_seats[idx]:
            cur_w += 1
        elif w in opp_seats[idx]:
            opp_w += 1
    return cur_w, opp_w


def tensorize_steps(steps: list[dict], device: str) -> dict[str, torch.Tensor]:
    """Convert list-of-dicts from play_batch into a GPU-resident tensor dict."""
    return {
        "nf": torch.from_numpy(np.stack([s["nf"] for s in steps])).to(device),
        "ef": torch.from_numpy(np.stack([s["ef"] for s in steps])).to(device),
        "ff": torch.from_numpy(np.stack([s["ff"] for s in steps])).to(device),
        "mask": torch.from_numpy(np.stack([s["mask"] for s in steps])).to(device),
        "action_idx": torch.tensor(
            [s["action_idx"] for s in steps], dtype=torch.long, device=device),
        "reward": torch.tensor(
            [s["reward"] for s in steps], dtype=torch.float32, device=device),
        "hz_flag": torch.tensor(
            [1.0 if s["is_hz"] else 0.0 for s in steps],
            dtype=torch.float32, device=device),
        "value_target": torch.from_numpy(
            np.stack([s["value_target"] for s in steps])).to(device),
    }


# ── Checkpoint I/O ────────────────────────────────────────────────────


def _save_checkpoint(net, out_dir: str, step: int, extra: dict) -> None:
    meta = {**extra, "step": step}
    # Atomic write: actors hot-reload latest.pt, so a partial write would crash them
    latest = os.path.join(out_dir, "latest.pt")
    tmp = latest + ".tmp"
    net.save_checkpoint(tmp, metadata=meta)
    os.replace(tmp, latest)
    net.save_checkpoint(
        os.path.join(out_dir, f"step_{step:06d}.pt"), metadata=meta)


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Self-play training (REINFORCE + BC)")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pretrained checkpoint from Phase 1")
    parser.add_argument("--output-dir", type=str, default="selfplay_checkpoints")

    # Game & training
    parser.add_argument("--games-per-step", type=int, default=32,
                        help="Concurrent games played before each gradient step")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Policy temperature for HZ during self-play")
    parser.add_argument("--entropy-weight", type=float, default=0.1,
                        help="Entropy bonus coefficient")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N training steps (0 = run forever)")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate every N training steps")
    parser.add_argument("--eval-games", type=int, default=25)
    parser.add_argument("--eval-temperature", type=float, default=0.01,
                        help="Policy temperature for HZ during evaluation")

    # Infrastructure
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gradient-checkpoint", action="store_true",
                        help="Trade compute for memory in trunk blocks")
    parser.add_argument("--wandb-key", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="hexazero-selfplay")
    parser.add_argument("--no-wandb", action="store_true")

    args = parser.parse_args()

    device = detect_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Late imports (after env setup) ────────────────────────────────
    from hexzero.config import get_default_config
    from hexzero.model.network import HexaZeroNet
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.elo.rating import EloRating, MatchResult
    from hexzero.bindings.lib_loader import load_library

    cfg = get_default_config()
    action_enc = ActionEncoder()
    lib = load_library()

    g = CatanGame(seed=0)
    g.reset()
    state_enc = g.make_state_encoder()
    edge_index_dev = state_enc._edge_index.to(device)

    gpu_name = "cpu"
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_properties(0).name

    # ── Model ─────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    net = HexaZeroNet.load_checkpoint(args.checkpoint, device=device)
    if args.gradient_checkpoint:
        net.gradient_checkpointing = True

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    _save_checkpoint(net, args.output_dir, 0, {})

    print("=" * 64, flush=True)
    print("  Phase 2: Self-play Training", flush=True)
    print(f"  Checkpoint : {args.checkpoint}", flush=True)
    print(f"  Device     : {device} ({gpu_name})", flush=True)
    print(f"  Parameters : {net.num_parameters:,}", flush=True)
    print(f"  Games/step : {args.games_per_step}   BS: {args.batch_size}"
          f"   LR: {args.lr}", flush=True)
    print(f"  Temperature: {args.temperature}   Eval: {args.eval_temperature}"
          f"   Entropy: {args.entropy_weight}", flush=True)
    print(f"  Output     : {args.output_dir}", flush=True)
    print("=" * 64, flush=True)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb and args.wandb_key:
        try:
            import wandb
            os.environ["WANDB_API_KEY"] = args.wandb_key
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"selfplay-{os.uname().nodename}",
                config=vars(args),
                tags=["selfplay", "phase2", device],
            )
            print(f"[wandb] {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] init failed: {e}", flush=True)

    # ── Training loop ─────────────────────────────────────────────────
    global_step = 0
    total_games = 0
    cum_hz_wins = 0
    cum_ab2_wins = 0
    best_eval_wr = -1.0
    seed_base = int(time.time()) % 10_000_000
    t_start = time.time()

    while args.max_steps == 0 or global_step < args.max_steps:
        t0 = time.time()

        # ── Play ──────────────────────────────────────────────────────
        data, hz_w, ab2_w = play_batch(
            net, state_enc, action_enc, lib, device, edge_index_dev,
            args.games_per_step, seed_base + total_games, args.temperature)
        total_games += args.games_per_step
        cum_hz_wins += hz_w
        cum_ab2_wins += ab2_w
        t_play = time.time() - t0

        # ── Train ─────────────────────────────────────────────────────
        t1 = time.time()
        gpu_data = tensorize_steps(data, device)
        metrics, n_mb = train_step(
            net, optimizer, gpu_data, edge_index_dev, device,
            args.batch_size, args.entropy_weight)
        del gpu_data
        t_train = time.time() - t1

        global_step += 1

        decided = hz_w + ab2_w
        wr = hz_w / max(decided, 1)
        elapsed = time.time() - t_start
        gps = total_games / max(elapsed, 0.01)

        print(
            f"[step {global_step:4d}] "
            f"loss={metrics.get('total_loss', 0):.4f} "
            f"hz_pl={metrics.get('hz_policy_loss', 0):.4f} "
            f"ab2_pl={metrics.get('ab2_policy_loss', 0):.4f} "
            f"vloss={metrics.get('value_loss', 0):.4f} "
            f"ent={metrics.get('entropy', 0):.3f} "
            f"vacc={metrics.get('value_accuracy', 0):.3f} "
            f"| HZ={hz_w} AB2={ab2_w} ({wr:.0%}) "
            f"| {len(data)} pos {n_mb} mb "
            f"| play {t_play:.1f}s train {t_train:.1f}s "
            f"| {gps:.1f} g/s",
            flush=True,
        )

        if wandb_run:
            import wandb
            wandb.log({
                "train/total_loss": metrics.get("total_loss", 0),
                "train/hz_policy_loss": metrics.get("hz_policy_loss", 0),
                "train/ab2_policy_loss": metrics.get("ab2_policy_loss", 0),
                "train/value_loss": metrics.get("value_loss", 0),
                "train/entropy": metrics.get("entropy", 0),
                "train/value_accuracy": metrics.get("value_accuracy", 0),
                "train/mean_advantage": metrics.get("mean_advantage", 0),
                "train/positions": len(data),
                "train/hz_wins": hz_w,
                "train/ab2_wins": ab2_w,
                "train/hz_winrate": wr,
                "train/total_games": total_games,
                "train/cum_hz_wr": cum_hz_wins / max(cum_hz_wins + cum_ab2_wins, 1),
                "train/games_per_sec": gps,
                "step": global_step,
            })

        # ── Evaluate ──────────────────────────────────────────────────
        if global_step % args.eval_every == 0:
            print(f"[eval] Running {args.eval_games} games ...", flush=True)
            t_eval = time.time()
            e_hz, e_ab2 = eval_batch(
                net, state_enc, action_enc, lib, device, edge_index_dev,
                args.eval_games, global_step, args.eval_temperature)
            t_eval = time.time() - t_eval

            eval_elo = EloRating(k_factor=32.0)
            eval_elo.register_player("AB2", 1000.0, pinned=True)
            eval_elo.register_player("HexaZero", 1000.0)
            for _ in range(e_hz):
                eval_elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2"], "HexaZero", 0, 0, 0, time.time()))
            for _ in range(e_ab2):
                eval_elo.update_ratings(MatchResult(
                    ["HexaZero", "AB2"], "AB2", 0, 0, 0, time.time()))
            hz_elo = eval_elo.get_rating("HexaZero")

            e_decided = e_hz + e_ab2
            e_wr = e_hz / max(e_decided, 1)

            print(f"[eval] HZ={e_hz} AB2={e_ab2} | "
                  f"WR={e_wr:.1%} | ELO={hz_elo:.0f} | "
                  f"{t_eval:.1f}s", flush=True)

            meta = {"eval_hz": e_hz, "eval_ab2": e_ab2,
                    "eval_wr": e_wr, "eval_elo": hz_elo}
            _save_checkpoint(net, args.output_dir, global_step, meta)

            if e_wr > best_eval_wr:
                best_eval_wr = e_wr
                net.save_checkpoint(
                    os.path.join(args.output_dir, "best.pt"),
                    metadata={**meta, "step": global_step})
                print(f"[eval] *** New best: WR={e_wr:.1%} ELO={hz_elo:.0f}",
                      flush=True)

            if wandb_run:
                import wandb
                wandb.log({
                    "eval/hz_wins": e_hz,
                    "eval/ab2_wins": e_ab2,
                    "eval/hz_winrate": e_wr,
                    "eval/hz_elo": hz_elo,
                    "step": global_step,
                })

    # ── Cleanup ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    cum_total = cum_hz_wins + cum_ab2_wins
    cum_wr = cum_hz_wins / max(cum_total, 1)
    print(f"\n[selfplay] Done: {global_step} steps, {total_games} games, "
          f"{elapsed:.0f}s", flush=True)
    print(f"[selfplay] Cumulative HZ win rate: {cum_wr:.1%} "
          f"({cum_hz_wins}/{cum_total})", flush=True)
    print(f"[selfplay] Best eval WR: {best_eval_wr:.1%}", flush=True)

    if wandb_run:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
