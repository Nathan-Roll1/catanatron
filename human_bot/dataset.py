"""Data pipeline for human Catan game data.

Supports two input formats:

1. **Pre-tensorised shards** (.pt) — identical layout to hexzero's
   ``collect_ab2_games.py`` output: ``{node_features, edge_features,
   flat_features, action_mask, action_idx, player, reward_vec}``.

2. **Colonist.io JSON archives** — raw game JSON from the Catan-data/dataset
   GitHub release (~44 k games).  Converted to shards via ``convert_colonist_games``.
"""

from __future__ import annotations

import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
import torch


# ======================================================================
# Value-target rotation helper
# ======================================================================

def rotate_value_targets_to_cp(
    vt: np.ndarray,
    players: np.ndarray,
    num_players: np.ndarray | int | None = None,
) -> np.ndarray:
    """Rotate absolute-seat value targets so slot 0 = cp, slot i = seat (cp+i) % N.

    This matches the state encoder convention
    ``flat_slot_i = player_state[(cp + i) % num_players]``, so training and
    inference both read/write "reward for seat (cp + i) % N" at output slot ``i``.

    Args:
        vt: absolute-seat value targets, shape ``(S, 4)``.
        players: current player index per row, shape ``(S,)``.
        num_players: per-row player count (shape ``(S,)`` or scalar), or None
            to default to 4-player rotation. When ``num_players[i] < 4``, slots
            ``>= num_players[i]`` in the output are zeroed.

    Returns:
        Rotated value targets, shape ``(S, 4)``.
    """
    vt = np.asarray(vt)
    players = np.asarray(players, dtype=np.int64)
    S = vt.shape[0]

    if num_players is None:
        # Legacy 4-player rotation (fixed sign: +players, not -players, so
        # slot 0 truly = cp's reward).
        shifts = (players % 4).astype(np.int64)
        idx_arr = (np.arange(4)[None, :] + shifts[:, None]) % 4
        return np.take_along_axis(vt, idx_arr, axis=1)

    if np.isscalar(num_players):
        num_players = np.full(S, int(num_players), dtype=np.int64)
    else:
        num_players = np.asarray(num_players, dtype=np.int64)

    out = np.zeros_like(vt)
    # Vectorised per-num_players grouping
    for n_p in np.unique(num_players):
        rows = num_players == n_p
        if not rows.any():
            continue
        cp_rows = players[rows] % n_p
        # For slot j in [0, n_p): out[i, j] = vt[i, (cp_rows[i] + j) % n_p]
        j_idx = np.arange(n_p)[None, :]  # (1, n_p)
        src = (cp_rows[:, None] + j_idx) % n_p  # (num_rows, n_p)
        # vt has 4 columns; for j in [0, n_p), take from src; for j >= n_p leave 0
        out_rows = np.zeros((rows.sum(), 4), dtype=vt.dtype)
        out_rows[:, :n_p] = np.take_along_axis(vt[rows], src, axis=1)
        out[rows] = out_rows
    return out


# ======================================================================
# In-memory dataset backed by concatenated tensors
# ======================================================================

def fix_action_masks(ff: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out action slots the player can't afford based on resources and dev cards.

    Skips examples during initial build phase (free placement).

    flat_features layout (current player = slot 0):
      1-5: resources/19 [wood, brick, sheep, wheat, ore]
      6-10: dev cards in hand /14 [knight, yop, monopoly, road_building, vp]
      19: has_played_dev
      109: is_initial_build_phase
    """
    from hexzero.encoder.action_encoder import _MARITIME_PAIRS

    is_initial = ff[:, 109] > 0.5
    is_main = ~is_initial

    res = (ff[:, 1:6] * 19).round()
    wood, brick, sheep, wheat, ore = res[:, 0], res[:, 1], res[:, 2], res[:, 3], res[:, 4]
    res_by_idx = [wood, brick, sheep, wheat, ore]

    dev_cards = (ff[:, 6:11] * 14).round()
    has_played_dev = ff[:, 19] > 0.5

    mask = mask.clone()

    # Only apply resource gating to main-game examples (not initial placement)
    cant_sett = is_main & ~((wood >= 1) & (brick >= 1) & (sheep >= 1) & (wheat >= 1))
    cant_city = is_main & ~((ore >= 3) & (wheat >= 2))
    cant_road = is_main & ~((wood >= 1) & (brick >= 1))
    cant_dev = is_main & ~((ore >= 1) & (sheep >= 1) & (wheat >= 1))

    mask[cant_sett, 5:59] = 0
    mask[cant_city, 59:113] = 0
    mask[cant_road, 113:185] = 0
    mask[cant_dev, 2:3] = 0

    for i, (give_res, _recv) in enumerate(_MARITIME_PAIRS):
        idx = 310 + i
        if idx < mask.shape[1]:
            mask[is_main & (res_by_idx[give_res] < 2), idx] = 0

    for block_base, min_give in [(337, 1), (357, 1), (377, 2)]:
        for i in range(20):
            idx = block_base + i
            if idx < mask.shape[1]:
                give_res = i // 4
                mask[is_main & (res_by_idx[give_res] < min_give), idx] = 0

    no_knight = is_main & ((dev_cards[:, 0] < 1) | has_played_dev)
    no_rb = is_main & ((dev_cards[:, 3] < 1) | has_played_dev)
    no_yop = is_main & ((dev_cards[:, 1] < 1) | has_played_dev)
    no_mono = is_main & ((dev_cards[:, 2] < 1) | has_played_dev)

    mask[no_knight, 3:4] = 0
    mask[no_rb, 4:5] = 0
    mask[no_yop, 285:305] = 0
    mask[no_mono, 305:310] = 0

    return mask


class HumanGameDataset:
    """Memory-mapped tensor dataset for training."""

    def __init__(
        self,
        nf: torch.Tensor,
        ef: torch.Tensor,
        ff: torch.Tensor,
        mask: torch.Tensor,
        action_idx: torch.Tensor,
        value_target: torch.Tensor,
    ) -> None:
        self.nf = nf
        self.ef = ef
        self.ff = ff
        self.mask = mask
        self.action_idx = action_idx
        self.value_target = value_target
        self.n = nf.shape[0]

    def __len__(self) -> int:
        return self.n

    def get_batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "nf": self.nf[indices],
            "ef": self.ef[indices],
            "ff": self.ff[indices],
            "mask": self.mask[indices],
            "action_idx": self.action_idx[indices],
            "value_target": self.value_target[indices],
        }

    def split(
        self, val_fraction: float = 0.10, seed: int = 42,
    ) -> tuple[HumanGameDataset, HumanGameDataset]:
        """Split into train / validation by shuffled index."""
        rng = np.random.RandomState(seed)
        perm = rng.permutation(self.n)
        n_val = max(1, int(self.n * val_fraction))
        val_idx = torch.from_numpy(perm[:n_val].copy()).long()
        trn_idx = torch.from_numpy(perm[n_val:].copy()).long()
        return (
            HumanGameDataset(
                self.nf[trn_idx], self.ef[trn_idx], self.ff[trn_idx],
                self.mask[trn_idx], self.action_idx[trn_idx],
                self.value_target[trn_idx],
            ),
            HumanGameDataset(
                self.nf[val_idx], self.ef[val_idx], self.ff[val_idx],
                self.mask[val_idx], self.action_idx[val_idx],
                self.value_target[val_idx],
            ),
        )


# ======================================================================
# Load pre-tensorised .pt shards (same format as supervised_train.py)
# ======================================================================

def load_tensor_shards(data_dir: str, max_examples: int = 0) -> HumanGameDataset:
    """Load .pt shard files and return a single HumanGameDataset."""
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".pt") and f != "metadata.pt"
    )
    if not files:
        raise FileNotFoundError(f"No .pt shard files in {data_dir}")

    all_nf, all_ef, all_ff, all_mask = [], [], [], []
    all_act, all_vt = [], []
    total = 0

    for fi, fname in enumerate(files):
        data = torch.load(
            os.path.join(data_dir, fname), weights_only=False, map_location="cpu",
        )

        players = data["player"].numpy()
        reward_vecs = data["reward_vec"].numpy()
        S = players.shape[0]
        winners = reward_vecs.argmax(axis=1)
        vt = np.zeros((S, 4), dtype=np.float32)
        vt[np.arange(S), winners] = 1.0
        bad = reward_vecs.max(axis=1) < 1e-8
        vt[bad] = 0.25
        n_p_tensor = data.get("num_players")
        vt = rotate_value_targets_to_cp(
            vt, players,
            n_p_tensor.numpy() if n_p_tensor is not None else None)

        all_nf.append(data["node_features"])
        all_ef.append(data["edge_features"])
        all_ff.append(data["flat_features"])
        # Pad mask to 397 if needed (AB2 data has 337, human data has 397)
        mask = data["action_mask"]
        if mask.shape[-1] < 397:
            pad = torch.zeros(mask.shape[0], 397 - mask.shape[-1], dtype=mask.dtype)
            mask = torch.cat([mask, pad], dim=-1)
        all_mask.append(mask)
        all_act.append(data["action_idx"])
        all_vt.append(torch.from_numpy(vt))
        total += S

        if (fi + 1) % 20 == 0 or fi + 1 == len(files):
            print(f"  {fi + 1}/{len(files)} files  ->  {total:,} examples", flush=True)

        if max_examples > 0 and total >= max_examples:
            break

    all_ff_t = torch.cat(all_ff)
    all_mask_t = torch.cat(all_mask)
    all_act_t = torch.cat(all_act)
    all_mask_t = fix_action_masks(all_ff_t, all_mask_t)
    # Ensure the ground-truth action is always marked legal
    all_mask_t[torch.arange(len(all_act_t)), all_act_t] = 1.0

    ds = HumanGameDataset(
        nf=torch.cat(all_nf),
        ef=torch.cat(all_ef),
        ff=all_ff_t,
        mask=all_mask_t,
        action_idx=all_act_t,
        value_target=torch.cat(all_vt),
    )
    if max_examples > 0 and ds.n > max_examples:
        idx = torch.arange(max_examples)
        return HumanGameDataset(
            ds.nf[idx], ds.ef[idx], ds.ff[idx],
            ds.mask[idx], ds.action_idx[idx], ds.value_target[idx],
        )
    return ds


# ======================================================================
# Colonist.io JSON -> tensor shards
# ======================================================================

# Colonist.io resource enum (1-indexed) -> hexzero resource index (0-indexed)
_COLONIST_RES = {1: 2, 2: 3, 3: 4, 4: 1, 5: 0}  # Brick=2, Wool=3, Grain=4, Ore=1, Lumber=0

# Colonist.io event types we can map to hexzero actions
_EVT_ROLL = "roll"
_EVT_BUILD_SETTLEMENT = "buildSettlement"
_EVT_BUILD_CITY = "buildCity"
_EVT_BUILD_ROAD = "buildRoad"
_EVT_BUY_DEV = "buyDevelopmentCard"
_EVT_PLAY_KNIGHT = "playKnight"
_EVT_MOVE_ROBBER = "moveRobber"
_EVT_MARITIME_TRADE = "maritimeTrade"
_EVT_END_TURN = "endTurn"
_EVT_DISCARD = "discard"
_EVT_MONOPOLY = "playMonopoly"
_EVT_YEAR_OF_PLENTY = "playYearOfPlenty"
_EVT_ROAD_BUILDING = "playRoadBuilding"


def convert_colonist_games(
    archive_dir: str,
    output_dir: str,
    games_per_shard: int = 200,
    min_turns: int = 20,
    max_turns: int = 500,
    require_four_players: bool = True,
    verbose: bool = True,
) -> dict[str, int]:
    """Convert raw Colonist.io JSON game files into hexzero-compatible .pt shards.

    Each JSON file represents one complete game with full event history.
    We replay each game through the hexzero C engine, recording the state
    tensor at each human decision point along with the action taken.

    Returns a summary dict with conversion statistics.
    """
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    os.makedirs(output_dir, exist_ok=True)
    ae = ActionEncoder()

    json_files = sorted(
        p for p in Path(archive_dir).rglob("*.json")
        if p.stat().st_size > 100
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON game files found in {archive_dir}")

    stats = {"total_files": len(json_files), "converted": 0, "skipped": 0, "errors": 0}
    shard_steps: list[dict] = []
    shard_idx = 0

    for fi, jpath in enumerate(json_files):
        try:
            with open(jpath) as f:
                game_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            stats["errors"] += 1
            continue

        events = game_data.get("eventHistory") or game_data.get("events")
        if not events:
            stats["skipped"] += 1
            continue

        n_players = len(game_data.get("playerUserStates", game_data.get("players", [])))
        if require_four_players and n_players != 4:
            stats["skipped"] += 1
            continue

        winner_idx = _extract_winner(game_data)
        if winner_idx is None:
            stats["skipped"] += 1
            continue

        reward_vec = np.zeros(4, dtype=np.float32)
        reward_vec[winner_idx] = 1.0

        try:
            steps = _replay_game_events(events, game_data, ae, reward_vec)
        except Exception:
            if verbose:
                traceback.print_exc()
            stats["errors"] += 1
            continue

        if len(steps) < min_turns or len(steps) > max_turns * 4:
            stats["skipped"] += 1
            continue

        shard_steps.extend(steps)
        stats["converted"] += 1

        if len(shard_steps) >= games_per_shard * 80:
            _save_shard(shard_steps, output_dir, shard_idx)
            shard_idx += 1
            shard_steps = []

        if verbose and (fi + 1) % 500 == 0:
            print(
                f"  [{fi + 1}/{len(json_files)}] converted={stats['converted']}  "
                f"skipped={stats['skipped']}  errors={stats['errors']}",
                flush=True,
            )

    if shard_steps:
        _save_shard(shard_steps, output_dir, shard_idx)

    stats["total_shards"] = shard_idx + (1 if shard_steps else 0)
    if verbose:
        print(f"Conversion complete: {stats}", flush=True)
    return stats


def _extract_winner(game_data: dict) -> int | None:
    """Determine which seat index (0-3) won the game."""
    end_state = game_data.get("endGameState") or game_data.get("winner")
    if isinstance(end_state, dict):
        vps = end_state.get("vpBreakdown") or end_state.get("victoryPoints")
        if isinstance(vps, list):
            return int(np.argmax([sum(v.values()) if isinstance(v, dict) else v for v in vps]))
        winner_idx = end_state.get("winnerIndex") or end_state.get("winner")
        if winner_idx is not None:
            return int(winner_idx)
    if isinstance(end_state, int):
        return end_state
    players = game_data.get("playerUserStates", game_data.get("players", []))
    for i, p in enumerate(players):
        if isinstance(p, dict) and p.get("isWinner"):
            return i
    return None


def _replay_game_events(
    events: list[dict],
    game_data: dict,
    ae: "ActionEncoder",
    reward_vec: np.ndarray,
) -> list[dict]:
    """Replay Colonist.io events through the hexzero C engine.

    For each human decision point, records the encoded state tensors,
    action mask, chosen action index, acting player, and outcome.

    This is the core conversion function.  It instantiates a fresh
    CatanGame with a fixed seed (we cannot reconstruct the original
    board, so we use seed=0 — the board layout will differ but the
    *action encoding* procedure is exercised correctly).

    NOTE: Full replay fidelity requires matching the Colonist board
    layout to the C engine board.  Until a board-layout injector is
    written, this function records training-format tensors from a
    *canonical* board while preserving the action-type distribution
    and relative player decisions.  This is sufficient for pre-training
    the policy head on action-type patterns and bootstrapping the
    value head.
    """
    from hexzero.game.interface import CatanGame

    game = CatanGame(seed=hash(json.dumps(game_data.get("gameSettings", {}))) & 0xFFFFFFFF)
    game.reset()
    se = game.make_state_encoder()

    steps: list[dict] = []
    N = se.num_nodes
    E = se.num_edges

    for event in events:
        if game.is_terminal():
            break

        if not isinstance(event, dict):
            continue

        evt_type = event.get("type") or event.get("eventType", "")
        player_idx = event.get("playerIndex", event.get("player", 0))
        if not isinstance(player_idx, int):
            player_idx = 0

        legal = game.get_legal_actions()
        if not legal:
            break

        cp = game.current_player()

        action_index = _match_event_to_action(evt_type, event, legal, ae, game)
        if action_index is None:
            if len(legal) == 1:
                action_index = 0
            else:
                continue

        sv = game.get_state_view()
        nf, ef, flat = se._encode_numpy(sv)
        mask = ae.get_action_mask(legal).numpy()
        act_idx = ae.encode(legal[action_index])

        steps.append({
            "nf": nf.copy(),
            "ef": ef.copy(),
            "ff": flat.copy(),
            "mask": mask.copy(),
            "action_idx": act_idx,
            "player": cp,
            "reward_vec": reward_vec.copy(),
        })

        game.step(action_index)

    return steps


def _match_event_to_action(
    evt_type: str,
    event: dict,
    legal: list,
    ae: "ActionEncoder",
    game: "CatanGame",
) -> int | None:
    """Match a Colonist.io event to a legal action index in the current game state."""
    from hexzero.encoder.action_encoder import (
        AT_ROLL, AT_END_TURN, AT_BUY_DEVELOPMENT_CARD,
        AT_PLAY_KNIGHT_CARD, AT_PLAY_ROAD_BUILDING,
        AT_BUILD_SETTLEMENT, AT_BUILD_CITY, AT_BUILD_ROAD,
        AT_DISCARD_RESOURCE, AT_PLAY_MONOPOLY,
        AT_PLAY_YEAR_OF_PLENTY, AT_MARITIME_TRADE,
        AT_MOVE_ROBBER,
    )

    target_type = {
        _EVT_ROLL: AT_ROLL,
        _EVT_END_TURN: AT_END_TURN,
        _EVT_BUY_DEV: AT_BUY_DEVELOPMENT_CARD,
        _EVT_PLAY_KNIGHT: AT_PLAY_KNIGHT_CARD,
        _EVT_ROAD_BUILDING: AT_PLAY_ROAD_BUILDING,
        _EVT_BUILD_SETTLEMENT: AT_BUILD_SETTLEMENT,
        _EVT_BUILD_CITY: AT_BUILD_CITY,
        _EVT_BUILD_ROAD: AT_BUILD_ROAD,
        _EVT_DISCARD: AT_DISCARD_RESOURCE,
        _EVT_MONOPOLY: AT_PLAY_MONOPOLY,
        _EVT_YEAR_OF_PLENTY: AT_PLAY_YEAR_OF_PLENTY,
        _EVT_MARITIME_TRADE: AT_MARITIME_TRADE,
        _EVT_MOVE_ROBBER: AT_MOVE_ROBBER,
    }.get(evt_type)

    if target_type is None:
        return None

    if target_type == AT_MOVE_ROBBER:
        # Prefer steal variants (value[3] >= 0) over no-steal (value[3] == -1).
        # The C engine lists no-steal first per tile, so a naive first-match
        # always picks no-steal. Humans steal ~90%+ of the time.
        steal_idx = None
        no_steal_idx = None
        for i, act in enumerate(legal):
            if act.type == AT_MOVE_ROBBER:
                if act.value[3] >= 0:
                    if steal_idx is None:
                        steal_idx = i
                else:
                    if no_steal_idx is None:
                        no_steal_idx = i
        return steal_idx if steal_idx is not None else no_steal_idx

    for i, act in enumerate(legal):
        if act.type == target_type:
            return i

    return None


def _save_shard(steps: list[dict], output_dir: str, shard_idx: int) -> None:
    """Stack step dicts into tensors and write a .pt shard."""
    S = len(steps)
    if S == 0:
        return

    nf_shape = steps[0]["nf"].shape
    ef_shape = steps[0]["ef"].shape
    ff_len = steps[0]["ff"].shape[0]

    nf = np.zeros((S, *nf_shape), dtype=np.float32)
    ef = np.zeros((S, *ef_shape), dtype=np.float32)
    ff = np.zeros((S, ff_len), dtype=np.float32)
    mask = np.zeros((S, 337), dtype=np.float32)
    action_idx = np.zeros(S, dtype=np.int64)
    player = np.zeros(S, dtype=np.int64)
    reward_vec = np.zeros((S, 4), dtype=np.float32)

    for i, step in enumerate(steps):
        nf[i] = step["nf"]
        ef[i] = step["ef"]
        ff[i] = step["ff"]
        mask[i] = step["mask"]
        action_idx[i] = step["action_idx"]
        player[i] = step["player"]
        reward_vec[i] = step["reward_vec"]

    data = {
        "node_features": torch.from_numpy(nf),
        "edge_features": torch.from_numpy(ef),
        "flat_features": torch.from_numpy(ff),
        "action_mask": torch.from_numpy(mask),
        "action_idx": torch.from_numpy(action_idx),
        "player": torch.from_numpy(player),
        "reward_vec": torch.from_numpy(reward_vec),
    }

    path = os.path.join(output_dir, f"shard_{shard_idx:05d}.pt")
    torch.save(data, path)
    print(f"  Saved {path}  ({S:,} steps)", flush=True)
