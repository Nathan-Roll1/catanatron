#!/usr/bin/env python3
"""Convert Colonist.io JSON game archives to hexzero training tensors.

v3: Pure-Python state encoder. Builds tensors directly from accumulated
Colonist game state (no C engine replay, no dice divergence).

Usage:
    python -m human_bot.colonist_converter \
        --input-dir data/colonist_raw/games \
        --output-dir data/human_games \
        --num-workers 14
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# ======================================================================
# Coordinate mapping tables (verified via permutation search + bipartite matching)
# ======================================================================

COL_CORNER_TO_NODE: dict[int, int] = {
    0: 47, 1: 48, 2: 21, 3: 19, 4: 44, 5: 45, 6: 18, 7: 16, 8: 42,
    9: 43, 10: 39, 11: 17, 13: 40, 14: 4, 15: 37, 16: 14, 18: 12,
    19: 35, 20: 34, 22: 13, 23: 10, 24: 33, 25: 32, 26: 11, 27: 28,
    28: 31, 29: 30, 30: 9, 31: 26, 32: 29, 33: 8, 34: 7, 36: 27,
    37: 24, 38: 23, 40: 1, 41: 52, 42: 49, 44: 20, 45: 50, 46: 46,
    47: 22, 48: 5, 49: 0, 50: 15, 51: 2, 52: 3, 53: 6,
}

COL_EDGE_TO_ACTION_IDX: dict[int, int] = {
    0: 64, 1: 34, 2: 1, 3: 35, 4: 36, 5: 38, 6: 2, 7: 9, 8: 29,
    9: 30, 10: 33, 11: 10, 12: 26, 13: 31, 14: 32, 15: 59, 16: 11,
    17: 7, 18: 24, 19: 27, 20: 28, 21: 8, 22: 21, 23: 52, 24: 25,
    25: 53, 26: 5, 27: 19, 28: 50, 29: 22, 30: 23, 31: 16, 32: 47,
    33: 49, 34: 20, 35: 48, 36: 14, 37: 45, 39: 17, 40: 18, 43: 46,
    44: 15, 48: 12, 49: 13, 50: 41, 53: 39, 54: 40, 55: 68, 56: 67,
    57: 0, 58: 37, 59: 66, 60: 4, 61: 3, 66: 6,
}

COL_RES_TO_ENGINE: dict[int, int] = {0: -1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 0}
COL_CARD_ENUM_TO_RES: dict[int, int] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 0}

NUM_NODES = 54
NUM_EDGES_DIRECTED = 144
NUM_RESOURCES = 5
NUM_PLAYERS = 4
ACTION_SPACE = 397  # 337 base + 60 trade offers (1:1=20, 1:2=20, 2:1=20)

_DICE_PROB = np.zeros(13, dtype=np.float32)
for _a in range(1, 7):
    for _b in range(1, 7):
        _DICE_PROB[_a + _b] += 1.0 / 36.0


# ======================================================================
# One-time topology precomputation (same for all games)
# ======================================================================

def _precompute_topology() -> dict:
    """Precompute static board topology from the C engine.

    Returns a dict with edge_index, tile data, and port info that the
    Python encoder needs.  Called once per worker process.
    """
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.state_encoder import StateEncoder
    from hexzero.encoder.action_encoder import ActionEncoder, _BASE_TOPO_COORDS

    g = CatanGame(seed=0)
    g.reset()
    se = g.make_state_encoder()
    ae = ActionEncoder()
    cmap = g._map_obj

    edge_index = se._edge_index.numpy().copy()  # (2, 144)

    # tile cube coords for robber matching
    tile_cubes = []
    for i in range(19):
        c = cmap.land_tile_coords[i]
        tile_cubes.append((c.x, c.y, c.z))

    # tile nodes in compact space (19, 6) — for production features
    ltiles = se._ltiles.copy()

    # Colonist hex (hx,hy) → C engine tile index
    col_hex_to_c_tile = {}
    for i, cube in enumerate(tile_cubes):
        col_hex = (cube[0], cube[2])  # cube (x, -(x+y), y) → Colonist (x, y=cube_z)
        col_hex_to_c_tile[col_hex] = i

    # Build directed edge lookup: (compact_src, compact_dst) → directed_edge_idx
    directed_lookup = {}
    for i in range(NUM_EDGES_DIRECTED):
        s, d = int(edge_index[0, i]), int(edge_index[1, i])
        directed_lookup[(s, d)] = i

    # Build Colonist edge → directed edge indices
    # Use the corner mapping to convert endpoint pairs
    node_to_col = {v: k for k, v in COL_CORNER_TO_NODE.items()}
    col_edge_to_directed = {}
    for col_eid, action_idx in COL_EDGE_TO_ACTION_IDX.items():
        edge_pair = ae._idx_to_edge[action_idx]
        na, nb = int(edge_pair[0]), int(edge_pair[1])
        d1 = directed_lookup.get((na, nb))
        d2 = directed_lookup.get((nb, na))
        if d1 is not None and d2 is not None:
            col_edge_to_directed[col_eid] = (d1, d2)

    return {
        "edge_index": edge_index,
        "ltiles": ltiles,
        "tile_cubes": tile_cubes,
        "col_hex_to_c_tile": col_hex_to_c_tile,
        "directed_lookup": directed_lookup,
        "col_edge_to_directed": col_edge_to_directed,
    }


# ======================================================================
# Python state encoder: build tensors from Colonist JSON state
# ======================================================================

def _encode_state(
    state: dict,
    topo: dict,
    current_player_color: int,
    play_order: list[int],
    production: np.ndarray,
    port_oh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (54,18) node, (144,5) edge, (115,) flat tensors from Colonist state."""

    nf = np.zeros((NUM_NODES, 18), dtype=np.float32)
    ef = np.zeros((NUM_EDGES_DIRECTED, 5), dtype=np.float32)
    flat = np.zeros(115, dtype=np.float32)

    color_to_seat = {c: i for i, c in enumerate(play_order)}
    cp_seat = color_to_seat.get(current_player_color, 0)

    # ── Node features ──────────────────────────────────────────────
    tcs = state.get("mapState", {}).get("tileCornerStates", {})
    occupied = set()
    for cid_str, cdata in tcs.items():
        if not isinstance(cdata, dict):
            continue
        owner = cdata.get("owner")
        btype = cdata.get("buildingType")
        if owner is None or btype is None:
            continue
        node = COL_CORNER_TO_NODE.get(int(cid_str))
        if node is None:
            continue
        occupied.add(node)
        owner_seat = color_to_seat.get(owner, 0)
        is_own = int(owner_seat == cp_seat)
        is_foe = int(owner_seat != cp_seat)
        is_sett = int(btype == 1)
        is_city = int(btype == 2)
        nf[node, 1] = is_own * is_sett
        nf[node, 2] = is_own * is_city
        nf[node, 3] = is_foe * is_sett
        nf[node, 4] = is_foe * is_city

    for n in range(NUM_NODES):
        if n not in occupied:
            nf[n, 0] = 1.0

    nf[:, 5:10] = production
    nf[:, 10:17] = port_oh

    robber_state = state.get("mechanicRobberState", {})
    robber_tile = robber_state.get("locationTileIndex")
    if robber_tile is not None:
        c_tile = topo["col_hex_to_c_tile"].get(
            _col_tile_to_hex(robber_tile, state), None
        )
        if c_tile is not None and c_tile < len(topo["ltiles"]):
            nf[topo["ltiles"][c_tile], 17] = 1.0

    # ── Edge features ──────────────────────────────────────────────
    ef[:, 0] = 1.0  # default: no road

    tes = state.get("mapState", {}).get("tileEdgeStates", {})
    for eid_str, edata in tes.items():
        if not isinstance(edata, dict):
            continue
        if edata.get("type") != 1:
            continue
        owner = edata.get("owner")
        if owner is None:
            continue
        dirs = topo["col_edge_to_directed"].get(int(eid_str))
        if dirs is None:
            continue
        owner_seat = color_to_seat.get(owner, 0)
        rel = (owner_seat - cp_seat) % NUM_PLAYERS
        for di in dirs:
            ef[di, 0] = 0.0
            if rel == 0:
                ef[di, 1] = 1.0
            else:
                ef[di, 1 + rel] = 1.0

    # ── Flat features (115) ────────────────────────────────────────
    ps = state.get("playerStates", {})
    rot = [(cp_seat + i) % NUM_PLAYERS for i in range(NUM_PLAYERS)]
    rot_colors = [play_order[s] for s in rot]

    o = 0
    for pi, pc in enumerate(rot_colors):
        pdata = ps.get(str(pc), {})
        if not isinstance(pdata, dict):
            o += 24
            continue

        vps = pdata.get("victoryPointsState", {})
        total_vp = sum(int(v) for v in vps.values()) if isinstance(vps, dict) else 0
        flat[o] = total_vp / 10.0

        rc = pdata.get("resourceCards", {})
        cards = rc.get("cards", []) if isinstance(rc, dict) else []
        res_counts = [0] * 5
        for c in cards:
            r = COL_CARD_ENUM_TO_RES.get(c)
            if r is not None:
                res_counts[r] += 1
        for r in range(5):
            flat[o + 1 + r] = res_counts[r] / 19.0

        dcs = state.get("mechanicDevelopmentCardsState", {}).get("players", {}).get(str(pc), {})
        dev_hand = dcs.get("developmentCards", {}).get("cards", []) if isinstance(dcs, dict) else []
        dev_used = dcs.get("developmentCardsUsed", []) if isinstance(dcs, dict) else []
        dev_counts = [0] * 5  # knight, yop, monopoly, road_building, vp
        for c in dev_hand:
            if c in (10, 11):
                dev_counts[0] += 1
            elif c == 15:
                dev_counts[1] += 1
            elif c in (12, 13):
                dev_counts[2] += 1
            elif c == 14:
                dev_counts[3] += 1
        for r in range(5):
            flat[o + 6 + r] = dev_counts[r] / 14.0

        played_counts = [0] * 5
        for c in dev_used:
            if c in (10, 11):
                played_counts[0] += 1
            elif c == 15:
                played_counts[1] += 1
            elif c in (12, 13):
                played_counts[2] += 1
            elif c == 14:
                played_counts[3] += 1
        for r in range(5):
            flat[o + 11 + r] = played_counts[r]

        # has_road, has_army
        lr = state.get("mechanicLongestRoadState", {}).get(str(pc), {})
        la = state.get("mechanicLargestArmyState", {}).get(str(pc), {})
        flat[o + 16] = 1.0 if (isinstance(lr, dict) and lr.get("hasLongestRoad")) else 0.0
        flat[o + 17] = 1.0 if (isinstance(la, dict) and la.get("hasLargestArmy")) else 0.0

        # has_rolled (approximate from diceState)
        ds = state.get("diceState", {})
        cs_data = state.get("currentState", {})
        is_current = (cs_data.get("currentTurnPlayerColor") == pc)
        flat[o + 18] = 1.0 if (is_current and ds.get("diceThrown")) else 0.0

        has_played_dev = dcs.get("hasUsedDevelopmentCardThisTurn", False) if isinstance(dcs, dict) else False
        flat[o + 19] = 1.0 if has_played_dev else 0.0

        # pieces available
        road_state = state.get("mechanicRoadState", {}).get(str(pc), {})
        sett_state = state.get("mechanicSettlementState", {}).get(str(pc), {})
        city_state = state.get("mechanicCityState", {}).get(str(pc), {})
        flat[o + 20] = (road_state.get("bankRoadAmount", 15) if isinstance(road_state, dict) else 15) / 15.0
        flat[o + 21] = (sett_state.get("bankSettlementAmount", 5) if isinstance(sett_state, dict) else 5) / 5.0
        flat[o + 22] = (city_state.get("bankCityAmount", 4) if isinstance(city_state, dict) else 4) / 4.0

        # longest road length
        flat[o + 23] = (lr.get("longestRoad", 0) if isinstance(lr, dict) else 0) / 15.0
        o += 24

    # Bank
    o1 = NUM_PLAYERS * 24  # 96
    bs = state.get("bankState", {}).get("resourceCards", {})
    for res_enum in range(1, 6):
        r = COL_CARD_ENUM_TO_RES.get(res_enum, 0)
        flat[o1 + r] = bs.get(str(res_enum), 19) / 19.0

    dev_bank = state.get("mechanicDevelopmentCardsState", {}).get("bankDevelopmentCards", {})
    dev_bank_cards = dev_bank.get("cards", []) if isinstance(dev_bank, dict) else []
    flat[o1 + 5] = len(dev_bank_cards) / 25.0

    # Phase
    o2 = o1 + 6  # 102
    cs_data = state.get("currentState", {})
    turn_count = cs_data.get("completedTurns", 0)
    flat[o2 + 12] = turn_count / 1000.0

    return nf, ef, flat


def _col_tile_to_hex(col_tile_idx: int, state: dict) -> tuple[int, int] | None:
    """Get Colonist hex (x, y) for a tile index from the mapState."""
    hexes = state.get("mapState", {}).get("tileHexStates", {})
    h = hexes.get(str(col_tile_idx))
    if h is None:
        return None
    return (h["x"], h["y"])


def _compute_production(state: dict, topo: dict) -> np.ndarray:
    """Compute per-node production features (54, 5) from hex tile data."""
    prod = np.zeros((NUM_NODES, NUM_RESOURCES), dtype=np.float32)
    hexes = state.get("mapState", {}).get("tileHexStates", {})
    for hi_str, hdata in hexes.items():
        hx, hy = hdata["x"], hdata["y"]
        res = COL_RES_TO_ENGINE.get(hdata.get("type", 0), -1)
        num = hdata.get("diceNumber", 0)
        if res < 0 or num <= 0 or num > 12:
            continue
        c_tile = topo["col_hex_to_c_tile"].get((hx, hy))
        if c_tile is None:
            continue
        prob = _DICE_PROB[num]
        nodes = topo["ltiles"][c_tile]
        for n in nodes:
            prod[n, res] += prob
    return prod


def _compute_port_oh(state: dict) -> np.ndarray:
    """Compute per-node port one-hot (54, 7) from Colonist port data."""
    port_oh = np.zeros((NUM_NODES, 7), dtype=np.float32)
    port_oh[:, 0] = 1.0  # default: no port

    ps = state.get("mapState", {}).get("portEdgeStates", {})
    if not ps:
        return port_oh

    # Colonist port type → feature index: 1=brick→2, 2=wool→3, 3=grain→4, 4=ore→5, 5=lumber→1, 6=generic→6
    port_type_map = {1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 6}

    for _, pdata in ps.items():
        if not isinstance(pdata, dict):
            continue
        ptype = pdata.get("type")
        feat_idx = port_type_map.get(ptype)
        if feat_idx is None:
            continue
        # Port is at an edge position. Find the two corners.
        # Use the Colonist playerStates.bankTradeRatiosState to determine
        # which corners have ports (when a player builds there, their ratio changes).
        # For now, skip per-node port assignment — use default "no port".
        # TODO: map port positions via edge endpoints

    return port_oh


# ======================================================================
# Event parsing (same as before)
# ======================================================================

def _parse_event(event: dict) -> list[dict]:
    sc = event.get("stateChange", {})
    gls = sc.get("gameLogState", {})
    ms = sc.get("mapState", {})
    tcs = ms.get("tileCornerStates", {})
    tes = ms.get("tileEdgeStates", {})
    robber = sc.get("mechanicRobberState", {})
    devs = sc.get("mechanicDevelopmentCardsState", {})

    actions: list[dict] = []

    for _key, entry in sorted(gls.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
        if not isinstance(entry, dict):
            continue
        text = entry.get("text", {})
        if not isinstance(text, dict):
            continue
        log_type = text.get("type")
        player_color = text.get("playerColor")

        if log_type == 4:
            piece = text.get("pieceEnum")
            if piece == 2:
                for cid_str, cdata in tcs.items():
                    if isinstance(cdata, dict) and cdata.get("buildingType") == 1:
                        actions.append({"action": "settlement", "corner_id": int(cid_str), "player_color": player_color})
            elif piece == 0:
                for eid_str, edata in tes.items():
                    if isinstance(edata, dict) and edata.get("type") == 1:
                        actions.append({"action": "road", "edge_id": int(eid_str), "player_color": player_color})

        elif log_type == 5:
            piece = text.get("pieceEnum")
            if piece == 2:
                for cid_str, cdata in tcs.items():
                    if isinstance(cdata, dict) and cdata.get("buildingType") == 1:
                        actions.append({"action": "settlement", "corner_id": int(cid_str), "player_color": player_color})
            elif piece == 3:
                for cid_str, cdata in tcs.items():
                    if isinstance(cdata, dict) and cdata.get("buildingType") == 2:
                        actions.append({"action": "city", "corner_id": int(cid_str), "player_color": player_color})

        elif log_type == 10:
            actions.append({"action": "roll", "player_color": player_color})

        elif log_type == 11:
            tile_idx = robber.get("locationTileIndex")
            if tile_idx is not None:
                actions.append({"action": "robber", "tile_idx": tile_idx, "player_color": player_color})

        elif log_type == 20:
            card_enum = text.get("cardEnum")
            if card_enum == 11:
                actions.append({"action": "play_knight", "player_color": player_color})
            elif card_enum == 13:
                actions.append({"action": "play_monopoly", "player_color": player_color})
            elif card_enum == 14:
                actions.append({"action": "play_road_building", "player_color": player_color})
            elif card_enum == 15:
                actions.append({"action": "play_year_of_plenty", "player_color": player_color})

        elif log_type == 1 and devs:
            if "bankDevelopmentCards" in devs or any(
                "developmentCardsBoughtThisTurn" in (devs.get("players", {}).get(str(c), {}) or {})
                for c in range(10)
            ):
                actions.append({"action": "buy_dev", "player_color": player_color})

        elif log_type == 44:
            actions.append({"action": "end_turn"})

        elif log_type == 55:
            card_enums = text.get("cardEnums", [])
            if card_enums:
                actions.append({"action": "discard", "card_enums": card_enums, "player_color": player_color})

        elif log_type == 116:
            given = text.get("givenCardEnums", [])
            received = text.get("receivedCardEnums", [])
            if given and received:
                actions.append({
                    "action": "maritime_trade", "give_resource": given[0],
                    "receive_resource": received[0], "player_color": player_color,
                })

        # --- Trade offer (player-to-player) ---
        elif log_type == 118:
            offered = text.get("offeredCardEnums", [])
            wanted = text.get("wantedCardEnums", [])
            no, nw = len(offered), len(wanted)
            if (no, nw) in ((1, 1), (1, 2), (2, 1)) and offered and wanted:
                actions.append({
                    "action": "trade_offer",
                    "give_enums": offered, "want_enums": wanted,
                    "ratio": f"{no}:{nw}",
                    "player_color": player_color,
                })

    return actions


# ======================================================================
# Action index computation
# ======================================================================

def _action_to_index(parsed: dict, state: dict, topo: dict) -> int | None:
    """Convert a parsed Colonist action to a 337-dim action index."""
    act = parsed["action"]

    if act == "roll":
        return 0
    if act == "end_turn":
        return 1
    if act == "buy_dev":
        return 2
    if act == "play_knight":
        return 3
    if act == "play_road_building":
        return 4

    if act == "settlement":
        node = COL_CORNER_TO_NODE.get(parsed["corner_id"])
        if node is not None:
            return 5 + node
        return None

    if act == "city":
        node = COL_CORNER_TO_NODE.get(parsed["corner_id"])
        if node is not None:
            return 59 + node
        return None

    if act == "road":
        eidx = COL_EDGE_TO_ACTION_IDX.get(parsed["edge_id"])
        if eidx is not None:
            return 113 + eidx
        return None

    if act == "robber":
        col_hex = _col_tile_to_hex(parsed["tile_idx"], state)
        if col_hex is not None:
            c_tile = topo["col_hex_to_c_tile"].get(col_hex)
            if c_tile is not None:
                return 185 + c_tile * 5 + 4  # steal slot 4 = no steal

    if act == "discard":
        card_enums = parsed.get("card_enums", [])
        if card_enums:
            res = COL_CARD_ENUM_TO_RES.get(card_enums[0])
            if res is not None:
                return 280 + res

    if act == "play_year_of_plenty":
        return 285  # first YoP slot

    if act == "play_monopoly":
        return 305  # first monopoly slot

    if act == "maritime_trade":
        give = COL_CARD_ENUM_TO_RES.get(parsed.get("give_resource"))
        recv = COL_CARD_ENUM_TO_RES.get(parsed.get("receive_resource"))
        if give is not None and recv is not None and give != recv:
            idx = give * 4 + (recv if recv < give else recv - 1)
            return 310 + idx

    # Trade offers: 60 new slots starting at 337
    # Layout: [337..356] = 1:1 (20), [357..376] = 1:2 (20), [377..396] = 2:1 (20)
    # Each block: give_res * 4 + (recv_res if recv < give else recv - 1)
    if act == "trade_offer":
        give_enums = parsed.get("give_enums", [])
        want_enums = parsed.get("want_enums", [])
        no, nw = len(give_enums), len(want_enums)
        give_res = COL_CARD_ENUM_TO_RES.get(give_enums[0])
        want_res = COL_CARD_ENUM_TO_RES.get(want_enums[0])
        if give_res is None or want_res is None or give_res == want_res:
            return None
        pair_idx = give_res * 4 + (want_res if want_res < give_res else want_res - 1)
        if (no, nw) == (1, 1):
            return 337 + pair_idx
        elif (no, nw) == (1, 2):
            return 357 + pair_idx
        elif (no, nw) == (2, 1):
            return 377 + pair_idx

    return None


# ======================================================================
# Action mask construction from Colonist state
# ======================================================================

def _build_action_mask(
    state: dict,
    parsed: dict,
    action_idx: int,
    play_order: list[int],
    current_color: int,
) -> np.ndarray:
    """Build an approximate action mask from the Colonist game state.

    The mask doesn't need to be perfectly legal — it just needs to
    reflect which ACTION TYPES are plausible so the model learns the
    conditional distribution: "given these types of actions are available,
    which does the human pick?"

    We determine the game phase from the parsed action and state, then
    enable the appropriate action type slots.
    """
    mask = np.zeros(ACTION_SPACE, dtype=np.float32)
    act = parsed["action"]

    cs = state.get("currentState", {})
    ds = state.get("diceState", {})
    has_rolled = ds.get("diceThrown", False)
    is_setup = cs.get("completedTurns", 0) < 8

    # The chosen action must always be legal
    mask[action_idx] = 1.0

    if is_setup:
        # Setup phase: only settlements and roads
        mask[5:59] = 1.0     # all settlement positions
        mask[113:185] = 1.0  # all road positions
    elif act in ("roll",):
        # Pre-roll: can roll or play dev cards
        mask[0] = 1.0        # roll
        mask[3] = 1.0        # play knight
        mask[4] = 1.0        # play road building
    elif act in ("discard",):
        # Discard phase
        mask[280:285] = 1.0  # all 5 discard options
    elif act in ("robber",):
        # Robber placement
        mask[185:280] = 1.0  # all robber positions
    elif act == "end_turn":
        # Post-roll: can build, trade, buy dev, or end turn
        mask[1] = 1.0        # end turn
        mask[2] = 1.0        # buy dev
        mask[5:59] = 1.0     # settlements
        mask[59:113] = 1.0   # cities
        mask[113:185] = 1.0  # roads
        mask[285:310] = 1.0  # yop, monopoly
        mask[310:330] = 1.0  # maritime
        mask[337:397] = 1.0  # trade offers
    else:
        # Any main-game action: enable all post-roll options
        mask[1] = 1.0        # end turn
        mask[2] = 1.0        # buy dev
        mask[5:59] = 1.0     # settlements
        mask[59:113] = 1.0   # cities
        mask[113:185] = 1.0  # roads
        mask[285:310] = 1.0  # yop, monopoly
        mask[310:330] = 1.0  # maritime
        mask[337:397] = 1.0  # trade offers

    return mask




def _deep_merge(base: dict, delta: dict) -> None:
    """Apply Colonist state delta to accumulated state."""
    for k, v in delta.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ======================================================================
# Per-game conversion
# ======================================================================

def convert_single_game(
    game_path: str,
    topo: dict,
) -> tuple[list[dict], dict] | None:
    """Convert one Colonist.io JSON game to training step dicts.

    Uses pure-Python state encoding — no C engine, no dice divergence.
    """
    import copy

    with open(game_path) as f:
        raw = json.load(f)

    data = raw.get("data", raw)
    eh = data.get("eventHistory", {})
    events = eh.get("events", [])
    if not events:
        return None

    players = data.get("playerUserStates", [])
    if len(players) != 4:
        return None

    settings = data.get("gameSettings", {})
    if settings.get("extensionSetting", 0) != 0:
        return None
    if settings.get("scenarioSetting", 0) != 0:
        return None

    initial = eh.get("initialState", {})
    hexes = initial.get("mapState", {}).get("tileHexStates", {})
    if len(hexes) != 19:
        return None

    play_order = data.get("playOrder", [])
    if len(play_order) != 4:
        return None

    end_state = eh.get("endGameState", {})
    winner_color = _find_winner(end_state)
    if winner_color is None:
        return None

    color_to_seat = {c: i for i, c in enumerate(play_order)}
    reward_vec = np.zeros(4, dtype=np.float32)
    if winner_color in color_to_seat:
        reward_vec[color_to_seat[winner_color]] = 1.0
    else:
        return None

    state = copy.deepcopy(initial)
    production = _compute_production(state, topo)
    port_oh = _compute_port_oh(state)

    stats: dict[str, int] = {}
    steps: list[dict] = []

    for evt in events:
        sc = evt.get("stateChange", {})
        parsed_actions = _parse_event(evt)

        for parsed in parsed_actions:
            cs = state.get("currentState", {})
            current_color = cs.get("currentTurnPlayerColor") or play_order[0]

            action_idx = _action_to_index(parsed, state, topo)
            if action_idx is None:
                continue

            nf, ef, flat = _encode_state(
                state, topo, current_color, play_order, production, port_oh,
            )

            act_name = parsed["action"]
            stats[act_name] = stats.get(act_name, 0) + 1

            cp_seat = color_to_seat.get(current_color, 0)
            mask = _build_action_mask(state, parsed, action_idx, play_order, current_color)

            steps.append({
                "nf": nf,
                "ef": ef,
                "ff": flat,
                "mask": mask,
                "action_idx": action_idx,
                "player": cp_seat,
                "reward_vec": reward_vec.copy(),
            })

        _deep_merge(state, sc)

    if len(steps) < 10:
        return None

    return steps, stats


def _find_winner(end_state: dict) -> int | None:
    players_info = end_state.get("players", {})
    for color_str, pdata in players_info.items():
        if isinstance(pdata, dict):
            if pdata.get("winningPlayer") or pdata.get("rank") == 1:
                return int(color_str)
    return None


# ======================================================================
# Shard I/O
# ======================================================================

def _save_shard(steps: list[dict], output_dir: str, shard_id: str) -> int:
    S = len(steps)
    if S == 0:
        return 0
    nf = np.stack([s["nf"] for s in steps])
    ef = np.stack([s["ef"] for s in steps])
    ff = np.stack([s["ff"] for s in steps])
    mask = np.stack([s["mask"] for s in steps])
    action_idx = np.array([s["action_idx"] for s in steps], dtype=np.int64)
    player = np.array([s["player"] for s in steps], dtype=np.int64)
    reward_vec = np.stack([s["reward_vec"] for s in steps])

    torch.save({
        "node_features": torch.from_numpy(nf),
        "edge_features": torch.from_numpy(ef),
        "flat_features": torch.from_numpy(ff),
        "action_mask": torch.from_numpy(mask),
        "action_idx": torch.from_numpy(action_idx),
        "player": torch.from_numpy(player),
        "reward_vec": torch.from_numpy(reward_vec),
    }, os.path.join(output_dir, f"{shard_id}.pt"))
    return S


# ======================================================================
# Worker
# ======================================================================

def _worker_fn(
    worker_id: int,
    file_list: list[str],
    output_dir: str,
    games_per_shard: int,
    counter,
    total_files: int,
) -> None:
    topo = _precompute_topology()

    shard_steps: list[dict] = []
    shard_idx = 0
    total_steps = 0
    converted = 0
    skipped = 0
    errors = 0
    all_stats: dict[str, int] = {}
    t_start = time.time()

    for fi, fpath in enumerate(file_list):
        try:
            result = convert_single_game(fpath, topo)
        except Exception:
            errors += 1
            if errors <= 3 and worker_id == 0:
                traceback.print_exc()
            continue

        if result is None:
            skipped += 1
            continue

        steps, game_stats = result
        shard_steps.extend(steps)
        converted += 1
        for k, v in game_stats.items():
            all_stats[k] = all_stats.get(k, 0) + v

        if len(shard_steps) >= games_per_shard * 100:
            sid = f"w{worker_id:02d}_{shard_idx:04d}"
            n = _save_shard(shard_steps, output_dir, sid)
            total_steps += n
            shard_steps = []
            shard_idx += 1

        if counter is not None:
            with counter.get_lock():
                counter.value += 1
                count = counter.value
        else:
            count = fi + 1

        if count % 2000 == 0:
            elapsed = time.time() - t_start
            gps = (fi + 1) / elapsed if elapsed > 0 else 0
            print(f"[progress] {count}/{total_files} "
                  f"({count * 100 // total_files}%) | "
                  f"w{worker_id}: {gps:.1f} f/s  "
                  f"conv={converted} skip={skipped} err={errors}  "
                  f"steps={total_steps + len(shard_steps):,}",
                  flush=True)

    if shard_steps:
        sid = f"w{worker_id:02d}_{shard_idx:04d}"
        n = _save_shard(shard_steps, output_dir, sid)
        total_steps += n
        shard_idx += 1

    elapsed = time.time() - t_start
    print(f"[worker {worker_id}] Done: {len(file_list)} files, "
          f"{converted} converted, {skipped} skipped, {errors} errors, "
          f"{total_steps:,} steps, {shard_idx} shards in {elapsed:.1f}s",
          flush=True)
    if worker_id == 0:
        print(f"  Action breakdown: {dict(sorted(all_stats.items(), key=lambda x: -x[1]))}")


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert Colonist.io games (v3: Python encoder)")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--games-per-shard", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.num_workers <= 0:
        args.num_workers = max(1, os.cpu_count() - 2)

    json_files = sorted(str(p) for p in Path(args.input_dir).glob("*.json"))
    if args.max_games > 0:
        json_files = json_files[:args.max_games]

    print(f"Colonist.io -> hexzero converter (v3: Python encoder)")
    print(f"  Input:    {args.input_dir} ({len(json_files):,} files)")
    print(f"  Output:   {args.output_dir}")
    print(f"  Workers:  {args.num_workers}")
    print(flush=True)

    t0 = time.time()

    if args.num_workers <= 1:
        _worker_fn(0, json_files, args.output_dir, args.games_per_shard,
                    None, len(json_files))
    else:
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        chunk_size = len(json_files) // args.num_workers
        procs = []
        for w in range(args.num_workers):
            start = w * chunk_size
            end = start + chunk_size if w < args.num_workers - 1 else len(json_files)
            chunk = json_files[start:end]
            if not chunk:
                continue
            p = ctx.Process(
                target=_worker_fn,
                args=(w, chunk, args.output_dir, args.games_per_shard,
                      counter, len(json_files)),
                daemon=True,
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    elapsed = time.time() - t0

    pt_files = sorted(f for f in os.listdir(args.output_dir) if f.endswith(".pt"))
    total_steps = 0
    action_counts: dict[int, int] = {}
    for f in pt_files:
        d = torch.load(os.path.join(args.output_dir, f), weights_only=False)
        acts = d["action_idx"].numpy()
        total_steps += len(acts)
        for a in acts:
            action_counts[int(a)] = action_counts.get(int(a), 0) + 1

    from human_bot.evaluate import action_type_label
    type_counts: dict[str, int] = {}
    for aidx, cnt in action_counts.items():
        label = action_type_label(aidx)
        type_counts[label] = type_counts.get(label, 0) + cnt

    print(f"\n{'=' * 65}")
    print(f"Conversion complete")
    print(f"  Input files:     {len(json_files):,}")
    print(f"  Total steps:     {total_steps:,}")
    print(f"  Avg steps/game:  {total_steps / max(len(json_files), 1):.1f}")
    print(f"  Shards:          {len(pt_files)}")
    print(f"  Wall time:       {elapsed:.1f}s")
    print(f"\n  Action type distribution:")
    for label, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {label:<14s}  {cnt:>10,}  ({cnt / max(total_steps, 1) * 100:.1f}%)")
    unique_corners = len(set(a for a in action_counts if 5 <= a < 113))
    unique_edges = len(set(a for a in action_counts if 113 <= a < 185))
    unique_robber = len(set(a for a in action_counts if 185 <= a < 280))
    print(f"\n  Spatial differentiation:")
    print(f"    Unique settlement/city positions: {unique_corners}")
    print(f"    Unique road positions:            {unique_edges}")
    print(f"    Unique robber positions:           {unique_robber}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
