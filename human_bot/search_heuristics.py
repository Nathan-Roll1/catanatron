"""Search-time heuristic bonuses for known value head blind spots."""

CITY_BONUS = 1.0
SETTLEMENT_BONUS = 0.4
ROAD_BONUS = 0.05
BUY_DEV_BONUS = 0.1
MARITIME_PENALTY = 0.0

AT_BUILD_ROAD = 3
AT_BUILD_SETTLEMENT = 4
AT_BUILD_CITY = 5
AT_BUY_DEV = 6
AT_MOVE_ROBBER = 1
AT_MARITIME_TRADE = 11


def apply_action_bonus(value: float, act) -> float:
    """Add heuristic bonus/penalty to a candidate action's value score."""
    if act.type == AT_BUILD_CITY:
        return value + CITY_BONUS
    if act.type == AT_BUILD_SETTLEMENT:
        return value + SETTLEMENT_BONUS
    if act.type == AT_BUILD_ROAD:
        return value + ROAD_BONUS
    if act.type == AT_BUY_DEV:
        return value + BUY_DEV_BONUS
    if act.type == AT_MARITIME_TRADE:
        return value + MARITIME_PENALTY
    return value


def fix_robber_steal(chosen_idx: int, le: list) -> int:
    """If chosen action is MOVE_ROBBER with no-steal, substitute a steal
    variant on the same tile if one exists."""
    act = le[chosen_idx]
    if act.type != AT_MOVE_ROBBER:
        return chosen_idx
    if act.value[3] >= 0:
        return chosen_idx

    tile = (act.value[0], act.value[1], act.value[2])
    for i, a in enumerate(le):
        if a.type == AT_MOVE_ROBBER and a.value[3] >= 0:
            if (a.value[0], a.value[1], a.value[2]) == tile:
                return i
    # No steal variant on same tile; try any steal on any tile
    for i, a in enumerate(le):
        if a.type == AT_MOVE_ROBBER and a.value[3] >= 0:
            return i
    return chosen_idx
