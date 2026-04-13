import random

from catanatron.models.player import Player
from catanatron.models.map import DICE_PROBAS
from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY

TRANSLATE_VARIETY = 4
_PROBA_POINT = 2.778 / 100
_VARIETY_BONUS = TRANSLATE_VARIETY * _PROBA_POINT
_DICE = DICE_PROBAS

DEFAULT_WEIGHTS = {
    "public_vps": 3e14,
    "production": 1e8,
    "enemy_production": -1e8,
    "num_tiles": 1,
    "reachable_production_0": 0,
    "reachable_production_1": 1e4,
    "buildable_nodes": 1e3,
    "longest_road": 10,
    "hand_synergy": 1e2,
    "hand_resources": 1,
    "discard_penalty": -5,
    "hand_devs": 10,
    "army_size": 10.1,
}

CONTENDER_WEIGHTS = {
    "public_vps": 300000000000001.94,
    "production": 100000002.04188395,
    "enemy_production": -99999998.03389844,
    "num_tiles": 2.91440418,
    "reachable_production_0": 2.03820085,
    "reachable_production_1": 10002.018773150001,
    "buildable_nodes": 1001.86278466,
    "longest_road": 12.127388499999999,
    "hand_synergy": 102.40606877,
    "hand_resources": 2.43644327,
    "discard_penalty": -3.00141993,
    "hand_devs": 10.721669799999999,
    "army_size": 12.93844622,
}


def _compute_production(state, board, color, robber_coordinate):
    """Compute total production + variety for a single color. No dicts."""
    settlements = state.buildings_by_color[color].get(SETTLEMENT, ())
    cities = state.buildings_by_color[color].get(CITY, ())
    adj = board.map.adjacent_tiles
    robber_tile = board.map.tiles.get(robber_coordinate)
    dice = _DICE
    total = 0.0
    variety = 0
    for resource in RESOURCES:
        res_prod = 0.0
        for node_id in settlements:
            for t in adj[node_id]:
                if t.resource is resource and t is not robber_tile and t.number is not None:
                    res_prod += dice[t.number]
        for node_id in cities:
            for t in adj[node_id]:
                if t.resource is resource and t is not robber_tile and t.number is not None:
                    res_prod += dice[t.number] + dice[t.number]
        total += res_prod
        if res_prod != 0.0:
            variety += 1
    return total, variety


def base_fn(params=DEFAULT_WEIGHTS):
    w_vps = params["public_vps"]
    w_prod = params["production"]
    w_eprod = params["enemy_production"]
    w_tiles = params["num_tiles"]
    w_reach0 = params["reachable_production_0"]
    w_reach1 = params["reachable_production_1"]
    w_buildable = params["buildable_nodes"]
    w_road = params["longest_road"]
    w_synergy = params["hand_synergy"]
    w_hand = params["hand_resources"]
    w_discard = params["discard_penalty"]
    w_devs = params["hand_devs"]
    w_army = params["army_size"]

    def fn(game, p0_color):
        state = game.state
        board = state.board
        ps = state.player_state
        robber = board.robber_coordinate
        colors = state.colors
        p0_idx = state.color_to_index[p0_color]
        key = f"P{p0_idx}"

        p0_prod, p0_var = _compute_production(state, board, p0_color, robber)
        production = p0_prod + p0_var * _VARIETY_BONUS

        enemy_color = colors[1] if colors[0] == p0_color else colors[0]
        e_prod, _ = _compute_production(state, board, enemy_color, robber)
        enemy_production = e_prod

        longest_road_length = ps[f"{key}_LONGEST_ROAD_LENGTH"]

        wheat = ps[f"{key}_WHEAT_IN_HAND"]
        ore = ps[f"{key}_ORE_IN_HAND"]
        sheep = ps[f"{key}_SHEEP_IN_HAND"]
        brick = ps[f"{key}_BRICK_IN_HAND"]
        wood = ps[f"{key}_WOOD_IN_HAND"]

        d_city = (max(2 - wheat, 0) + max(3 - ore, 0)) / 5.0
        d_settle = (max(1-wheat,0) + max(1-sheep,0) + max(1-brick,0) + max(1-wood,0)) / 4.0
        hand_synergy = (2 - d_city - d_settle) / 2

        num_in_hand = wood + brick + sheep + wheat + ore

        buildings = state.buildings_by_color[p0_color]
        owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
        owned_tiles = set()
        for n in owned_nodes:
            owned_tiles.update(board.map.adjacent_tiles[n])

        num_buildable_nodes = len(board.buildable_node_ids(p0_color))
        longest_road_factor = w_road if num_buildable_nodes == 0 else 0.1

        num_devs = (
            ps[f"{key}_KNIGHT_IN_HAND"]
            + ps[f"{key}_YEAR_OF_PLENTY_IN_HAND"]
            + ps[f"{key}_ROAD_BUILDING_IN_HAND"]
            + ps[f"{key}_MONOPOLY_IN_HAND"]
            + ps[f"{key}_VICTORY_POINT_IN_HAND"]
        )
        army = ps[f"{key}_PLAYED_KNIGHT"]

        return float(
            ps[f"{key}_VICTORY_POINTS"] * w_vps
            + production * w_prod
            + enemy_production * w_eprod
            + hand_synergy * w_synergy
            + num_buildable_nodes * w_buildable
            + len(owned_tiles) * w_tiles
            + num_in_hand * w_hand
            + (w_discard if num_in_hand > 7 else 0)
            + longest_road_length * longest_road_factor
            + num_devs * w_devs
            + army * w_army
        )

    return fn


def value_production(sample, player_name="P0", include_variety=True):
    features = [
        f"EFFECTIVE_{player_name}_WHEAT_PRODUCTION",
        f"EFFECTIVE_{player_name}_ORE_PRODUCTION",
        f"EFFECTIVE_{player_name}_SHEEP_PRODUCTION",
        f"EFFECTIVE_{player_name}_WOOD_PRODUCTION",
        f"EFFECTIVE_{player_name}_BRICK_PRODUCTION",
    ]
    prod_sum = sum(sample[f] for f in features)
    prod_variety = sum(1 for f in features if sample[f] != 0) * _VARIETY_BONUS
    return prod_sum + (prod_variety if include_variety else 0)


def contender_fn(params):
    return base_fn(params or CONTENDER_WEIGHTS)


class ValueFunctionPlayer(Player):
    """
    Player that selects the move that maximizes a heuristic value function.

    For now, the base value function only considers 1 enemy player.
    """

    def __init__(
        self, color, value_fn_builder_name=None, params=None, is_bot=True, epsilon=None
    ):
        super().__init__(color, is_bot)
        self.value_fn_builder_name = (
            "contender_fn" if value_fn_builder_name == "C" else "base_fn"
        )
        self.params = params
        self.epsilon = epsilon

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        value_fn = get_value_fn(self.value_fn_builder_name, self.params)
        best_value = float("-inf")
        best_action = None
        for action in playable_actions:
            game_copy = game.copy()
            game_copy.execute(action)
            value = value_fn(game_copy, self.color)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def __str__(self):
        return super().__str__() + f"(value_fn={self.value_fn_builder_name})"


_BASE_FN_CACHE = {}


def get_value_fn(name, params, value_function=None):
    if value_function is not None:
        return value_function
    elif name == "base_fn":
        if "base" not in _BASE_FN_CACHE:
            _BASE_FN_CACHE["base"] = base_fn(DEFAULT_WEIGHTS)
        return _BASE_FN_CACHE["base"]
    elif name == "contender_fn":
        return contender_fn(params)
    else:
        raise ValueError
