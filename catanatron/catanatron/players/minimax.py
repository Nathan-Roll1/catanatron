import time
import random
from typing import Any

from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.player import Player
from catanatron.players.tree_search_utils import execute_spectrum, list_prunned_actions
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,
)


ALPHABETA_DEFAULT_DEPTH = 2
MAX_SEARCH_TIME_SECS = 120

FREE_ACTIONS = frozenset({ActionType.ROLL})

ACTION_ORDER = {
    ActionType.BUILD_CITY: 0,
    ActionType.BUILD_SETTLEMENT: 1,
    ActionType.BUY_DEVELOPMENT_CARD: 2,
    ActionType.BUILD_ROAD: 3,
    ActionType.PLAY_KNIGHT_CARD: 4,
    ActionType.PLAY_MONOPOLY: 5,
    ActionType.PLAY_YEAR_OF_PLENTY: 6,
    ActionType.PLAY_ROAD_BUILDING: 7,
    ActionType.MARITIME_TRADE: 8,
    ActionType.MOVE_ROBBER: 9,
    ActionType.END_TURN: 10,
    ActionType.ROLL: 11,
    ActionType.DISCARD_RESOURCE: 12,
}


def _action_sort_key(action):
    return ACTION_ORDER.get(action.action_type, 20)


class AlphaBetaPlayer(Player):
    """
    Player that executes an AlphaBeta Search where the value of each node
    is taken to be the expected value (using the probability of rolls, etc...)
    of its children. At leafs we simply use the heuristic function given.

    ROLL actions don't consume depth since they're forced (no decision).
    This means depth=2 sees a full turn (roll+build+end) and depth=3
    additionally sees the opponent's response.
    """

    def __init__(
        self,
        color,
        depth=ALPHABETA_DEFAULT_DEPTH,
        prunning=False,
        value_fn_builder_name=None,
        params=DEFAULT_WEIGHTS,
        epsilon=None,
        time_limit=MAX_SEARCH_TIME_SECS,
    ):
        super().__init__(color)
        self.depth = int(depth)
        self.prunning = str(prunning).lower() != "false" or self.depth >= 3
        if value_fn_builder_name == "C":
            self.value_fn_builder_name = "contender_fn"
        else:
            self.value_fn_builder_name = value_fn_builder_name or "base_fn"
        self.params = params
        self.use_value_function = None
        self.epsilon = epsilon
        self.time_limit = int(time_limit)

    def value_function(self, game, p0_color):
        raise NotImplementedError

    def get_actions(self, game):
        if self.prunning:
            actions = list_prunned_actions(game)
        else:
            actions = game.playable_actions
        return sorted(actions, key=_action_sort_key)

    def decide(self, game: Game, playable_actions):
        actions = self.get_actions(game)
        if len(actions) == 1:
            return actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        deadline = time.time() + self.time_limit
        self._value_fn = get_value_fn(
            self.value_fn_builder_name,
            self.params,
            self.value_function if self.use_value_function else None,
        )
        best_action, best_value = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline
        )
        if best_action is None:
            return playable_actions[0]
        return best_action

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(depth={self.depth},value_fn={self.value_fn_builder_name},prunning={self.prunning})"
        )

    def alphabeta(self, game, depth, alpha, beta, deadline):
        if depth <= 0 or game.winning_color() is not None or time.time() >= deadline:
            return None, self._value_fn(game, self.color)

        maximizingPlayer = game.state.current_color() == self.color
        actions = self.get_actions(game)

        if maximizingPlayer:
            best_action = None
            best_value = float("-inf")
            for action in actions:
                outcomes = execute_spectrum(game, action)
                child_depth = depth if action.action_type in FREE_ACTIONS else depth - 1

                expected_value = 0
                for outcome, proba in outcomes:
                    _, value = self.alphabeta(
                        outcome, child_depth, alpha, beta, deadline
                    )
                    expected_value += proba * value

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for action in actions:
                outcomes = execute_spectrum(game, action)
                child_depth = depth if action.action_type in FREE_ACTIONS else depth - 1

                expected_value = 0
                for outcome, proba in outcomes:
                    _, value = self.alphabeta(
                        outcome, child_depth, alpha, beta, deadline
                    )
                    expected_value += proba * value

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_action, best_value


class SameTurnAlphaBetaPlayer(AlphaBetaPlayer):
    """AlphaBeta but only searches within the current turn."""

    def alphabeta(self, game, depth, alpha, beta, deadline):
        if (
            depth <= 0
            or game.state.current_color() != self.color
            or game.winning_color() is not None
            or time.time() >= deadline
        ):
            return None, self._value_fn(game, self.color)

        actions = self.get_actions(game)

        best_action = None
        best_value = float("-inf")
        for action in actions:
            outcomes = execute_spectrum(game, action)
            child_depth = depth if action.action_type in FREE_ACTIONS else depth - 1

            expected_value = 0
            for outcome, proba in outcomes:
                _, value = self.alphabeta(
                    outcome, child_depth, alpha, beta, deadline
                )
                expected_value += proba * value

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        return best_action, best_value
