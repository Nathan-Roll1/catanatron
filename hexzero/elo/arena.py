"""Arena for running evaluation matches and tracking ELO ratings.

Provides abstract Player interface, concrete player implementations
(Random, AB2, GreedyValue, HexaZero), and the Arena orchestrator that
manages match execution, seat rotation, and rating updates.
"""

from __future__ import annotations

import ctypes
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hexzero.bindings import (
    MAX_ACTIONS,
    Action,
    CatanMap,
    Game,
    RngState,
    SearchCtx,
    SearchResult,
    ValueFn,
    load_library,
)
from hexzero.bindings.structs import COLOR_NONE
from hexzero.config import EloConfig

from .rating import EloRating, MatchResult

if TYPE_CHECKING:
    from hexzero.config import MCTSConfig, NetworkConfig

log = logging.getLogger(__name__)


# ============================================================================
# Game protocol — matches the CatanGame interface from the spec.
# Concrete implementation lives elsewhere; we only depend on the protocol.
# ============================================================================


@runtime_checkable
class CatanGame(Protocol):
    """Minimal game interface expected by arena players."""

    def reset(self, seed: int | None = None) -> None: ...
    def step(self, action_index: int) -> tuple[object, bool]: ...
    def get_legal_actions(self) -> list[Action]: ...
    def current_player(self) -> int: ...
    def is_terminal(self) -> bool: ...
    def winner(self) -> int | None: ...
    def clone(self) -> CatanGame: ...
    @property
    def num_players(self) -> int: ...


# ============================================================================
# Abstract player
# ============================================================================


class Player(ABC):
    """Abstract base class for arena players."""

    @abstractmethod
    def select_action(self, game: CatanGame, legal_actions: list[Action]) -> int:
        """Return the *index* into ``legal_actions`` of the chosen move."""

    @property
    @abstractmethod
    def name(self) -> str: ...


# ============================================================================
# Concrete players
# ============================================================================


class RandomPlayer(Player):
    """Selects actions uniformly at random."""

    def __init__(self, seed: int = 42, name: str = "Random") -> None:
        self._rng = random.Random(seed)
        self._name = name

    def select_action(self, game: CatanGame, legal_actions: list[Action]) -> int:
        return self._rng.randrange(len(legal_actions))

    @property
    def name(self) -> str:
        return self._name


class AB2Player(Player):
    """Alpha-Beta depth-2 using the C engine's ``alphabeta_search``.

    This is the BASELINE player, pinned at 100 ELO.
    Falls back to greedy depth-1 (``base_value_fn`` on each successor) if
    the full search binding is unavailable.
    """

    def __init__(self, name: str = "AB2") -> None:
        self._name = name
        try:
            self._lib = load_library()
            self._search_available = True
        except (FileNotFoundError, OSError):
            self._lib = None
            self._search_available = False
            log.warning(
                "C library not found — AB2Player will use greedy fallback"
            )

    def select_action(self, game: CatanGame, legal_actions: list[Action]) -> int:
        if self._search_available:
            return self._ab_search(game, legal_actions)
        return self._greedy_fallback(game, legal_actions)

    def _ab_search(self, game: CatanGame, legal_actions: list[Action]) -> int:
        """Run alphabeta_search at depth 2 via ctypes."""
        assert self._lib is not None

        c_game: Game = self._extract_c_game(game)
        bot_color: int = c_game.state.colors[c_game.state.current_player_index]

        actions_arr = (Action * MAX_ACTIONS)()
        for i, a in enumerate(legal_actions):
            actions_arr[i] = a

        ctx = SearchCtx()
        ctx.depth_counter = 0

        wrapped_eval = ValueFn(self._lib.base_value_fn)

        result: SearchResult = self._lib.alphabeta_search(
            ctypes.byref(ctx),
            ctypes.byref(c_game),
            actions_arr,
            len(legal_actions),
            2,          # depth
            -1e30,      # alpha
            1e30,       # beta
            bot_color,
            wrapped_eval,
        )

        best_action = result.action
        for i, a in enumerate(legal_actions):
            if _actions_equal(a, best_action):
                return i

        return 0

    def _greedy_fallback(
        self, game: CatanGame, legal_actions: list[Action]
    ) -> int:
        """Pick the action that maximises base_value_fn on the successor.

        Only available when the library is loaded but full search isn't
        desired (or for testing).
        """
        if self._lib is None:
            return random.randrange(len(legal_actions))

        c_game: Game = self._extract_c_game(game)
        bot_color: int = c_game.state.colors[c_game.state.current_player_index]

        best_idx = 0
        best_val = -math.inf
        child = Game()
        child_actions = (Action * MAX_ACTIONS)()
        child_n = ctypes.c_int(0)

        for i, action in enumerate(legal_actions):
            self._lib.game_copy(ctypes.byref(child), ctypes.byref(c_game))
            self._lib.game_execute(
                ctypes.byref(child), action, child_actions, ctypes.byref(child_n)
            )
            val: float = self._lib.base_value_fn(ctypes.byref(child), bot_color)
            if val > best_val:
                best_val = val
                best_idx = i

        return best_idx

    @staticmethod
    def _extract_c_game(game: CatanGame) -> Game:
        """Get the underlying ctypes ``Game`` struct from a CatanGame.

        The concrete CatanGame is expected to expose a ``._game`` attribute
        holding the ctypes Game struct.  If not, we raise with a clear
        message so integrators know what to wire up.
        """
        inner = getattr(game, "_game", None)
        if inner is None:
            raise AttributeError(
                "CatanGame must expose a ._game attribute with the ctypes "
                "Game struct for AB2Player to use."
            )
        return inner  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return self._name


class GreedyValuePlayer(Player):
    """Picks the legal action that maximises ``base_value_fn`` on the successor."""

    def __init__(self, name: str = "Greedy") -> None:
        self._name = name
        try:
            self._lib = load_library()
        except (FileNotFoundError, OSError):
            self._lib = None
            log.warning(
                "C library not found — GreedyValuePlayer will play randomly"
            )

    def select_action(self, game: CatanGame, legal_actions: list[Action]) -> int:
        if self._lib is None:
            return random.randrange(len(legal_actions))

        c_game: Game = AB2Player._extract_c_game(game)
        bot_color: int = c_game.state.colors[c_game.state.current_player_index]

        best_idx = 0
        best_val = -math.inf
        child = Game()
        child_actions = (Action * MAX_ACTIONS)()
        child_n = ctypes.c_int(0)

        for i, action in enumerate(legal_actions):
            self._lib.game_copy(ctypes.byref(child), ctypes.byref(c_game))
            self._lib.game_execute(
                ctypes.byref(child), action, child_actions, ctypes.byref(child_n)
            )
            val: float = self._lib.base_value_fn(ctypes.byref(child), bot_color)
            if val > best_val:
                best_val = val
                best_idx = i

        return best_idx

    @property
    def name(self) -> str:
        return self._name


class HexaZeroPlayer(Player):
    """Neural MCTS player using a HexaZero network.

    Wraps an ``MCTSSearch`` instance to select actions with a near-zero
    temperature for deterministic competitive play.
    """

    def __init__(
        self,
        network: object,
        state_encoder: object,
        action_encoder: object,
        mcts_config: MCTSConfig,
        name: str = "HexaZero",
        device: str = "cuda",
        temperature: float = 0.01,
    ) -> None:
        self._network = network
        self._state_encoder = state_encoder
        self._action_encoder = action_encoder
        self._mcts_config = mcts_config
        self._name = name
        self._device = device
        self._temperature = temperature
        self._mcts: object | None = None

    def _get_mcts(self) -> object:
        """Lazily build the MCTSSearch instance."""
        if self._mcts is None:
            try:
                from hexzero.mcts import MCTSSearch  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "hexzero.mcts.MCTSSearch is required for HexaZeroPlayer"
                ) from exc
            self._mcts = MCTSSearch(
                network=self._network,
                state_encoder=self._state_encoder,
                action_encoder=self._action_encoder,
                config=self._mcts_config,
                device=self._device,
            )
        return self._mcts

    def select_action(self, game: CatanGame, legal_actions: list[Action]) -> int:
        mcts = self._get_mcts()
        result = mcts.search(game)  # type: ignore[union-attr]
        turn = getattr(game, "turn_number", 0)
        return mcts.select_action(  # type: ignore[union-attr]
            result.action_probs, self._temperature, turn
        )

    @property
    def name(self) -> str:
        return self._name


# ============================================================================
# Evaluation result
# ============================================================================


@dataclass
class EvalResult:
    """Aggregated result of an evaluation run between two players."""

    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int
    total_games: int
    win_rate_a: float
    win_rate_b: float
    elo_a: float
    elo_b: float
    elo_diff: float
    confidence_interval: tuple[float, float]
    avg_game_length: float


# ============================================================================
# Arena
# ============================================================================


class Arena:
    """Runs evaluation matches between players and tracks ELO ratings.

    Manages game execution, seat rotation for fairness, rating updates,
    and statistical reporting.
    """

    BASELINE_NAME = "AB2"
    BASELINE_ELO = 100.0

    def __init__(self, elo_system: EloRating, config: EloConfig) -> None:
        self.elo = elo_system
        self.config = config
        self.elo.register_player(
            self.BASELINE_NAME, self.BASELINE_ELO, pinned=True
        )

    # ------------------------------------------------------------------
    # Single match
    # ------------------------------------------------------------------

    def run_match(
        self, players: list[Player], game: CatanGame, seed: int = 0
    ) -> MatchResult:
        """Play a single 4-player game and return the result.

        ``players`` maps seat index to the Player that controls it.
        """
        game.reset(seed=seed)
        turn = 0

        while not game.is_terminal():
            current = game.current_player()
            legal = game.get_legal_actions()
            if not legal:
                break
            player = players[current]
            action_idx = player.select_action(game, legal)
            game.step(action_idx)
            turn += 1

        winner_seat = game.winner()
        winner_name = players[winner_seat].name if winner_seat is not None else ""

        return MatchResult(
            players=[p.name for p in players],
            winner=winner_name,
            winner_seat=winner_seat if winner_seat is not None else -1,
            num_turns=turn,
            game_seed=seed,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        player_a: Player,
        player_b: Player,
        game: CatanGame,
        num_games: int = 50,
        fill_with: Player | None = None,
    ) -> EvalResult:
        """Evaluate *player_a* vs *player_b* over many games.

        Remaining seats are filled with *fill_with* (default: RandomPlayer).
        Seating is rotated each game so every player gets equal time in
        each seat position.
        """
        filler = fill_with or RandomPlayer(seed=123, name="Filler")

        for p in (player_a, player_b, filler):
            if p.name not in self.elo.ratings:
                self.elo.register_player(p.name, self.config.initial_elo)

        wins_a = 0
        wins_b = 0
        draws = 0
        total_turns = 0

        for i in range(num_games):
            roster = self._build_roster(player_a, player_b, filler, rotation=i)
            result = self.run_match(roster, game, seed=i)

            if result.winner == player_a.name:
                wins_a += 1
            elif result.winner == player_b.name:
                wins_b += 1
            else:
                draws += 1

            if result.winner:
                self.elo.update_ratings(result)

            total_turns += result.num_turns

            if (i + 1) % 10 == 0:
                log.info(
                    "Game %d/%d | %s: %d  %s: %d  draws: %d | ELO %s=%.0f  %s=%.0f",
                    i + 1, num_games,
                    player_a.name, wins_a, player_b.name, wins_b, draws,
                    player_a.name, self.elo.get_rating(player_a.name),
                    player_b.name, self.elo.get_rating(player_b.name),
                )

        total = wins_a + wins_b + draws
        wr_a = wins_a / max(total, 1)
        wr_b = wins_b / max(total, 1)

        elo_a = self.elo.get_rating(player_a.name)
        elo_b = self.elo.get_rating(player_b.name)

        ci = _elo_confidence_interval(wr_a, total)

        return EvalResult(
            player_a=player_a.name,
            player_b=player_b.name,
            wins_a=wins_a,
            wins_b=wins_b,
            draws=draws,
            total_games=total,
            win_rate_a=wr_a,
            win_rate_b=wr_b,
            elo_a=elo_a,
            elo_b=elo_b,
            elo_diff=elo_a - elo_b,
            confidence_interval=ci,
            avg_game_length=total_turns / max(total, 1),
        )

    def evaluate_against_baseline(
        self,
        player: Player,
        game: CatanGame,
        num_games: int = 50,
    ) -> EvalResult:
        """Evaluate *player* against the AB2 baseline (100 ELO).

        Fills remaining seats with random players.
        """
        baseline = AB2Player(name=self.BASELINE_NAME)
        return self.evaluate(
            player_a=player,
            player_b=baseline,
            game=game,
            num_games=num_games,
            fill_with=RandomPlayer(seed=99, name="Filler"),
        )

    def round_robin(
        self,
        players: list[Player],
        game: CatanGame,
        games_per_matchup: int = 20,
    ) -> dict[str, object]:
        """Run a round-robin tournament between all players.

        Every pair plays *games_per_matchup* head-to-head games (remaining
        seats filled with RandomPlayer).  Returns a dict with per-matchup
        EvalResults and the final ratings table.
        """
        results: dict[str, EvalResult] = {}

        for i, pa in enumerate(players):
            for pb in players[i + 1 :]:
                key = f"{pa.name}_vs_{pb.name}"
                log.info("Round-robin: %s vs %s (%d games)", pa.name, pb.name, games_per_matchup)
                res = self.evaluate(
                    pa, pb, game, num_games=games_per_matchup
                )
                results[key] = res

        return {
            "matchups": results,
            "ratings": self.elo.get_ratings_table(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_roster(
        player_a: Player,
        player_b: Player,
        filler: Player,
        rotation: int,
    ) -> list[Player]:
        """Create a 4-seat roster rotating A and B through all positions."""
        base: list[Player] = [filler, filler, player_a, player_b]
        shift = rotation % 4
        return base[shift:] + base[:shift]


# ============================================================================
# Utilities
# ============================================================================


def _actions_equal(a: Action, b: Action) -> bool:
    """Compare two ctypes Action structs field-by-field."""
    if a.color != b.color or a.type != b.type:
        return False
    for i in range(5):
        if a.value[i] != b.value[i]:
            return False
    return True


def _elo_confidence_interval(
    win_rate: float, num_games: int, z: float = 1.96
) -> tuple[float, float]:
    """95 % confidence interval on ELO difference from observed win rate.

    Uses the normal approximation:
        ELO_diff = -400 * log10(1/win_rate - 1)
        SE       = 400 / (ln(10) * sqrt(n * p * (1-p)))
        CI       = ELO_diff +/- z * SE
    """
    if num_games == 0:
        return (0.0, 0.0)

    p = max(min(win_rate, 0.999), 0.001)
    elo_diff = -400.0 * math.log10(1.0 / p - 1.0)

    se = 400.0 / (math.log(10.0) * math.sqrt(num_games * p * (1.0 - p)))
    return (round(elo_diff - z * se, 1), round(elo_diff + z * se, 1))
