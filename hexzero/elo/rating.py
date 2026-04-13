"""ELO rating system for tracking player strength in Catan.

Implements standard ELO with extensions for multi-player (4-player) games.
The AB2 bot (alpha-beta depth 2) is the baseline at 100 ELO.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RatingSnapshot:
    """Point-in-time rating record."""

    rating: float
    timestamp: float
    games_played: int


@dataclass
class MatchResult:
    """Result of a single game."""

    players: list[str]
    winner: str
    winner_seat: int
    num_turns: int
    game_seed: int
    timestamp: float


class EloRating:
    """ELO rating system for tracking player strength.

    Standard ELO with configurable K-factor and multi-player support.
    For 4-player Catan the winner is compared pairwise against each loser,
    and K is divided by the number of comparisons to prevent inflation.
    """

    def __init__(self, k_factor: float = 32.0) -> None:
        self.k_factor = k_factor
        self.ratings: dict[str, float] = {}
        self.games_played: dict[str, int] = {}
        self.history: dict[str, list[RatingSnapshot]] = {}
        self.match_history: list[MatchResult] = []
        self._pinned: set[str] = set()

    def register_player(
        self, name: str, initial_rating: float = 1500.0, *, pinned: bool = False
    ) -> None:
        """Register a new player with initial rating.

        If *pinned* is True the player's rating is never updated (used for
        the AB2 baseline so all other ratings are measured relative to it).
        """
        if name in self.ratings:
            return
        self.ratings[name] = initial_rating
        self.games_played[name] = 0
        self.history[name] = [
            RatingSnapshot(
                rating=initial_rating, timestamp=time.time(), games_played=0
            )
        ]
        if pinned:
            self._pinned.add(name)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score for player A against player B.

        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))

    def update_ratings(self, match: MatchResult) -> dict[str, float]:
        """Update ratings after a game and return the deltas.

        For 4-player Catan the winner (score 1) is compared pairwise against
        each loser (score 0).  K is divided by (num_players - 1) so that
        total rating change stays bounded regardless of player count.
        """
        self.match_history.append(match)

        for name in match.players:
            if name not in self.ratings:
                self.register_player(name)

        losers = [p for p in match.players if p != match.winner]
        num_comparisons = len(losers)
        k_adj = self.k_factor / max(num_comparisons, 1)

        deltas: dict[str, float] = {p: 0.0 for p in match.players}

        winner_rating = self.ratings[match.winner]
        for loser in losers:
            loser_rating = self.ratings[loser]

            e_w = self.expected_score(winner_rating, loser_rating)
            e_l = self.expected_score(loser_rating, winner_rating)

            deltas[match.winner] += k_adj * (1.0 - e_w)
            deltas[loser] += k_adj * (0.0 - e_l)

        now = time.time()
        for name in match.players:
            self.games_played[name] += 1
            if name not in self._pinned:
                self.ratings[name] += deltas[name]
            self.history[name].append(
                RatingSnapshot(
                    rating=self.ratings[name],
                    timestamp=now,
                    games_played=self.games_played[name],
                )
            )

        return deltas

    def get_rating(self, name: str) -> float:
        if name not in self.ratings:
            raise KeyError(f"Player '{name}' is not registered")
        return self.ratings[name]

    def get_ratings_table(self) -> list[dict[str, object]]:
        """Return ratings sorted descending with uncertainty estimates."""
        rows: list[dict[str, object]] = []
        for name, rating in self.ratings.items():
            gp = self.games_played[name]
            uncertainty = 400.0 / math.sqrt(max(gp, 1))
            rows.append(
                {
                    "name": name,
                    "rating": round(rating, 1),
                    "games_played": gp,
                    "uncertainty": round(uncertainty, 1),
                    "pinned": name in self._pinned,
                }
            )
        rows.sort(key=lambda r: r["rating"], reverse=True)  # type: ignore[arg-type]
        return rows

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize full state to JSON."""
        data = {
            "k_factor": self.k_factor,
            "ratings": self.ratings,
            "games_played": self.games_played,
            "pinned": list(self._pinned),
            "history": {
                name: [asdict(s) for s in snaps]
                for name, snaps in self.history.items()
            },
            "match_history": [asdict(m) for m in self.match_history],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> EloRating:
        """Deserialize from JSON."""
        data = json.loads(Path(path).read_text())
        obj = cls(k_factor=data["k_factor"])
        obj.ratings = data["ratings"]
        obj.games_played = data.get("games_played", {n: 0 for n in obj.ratings})
        obj._pinned = set(data.get("pinned", []))
        obj.history = {
            name: [RatingSnapshot(**s) for s in snaps]
            for name, snaps in data.get("history", {}).items()
        }
        obj.match_history = [
            MatchResult(**m) for m in data.get("match_history", [])
        ]
        return obj
