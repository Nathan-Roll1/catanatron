# Catanatron

Fast Settlers of Catan simulator and strong AI player. Run thousands of games in seconds.

## Installation

```bash
python -m venv venv
source ./venv/bin/activate
pip install -e .
```

Requires Python 3.11+.

## Usage

```bash
# Run 1000 games with random players
catanatron-play --players=R,R,R,R --num=1000 --quiet

# Pit bots against each other
catanatron-play --players=AB:2,F,W,R --num=500

# Save game data as JSON
catanatron-play --players=R,R,R,R --num=100 --output ./data

# See all player types
catanatron-play --help-players
```

## Player Types

| Code | Player | Description |
|------|--------|-------------|
| R | RandomPlayer | Chooses actions at random |
| W | WeightedRandomPlayer | Favors buying cities, settlements, dev cards |
| VP | VictoryPointPlayer | Greedy on immediate VP gain |
| F | ValueFunctionPlayer | Hand-crafted heuristic evaluation |
| G:N | GreedyPlayoutsPlayer | N random playouts per action |
| M:N | MCTSPlayer | Monte Carlo Tree Search with N simulations |
| AB:N | AlphaBetaPlayer | Alpha-beta search at depth N |
| SAB | SameTurnAlphaBetaPlayer | Alpha-beta within current turn only |

## Python API

```python
from catanatron import Game, RandomPlayer, Color

players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())
```

## Custom Players

```python
from catanatron.models.player import Player
from catanatron.cli.cli_players import register_cli_player

class MyBot(Player):
    def decide(self, game, playable_actions):
        return playable_actions[0]

register_cli_player("MY", MyBot)
```

```bash
catanatron-play --code=my_bot.py --players=MY,R,R,R --num=100
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```
