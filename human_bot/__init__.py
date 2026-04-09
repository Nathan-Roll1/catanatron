"""human_bot — Behavioral cloning from human Catan games.

Trains a dual-head neural network (shared GNN encoder + ResNet trunk) to
predict both *what action a human would take* (policy head) and *who will
win from this board state* (value head).  Built on top of the hexzero
game engine and HexaZeroNet architecture.
"""

__version__ = "0.1.0"
