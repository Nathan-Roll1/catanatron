from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass
class GameConfig:
    num_players: int = 4
    vps_to_win: int = 10
    discard_limit: int = 7
    friendly_robber: bool = False
    max_actions: int = 128
    turns_limit: int = 1000
    num_nodes: int = 54
    num_edges: int = 72
    num_land_tiles: int = 19
    num_ports: int = 9
    total_board_nodes: int = 96
    max_degree: int = 3
    num_resources: int = 5
    num_dev_types: int = 5


@dataclass
class NetworkConfig:
    gnn_layers: int = 6
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 256
    trunk_blocks: int = 20
    trunk_channels: int = 256
    trunk_activation: str = "mish"
    policy_head_hidden: int = 256
    value_head_hidden: int = 256
    flat_feature_dim: int = 115
    node_feature_dim: int = 18
    edge_feature_dim: int = 5
    action_space_size: int = 337


@dataclass
class MCTSConfig:
    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 30
    temperature_init: float = 1.0
    temperature_final: float = 0.01
    max_tree_reuse: bool = True
    num_determinizations: int = 8
    virtual_loss: float = 3.0


@dataclass
class TrainingConfig:
    replay_buffer_size: int = 1_000_000
    batch_size: int = 2048
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 1000
    num_epochs_per_iteration: int = 10
    checkpoint_interval: int = 100
    eval_games: int = 50


@dataclass
class SelfPlayConfig:
    num_workers: int = 8
    games_per_iteration: int = 100
    max_game_length: int = 1000
    resign_threshold: float = -0.95
    resign_enabled: bool = False


@dataclass
class EloConfig:
    initial_elo: float = 1000.0
    k_factor: float = 32.0
    num_eval_games: int = 50
    confidence_games: int = 100


@dataclass
class RNaDConfig:
    eta: float = 0.2
    clip_bound: float = 10_000.0
    value_weight: float = 0.5
    anchor_interval: int = 200
    concurrent_games: int = 64
    gamma: float = 1.0
    rho_bar: float = 1.0
    c_bar: float = 1.0
    graded_rewards: bool = True


@dataclass
class HexaZeroConfig:
    game: GameConfig = field(default_factory=GameConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    elo: EloConfig = field(default_factory=EloConfig)
    rnad: RNaDConfig = field(default_factory=RNaDConfig)
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "runs/hexazero"

    @classmethod
    def from_dict(cls, d: dict) -> HexaZeroConfig:
        sub_configs = {
            "game": GameConfig,
            "network": NetworkConfig,
            "mcts": MCTSConfig,
            "training": TrainingConfig,
            "selfplay": SelfPlayConfig,
            "elo": EloConfig,
            "rnad": RNaDConfig,
        }
        kwargs = {}
        for f in fields(cls):
            if f.name not in d:
                continue
            val = d[f.name]
            if f.name in sub_configs and isinstance(val, dict):
                kwargs[f.name] = sub_configs[f.name](**val)
            else:
                kwargs[f.name] = val
        return cls(**kwargs)


def get_default_config() -> HexaZeroConfig:
    return HexaZeroConfig()
