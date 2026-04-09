"""Configuration dataclasses for human_bot training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HumanBotTrainingConfig:
    # Data
    data_dir: str = "data/human_games"
    val_fraction: float = 0.10
    max_examples: int = 0  # 0 = no limit

    # Model (uses HumanBotNet ~192k params by default)
    pretrained_checkpoint: str = ""
    freeze_encoder_epochs: int = 2

    # Optimiser — tuned for M5 Max MPS
    lr_encoder_frozen: float = 2e-3
    lr_finetune: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 4096  # large batches exploit unified memory

    # Schedule
    epochs: int = 20
    lr_warmup_steps: int = 300
    lr_schedule: str = "cosine"

    # Loss
    label_smoothing: float = 0.05
    entropy_weight: float = 0.01
    use_uncertainty_weighting: bool = True
    policy_weight: float = 1.0
    value_weight: float = 0.5

    # Evaluation
    eval_games: int = 25
    eval_temperature: float = 0.1
    eval_every: int = 1

    # MPS hardware optimisation (tuned for M5 Max, 192k-param model)
    device: str = "auto"
    use_fp16: bool = False       # fp32 is faster for this model size (no cast overhead)
    compile_model: bool = False  # compile overhead exceeds gains at this scale
    num_workers: int = 0         # unified memory = direct tensor indexing, no workers

    # Infrastructure
    checkpoint_dir: str = "checkpoints/human_bot"
    wandb_key: str = ""
    wandb_project: str = "human-bot"
    seed: int = 42


@dataclass
class ColonistDataConfig:
    """Settings for ingesting Colonist.io JSON game archives."""

    archive_dir: str = "data/colonist_raw"
    output_dir: str = "data/human_games"
    games_per_shard: int = 200
    min_turns: int = 20
    max_turns: int = 500
    require_four_players: bool = True
    skip_incomplete: bool = True
