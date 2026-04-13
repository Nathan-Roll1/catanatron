"""Evaluation entry point: pit HexaZero against AB2 baseline.

Runs arena matches and updates the ELO tracking file.

Usage:
    python -m hexzero.scripts.evaluate \
        --checkpoint checkpoints/latest.pt \
        --num-games 100 \
        --elo-file elo_history/ratings.json \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="HexaZero evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument("--elo-file", type=str, required=True)
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="MCTS sims for eval (lower than training for speed)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    elo_path = Path(args.elo_file)
    elo_path.parent.mkdir(parents=True, exist_ok=True)

    from hexzero.config import get_default_config, MCTSConfig
    from hexzero.elo.arena import AB2Player, Arena, HexaZeroPlayer, RandomPlayer
    from hexzero.elo.rating import EloRating
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.model.network import HexaZeroNet

    cfg = get_default_config()

    if elo_path.exists():
        elo = EloRating.load(str(elo_path))
        log.info("Loaded ELO history: %d matches", len(elo.match_history))
    else:
        elo = EloRating(k_factor=cfg.elo.k_factor)

    arena = Arena(elo, cfg.elo)

    net = HexaZeroNet.load_checkpoint(args.checkpoint, device=device)
    net.eval()

    game_for_encoder = CatanGame(seed=0)
    game_for_encoder.reset()
    state_encoder = game_for_encoder.make_state_encoder()
    action_encoder = ActionEncoder()

    eval_mcts_cfg = MCTSConfig(
        num_simulations=args.mcts_sims,
        num_determinizations=1,
        c_puct=cfg.mcts.c_puct,
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=0.0,
        temperature_threshold=0,
        temperature_init=0.01,
        temperature_final=0.01,
    )

    hz_player = HexaZeroPlayer(
        network=net,
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        mcts_config=eval_mcts_cfg,
        name="HexaZero",
        device=device,
        temperature=0.01,
    )

    game = CatanGame(seed=42)
    result = arena.evaluate_against_baseline(hz_player, game, num_games=args.num_games)

    log.info("=" * 60)
    log.info("EVALUATION RESULTS")
    log.info("=" * 60)
    log.info("  %s wins: %d / %d (%.1f%%)",
             result.player_a, result.wins_a, result.total_games, result.win_rate_a * 100)
    log.info("  %s wins: %d / %d (%.1f%%)",
             result.player_b, result.wins_b, result.total_games, result.win_rate_b * 100)
    log.info("  ELO: %s=%.0f  %s=%.0f  (diff=%.0f)",
             result.player_a, result.elo_a, result.player_b, result.elo_b, result.elo_diff)
    log.info("  95%% CI: [%.0f, %.0f]", *result.confidence_interval)
    log.info("  Avg game length: %.1f turns", result.avg_game_length)

    elo.save(str(elo_path))
    log.info("Saved ELO history -> %s", elo_path)

    table = elo.get_ratings_table()
    log.info("Ratings table:")
    for row in table:
        pin = " [PINNED]" if row["pinned"] else ""
        log.info("  %-20s %8.1f  (%d games)%s",
                 row["name"], row["rating"], row["games_played"], pin)


if __name__ == "__main__":
    main()
