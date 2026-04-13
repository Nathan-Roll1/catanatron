"""Self-play worker entry point for Slurm jobs.

Loads the latest checkpoint, plays N games via MCTS, and writes
training examples to an output directory. Designed to be launched
as multiple parallel Slurm jobs.

Usage:
    python -m hexzero.scripts.selfplay_worker \
        --checkpoint checkpoints/latest.pt \
        --output-dir /scr-ssd/hexazero_sp \
        --games 25 \
        --mcts-sims 200 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("selfplay_worker")


def main() -> None:
    parser = argparse.ArgumentParser(description="HexaZero self-play worker")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (None = random init)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write game data")
    parser.add_argument("--games", type=int, default=25,
                        help="Number of games to play")
    parser.add_argument("--mcts-sims", type=int, default=200,
                        help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed (default: Slurm job ID or time)")
    args = parser.parse_args()

    seed = args.seed or int(os.environ.get("SLURM_JOB_ID", int(time.time()) % 100000))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from hexzero.config import get_default_config
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.model.network import HexaZeroNet
    from hexzero.mcts.search import MCTSSearch

    cfg = get_default_config()
    cfg.mcts.num_simulations = args.mcts_sims
    cfg.mcts.num_determinizations = 1
    device = args.device if torch.cuda.is_available() else "cpu"

    if args.checkpoint and Path(args.checkpoint).exists():
        log.info("Loading checkpoint: %s", args.checkpoint)
        net = HexaZeroNet.load_checkpoint(args.checkpoint, device=device)
    else:
        log.info("No checkpoint -- using random initialization")
        net = HexaZeroNet(cfg.network)
        net.to(device)
    net.eval()

    game = CatanGame(seed=seed)
    game.reset()
    state_encoder = game.make_state_encoder()
    action_encoder = ActionEncoder()

    mcts = MCTSSearch(
        network=net,
        encoder=state_encoder,
        action_encoder=action_encoder,
        config=cfg.mcts,
        device=device,
    )

    all_examples = []
    stats = {"games": 0, "total_turns": 0, "winners": [0, 0, 0, 0], "timeouts": 0}

    for gi in range(args.games):
        game_seed = seed * 10000 + gi
        game = CatanGame(seed=game_seed)
        game.reset()

        history = []
        t0 = time.time()

        while not game.is_terminal() and game.turn_number < cfg.selfplay.max_game_length:
            cp = game.current_player()
            sv = game.get_state_view()
            state_tensors = state_encoder.encode(sv)
            state_tensors = {k: v.detach().cpu() for k, v in state_tensors.items()}

            legal = game.get_legal_actions()
            mask = action_encoder.get_action_mask(legal)
            state_tensors["action_masks"] = mask

            result = mcts.search(game)

            temp = (cfg.mcts.temperature_init
                    if game.turn_number < cfg.mcts.temperature_threshold
                    else cfg.mcts.temperature_final)
            action_space_idx = mcts.select_action(result.action_probs, temp, game.turn_number)

            policy = torch.from_numpy(result.action_probs).float()
            history.append((state_tensors, policy, cp))

            legal_idx = 0
            for i, a in enumerate(legal):
                if action_encoder.encode(a) == action_space_idx:
                    legal_idx = i
                    break
            game.step(legal_idx)

        elapsed = time.time() - t0
        winner = game.winner() if game.is_terminal() else None

        examples = _build_examples(history, winner if winner is not None else -1, 4)
        all_examples.extend(examples)

        if winner is not None:
            stats["winners"][winner] += 1
        else:
            stats["timeouts"] += 1
        stats["games"] += 1
        stats["total_turns"] += game.turn_number

        log.info(
            "Game %d/%d: %d turns, winner=%s, %.1fs (%.1f moves/s)",
            gi + 1, args.games, game.turn_number,
            str(winner), elapsed,
            game.turn_number / max(elapsed, 0.001),
        )

        game_file = out_dir / f"game_{game_seed}.pt"
        torch.save(examples, game_file)

    stats_file = out_dir / f"stats_{seed}.json"
    stats_file.write_text(json.dumps(stats, indent=2))
    log.info("Done: %d games, %d examples, stats -> %s", stats["games"], len(all_examples), stats_file)


def _build_examples(history, winner, num_players):
    from hexzero.selfplay.replay_buffer import TrainingExample
    examples = []
    for state_tensors, policy, player_at_turn in history:
        value_target = torch.zeros(num_players, dtype=torch.float32)
        if winner >= 0:
            rotated = (winner - player_at_turn) % num_players
            value_target[rotated] = 1.0
        examples.append(TrainingExample(
            state_tensors=state_tensors,
            policy_target=policy,
            value_target=value_target,
        ))
    return examples


if __name__ == "__main__":
    main()
