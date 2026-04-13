#!/usr/bin/env python3
"""Phase 3 self-play actor for the no-lookahead pipeline.

Runs batches of 2 HZ + 2 AB2 games on a single GPU, writing completed step
batches to a shared trajectory directory on local SSD. The learner consumes
those files asynchronously.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch


def _wait_for_checkpoint(path: str, timeout_s: int = 300) -> None:
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for checkpoint: {path}")
        time.sleep(1.0)


def _maybe_reload(path: str, device: str, net, last_mtime: float):
    from hexzero.model.network import HexaZeroNet

    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return net, last_mtime, False

    if net is None or mtime > last_mtime:
        try:
            net = HexaZeroNet.load_checkpoint(path, device=device)
            net.eval()
            return net, mtime, True
        except Exception as e:
            print(f"[reload] Failed to load {path}: {e}", flush=True)
            if net is not None:
                return net, last_mtime, False
            time.sleep(2.0)
            return net, last_mtime, False
    return net, last_mtime, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="No-lookahead self-play actor (2 HZ + 2 AB2)"
    )
    parser.add_argument("--actor-id", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--trajectory-dir", type=str, required=True)
    parser.add_argument("--games-per-batch", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top policy actions to evaluate with 1-ply search")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--reload-every", type=int, default=2)
    parser.add_argument("--max-pending", type=int, default=200,
                        help="Stop writing if this many trajectory files are pending")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.set_num_threads(1)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from hexzero.game.interface import CatanGame
    from hexzero.scripts.selfplay_train import detect_device, play_batch_exit

    os.makedirs(args.trajectory_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pt")
    device = detect_device(args.device)

    tmp = CatanGame(seed=0)
    tmp.reset()
    state_enc = tmp.make_state_encoder()
    action_enc = ActionEncoder()
    lib = load_library()
    edge_index_dev = state_enc._edge_index.to(device)

    print(
        f"[actor {args.actor_id}] device={device} games/batch={args.games_per_batch} "
        f"traj_dir={args.trajectory_dir}",
        flush=True,
    )
    print(f"[actor {args.actor_id}] waiting for {ckpt_path}", flush=True)
    _wait_for_checkpoint(ckpt_path)

    net = None
    ckpt_mtime = 0.0
    total_games = 0
    batch_id = 0
    seed_base = args.actor_id * 10_000_000 + int(time.time()) % 10_000_000
    t_start = time.time()

    while args.max_batches == 0 or batch_id < args.max_batches:
        if batch_id % max(args.reload_every, 1) == 0:
            net, ckpt_mtime, reloaded = _maybe_reload(
                ckpt_path, device, net, ckpt_mtime
            )
            if reloaded:
                print(
                    f"[actor {args.actor_id}] reloaded checkpoint "
                    f"(mtime={int(ckpt_mtime)})",
                    flush=True,
                )

        t0 = time.time()
        steps = play_batch_exit(
            net=net,
            state_enc=state_enc,
            action_enc=action_enc,
            lib=lib,
            device=device,
            edge_index_dev=edge_index_dev,
            num_games=args.games_per_batch,
            seed_base=seed_base + total_games,
            top_k=args.top_k,
            temperature=args.temperature,
        )

        # Backpressure: wait if too many unprocessed files on disk
        pending = len(glob.glob(os.path.join(args.trajectory_dir, "actor*_batch*.pt")))
        if pending >= args.max_pending:
            print(
                f"[actor {args.actor_id}] backpressure: {pending} pending files, "
                f"waiting for learner to catch up",
                flush=True,
            )
            while pending >= args.max_pending:
                time.sleep(5.0)
                pending = len(glob.glob(
                    os.path.join(args.trajectory_dir, "actor*_batch*.pt")))

        payload = {
            "steps": steps,
            "num_games": args.games_per_batch,
            "actor_id": args.actor_id,
            "batch_id": batch_id,
        }
        final_path = os.path.join(
            args.trajectory_dir,
            f"actor{args.actor_id}_batch{batch_id:06d}.pt",
        )
        tmp_path = f"{final_path}.tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, final_path)
        except (RuntimeError, OSError) as e:
            print(f"[actor {args.actor_id}] write failed ({e}), skipping batch",
                  flush=True)
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            time.sleep(5.0)

        total_games += args.games_per_batch
        batch_id += 1

        elapsed = time.time() - t_start
        gps = total_games / max(elapsed, 0.01)
        print(
            f"[actor {args.actor_id}] batch={batch_id} "
            f"{args.games_per_batch} games {len(steps)} steps "
            f"top_k={args.top_k} | "
            f"{time.time() - t0:.1f}s | total={total_games} games ({gps:.1f} g/s)",
            flush=True,
        )

    print(
        f"[actor {args.actor_id}] done: {batch_id} batches, {total_games} games",
        flush=True,
    )


if __name__ == "__main__":
    main()
