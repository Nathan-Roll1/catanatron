"""One-shot diagnostic: figure out why a checkpoint looks weak in the new
eval harness.

Prints:
  - libcatan path + md5 of the checkpoint
  - M2 inference output on a fixed mid-game 4p state
  - 50-game 0-ply WR vs proper-AB2 at two fresh seed ranges

Run on the cluster with:
  cd /nlp/scr/nroll/catan_training_big
  PYTHONPATH=$PWD python3 -u human_bot/_diag_m2.py [CKPT_PATH]

Defaults to checkpoints/exit_v2/init.pt so it actually checks the file
the learner would seed from.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import torch


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    from hexzero.config import GameConfig
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.bindings.lib_loader import load_library
    from human_bot.eval_search import evaluate_search_vs_ab2
    from human_bot.model import HumanBotNet

    print("=" * 60)
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/exit_v2/init.pt"
    if not os.path.exists(ckpt_path):
        print(f"!!! No file at {ckpt_path}")
        return
    sz_mb = os.path.getsize(ckpt_path) / 1e6
    with open(ckpt_path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print(f"Checkpoint:  {ckpt_path}")
    print(f"  size:      {sz_mb:.2f} MB")
    print(f"  md5:       {md5}")

    lib = load_library()
    print(f"  libcatan:  {lib._name}")
    print()

    ae = ActionEncoder()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = HumanBotNet.load_checkpoint(ckpt_path, device=device)
    net.eval()
    print(f"Loaded:      {net.num_parameters:,} params on {device}")

    # Build a deterministic 4p state ~30 turns in
    g = CatanGame(seed=12345, config=GameConfig(num_players=4))
    g.reset()
    se = g.make_state_encoder()
    for _ in range(30):
        if g.is_terminal():
            break
        le = g.get_legal_actions()
        if not le:
            break
        g.step(0)

    sv = g.get_state_view()
    nf = np.zeros((se.num_nodes, 18), np.float32)
    ef = np.zeros((se.num_edges, 5), np.float32)
    ff = np.zeros(115, np.float32)
    se.encode_into(sv, nf, ef, ff)
    print()
    print(f"Mid-game probe: cp={sv.current_player}, n_players={sv.num_players}, "
          f"turn={g.turn_number}")
    print(f"  ef channel sums: {ef.sum(axis=0).tolist()}")
    print(f"  flat per-player block sums: "
          f"{[round(float(ff[24*i:24*(i+1)].sum()), 3) for i in range(4)]}")
    print(f"  nf[:,5:10].sum()={float(nf[:, 5:10].sum()):.2f}  "
          f"nf[:,17].sum()={float(nf[:, 17].sum()):.2f}")

    le = g.get_legal_actions()
    mk = np.zeros(397, np.float32)
    mn = ae.get_action_mask(le).numpy()
    mk[:len(mn)] = mn

    with torch.no_grad():
        out = net({
            "node_features": torch.from_numpy(nf[None]).to(device),
            "edge_index": se._edge_index.to(device),
            "edge_features": torch.from_numpy(ef[None]).to(device),
            "flat_features": torch.from_numpy(ff[None]).to(device),
            "action_mask": torch.from_numpy(mk[None]).to(device),
        })
    top5 = out["policy_logits"][0, :337].topk(5)
    print(f"  top-5 action logits: "
          f"{[round(float(v), 3) for v in top5.values.cpu().tolist()]}")
    print(f"  top-5 action ids:    {top5.indices.cpu().tolist()}")
    print(f"  value logits:        "
          f"{[round(float(v), 3) for v in out['value'][0].cpu().tolist()]}")
    print()

    print("Eval at 50 games (seed_offset=9, search_depth=0):")
    r = evaluate_search_vs_ab2(net, se, ae, device, lib,
        num_games=50, search_depth=0, seed_offset=9)
    print(f"  WR = {r['win_rate']:.1%}  "
          f"(NN={r['hz_wins']}, AB2={r['ab2_wins']})")

    print()
    print("Eval at 50 games (seed_offset=99, search_depth=0):")
    r2 = evaluate_search_vs_ab2(net, se, ae, device, lib,
        num_games=50, search_depth=0, seed_offset=99)
    print(f"  WR = {r2['win_rate']:.1%}  "
          f"(NN={r2['hz_wins']}, AB2={r2['ab2_wins']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
