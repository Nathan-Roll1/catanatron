"""Policy-only inference model: stripped value head, int8 quantized.

Loads a full HumanBotNet checkpoint, removes value_head and vp_head,
applies dynamic int8 quantization to all Linear layers, and exports
a compact .pt file (~250KB vs ~4MB full).

Usage:
    # Export once:
    python -m human_bot.policy_net --checkpoint checkpoints/exit_v2/latest.pt \
        --output checkpoints/policy_int8.pt

    # Use in code:
    from human_bot.policy_net import PolicyNet
    net = PolicyNet.load("checkpoints/policy_int8.pt")
    action_idx = net.pick(game, state_encoder, action_encoder)
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


AD = 337
MASK_DIM = 397


class PolicyNet(nn.Module):
    """Policy-only inference wrapper. No value head, int8 quantized."""

    def __init__(self, board_encoder, trunk, policy_head, config):
        super().__init__()
        self.board_encoder = board_encoder
        self.trunk = trunk
        self.policy_head = policy_head
        self.config = config
        self._edge_index = None

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        nf = batch["node_features"]
        ef = batch["edge_features"]
        ff = batch["flat_features"]
        mask = batch.get("action_mask")

        param_dtype = next(self.parameters()).dtype
        if param_dtype != nf.dtype:
            nf = nf.to(param_dtype)
            ef = ef.to(param_dtype)
            ff = ff.to(param_dtype)
            if mask is not None:
                mask = mask.to(param_dtype)

        board_emb, node_emb = self.board_encoder(
            nf, batch["edge_index"], ef)
        parts = [board_emb, ff]
        if self.config.mask_as_input and mask is not None:
            parts.append(mask)
        combined = torch.cat(parts, dim=-1)
        trunk_out = self.trunk(combined)
        raw_logits = self.policy_head(trunk_out, node_emb)

        if mask is not None:
            masked_logits = raw_logits.float().masked_fill(~mask.bool(), -1e9)
        else:
            masked_logits = raw_logits.float()
        return masked_logits

    @torch.inference_mode()
    def pick(self, game, se, ae, device="cpu") -> int:
        """Play one move: encode state, forward, return legal action index."""
        le = game.get_legal_actions()
        if not le:
            return -1
        if len(le) == 1:
            return 0

        N, E = se.num_nodes, se.num_edges
        nf = np.zeros((1, N, se.NODE_FEATURE_DIM), dtype=np.float32)
        ef = np.zeros((1, E, se.EDGE_FEATURE_DIM), dtype=np.float32)
        ff = np.zeros((1, se.FLAT_FEATURE_DIM), dtype=np.float32)
        se.encode_into(game.get_state_view(), nf[0], ef[0], ff[0])
        mn = ae.get_action_mask(le).numpy()
        mk = np.zeros((1, MASK_DIM), dtype=np.float32)
        mk[0, :len(mn)] = mn

        if self._edge_index is None:
            self._edge_index = se._edge_index.to(device)

        logits = self({
            "node_features": torch.from_numpy(nf).to(device),
            "edge_index": self._edge_index,
            "edge_features": torch.from_numpy(ef).to(device),
            "flat_features": torch.from_numpy(ff).to(device),
            "action_mask": torch.from_numpy(mk).to(device),
        })

        lo = logits[0, :AD].cpu().numpy()
        lo[mn[:AD] < 0.5] = -1e9
        best_enc = int(np.argmax(lo))
        for i, a in enumerate(le):
            try:
                if ae.encode(a) == best_enc:
                    return i
            except ValueError:
                continue
        return 0

    def save(self, path: str) -> None:
        torch.save({
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
            "model_type": "PolicyNet_int8",
        }, path)
        sz = os.path.getsize(path)
        print(f"Saved {path} ({sz / 1024:.0f} KB)")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> PolicyNet:
        from human_bot.model import SmallNetworkConfig, SmallBoardEncoder, SmallTrunk, SmallSpatialPolicyHead
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = SmallNetworkConfig(**ckpt["config"])

        road_pairs, tile_nodes = _compute_topology()
        net = cls(
            board_encoder=SmallBoardEncoder(config),
            trunk=SmallTrunk(
                config.gnn_output_dim + config.flat_feature_dim +
                (config.action_space_size if config.mask_as_input else 0),
                config.trunk_channels, config.trunk_blocks),
            policy_head=SmallSpatialPolicyHead(config, road_pairs, tile_nodes),
            config=config,
        )
        net.load_state_dict(ckpt["model_state_dict"], strict=False)
        net.to(device)
        net.eval()
        return net

    @classmethod
    def from_full_checkpoint(cls, path: str, device: str = "cpu") -> PolicyNet:
        """Load a full HumanBotNet checkpoint, strip value heads, half-precision."""
        from human_bot.model import HumanBotNet
        full = HumanBotNet.load_checkpoint(path, device="cpu")
        full.eval()

        net = cls(
            board_encoder=full.board_encoder,
            trunk=full.trunk,
            policy_head=full.policy_head,
            config=full.config,
        )
        net.eval()

        full_params = sum(p.numel() for p in full.parameters())
        policy_params = sum(p.numel() for p in net.parameters())
        value_params = full_params - policy_params
        print(f"Full model:    {full_params:,} params")
        print(f"Policy only:   {policy_params:,} params ({value_params:,} removed)")

        use_int8 = False
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                net, {nn.Linear}, dtype=torch.qint8)
            quantized.eval()
            net = quantized
            use_int8 = True
            print(f"Quantized to int8 (Linear layers)")
        except RuntimeError:
            for p in net.parameters():
                p.data = p.data.half()
            print(f"Converted to float16 (int8 not available on this platform)")

        net.to(device)
        return net


def _compute_topology():
    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    ae = ActionEncoder()
    g = CatanGame(seed=0); g.reset()
    se = g.make_state_encoder()
    full_to_compact = {int(fi): ci for ci, fi in enumerate(se._land)}
    road_pairs = torch.zeros(72, 2, dtype=torch.long)
    for i in range(72):
        a, b = ae._idx_to_edge[i]
        road_pairs[i, 0] = full_to_compact[int(a)]
        road_pairs[i, 1] = full_to_compact[int(b)]
    tile_nodes = torch.from_numpy(se._ltiles.copy()).long()
    return road_pairs, tile_nodes


def main():
    parser = argparse.ArgumentParser(
        description="Export policy-only int8 quantized model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Full HumanBotNet checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <ckpt_dir>/policy_int8.pt)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run a quick inference benchmark after export")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(args.checkpoint), "policy_int8.pt")

    net = PolicyNet.from_full_checkpoint(args.checkpoint)
    net.save(args.output)

    # Verify roundtrip
    net2 = PolicyNet.load(args.output)
    print(f"Reload OK: {sum(1 for _ in net2.parameters())} param tensors")

    if args.benchmark:
        import time
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder
        from hexzero.bindings.lib_loader import load_library
        load_library()
        ae = ActionEncoder()
        g = CatanGame(seed=42); g.reset()
        se = g.make_state_encoder()

        # Warmup
        for _ in range(5):
            net2.pick(g, se, ae)

        t0 = time.perf_counter()
        N = 100
        for _ in range(N):
            net2.pick(g, se, ae)
        dt = (time.perf_counter() - t0) / N * 1000
        print(f"Benchmark: {dt:.1f} ms/inference ({1000/dt:.0f} infer/sec)")

        # Compare with full model
        from human_bot.model import HumanBotNet
        full = HumanBotNet.load_checkpoint(args.checkpoint)
        full.eval()
        for _ in range(5):
            nf = np.zeros((1, se.num_nodes, se.NODE_FEATURE_DIM), np.float32)
            ef = np.zeros((1, se.num_edges, se.EDGE_FEATURE_DIM), np.float32)
            ff = np.zeros((1, se.FLAT_FEATURE_DIM), np.float32)
            se.encode_into(g.get_state_view(), nf[0], ef[0], ff[0])
            mk = np.zeros((1, MASK_DIM), np.float32)
            with torch.no_grad():
                full({"node_features": torch.from_numpy(nf),
                      "edge_index": se._edge_index,
                      "edge_features": torch.from_numpy(ef),
                      "flat_features": torch.from_numpy(ff),
                      "action_mask": torch.from_numpy(mk)})

        t0 = time.perf_counter()
        for _ in range(N):
            with torch.no_grad():
                full({"node_features": torch.from_numpy(nf),
                      "edge_index": se._edge_index,
                      "edge_features": torch.from_numpy(ef),
                      "flat_features": torch.from_numpy(ff),
                      "action_mask": torch.from_numpy(mk)})
        dt_full = (time.perf_counter() - t0) / N * 1000
        print(f"Full model:    {dt_full:.1f} ms/inference")
        print(f"Speedup:       {dt_full/dt:.1f}x")

        sz_full = os.path.getsize(args.checkpoint) / 1024
        sz_q = os.path.getsize(args.output) / 1024
        print(f"Size:          {sz_full:.0f} KB -> {sz_q:.0f} KB ({sz_full/sz_q:.1f}x smaller)")


if __name__ == "__main__":
    main()
