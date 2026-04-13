#!/usr/bin/env python3
"""Export HumanBotNet weights + topology to a flat binary file for C inference.

Binary format (all little-endian):
  Header: magic(4B) version(u32) then architecture constants (u32 each)
  Topology: edge_index, road_pairs, tile_nodes, land_nodes (int32 arrays)
  Weights: all parameters + BN running stats as contiguous float32 blocks.
           BatchNorm is fused to (scale, shift) for eval-mode inference.

Usage:
    python -m human_bot.export_nn --checkpoint path/to/model.pt --output csrc/nn_weights.bin
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np
import torch

from human_bot.model import HumanBotNet


def fuse_bn(weight, bias, running_mean, running_var, eps=1e-5):
    """Fuse BatchNorm into (scale, shift) for eval-mode: y = x*scale + shift."""
    scale = weight / torch.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale.numpy(), shift.numpy()


def write_f32(f, tensor):
    """Write a tensor as contiguous float32 bytes."""
    arr = tensor.detach().cpu().float().numpy()
    f.write(arr.tobytes())


def write_i32(f, arr):
    """Write an int32 numpy array."""
    f.write(np.ascontiguousarray(arr, dtype=np.int32).tobytes())


def export(checkpoint_path: str, output_path: str):
    net = HumanBotNet.load_checkpoint(checkpoint_path, device="cpu")
    net.eval()
    sd = net.state_dict()

    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder

    g = CatanGame(seed=0); g.reset()
    se = g.make_state_encoder()
    ae = ActionEncoder()

    ei = se._edge_index.numpy().astype(np.int32)  # (2, E)
    num_nodes = se.num_nodes          # 54
    num_edges = ei.shape[1]           # 144
    road_pairs = sd["policy_head.road_pairs"].numpy().astype(np.int32)  # (72, 2)
    tile_nodes = sd["policy_head.tile_nodes"].numpy().astype(np.int32)  # (19, 6)
    land_nodes = se._land.astype(np.int32)  # (54,) global node IDs

    cfg = net.config
    H = cfg.gnn_hidden_dim       # 64
    GO = cfg.gnn_output_dim      # 128
    GL = cfg.gnn_layers          # 4
    TC = cfg.trunk_channels      # 128
    TB = cfg.trunk_blocks        # 6
    VH = cfg.value_head_hidden   # 128
    FD = cfg.flat_feature_dim    # 115
    MD = cfg.action_space_size   # 397
    NF = cfg.node_feature_dim    # 18
    EF = cfg.edge_feature_dim    # 5
    PH = cfg.policy_hidden_dim   # 128
    SH = cfg.scorer_hidden_dim   # 48

    # Node/edge mapping tables for action encoder
    node_to_compact = ae._node_to_compact.astype(np.int32)  # (96,)
    edge_lut = ae._edge_lut.astype(np.int32)                # (96, 96)
    coord_to_tile = np.full((7, 7, 7), -1, dtype=np.int32)  # offset by 3
    for (x, y, z), ti in ae._coord_to_tile.items():
        coord_to_tile[x+3, y+3, z+3] = ti
    mar_lut = ae._mar_lut.astype(np.int32)  # (5, 5)
    idx_to_edge = ae._idx_to_edge.astype(np.int32)  # (72, 2)

    with open(output_path, "wb") as f:
        # -- Header --
        f.write(b"HBOT")
        for v in [1, num_nodes, num_edges, H, GO, GL, TC, TB, VH, FD, MD, NF, EF, PH, SH]:
            f.write(struct.pack("<I", v))

        # -- Topology --
        write_i32(f, ei.T)          # (E, 2) src,dst pairs
        write_i32(f, road_pairs)    # (72, 2)
        write_i32(f, tile_nodes)    # (19, 6)
        write_i32(f, land_nodes)    # (54,)
        write_i32(f, node_to_compact)  # (96,)
        write_i32(f, edge_lut)      # (96, 96)
        write_i32(f, coord_to_tile) # (7, 7, 7)
        write_i32(f, mar_lut)       # (5, 5)
        write_i32(f, idx_to_edge)   # (72, 2)

        # -- GNN weights --
        # node_proj: Linear(NF, H) + Mish
        write_f32(f, sd["board_encoder.node_proj.0.weight"])  # (H, NF)
        write_f32(f, sd["board_encoder.node_proj.0.bias"])    # (H,)
        # edge_proj: Linear(EF, H)
        write_f32(f, sd["board_encoder.edge_proj.weight"])    # (H, EF)
        write_f32(f, sd["board_encoder.edge_proj.bias"])      # (H,)

        # 4 EdgeConvLayers
        for i in range(GL):
            pre = f"board_encoder.layers.{i}"
            # msg_mlp: Linear(3H, H) + Mish + Linear(H, H) + Mish
            write_f32(f, sd[f"{pre}.msg_mlp.0.weight"])   # (H, 3H)
            write_f32(f, sd[f"{pre}.msg_mlp.0.bias"])     # (H,)
            write_f32(f, sd[f"{pre}.msg_mlp.2.weight"])   # (H, H)
            write_f32(f, sd[f"{pre}.msg_mlp.2.bias"])     # (H,)
            # update_mlp: Linear(2H, H) + Mish + Linear(H, H)
            write_f32(f, sd[f"{pre}.update_mlp.0.weight"])  # (H, 2H)
            write_f32(f, sd[f"{pre}.update_mlp.0.bias"])    # (H,)
            write_f32(f, sd[f"{pre}.update_mlp.2.weight"])  # (H, H)
            write_f32(f, sd[f"{pre}.update_mlp.2.bias"])    # (H,)
            # LayerNorm(H)
            write_f32(f, sd[f"{pre}.norm.weight"])          # (H,)
            write_f32(f, sd[f"{pre}.norm.bias"])            # (H,)

        # output_proj: Linear(2H, GO) + Mish + Linear(GO, GO)
        write_f32(f, sd["board_encoder.output_proj.0.weight"])  # (GO, 2H)
        write_f32(f, sd["board_encoder.output_proj.0.bias"])    # (GO,)
        write_f32(f, sd["board_encoder.output_proj.2.weight"])  # (GO, GO)
        write_f32(f, sd["board_encoder.output_proj.2.bias"])    # (GO,)

        # -- Trunk weights (fused BN) --
        # input_proj: Linear(640, TC) + BN(TC) + Mish
        write_f32(f, sd["trunk.input_proj.0.weight"])  # (TC, 640)
        write_f32(f, sd["trunk.input_proj.0.bias"])    # (TC,)
        s, sh = fuse_bn(sd["trunk.input_proj.1.weight"], sd["trunk.input_proj.1.bias"],
                        sd["trunk.input_proj.1.running_mean"], sd["trunk.input_proj.1.running_var"])
        f.write(s.tobytes()); f.write(sh.tobytes())

        # 6 ResBlocks: fc1 + bn1 + fc2 + bn2
        for i in range(TB):
            pre = f"trunk.blocks.{i}"
            write_f32(f, sd[f"{pre}.fc1.weight"])
            write_f32(f, sd[f"{pre}.fc1.bias"])
            s, sh = fuse_bn(sd[f"{pre}.bn1.weight"], sd[f"{pre}.bn1.bias"],
                            sd[f"{pre}.bn1.running_mean"], sd[f"{pre}.bn1.running_var"])
            f.write(s.tobytes()); f.write(sh.tobytes())
            write_f32(f, sd[f"{pre}.fc2.weight"])
            write_f32(f, sd[f"{pre}.fc2.bias"])
            s, sh = fuse_bn(sd[f"{pre}.bn2.weight"], sd[f"{pre}.bn2.bias"],
                            sd[f"{pre}.bn2.running_mean"], sd[f"{pre}.bn2.running_var"])
            f.write(s.tobytes()); f.write(sh.tobytes())

        # -- Value head (fused BN) --
        write_f32(f, sd["value_head.fc1.weight"])
        write_f32(f, sd["value_head.fc1.bias"])
        s, sh = fuse_bn(sd["value_head.bn1.weight"], sd["value_head.bn1.bias"],
                        sd["value_head.bn1.running_mean"], sd["value_head.bn1.running_var"])
        f.write(s.tobytes()); f.write(sh.tobytes())
        for r in ["res1", "res2"]:
            pre = f"value_head.{r}"
            write_f32(f, sd[f"{pre}.fc1.weight"])
            write_f32(f, sd[f"{pre}.fc1.bias"])
            s, sh = fuse_bn(sd[f"{pre}.bn1.weight"], sd[f"{pre}.bn1.bias"],
                            sd[f"{pre}.bn1.running_mean"], sd[f"{pre}.bn1.running_var"])
            f.write(s.tobytes()); f.write(sh.tobytes())
            write_f32(f, sd[f"{pre}.fc2.weight"])
            write_f32(f, sd[f"{pre}.fc2.bias"])
            s, sh = fuse_bn(sd[f"{pre}.bn2.weight"], sd[f"{pre}.bn2.bias"],
                            sd[f"{pre}.bn2.running_mean"], sd[f"{pre}.bn2.running_var"])
            f.write(s.tobytes()); f.write(sh.tobytes())
        write_f32(f, sd["value_head.fc_out.weight"])  # (4, VH)
        write_f32(f, sd["value_head.fc_out.bias"])    # (4,)

        # -- Policy head --
        # trunk_norm (LayerNorm)
        write_f32(f, sd["policy_head.trunk_norm.weight"])
        write_f32(f, sd["policy_head.trunk_norm.bias"])
        # node_norm (LayerNorm)
        write_f32(f, sd["policy_head.node_norm.weight"])
        write_f32(f, sd["policy_head.node_norm.bias"])
        # type_fc: Linear(TC, PH) + BN(PH) + Mish + Linear(PH, 12)
        write_f32(f, sd["policy_head.type_fc.0.weight"])
        write_f32(f, sd["policy_head.type_fc.0.bias"])
        s, sh = fuse_bn(sd["policy_head.type_fc.1.weight"], sd["policy_head.type_fc.1.bias"],
                        sd["policy_head.type_fc.1.running_mean"], sd["policy_head.type_fc.1.running_var"])
        f.write(s.tobytes()); f.write(sh.tobytes())
        write_f32(f, sd["policy_head.type_fc.3.weight"])
        write_f32(f, sd["policy_head.type_fc.3.bias"])
        # sub-action heads
        for name in ["discard_yop_mono_fc", "maritime_fc", "trade_fc"]:
            write_f32(f, sd[f"policy_head.{name}.0.weight"])
            write_f32(f, sd[f"policy_head.{name}.0.bias"])
            write_f32(f, sd[f"policy_head.{name}.2.weight"])
            write_f32(f, sd[f"policy_head.{name}.2.bias"])
        # spatial scorers
        for name in ["settlement_scorer", "city_scorer", "road_scorer", "robber_scorer"]:
            write_f32(f, sd[f"policy_head.{name}.0.weight"])
            write_f32(f, sd[f"policy_head.{name}.0.bias"])
            write_f32(f, sd[f"policy_head.{name}.2.weight"])
            write_f32(f, sd[f"policy_head.{name}.2.bias"])

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported {output_path} ({size_kb:.0f} KB)")

    # -- Write test vectors for C verification --
    test_path = output_path.replace(".bin", "_test.bin")
    game = CatanGame(seed=42); game.reset()
    for _ in range(20):
        le = game.get_legal_actions()
        if not le: break
        game.step(0)
    nf = np.zeros((1, num_nodes, NF), dtype=np.float32)
    ef = np.zeros((1, num_edges, EF), dtype=np.float32)
    ff = np.zeros((1, FD), dtype=np.float32)
    se.encode_into(game.get_state_view(), nf[0], ef[0], ff[0])
    le = game.get_legal_actions()
    mask_np = ae.get_action_mask(le).numpy()
    mask_397 = np.zeros(MD, dtype=np.float32)
    mask_397[:len(mask_np)] = mask_np

    with torch.no_grad():
        out = net({
            "node_features": torch.from_numpy(nf),
            "edge_index": se._edge_index,
            "edge_features": torch.from_numpy(ef),
            "flat_features": torch.from_numpy(ff),
            "action_mask": torch.from_numpy(mask_397).unsqueeze(0),
        })
    value = out["value"][0].numpy()
    logits = out["policy_logits"][0].numpy()

    with open(test_path, "wb") as f:
        f.write(nf.tobytes())
        f.write(ef.tobytes())
        f.write(ff.tobytes())
        f.write(mask_397.tobytes())
        f.write(value.tobytes())
        f.write(logits.tobytes())
    print(f"Test vectors: {test_path}")
    print(f"  value = {value.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="csrc/nn_weights.bin")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
