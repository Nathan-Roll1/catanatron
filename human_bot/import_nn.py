#!/usr/bin/env python3
"""Import fp32 C-format HumanBotNet weights back into a PyTorch checkpoint.

This is intentionally conservative: it supports the v1 fp32 files emitted by
`export_nn.py`. BatchNorm layers in the C file are stored as fused
scale/shift, so we seed PyTorch BatchNorm with running_mean=0, running_var=1
and affine weight/bias equal to the fused pair.
"""
from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import torch

from human_bot.model import HumanBotNet, SmallNetworkConfig


def _read_i32(f, shape):
    n = int(np.prod(shape))
    data = f.read(n * 4)
    if len(data) != n * 4:
        raise EOFError("unexpected EOF while reading int32 block")
    return np.frombuffer(data, dtype="<i4").reshape(shape).copy()


def _read_f32(f, shape):
    n = int(np.prod(shape))
    data = f.read(n * 4)
    if len(data) != n * 4:
        raise EOFError("unexpected EOF while reading float32 block")
    return torch.from_numpy(np.frombuffer(data, dtype="<f4").reshape(shape).copy())


def _set(sd, name, value):
    sd[name].copy_(value.to(dtype=sd[name].dtype))


def _set_bn_fused(sd, prefix, scale, shift):
    _set(sd, f"{prefix}.weight", scale)
    _set(sd, f"{prefix}.bias", shift)
    if f"{prefix}.running_mean" in sd:
        sd[f"{prefix}.running_mean"].zero_()
    if f"{prefix}.running_var" in sd:
        sd[f"{prefix}.running_var"].fill_(1.0)
    if f"{prefix}.num_batches_tracked" in sd:
        sd[f"{prefix}.num_batches_tracked"].zero_()


def import_checkpoint(input_path: str, checkpoint_path: str) -> None:
    with open(input_path, "rb") as f:
        magic = f.read(4)
        if magic != b"HBOT":
            raise ValueError(f"{input_path} is not an HBOT weights file")
        header = struct.unpack("<15I", f.read(15 * 4))
        (file_ver, num_nodes, num_edges, H, GO, GL, TC, TB, VH, FD, MD,
         NF, EF, PH, SH) = header
        if file_ver != 1:
            raise ValueError(
                f"only fp32 v1 weights are importable, got version {file_ver}")

        cfg = SmallNetworkConfig(
            gnn_layers=GL,
            gnn_hidden_dim=H,
            gnn_output_dim=GO,
            trunk_blocks=TB,
            trunk_channels=TC,
            policy_hidden_dim=PH,
            scorer_hidden_dim=SH,
            value_head_hidden=VH,
            flat_feature_dim=FD,
            action_space_size=MD,
            node_feature_dim=NF,
            edge_feature_dim=EF,
            mask_as_input=True,
        )
        net = HumanBotNet(cfg)
        sd = net.state_dict()

        _read_i32(f, (num_edges, 2))
        road_pairs = _read_i32(f, (72, 2))
        tile_nodes = _read_i32(f, (19, 6))
        _read_i32(f, (54,))
        _read_i32(f, (96,))
        _read_i32(f, (96, 96))
        _read_i32(f, (7, 7, 7))
        _read_i32(f, (5, 5))
        _read_i32(f, (72, 2))
        _set(sd, "policy_head.road_pairs", torch.from_numpy(road_pairs).long())
        _set(sd, "policy_head.tile_nodes", torch.from_numpy(tile_nodes).long())

        _set(sd, "board_encoder.node_proj.0.weight", _read_f32(f, (H, NF)))
        _set(sd, "board_encoder.node_proj.0.bias", _read_f32(f, (H,)))
        _set(sd, "board_encoder.edge_proj.weight", _read_f32(f, (H, EF)))
        _set(sd, "board_encoder.edge_proj.bias", _read_f32(f, (H,)))

        for i in range(GL):
            pre = f"board_encoder.layers.{i}"
            _set(sd, f"{pre}.msg_mlp.0.weight", _read_f32(f, (H, 3 * H)))
            _set(sd, f"{pre}.msg_mlp.0.bias", _read_f32(f, (H,)))
            _set(sd, f"{pre}.msg_mlp.2.weight", _read_f32(f, (H, H)))
            _set(sd, f"{pre}.msg_mlp.2.bias", _read_f32(f, (H,)))
            _set(sd, f"{pre}.update_mlp.0.weight", _read_f32(f, (H, 2 * H)))
            _set(sd, f"{pre}.update_mlp.0.bias", _read_f32(f, (H,)))
            _set(sd, f"{pre}.update_mlp.2.weight", _read_f32(f, (H, H)))
            _set(sd, f"{pre}.update_mlp.2.bias", _read_f32(f, (H,)))
            _set(sd, f"{pre}.norm.weight", _read_f32(f, (H,)))
            _set(sd, f"{pre}.norm.bias", _read_f32(f, (H,)))

        _set(sd, "board_encoder.output_proj.0.weight", _read_f32(f, (GO, 2 * H)))
        _set(sd, "board_encoder.output_proj.0.bias", _read_f32(f, (GO,)))
        _set(sd, "board_encoder.output_proj.2.weight", _read_f32(f, (GO, GO)))
        _set(sd, "board_encoder.output_proj.2.bias", _read_f32(f, (GO,)))

        trunk_input = GO + FD + MD
        _set(sd, "trunk.input_proj.0.weight", _read_f32(f, (TC, trunk_input)))
        _set(sd, "trunk.input_proj.0.bias", _read_f32(f, (TC,)))
        _set_bn_fused(sd, "trunk.input_proj.1", _read_f32(f, (TC,)), _read_f32(f, (TC,)))

        for i in range(TB):
            pre = f"trunk.blocks.{i}"
            _set(sd, f"{pre}.fc1.weight", _read_f32(f, (TC, TC)))
            _set(sd, f"{pre}.fc1.bias", _read_f32(f, (TC,)))
            _set_bn_fused(sd, f"{pre}.bn1", _read_f32(f, (TC,)), _read_f32(f, (TC,)))
            _set(sd, f"{pre}.fc2.weight", _read_f32(f, (TC, TC)))
            _set(sd, f"{pre}.fc2.bias", _read_f32(f, (TC,)))
            _set_bn_fused(sd, f"{pre}.bn2", _read_f32(f, (TC,)), _read_f32(f, (TC,)))

        _set(sd, "value_head.fc1.weight", _read_f32(f, (VH, TC)))
        _set(sd, "value_head.fc1.bias", _read_f32(f, (VH,)))
        _set_bn_fused(sd, "value_head.bn1", _read_f32(f, (VH,)), _read_f32(f, (VH,)))
        for r in ("res1", "res2"):
            pre = f"value_head.{r}"
            _set(sd, f"{pre}.fc1.weight", _read_f32(f, (VH, VH)))
            _set(sd, f"{pre}.fc1.bias", _read_f32(f, (VH,)))
            _set_bn_fused(sd, f"{pre}.bn1", _read_f32(f, (VH,)), _read_f32(f, (VH,)))
            _set(sd, f"{pre}.fc2.weight", _read_f32(f, (VH, VH)))
            _set(sd, f"{pre}.fc2.bias", _read_f32(f, (VH,)))
            _set_bn_fused(sd, f"{pre}.bn2", _read_f32(f, (VH,)), _read_f32(f, (VH,)))
        _set(sd, "value_head.fc_out.weight", _read_f32(f, (4, VH)))
        _set(sd, "value_head.fc_out.bias", _read_f32(f, (4,)))

        _set(sd, "policy_head.trunk_norm.weight", _read_f32(f, (TC,)))
        _set(sd, "policy_head.trunk_norm.bias", _read_f32(f, (TC,)))
        _set(sd, "policy_head.node_norm.weight", _read_f32(f, (H,)))
        _set(sd, "policy_head.node_norm.bias", _read_f32(f, (H,)))
        _set(sd, "policy_head.type_fc.0.weight", _read_f32(f, (PH, TC)))
        _set(sd, "policy_head.type_fc.0.bias", _read_f32(f, (PH,)))
        _set_bn_fused(sd, "policy_head.type_fc.1", _read_f32(f, (PH,)), _read_f32(f, (PH,)))
        _set(sd, "policy_head.type_fc.3.weight", _read_f32(f, (12, PH)))
        _set(sd, "policy_head.type_fc.3.bias", _read_f32(f, (12,)))

        for name, out_dim in (("discard_yop_mono_fc", 30),
                              ("maritime_fc", 20),
                              ("trade_fc", 67)):
            pre = f"policy_head.{name}"
            _set(sd, f"{pre}.0.weight", _read_f32(f, (SH, TC)))
            _set(sd, f"{pre}.0.bias", _read_f32(f, (SH,)))
            _set(sd, f"{pre}.2.weight", _read_f32(f, (out_dim, SH)))
            _set(sd, f"{pre}.2.bias", _read_f32(f, (out_dim,)))

        for name, in_dim, out_dim in (
            ("settlement_scorer", TC + H, 1),
            ("city_scorer", TC + H, 1),
            ("road_scorer", TC + 2 * H, 1),
            ("robber_scorer", TC + H, 5),
        ):
            pre = f"policy_head.{name}"
            _set(sd, f"{pre}.0.weight", _read_f32(f, (SH, in_dim)))
            _set(sd, f"{pre}.0.bias", _read_f32(f, (SH,)))
            _set(sd, f"{pre}.2.weight", _read_f32(f, (out_dim, SH)))
            _set(sd, f"{pre}.2.bias", _read_f32(f, (out_dim,)))

        extra = f.read(1)
        if extra:
            raise ValueError("unparsed trailing bytes in weights file")

    net.load_state_dict(sd, strict=True)
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    net.save_checkpoint(checkpoint_path, {
        "source": "import_nn",
        "input": os.path.abspath(input_path),
        "note": "Imported from fp32 C weights; BatchNorm seeded from fused scale/shift.",
    })
    print(f"Imported {input_path} -> {checkpoint_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    import_checkpoint(args.input, args.checkpoint)


if __name__ == "__main__":
    main()
