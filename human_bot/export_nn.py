#!/usr/bin/env python3
"""Export HumanBotNet weights + topology to a flat binary file for C inference.

Binary format (all little-endian):
  Header: magic(4B) version(u32): 1=fp32, 2=fp16, 3=symmetric int8 + f32 scale per `read_tensor_block`
          in `nn_load` (one block for GNN/Res/val_res *structs*; v2/v3 expand to fp32 for math)
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


def write_f16(f, tensor):
    """Write a tensor as float16 (file version 2); C nn_load expands to fp32."""
    arr = tensor.detach().cpu().float().numpy().astype(np.float16)
    f.write(arr.tobytes())


def write_int8_symmetric(f, arr_f32: np.ndarray):
    """Per-tensor minmax int8 + float32 scale (file version 3); C nn_load expands to fp32."""
    arr = np.asarray(arr_f32, dtype=np.float32).ravel()
    amax = float(np.max(np.abs(arr))) if arr.size else 0.0
    if amax < 1e-20:
        scale = 1.0
        q = np.zeros(arr.shape[0], dtype=np.int8)
    else:
        scale = amax / 127.0
        q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    f.write(q.tobytes())
    f.write(struct.pack("<f", scale))


def write_i32(f, arr):
    """Write an int32 numpy array."""
    f.write(np.ascontiguousarray(arr, dtype=np.int32).tobytes())


def export(
    checkpoint_path: str,
    output_path: str,
    weight_format: str = "fp32",
    write_test_vectors: bool = True,
):
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

    wf = weight_format.lower()
    if wf not in ("fp32", "fp16", "int8"):
        raise ValueError("weight_format must be fp32, fp16, or int8")
    file_ver = 3 if wf == "int8" else (2 if wf == "fp16" else 1)

    def w_tensor(fh, tensor):
        if wf == "int8":
            write_int8_symmetric(fh, tensor.detach().cpu().float().numpy())
        elif wf == "fp16":
            write_f16(fh, tensor)
        else:
            write_f32(fh, tensor)

    def w_fused(fh, s, sh):
        if wf == "int8":
            a = np.concatenate(
                [np.asarray(s, dtype=np.float32).ravel(), np.asarray(sh, dtype=np.float32).ravel()]
            )
            write_int8_symmetric(fh, a)
        elif wf == "fp16":
            fh.write(np.asarray(s, dtype=np.float16).tobytes())
            fh.write(np.asarray(sh, dtype=np.float16).tobytes())
        else:
            fh.write(s.tobytes())
            fh.write(sh.tobytes())

    with open(output_path, "wb") as f:
        # -- Header --
        f.write(b"HBOT")
        for v in [file_ver, num_nodes, num_edges, H, GO, GL, TC, TB, VH, FD, MD, NF, EF, PH, SH]:
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
        w_tensor(f, sd["board_encoder.node_proj.0.weight"])  # (H, NF)
        w_tensor(f, sd["board_encoder.node_proj.0.bias"])    # (H,)
        # edge_proj: Linear(EF, H)
        w_tensor(f, sd["board_encoder.edge_proj.weight"])    # (H, EF)
        w_tensor(f, sd["board_encoder.edge_proj.bias"])      # (H,)

        # 4 EdgeConvLayers — C loads one EdgeConvWeights struct per layer (see nn.c)
        for i in range(GL):
            pre = f"board_encoder.layers.{i}"
            if wf == "int8":
                tns = [
                    sd[f"{pre}.msg_mlp.0.weight"],
                    sd[f"{pre}.msg_mlp.0.bias"],
                    sd[f"{pre}.msg_mlp.2.weight"],
                    sd[f"{pre}.msg_mlp.2.bias"],
                    sd[f"{pre}.update_mlp.0.weight"],
                    sd[f"{pre}.update_mlp.0.bias"],
                    sd[f"{pre}.update_mlp.2.weight"],
                    sd[f"{pre}.update_mlp.2.bias"],
                    sd[f"{pre}.norm.weight"],
                    sd[f"{pre}.norm.bias"],
                ]
                arr = np.concatenate(
                    [x.detach().cpu().float().numpy().ravel() for x in tns]
                )
                write_int8_symmetric(f, arr)
            else:
                w_tensor(f, sd[f"{pre}.msg_mlp.0.weight"])
                w_tensor(f, sd[f"{pre}.msg_mlp.0.bias"])
                w_tensor(f, sd[f"{pre}.msg_mlp.2.weight"])
                w_tensor(f, sd[f"{pre}.msg_mlp.2.bias"])
                w_tensor(f, sd[f"{pre}.update_mlp.0.weight"])
                w_tensor(f, sd[f"{pre}.update_mlp.0.bias"])
                w_tensor(f, sd[f"{pre}.update_mlp.2.weight"])
                w_tensor(f, sd[f"{pre}.update_mlp.2.bias"])
                w_tensor(f, sd[f"{pre}.norm.weight"])
                w_tensor(f, sd[f"{pre}.norm.bias"])

        # output_proj: Linear(2H, GO) + Mish + Linear(GO, GO)
        w_tensor(f, sd["board_encoder.output_proj.0.weight"])  # (GO, 2H)
        w_tensor(f, sd["board_encoder.output_proj.0.bias"])    # (GO,)
        w_tensor(f, sd["board_encoder.output_proj.2.weight"])  # (GO, GO)
        w_tensor(f, sd["board_encoder.output_proj.2.bias"])    # (GO,)

        # -- Trunk weights (fused BN) --
        # input_proj: Linear(640, TC) + BN(TC) + Mish
        w_tensor(f, sd["trunk.input_proj.0.weight"])  # (TC, 640)
        w_tensor(f, sd["trunk.input_proj.0.bias"])    # (TC,)
        s, sh = fuse_bn(sd["trunk.input_proj.1.weight"], sd["trunk.input_proj.1.bias"],
                        sd["trunk.input_proj.1.running_mean"], sd["trunk.input_proj.1.running_var"])
        w_fused(f, s, sh)

        # 6 ResBlocks: C loads one ResBlockWeights struct per block
        for i in range(TB):
            pre = f"trunk.blocks.{i}"
            if wf == "int8":
                s1, sh1 = fuse_bn(
                    sd[f"{pre}.bn1.weight"], sd[f"{pre}.bn1.bias"],
                    sd[f"{pre}.bn1.running_mean"], sd[f"{pre}.bn1.running_var"]
                )
                s2, sh2 = fuse_bn(
                    sd[f"{pre}.bn2.weight"], sd[f"{pre}.bn2.bias"],
                    sd[f"{pre}.bn2.running_mean"], sd[f"{pre}.bn2.running_var"]
                )
                arr = np.concatenate(
                    [
                        sd[f"{pre}.fc1.weight"].detach().cpu().float().numpy().ravel(),
                        sd[f"{pre}.fc1.bias"].detach().cpu().float().numpy().ravel(),
                        np.asarray(s1, dtype=np.float32).ravel(),
                        np.asarray(sh1, dtype=np.float32).ravel(),
                        sd[f"{pre}.fc2.weight"].detach().cpu().float().numpy().ravel(),
                        sd[f"{pre}.fc2.bias"].detach().cpu().float().numpy().ravel(),
                        np.asarray(s2, dtype=np.float32).ravel(),
                        np.asarray(sh2, dtype=np.float32).ravel(),
                    ]
                )
                write_int8_symmetric(f, arr)
            else:
                w_tensor(f, sd[f"{pre}.fc1.weight"])
                w_tensor(f, sd[f"{pre}.fc1.bias"])
                s, sh = fuse_bn(
                    sd[f"{pre}.bn1.weight"], sd[f"{pre}.bn1.bias"],
                    sd[f"{pre}.bn1.running_mean"], sd[f"{pre}.bn1.running_var"]
                )
                w_fused(f, s, sh)
                w_tensor(f, sd[f"{pre}.fc2.weight"])
                w_tensor(f, sd[f"{pre}.fc2.bias"])
                s, sh = fuse_bn(
                    sd[f"{pre}.bn2.weight"], sd[f"{pre}.bn2.bias"],
                    sd[f"{pre}.bn2.running_mean"], sd[f"{pre}.bn2.running_var"]
                )
                w_fused(f, s, sh)

        # -- Value head (fused BN) --
        w_tensor(f, sd["value_head.fc1.weight"])
        w_tensor(f, sd["value_head.fc1.bias"])
        s, sh = fuse_bn(sd["value_head.bn1.weight"], sd["value_head.bn1.bias"],
                        sd["value_head.bn1.running_mean"], sd["value_head.bn1.running_var"])
        w_fused(f, s, sh)
        for r in ["res1", "res2"]:
            pre = f"value_head.{r}"
            if wf == "int8":
                s1, sh1 = fuse_bn(
                    sd[f"{pre}.bn1.weight"], sd[f"{pre}.bn1.bias"],
                    sd[f"{pre}.bn1.running_mean"], sd[f"{pre}.bn1.running_var"]
                )
                s2, sh2 = fuse_bn(
                    sd[f"{pre}.bn2.weight"], sd[f"{pre}.bn2.bias"],
                    sd[f"{pre}.bn2.running_mean"], sd[f"{pre}.bn2.running_var"]
                )
                arr = np.concatenate(
                    [
                        sd[f"{pre}.fc1.weight"].detach().cpu().float().numpy().ravel(),
                        sd[f"{pre}.fc1.bias"].detach().cpu().float().numpy().ravel(),
                        np.asarray(s1, dtype=np.float32).ravel(),
                        np.asarray(sh1, dtype=np.float32).ravel(),
                        sd[f"{pre}.fc2.weight"].detach().cpu().float().numpy().ravel(),
                        sd[f"{pre}.fc2.bias"].detach().cpu().float().numpy().ravel(),
                        np.asarray(s2, dtype=np.float32).ravel(),
                        np.asarray(sh2, dtype=np.float32).ravel(),
                    ]
                )
                write_int8_symmetric(f, arr)
            else:
                w_tensor(f, sd[f"{pre}.fc1.weight"])
                w_tensor(f, sd[f"{pre}.fc1.bias"])
                s, sh = fuse_bn(
                    sd[f"{pre}.bn1.weight"], sd[f"{pre}.bn1.bias"],
                    sd[f"{pre}.bn1.running_mean"], sd[f"{pre}.bn1.running_var"]
                )
                w_fused(f, s, sh)
                w_tensor(f, sd[f"{pre}.fc2.weight"])
                w_tensor(f, sd[f"{pre}.fc2.bias"])
                s, sh = fuse_bn(
                    sd[f"{pre}.bn2.weight"], sd[f"{pre}.bn2.bias"],
                    sd[f"{pre}.bn2.running_mean"], sd[f"{pre}.bn2.running_var"]
                )
                w_fused(f, s, sh)
        w_tensor(f, sd["value_head.fc_out.weight"])  # (4, VH)
        w_tensor(f, sd["value_head.fc_out.bias"])    # (4,)

        # -- Policy head --
        # trunk_norm (LayerNorm)
        w_tensor(f, sd["policy_head.trunk_norm.weight"])
        w_tensor(f, sd["policy_head.trunk_norm.bias"])
        # node_norm (LayerNorm)
        w_tensor(f, sd["policy_head.node_norm.weight"])
        w_tensor(f, sd["policy_head.node_norm.bias"])
        # type_fc: Linear(TC, PH) + BN(PH) + Mish + Linear(PH, 12)
        w_tensor(f, sd["policy_head.type_fc.0.weight"])
        w_tensor(f, sd["policy_head.type_fc.0.bias"])
        s, sh = fuse_bn(sd["policy_head.type_fc.1.weight"], sd["policy_head.type_fc.1.bias"],
                        sd["policy_head.type_fc.1.running_mean"], sd["policy_head.type_fc.1.running_var"])
        w_fused(f, s, sh)
        w_tensor(f, sd["policy_head.type_fc.3.weight"])
        w_tensor(f, sd["policy_head.type_fc.3.bias"])
        # sub-action heads
        for name in ["discard_yop_mono_fc", "maritime_fc", "trade_fc"]:
            w_tensor(f, sd[f"policy_head.{name}.0.weight"])
            w_tensor(f, sd[f"policy_head.{name}.0.bias"])
            w_tensor(f, sd[f"policy_head.{name}.2.weight"])
            w_tensor(f, sd[f"policy_head.{name}.2.bias"])
        # spatial scorers
        for name in ["settlement_scorer", "city_scorer", "road_scorer", "robber_scorer"]:
            w_tensor(f, sd[f"policy_head.{name}.0.weight"])
            w_tensor(f, sd[f"policy_head.{name}.0.bias"])
            w_tensor(f, sd[f"policy_head.{name}.2.weight"])
            w_tensor(f, sd[f"policy_head.{name}.2.bias"])

    import os
    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported {output_path} ({size_kb:.0f} KB)")

    if not write_test_vectors:
        return

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
    parser.add_argument(
        "--weight-format",
        choices=("fp32", "fp16", "int8"),
        default="fp32",
        help="fp16=v2, int8=v3 symmetric per-tensor quant; C nn_load expands to fp32 (same runtime as fp32).",
    )
    args = parser.parse_args()
    export(args.checkpoint, args.output, weight_format=args.weight_format)


if __name__ == "__main__":
    main()
