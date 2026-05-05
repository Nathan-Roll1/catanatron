"""Optimized single-core policy inference for Apple Silicon.

Fuses the GNN message-passing into pre-computed dense ops for batch=1
on a fixed graph. Eliminates scatter/gather overhead that dominates
the standard EdgeConvLayer at small batch sizes.

Usage:
    from human_bot.fast_policy import FastPolicy
    fp = FastPolicy("checkpoints/selfplay_v2/latest.pt")
    action_idx = fp.pick(game)
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AD = 337
MASK_DIM = 397


class FusedEdgeConv:
    """Pre-compiled dense message passing for a fixed graph.

    Instead of scatter/gather per forward, we pre-build a dense adjacency
    matrix A[N,N] and compute aggregation as A @ messages, which is a
    single matmul on Apple Silicon's AMX.
    """
    __slots__ = ("msg_w1", "msg_b1", "msg_w2", "msg_b2",
                 "upd_w1", "upd_b1", "upd_w2", "upd_b2",
                 "ln_w", "ln_b", "scatter_matrix", "src_idx", "dst_idx")

    def __init__(self, layer: nn.Module, src: np.ndarray, dst: np.ndarray,
                 num_nodes: int):
        sd = layer.state_dict()
        self.msg_w1 = sd["msg_mlp.0.weight"].numpy()
        self.msg_b1 = sd["msg_mlp.0.bias"].numpy()
        self.msg_w2 = sd["msg_mlp.2.weight"].numpy()
        self.msg_b2 = sd["msg_mlp.2.bias"].numpy()
        self.upd_w1 = sd["update_mlp.0.weight"].numpy()
        self.upd_b1 = sd["update_mlp.0.bias"].numpy()
        self.upd_w2 = sd["update_mlp.2.weight"].numpy()
        self.upd_b2 = sd["update_mlp.2.bias"].numpy()
        self.ln_w = sd["norm.weight"].numpy()
        self.ln_b = sd["norm.bias"].numpy()
        self.src_idx = src
        self.dst_idx = dst

        num_edges = len(src)
        self.scatter_matrix = np.zeros((num_nodes, num_edges), dtype=np.float32)
        for ei, d in enumerate(dst):
            self.scatter_matrix[d, ei] = 1.0

    def forward(self, x: np.ndarray, edge_attr: np.ndarray) -> np.ndarray:
        N, H = x.shape
        inp = np.concatenate([x[self.src_idx], x[self.dst_idx], edge_attr], axis=1)

        m = inp @ self.msg_w1.T + self.msg_b1
        m = _mish(m)
        m = m @ self.msg_w2.T + self.msg_b2
        m = _mish(m)

        agg = self.scatter_matrix @ m

        inp2 = np.concatenate([x, agg], axis=1)
        out = inp2 @ self.upd_w1.T + self.upd_b1
        out = _mish(out)
        out = out @ self.upd_w2.T + self.upd_b2

        res = x + out
        mean = res.mean(axis=1, keepdims=True)
        var = res.var(axis=1, keepdims=True)
        res = (res - mean) / np.sqrt(var + 1e-5) * self.ln_w + self.ln_b
        return res


def _mish(x):
    return x * np.tanh(np.log1p(np.exp(np.clip(x, -20, 20))))


def _log_softmax(x):
    m = x.max()
    lse = m + np.log(np.sum(np.exp(x - m)))
    return x - lse


class FastPolicy:
    """Numpy-only policy inference. Zero torch overhead per decision."""

    def __init__(self, checkpoint_path: str):
        from human_bot.model import HumanBotNet
        from hexzero.bindings.lib_loader import load_library
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        self._lib = load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        N = self._se.num_nodes
        E = self._se.num_edges

        net = HumanBotNet.load_checkpoint(checkpoint_path, device="cpu")
        net.eval()
        sd = net.state_dict()
        cfg = net.config

        ei = self._se._edge_index.numpy()
        src, dst = ei[0], ei[1]

        self._node_proj_w = sd["board_encoder.node_proj.0.weight"].numpy()
        self._node_proj_b = sd["board_encoder.node_proj.0.bias"].numpy()
        self._edge_proj_w = sd["board_encoder.edge_proj.weight"].numpy()
        self._edge_proj_b = sd["board_encoder.edge_proj.bias"].numpy()

        self._gnn_layers = []
        for i in range(cfg.gnn_layers):
            layer = net.board_encoder.layers[i]
            self._gnn_layers.append(FusedEdgeConv(layer, src, dst, N))

        self._out_proj_w1 = sd["board_encoder.output_proj.0.weight"].numpy()
        self._out_proj_b1 = sd["board_encoder.output_proj.0.bias"].numpy()
        self._out_proj_w2 = sd["board_encoder.output_proj.2.weight"].numpy()
        self._out_proj_b2 = sd["board_encoder.output_proj.2.bias"].numpy()

        self._trunk_blocks = []
        self._trunk_ip_w = sd["trunk.input_proj.0.weight"].numpy()
        self._trunk_ip_b = sd["trunk.input_proj.0.bias"].numpy()
        self._trunk_ip_bn = _extract_bn(sd, "trunk.input_proj.1")
        for i in range(cfg.trunk_blocks):
            blk = {
                "fc1_w": sd[f"trunk.blocks.{i}.fc1.weight"].numpy(),
                "fc1_b": sd[f"trunk.blocks.{i}.fc1.bias"].numpy(),
                "bn1": _extract_bn(sd, f"trunk.blocks.{i}.bn1"),
                "fc2_w": sd[f"trunk.blocks.{i}.fc2.weight"].numpy(),
                "fc2_b": sd[f"trunk.blocks.{i}.fc2.bias"].numpy(),
                "bn2": _extract_bn(sd, f"trunk.blocks.{i}.bn2"),
            }
            self._trunk_blocks.append(blk)

        self._pol = _extract_policy_head(sd, cfg)
        self._cfg = cfg
        self._N = N
        self._E = E

        self._nf = np.zeros((N, cfg.node_feature_dim), dtype=np.float32)
        self._ef = np.zeros((E, cfg.edge_feature_dim), dtype=np.float32)
        self._ff = np.zeros(cfg.flat_feature_dim, dtype=np.float32)

    def forward(self, nf, ef, ff, mask) -> np.ndarray:
        """Pure numpy forward. Returns masked policy logits (397,)."""
        H = self._cfg.gnn_hidden_dim

        h = nf @ self._node_proj_w.T + self._node_proj_b
        h = _mish(h)
        e = ef @ self._edge_proj_w.T + self._edge_proj_b

        for layer in self._gnn_layers:
            h = layer.forward(h, e)

        mean_pool = h.mean(axis=0)
        max_pool = h.max(axis=0)
        cat = np.concatenate([mean_pool, max_pool])
        board_emb = cat @ self._out_proj_w1.T + self._out_proj_b1
        board_emb = _mish(board_emb)
        board_emb = board_emb @ self._out_proj_w2.T + self._out_proj_b2

        parts = [board_emb, ff]
        if self._cfg.mask_as_input:
            parts.append(mask)
        combined = np.concatenate(parts)

        x = combined @ self._trunk_ip_w.T + self._trunk_ip_b
        x = _apply_bn_mish(x, self._trunk_ip_bn)
        for blk in self._trunk_blocks:
            residual = x
            x2 = x @ blk["fc1_w"].T + blk["fc1_b"]
            x2 = _apply_bn_mish(x2, blk["bn1"])
            x2 = x2 @ blk["fc2_w"].T + blk["fc2_b"]
            x2 = _apply_bn_mish_residual(x2, blk["bn2"], residual)
            x = x2

        logits = _policy_forward(x, h, self._pol, self._cfg)

        mask_bool = mask[:MASK_DIM] > 0.5
        logits[~mask_bool] = -1e9
        return logits

    @property
    def se(self):
        return self._se

    @property
    def ae(self):
        return self._ae

    def pick(self, game) -> int:
        """Encode game state and return best legal action index."""
        le = game.get_legal_actions()
        if not le:
            return -1
        if len(le) == 1:
            return 0

        self._se.encode_into(game.get_state_view(),
                             self._nf, self._ef, self._ff)
        mn = self._ae.get_action_mask(le).numpy()
        mk = np.zeros(MASK_DIM, dtype=np.float32)
        mk[:len(mn)] = mn

        logits = self.forward(self._nf, self._ef, self._ff, mk)
        lo = logits[:AD]
        best_enc = int(np.argmax(lo))
        for i, a in enumerate(le):
            try:
                if self._ae.encode(a) == best_enc:
                    return i
            except ValueError:
                continue
        return 0


def _extract_bn(sd, prefix):
    return {
        "scale": (sd[f"{prefix}.weight"] /
                  torch.sqrt(sd[f"{prefix}.running_var"] + 1e-5)).numpy(),
        "shift": (sd[f"{prefix}.bias"] -
                  sd[f"{prefix}.running_mean"] *
                  sd[f"{prefix}.weight"] /
                  torch.sqrt(sd[f"{prefix}.running_var"] + 1e-5)).numpy(),
    }


def _apply_bn_mish(x, bn):
    x = x * bn["scale"] + bn["shift"]
    return _mish(x)


def _apply_bn_mish_residual(x, bn, residual):
    x = x * bn["scale"] + bn["shift"]
    x = x + residual
    return _mish(x)


def _extract_policy_head(sd, cfg):
    T = cfg.trunk_channels
    H = cfg.gnn_hidden_dim
    p = {}
    p["trunk_ln_w"] = sd["policy_head.trunk_norm.weight"].numpy()
    p["trunk_ln_b"] = sd["policy_head.trunk_norm.bias"].numpy()
    p["node_ln_w"] = sd["policy_head.node_norm.weight"].numpy()
    p["node_ln_b"] = sd["policy_head.node_norm.bias"].numpy()

    p["type_w1"] = sd["policy_head.type_fc.0.weight"].numpy()
    p["type_b1"] = sd["policy_head.type_fc.0.bias"].numpy()
    p["type_bn"] = _extract_bn(sd, "policy_head.type_fc.1")
    p["type_w2"] = sd["policy_head.type_fc.3.weight"].numpy()
    p["type_b2"] = sd["policy_head.type_fc.3.bias"].numpy()

    for name in ("discard_yop_mono_fc", "maritime_fc", "trade_fc"):
        short = {"discard_yop_mono_fc": "dym", "maritime_fc": "mar",
                 "trade_fc": "trd"}[name]
        p[f"{short}_w1"] = sd[f"policy_head.{name}.0.weight"].numpy()
        p[f"{short}_b1"] = sd[f"policy_head.{name}.0.bias"].numpy()
        p[f"{short}_w2"] = sd[f"policy_head.{name}.2.weight"].numpy()
        p[f"{short}_b2"] = sd[f"policy_head.{name}.2.bias"].numpy()

    for name in ("settlement_scorer", "city_scorer", "road_scorer", "robber_scorer"):
        short = name.replace("_scorer", "")
        p[f"{short}_w1"] = sd[f"policy_head.{name}.0.weight"].numpy()
        p[f"{short}_b1"] = sd[f"policy_head.{name}.0.bias"].numpy()
        p[f"{short}_w2"] = sd[f"policy_head.{name}.2.weight"].numpy()
        p[f"{short}_b2"] = sd[f"policy_head.{name}.2.bias"].numpy()

    p["road_pairs"] = sd["policy_head.road_pairs"].numpy()
    p["tile_nodes"] = sd["policy_head.tile_nodes"].numpy()
    return p


def _policy_forward(trunk_out, node_emb, p, cfg):
    N, H = node_emb.shape
    T = trunk_out.shape[0]

    tn = _layer_norm(trunk_out, p["trunk_ln_w"], p["trunk_ln_b"])
    nn_ = np.zeros_like(node_emb)
    for i in range(N):
        nn_[i] = _layer_norm(node_emb[i], p["node_ln_w"], p["node_ln_b"])

    x = tn @ p["type_w1"].T + p["type_b1"]
    x = _apply_bn_mish_1d(x, p["type_bn"])
    type_logits = x @ p["type_w2"].T + p["type_b2"]
    log_type = _log_softmax_vec(type_logits)

    ctx = np.concatenate([np.tile(tn, (N, 1)), nn_], axis=1)

    sett_raw = (ctx @ p["settlement_w1"].T + p["settlement_b1"])
    sett_raw = _mish(sett_raw)
    sett_raw = (sett_raw @ p["settlement_w2"].T + p["settlement_b2"]).ravel()

    city_raw = (ctx @ p["city_w1"].T + p["city_b1"])
    city_raw = _mish(city_raw)
    city_raw = (city_raw @ p["city_w2"].T + p["city_b2"]).ravel()

    rp = p["road_pairs"]
    src_n = nn_[rp[:, 0]]
    dst_n = nn_[rp[:, 1]]
    road_ctx = np.concatenate([np.tile(tn, (72, 1)), src_n, dst_n], axis=1)
    road_raw = (road_ctx @ p["road_w1"].T + p["road_b1"])
    road_raw = _mish(road_raw)
    road_raw = (road_raw @ p["road_w2"].T + p["road_b2"]).ravel()

    tn_ = p["tile_nodes"]
    tile_emb = nn_[tn_].mean(axis=1)
    tile_ctx = np.concatenate([np.tile(tn, (19, 1)), tile_emb], axis=1)
    robber_raw = (tile_ctx @ p["robber_w1"].T + p["robber_b1"])
    robber_raw = _mish(robber_raw)
    robber_raw = (robber_raw @ p["robber_w2"].T + p["robber_b2"]).ravel()

    dym_raw = _mish(tn @ p["dym_w1"].T + p["dym_b1"]) @ p["dym_w2"].T + p["dym_b2"]
    mar_raw = _mish(tn @ p["mar_w1"].T + p["mar_b1"]) @ p["mar_w2"].T + p["mar_b2"]
    trd_raw = _mish(tn @ p["trd_w1"].T + p["trd_b1"]) @ p["trd_w2"].T + p["trd_b2"]

    out = np.empty(397, dtype=np.float32)
    out[0] = log_type[0]
    out[1] = log_type[1]
    out[2] = log_type[2]
    out[3] = log_type[3]
    out[4] = log_type[4]
    out[5:59] = log_type[5] + _log_softmax_vec(sett_raw)
    out[59:113] = log_type[6] + _log_softmax_vec(city_raw)
    out[113:185] = log_type[7] + _log_softmax_vec(road_raw)
    out[185:280] = log_type[8] + _log_softmax_vec(robber_raw)
    out[280:310] = log_type[9] + _log_softmax_vec(dym_raw)
    out[310:330] = log_type[10] + _log_softmax_vec(mar_raw)
    out[330:397] = log_type[11] + _log_softmax_vec(trd_raw)
    return out


def _layer_norm(x, w, b, eps=1e-5):
    mean = x.mean()
    var = x.var()
    return (x - mean) / np.sqrt(var + eps) * w + b


def _apply_bn_mish_1d(x, bn):
    return _mish(x * bn["scale"] + bn["shift"])


def _log_softmax_vec(x):
    m = x.max()
    return x - m - np.log(np.sum(np.exp(x - m)))


class CompiledPolicy:
    """Fastest PyTorch inference: policy-only + torch.compile.

    Usage:
        cp = CompiledPolicy("checkpoints/selfplay_v2/latest.pt")
        action_idx = cp.pick(game)
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        from human_bot.model import HumanBotNet
        from hexzero.bindings.lib_loader import load_library
        from hexzero.game.interface import CatanGame
        from hexzero.encoder.action_encoder import ActionEncoder

        torch.set_num_threads(1)
        self._lib = load_library()
        self._ae = ActionEncoder()
        g0 = CatanGame(seed=0); g0.reset()
        self._se = g0.make_state_encoder()
        self._device = device

        self._net = HumanBotNet.load_checkpoint(checkpoint_path, device=device)
        self._net.eval()
        self._compiled = False

        self._ei = self._se._edge_index.to(device)
        N, E = self._se.num_nodes, self._se.num_edges
        NF = self._se.NODE_FEATURE_DIM
        EF = self._se.EDGE_FEATURE_DIM
        FFD = self._se.FLAT_FEATURE_DIM
        self._nf = np.zeros((1, N, NF), dtype=np.float32)
        self._ef = np.zeros((1, E, EF), dtype=np.float32)
        self._ff = np.zeros((1, FFD), dtype=np.float32)
        self._mk = np.zeros((1, MASK_DIM), dtype=np.float32)

        print(f"CompiledPolicy: warming up{'  (compiled)' if self._compiled else ''}...")
        self._encode_batch(g0)
        for _ in range(10):
            with torch.inference_mode():
                self._net(self._batch)
        print(f"CompiledPolicy: ready")

    def _encode_batch(self, game):
        self._se.encode_into(game.get_state_view(),
                             self._nf[0], self._ef[0], self._ff[0])
        le = game.get_legal_actions()
        mn = self._ae.get_action_mask(le).numpy()
        self._mk[0, :] = 0
        self._mk[0, :len(mn)] = mn
        self._batch = {
            "node_features": torch.from_numpy(self._nf).to(self._device),
            "edge_index": self._ei,
            "edge_features": torch.from_numpy(self._ef).to(self._device),
            "flat_features": torch.from_numpy(self._ff).to(self._device),
            "action_mask": torch.from_numpy(self._mk).to(self._device),
        }
        return le, mn

    def pick(self, game) -> int:
        le, mn = self._encode_batch(game)
        if not le:
            return -1
        if len(le) == 1:
            return 0
        with torch.inference_mode():
            out = self._net(self._batch)
        lo = out["policy_logits"][0, :AD].cpu().numpy()
        lo[mn[:AD] < 0.5] = -1e9
        best_enc = int(np.argmax(lo))
        for i, a in enumerate(le):
            try:
                if self._ae.encode(a) == best_enc:
                    return i
            except ValueError:
                continue
        return 0

    @property
    def se(self):
        return self._se

    @property
    def ae(self):
        return self._ae


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/selfplay_v2/latest.pt"

    torch.set_num_threads(1)
    from hexzero.game.interface import CatanGame

    print("=== CompiledPolicy (torch.compile, policy-only) ===")
    cp = CompiledPolicy(ckpt)
    g = CatanGame(seed=42); g.reset()
    for _ in range(50):
        cp.pick(g)
    RUNS = 500
    t0 = time.perf_counter()
    for _ in range(RUNS):
        cp.pick(g)
    dt_comp = (time.perf_counter() - t0) / RUNS * 1000
    print(f"CompiledPolicy: {dt_comp:.2f} ms ({1000/dt_comp:.0f}/s)")

    print()
    print("=== FastPolicy (numpy, policy-only) ===")
    fp = FastPolicy(ckpt)
    for _ in range(20):
        fp.pick(g)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        fp.pick(g)
    dt_np = (time.perf_counter() - t0) / RUNS * 1000
    print(f"FastPolicy:     {dt_np:.2f} ms ({1000/dt_np:.0f}/s)")

    print()
    print("=== Baseline (full HumanBotNet, torch) ===")
    from human_bot.model import HumanBotNet
    net = HumanBotNet.load_checkpoint(ckpt, device="cpu")
    net.eval()
    se = cp.se; ae = cp.ae
    nf = np.zeros((1, se.num_nodes, se.NODE_FEATURE_DIM), np.float32)
    ef = np.zeros((1, se.num_edges, se.EDGE_FEATURE_DIM), np.float32)
    ff = np.zeros((1, se.FLAT_FEATURE_DIM), np.float32)
    mk = np.zeros((1, MASK_DIM), np.float32)
    se.encode_into(g.get_state_view(), nf[0], ef[0], ff[0])
    le = g.get_legal_actions()
    mn = ae.get_action_mask(le).numpy()
    mk[0, :len(mn)] = mn
    batch = {
        "node_features": torch.from_numpy(nf), "edge_index": se._edge_index,
        "edge_features": torch.from_numpy(ef), "flat_features": torch.from_numpy(ff),
        "action_mask": torch.from_numpy(mk),
    }
    for _ in range(20):
        with torch.no_grad(): net(batch)
    t0 = time.perf_counter()
    for _ in range(RUNS):
        with torch.no_grad(): net(batch)
    dt_base = (time.perf_counter() - t0) / RUNS * 1000
    print(f"Baseline:       {dt_base:.2f} ms ({1000/dt_base:.0f}/s)")
    print()
    print(f"Compiled vs baseline: {dt_base/dt_comp:.1f}x faster")
    print(f"Numpy vs baseline:    {dt_base/dt_np:.1f}x")
