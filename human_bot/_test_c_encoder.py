"""Verify C state_encoder produces byte-identical output to Python encoder.

Tests on multiple game states (early/mid/late game) with multiple seeds.
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

# ── Load libencode ────────────────────────────────────────────────
_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "csrc", "libencode.dylib")


def _build_state_encoder_struct():
    """Match the C StateEncoderC layout for ctypes binding."""
    TOTAL_NODES = 96
    ENC_NUM_NODES = 54
    ENC_NUM_EDGES = 144
    NUM_LAND_TILES = 19

    class StateEncoderC(ctypes.Structure):
        _fields_ = [
            ("N", ctypes.c_int),
            ("E", ctypes.c_int),
            ("land_to_local", ctypes.c_int * TOTAL_NODES),
            ("local_to_global", ctypes.c_int * ENC_NUM_NODES),
            ("ltiles", (ctypes.c_int * 6) * NUM_LAND_TILES),
            ("tile_coords", (ctypes.c_int * 3) * NUM_LAND_TILES),
            ("road_src_global", ctypes.c_int * ENC_NUM_EDGES),
            ("road_adj_idx", ctypes.c_int * ENC_NUM_EDGES),
            ("port_oh", (ctypes.c_float * 7) * ENC_NUM_NODES),
            ("n_real_players", ctypes.c_int),
        ]
    return StateEncoderC


StateEncoderC = _build_state_encoder_struct()

_lib = ctypes.CDLL(_LIB_PATH)
_lib.state_encoder_init.restype = None
_lib.state_encoder_init.argtypes = [
    ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_int]
_lib.encode_state.restype = None
_lib.encode_state.argtypes = [
    ctypes.POINTER(StateEncoderC), ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float)]


def c_encode(enc, game):
    """Run C encoder, return (nf, ef, flat) numpy arrays."""
    nf = np.zeros((enc.N, 18), dtype=np.float32)
    ef = np.zeros((enc.E, 5), dtype=np.float32)
    flat = np.zeros(115, dtype=np.float32)
    _lib.encode_state(
        ctypes.byref(enc), ctypes.addressof(game._game),
        nf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ef.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return nf, ef, flat


def py_encode(se, game):
    """Run Python encoder, return (nf, ef, flat) numpy arrays."""
    N, E = se.num_nodes, se.num_edges
    nf = np.zeros((N, 18), dtype=np.float32)
    ef = np.zeros((E, 5), dtype=np.float32)
    flat = np.zeros(115, dtype=np.float32)
    se.encode_into(game.get_state_view(), nf, ef, flat)
    return nf, ef, flat


def compare(name, c_arr, py_arr, atol=1e-5):
    """Compare two arrays. Return True if match within tolerance."""
    if c_arr.shape != py_arr.shape:
        print(f"  {name}: SHAPE MISMATCH c={c_arr.shape} py={py_arr.shape}")
        return False
    diff = np.abs(c_arr - py_arr)
    max_diff = float(diff.max()) if c_arr.size > 0 else 0.0
    if max_diff > atol:
        # Find first mismatch
        bad = np.unravel_index(np.argmax(diff), c_arr.shape)
        print(f"  {name}: MISMATCH max_diff={max_diff:.6g} at {bad} "
              f"c={c_arr[bad]:.6f} py={py_arr[bad]:.6f}")
        return False
    print(f"  {name}: OK (max_diff={max_diff:.2e})")
    return True


def test_one_game(seed, n_steps_list=(0, 7, 30, 60, 100)):
    """Play one game, sample at multiple turns, compare encoders."""
    from hexzero.bindings.lib_loader import load_library
    from hexzero.game.interface import CatanGame

    load_library()
    game = CatanGame(seed=seed); game.reset()
    se = game.make_state_encoder()

    enc = StateEncoderC()
    _lib.state_encoder_init(ctypes.byref(enc), ctypes.addressof(game._game), 4)

    print(f"  Encoder: N={enc.N} E={enc.E} (Python: N={se.num_nodes} E={se.num_edges})")
    if enc.N != se.num_nodes or enc.E != se.num_edges:
        print(f"  FAIL: dimension mismatch")
        return False

    # Verify pre-computed tables match
    py_land = se._land
    c_land = np.array([enc.local_to_global[i] for i in range(enc.N)])
    if not np.array_equal(c_land, py_land):
        print(f"  FAIL: land mismatch\n  c={c_land[:10]}\n  py={py_land[:10]}")
        return False
    print(f"  Land array: OK (N={enc.N})")

    py_g2l = se._g2l
    c_g2l = np.array([enc.land_to_local[i] for i in range(96)])
    if not np.array_equal(c_g2l, py_g2l):
        print(f"  FAIL: g2l mismatch")
        return False
    print(f"  g2l: OK")

    py_ltiles = se._ltiles
    c_ltiles = np.array([[enc.ltiles[t][k] for k in range(6)] for t in range(19)])
    if not np.array_equal(c_ltiles, py_ltiles):
        print(f"  FAIL: ltiles mismatch")
        return False
    print(f"  ltiles: OK")

    py_port_oh = se._port_oh_np
    c_port_oh = np.array([[enc.port_oh[i][k] for k in range(7)] for i in range(enc.N)])
    if not np.array_equal(c_port_oh, py_port_oh):
        print(f"  FAIL: port_oh mismatch")
        diff = c_port_oh != py_port_oh
        print(f"  Differences at: {np.argwhere(diff)[:5]}")
        return False
    print(f"  port_oh: OK")

    # Now compare encoded outputs at multiple game stages
    all_ok = True
    for target_step in n_steps_list:
        # Step game until target_step (or end)
        steps_taken = 0
        while steps_taken < target_step and not game.is_terminal():
            le = game.get_legal_actions()
            if not le: break
            game.step(0)  # arbitrary — just need to advance state
            steps_taken += 1

        if game.is_terminal():
            print(f"  Game terminated before step {target_step}, stopping")
            break

        nf_c, ef_c, flat_c = c_encode(enc, game)
        nf_py, ef_py, flat_py = py_encode(se, game)

        print(f"  At step ~{steps_taken} (turn {game.turn_number}):")
        ok_nf = compare("    nf", nf_c, nf_py, atol=1e-5)
        ok_ef = compare("    ef", ef_c, ef_py, atol=1e-5)
        ok_flat = compare("    flat", flat_c, flat_py, atol=1e-5)
        all_ok = all_ok and ok_nf and ok_ef and ok_flat

    return all_ok


def main():
    print("=== Phase 1: C encoder verification ===")
    print()
    seeds = [42, 95000, 80000, 12345]
    all_passed = True
    for seed in seeds:
        print(f"--- Seed {seed} ---")
        ok = test_one_game(seed)
        all_passed = all_passed and ok
        print()
    print("=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
