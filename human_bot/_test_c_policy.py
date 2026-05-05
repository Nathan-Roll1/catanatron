"""Phase 2 verification: C policy_top_k matches Python policy_top_k.

For each test state, compute top-K via:
  - C: encode + nn_forward + top-K all in libpolicy.dylib
  - Python: encode_into + nn_forward (libnn) + top-K via numpy
And verify the indices returned are identical.
"""
import ctypes
import os
import sys

import numpy as np

from human_bot._test_c_encoder import StateEncoderC

# ── Load libpolicy ────────────────────────────────────────────────
_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "csrc", "libpolicy.dylib")
_lib = ctypes.CDLL(_LIB_PATH)

# Re-bind state_encoder_init since libpolicy includes state_encode.c
_lib.state_encoder_init.restype = None
_lib.state_encoder_init.argtypes = [
    ctypes.POINTER(StateEncoderC), ctypes.c_void_p, ctypes.c_int]

# Existing nn_load (libpolicy includes nn.c so this works)
_lib.nn_load.restype = ctypes.c_int
_lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# policy_top_k signature
FP = ctypes.POINTER(ctypes.c_float)
_lib.policy_top_k.restype = ctypes.c_int
_lib.policy_top_k.argtypes = [
    ctypes.POINTER(StateEncoderC),       # enc
    ctypes.c_void_p,                      # nn model
    ctypes.c_void_p,                      # game
    ctypes.c_void_p,                      # actions
    ctypes.c_int,                         # n_actions
    ctypes.c_int,                         # k
    ctypes.POINTER(ctypes.c_int),         # out_indices
    FP, FP, FP, FP, FP                    # nf, ef, ff, mk, out
]

# policy_action_encode (for individual action encoding tests)
_lib.policy_action_encode.restype = ctypes.c_int
_lib.policy_action_encode.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p
]


def py_top_k(net_lib, mptr, se, ae, game, le, k, scratch):
    """Python top-K, mirrors the Python superbot_v3_c policy callback."""
    nf, ef, ff, mk, out_arr = scratch
    se.encode_into(game.get_state_view(), nf, ef, ff)
    mn = ae.get_action_mask(le).numpy()
    mk[:] = 0
    mk[:len(mn)] = mn
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)
    outp = out_arr.ctypes.data_as(ctypes.c_void_p)
    net_lib.nn_forward(mptr, nfp, efp, ffp, mkp, outp)
    AD = 337
    logits = out_arr[4:4 + AD]
    a2i = {}
    for i, a in enumerate(le):
        try:
            a2i[ae.encode(a)] = i
        except ValueError:
            continue
    if not a2i:
        return []
    scored = sorted([(logits[e], i) for e, i in a2i.items()], reverse=True)
    return [i for _, i in scored[:k]]


def c_top_k(enc, mptr, game, le, k):
    """C top-K via libpolicy."""
    from hexzero.bindings.structs import Action as CAction, MAX_ACTIONS

    n = len(le)
    actions_arr = (CAction * MAX_ACTIONS)()
    for i, a in enumerate(le):
        actions_arr[i] = a
    out_indices = (ctypes.c_int * 64)()
    nf = np.zeros(54 * 18, dtype=np.float32)
    ef = np.zeros(144 * 5, dtype=np.float32)
    ff = np.zeros(115, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    out = np.zeros(4 + 397, dtype=np.float32)

    n_top = _lib.policy_top_k(
        ctypes.byref(enc), mptr, ctypes.addressof(game._game),
        ctypes.cast(actions_arr, ctypes.c_void_p),
        n, k, out_indices,
        nf.ctypes.data_as(FP), ef.ctypes.data_as(FP), ff.ctypes.data_as(FP),
        mk.ctypes.data_as(FP), out.ctypes.data_as(FP))
    return [out_indices[i] for i in range(n_top)]


def test_action_encoding(ae, mptr_for_libpolicy, sample_actions):
    """Test that policy_action_encode matches Python's ae.encode for each action."""
    print("  Testing action encoding round-trip...")
    n_pass = n_fail = 0
    for a in sample_actions:
        try:
            py_idx = ae.encode(a)
        except ValueError:
            py_idx = -1
        c_idx = _lib.policy_action_encode(
            mptr_for_libpolicy, ctypes.byref(a))
        if py_idx != c_idx and py_idx >= 0:
            print(f"    MISMATCH: type={a.type} value={[a.value[i] for i in range(5)]} "
                  f"py_idx={py_idx} c_idx={c_idx}")
            n_fail += 1
        else:
            n_pass += 1
    print(f"  Action encoding: {n_pass} pass, {n_fail} fail "
          f"(out of {len(sample_actions)})")
    return n_fail == 0


def test_one_game(seed, weights_path):
    from hexzero.bindings.lib_loader import load_library
    from hexzero.game.interface import CatanGame
    from hexzero.encoder.action_encoder import ActionEncoder

    # Load libcatan (for state)
    load_library()
    ae = ActionEncoder()

    # Load libpolicy's NN model (libpolicy has its own nn.c)
    mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    rc = _lib.nn_load(mptr, weights_path.encode())
    assert rc == 0, f"libpolicy nn_load failed: {rc}"

    # Load libnn (Python's NN model — separate .dylib but same weights)
    libnn = ctypes.CDLL(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "csrc", "libnn.dylib"))
    libnn.nn_load.restype = ctypes.c_int
    libnn.nn_forward.restype = None
    libnn.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    py_mbuf = (ctypes.c_char * (16 * 1024 * 1024))()
    py_mptr = ctypes.cast(py_mbuf, ctypes.c_void_p)
    rc = libnn.nn_load(py_mptr, weights_path.encode())
    assert rc == 0

    game = CatanGame(seed=seed); game.reset()
    se = game.make_state_encoder()
    enc = StateEncoderC()
    _lib.state_encoder_init(ctypes.byref(enc), ctypes.addressof(game._game), 4)

    # Scratch buffers for Python path
    scratch = (
        np.zeros((54, 18), dtype=np.float32),
        np.zeros((144, 5), dtype=np.float32),
        np.zeros(115, dtype=np.float32),
        np.zeros(397, dtype=np.float32),
        np.zeros(4 + 397, dtype=np.float32),
    )

    n_total = 0
    n_match = 0
    n_mismatch = 0

    # Sample actions from a few game stages for action encoding test
    sample_actions = []

    for target in [0, 7, 30, 60, 100]:
        steps = 0
        while steps < target and not game.is_terminal():
            le = game.get_legal_actions()
            if not le: break
            # Mostly play first action, occasionally last
            game.step(0)
            steps += 1
        if game.is_terminal():
            break

        for _ in range(5):  # test 5 different decisions per stage
            le = game.get_legal_actions()
            if not le or len(le) <= 1:
                # Force advance
                if le: game.step(0)
                continue

            sample_actions.extend(le[:3])

            for k in [3, 5, 10]:
                k_eff = min(k, len(le))
                py_idxs = py_top_k(libnn, py_mptr, se, ae, game, le, k_eff, scratch)
                c_idxs = c_top_k(enc, mptr, game, le, k_eff)

                n_total += 1
                if py_idxs == c_idxs:
                    n_match += 1
                else:
                    n_mismatch += 1
                    if n_mismatch <= 3:
                        print(f"    Mismatch turn={game.turn_number} k={k_eff}:")
                        print(f"      py: {py_idxs}")
                        print(f"      c : {c_idxs}")
            game.step(0)

    print(f"  Top-K results: {n_match}/{n_total} matched ({n_mismatch} mismatched)")

    # Action encoding spot check
    test_action_encoding(ae, mptr, sample_actions[:30])

    return n_mismatch == 0


def main():
    weights = os.path.abspath("csrc/nn_weights_m2.bin")
    print("=== Phase 2: C policy_top_k verification ===")
    print()
    seeds = [42, 95000, 80000]
    all_passed = True
    for seed in seeds:
        print(f"--- Seed {seed} ---")
        ok = test_one_game(seed, weights)
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
