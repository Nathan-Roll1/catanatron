"""Diagnose ef mismatch."""
import ctypes
import numpy as np
from human_bot._test_c_encoder import StateEncoderC, _lib, c_encode

from hexzero.bindings.lib_loader import load_library
from hexzero.game.interface import CatanGame

load_library()
game = CatanGame(seed=42); game.reset()
se = game.make_state_encoder()
enc = StateEncoderC()
_lib.state_encoder_init(ctypes.byref(enc), ctypes.addressof(game._game), 4)

# Step a few turns to get some roads on the board
for _ in range(10):
    if game.is_terminal(): break
    le = game.get_legal_actions()
    if not le: break
    game.step(0)

print(f"Turn {game.turn_number}, cp={game.current_player()}")

# Compare the road owner lookups directly
state_view = game.get_state_view()
print(f"\n_road_src[:10] = {se._road_src[:10]}")
print(f"_road_adj[:10] = {se._road_adj[:10]}")

# C-side pre-computed tables
c_rsrc = np.array([enc.road_src_global[i] for i in range(enc.E)])
c_radj = np.array([enc.road_adj_idx[i] for i in range(enc.E)])
print(f"\nC road_src_global[:10] = {c_rsrc[:10]}")
print(f"C road_adj_idx[:10] = {c_radj[:10]}")

print(f"\nMatch: src={np.array_equal(c_rsrc, se._road_src)}  adj={np.array_equal(c_radj, se._road_adj)}")
if not np.array_equal(c_rsrc, se._road_src):
    diff = np.where(c_rsrc != se._road_src)[0]
    print(f"  First src diff: idx={diff[0]} c={c_rsrc[diff[0]]} py={se._road_src[diff[0]]}")
if not np.array_equal(c_radj, se._road_adj):
    diff = np.where(c_radj != se._road_adj)[0]
    print(f"  First adj diff: idx={diff[0]} c={c_radj[diff[0]]} py={se._road_adj[diff[0]]}")

# Now compare actual road-owner reads
print(f"\nstate.road_owners[:5, :] =\n{state_view.road_owners[:5]}")
print(f"\nLook at edge 0:")
print(f"  py: src={se._road_src[0]} adj={se._road_adj[0]}")
print(f"      road_owners[{se._road_src[0]}, {se._road_adj[0]}] = {state_view.road_owners[se._road_src[0], se._road_adj[0]]}")
print(f"  c:  src={c_rsrc[0]} adj={c_radj[0]}")
print(f"      board.road_owner[{c_rsrc[0]}][{c_radj[0]}] = {game._game.state.board.road_owner[c_rsrc[0]][c_radj[0]]}")

# Now actually run both encoders and look at edge 0
nf_c, ef_c, flat_c = c_encode(enc, game)
nf_py = np.zeros((54, 18), dtype=np.float32)
ef_py = np.zeros((144, 5), dtype=np.float32)
flat_py = np.zeros(115, dtype=np.float32)
se.encode_into(state_view, nf_py, ef_py, flat_py)

print(f"\nef[0] (edge 0):")
print(f"  C : {ef_c[0]}")
print(f"  py: {ef_py[0]}")
print(f"\nef[1] (edge 1, reverse direction):")
print(f"  C : {ef_c[1]}")
print(f"  py: {ef_py[1]}")

# Where do they actually differ?
diffs = (ef_c != ef_py).any(axis=1)
n_diff = diffs.sum()
print(f"\n{n_diff} of {len(ef_c)} edges differ")
if n_diff > 0:
    diff_idxs = np.where(diffs)[0][:5]
    for i in diff_idxs:
        print(f"  edge {i}: src_g(py)={se._road_src[i]} adj(py)={se._road_adj[i]}  "
              f"src_g(c)={c_rsrc[i]} adj(c)={c_radj[i]}")
        print(f"    py ef[{i}]={ef_py[i]}")
        print(f"    c  ef[{i}]={ef_c[i]}")
        print(f"    road_owners(py)[{se._road_src[i]}, {se._road_adj[i]}] = {state_view.road_owners[se._road_src[i], se._road_adj[i]]}")
