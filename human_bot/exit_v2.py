#!/usr/bin/env python3
"""ExIt v2: 5-round Expert Iteration on the fixed engine.

Each round:
  1. Export model to C weights
  2. Collect 1k ABt5 self-play games (16 parallel workers, C engine)
  3. Train 1 epoch (batch-at-a-time, RAM-safe)
  4. Quick eval: 50 games NNt10 vs AB2

Usage:
    python3 -u human_bot/exit_v2.py
    python3 -u human_bot/exit_v2.py --rounds 3 --games-per-round 500
"""

import argparse
import ctypes
import gc
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(PROJECT_ROOT, "csrc")


# ── Data collection worker ────────────────────────────────────────

def collect_one_game(args):
    seed, nn_weights, depth, top_k, seed_base, temperature, use_ab_value, flat_k = args

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    nn_lib_path = os.path.join(PROJECT_ROOT, "catan_player", "libcatan_nn.dylib")
    if not os.path.exists(nn_lib_path):
        nn_lib_path = os.path.join(CSRC, "libnn.dylib")
    nn_lib = ctypes.CDLL(nn_lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, nn_weights.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)

    def _enc(game): se.encode_into(game.get_state_view(), nf, ef, ff)
    def _mask(le):
        mk[:] = 0; mn = ae.get_action_mask(le).numpy(); mk[:len(mn)] = mn; return mn
    def c_topk(game, le, k):
        _enc(game); mn = _mask(le)
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; a2i = {ae.encode(a): i for i, a in enumerate(le)}
        return [li for _, li in sorted([(lo[e], li) for e, li in a2i.items()], reverse=True)[:k]]
    def c_argmax(gc):
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        _enc(gc); mn = _mask(le)
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        gc.step(next((i for i, a in enumerate(le) if ae.encode(a) == int(np.argmax(lo))), 0))

    def c_sample(gc, temp):
        """Sample from policy with temperature instead of argmax."""
        le = gc.get_legal_actions()
        if not le: return
        if len(le) == 1: gc.step(0); return
        _enc(gc); mn = _mask(le)
        out = np.zeros(4+397, dtype=np.float32)
        nn_lib.nn_forward(mptr, nfp, efp, ffp, mkp, out.ctypes.data_as(ctypes.c_void_p))
        lo = out[4:4+AD]; lo[mn < 0.5] = -1e9
        a2i = {ae.encode(a): i for i, a in enumerate(le)}
        legal_logits = np.array([lo[enc] for enc in a2i.keys()])
        legal_indices = list(a2i.values())
        legal_logits = legal_logits / temp
        legal_logits -= legal_logits.max()
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        chosen_pos = np.random.choice(len(legal_indices), p=probs)
        gc.step(legal_indices[chosen_pos])
    def c_val(game):
        _enc(game); _mask(game.get_legal_actions())
        nn_lib.nn_value_only(mptr, nfp, efp, ffp, mkp, vlp)
        return vl.copy()

    lib = load_library() if use_ab_value else None

    def ab_leaf_val(game, seat):
        cg = game._game; bc = cg.state.colors[seat]
        return float(lib.base_value_fn(ctypes.byref(cg), bc))

    def _flat_eval(game, root_seat, plies_left, k):
        if plies_left == 0 or game.is_terminal():
            if game.is_terminal():
                w = game.winner()
                return 10.0 if (w is not None and w == root_seat) else (-10.0 if w is not None else 0.0)
            if use_ab_value:
                return ab_leaf_val(game, root_seat)
            vs = c_val(game); off = (root_seat - game.current_player()) % 4
            return float(vs[off])
        le2 = game.get_legal_actions()
        if not le2: return 0.0
        if len(le2) == 1:
            gc2 = game.clone(); gc2.step(0)
            return _flat_eval(gc2, root_seat, plies_left - 1, k)
        cp = game.current_player()
        cands2 = list(range(len(le2)))
        if len(le2) > k:
            cands2 = c_topk(game, le2, k)
        best_ci, best_own = cands2[0], -1e30
        for ci2 in cands2:
            gc2 = game.clone(); gc2.step(ci2)
            own_v = ab_leaf_val(gc2, cp) if use_ab_value else float(c_val(gc2)[(cp - gc2.current_player()) % 4])
            if own_v > best_own: best_own = own_v; best_ci = ci2
        gc2 = game.clone(); gc2.step(best_ci)
        return _flat_eval(gc2, root_seat, plies_left - 1, k)

    def nn_search(game, le):
        """Search with configurable leaf eval and branching."""
        seat = game.current_player()
        cands = list(range(len(le)))
        if len(le) > top_k and depth >= 2:
            cands = c_topk(game, le, top_k)
        if flat_k > 0:
            bp, bv = 0, -1e30
            for p, ci in enumerate(cands):
                gc = game.clone(); gc.step(ci)
                v = _flat_eval(gc, seat, depth - 1, flat_k)
                v = apply_action_bonus(v, le[ci])
                if v > bv: bv = v; bp = p
            return fix_robber_steal(cands[bp], le)
        bp, bv = 0, -1e30
        for p, ci in enumerate(cands):
            gc = game.clone(); gc.step(ci)
            for ply in range(2, depth + 1):
                if gc.is_terminal(): break
                if temperature > 0:
                    c_sample(gc, temperature)
                else:
                    c_argmax(gc)
            if gc.is_terminal():
                w = gc.winner()
                v = 10.0 if (w is not None and w == seat) else (-10.0 if w is not None else 0.0)
            elif use_ab_value:
                v = ab_leaf_val(gc, seat)
            else:
                vs = c_val(gc)
                off = (seat - gc.current_player()) % 4
                v = float(vs[off])
            v = apply_action_bonus(v, le[ci])
            if v > bv: bv = v; bp = p
        return fix_robber_steal(cands[bp], le)

    game = CatanGame(seed=seed); game.reset()

    game_nf, game_ef, game_ff, game_mk = [], [], [], []
    game_action, game_player = [], []

    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if not le: break
        cp = game.current_player()
        if len(le) == 1: game.step(0); continue

        nf_rec = np.zeros((N, NF), dtype=np.float32)
        ef_rec = np.zeros((E, EF), dtype=np.float32)
        ff_rec = np.zeros(FFD, dtype=np.float32)
        se.encode_into(game.get_state_view(), nf_rec, ef_rec, ff_rec)
        mn = ae.get_action_mask(le).numpy()
        mk_rec = np.zeros(397, dtype=np.float32); mk_rec[:len(mn)] = mn

        chosen = nn_search(game, le)
        game_nf.append(nf_rec); game_ef.append(ef_rec)
        game_ff.append(ff_rec); game_mk.append(mk_rec)
        game_action.append(ae.encode(le[chosen])); game_player.append(cp)
        game.step(chosen)

    winner = game.winner()
    n = len(game_action)
    if n == 0: return None

    rewards = []
    for di in range(n):
        p = game_player[di]
        rv = np.zeros(4, dtype=np.float32)
        if winner is not None: rv[winner] = 1.0
        else: rv[:] = 0.25
        rv = np.roll(rv, (-p) % 4)
        rewards.append(rv)

    return {
        "nf": np.stack(game_nf), "ef": np.stack(game_ef),
        "ff": np.stack(game_ff), "mk": np.stack(game_mk),
        "action": np.array(game_action, dtype=np.int64),
        "reward": np.stack(rewards),
        "n": n, "winner": winner,
    }


# ── Eval worker ───────────────────────────────────────────────────

def eval_one_game(args):
    seed, nn_weights, nn_seats_list, ab2_seats_list, eval_depth = args

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS
    from human_bot.search_heuristics import apply_action_bonus, fix_robber_steal

    FP = ctypes.POINTER(ctypes.c_float)
    ae = ActionEncoder(); lib = load_library()
    g0 = CatanGame(seed=0); g0.reset(); se = g0.make_state_encoder()
    N, E = se.num_nodes, se.num_edges
    NF, EF, FFD = se.NODE_FEATURE_DIM, se.EDGE_FEATURE_DIM, se.FLAT_FEATURE_DIM
    AD = 337

    nn_lib_path = os.path.join(PROJECT_ROOT, "catan_player", "libcatan_nn.dylib")
    if not os.path.exists(nn_lib_path):
        nn_lib_path = os.path.join(CSRC, "libnn.dylib")
    nn_lib = ctypes.CDLL(nn_lib_path)
    nn_lib.nn_load.restype = ctypes.c_int
    nn_lib.nn_forward.restype = None
    nn_lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]
    nn_lib.nn_value_only.restype = None
    nn_lib.nn_value_only.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, FP]
    mbuf = (ctypes.c_char * (8 * 1024 * 1024))()
    mptr = ctypes.cast(mbuf, ctypes.c_void_p)
    assert nn_lib.nn_load(mptr, nn_weights.encode()) == 0

    nf = np.zeros((N, NF), dtype=np.float32)
    ef = np.zeros((E, EF), dtype=np.float32)
    ff = np.zeros(FFD, dtype=np.float32)
    mk = np.zeros(397, dtype=np.float32)
    vl = np.zeros(4, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP); efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP); mkp = mk.ctypes.data_as(FP); vlp = vl.ctypes.data_as(FP)
    ch = CGame(); ca = (Action * MAX_ACTIONS)(); cn = ctypes.c_int(0)

    def _enc(g): se.encode_into(g.get_state_view(), nf, ef, ff)
    def _mask(le):
        mk[:]=0; mn=ae.get_action_mask(le).numpy(); mk[:len(mn)]=mn; return mn
    def c_topk(g, le, k):
        _enc(g); mn=_mask(le)
        out=np.zeros(4+397,dtype=np.float32)
        nn_lib.nn_forward(mptr,nfp,efp,ffp,mkp,out.ctypes.data_as(ctypes.c_void_p))
        lo=out[4:4+AD]; a2i={ae.encode(a):i for i,a in enumerate(le)}
        return[li for _,li in sorted([(lo[e],li) for e,li in a2i.items()],reverse=True)[:k]]
    def c_argmax(gc):
        le=gc.get_legal_actions()
        if not le:return
        if len(le)==1:gc.step(0);return
        _enc(gc);mn=_mask(le)
        out=np.zeros(4+397,dtype=np.float32)
        nn_lib.nn_forward(mptr,nfp,efp,ffp,mkp,out.ctypes.data_as(ctypes.c_void_p))
        lo=out[4:4+AD];lo[mn<0.5]=-1e9
        gc.step(next((i for i,a in enumerate(le) if ae.encode(a)==int(np.argmax(lo))),0))
    def c_val(g):
        _enc(g);_mask(g.get_legal_actions())
        nn_lib.nn_value_only(mptr,nfp,efp,ffp,mkp,vlp);return vl.copy()
    def ab2_choose(g, le):
        cg=g._game; bc=cg.state.colors[cg.state.current_player_index]
        bi,bv=0,-1e30
        for i,act in enumerate(le):
            lib.game_copy(ctypes.byref(ch),ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch),act,ca,ctypes.byref(cn))
            v=lib.base_value_fn(ctypes.byref(ch),bc)
            if v>bv:bv=v;bi=i
        return bi
    def nn_argmax_idx(g, le):
        _enc(g);mn=_mask(le)
        out=np.zeros(4+397,dtype=np.float32)
        nn_lib.nn_forward(mptr,nfp,efp,ffp,mkp,out.ctypes.data_as(ctypes.c_void_p))
        lo=out[4:4+AD];lo[mn<0.5]=-1e9
        ai=int(np.argmax(lo))
        return next((i for i,a in enumerate(le) if ae.encode(a)==ai),0)

    def nnt(g, le, depth):
        if depth == 0:
            return nn_argmax_idx(g, le)
        seat=g.current_player();cands=list(range(len(le)))
        if len(le)>5 and depth>=2:cands=c_topk(g,le,5)
        bp,bv=0,-1e30
        for p,ci in enumerate(cands):
            gc=g.clone();gc.step(ci)
            for ply in range(2,depth+1):
                if gc.is_terminal():break
                c_argmax(gc)
            if gc.is_terminal():
                w=gc.winner();v=10.0 if(w is not None and w==seat)else(-10.0 if w is not None else 0.0)
            else:
                vs=c_val(gc);off=(seat-gc.current_player())%4;v=float(vs[off])
            v=apply_action_bonus(v,le[ci])
            if v>bv:bv=v;bp=p
        return fix_robber_steal(cands[bp],le)

    nn_seats=set(nn_seats_list); ab2_seats=set(ab2_seats_list)
    game=CatanGame(seed=seed);game.reset()
    while not game.is_terminal() and game.turn_number<1000:
        le=game.get_legal_actions()
        if not le:break
        if len(le)==1:game.step(0);continue
        cp=game.current_player()
        if cp in nn_seats: game.step(nnt(game,le,eval_depth))
        else: game.step(ab2_choose(game,le))
    w=game.winner()
    return "NN" if(w is not None and w in nn_seats) else("AB2" if w is not None else "draw")


# ── Training ──────────────────────────────────────────────────────

_ACT_MOD = np.ones(397, dtype=np.float32)
_ACT_MOD[0] = 0.2; _ACT_MOD[1] = 0.5
_ACT_MOD[2:5] = 1.5; _ACT_MOD[5:113] = 1.5; _ACT_MOD[113:185] = 1.5
_ACT_MOD[185:280] = 1.5; _ACT_MOD[280:285] = 0.3
_ACT_MOD[285:310] = 1.5; _ACT_MOD[310:397] = 1.3


def _compute_step_weights(data):
    """Selfplay-style per-example weights: winner progression + action-type mods."""
    S = data["action"].shape[0]
    rewards = data["reward"]
    actions = data["action"]
    weights = np.ones(S, dtype=np.float32)
    game_starts = np.where(np.diff(np.concatenate([[0], data.get("_game_id", np.zeros(S))])))[0]
    if len(game_starts) == 0:
        game_starts = np.array([0])
    for gs_idx in range(len(game_starts)):
        start = game_starts[gs_idx]
        end = game_starts[gs_idx + 1] if gs_idx + 1 < len(game_starts) else S
        seg_len = end - start
        for j in range(seg_len):
            i = start + j
            progress = j / max(seg_len - 1, 1)
            is_winner = rewards[i, 0] > 0.9
            if is_winner:
                base = 1.0 + 0.5 * progress
            else:
                base = max(0.3, 0.6 - 0.3 * progress)
            act_idx = min(int(actions[i]), 396)
            weights[i] = base * _ACT_MOD[act_idx]
    return weights


def train_one_epoch(net, data, edge_index, device, lr, batch_size, winner_boost=2.0):
    import math
    from human_bot.loss import UncertaintyWeightedLoss, human_policy_loss, value_loss, masked_entropy

    loss_combiner = UncertaintyWeightedLoss().to(device)
    all_params = list(net.parameters()) + list(loss_combiner.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)

    S = data["nf"].shape[0]
    n_steps = max(1, S // batch_size)
    warmup = min(50, n_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(n_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step_weights = _compute_step_weights(data)

    net.train()
    perm = np.random.permutation(S)
    sums = {"ploss": 0, "vloss": 0, "pacc": 0, "vacc": 0}
    nb = 0

    for i in range(0, S, batch_size):
        idx = perm[i:i+batch_size]
        if len(idx) < 16: continue
        nf_b = torch.from_numpy(data["nf"][idx])
        ef_b = torch.from_numpy(data["ef"][idx])
        ff_b = torch.from_numpy(data["ff"][idx])
        mk_b = torch.from_numpy(data["mk"][idx])
        act_b = torch.from_numpy(data["action"][idx])
        vt_b = torch.from_numpy(data["reward"][idx])
        sw_b = torch.from_numpy(step_weights[idx])

        out = net({"node_features": nf_b, "edge_index": edge_index,
                   "edge_features": ef_b, "flat_features": ff_b, "action_mask": mk_b})
        p_loss = human_policy_loss(out["policy_logits"], act_b, mk_b,
                                   label_smoothing=0.02, winner_boost=sw_b)
        tp = ff_b[:, 114] if ff_b.shape[1] > 114 else None
        v_loss = value_loss(out["value"], vt_b, turn_progress=tp)
        ent = masked_entropy(out["policy_logits"], mk_b)
        total, _ = loss_combiner(p_loss, v_loss, ent, 0.01)

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            pacc = (out["policy_logits"].argmax(-1) == act_b).float().mean().item()
            vacc = (out["value"].argmax(-1) == vt_b.argmax(-1)).float().mean().item()
        sums["ploss"] += p_loss.item(); sums["vloss"] += v_loss.item()
        sums["pacc"] += pacc; sums["vacc"] += vacc; nb += 1

        del nf_b, ef_b, ff_b, mk_b, act_b, vt_b, sw_b, out
        if nb % 5 == 0: gc.collect()

    return {k: v / max(nb, 1) for k, v in sums.items()}


# ── Main ──────────────────────────────────────────────────────────

def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ckpt-dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "v2"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--games-per-round", type=int, default=1000)
    parser.add_argument("--search-depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Initial rollout temperature (0 = argmax)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--ab-value", action="store_true",
                        help="Use base_value_fn at leaf instead of NN value")
    parser.add_argument("--flat-k", type=int, default=0,
                        help="Flat search width per ply (0 = tapered argmax)")
    parser.add_argument("--winner-boost", type=float, default=2.0,
                        help="Weight multiplier for winner's actions in policy loss")
    args = parser.parse_args()

    sys.path.insert(0, PROJECT_ROOT)
    from human_bot.model import HumanBotNet
    from human_bot.export_nn import export
    from hexzero.game.interface import CatanGame

    device = "cpu"
    current_ckpt = args.checkpoint
    g0 = CatanGame(seed=0); g0.reset()
    edge_index = g0.make_state_encoder()._edge_index.to(device)

    t_total = time.perf_counter()

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ExIt Round {rnd}/{args.rounds}")
        print(f"{'='*60}")

        # 1. Export to C weights
        weights_path = os.path.join(CSRC, f"nn_weights_v2_r{rnd}.bin")
        print(f"\n  [1/4] Exporting {current_ckpt} -> {os.path.basename(weights_path)}")
        export(current_ckpt, weights_path)

        # 2. Collect self-play data
        seed_base = 700000 + rnd * 100000
        temp = max(0.0, args.temperature * (1.0 - (rnd - 1) / max(args.rounds - 1, 1)))
        sfx = f"f{args.flat_k}" if args.flat_k > 0 else ""
        vfn = "AB" if args.ab_value else "NN"
        print(f"  [2/4] Collecting {args.games_per_round} {vfn}t{args.search_depth}{sfx} games "
              f"(temp={temp:.2f}, {args.workers} workers, wb={args.winner_boost})...")
        jobs = [(seed_base + gi, weights_path, args.search_depth, args.top_k, seed_base, temp,
                 args.ab_value, args.flat_k)
                for gi in range(args.games_per_round)]
        t0 = time.perf_counter()
        with mp.Pool(args.workers) as pool:
            results = pool.map(collect_one_game, jobs)
        collect_time = time.perf_counter() - t0

        results = [r for r in results if r is not None]
        total_dec = sum(r["n"] for r in results)
        avg_turns = np.mean([r["n"] for r in results]) if results else 0
        print(f"         {total_dec:,} decisions from {len(results)} games, "
              f"avg {avg_turns:.0f} dec/game, {collect_time:.0f}s")

        data = {
            "nf": np.concatenate([r["nf"] for r in results]),
            "ef": np.concatenate([r["ef"] for r in results]),
            "ff": np.concatenate([r["ff"] for r in results]),
            "mk": np.concatenate([r["mk"] for r in results]),
            "action": np.concatenate([r["action"] for r in results]),
            "reward": np.concatenate([r["reward"] for r in results]),
        }
        del results; gc.collect()

        # 3. Train
        print(f"  [3/4] Training on {data['nf'].shape[0]:,} samples "
              f"({args.epochs} epoch(s), lr={args.lr})")
        net = HumanBotNet.load_checkpoint(current_ckpt, device=device)
        for ep in range(args.epochs):
            avg = train_one_epoch(net, data, edge_index, device, args.lr, args.batch_size,
                                 winner_boost=args.winner_boost)
        print(f"         ploss={avg['ploss']:.3f} pacc={avg['pacc']:.3f} "
              f"vloss={avg['vloss']:.3f} vacc={avg['vacc']:.3f}")
        del data; gc.collect()

        out_ckpt = os.path.join(args.ckpt_dir, f"r{rnd}.pt")
        net.eval()
        net.save_checkpoint(out_ckpt, {
            "method": f"exit_v2_r{rnd}", "base": current_ckpt,
            "games": args.games_per_round, "depth": args.search_depth,
        })
        current_ckpt = out_ckpt
        print(f"         Saved: {out_ckpt}")

        # Re-export for eval
        export(current_ckpt, weights_path)

        # 4. Multi-depth eval vs AB2
        if args.eval_games > 0:
            print(f"  [4/4] Eval: {args.eval_games} games vs AB2 at each depth...")
            depth_results = {}
            for eval_depth in [0, 1, 2, 5, 10]:
                eval_jobs = []
                for gi in range(args.eval_games):
                    seed = 900000 + rnd * 50000 + eval_depth * 1000 + gi
                    eval_jobs.append((seed, weights_path,
                                      [gi % 4, (gi + 2) % 4],
                                      [(gi + 1) % 4, (gi + 3) % 4],
                                      eval_depth))
                with mp.Pool(args.workers) as pool:
                    eval_results = pool.map(eval_one_game, eval_jobs)
                nn_w = sum(1 for r in eval_results if r == "NN")
                depth_results[eval_depth] = nn_w
            parts = " | ".join(f"t{d}={depth_results[d]}/{args.eval_games}"
                               for d in [0, 1, 2, 5, 10])
            print(f"         {parts}")

        del net; gc.collect()

    total = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  All {args.rounds} rounds complete in {total:.0f}s ({total/60:.1f} min)")
    print(f"  Final model: {current_ckpt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
