"""Mixed 4-way no-search league evaluation for many NN binaries.

Each default game seats exactly one candidate plus three anchors:

  - current incumbent
  - original M2 anchor
  - one historical league member

The no-incumbent mode instead seats:

  - candidate
  - original M2 anchor
  - two historical league members

For each candidate/league-member pair, every seed is replayed with the
candidate in all four seats.  The three non-candidate models rotate around the
remaining seats to reduce assignment artifacts.  The raw candidate win rate is
reported, along with a normalized score where 0.50 is the four-player baseline:

    score = 0.5 + 2 * (candidate_winrate - 0.25)
"""
from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

import numpy as np


AD = 337
MASK_DIM = 397
MODEL_BYTES = 16 * 1024 * 1024
FP = ctypes.POINTER(ctypes.c_float)

_G = {}


def _resolve_libnn_path():
    import platform
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hostname = platform.node().split(".")[0]
    candidates = [
        os.path.join(project_root, "csrc", f"libnn_{hostname}.so"),
        os.path.join(project_root, "csrc", "libnn.so"),
        os.path.join(project_root, "csrc", "libnn.dylib"),
        os.path.join(project_root, "catan_player", "libcatan_nn.so"),
        os.path.join(project_root, "catan_player", "libcatan_nn.dylib"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No libnn found, tried: {candidates}")


def _load_model(lib, weights):
    buf = (ctypes.c_char * MODEL_BYTES)()
    ptr = ctypes.cast(buf, ctypes.c_void_p)
    rc = lib.nn_load(ptr, weights.encode())
    if rc != 0:
        raise RuntimeError(f"nn_load failed for {weights}: {rc}")
    return buf, ptr


def _init_worker(candidate_weights, support_weights):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from hexzero.encoder.action_encoder import ActionEncoder
    from hexzero.game.interface import CatanGame

    lib = ctypes.CDLL(_resolve_libnn_path())
    lib.nn_load.restype = ctypes.c_int
    lib.nn_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.nn_forward.restype = None
    lib.nn_forward.argtypes = [ctypes.c_void_p, FP, FP, FP, FP, ctypes.c_void_p]

    candidate_models = [_load_model(lib, path) for path in candidate_weights]
    support_models = [_load_model(lib, path) for path in support_weights]
    _G.update({
        "lib": lib,
        "candidate_bufs": [buf for buf, _ptr in candidate_models],
        "candidate_ptrs": [ptr for _buf, ptr in candidate_models],
        "support_bufs": [buf for buf, _ptr in support_models],
        "support_ptrs": [ptr for _buf, ptr in support_models],
        "ae": ActionEncoder(),
        "CatanGame": CatanGame,
    })


def _play_one(job):
    cand_idx, member_idx, seed_idx, seed, candidate_seat = job
    CatanGame = _G["CatanGame"]
    ae = _G["ae"]
    lib = _G["lib"]

    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)
    out = np.zeros(4 + MASK_DIM, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)
    outp = out.ctypes.data_as(ctypes.c_void_p)

    cand_ptr = _G["candidate_ptrs"][cand_idx]
    # support[0] = incumbent, support[1] = original, support[2 + member] = league member
    support_ptrs = _G["support_ptrs"]
    others = [
        ("incumbent", support_ptrs[0]),
        ("league", support_ptrs[2 + member_idx]),
        ("original", support_ptrs[1]),
    ]
    rot = (seed_idx + candidate_seat + member_idx) % 3
    others = others[rot:] + others[:rot]

    model_by_seat = [None, None, None, None]
    label_by_seat = ["", "", "", ""]
    model_by_seat[candidate_seat] = cand_ptr
    label_by_seat[candidate_seat] = "candidate"
    for seat, (label, ptr) in zip([s for s in range(4) if s != candidate_seat], others):
        model_by_seat[seat] = ptr
        label_by_seat[seat] = label

    def nn_argmax(le, model_ptr):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        lib.nn_forward(model_ptr, nfp, efp, ffp, mkp, outp)
        logits = out[4:4 + AD].copy()
        logits[mn[:AD] < 0.5] = -1e9
        action_idx = int(np.argmax(logits))
        for i, action in enumerate(le):
            try:
                if ae.encode(action) == action_idx:
                    return i
            except ValueError:
                continue
        return 0

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        if len(legal) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        game.step(nn_argmax(legal, model_by_seat[cp]))

    winner = game.winner()
    return {
        "candidate": cand_idx,
        "member": member_idx,
        "seed_idx": seed_idx,
        "candidate_seat": candidate_seat,
        "winner": winner,
        "winner_label": label_by_seat[winner] if winner is not None else None,
        "candidate_won": winner == candidate_seat if winner is not None else False,
        "seat_labels": list(label_by_seat),
    }


def _play_one_no_incumbent(job):
    cand_idx, pair_idx, member_a_idx, member_b_idx, seed_idx, seed, candidate_seat = job
    CatanGame = _G["CatanGame"]
    ae = _G["ae"]
    lib = _G["lib"]

    game = CatanGame(seed=seed)
    game.reset()
    se = game.make_state_encoder()
    nf = np.zeros((se.num_nodes, se.NODE_FEATURE_DIM), dtype=np.float32)
    ef = np.zeros((se.num_edges, se.EDGE_FEATURE_DIM), dtype=np.float32)
    ff = np.zeros(se.FLAT_FEATURE_DIM, dtype=np.float32)
    mk = np.zeros(MASK_DIM, dtype=np.float32)
    out = np.zeros(4 + MASK_DIM, dtype=np.float32)
    nfp = nf.ctypes.data_as(FP)
    efp = ef.ctypes.data_as(FP)
    ffp = ff.ctypes.data_as(FP)
    mkp = mk.ctypes.data_as(FP)
    outp = out.ctypes.data_as(ctypes.c_void_p)

    cand_ptr = _G["candidate_ptrs"][cand_idx]
    # support[0] = original, support[1 + idx] = league member idx
    support_ptrs = _G["support_ptrs"]
    others = [
        ("original", support_ptrs[0]),
        ("league_a", support_ptrs[1 + member_a_idx]),
        ("league_b", support_ptrs[1 + member_b_idx]),
    ]
    rot = (seed_idx + candidate_seat + pair_idx) % 3
    others = others[rot:] + others[:rot]

    model_by_seat = [None, None, None, None]
    label_by_seat = ["", "", "", ""]
    model_by_seat[candidate_seat] = cand_ptr
    label_by_seat[candidate_seat] = "candidate"
    for seat, (label, ptr) in zip([s for s in range(4) if s != candidate_seat], others):
        model_by_seat[seat] = ptr
        label_by_seat[seat] = label

    def nn_argmax(le, model_ptr):
        se.encode_into(game.get_state_view(), nf, ef, ff)
        mn = ae.get_action_mask(le).numpy()
        mk[:] = 0
        mk[:len(mn)] = mn
        lib.nn_forward(model_ptr, nfp, efp, ffp, mkp, outp)
        logits = out[4:4 + AD].copy()
        logits[mn[:AD] < 0.5] = -1e9
        action_idx = int(np.argmax(logits))
        for i, action in enumerate(le):
            try:
                if ae.encode(action) == action_idx:
                    return i
            except ValueError:
                continue
        return 0

    while not game.is_terminal() and game.turn_number < 500:
        legal = game.get_legal_actions()
        if not legal:
            break
        if len(legal) == 1:
            game.step(0)
            continue
        cp = game.current_player()
        game.step(nn_argmax(legal, model_by_seat[cp]))

    winner = game.winner()
    return {
        "candidate": cand_idx,
        "pair": pair_idx,
        "member_a": member_a_idx,
        "member_b": member_b_idx,
        "seed_idx": seed_idx,
        "candidate_seat": candidate_seat,
        "winner": winner,
        "winner_label": label_by_seat[winner] if winner is not None else None,
        "candidate_won": winner == candidate_seat if winner is not None else False,
        "seat_labels": list(label_by_seat),
    }


def _normalized_score(raw_winrate: float) -> float:
    return 0.5 + 2.0 * (float(raw_winrate) - 0.25)


def _league_pairs(league_weights, max_pairings):
    if not league_weights:
        return [(0, 0)]
    if len(league_weights) == 1:
        return [(0, 0)]
    pairs = [(i, j) for i in range(len(league_weights)) for j in range(i + 1, len(league_weights))]
    if max_pairings and max_pairings > 0:
        pairs = pairs[:max_pairings]
    return pairs


def run_no_incumbent(candidate_weights, original_weights, league_weights,
                     games, workers, seed_base, max_pairings=0):
    candidate_weights = [os.path.abspath(p) for p in candidate_weights]
    original_weights = os.path.abspath(original_weights)
    league_weights = [os.path.abspath(p) for p in league_weights]
    if not league_weights:
        league_weights = [original_weights]
    support_weights = [original_weights] + league_weights
    pairs = _league_pairs(league_weights, max_pairings)

    jobs = []
    for ci in range(len(candidate_weights)):
        for pi, (mi, mj) in enumerate(pairs):
            for gi in range(games):
                seed = seed_base + pi * 100_000 + gi
                for candidate_seat in range(4):
                    jobs.append((ci, pi, mi, mj, gi, seed, candidate_seat))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(candidate_weights, support_weights),
    ) as pool:
        raw = list(pool.imap_unordered(_play_one_no_incumbent, jobs))
    elapsed = time.time() - t0

    results = []
    total_games = 4 * games * len(pairs)
    for ci, path in enumerate(candidate_weights):
        rows = [r for r in raw if r["candidate"] == ci]
        candidate_wins = sum(1 for r in rows if r["candidate_won"])
        no_winner = sum(1 for r in rows if r["winner"] is None)
        seat_wins = [
            sum(1 for r in rows if r["candidate_seat"] == seat and r["candidate_won"])
            for seat in range(4)
        ]
        per_pair = []
        for pi, (mi, mj) in enumerate(pairs):
            prows = [r for r in rows if r["pair"] == pi]
            pwins = sum(1 for r in prows if r["candidate_won"])
            pgames = len(prows)
            p_wr = pwins / max(1, pgames)
            per_pair.append({
                "pair": pi,
                "member_a": mi,
                "member_b": mj,
                "league_weights": [league_weights[mi], league_weights[mj]],
                "games": pgames,
                "candidate_wins": pwins,
                "no_winner": sum(1 for r in prows if r["winner"] is None),
                "candidate_winrate": p_wr,
                "score": _normalized_score(p_wr),
                "candidate_seat_wins": [
                    sum(1 for r in prows if r["candidate_seat"] == seat and r["candidate_won"])
                    for seat in range(4)
                ],
                "winner_labels": dict(Counter(r["winner_label"] for r in prows)),
            })
        wr = candidate_wins / max(1, total_games)
        results.append({
            "candidate": ci,
            "candidate_weights": path,
            "original_weights": original_weights,
            "league_members": len(league_weights),
            "league_pairings": len(pairs),
            "seed_count_per_pairing": games,
            "games": total_games,
            "candidate_wins": candidate_wins,
            "no_winner": no_winner,
            "candidate_winrate": wr,
            "score": _normalized_score(wr),
            "candidate_seat_wins": seat_wins,
            "per_pair": per_pair,
        })

    return {
        "seed_count_per_pairing": games,
        "games_per_candidate": total_games,
        "candidates": len(candidate_weights),
        "league_members": len(league_weights),
        "league_pairings": len(pairs),
        "elapsed_sec": elapsed,
        "mode": "no_fixed_incumbent",
        "results": results,
    }


def run(candidate_weights, incumbent_weights, original_weights, league_weights,
        games, workers, seed_base):
    candidate_weights = [os.path.abspath(p) for p in candidate_weights]
    incumbent_weights = os.path.abspath(incumbent_weights)
    original_weights = os.path.abspath(original_weights)
    league_weights = [os.path.abspath(p) for p in league_weights]
    if not league_weights:
        league_weights = [original_weights]
    support_weights = [incumbent_weights, original_weights] + league_weights

    jobs = []
    for ci in range(len(candidate_weights)):
        for mi in range(len(league_weights)):
            for gi in range(games):
                seed = seed_base + mi * 100_000 + gi
                for candidate_seat in range(4):
                    jobs.append((ci, mi, gi, seed, candidate_seat))

    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(candidate_weights, support_weights),
    ) as pool:
        raw = list(pool.imap_unordered(_play_one, jobs))
    elapsed = time.time() - t0

    results = []
    total_games = 4 * games * len(league_weights)
    for ci, path in enumerate(candidate_weights):
        rows = [r for r in raw if r["candidate"] == ci]
        candidate_wins = sum(1 for r in rows if r["candidate_won"])
        no_winner = sum(1 for r in rows if r["winner"] is None)
        seat_wins = [
            sum(1 for r in rows if r["candidate_seat"] == seat and r["candidate_won"])
            for seat in range(4)
        ]
        per_member = []
        for mi, league_path in enumerate(league_weights):
            mrows = [r for r in rows if r["member"] == mi]
            mwins = sum(1 for r in mrows if r["candidate_won"])
            mgames = len(mrows)
            m_wr = mwins / max(1, mgames)
            winner_labels = Counter(r["winner_label"] for r in mrows)
            incumbent_wins = int(winner_labels.get("incumbent", 0))
            incumbent_wr = incumbent_wins / max(1, mgames)
            per_member.append({
                "member": mi,
                "league_weights": league_path,
                "games": mgames,
                "candidate_wins": mwins,
                "incumbent_wins": incumbent_wins,
                "no_winner": sum(1 for r in mrows if r["winner"] is None),
                "candidate_winrate": m_wr,
                "incumbent_winrate": incumbent_wr,
                "candidate_vs_incumbent_winrate_delta": m_wr - incumbent_wr,
                "score": _normalized_score(m_wr),
                "candidate_seat_wins": [
                    sum(1 for r in mrows if r["candidate_seat"] == seat and r["candidate_won"])
                    for seat in range(4)
                ],
                "winner_labels": dict(winner_labels),
            })
        wr = candidate_wins / max(1, total_games)
        winner_labels = Counter(r["winner_label"] for r in rows)
        incumbent_wins = int(winner_labels.get("incumbent", 0))
        incumbent_wr = incumbent_wins / max(1, total_games)
        results.append({
            "candidate": ci,
            "candidate_weights": path,
            "incumbent_weights": incumbent_weights,
            "original_weights": original_weights,
            "league_members": len(league_weights),
            "seed_count_per_member": games,
            "games": total_games,
            "candidate_wins": candidate_wins,
            "incumbent_wins": incumbent_wins,
            "no_winner": no_winner,
            "candidate_winrate": wr,
            "incumbent_winrate": incumbent_wr,
            "candidate_vs_incumbent_winrate_delta": wr - incumbent_wr,
            "score": _normalized_score(wr),
            "candidate_seat_wins": seat_wins,
            "winner_labels": dict(winner_labels),
            "per_member": per_member,
        })

    return {
        "seed_count_per_member": games,
        "games_per_candidate": total_games,
        "candidates": len(candidate_weights),
        "league_members": len(league_weights),
        "elapsed_sec": elapsed,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-weight", action="append", required=True)
    parser.add_argument("--incumbent-weights")
    parser.add_argument("--original-weights", required=True)
    parser.add_argument("--league-weight", action="append", required=True)
    parser.add_argument("--no-fixed-incumbent", action="store_true")
    parser.add_argument("--max-pairings", type=int, default=0)
    parser.add_argument("--games", type=int, default=64,
                        help="seeds per league member; each seed runs all 4 candidate seats")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=3000000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.no_fixed_incumbent:
        res = run_no_incumbent(
            args.candidate_weight,
            args.original_weights,
            args.league_weight,
            args.games,
            args.workers,
            args.seed_base,
            args.max_pairings,
        )
    else:
        if not args.incumbent_weights:
            raise SystemExit("--incumbent-weights is required unless --no-fixed-incumbent is set")
        res = run(
            args.candidate_weight,
            args.incumbent_weights,
            args.original_weights,
            args.league_weight,
            args.games,
            args.workers,
            args.seed_base,
        )
    if args.json:
        print(json.dumps(res, sort_keys=True))
    else:
        print(f"=== eval_4way_league_many_nn_fast: {res['candidates']} candidates ===")
        for row in res["results"]:
            print(
                f"{row['candidate']:>3}: {row['candidate_wins']}/{row['games']} "
                f"raw={100 * row['candidate_winrate']:.1f}% "
                f"score={row['score']:.3f}  {row['candidate_weights']}")
        print(f"elapsed: {res['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
