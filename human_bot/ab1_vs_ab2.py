"""AB1 (1-ply greedy) vs AB2 (2-ply greedy) benchmark.

Usage:
    python3 -u human_bot/ab1_vs_ab2.py --games 100 --workers 8
"""
import argparse
import ctypes
import multiprocessing as mp
import time

def play_one(args):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    seed, ab1_seats, ab2_seats = args
    from hexzero.game.interface import CatanGame
    from hexzero.bindings.lib_loader import load_library
    from hexzero.bindings.structs import Game as CGame, Action, MAX_ACTIONS
    lib = load_library()

    ch = CGame()
    ca = (Action * MAX_ACTIONS)()
    cn = ctypes.c_int(0)
    ch2 = CGame()
    ca2 = (Action * MAX_ACTIONS)()
    cn2 = ctypes.c_int(0)

    def ab_choose(game, le, depth):
        cg = game._game
        bc = cg.state.colors[cg.state.current_player_index]
        bi, bv = 0, -1e30
        for i, act in enumerate(le):
            lib.game_copy(ctypes.byref(ch), ctypes.byref(cg))
            lib.game_execute(ctypes.byref(ch), act, ca, ctypes.byref(cn))
            if depth >= 2 and cn.value > 0:
                if cn.value > 1:
                    best_resp, best_rv = 0, -1e30
                    bc2 = ch.state.colors[ch.state.current_player_index]
                    for j in range(cn.value):
                        lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                        lib.game_execute(ctypes.byref(ch2), ca[j], ca2, ctypes.byref(cn2))
                        rv = lib.base_value_fn(ctypes.byref(ch2), bc2)
                        if rv > best_rv: best_rv = rv; best_resp = j
                    lib.game_copy(ctypes.byref(ch2), ctypes.byref(ch))
                    lib.game_execute(ctypes.byref(ch2), ca[best_resp], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch2), bc)
                else:
                    lib.game_execute(ctypes.byref(ch), ca[0], ca2, ctypes.byref(cn2))
                    v = lib.base_value_fn(ctypes.byref(ch), bc)
            else:
                v = lib.base_value_fn(ctypes.byref(ch), bc)
            if v > bv: bv = v; bi = i
        return bi

    ab1_set = set(ab1_seats)
    game = CatanGame(seed=seed)
    game.reset()
    while not game.is_terminal() and game.turn_number < 1000:
        le = game.get_legal_actions()
        if len(le) == 1:
            game.step(0)
        elif game.current_player() in ab1_set:
            game.step(ab_choose(game, le, 1))
        else:
            game.step(ab_choose(game, le, 2))
    w = game.winner()
    if w is None:
        return "draw"
    return "AB1" if w in ab1_set else "AB2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=200000)
    args = parser.parse_args()

    jobs = []
    for gi in range(args.games):
        ab1_s = [gi % 4, (gi + 2) % 4]
        ab2_s = [(gi + 1) % 4, (gi + 3) % 4]
        jobs.append((args.seed_base + gi, ab1_s, ab2_s))

    print(f"Running {args.games} games: AB1 (1-ply) vs AB2 (2-ply), 2 seats each, {args.workers} workers...",
          flush=True)
    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        results = pool.map(play_one, jobs)
    wall = time.perf_counter() - t0

    ab1_w = sum(1 for r in results if r == "AB1")
    ab2_w = sum(1 for r in results if r == "AB2")
    draws = sum(1 for r in results if r == "draw")
    print(f"\n{'='*50}")
    print(f"  AB1 (1-ply) vs AB2 (2-ply) — {args.games} games")
    print(f"{'='*50}")
    print(f"  AB1: {ab1_w} wins ({100*ab1_w/args.games:.0f}%)")
    print(f"  AB2: {ab2_w} wins ({100*ab2_w/args.games:.0f}%)")
    print(f"  Draws: {draws}")
    print(f"  Wall time: {wall:.1f}s ({args.games/wall*60:.0f} games/min)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
