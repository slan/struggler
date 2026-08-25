"""Evaluate the search player inside the repo: full engine games, no DLL.

    python -m wopr.search_eval --policy search=baselines/r3-bid2/v3/joshua.pt \
        --opponent greedy --games 200 --bid 2

The sanity check the pre-registered search experiment runs before any
Playdek time (docs/JOSHUA.md, 2026-08-25): `search=`/`veto=` against
Greedy or against the raw checkpoint (`joshua=<ckpt>`). Games go through
`struggler.runner.play_game` rather than the arena's batched backends --
the search player needs the live engine to `determinize()` -- so this is
slower per game than `wopr.eval` and meant for hundreds of games, not
tens of thousands. Seats alternate by game index, the engine seed is
`--seed + index`, and results are per-seat win rates with Wilson 95%
intervals, `wopr.playdek.eval`-style.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from struggler.engine import Engine, Side
from struggler.engine.player import Player

WORKER_THREADS = 2  # torch threads per worker: single-row inference


@dataclass(frozen=True)
class Job:
    index: int
    seed: int
    side: str  # the policy's seat
    policy: str
    opponent: str
    us_bid: int


def build_player(spec: str, *, seed: int) -> Player:
    if spec == "random":
        from struggler.bots.naive import RandomPlayer

        return RandomPlayer(seed=seed)
    if spec == "greedy":
        from struggler.bots.greedy import GreedyPlayer

        return GreedyPlayer()
    if spec == "first":
        from struggler.bots.naive import FirstLegalPlayer

        return FirstLegalPlayer()
    if "=" not in spec:
        raise ValueError(f"policy {spec!r}: expected random|greedy|first or search=|veto=|joshua=checkpoint.pt")
    name, checkpoint = spec.split("=", 1)
    if name in ("search", "veto"):
        from struggler.bots.joshua.search import SearchPlayer

        return SearchPlayer.from_checkpoint(
            checkpoint, evaluator="value" if name == "search" else "terminal", seed=seed
        )
    from struggler.bots.joshua.player import JoshuaPlayer

    return JoshuaPlayer.from_checkpoint(checkpoint, seed=seed)


def run_job(job: Job) -> dict:
    import torch

    from struggler.runner import play_game

    torch.set_num_threads(WORKER_THREADS)
    side = Side(job.side)
    engine = Engine.new_game(seed=job.seed, us_bid=job.us_bid)
    players = {
        side: build_player(job.policy, seed=job.seed * 7919 + 1),
        side.opponent: build_player(job.opponent, seed=job.seed * 104729 + 2),
    }
    for player in players.values():
        if hasattr(player, "bind"):
            player.bind(engine)
    start = time.monotonic()
    winner = play_game(engine, players)
    return {
        "index": job.index, "seed": job.seed, "side": job.side,
        "winner": winner.value if winner is not None else None,
        "turn": engine.turn, "seconds": time.monotonic() - start,
    }


def main(argv: list[str] | None = None) -> None:
    from wopr.playdek.eval import wilson  # the same interval on the same kind of tally

    p = argparse.ArgumentParser(description="The search player against an in-repo opponent.")
    p.add_argument("--policy", required=True, help="search=ckpt.pt | veto=ckpt.pt | joshua=ckpt.pt | greedy | random | first")
    p.add_argument("--opponent", default="greedy", help="same specs as --policy")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--side", choices=["ussr", "us", "both"], default="both")
    p.add_argument("--workers", type=int, default=None, help="games in parallel (default: CPUs / 2)")
    p.add_argument("--bid", type=int, default=0, help="the tournament bid (Engine.new_game(us_bid=N))")
    args = p.parse_args(argv)

    sides = {"ussr": ["USSR"], "us": ["US"], "both": ["USSR", "US"]}[args.side]
    jobs = [Job(i, args.seed + i, sides[i % len(sides)], args.policy, args.opponent, args.bid)
            for i in range(args.games)]
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 4) // 2)
    start = time.monotonic()
    results: list[dict] = []
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            results.append(run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for result in pool.map(run_job, jobs):
                results.append(result)
                if len(results) % 20 == 0:
                    print(f"[{len(results)}/{len(jobs)}]", flush=True)
    for side in ("USSR", "US", "both"):
        rows = results if side == "both" else [r for r in results if r["side"] == side]
        decided = [r for r in rows if r["winner"] is not None]
        wins = sum(1 for r in decided if r["winner"] == r["side"])
        if decided:
            lo, hi = wilson(wins, len(decided))
            print(f"{args.policy} as {side}: {wins}/{len(decided)} = {wins / len(decided):.3f} "
                  f"[{lo:.2f}, {hi:.2f}] vs {args.opponent}, mean turn "
                  f"{sum(r['turn'] for r in rows) / len(rows):.1f}")
    print(f"{len(results)} games, {(time.monotonic() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
