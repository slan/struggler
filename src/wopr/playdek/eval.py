"""Evaluate a policy against Playdek's AI.

    python -m wopr.playdek.eval --games 20 --policy joshua=baselines/v16/joshua.pt --difficulty hard --out runs/playdek/v16-hard

Each game is one `operator.play_match`: the DLL's AI on one seat, the
policy on the other, the engine refereeing in physical mode. Games are
independent and the DLL is one game per process, so they run in a process
pool (`--workers`), one Playdek instance per worker; the AI's 15 s per
decision (docs/WOPR.md) makes a game ~3-5 minutes of one core. The seats
alternate by game index (`--side both`), the seed is `--seed + index`.

Every game's replay log goes to `<out>/games/` (`GameLogWriter`, the
engine's record of the game), every result to `<out>/results.jsonl`, and
the tally to `<out>/summary.json` and stdout: the policy's win rate per
seat with a Wilson 95% interval, how many games desynced (the two
programs disagreed fatally; those games do not count) and how many are
void (ended on a documented difference between the two rule sets, by
reason; they do not count either, and are not desyncs). The AI is not
deterministic for a seed, so this is a sample, not a replay.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from struggler.engine.player import Player
from struggler.engine.types import Side
from wopr.playdek.ffi import AIDifficulty

WORKER_THREADS = 2  # torch threads per worker: the game is DLL-bound, the inference a single row


@dataclass(frozen=True)
class Job:
    index: int
    seed: int
    side: str  # the policy's seat
    policy: str  # random | greedy | first | name=checkpoint.pt
    difficulty: str  # easy | hard | hotseat (a random DLL policy on the other seat, for testing)
    out: str | None
    deterministic: bool = True
    max_divergences: int = 40
    trace: bool = False
    us_bid: int = 0  # the tournament bid, on both boards


def build_player(spec: str, *, seed: int, deterministic: bool) -> Player:
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
        raise ValueError(f"policy {spec!r}: expected random|greedy|first or name=checkpoint.pt")
    name, checkpoint = spec.split("=", 1)
    if name in ("search", "veto"):
        # Inference-time lookahead over the checkpoint's value head
        # (docs/WOPR.md, "Search over the learned value head"); `veto` is
        # its terminal-only ablation. `play_match` binds the engine.
        from struggler.bots.joshua.search import SearchPlayer

        return SearchPlayer.from_checkpoint(
            checkpoint, evaluator="value" if name == "search" else "terminal", seed=seed
        )
    from struggler.bots.joshua.player import JoshuaPlayer

    return JoshuaPlayer.from_checkpoint(checkpoint, deterministic=deterministic, seed=seed)


def run_job(job: Job) -> dict:
    import random

    import torch

    from wopr.playdek.game import Playdek
    from wopr.playdek.lockstep import random_policy
    from wopr.playdek.operator import play_match

    torch.set_num_threads(WORKER_THREADS)
    pd = Playdek._instance or Playdek()
    side = Side(job.side)
    player = build_player(job.policy, seed=job.seed * 7919 + 1, deterministic=job.deterministic)
    log_path = None
    if job.out:
        games = Path(job.out) / "games"
        games.mkdir(parents=True, exist_ok=True)
        log_path = str(games / f"{job.index:04d}_seed{job.seed}_{job.side}.json")
    if job.difficulty == "hotseat":
        result = play_match(pd, player, seed=job.seed, side=side, emulate=random_policy(random.Random(job.seed)),
                            log_path=log_path, max_divergences=job.max_divergences, trace=job.trace, us_bid=job.us_bid)
    else:
        result = play_match(pd, player, seed=job.seed, side=side, difficulty=AIDifficulty[job.difficulty.upper()],
                            log_path=log_path, max_divergences=job.max_divergences, trace=job.trace, us_bid=job.us_bid)
    return {"index": job.index, **dataclasses.asdict(result)}


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def summarize(results: list[dict]) -> dict:
    out: dict = {"games": len(results), "desyncs": sum(1 for r in results if r["desync"]),
                 "void": dict(collections.Counter(r["void"] for r in results if r.get("void"))), "by_side": {}}
    clean = [r for r in results if not r["desync"] and not r.get("void") and r["playdek_winner"] is not None]
    for side in ("USSR", "US", "both"):
        rows = clean if side == "both" else [r for r in clean if r["side"] == side]
        wins = sum(1 for r in rows if r["playdek_winner"] == r["side"])
        lo, hi = wilson(wins, len(rows))
        out["by_side"][side] = {"games": len(rows), "wins": wins, "win_rate": wins / len(rows) if rows else None,
                               "wilson95": [lo, hi], "mean_turn": sum(r["turn"] for r in rows) / len(rows) if rows else None}
    out["win_types"] = dict(collections.Counter(r["win_type"] for r in clean))
    out["known"] = dict(collections.Counter(d.split("known: ", 1)[1].rsplit(" (", 1)[0] for r in results for d in r["divergences"] if "known: " in d))
    out["fatal"] = [d for r in results for d in r["divergences"] if "FATAL" in d]
    out["seconds"] = sum(r["seconds"] for r in results)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="A policy against Playdek's AI.")
    p.add_argument("--policy", default="greedy",
                   help="random | greedy | first | name=checkpoint.pt (search=/veto= for the lookahead player)")
    p.add_argument("--games", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--side", choices=["ussr", "us", "both"], default="both", help="the policy's seat (both: alternating by game)")
    p.add_argument("--difficulty", choices=["easy", "hard", "hotseat"], default="hard", help="hotseat: a random DLL policy instead of the AI (fast; for testing)")
    p.add_argument("--workers", type=int, default=None, help="games in parallel (default: CPUs / 4)")
    p.add_argument("--sample", action="store_true", help="sample the network policy instead of taking its argmax")
    p.add_argument("--out", default=None, help="directory for the replay logs, results.jsonl and summary.json")
    p.add_argument("--max-divergences", type=int, default=40)
    p.add_argument("--trace", action="store_true", help="print every record, prompt, reply and inference (one game at a time: --workers 1)")
    p.add_argument("--bid", type=int, default=0, help="the tournament bid: the US places this much extra influence after setup, on the DLL's board and the engine's")
    args = p.parse_args(argv)

    sides = {"ussr": ["USSR"], "us": ["US"], "both": ["USSR", "US"]}[args.side]
    jobs = [Job(i, args.seed + i, sides[i % len(sides)], args.policy, args.difficulty, args.out, not args.sample, args.max_divergences, args.trace, us_bid=args.bid)
            for i in range(args.games)]
    workers = args.workers if args.workers is not None else max(1, (os.cpu_count() or 4) // 4)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    start = time.monotonic()
    results: list[dict] = []

    def report(r: dict) -> None:
        results.append(r)
        won = "won" if r["playdek_winner"] == r["side"] else "lost" if r["playdek_winner"] else "draw"
        print(f"[{len(results)}/{len(jobs)}] seed {r['seed']} as {r['side']}: {'DESYNC' if r['desync'] else 'VOID' if r.get('void') else won} "
              f"({r['win_type']}, score {r['score']}, turn {r['turn']}, {r['seconds']:.0f}s)", flush=True)
        for d in r["divergences"]:
            # known counters and the grain/hand-drift evidence lines stay in
            # results.jsonl; the console shows what needs eyes.
            if "known: " not in d and "] grain: " not in d and "] hand-drift: " not in d:
                print("    " + d, flush=True)
        if out:
            with (out / "results.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            report(run_job(job))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            by_future = {pool.submit(run_job, job): job for job in jobs}
            for future in as_completed(by_future):
                try:
                    result = future.result()
                except Exception as e:  # one crashed game must not kill the batch
                    job = by_future[future]
                    print(f"[!] seed {job.seed} as {job.side}: CRASHED ({e!r})", flush=True)
                    continue
                report(result)
    results.sort(key=lambda r: r["index"])
    summary = summarize(results)
    summary["wall_seconds"] = time.monotonic() - start
    summary["policy"], summary["difficulty"] = args.policy, args.difficulty
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for side, s in summary["by_side"].items():
        if s["games"]:
            lo, hi = s["wilson95"]
            print(f"{args.policy} as {side}: {s['wins']}/{s['games']} = {s['win_rate']:.3f} [{lo:.2f}, {hi:.2f}], mean turn {s['mean_turn']:.1f}")
    print(f"desyncs {summary['desyncs']}/{summary['games']}; void {sum(summary['void'].values())}; win types {summary['win_types']}; "
          f"{summary['wall_seconds'] / 60:.1f} min")
    for what, n in summary["known"].items():
        print(f"  known: {what} ({n}x)")


if __name__ == "__main__":
    main()
