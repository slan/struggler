"""Freeze a training run as a numbered baseline under `baselines/`.

    python -m wopr.baseline v1 --run signal
    python -m wopr.baseline v2 --run longer --games 200 --seeds 0 1 2

A baseline is what survives a run: `runs/` is gitignored scratch, while
`baselines/<version>/` is committed and holds the training trajectory
(`metrics.csv`), the configuration, the checkpoint itself, and a fixed
evaluation protocol's results. Keeping the checkpoint is the point --
every later version is evaluated *against* every earlier one, so the Elo
numbers form one chain across versions instead of a fresh scale per run.

The protocol, per evaluation seed: the new version plays `--games` games
(half on each seat, shared decks) against each anchor (`random`, `first`,
`greedy`) and against each earlier baseline, deterministically (argmax).
Seed 0 is additionally played with sampling, to see how much of the
strength is the argmax line. `summary.json` aggregates across seeds and
the script prints a ready-to-paste entry for `baselines/README.md`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from wopr.eval import PairJob, play_pairs
from wopr.ladder import Match, elo_ratings, summarize

BASELINES_DIR = Path("baselines")
RUNS_DIR = Path("runs")
RUN_FILES = ("config.json", "metrics.csv", "joshua.pt")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def earlier_baselines(version: str) -> list[tuple[str, Path]]:
    found = []
    for checkpoint in sorted(BASELINES_DIR.glob("v*/joshua.pt")):
        name = checkpoint.parent.name
        if name != version:
            found.append((name, checkpoint))
    return found


def star_jobs(
    version: str, opponents: Mapping[str, str], *, games: int, seed: int, deterministic: bool, device: str, events: bool
) -> list[PairJob]:
    """Star protocol: `version` against each other opponent, one job per pair."""
    return [
        PairJob(opponents[version], spec, games, seed, events=events, deterministic=deterministic, device=device)
        for name, spec in opponents.items()
        if name != version
    ]


def tabulate(
    version: str, jobs: Sequence[PairJob], results: Sequence[list[Match]], *, anchored: bool
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Per-opponent summaries (overall / as US / as USSR) and Elo with
    `random` anchored at 0 when it played."""
    matches: list[Match] = []
    table: dict[str, dict[str, float]] = {}
    for job, pair in zip(jobs, results):
        name = job.b
        matches.extend(pair)
        table[name] = {
            "win_rate": summarize(pair, version, name)["win_rate"],
            "as_us": summarize([m for m in pair if m.a == version], version, name)["win_rate"],
            "as_ussr": summarize([m for m in pair if m.b == version], version, name)["win_rate"],
            "games": summarize(pair, version, name)["games"],
        }
    return table, elo_ratings(matches, anchors={"random": 0.0} if anchored else None)


def render(version: str, table: Mapping[str, Mapping[str, float]], ratings: Mapping[str, float], *, header: str) -> str:
    lines = [header, ""]
    for name, row in table.items():
        lines.append(
            f"{version} vs {name}: {row['win_rate']:.3f} over {int(row['games'])} "
            f"| as US {row['as_us']:.3f} | as USSR {row['as_ussr']:.3f}"
        )
    lines.append("Elo: " + ", ".join(f"{n} {r:+.0f}" for n, r in sorted(ratings.items(), key=lambda kv: -kv[1])))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Freeze a run as a committed baseline with a fixed evaluation.")
    p.add_argument("version", help="baseline name, e.g. v1")
    p.add_argument("--run", required=True, help="run name under runs/")
    p.add_argument("--games", type=int, default=200, help="games per opponent per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--anchors", nargs="+", default=["random", "first", "greedy"])
    p.add_argument("--no-sampled", action="store_true", help="skip the sampled (stochastic) evaluation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-events", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing baseline folder")
    p.add_argument("--workers", type=int, default=None, help="pair-playing processes (default: CPUs / 4)")
    args = p.parse_args(argv)

    src = RUNS_DIR / args.run
    dst = BASELINES_DIR / args.version
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} exists; pass --force to overwrite")
    for name in RUN_FILES:
        if not (src / name).exists():
            raise SystemExit(f"{src / name} missing; is {args.run!r} a finished run?")
    dst.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        shutil.copy2(src / name, dst / name)

    opponents = {args.version: f"{args.version}={dst / 'joshua.pt'}"}
    opponents.update({name: name for name in args.anchors})
    opponents.update({name: f"{name}={path}" for name, path in earlier_baselines(args.version)})

    # Every (seed, argmax/sampled) pass is a star of independent pairs; all
    # of them go to one process pool at once.
    passes: list[tuple[int, bool]] = [(seed, True) for seed in args.seeds]
    if not args.no_sampled:
        passes.append((args.seeds[0], False))
    jobs_per_pass = [
        star_jobs(args.version, opponents, games=args.games, seed=seed, deterministic=deterministic,
                  device=args.device, events=not args.no_events)
        for seed, deterministic in passes
    ]
    started = time.perf_counter()
    results = play_pairs([job for jobs in jobs_per_pass for job in jobs], workers=args.workers)
    elapsed = time.perf_counter() - started
    anchored = "random" in opponents

    per_seed = []
    sampled = None
    offset = 0
    for (seed, deterministic), jobs in zip(passes, jobs_per_pass):
        table, ratings = tabulate(args.version, jobs, results[offset:offset + len(jobs)], anchored=anchored)
        offset += len(jobs)
        if deterministic:
            header = f"Deterministic (argmax), {args.games} games per opponent, eval seed {seed}"
            path = dst / f"eval_seed_{seed}.txt"
            per_seed.append({"seed": seed, "table": table, "elo": ratings})
        else:
            header = f"Sampled (stochastic), {args.games} games per opponent, eval seed {seed}"
            path = dst / f"eval_sampled_seed_{seed}.txt"
            sampled = {"seed": seed, "table": table, "elo": ratings}
        text = render(args.version, table, ratings, header=header)
        path.write_text(text)
        print(text, flush=True)
    print(f"{len(results)} pairs in {elapsed:.0f} s", flush=True)

    config = json.loads((dst / "config.json").read_text())
    opponents_seen = [n for n in opponents if n != args.version]

    def mean_std(values: list[float]) -> dict[str, float]:
        return {"mean": round(statistics.fmean(values), 4), "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0}

    summary = {
        "version": args.version,
        "commit": git_commit(),
        "run": args.run,
        "games_trained": config.get("games_done"),
        "protocol": {"games_per_opponent": args.games, "seeds": args.seeds, "anchors": args.anchors,
                     "events": not args.no_events},
        "elo": {n: mean_std([s["elo"][n] for s in per_seed]) for n in [args.version, *opponents_seen]},
        "win_rate": {n: {k: mean_std([s["table"][n][k] for s in per_seed]) for k in ("win_rate", "as_us", "as_ussr")}
                     for n in opponents_seen},
        "sampled": sampled,
    }
    (dst / "summary.json").write_text(json.dumps(summary, indent=2))

    elo = summary["elo"][args.version]
    vs = summary["win_rate"]
    print(f"## {args.version}\n")
    # ASCII only: this is pasted from a console whose code page may not be UTF-8.
    print(f"Commit {summary['commit']} -- run `{args.run}`, {summary['games_trained']} games trained\n")
    print("- (what changed)")
    print(f"- Elo vs random: {elo['mean']:+.0f} +/- {elo['std']:.0f} over seeds {args.seeds}")
    for name in opponents_seen:
        w = vs[name]
        print(f"- vs {name}: {w['win_rate']['mean']:.3f} (US {w['as_us']['mean']:.3f} / USSR {w['as_ussr']['mean']:.3f})")


if __name__ == "__main__":
    main()
