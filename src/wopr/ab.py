"""A clean run and its comparison, in one command and one ledger row.

    python -m wopr.ab --run fix-xyz --control v11 --champion v16

trains `runs/<run>/` from scratch with a recipe (default `v11`: 8,000
games, hidden 256, 4 epochs, self-play and the pool), then plays it --
argmax, every eval seed -- against the control (the frozen version trained
with the same recipe and budget: the number that says whether what
changed between the two commits changed the learned game), the champion,
Greedy, and itself (its USSR edge), and writes the result to
`runs/<run>/ab.json` **and** a row in `baselines/EXPERIMENTS.md`, the
committed ledger of every experiment whether or not it was frozen.
Anything after `--` goes to `train.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from wopr import baseline, train
from wopr.eval import PairJob, play_pairs
from wopr.ladder import summarize
from wopr.repo import git_commit

LEDGER = baseline.BASELINES_DIR / "EXPERIMENTS.md"
LEDGER_HEADER = (
    "# Experiments\n\n"
    "One row per `wopr.ab` run, frozen or not (`baselines/README.md` has the\n"
    "frozen ones in full). Win rates are the run's, argmax play, mean over the\n"
    "eval seeds with the worst seed in brackets; `USSR edge` is the run against\n"
    "itself, as USSR. `rules` is the engine's rules version the row was\n"
    "measured on (ratings do not cross it). The reading of each row: docs/JOSHUA.md.\n\n"
    "| date | rules | run | commit | recipe | games | vs control | vs champion | vs greedy | USSR edge | note |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def compare(
    checkpoint: Path, opponents: Mapping[str, str], *, games: int, seeds: Sequence[int], workers: int | None
) -> dict[str, Any]:
    """`checkpoint` (as `run`) against each opponent spec on every seed, plus
    itself on the first seed: per opponent the mean and worst-seed win rate
    and the per-seat split; `ussr_edge` for the self pair."""
    me = f"run={checkpoint}"
    pairs = [(name, seed) for seed in seeds for name in opponents]
    jobs = [PairJob(me, opponents[name], games, seed) for name, seed in pairs]
    jobs.append(PairJob(me, f"self={checkpoint}", games, seeds[0]))
    results = play_pairs(jobs, workers=workers)
    return tabulate(pairs, results[:-1], results[-1])


def tabulate(pairs: Sequence[tuple[str, int]], results: Sequence[list], self_pair: list) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for (name, _seed), pair in zip(pairs, results):
        entry = report.setdefault(name, {"per_seed": []})
        entry["per_seed"].append(
            {
                "win_rate": summarize(pair, "run", name)["win_rate"],
                "as_us": summarize([m for m in pair if m.a == "run"], "run", name)["win_rate"],
                "as_ussr": summarize([m for m in pair if m.b == "run"], "run", name)["win_rate"],
            }
        )
    for entry in report.values():
        for key in ("win_rate", "as_us", "as_ussr"):
            entry[key] = statistics.fmean(r[key] for r in entry["per_seed"])
        entry["min_seed"] = min(r["win_rate"] for r in entry["per_seed"])
    report["ussr_edge"] = summarize([m for m in self_pair if m.b == "run"], "run", "self")["win_rate"]
    return report


def ledger_row(summary: Mapping[str, Any]) -> str:
    def cell(name: str) -> str:
        entry = summary["results"].get(name)
        if entry is None:
            return "—"
        return f"{entry['win_rate']:.3f} [{entry['min_seed']:.3f}] (US {entry['as_us']:.2f} / USSR {entry['as_ussr']:.2f})"

    return (
        f"| {summary['date']} | r{summary['rules_version']} | `{summary['run']}` | `{summary['commit'][:7]}` | {summary['recipe'] or '—'} | "
        f"{summary['games']:,} | {cell('control')} | {cell('champion')} | {cell('greedy')} | "
        f"{summary['results']['ussr_edge']:.3f} | {summary['note']} |\n"
    )


def append_ledger(row: str, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(LEDGER_HEADER, encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(row)


def latest_version() -> str | None:
    versions = sorted((d.name for d in baseline.ladder_dir().glob("v*") if d.is_dir()), key=lambda v: int(v[1:]))
    return versions[-1] if versions else None


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="A clean run, compared, and a ledger row.")
    p.add_argument("--run", required=True, help="new run name under runs/ (must not exist)")
    p.add_argument("--games", type=int, default=8000)
    p.add_argument("--recipe", default="v11", choices=sorted(train.RECIPES))
    p.add_argument("--control", default=None, help="baseline (this ladder) trained with the same recipe and budget; none on a fresh ladder")
    p.add_argument("--champion", default=None, help="baseline to compare against as well (default: the newest vN)")
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--note", default="", help="one line for the ledger: what this run is testing")
    p.add_argument("--no-ledger", action="store_true", help="write ab.json only")
    p.add_argument("--existing", action="store_true", help="compare runs/<run> as it is instead of training it")
    p.add_argument("train_args", nargs="*", help="arguments passed to train.py after `--`")
    args = p.parse_args(argv)

    run_dir = train.RUNS_DIR / args.run
    if args.existing:
        if not (run_dir / "joshua.pt").exists():
            raise SystemExit(f"--existing: no runs/{args.run}/joshua.pt")
    elif run_dir.exists():
        raise SystemExit(f"runs/{args.run} exists: an A/B run is trained from scratch, pick a new name (or --existing)")
    champion = args.champion or latest_version()
    opponents = {"greedy": "greedy"}
    if args.control:
        opponents["control"] = str(baseline.ladder_dir() / args.control / "joshua.pt")
    if champion and champion != args.control:
        opponents["champion"] = str(baseline.ladder_dir() / champion / "joshua.pt")
    for name, spec in opponents.items():
        if spec != "greedy" and not Path(spec).exists():
            raise SystemExit(f"{name}: no {spec}")

    if not args.existing:
        train.main(["--run", args.run, "--games", str(args.games), "--recipe", args.recipe, "--workers", str(args.workers), *args.train_args])
    config = json.loads((run_dir / "config.json").read_text())
    report = compare(run_dir / "joshua.pt", opponents, games=args.eval_games, seeds=args.eval_seeds, workers=args.workers)
    summary = {
        "date": dt.date.today().isoformat(),
        "run": args.run,
        "commit": config.get("commit", git_commit()),
        "rules_version": config.get("rules_version", baseline.RULES_VERSION),
        "recipe": config.get("recipe") if args.existing else args.recipe,
        "games": int(config.get("games_done", args.games)),
        "control": args.control,
        "champion": champion,
        "train_args": args.train_args,
        "note": args.note,
        "results": report,
    }
    (run_dir / "ab.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    row = ledger_row(summary)
    print(row, end="")
    if not args.no_ledger:
        append_ledger(row)
        print(f"[ab] appended to {LEDGER}")


if __name__ == "__main__":
    main()
