"""The self-improvement loop: train, evaluate, gate, promote, repeat.

    python -m wopr.loop --run pure --champion v5 --generations 5 --generation-games 4000 --workers 8

One generation continues the run for `--generation-games` games (the
same `train.py` path: optimizer state and pool carried over), evaluates
the latest checkpoint -- the *challenger* -- against the *champion* (the
newest frozen baseline, or whatever `--champion` names) and against
Greedy on fixed seeds with the parallel ladder, and applies the gate: a
challenger that wins at least `--gate` of its games against the champion
over every eval seed is frozen as the next `vN` (`baseline.py`, full
protocol), its README entry appended, and becomes the champion. One that
does not is simply trained further; the run never rolls back -- the PFSP
pool is what guards against regression -- but a challenger that keeps
*losing* to the champion (`--patience` generations below 0.5) stops the
loop, since that is a regression to look at, not to train through --
and so does the plateau rule of docs/WOPR.md ("Decision points"): two
misses in the last three generations (`--plateau-misses`, 0 to disable).

Training arguments after `--` go to `train.py` unchanged, which is how a
hyperparameter experiment runs through the loop:

    python -m wopr.loop --run pure --generations 2 -- --n-epochs 2

`--no-promote` evaluates and logs the gate without freezing anything, so
two arms of an experiment can be gated against the same champion.
`runs/<run>/loop.csv` records every generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from wopr import baseline, train
from wopr.baseline import append_readme, latest_version, next_version, readme_entry  # noqa: F401 -- the loop's names
from wopr.eval import PairJob, play_pairs
from wopr.ladder import summarize

LOOP_COLUMNS = (
    "generation", "games", "train_s", "eval_s", "champion",
    "vs_champion", "vs_champion_us", "vs_champion_ussr", "vs_champion_min_seed",
    "vs_greedy", "vs_greedy_us", "vs_greedy_ussr",
    "promoted", "version",
)


def games_done(run_dir: Path) -> int:
    config = run_dir / "config.json"
    return int(json.loads(config.read_text()).get("games_done", 0)) if config.exists() else 0


def evaluate_challenger(
    challenger: Path, champion: Path | None, *, games: int, seeds: Sequence[int], workers: int | None
) -> dict[str, Any]:
    """Argmax play of the challenger against the champion (every seed) and
    against Greedy (first seed): per-seed and per-seat win rates."""
    jobs = [PairJob(f"challenger={challenger}", "greedy", games, seeds[0])]
    if champion is not None:
        jobs += [PairJob(f"challenger={challenger}", f"champion={champion}", games, seed) for seed in seeds]
    results = play_pairs(jobs, workers=workers)

    def rates(pair, opponent: str) -> dict[str, float]:
        return {
            "win_rate": summarize(pair, "challenger", opponent)["win_rate"],
            "as_us": summarize([m for m in pair if m.a == "challenger"], "challenger", opponent)["win_rate"],
            "as_ussr": summarize([m for m in pair if m.b == "challenger"], "challenger", opponent)["win_rate"],
        }

    report: dict[str, Any] = {"vs_greedy": rates(results[0], "greedy")}
    if champion is not None:
        per_seed = [rates(pair, "champion") for pair in results[1:]]
        report["vs_champion"] = {
            "win_rate": statistics.fmean(r["win_rate"] for r in per_seed),
            "as_us": statistics.fmean(r["as_us"] for r in per_seed),
            "as_ussr": statistics.fmean(r["as_ussr"] for r in per_seed),
            "min_seed": min(r["win_rate"] for r in per_seed),
            "per_seed": per_seed,
        }
    return report


def gate_passes(report: dict[str, Any], gate: float) -> bool:
    """Promote when every eval seed's win rate against the champion clears
    `gate` -- the mean alone lets one lucky deck carry a generation. With
    no champion yet, the gate is the same bar against Greedy."""
    versus = report.get("vs_champion")
    if versus is None:
        return report["vs_greedy"]["win_rate"] >= gate
    return versus["min_seed"] >= gate


def plateaued(misses: Sequence[bool], threshold: int) -> bool:
    """The decision point's plateau: `threshold` misses among the last
    three generations (0 disables the rule)."""
    return threshold > 0 and sum(misses[-3:]) >= threshold


def append_row(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOOP_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train, evaluate, gate, promote -- repeat.")
    p.add_argument("--run", required=True, help="run name under runs/ (continued; created if new)")
    p.add_argument("--champion", default=None, help="baseline to beat (default: the newest vN; none means Greedy)")
    p.add_argument("--generations", type=int, default=1)
    p.add_argument("--generation-games", type=int, default=4000, help="games trained per generation")
    p.add_argument("--gate", type=float, default=0.55, help="win rate against the champion, on every eval seed, that promotes")
    p.add_argument("--patience", type=int, default=3, help="stop after this many generations below 0.5 against the champion")
    p.add_argument("--plateau-misses", type=int, default=2, help="stop once this many of the last three generations missed the gate (0: never)")
    p.add_argument("--eval-games", type=int, default=200, help="games per pair in the gate evaluation")
    p.add_argument("--eval-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=8, help="collector processes for training and pair processes for evaluation")
    p.add_argument("--no-promote", action="store_true", help="evaluate and log the gate, freeze nothing: an experiment, not a generation")
    p.add_argument("train_args", nargs="*", help="arguments passed to train.py after `--`")
    args = p.parse_args(argv)

    run_dir = train.RUNS_DIR / args.run
    champion = args.champion or latest_version()
    if champion is not None and not (baseline.ladder_dir() / champion / "joshua.pt").exists():
        raise SystemExit(f"champion {champion!r}: no {baseline.ladder_dir() / champion}/joshua.pt (the ladder of rules version {baseline.RULES_VERSION})")
    losing = 0
    misses: list[bool] = []
    for generation in range(1, args.generations + 1):
        target = games_done(run_dir) + args.generation_games
        print(f"[loop] generation {generation}: training {args.run!r} to {target} games, champion {champion or 'greedy'}", flush=True)
        started = time.perf_counter()
        train.main(["--run", args.run, "--games", str(target), "--workers", str(args.workers), *args.train_args])
        train_s = time.perf_counter() - started

        started = time.perf_counter()
        champion_path = None if champion is None else baseline.ladder_dir() / champion / "joshua.pt"
        report = evaluate_challenger(
            run_dir / "joshua.pt", champion_path, games=args.eval_games, seeds=args.eval_seeds, workers=args.workers
        )
        eval_s = time.perf_counter() - started
        versus = report.get("vs_champion")
        greedy = report["vs_greedy"]
        line = f"[loop] generation {generation}: vs greedy {greedy['win_rate']:.3f} (US {greedy['as_us']:.3f} / USSR {greedy['as_ussr']:.3f})"
        if versus is not None:
            line += (f" | vs {champion} {versus['win_rate']:.3f} (US {versus['as_us']:.3f} / USSR {versus['as_ussr']:.3f}, "
                     f"worst seed {versus['min_seed']:.3f})")
        print(line, flush=True)

        promoted = gate_passes(report, args.gate)
        version = ""
        if promoted and args.no_promote:
            print(f"[loop] gate cleared; not promoting (--no-promote)", flush=True)
        elif promoted:
            version = next_version()
            note = (f"Loop generation {generation}: {champion or 'Greedy'} continued for {args.generation_games:,} games; "
                    f"gate {args.gate:.2f} cleared at "
                    + (f"{versus['min_seed']:.3f} (worst seed) against {champion}" if versus else f"{greedy['win_rate']:.3f} against Greedy")
                    + ".")
            print(f"[loop] promoting to {version}", flush=True)
            baseline.main([version, "--run", args.run, "--workers", str(args.workers)])
            append_readme(version, note)
            champion = version
            losing = 0
        elif versus is not None and versus["win_rate"] < 0.5:
            losing += 1
        else:
            losing = 0

        append_row(run_dir / "loop.csv", {
            "generation": generation, "games": games_done(run_dir), "train_s": round(train_s, 1), "eval_s": round(eval_s, 1),
            "champion": version or champion,
            "vs_champion": None if versus is None else round(versus["win_rate"], 4),
            "vs_champion_us": None if versus is None else round(versus["as_us"], 4),
            "vs_champion_ussr": None if versus is None else round(versus["as_ussr"], 4),
            "vs_champion_min_seed": None if versus is None else round(versus["min_seed"], 4),
            "vs_greedy": round(greedy["win_rate"], 4), "vs_greedy_us": round(greedy["as_us"], 4), "vs_greedy_ussr": round(greedy["as_ussr"], 4),
            "promoted": int(promoted), "version": version,
        })
        if losing >= args.patience:
            print(f"[loop] stopping: {losing} generations below 0.5 against {champion}", flush=True)
            return
        misses.append(not promoted)
        if plateaued(misses, args.plateau_misses):
            print(f"[loop] stopping: plateau -- {sum(misses[-3:])} of the last 3 generations missed the gate (docs/WOPR.md)", flush=True)
            return


if __name__ == "__main__":
    main()
