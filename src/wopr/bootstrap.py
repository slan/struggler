"""From nothing to a ladder's first version, stopped by the yardstick.

    python -m wopr.bootstrap --run r3 --workers 8 --torch-threads 16

A fresh run with the recipe (`v11`: hidden 256, 4 PPO epochs, half
self-play, half the PFSP pool, no anchors), evaluated against Greedy
every `--eval-every` games *as it trains* (`train.py --eval-every`: the
collectors play the latest checkpoint, argmax, `--eval-games` games half
on each seat, through the PPO update), and stopped by a rule rather than
a budget:

- The signal is the **per-seat rolling mean** over the last `--window`
  evaluations (two of 200 games: 200 a seat, about ±0.07), not one tick.
- Rolling mean at or above `--target` on **both** seats → a confirmatory
  evaluation of `--confirm-games` (600: 300 a seat) on fresh decks; both
  seats at or above the target there → **stop, confirmed**. A miss trains
  on; the next tick may ask again.
- No new best of the rolling mean's weaker seat for `--plateau`
  evaluations (four, about 2,000 games) → **stop, plateau**.
- `--games` is the budget cap → **stop, cap**.

Whatever stopped it, the last evaluated checkpoint (`runs/<run>/joshua.pt`)
is frozen as the ladder's next version with the full protocol
(`wopr.baseline`) and gets its README entry; `runs/<run>/bootstrap.csv`
records every tick's decision and `bootstrap.json` the outcome. Anything
after `--` goes to `train.py`; the run resumes if it exists (the ticks
already in `metrics.csv` count).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from struggler.engine import RULES_VERSION
from wopr import baseline, train
from wopr.callback import EvalTick, WoprCallback
from wopr.eval import EvalCounts
from wopr.repo import git_commit

BOOTSTRAP_COLUMNS = (
    "tick", "games", "seed", "eval_us", "eval_ussr", "rolling_us", "rolling_ussr", "signal", "best", "since_best",
    "confirm_games", "confirm_us", "confirm_ussr", "decision",
)


@dataclass(frozen=True)
class Assessment:
    """The stop rule read off the ticks so far."""

    rolling_us: float | None
    rolling_ussr: float | None
    best: float | None  # the best signal (weaker seat's rolling mean) so far, this tick included
    since_best: int  # ticks since the best was set (0: this tick set it)

    @property
    def signal(self) -> float | None:
        return None if self.rolling_us is None or self.rolling_ussr is None else min(self.rolling_us, self.rolling_ussr)


@dataclass(frozen=True)
class StopRule:
    target: float = 0.75
    window: int = 2
    plateau: int = 4

    def rolling(self, ticks: Sequence[EvalTick]) -> EvalCounts | None:
        """The last `window` ticks added up (games-weighted), once there are that many."""
        if len(ticks) < self.window:
            return None
        return sum((tick.counts for tick in ticks[-self.window:]), EvalCounts())

    def assess(self, ticks: Sequence[EvalTick]) -> Assessment:
        best: float | None = None
        best_at = 0
        rolling = None
        for i in range(len(ticks)):
            rolling = self.rolling(ticks[: i + 1])
            if rolling is None:
                continue
            signal = min(rolling.as_us, rolling.as_ussr)
            if best is None or signal > best:
                best, best_at = signal, i
        if rolling is None:
            return Assessment(None, None, None, 0)
        return Assessment(rolling.as_us, rolling.as_ussr, best, len(ticks) - 1 - best_at)

    def ready(self, assessment: Assessment) -> bool:
        """Both seats' rolling means at the target: ask for the confirmation."""
        return assessment.signal is not None and assessment.signal >= self.target

    def confirmed(self, counts: EvalCounts) -> bool:
        return counts.as_us >= self.target and counts.as_ussr >= self.target

    def plateaued(self, assessment: Assessment) -> bool:
        return assessment.best is not None and assessment.since_best >= self.plateau


class BootstrapGate(BaseCallback):
    """Reads each new evaluation tick off the tracker, applies the rule,
    runs the confirmation when asked, and ends training when it says so."""

    def __init__(self, tracker: WoprCallback, rule: StopRule, *, confirm_games: int, confirm_seed: int, log_path: Path) -> None:
        super().__init__(verbose=1)
        self.tracker = tracker
        self.rule = rule
        self.confirm_games = confirm_games
        self.confirm_seed = confirm_seed
        self.log_path = log_path
        self.stopped_by: str | None = None
        self.confirmations: list[dict[str, Any]] = []
        self._seen = len(tracker.evals)

    def _on_step(self) -> bool:
        return self.stopped_by is None

    # The tracker's `on_rollout_start` runs first (it is earlier in the
    # callback list) and collects the tick; by now it is in `evals`.
    def on_rollout_start(self) -> None:
        super().on_rollout_start()
        if len(self.tracker.evals) == self._seen:
            return
        self._seen = len(self.tracker.evals)
        self.judge()

    def judge(self) -> None:
        ticks = self.tracker.evals
        tick = ticks[-1]
        assessment = self.rule.assess(ticks)
        row: dict[str, Any] = {
            "tick": len(ticks), "games": tick.games, "seed": tick.seed,
            "eval_us": _r(tick.counts.as_us), "eval_ussr": _r(tick.counts.as_ussr),
            "rolling_us": _r(assessment.rolling_us), "rolling_ussr": _r(assessment.rolling_ussr),
            "signal": _r(assessment.signal), "best": _r(assessment.best), "since_best": assessment.since_best,
        }
        decision = "train"
        if self.rule.ready(assessment):
            seed = self.confirm_seed + len(self.confirmations)
            print(f"[bootstrap] rolling mean US {assessment.rolling_us:.3f} / USSR {assessment.rolling_ussr:.3f} "
                  f">= {self.rule.target}: confirming over {self.confirm_games} games (seed {seed})", flush=True)
            started = time.perf_counter()
            counts = self.tracker.play_eval(self.confirm_games, seed)
            passed = self.rule.confirmed(counts)
            self.confirmations.append({"games": tick.games, "seed": seed, "counts": counts._asdict(), "passed": passed})
            print(f"[bootstrap] confirmation: {counts.win_rate:.3f} (US {counts.as_us:.3f} / USSR {counts.as_ussr:.3f}) "
                  f"over {counts.games} in {time.perf_counter() - started:.0f}s -- {'passed' if passed else 'missed'}", flush=True)
            row.update(confirm_games=counts.games, confirm_us=_r(counts.as_us), confirm_ussr=_r(counts.as_ussr))
            decision = "confirmed" if passed else "confirm-missed"
            if passed:
                self.stopped_by = "confirmed"
        if self.stopped_by is None and self.rule.plateaued(assessment):
            decision = "plateau"
            self.stopped_by = "plateau"
        row["decision"] = decision
        append_row(self.log_path, row)
        if decision != "train":
            print(f"[bootstrap] {decision} at {tick.games} games", flush=True)


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def append_row(path: Path, row: dict[str, Any]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BOOTSTRAP_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train from scratch until the Greedy yardstick says stop, then freeze.")
    p.add_argument("--run", required=True, help="run name under runs/ (resumed if it exists)")
    p.add_argument("--games", type=int, default=20000, help="the budget cap")
    p.add_argument("--recipe", default="v11", choices=sorted(train.RECIPES))
    p.add_argument("--target", type=float, default=0.75, help="win rate against Greedy wanted on both seats")
    p.add_argument("--window", type=int, default=2, help="evaluations the rolling mean spans")
    p.add_argument("--plateau", type=int, default=4, help="evaluations without a new best rolling mean (weaker seat) that stop the run")
    p.add_argument("--eval-every", type=int, default=500, help="training games between evaluations")
    p.add_argument("--eval-games", type=int, default=200, help="games per evaluation (half a seat)")
    p.add_argument("--eval-seed", type=int, default=1000, help="deck seed of the first evaluation (+ the tick number after)")
    p.add_argument("--confirm-games", type=int, default=600, help="games of the confirmatory evaluation (half a seat)")
    p.add_argument("--confirm-seed", type=int, default=5000, help="deck seed of the first confirmation (+ its number after)")
    p.add_argument("--workers", type=int, default=8, help="collector processes (training and evaluations) and pair processes for the freeze")
    p.add_argument("--torch-threads", type=int, default=None)
    p.add_argument("--version", default=None, help="freeze as this version (default: the ladder's next)")
    p.add_argument("--no-freeze", action="store_true", help="stop and report; freeze nothing")
    p.add_argument("train_args", nargs="*", help="arguments passed to train.py after `--`")
    args = p.parse_args(argv)

    rule = StopRule(target=args.target, window=args.window, plateau=args.plateau)
    run_dir = train.RUNS_DIR / args.run
    train_argv = [
        "--run", args.run, "--games", str(args.games), "--recipe", args.recipe, "--workers", str(args.workers),
        "--eval-every", str(args.eval_every), "--eval-games", str(args.eval_games), "--eval-seed", str(args.eval_seed),
        "--eval-opponent", "greedy", *args.train_args,
    ]
    if args.torch_threads:
        train_argv += ["--torch-threads", str(args.torch_threads)]
    version = args.version or baseline.next_version()
    if not args.no_freeze and (baseline.ladder_dir() / version).exists():
        raise SystemExit(f"{baseline.ladder_dir() / version} exists; pass --version or --no-freeze")
    print(f"[bootstrap] {args.run!r}: recipe {args.recipe}, target {rule.target} on both seats over {rule.window} evaluations "
          f"of {args.eval_games} every {args.eval_every} games, confirm {args.confirm_games}, plateau {rule.plateau}, cap {args.games:,}; "
          f"freezing as {'nothing' if args.no_freeze else version}", flush=True)

    gate: BootstrapGate | None = None

    def callbacks(tracker: WoprCallback) -> list[BaseCallback]:
        nonlocal gate
        gate = BootstrapGate(tracker, rule, confirm_games=args.confirm_games, confirm_seed=args.confirm_seed, log_path=run_dir / "bootstrap.csv")
        return [gate]

    started = time.perf_counter()
    tracker = train.run(train.parse_args(train_argv), callbacks=callbacks)
    train_s = time.perf_counter() - started
    if tracker is None or gate is None:
        raise SystemExit(f"runs/{args.run} is already at the cap; nothing trained")
    stopped_by = gate.stopped_by or "cap"
    assessment = rule.assess(tracker.evals)
    outcome: dict[str, Any] = {
        "run": args.run, "commit": git_commit(), "rules_version": RULES_VERSION, "recipe": args.recipe,
        "games": tracker.games, "train_s": round(train_s, 1), "stopped_by": stopped_by,
        "rule": {"target": rule.target, "window": rule.window, "plateau": rule.plateau, "cap": args.games,
                 "eval_every": args.eval_every, "eval_games": args.eval_games, "confirm_games": args.confirm_games},
        "rolling": {"us": assessment.rolling_us, "ussr": assessment.rolling_ussr, "best": assessment.best, "since_best": assessment.since_best},
        "ticks": [{"games": t.games, "seed": t.seed, **t.counts._asdict()} for t in tracker.evals],
        "confirmations": gate.confirmations,
        "version": None if args.no_freeze else version,
    }
    (run_dir / "bootstrap.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(f"[bootstrap] stopped by {stopped_by} at {tracker.games:,} games after {train_s / 60:.0f} min; "
          f"rolling mean vs Greedy US {_fmt(assessment.rolling_us)} / USSR {_fmt(assessment.rolling_ussr)}", flush=True)
    if args.no_freeze:
        return
    print(f"[bootstrap] freezing {args.run!r} as {version}", flush=True)
    baseline.main([version, "--run", args.run, "--workers", str(args.workers)])
    baseline.append_readme(version, bootstrap_note(outcome))
    print(f"[bootstrap] {version} frozen; README entry appended", flush=True)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def bootstrap_note(outcome: dict[str, Any]) -> str:
    rule = outcome["rule"]
    rolling = outcome["rolling"]
    note = (f"Bootstrap of the r{outcome['rules_version']} ladder: recipe {outcome['recipe']} from scratch, stopped by "
            f"**{outcome['stopped_by']}** at {outcome['games']:,} games (rule: rolling mean over {rule['window']} evaluations of "
            f"{rule['eval_games']} every {rule['eval_every']} games ≥ {rule['target']} on both seats, confirmed over "
            f"{rule['confirm_games']}; plateau {rule['plateau']}; cap {rule['cap']:,}). Last rolling mean vs Greedy: "
            f"US {_fmt(rolling['us'])} / USSR {_fmt(rolling['ussr'])}.")
    confirmations = outcome["confirmations"]
    if confirmations:
        last = confirmations[-1]
        counts = EvalCounts(**last["counts"])
        note += (f" Confirmation: {counts.win_rate:.3f} (US {counts.as_us:.3f} / USSR {counts.as_ussr:.3f}) over {counts.games} "
                 f"-- {'passed' if last['passed'] else 'missed'}" + (f", {len(confirmations)} asked." if len(confirmations) > 1 else "."))
    return note


if __name__ == "__main__":
    main()
