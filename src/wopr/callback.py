"""Per-rollout bookkeeping: game results, policy health, pool snapshots, CSV.

What gets logged, and why each number is there:

- games / win rates *per seat* and per opponent kind -- Twilight Struggle is
  asymmetric, so one pooled win rate hides a policy that only learned USSR.
- `entropy`, `k_valid`, `entropy_ratio` (H / ln K) and `k_eff` (exp H):
  exploration health. H/lnK near 0 with many legal options means the policy
  has collapsed onto a line; near 1 means it has not started choosing.
- SB3's `approx_kl`, `clip_fraction`, `explained_variance`, losses:
  update stability; a KL well above target or a clip fraction above ~0.3
  says the step size is too big, EV near 0 says the value head is lost.

Snapshots go to the `CheckpointPool` every `snapshot_every` rollouts and
to `<run>/joshua.pt` (the latest, what `JoshuaPlayer` loads) every rollout;
`on_snapshot` (train.py: save `ppo.zip` and the game count) runs with
each pool snapshot, so a run killed mid-way resumes from its last one
rather than from nothing.

With `eval_every` set, every time the game count crosses a multiple of it
the just-saved `joshua.pt` plays `eval_games` against `eval_opponent`
(argmax, half on each seat, the deck seed rotating tick by tick): started
on the backend at the end of the rollout, so the collectors play it while
the PPO update runs, and collected at the start of the next rollout into
the same row (`eval_*` columns). `evals` keeps every tick, the earlier
ones read back from `metrics.csv` when a run resumes -- the bootstrap's
stop rule reads them.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from struggler.bots.joshua.model import save_checkpoint
from wopr.eval import EvalCounts, EvalJob
from wopr.pool import POOL_PREFIX, AnchorSchedule, CheckpointPool
from wopr.vec_env import LEARNER, WoprVecEnv

CSV_COLUMNS = (
    "update", "timesteps", "games", "games_in_rollout", "elapsed_s", "steps_per_s", "rollout_s", "update_s", "wait_s",
    "win_rate", "win_rate_us", "win_rate_ussr", "draw_rate", "win_rate_vs_pool", "win_rate_vs_anchor",
    "ep_len_mean", "turn_mean", "vp_mean",
    "entropy", "k_valid", "entropy_ratio", "k_eff",
    "approx_kl", "clip_fraction", "explained_variance", "policy_loss", "value_loss", "entropy_loss",
    "pool_size", "anchor",
    "eval_seed", "eval_games", "eval_win_rate", "eval_win_rate_us", "eval_win_rate_ussr", "eval_s",
)


class EvalTick(NamedTuple):
    """One evaluation of the run against its yardstick: after `games`
    training games, `counts` over `EvalJob.games` games on deck seed `seed`."""

    games: int
    seed: int
    counts: EvalCounts

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "EvalTick | None":
        """The tick a `metrics.csv` row recorded, if it recorded one."""
        if not row.get("eval_games"):
            return None
        games, us, ussr = int(row["games"]), float(row["eval_win_rate_us"]), float(row["eval_win_rate_ussr"])
        half = int(row["eval_games"]) // 2
        # Wins and draws are not kept apart in the CSV: two draws count as a win.
        return cls(games, int(row["eval_seed"]), EvalCounts(round(us * half), 0, half, round(ussr * half), 0, half))


def read_evals(path: Path) -> list[EvalTick]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        ticks = (EvalTick.from_row(row) for row in csv.DictReader(f))
        return [tick for tick in ticks if tick is not None]


def ensure_columns(path: Path) -> None:
    """Create `metrics.csv`, or rewrite an older run's file under the
    current `CSV_COLUMNS` if its header differs. Rows are written in
    `CSV_COLUMNS` order, so resuming a run whose file predates a column
    would otherwise file every new value under the wrong name."""
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        return
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) == CSV_COLUMNS:
            return
        rows = list(reader)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class WoprCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path,
        env: WoprVecEnv,
        pool: CheckpointPool,
        target_games: int,
        snapshot_every: int,
        games_done: int = 0,
        updates_done: int = 0,
        anchor_schedule: AnchorSchedule | None = None,
        eval_every: int = 0,
        eval_games: int = 200,
        eval_seed: int = 1000,
        eval_opponent: str = "greedy",
        on_snapshot: Callable[[int], None] | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.run_dir = run_dir
        self.env = env
        self.pool = pool
        self.anchor_schedule = anchor_schedule
        self.target_games = target_games
        self.snapshot_every = snapshot_every
        self.games = games_done
        self.update = updates_done
        self.eval_every = eval_every
        self.eval_games = eval_games
        self.eval_seed = eval_seed
        self.eval_opponent = eval_opponent
        self.on_snapshot = on_snapshot
        self.evals: list[EvalTick] = read_evals(run_dir / "metrics.csv") if eval_every else []
        self._eval_tick = max((t.games // eval_every for t in self.evals), default=games_done // eval_every) if eval_every else 0
        self._eval_in_flight: tuple[int, float] | None = None  # (deck seed, started at)
        self._rollout_games: list[dict[str, Any]] = []
        self._rollout_start = time.perf_counter()
        self._start = time.perf_counter()
        self._start_timesteps = 0
        self._csv_path = run_dir / "metrics.csv"
        ensure_columns(self._csv_path)

    # -- SB3 hooks -------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._start = time.perf_counter()
        self._start_timesteps = self.model.num_timesteps

    def _on_rollout_start(self) -> None:
        self._rollout_games = []
        self._rollout_start = time.perf_counter()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            episode = info.get("episode")
            if episode is None:
                continue
            self.games += 1
            self._rollout_games.append(episode)
            self._record_pool(episode)
            self._record_anchor(episode)
        return True

    def _on_rollout_end(self) -> None:
        self.update += 1
        rollout_s = time.perf_counter() - self._rollout_start
        row = self._game_metrics()
        row.update(self._policy_health())
        row.update(
            update=self.update,
            timesteps=self.model.num_timesteps,
            games=self.games,
            games_in_rollout=len(self._rollout_games),
            elapsed_s=round(time.perf_counter() - self._start, 1),
            steps_per_s=round(self.model.n_steps * self.env.num_envs / rollout_s, 1),
            rollout_s=round(rollout_s, 1),
            # Shared-memory backend: seconds the rollout spent waiting on its collectors.
            wait_s=round(take_wait(), 2) if (take_wait := getattr(self.env.backend, "take_wait", None)) else None,
            pool_size=len(self.pool),
            anchor=None if self.anchor_schedule is None else self.anchor_schedule.current,
        )
        self._pending_row = row
        # The PPO update runs between here and the row's flush; `_flush`
        # measures it as `update_s`. Training has been update-bound
        # (docs/JOSHUA.md), so it is logged alongside the rollout rate.
        self._rollout_end = time.perf_counter()
        save_checkpoint(self.model.policy.net, self.run_dir / "joshua.pt", extra={"games": self.games, "update": self.update})
        if self.snapshot_every > 0 and self.update % self.snapshot_every == 0:
            self.pool.add(f"u{self.update:05d}", self.model.policy.net, extra={"games": self.games, "update": self.update})
            if self.on_snapshot is not None:
                self.on_snapshot(self.games)
        if self.eval_every and self.games // self.eval_every > self._eval_tick:
            self._eval_tick = self.games // self.eval_every
            self._start_eval(self.eval_seed + self._eval_tick)

    def _on_training_end(self) -> None:
        self._flush(final=True)

    # PPO logs its train/* values after _on_rollout_end; the row is flushed
    # at the start of the next rollout so it carries them too.
    def on_rollout_start(self) -> None:
        self._flush(final=False)
        super().on_rollout_start()

    # -- the evaluation on the collectors ------------------------------------------------

    def _start_eval(self, seed: int) -> None:
        """The checkpoint just saved against the yardstick, on the backend:
        the collectors play it through the PPO update."""
        job = EvalJob(str(self.run_dir / "joshua.pt"), self.eval_games, seed, opponent=self.eval_opponent)
        self.env.backend.start_eval(job)
        self._eval_in_flight = (seed, time.perf_counter())

    def _collect_eval(self) -> dict[str, Any]:
        """Wait for the evaluation in flight, if any, record its tick and
        return its row columns."""
        if self._eval_in_flight is None:
            return {}
        seed, started = self._eval_in_flight
        self._eval_in_flight = None
        waited = time.perf_counter()
        counts = self.env.backend.finish_eval()
        tick = EvalTick(self.games, seed, counts)
        self.evals.append(tick)
        if self.verbose:
            print(
                f"[wopr] eval @ {tick.games} games vs {self.eval_opponent}: {counts.win_rate:.3f} "
                f"(US {counts.as_us:.3f} / USSR {counts.as_ussr:.3f}) over {counts.games}, seed {seed}, "
                f"{time.perf_counter() - started:.0f}s ({time.perf_counter() - waited:.0f}s waited)",
                flush=True,
            )
        return {
            "eval_seed": seed, "eval_games": counts.games,
            "eval_win_rate": _round(counts.win_rate), "eval_win_rate_us": _round(counts.as_us), "eval_win_rate_ussr": _round(counts.as_ussr),
            "eval_s": round(time.perf_counter() - waited, 1),
        }

    def play_eval(self, games: int, seed: int) -> EvalCounts:
        """An evaluation played to completion now (the bootstrap's confirmatory
        one): only between rollouts, when the collectors are free."""
        if self._eval_in_flight is not None:
            raise RuntimeError("an evaluation is already in flight")
        self.env.backend.start_eval(EvalJob(str(self.run_dir / "joshua.pt"), games, seed, opponent=self.eval_opponent))
        return self.env.backend.finish_eval()

    # -- metrics -----------------------------------------------------------------------

    def _record_anchor(self, episode: dict[str, Any]) -> None:
        schedule = self.anchor_schedule
        if schedule is None:
            return
        seats = episode["seats"]
        for side, policy_id in seats.items():
            if schedule.is_anchor(policy_id):
                learner_side = next(s for s, p in seats.items() if p == LEARNER)
                promoted = schedule.record(episode["winner"] == learner_side)
                if promoted is not None and self.verbose:
                    print(f"[wopr] anchor promoted to {promoted!r} after {self.games} games", flush=True)

    def _record_pool(self, episode: dict[str, Any]) -> None:
        seats = episode["seats"]
        for side, policy_id in seats.items():
            if policy_id.startswith(POOL_PREFIX):
                learner_side = next(s for s, p in seats.items() if p == LEARNER)
                self.pool.record(policy_id, episode["winner"] == learner_side)
        self.pool.save()

    def _game_metrics(self) -> dict[str, Any]:
        games = self._rollout_games
        if not games:
            return {}
        outcomes: Counter = Counter()
        by_seat = {"US": [0, 0], "USSR": [0, 0]}
        vs_pool = [0, 0]
        vs_anchor = [0, 0]
        vps: list[int] = []
        for g in games:
            seats = g["seats"]
            learner_sides = [s for s, p in seats.items() if p == LEARNER]
            if len(learner_sides) == 2:
                outcomes["self_play"] += 1
                continue
            side = learner_sides[0]
            won = g["winner"] == side
            vps.append(g["vp"] if side == "US" else -g["vp"])
            by_seat[side][0] += int(won)
            by_seat[side][1] += 1
            if g["winner"] is None:
                outcomes["draw"] += 1
            elif won:
                outcomes["win"] += 1
            else:
                outcomes["loss"] += 1
            opponent = seats["US" if side == "USSR" else "USSR"]
            bucket = vs_pool if opponent.startswith(POOL_PREFIX) else vs_anchor
            bucket[0] += int(won)
            bucket[1] += 1

        def rate(wins: int, total: int) -> float | None:
            return round(wins / total, 4) if total else None

        decided = outcomes["win"] + outcomes["loss"] + outcomes["draw"]
        return {
            "win_rate": rate(outcomes["win"], decided),
            "draw_rate": rate(outcomes["draw"], decided),
            "win_rate_us": rate(*by_seat["US"]),
            "win_rate_ussr": rate(*by_seat["USSR"]),
            "win_rate_vs_pool": rate(*vs_pool),
            "win_rate_vs_anchor": rate(*vs_anchor),
            "ep_len_mean": round(float(np.mean([g["l"] for g in games])), 1),
            "turn_mean": round(float(np.mean([g["turn"] for g in games])), 2),
            "vp_mean": round(float(np.mean(vps)), 2) if vps else None,
        }

    def _policy_health(self, sample: int = 2048) -> dict[str, Any]:
        buffer = self.model.rollout_buffer
        masks = buffer.observations["opt_mask"].reshape(-1, buffer.observations["opt_mask"].shape[-1])
        k_valid = masks.sum(-1)
        multi = k_valid > 1
        if not multi.any():
            return {}
        rng = np.random.default_rng(self.update)
        index = np.flatnonzero(multi)
        if len(index) > sample:
            index = rng.choice(index, size=sample, replace=False)
        obs = {
            name: torch.as_tensor(array.reshape(-1, *array.shape[2:])[index], device=self.model.device)
            for name, array in buffer.observations.items()
        }
        with torch.no_grad():
            entropy = float(self.model.policy.get_distribution(obs).entropy().mean())
        k_mean = float(k_valid[index].mean())
        return {
            "entropy": round(entropy, 4),
            "k_valid": round(k_mean, 2),
            "entropy_ratio": round(entropy / np.log(k_mean), 4) if k_mean > 1 else None,
            "k_eff": round(float(np.exp(entropy)), 2),
        }

    def _flush(self, *, final: bool) -> None:
        row = getattr(self, "_pending_row", None)
        if row is None:
            return
        logged = self.model.logger.name_to_value
        update_s = round(time.perf_counter() - self._rollout_end, 1)
        row.update(self._collect_eval())
        row.update(
            update_s=update_s,
            approx_kl=_round(logged.get("train/approx_kl")),
            clip_fraction=_round(logged.get("train/clip_fraction")),
            explained_variance=_round(logged.get("train/explained_variance")),
            policy_loss=_round(logged.get("train/policy_gradient_loss")),
            value_loss=_round(logged.get("train/value_loss")),
            entropy_loss=_round(logged.get("train/entropy_loss")),
        )
        with self._csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)
        if self.verbose:
            print(
                f"[wopr] upd {row['update']:>4} | games {row['games']:>6} (+{row['games_in_rollout']}) "
                f"| wr {row.get('win_rate')} us {row.get('win_rate_us')} ussr {row.get('win_rate_ussr')} "
                f"pool {row.get('win_rate_vs_pool')} anchor {row.get('win_rate_vs_anchor')} "
                f"| len {row.get('ep_len_mean')} turn {row.get('turn_mean')} "
                f"| H {row.get('entropy')} K {row.get('k_valid')} H/lnK {row.get('entropy_ratio')} "
                f"| kl {row.get('approx_kl')} clip {row.get('clip_fraction')} ev {row.get('explained_variance')} "
                f"| {row['steps_per_s']} st/s | rollout {row['rollout_s']}s update {row.get('update_s')}s",
                flush=True,
            )
        self._pending_row = None


def _round(value: Any, digits: int = 4) -> Any:
    return None if value is None else round(float(value), digits)


class StopAtGames(BaseCallback):
    """Ends `learn()` once the paired WoprCallback has counted `target` games."""

    def __init__(self, tracker: WoprCallback) -> None:
        super().__init__()
        self._tracker = tracker

    def _on_step(self) -> bool:
        return self._tracker.games < self._tracker.target_games
