"""Evaluate checkpoints head to head on fixed seeds.

    python -m wopr.eval --games 200 joshua=runs/first/joshua.pt random
    python -m wopr.eval --games 200 new=runs/b/joshua.pt old=runs/a/pool/u00010.pt greedy

Each positional argument is a policy: `random`, `greedy`, `first`, or
`name=path/to/checkpoint.pt`. Every pair plays `--games` games, half with
each seat assignment (the seeds are shared across pairs, so two policies
facing the same third one see the same decks). Prints per-pair, per-seat
win rates and Elo with `random` anchored at 0 when present.

Pairs are independent jobs: each builds its own two policies, seeded from
the eval seed and the policy names, so a pair's result does not depend on
which other pairs ran or in what order -- and the jobs run in a process
pool (`--workers`), one pair per process, since nothing crosses between
them but the list of results.
"""

from __future__ import annotations

import argparse
import itertools
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from struggler.bots.greedy import GreedyPlayer
from struggler.bots.naive import FirstLegalPlayer
from struggler.engine import Side
from wopr.arena import Arena, Opponent, play_out
from wopr.ladder import Match, elo_ratings, summarize
from wopr.opponents import NetOpponent, PlayerOpponent, RandomOpponent

#: torch threads per evaluation worker: the work is engine-bound Python with
#: small-batch inference; more threads per process only fight each other.
WORKER_THREADS = 2


def policy_name(spec: str) -> str:
    return spec.split("=", 1)[0]


def parse_policy(spec: str, *, seed: int, device: str, deterministic: bool) -> tuple[str, Opponent]:
    if spec == "random":
        return spec, RandomOpponent(seed)
    if spec == "greedy":
        return spec, PlayerOpponent(GreedyPlayer())
    if spec == "first":
        return spec, PlayerOpponent(FirstLegalPlayer())
    if "=" not in spec:
        raise ValueError(f"policy {spec!r}: expected random|greedy|first or name=checkpoint.pt")
    name, path = spec.split("=", 1)
    return name, NetOpponent.from_checkpoint(path, seed=seed, device=device, deterministic=deterministic)


def play_pair(
    a: str, b: str, policies: dict[str, Opponent], *, games: int, seed: int, events: bool, starting_vp: int = 0,
    us_bid: int = 0,
) -> list[Match]:
    """`games` games between `a` and `b`, half with each seat assignment on
    shared decks: the whole of `play_slice` in one process."""
    half = max(1, games // 2)
    return play_slice(a, b, policies, part=range(half), half=half, seed=seed, events=events, starting_vp=starting_vp,
                      us_bid=us_bid)


def play_slice(
    a: str, b: str, policies: dict[str, Opponent], *, part: range, half: int, seed: int, events: bool, starting_vp: int = 0,
    us_bid: int = 0,
) -> list[Match]:
    """The games of `part` -- a sub-range of the `half` decks of a pair --
    played with `a` as US and again with `a` as USSR. The decks are the
    pair's (`Arena` seeds by global slot), so the slices of [0, half)
    played anywhere, in any order, add up to exactly the pair's games:
    that is what lets the collectors play an evaluation between rollouts."""
    if not part:
        return []

    def assign_a_us(slot: int, episode: int, rng) -> dict[Side, str]:
        return {Side.US: a, Side.USSR: b}

    def assign_a_ussr(slot: int, episode: int, rng) -> dict[Side, str]:
        return {Side.US: b, Side.USSR: a}

    matches: list[Match] = []
    for assigner in (assign_a_us, assign_a_ussr):
        arena = Arena(
            len(part), seed=seed, seat_assigner=assigner, events=events, starting_vp=starting_vp, us_bid=us_bid,
            slot_offset=part.start, total_slots=half,
        )
        for result in play_out(arena, policies):
            # Recorded US-first: `Match.a` is whoever sat as US.
            score_us = 0.5 if result.winner is None else (1.0 if result.winner is Side.US else 0.0)
            matches.append(Match(result.seats[Side.US], result.seats[Side.USSR], score_us))
    return matches


@dataclass(frozen=True)
class EvalJob:
    """A checkpoint against a scripted opponent, argmax, half the games on
    each seat: the evaluation a run plays on its collectors every
    `--eval-every` games (`backend.start_eval`). Plain data; a collector
    plays its `part` of the decks with `play_slice`."""

    checkpoint: str
    games: int
    seed: int
    opponent: str = "greedy"  # random | greedy | first
    events: bool = True
    us_bid: int = 0  # the game's tournament bid, as trained

    @property
    def half(self) -> int:
        return max(1, self.games // 2)

    def play(self, part: range) -> "EvalCounts":
        """The counts of `part` (a sub-range of [0, half)), the checkpoint
        as `learner` -- freshly built policies, so the result is the same
        whichever process plays it."""
        policies = {
            "learner": NetOpponent.from_checkpoint(self.checkpoint, seed=self.seed, deterministic=True),
            self.opponent: parse_policy(self.opponent, seed=self.seed, device="cpu", deterministic=True)[1],
        }
        matches = play_slice("learner", self.opponent, policies, part=part, half=self.half, seed=self.seed,
                             events=self.events, us_bid=self.us_bid)
        return EvalCounts.from_matches(matches, "learner", self.opponent)


class EvalCounts(NamedTuple):
    """Per-seat wins and draws of an `EvalJob`: what the collectors hand
    back, and what adds up across them."""

    us_wins: int = 0
    us_draws: int = 0
    us_games: int = 0
    ussr_wins: int = 0
    ussr_draws: int = 0
    ussr_games: int = 0

    @classmethod
    def from_matches(cls, matches: Sequence[Match], learner: str, opponent: str) -> "EvalCounts":
        as_us = summarize([m for m in matches if m.a == learner], learner, opponent)
        as_ussr = summarize([m for m in matches if m.b == learner], learner, opponent)
        return cls(as_us["wins"], as_us["draws"], as_us["games"], as_ussr["wins"], as_ussr["draws"], as_ussr["games"])

    def __add__(self, other: tuple) -> "EvalCounts":  # type: ignore[override]
        return EvalCounts(*(a + b for a, b in zip(self, other)))

    @property
    def games(self) -> int:
        return self.us_games + self.ussr_games

    @staticmethod
    def _rate(wins: int, draws: int, games: int) -> float | None:
        return (wins + 0.5 * draws) / games if games else None

    @property
    def as_us(self) -> float | None:
        return self._rate(self.us_wins, self.us_draws, self.us_games)

    @property
    def as_ussr(self) -> float | None:
        return self._rate(self.ussr_wins, self.ussr_draws, self.ussr_games)

    @property
    def win_rate(self) -> float | None:
        return self._rate(self.us_wins + self.ussr_wins, self.us_draws + self.ussr_draws, self.games)


@dataclass(frozen=True)
class PairJob:
    """One pair to play: two policy specs (as `parse_policy` reads them) and
    the protocol. Plain data, so it crosses to a worker process as is."""

    spec_a: str
    spec_b: str
    games: int
    seed: int
    events: bool = True
    deterministic: bool = True
    device: str = "cpu"
    starting_vp: int = 0
    us_bid: int = 0

    @property
    def a(self) -> str:
        return policy_name(self.spec_a)

    @property
    def b(self) -> str:
        return policy_name(self.spec_b)


def _policy_seed(eval_seed: int, name: str) -> int:
    # Stable per (eval seed, policy name): the same policy gets the same
    # RNG stream in every pair of the same evaluation, whatever else ran.
    return eval_seed * 100_003 + zlib.crc32(name.encode()) % 100_003


def run_pair(job: PairJob) -> list[Match]:
    import torch

    torch.set_num_threads(WORKER_THREADS)
    policies = {
        policy_name(spec): parse_policy(
            spec, seed=_policy_seed(job.seed, policy_name(spec)), device=job.device, deterministic=job.deterministic
        )[1]
        for spec in (job.spec_a, job.spec_b)
    }
    return play_pair(
        job.a, job.b, policies, games=job.games, seed=job.seed, events=job.events, starting_vp=job.starting_vp,
        us_bid=job.us_bid,
    )


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 4)


def play_pairs(jobs: Sequence[PairJob], *, workers: int | None = None) -> list[list[Match]]:
    """Play every job, in a process pool of `workers` (default: a quarter of
    the CPUs; 1 runs them in this process). Results are in job order."""
    workers = default_workers() if workers is None else workers
    if workers <= 1 or len(jobs) <= 1:
        return [run_pair(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(run_pair, jobs))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Head-to-head evaluation of WOPR policies.")
    p.add_argument("policies", nargs="+")
    p.add_argument("--games", type=int, default=100, help="games per pair (split evenly between seat assignments)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample", action="store_true", help="sample network policies instead of taking their argmax")
    p.add_argument("--no-events", action="store_true")
    p.add_argument("--handicap", type=int, default=0, help="open every game with the US this many VP ahead (measures the USSR edge under a bid)")
    p.add_argument("--bid", type=int, default=0, help="the tournament bid: this much extra US influence placed after setup (11.1.4)")
    p.add_argument("--workers", type=int, default=None, help="pair-playing processes (default: CPUs / 4)")
    args = p.parse_args(argv)

    names = [policy_name(spec) for spec in args.policies]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate policy names in {names}")
    if len(names) < 2:
        raise ValueError("need at least two policies")
    for spec in args.policies:
        if spec not in ("random", "greedy", "first") and "=" not in spec:
            raise ValueError(f"policy {spec!r}: expected random|greedy|first or name=checkpoint.pt")

    jobs = [
        PairJob(spec_a, spec_b, args.games, args.seed, events=not args.no_events,
                deterministic=not args.sample, device=args.device, starting_vp=args.handicap, us_bid=args.bid)
        for spec_a, spec_b in itertools.combinations(args.policies, 2)
    ]
    matches: list[Match] = []
    for job, pair in zip(jobs, play_pairs(jobs, workers=args.workers)):
        a, b = job.a, job.b
        matches.extend(pair)
        overall = summarize(pair, a, b)
        as_us = summarize([m for m in pair if m.a == a], a, b)
        as_ussr = summarize([m for m in pair if m.b == a], a, b)
        print(
            f"{a} vs {b}: {overall['win_rate']:.3f} over {overall['games']} "
            f"(W{overall['wins']} D{overall['draws']} L{overall['losses']}) | "
            f"as US {as_us['win_rate']:.3f} | as USSR {as_ussr['win_rate']:.3f}"
        )
    anchors = {"random": 0.0} if "random" in names else None
    ratings = elo_ratings(matches, anchors=anchors)
    print("Elo:", ", ".join(f"{name} {rating:+.0f}" for name, rating in sorted(ratings.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
