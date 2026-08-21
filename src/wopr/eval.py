"""Evaluate checkpoints head to head on fixed seeds.

    python -m wopr.eval --games 200 joshua=runs/first/joshua.pt random
    python -m wopr.eval --games 200 new=runs/b/joshua.pt old=runs/a/pool/u00010.pt greedy

Each positional argument is a policy: `random`, `greedy`, `first`, or
`name=path/to/checkpoint.pt`. Every pair plays `--games` games, half with
each seat assignment (the seeds are shared across pairs, so two policies
facing the same third one see the same decks). Prints per-pair, per-seat
win rates and Elo with `random` anchored at 0 when present.
"""

from __future__ import annotations

import argparse
import itertools
from typing import Mapping

from struggler.bots.greedy import GreedyPlayer
from struggler.bots.naive import FirstLegalPlayer
from struggler.engine import Side
from wopr.arena import Arena, Opponent, play_out
from wopr.ladder import Match, elo_ratings, summarize
from wopr.opponents import NetOpponent, PlayerOpponent, RandomOpponent


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


def play_pair(a: str, b: str, policies: Mapping[str, Opponent], *, games: int, seed: int, events: bool) -> list[Match]:
    half = max(1, games // 2)

    def assign_a_us(slot: int, episode: int, rng) -> dict[Side, str]:
        return {Side.US: a, Side.USSR: b}

    def assign_a_ussr(slot: int, episode: int, rng) -> dict[Side, str]:
        return {Side.US: b, Side.USSR: a}

    matches: list[Match] = []
    for assigner in (assign_a_us, assign_a_ussr):
        arena = Arena(half, seed=seed, seat_assigner=assigner, events=events)
        for result in play_out(arena, policies):
            # Recorded US-first: `Match.a` is whoever sat as US.
            score_us = 0.5 if result.winner is None else (1.0 if result.winner is Side.US else 0.0)
            matches.append(Match(result.seats[Side.US], result.seats[Side.USSR], score_us))
    return matches


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Head-to-head evaluation of WOPR policies.")
    p.add_argument("policies", nargs="+")
    p.add_argument("--games", type=int, default=100, help="games per pair (split evenly between seat assignments)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample", action="store_true", help="sample network policies instead of taking their argmax")
    p.add_argument("--no-events", action="store_true")
    args = p.parse_args(argv)

    policies: dict[str, Opponent] = {}
    for i, spec in enumerate(args.policies):
        name, policy = parse_policy(spec, seed=args.seed + i, device=args.device, deterministic=not args.sample)
        if name in policies:
            raise ValueError(f"duplicate policy name {name!r}")
        policies[name] = policy
    if len(policies) < 2:
        raise ValueError("need at least two policies")

    matches: list[Match] = []
    for a, b in itertools.combinations(policies, 2):
        pair = play_pair(a, b, policies, games=args.games, seed=args.seed, events=not args.no_events)
        matches.extend(pair)
        overall = summarize(pair, a, b)
        as_us = summarize([m for m in pair if m.a == a], a, b)
        as_ussr = summarize([m for m in pair if m.b == a], a, b)
        print(
            f"{a} vs {b}: {overall['win_rate']:.3f} over {overall['games']} "
            f"(W{overall['wins']} D{overall['draws']} L{overall['losses']}) | "
            f"as US {as_us['win_rate']:.3f} | as USSR {as_ussr['win_rate']:.3f}"
        )
    anchors = {"random": 0.0} if "random" in policies else None
    ratings = elo_ratings(matches, anchors=anchors)
    print("Elo:", ", ".join(f"{name} {rating:+.0f}" for name, rating in sorted(ratings.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
