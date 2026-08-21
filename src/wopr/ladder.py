"""Ratings over a set of policies from head-to-head results.

"Win rate against the scripted bot" stops meaning anything once the
learner beats it every time; a rating ladder with fixed anchors (Random,
Greedy) keeps measuring progress after that, and tells apart "the pool is
climbing" from "the pool is rotating". Plain Elo, fitted by repeated
sweeps over all results rather than one online pass, so the order games
were played in does not matter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Match:
    a: str
    b: str
    score_a: float  # 1 win, 0.5 draw, 0 loss, from a's perspective


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def elo_ratings(
    matches: Iterable[Match],
    *,
    anchors: Mapping[str, float] | None = None,
    k: float = 8.0,
    sweeps: int = 50,
    initial: float = 1000.0,
) -> dict[str, float]:
    """Fit Elo ratings to `matches`. `anchors` pins named policies to fixed
    ratings (e.g. `{"random": 0.0}`) so numbers stay comparable across runs."""
    matches = list(matches)
    anchors = dict(anchors or {})
    ratings: dict[str, float] = {}
    for m in matches:
        ratings.setdefault(m.a, initial)
        ratings.setdefault(m.b, initial)
    ratings.update(anchors)
    for _ in range(sweeps):
        for m in matches:
            expected = expected_score(ratings[m.a], ratings[m.b])
            delta = k * (m.score_a - expected)
            if m.a not in anchors:
                ratings[m.a] += delta
            if m.b not in anchors:
                ratings[m.b] -= delta
    return ratings


def summarize(matches: Iterable[Match], a: str, b: str) -> dict[str, float]:
    """Win/draw/loss counts of `a` against `b` over `matches` (either order)."""
    wins = draws = losses = 0
    for m in matches:
        if {m.a, m.b} != {a, b}:
            continue
        score = m.score_a if m.a == a else 1.0 - m.score_a
        if score == 1.0:
            wins += 1
        elif score == 0.0:
            losses += 1
        else:
            draws += 1
    games = wins + draws + losses
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": (wins + 0.5 * draws) / games if games else float("nan"),
    }
