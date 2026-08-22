"""How a policy's games actually go: the probes behind every r1 finding.

    python -m wopr.diagnose runs/first/joshua.pt [--games 120] [--vs greedy]

Plays the checkpoint against itself (argmax, seats alternating) and, with
`--vs`, against an opponent, and reports what the win rate cannot:

- how the games end (by VP, final scoring, DEFCON, a held scoring card, a
  draw), by winner, with the mean final turn of each;
- the USSR edge: the checkpoint's win rate as USSR against itself;
- the VP track at the start of each turn, averaged;
- where the VP came from: net VP by card, to each side.

`--json` writes the same as a file (what a ledger row's note should point
at). The loop's gate says whether a version is better; this says *how it
plays*, which is what decides the next experiment (WOPR.md, "Decision
points").
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any

from struggler.engine import Side
from struggler.engine.types import DecisionKind
from wopr.arena import Arena, Opponent
from wopr.eval import parse_policy


def play_traced(arena: Arena, policies: dict[str, Opponent]) -> dict[str, Any]:
    """`play_out`, with the VP track watched: every change attributed to
    the turn, the actor and the card the actor last chose to play."""
    n = arena.n_games
    last_card = [{Side.US: "?", Side.USSR: "?"} for _ in range(n)]
    by_card: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # card -> [to USSR, to US]
    track: dict[int, list[int]] = collections.defaultdict(list)  # turn -> vp at its start
    seen_turn = [0] * n
    while True:
        pending = arena.pending()
        if not pending:
            break
        for policy_id, rows in pending.items():
            choices = policies[policy_id].choose(rows)
            for row, choice in zip(rows, choices):
                engine = arena.engine(row.slot)
                decision = engine.pending_decision
                action = decision.options[int(choice)]
                card = action.payload.get("card")
                if card and decision.kind in (DecisionKind.HEADLINE_PLAY, DecisionKind.ACTION_ROUND_PLAY):
                    last_card[row.slot][decision.actor] = str(card)
                before, turn = engine.vp, engine.turn
                if turn > seen_turn[row.slot]:
                    seen_turn[row.slot] = turn
                    track[turn].append(before)
                arena.apply(row.slot, int(choice))
                delta = engine.vp - before
                if delta:
                    key = last_card[row.slot][decision.actor]
                    if decision.kind is DecisionKind.HEADLINE_PLAY:
                        key = f"headline:{key}"
                    by_card[key][0 if delta < 0 else 1] += abs(delta)
    endings: collections.Counter = collections.Counter()
    turns: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    results = []
    for slot in range(n):
        result = arena.result(slot)
        engine = arena.engine(slot)
        reason = engine.serialize()["game_over_reason"] or "draw"
        winner = "-" if result.winner is None else result.winner.value
        endings[(winner, reason)] += 1
        turns[(winner, reason)].append(result.turn)
        results.append(result)
    return {
        "games": n,
        "endings": [
            {"winner": w, "reason": r, "games": c, "mean_turn": round(statistics.fmean(turns[(w, r)]), 2)}
            for (w, r), c in endings.most_common()
        ],
        "mean_final_turn": round(statistics.fmean(r.turn for r in results), 2),
        "track": {t: round(statistics.fmean(v), 2) for t, v in sorted(track.items())},
        "vp_by_card": sorted(
            ({"card": k, "to_ussr": v[0], "to_us": v[1]} for k, v in by_card.items()),
            key=lambda e: -(e["to_ussr"] + e["to_us"]),
        ),
        "results": results,
    }


def diagnose(checkpoint: str, *, games: int, seed: int, versus: str | None, device: str) -> dict[str, Any]:
    def seats(slot: int, episode: int, rng) -> dict[Side, str]:
        return {Side.US: "a", Side.USSR: "b"} if slot % 2 == 0 else {Side.US: "b", Side.USSR: "a"}

    a = parse_policy(f"a={checkpoint}", seed=seed, device=device, deterministic=True)[1]
    b = parse_policy(f"b={checkpoint}", seed=seed + 1, device=device, deterministic=True)[1]
    report: dict[str, Any] = {"checkpoint": checkpoint, "games": games, "seed": seed}
    traced = play_traced(Arena(games, seed=seed, seat_assigner=seats), {"a": a, "b": b})
    ussr_wins = sum(1 for r in traced["results"] if r.winner is Side.USSR)
    draws = sum(1 for r in traced["results"] if r.winner is None)
    report["self"] = {k: v for k, v in traced.items() if k != "results"}
    report["self"]["ussr_edge"] = round(ussr_wins / games, 3)
    report["self"]["draw_rate"] = round(draws / games, 3)
    if versus:
        name, opponent = parse_policy(versus if versus in ("random", "greedy", "first") else f"vs={versus}", seed=seed + 2, device=device, deterministic=True)
        traced = play_traced(Arena(games, seed=seed, seat_assigner=seats), {"a": a, "b": opponent})
        wins = sum(1 for r in traced["results"] if r.winner is not None and r.seats[r.winner] == "a")
        report["versus"] = {"opponent": name, "win_rate": round(wins / games, 3), **{k: v for k, v in traced.items() if k != "results"}}
    return report


def render(report: dict[str, Any]) -> str:
    lines = [f"{report['checkpoint']}: {report['games']} games against itself (seed {report['seed']})"]
    me = report["self"]
    lines.append(f"  USSR edge {me['ussr_edge']:.3f}, draws {me['draw_rate']:.3f}, mean final turn {me['mean_final_turn']}")
    lines.append("  endings: " + ", ".join(f"{e['winner']} by {e['reason']} {e['games']} (turn {e['mean_turn']})" for e in me["endings"]))
    lines.append("  VP at start of turn: " + " ".join(f"{t}:{v:+.1f}" for t, v in me["track"].items()))
    lines.append("  net VP by card (to USSR / to US):")
    for e in me["vp_by_card"][:12]:
        lines.append(f"    {e['card']:36s} {e['to_ussr']:5d} / {e['to_us']:5d}")
    if "versus" in report:
        vs = report["versus"]
        lines.append(f"  vs {vs['opponent']}: win rate {vs['win_rate']:.3f}, mean final turn {vs['mean_final_turn']}")
        lines.append("    endings: " + ", ".join(f"{e['winner']} by {e['reason']} {e['games']}" for e in vs["endings"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="How a policy's games go: endings, the USSR edge, the VP track, VP by card.")
    p.add_argument("checkpoint")
    p.add_argument("--games", type=int, default=120, help="games against itself (and against --vs)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vs", default=None, help="also play this opponent: random | greedy | first | a checkpoint")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", default=None, help="write the report here as JSON")
    args = p.parse_args(argv)
    report = diagnose(args.checkpoint, games=args.games, seed=args.seed, versus=args.vs, device=args.device)
    print(render(report), end="")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
