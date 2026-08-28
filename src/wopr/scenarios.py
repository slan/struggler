"""Scenario banks: a prior over training start states.

The scenario-seeded self-play arc (docs/JOSHUA.md) trains a fraction of
games from positions where a known failure is on the table instead of the
printed setup. A *bank* is a JSONL file of serialized engine states
harvested from scripted games:

    python -m wopr.scenarios --out scenarios/defcon2-gift.jsonl \
        --games 400 --seed 1 --bid 2 --policy random

Line 1 is a header recording the game spec the states were generated
under (`us_bid`, `events`, `include_optional`, `starting_vp`) and the
generator (predicate, policy, seed); each later line is one entry: the
mover, the predicate that matched, where in the game it was
(`turn`/`action_round`), and `Engine.serialize()`.

Starting a training game from an entry does NOT replay the entry's game:
`ScenarioBank.start(index, seed)` deserializes the state and re-hides it
with `Engine.determinize(mover, seed)` -- the mover's own observation is
preserved exactly while the draw pile's order, the opponent's hand and
every future roll are resampled from `seed`. One bank entry therefore
yields as many distinct games as there are seeds, all consistent with
what the mover could know (mandate #4), and the arena's
(seed, slot, episode) determinism carries over unchanged.

Predicates (v1):

- ``defcon2_gift``: at an ACTION_ROUND_PLAY decision, DEFCON is 2 and
  the mover's hand holds a granted-op gift card -- the opponent's event
  that hands them a free coup (CIA Created for the USSR seat, Lone
  Gunman for the US). The failure shape the search arc closed on: the
  gift must be spent or spaced while it is still safe, which is a
  scheduling lesson no inference-time lookahead reaches.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from struggler.engine import Engine, Side
from struggler.engine.types import DecisionKind

#: The granted-op gift each seat can be caught holding: playing it for
#: ops fires the opponent's event, a free 1-Op coup at DEFCON 2.
GIFT_CARDS = {Side.USSR: ("CIA_Created",), Side.US: ("Lone_Gunman",)}


def defcon2_gift(engine: Engine) -> Side | None:
    """The mover, when the pending decision is an action-round card pick
    at DEFCON 2 with a gift card in the mover's hand; else None."""
    decision = engine.pending_decision
    if decision is None or decision.kind is not DecisionKind.ACTION_ROUND_PLAY:
        return None
    if engine.defcon != 2:
        return None
    actor = decision.actor
    if actor not in (Side.US, Side.USSR):
        return None
    hand = engine.hands[actor.value]
    return actor if any(card in hand for card in GIFT_CARDS[actor]) else None


PREDICATES: dict[str, Callable[[Engine], Side | None]] = {"defcon2_gift": defcon2_gift}

#: Header fields that must match the arena's game spec exactly.
SPEC_FIELDS = ("us_bid", "starting_vp", "events", "include_optional")


@dataclass(frozen=True)
class ScenarioBank:
    header: dict
    entries: tuple[dict, ...]

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioBank":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"scenario bank {path}: empty file")
        header = json.loads(lines[0])
        if header.get("kind") != "scenario-bank":
            raise ValueError(f"scenario bank {path}: missing header line")
        entries = tuple(json.loads(line) for line in lines[1:] if line)
        if not entries:
            raise ValueError(f"scenario bank {path}: no entries")
        return cls(header=header, entries=entries)

    def __len__(self) -> int:
        return len(self.entries)

    def validate(self, *, us_bid: int, starting_vp: int, events: bool, include_optional: bool) -> None:
        """A bank is only a prior over the game it was harvested from: the
        arena's spec must match the header's, field for field."""
        wanted = {"us_bid": us_bid, "starting_vp": starting_vp,
                  "events": events, "include_optional": include_optional}
        for field in SPEC_FIELDS:
            if self.header.get(field) != wanted[field]:
                raise ValueError(
                    f"scenario bank {field}={self.header.get(field)!r} does not match the arena's {wanted[field]!r}"
                )

    def start(self, index: int, seed: int) -> Engine:
        """A fresh game from entry `index`: the mover's observation as
        recorded, everything hidden from that seat resampled from `seed`."""
        entry = self.entries[index]
        engine = Engine.deserialize(entry["state"]).determinize(Side(entry["mover"]), seed)
        # `determinize` prepares a search copy; a training game rolls its
        # own dice one pre-drawn option at a time (mandate #3).
        engine.expose_chance_outcomes = False
        return engine


# -- generation ----------------------------------------------------------------


def _build_player(spec: str, seed: int):
    if spec == "random":
        from struggler.bots.naive import RandomPlayer

        return RandomPlayer(seed=seed)
    if spec == "greedy":
        from struggler.bots.greedy import GreedyPlayer

        return GreedyPlayer()
    if spec.startswith("joshua="):
        from struggler.bots.joshua.player import JoshuaPlayer

        # Sampled, not argmax: distinct games from distinct seeds.
        return JoshuaPlayer.from_checkpoint(spec.split("=", 1)[1], deterministic=False, seed=seed)
    raise ValueError(f"policy {spec!r}: expected random | greedy | joshua=checkpoint.pt")


def harvest(
    *,
    games: int,
    seed: int,
    policy: str,
    predicate: str,
    us_bid: int = 0,
    per_game: int = 4,
) -> ScenarioBank:
    """Play `games` scripted games and snapshot every first state per
    (turn, action round, mover) where `predicate` matches, up to
    `per_game` per game -- consecutive decisions of one stuck position
    would otherwise flood the bank with near-duplicates."""
    match = PREDICATES[predicate]
    entries: list[dict] = []
    for g in range(games):
        game_seed = seed + g
        engine = Engine.new_game(seed=game_seed, us_bid=us_bid)
        players = {
            Side.US: _build_player(policy, seed=game_seed * 2 + 1),
            Side.USSR: _build_player(policy, seed=game_seed * 2 + 2),
        }
        taken: set[tuple[int, int, Side]] = set()
        while not engine.is_terminal and len(taken) < per_game:
            decision = engine.pending_decision
            if decision.actor is Side.CHANCE:
                engine.step(decision.options[0])
                continue
            mover = match(engine)
            key = (engine.turn, engine.action_round, mover)
            if mover is not None and key not in taken:
                taken.add(key)
                entries.append({
                    "mover": mover.value,
                    "predicate": predicate,
                    "turn": engine.turn,
                    "action_round": engine.action_round,
                    "source_seed": game_seed,
                    "state": engine.serialize(),
                })
            actor = decision.actor
            engine.step(players[actor].choose_action(engine.observe(actor), ()))
    header = {
        "kind": "scenario-bank",
        "predicate": predicate,
        "policy": policy,
        "seed": seed,
        "games": games,
        "us_bid": us_bid,
        "starting_vp": 0,
        "events": True,
        "include_optional": True,
    }
    return ScenarioBank(header=header, entries=tuple(entries))


def save(bank: ScenarioBank, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(bank.header) + "\n")
        for entry in bank.entries:
            f.write(json.dumps(entry) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Harvest a scenario bank from scripted games.")
    p.add_argument("--out", required=True, help="the bank file (JSONL)")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--policy", default="random", help="random | greedy | joshua=checkpoint.pt (sampled)")
    p.add_argument("--predicate", choices=sorted(PREDICATES), default="defcon2_gift")
    p.add_argument("--bid", type=int, default=0, help="the tournament bid the states are harvested under")
    p.add_argument("--per-game", type=int, default=4, help="most snapshots taken from one game")
    args = p.parse_args(argv)
    bank = harvest(games=args.games, seed=args.seed, policy=args.policy,
                   predicate=args.predicate, us_bid=args.bid, per_game=args.per_game)
    save(bank, args.out)
    by_mover = {side.value: sum(1 for e in bank.entries if e["mover"] == side.value) for side in (Side.US, Side.USSR)}
    print(f"{len(bank)} states from {args.games} games -> {args.out} (by mover: {by_mover})")


if __name__ == "__main__":
    main()
