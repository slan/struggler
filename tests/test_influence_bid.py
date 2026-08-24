"""The tournament bid (rule 11.1.4): extra US influence after regular setup.

A bid of N gives the US N extra influence, placed once the regular setup
placements (USSR 6 in Eastern Europe, US 7 in Western Europe) are done,
only into countries that hold US influence at that moment, and never past
two more than what control of the country needs (11.1.4.1).
"""

from __future__ import annotations

import pytest

from struggler.engine import Engine, Side
from struggler.engine.replay import make_engine


def place(engine: Engine, country: str) -> None:
    decision = engine.pending_decision
    action = next(a for a in decision.options if a.payload["country"] == country)
    engine.step(action)


def run_setup(engine: Engine, us: tuple[str, ...] = ("Italy",) * 7) -> None:
    for _ in range(6):
        place(engine, "East_Germany")
    for country in us:
        place(engine, country)


def test_bid_influence_goes_only_where_the_us_already_is_and_counts():
    engine = Engine.new_game(seed=5, us_bid=2)
    # Spread below every cap, so the option set is exactly "has US influence".
    run_setup(engine, ("West_Germany",) * 4 + ("France",) * 3)
    decision = engine.pending_decision
    assert decision.actor is Side.US and decision.context.get("bid") and decision.context["remaining"] == 2
    offered = {a.payload["country"] for a in decision.options}
    assert offered == {cid for cid, inf in engine.board.influence.items() if inf["US"] > 0}
    assert "East_Germany" not in offered  # USSR influence alone does not qualify
    before = engine.board.influence["UK"]["US"]
    place(engine, "UK")
    place(engine, "UK")
    assert engine.board.influence["UK"]["US"] == before + 2
    assert engine.pending_decision.context.get("bid") is None  # the bid is spent; on to the headline


def test_bid_placement_stops_two_past_what_control_needs():
    engine = Engine.new_game(seed=5, us_bid=3)
    run_setup(engine)  # Italy (stability 2, USSR 0): 7 setup influence, far past control + 2 = 4
    offered = {a.payload["country"] for a in engine.pending_decision.options}
    assert "Italy" in {cid for cid, inf in engine.board.influence.items() if inf["US"] > 0}
    assert "Italy" not in offered  # 11.1.4.1: already at 7 >= 2 + 0 + 2
    # Panama: printed 1, stability 2, USSR 0 -> capped at 4: two placements offered, not a third.
    place(engine, "Panama")
    place(engine, "Panama")
    assert engine.board.influence["Panama"]["US"] == 3
    assert "Panama" in {a.payload["country"] for a in engine.pending_decision.options}
    place(engine, "Panama")
    assert engine.board.influence["Panama"]["US"] == 4
    assert engine.pending_decision.context.get("bid") is None


def test_bid_survives_serialization_and_replay_logs():
    engine = Engine.new_game(seed=9, us_bid=2)
    run_setup(engine)
    copy = Engine.deserialize(engine.serialize())
    assert copy.us_bid == 2
    assert [a.payload for a in copy.pending_decision.options] == [a.payload for a in engine.pending_decision.options]
    replayed = make_engine({"seed": 9, "new_game": True, "include_optional": True, "events": True, "us_bid": 2})
    assert replayed.us_bid == 2

    plain = Engine.new_game(seed=9)
    assert plain.us_bid == 0
    run_setup(plain)
    assert plain.pending_decision.context.get("bid") is None  # bid 0 is the printed game


def test_new_game_rejects_a_negative_bid():
    with pytest.raises(ValueError):
        Engine.new_game(seed=1, us_bid=-1)
