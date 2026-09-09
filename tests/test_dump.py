"""The dump: the turn-horizon rider over the gift cards on the veto player
(`bots/joshua/search.py`, docs/JOSHUA.md 2026-09-09). Positions are built
directly: `side` to pick its action-round card with a chosen hand, DEFCON
and number of its own action rounds still to come."""

from __future__ import annotations

import pytest
import torch

from conftest import bare_engine  # noqa: E402
from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet  # noqa: E402
from struggler.bots.joshua.search import GIFT_CARDS, SearchPlayer  # noqa: E402
from struggler.engine import Engine, Side  # noqa: E402
from struggler.engine.cards import action_rounds  # noqa: E402
from struggler.engine.types import DecisionKind  # noqa: E402

USSR_FILLERS = ("Nasser", "Fidel", "Decolonization", "Blockade", "Warsaw_Pact_Formed", "Suez_Crisis")
US_FILLERS = ("Truman_Doctrine", "Marshall_Plan", "Containment", "NATO", "Independent_Reds", "Formosan_Resolution")


@pytest.fixture(scope="module")
def net() -> JoshuaNet:
    torch.manual_seed(0)
    return JoshuaNet(JoshuaConfig())


def action_round(side: Side, hand: list[str], *, defcon: int, plays_left: int, turn: int = 1) -> Engine:
    """`side` to pick its action-round card on `turn`, `plays_left` of its
    own action rounds still to come (this one included)."""
    engine = bare_engine()
    engine.turn = turn
    engine.defcon = defcon
    engine.phase = "action_rounds"
    k = action_rounds(turn) - plays_left  # the mover's 0-based round index
    engine._ars_played = 2 * k + (1 if side is Side.USSR else 2)  # play index + 1; the USSR plays the even indices
    engine.action_round = k + 1
    fillers = US_FILLERS if side is Side.USSR else USSR_FILLERS
    engine.hands = {side.value: list(hand), side.opponent.value: list(fillers[:4])}
    engine._push_action_round_play(side)
    assert engine.pending_decision.kind is DecisionKind.ACTION_ROUND_PLAY and engine.pending_decision.actor is side
    return engine


def dump_of(net: JoshuaNet, engine: Engine, side: Side) -> tuple[str, str | None] | None:
    """What the rider would do at the pending card decision: (card, mode) or None."""
    player = SearchPlayer(net, evaluator="terminal", dump=True, seed=0)
    player.bind(engine)
    decision = engine.pending_decision
    picked = player._choose_dump(side, decision, player._policy_logits(engine.observe(side)))
    if picked is None:
        return None
    index, mode = picked
    return decision.options[index].payload["card"], mode


def test_the_gift_cards_are_opponent_events():
    engine = bare_engine()
    for side, cards in GIFT_CARDS.items():
        for cid in cards:
            assert engine._is_opponent_event(side, engine.cards[cid]), (side, cid)


def test_dump_plays_the_unspaceable_gift_in_the_window(net):
    # CIA Created (1 Op) can never be spaced: it leaves at the first window
    # (DEFCON >= 3), whether or not the turn's arithmetic forces it.
    for plays_left in (4, 3):  # hand 4: forced at 4 plays left, holdable at 3
        engine = action_round(Side.USSR, ["CIA_Created", *USSR_FILLERS[:3]], defcon=4, plays_left=plays_left)
        assert dump_of(net, engine, Side.USSR) == ("CIA_Created", None)
        player = SearchPlayer(net, evaluator="terminal", dump=True, seed=0)
        player.bind(engine)
        assert player.choose_action(engine.observe(Side.USSR), []).payload["card"] == "CIA_Created"
    # At DEFCON 2 the window is shut and the card has no safe door: the
    # rider stays out of it (the veto's question now).
    engine = action_round(Side.USSR, ["CIA_Created", *USSR_FILLERS[:3]], defcon=2, plays_left=4)
    assert dump_of(net, engine, Side.USSR) is None


def test_dump_leaves_a_disposable_gift_until_the_turn_forces_it(net):
    # Tear Down This Wall (3 Ops) can be spaced: with a card to spare it is
    # held; once every card must be played this turn it leaves in the window.
    engine = action_round(Side.USSR, ["Tear_Down_This_Wall", *USSR_FILLERS[:3]], defcon=4, plays_left=3)
    assert dump_of(net, engine, Side.USSR) is None
    engine = action_round(Side.USSR, ["Tear_Down_This_Wall", *USSR_FILLERS[:3]], defcon=4, plays_left=4)
    assert dump_of(net, engine, Side.USSR) == ("Tear_Down_This_Wall", None)


def test_dump_spaces_a_forced_gift_at_defcon_2(net):
    # Two cards, two plays left, DEFCON 2: Grain Sales cannot be held and
    # goes into the Space Race now -- the card pick, then its PLAY_MODE.
    engine = action_round(Side.USSR, ["Grain_Sales_to_Soviets", "Nasser"], defcon=2, plays_left=2)
    assert dump_of(net, engine, Side.USSR) == ("Grain_Sales_to_Soviets", "space_race")
    player = SearchPlayer(net, evaluator="terminal", dump=True, seed=0)
    player.bind(engine)
    action = player.choose_action(engine.observe(Side.USSR), [])
    assert action.payload["card"] == "Grain_Sales_to_Soviets"
    engine.step(action)
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.PLAY_MODE and decision.context["card"] == "Grain_Sales_to_Soviets"
    assert player.choose_action(engine.observe(Side.USSR), []).payload["mode"] == "space_race"
    assert player._dump_intent is None
    # One play left: the gift is the card carried over, nothing to do.
    engine = action_round(Side.USSR, ["Grain_Sales_to_Soviets", "Nasser"], defcon=2, plays_left=1)
    assert dump_of(net, engine, Side.USSR) is None
    # Forced but with no safe door (a 1-Op gift): the rider stays out of it.
    engine = action_round(Side.USSR, ["CIA_Created", "Nasser"], defcon=2, plays_left=2)
    assert dump_of(net, engine, Side.USSR) is None


def test_dump_reads_the_us_seat_too(net):
    engine = action_round(Side.US, ["Lone_Gunman", *US_FILLERS[:2]], defcon=3, plays_left=2)
    assert dump_of(net, engine, Side.US) == ("Lone_Gunman", None)
    engine = action_round(Side.US, list(US_FILLERS[:3]), defcon=3, plays_left=2)
    assert dump_of(net, engine, Side.US) is None


def test_dump_intent_is_dropped_when_the_next_decision_is_not_the_gifts_mode(net):
    player = SearchPlayer(net, evaluator="terminal", dump=True, seed=0)
    player._dump_intent = ("Grain_Sales_to_Soviets", "space_race")
    engine = action_round(Side.USSR, ["Nasser", "Fidel", "Blockade"], defcon=4, plays_left=3)
    player.bind(engine)
    action = player.choose_action(engine.observe(Side.USSR), [])
    assert action in engine.pending_decision.options
    assert player._dump_intent is None
