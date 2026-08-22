"""Tests for GreedyPlayer: the board evaluator (and its local `_swing`
shortcut), DEFCON-safety heuristic, fallback behavior, and a win-rate sanity
check against RandomPlayer."""

from __future__ import annotations

import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from struggler.bots.greedy import GreedyPlayer, GreedyWeights, _swing, board_value
from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
from struggler.engine import Action, Decision, DecisionKind, Engine, Region, ScoringTier, Side
from struggler.engine.board import Board
from struggler.runner import play_game


def test_board_value_zero_with_no_influence_anywhere():
    board = Board()
    assert board_value(GreedyWeights(), board, Side.US) == 0.0


def test_board_value_rewards_controlling_a_battleground_over_a_non_battleground():
    weights = GreedyWeights()
    battleground = next(cid for cid, info in Board().countries.items() if info.battleground)
    non_battleground = next(cid for cid, info in Board().countries.items() if not info.battleground)

    bg_board = Board()
    bg_board.influence[battleground]["US"] = bg_board.countries[battleground].stability
    bg_value = board_value(weights, bg_board, Side.US)

    plain_board = Board()
    plain_board.influence[non_battleground]["US"] = plain_board.countries[non_battleground].stability
    plain_value = board_value(weights, plain_board, Side.US)

    assert bg_value > plain_value > 0.0


def test_greedy_avoids_coup_as_an_ops_type_at_defcon_2():
    """DEFCON 2 -> 1 loses the game for whoever caused the drop (mandate:
    priority #1: never die to DEFCON). Even
    with a juicy Coup target on offer, GreedyPlayer must pick something
    else at the OPS_TYPE decision."""
    engine = Engine(seed=1)
    # Mexico: a Battleground with no DEFCON region-lock, so a Coup here risks DEFCON 1
    # even at DEFCON 2 (unlike Europe/Asia/Middle East, which lock out first).
    engine.board.influence["Mexico"]["USSR"] = 3
    engine._change_defcon(-3, caused_by=Side.US)  # 5 -> 2
    assert engine.defcon == 2

    engine._push_ops_type(Side.US, ops=3)
    observation = engine.observe(Side.US)
    assert observation.pending_decision.kind is DecisionKind.OPS_TYPE
    offered = {a.payload["type"] for a in observation.pending_decision.options}
    assert "coup" in offered  # the engine itself doesn't forbid the suicidal option

    action = GreedyPlayer().choose_action(observation, [])
    assert action.payload["type"] != "coup"


def test_greedy_falls_back_to_first_option_for_unmapped_decision_kinds():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    fallback_decision = Decision(
        id=999,
        actor=observation.side,
        kind=DecisionKind.EVENT_CHOICE,
        options=(
            Action(DecisionKind.EVENT_CHOICE, {"choice": "a"}),
            Action(DecisionKind.EVENT_CHOICE, {"choice": "b"}),
        ),
    )
    observation = dataclasses.replace(observation, pending_decision=fallback_decision)

    action = GreedyPlayer().choose_action(observation, [])

    assert action == fallback_decision.options[0]


def test_greedy_aldrich_ames_remix_discards_the_opponents_highest_ops_card():
    """Unlike the generic EVENT_CHOICE fallback above, Aldrich Ames Remix has
    its own heuristic: force the discard of the US hand's highest-Ops card,
    rather than blindly taking whichever option came first."""
    engine = Engine.new_game(seed=1)
    engine.hands["US"] = ["Nasser", "Fidel", "Duck_and_Cover"]  # Ops 1, 2, 3
    engine._fire_event(Side.USSR, "Aldrich_Ames_Remix")
    observation = engine.observe(Side.USSR)
    assert observation.pending_decision.kind is DecisionKind.EVENT_CHOICE

    action = GreedyPlayer().choose_action(observation, [])

    assert action.payload["choice"] == "Duck_and_Cover"


_N_COUNTRIES = len(Board().countries)


@given(
    influence=st.lists(st.tuples(st.integers(0, 6), st.integers(0, 6)), min_size=_N_COUNTRIES, max_size=_N_COUNTRIES),
    index=st.integers(0, _N_COUNTRIES - 1),
    deltas=st.fixed_dictionaries({}, optional={"US": st.integers(-6, 6), "USSR": st.integers(-6, 6)}),
    us=st.booleans(),
)
@settings(max_examples=300, deadline=None)
def test_swing_equals_the_board_value_difference_of_a_full_recount(influence, index, deltas, us):
    """`_swing` is `board_value` computed locally (one country's Control
    bonus and its region's tiers); it must agree exactly with the full
    recount and leave the board as it found it."""
    weights = GreedyWeights()
    side = Side.US if us else Side.USSR
    board = Board()
    ids = list(board.countries)
    for cid, (us_inf, ussr_inf) in zip(ids, influence):
        board.influence[cid]["US"] = us_inf
        board.influence[cid]["USSR"] = ussr_inf
    country = ids[index]
    # Influence never goes negative on a real board; clamp the random deltas.
    deltas = {key: max(delta, -board.influence[country][key]) for key, delta in deltas.items()}
    snapshot = board.snapshot_influence()

    before = board_value(weights, board, side)
    swing = _swing(weights, board, side, country, deltas)
    assert board.influence == snapshot

    for key, delta in deltas.items():
        board.influence[country][key] += delta
    after = board_value(weights, board, side)
    assert swing == after - before


@pytest.mark.parametrize("holder", [Side.US, Side.USSR])
def test_greedy_treats_europe_scoring_at_control_as_the_game(holder):
    """Europe has no Control scoring value (`Board.score_region` raises
    rather than guess), and a learned opponent reached exactly that board
    against Greedy. Whoever holds the tier should play the card at once;
    the other side should hold anything else instead."""
    engine = Engine.new_game(seed=1)
    for cid in engine.board.countries_in(Region.EUROPE):
        if engine.board.countries[cid].battleground or cid in ("Austria", "Finland"):
            influence = engine.board.influence[cid]
            influence[holder.value] = influence[holder.opponent.value] + engine.board.countries[cid].stability
    assert engine.board.region_tier(holder, Region.EUROPE) is ScoringTier.CONTROL
    assert engine.board.controls_all_of_europe() is None  # Control tier, not every country

    for side in (Side.US, Side.USSR):
        observation = engine.observe(side)
        decision = Decision(
            id=999,
            actor=side,
            kind=DecisionKind.ACTION_ROUND_PLAY,
            options=(
                Action(DecisionKind.ACTION_ROUND_PLAY, {"card": "Duck_and_Cover"}),
                Action(DecisionKind.ACTION_ROUND_PLAY, {"card": "Europe_Scoring"}),
            ),
        )
        observation = dataclasses.replace(observation, pending_decision=decision)
        action = GreedyPlayer().choose_action(observation, [])
        expected = "Europe_Scoring" if side is holder else "Duck_and_Cover"
        assert action.payload["card"] == expected
