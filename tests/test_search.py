"""Engine.determinize and the SearchPlayer (docs/WOPR.md, "Search over
the learned value head"): the determinized copy preserves exactly what
`observe(side)` shows and resamples everything else; the veto masks
provable losses; the value evaluator never prefers a certain terminal
loss and runs one exact simulation on branches that consume no
randomness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import bare_engine  # noqa: E402
from struggler.bots.joshua.model import JoshuaConfig, JoshuaNet  # noqa: E402
from struggler.bots.joshua.search import SearchPlayer  # noqa: E402
from struggler.engine import Engine, Side  # noqa: E402
from struggler.engine.core import HIDDEN_CARD  # noqa: E402
from struggler.engine.types import DecisionKind  # noqa: E402


def advance_to_decision(engine: Engine, steps: int) -> None:
    """Deterministically play `steps` first-legal options into the game,
    then continue to the next US/USSR decision."""
    for _ in range(steps):
        if engine.is_terminal:
            break
        engine.step(engine.pending_decision.options[0])
    while not engine.is_terminal and engine.pending_decision.actor is Side.CHANCE:
        engine.step(engine.pending_decision.options[0])


def card_multiset(data: dict) -> list[str]:
    cards = (
        list(data["draw_pile"]) + list(data["discard_pile"]) + list(data["removed_cards"])
        + list(data["hands"]["US"]) + list(data["hands"]["USSR"])
        + list(data["our_man_queue"]) + list(data["our_man_kept"])
        + [cid for cid in data["headline"].values() if cid is not None]
    )
    return sorted(cards)


# -- Engine.determinize -------------------------------------------------------


def test_determinize_preserves_the_movers_view_and_the_card_multiset():
    engine = Engine.new_game(seed=11)
    advance_to_decision(engine, 120)
    side = engine.pending_decision.actor
    det = engine.determinize(side, seed=7)
    assert det.observe(side) == engine.observe(side)
    assert card_multiset(det.serialize()) == card_multiset(engine.serialize())
    assert det.expose_chance_outcomes and not engine.expose_chance_outcomes


def test_determinize_resamples_hidden_state_and_is_deterministic_per_seed():
    engine = Engine.new_game(seed=11)
    advance_to_decision(engine, 120)
    side = engine.pending_decision.actor
    opponent = side.opponent.value
    hidden = (engine.serialize()["hands"][opponent], engine.serialize()["draw_pile"])
    resampled = [engine.determinize(side, seed=s).serialize() for s in range(6)]
    assert any((d["hands"][opponent], d["draw_pile"]) != hidden for d in resampled)
    assert engine.determinize(side, seed=3).serialize() == engine.determinize(side, seed=3).serialize()


def test_determinize_exposes_all_die_outcomes():
    engine = bare_engine()
    engine.board.influence["Angola"]["US"] = 2
    engine.begin_coup(Side.USSR, ops=2)
    det = engine.determinize(Side.USSR, seed=1)
    det.step(det.pending_decision.options[0])
    frame = det.pending_decision
    assert frame.actor is Side.CHANCE and frame.kind is DecisionKind.COUP_ROLL
    assert len(frame.options) == 6
    # The live engine still pre-rolls: one option, from its own seeded RNG.
    engine.step(engine.pending_decision.options[0])
    assert len(engine.pending_decision.options) == 1


def test_expose_chance_outcomes_serializes_only_when_set():
    engine = bare_engine()
    assert "expose_chance_outcomes" not in engine.serialize()
    engine.expose_chance_outcomes = True
    data = engine.serialize()
    assert data["expose_chance_outcomes"] is True
    assert Engine.deserialize(data).expose_chance_outcomes


def test_determinize_converts_a_physical_game():
    engine = bare_engine()
    engine.physical_mode = True
    engine.physical_side = Side.US
    real = sorted(engine.cards)[:4]
    engine.hands["US"] = [HIDDEN_CARD, HIDDEN_CARD]
    engine.hands["USSR"] = [real[0]]
    engine.draw_pile = [HIDDEN_CARD]
    engine.hidden_pool = real[1:]
    det = engine.determinize(Side.USSR, seed=5)
    assert not det.physical_mode and det.physical_side is None and det.hidden_pool == []
    assert det.hands["USSR"] == [real[0]]
    assert HIDDEN_CARD not in det.hands["US"] + det.draw_pile
    assert sorted(det.hands["US"] + det.draw_pile) == sorted(real[1:])


# -- SearchPlayer -------------------------------------------------------------


@pytest.fixture(scope="module")
def net() -> JoshuaNet:
    torch.manual_seed(0)
    return JoshuaNet(JoshuaConfig())


def coup_engine_defcon2() -> Engine:
    """USSR to pick a coup target at DEFCON 2: Angola (battleground --
    couping it drops DEFCON to 1 and loses the game for the couping side
    in the bare harness) beside non-battleground Kenya and Cameroon."""
    engine = bare_engine()
    engine.defcon = 2
    for country in ("Angola", "Kenya", "Cameroon"):
        engine.board.influence[country]["US"] = 2
    engine.begin_coup(Side.USSR, ops=2)
    return engine


@pytest.mark.parametrize("evaluator", ["terminal", "value"])
def test_search_never_picks_the_suicide_coup(net, evaluator):
    engine = coup_engine_defcon2()
    player = SearchPlayer(net, evaluator=evaluator, seed=0)
    player.bind(engine)
    action = player.choose_action(engine.observe(Side.USSR), [])
    assert action in engine.pending_decision.options
    assert action.payload["country"] != "Angola"


def test_probe_proves_the_granted_coup_mate(net):
    # The DEFCON gift's shape: during the USSR's action round the US holds
    # a granted coup with a battleground available at DEFCON 2 -- every die
    # outcome reaches DEFCON 1 and the *phasing* USSR loses. The probe must
    # prove it; the same position at DEFCON 4 proves nothing.
    for defcon, expected in ((2, True), (4, False)):
        engine = bare_engine()
        engine.defcon = defcon
        engine.phase = "action_rounds"
        engine._ars_played = 1  # play index 0 was the USSR's: the USSR is phasing
        engine.board.influence["Angola"]["USSR"] = 2
        engine.begin_coup(Side.US, ops=2)
        player = SearchPlayer(net, evaluator="terminal", seed=0)
        sim = engine.determinize(Side.USSR, seed=1)
        assert player._probe(sim, Side.USSR, player._boundary(sim), [500]) is expected


def test_value_search_runs_one_exact_simulation_on_deterministic_branches(net, monkeypatch):
    engine = bare_engine()
    engine.board.influence["Poland"]["USSR"] = 1
    engine.begin_influence_operations(Side.USSR, ops=1)
    n_options = len(engine.pending_decision.options)
    assert n_options > 1
    calls = []
    original = Engine.determinize

    def counting(self, side, seed):
        calls.append(seed)
        return original(self, side, seed)

    monkeypatch.setattr(Engine, "determinize", counting)
    player = SearchPlayer(net, evaluator="value", k=6, seed=0)
    player.bind(engine)
    action = player.choose_action(engine.observe(Side.USSR), [])
    assert action in engine.pending_decision.options
    # Placing one influence point rolls no die and draws no card: every
    # branch is exact from a single determinization, k notwithstanding.
    assert len(calls) == n_options
