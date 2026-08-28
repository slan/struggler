"""Scenario banks (wopr.scenarios): the harvest predicate, determinized
restarts, and the arena's seeded scenario draw. Torch-free."""

from __future__ import annotations

import pytest

from struggler.engine import Engine, Side
from struggler.engine.types import DecisionKind
from wopr.arena import Arena
from wopr.scenarios import GIFT_CARDS, ScenarioBank, harvest, save

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def bank() -> ScenarioBank:
    harvested = harvest(games=10, seed=11, policy="random", predicate="defcon2_gift")
    assert len(harvested) > 0, "no defcon2_gift states in 10 random games; pick another seed"
    return harvested


def test_harvested_states_match_the_predicate(bank: ScenarioBank):
    for entry in bank.entries:
        engine = Engine.deserialize(entry["state"])
        mover = Side(entry["mover"])
        assert engine.defcon == 2
        assert engine.pending_decision.kind is DecisionKind.ACTION_ROUND_PLAY
        assert engine.pending_decision.actor is mover
        assert any(card in engine.hands[mover.value] for card in GIFT_CARDS[mover])


def test_start_preserves_the_movers_observation_and_resamples_the_rest(bank: ScenarioBank):
    entry = bank.entries[0]
    mover = Side(entry["mover"])
    original = Engine.deserialize(entry["state"])

    first, second = bank.start(0, seed=5), bank.start(0, seed=5)
    assert first.serialize() == second.serialize()  # deterministic in the seed
    assert not first.expose_chance_outcomes  # a training game, not a search copy
    assert first.observe(mover) == original.observe(mover)  # mandate #4: nothing the mover knows moved

    hidden = [bank.start(0, seed=s).serialize() for s in (5, 6, 7)]
    assert any(d["draw_pile"] != hidden[0]["draw_pile"] or d["hands"] != hidden[0]["hands"]
               for d in hidden[1:]), "three seeds, one hidden world: not resampled"


def test_scenario_game_plays_to_completion(bank: ScenarioBank):
    from struggler.bots.naive import RandomPlayer

    engine = bank.start(0, seed=9)
    players = {Side.US: RandomPlayer(seed=1), Side.USSR: RandomPlayer(seed=2)}
    while not engine.is_terminal:
        decision = engine.pending_decision
        if decision.actor is Side.CHANCE:
            engine.step(decision.options[0])
        else:
            actor = decision.actor
            engine.step(players[actor].choose_action(engine.observe(actor), ()))
    assert engine.winner in (Side.US, Side.USSR, None)


def test_arena_scenario_draw_is_a_function_of_the_game_seed(bank: ScenarioBank, tmp_path):
    path = tmp_path / "bank.jsonl"
    save(bank, path)
    loaded = ScenarioBank.load(path)
    assert loaded.entries == bank.entries and loaded.header == bank.header

    kwargs = dict(seed=3, scenario_bank=loaded, scenario_frac=1.0)
    whole = Arena(4, **kwargs)
    sliced = [Arena(2, slot_offset=0, total_slots=4, **kwargs),
              Arena(2, slot_offset=2, total_slots=4, **kwargs)]
    for slot in range(4):
        want = whole.engine(slot).serialize()
        part = sliced[slot // 2].engine(slot % 2).serialize()
        assert part == want  # k sliced arenas play the whole arena's games
        assert want["defcon"] == 2  # frac 1.0: every game starts in the scenario

    again = Arena(4, **kwargs)
    assert [again.engine(i).serialize() for i in range(4)] == [whole.engine(i).serialize() for i in range(4)]


def test_arena_validates_the_banks_game_spec(bank: ScenarioBank):
    with pytest.raises(ValueError, match="us_bid"):
        Arena(2, seed=3, scenario_bank=bank, scenario_frac=0.5, us_bid=2)
    with pytest.raises(ValueError, match="scenario_bank"):
        Arena(2, seed=3, scenario_frac=0.5)


def test_arena_without_scenarios_is_unchanged(bank: ScenarioBank):
    plain = Arena(3, seed=5)
    with_bank_off = Arena(3, seed=5, scenario_bank=bank, scenario_frac=0.0)
    for slot in range(3):
        assert plain.engine(slot).serialize() == with_bank_off.engine(slot).serialize()
