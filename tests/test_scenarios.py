"""Scenario banks (wopr.scenarios): the harvest predicate, determinized
restarts, and the arena's seeded scenario draw. Torch-free."""

from __future__ import annotations

import pytest

from struggler.engine import Engine, Side
from struggler.engine.types import DecisionKind
from wopr.arena import Arena
from wopr.scenarios import GIFT_CARDS, ScenarioBank, harvest, harvest_logs, log_paths, save

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


def test_us_opening_states_from_a_replay_log(tmp_path):
    # A log the engine's own runner wrote is a source like a batch's: the
    # US's first card pick of turns 1-3, three states from one game, each
    # a game the arena can start (the hidden seat resampled) and play on.
    from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
    from struggler.runner import play_game

    engine = Engine.new_game(seed=3, events=True, us_bid=2)
    play_game(engine, {Side.US: FirstLegalPlayer(), Side.USSR: RandomPlayer(seed=4)}, log_path=str(tmp_path / "g.json"))
    bank = harvest_logs(log_paths([tmp_path]), predicate="us_opening")
    assert bank.header["source"] == "logs" and bank.header["us_bid"] == 2 and bank.header["games"] == 1
    assert [(e["mover"], e["turn"], e["action_round"]) for e in bank.entries] == [("US", t, 1) for t in range(1, len(bank) + 1)]
    assert 1 <= len(bank) <= 3
    for index, entry in enumerate(bank.entries):
        state = Engine.deserialize(entry["state"])
        assert state.pending_decision.actor is Side.US and state.pending_decision.kind is DecisionKind.ACTION_ROUND_PLAY
        started = bank.start(index, seed=index)
        assert started.observe(Side.US) == state.observe(Side.US)
    # The deck is the log's: a spec check does not hold it against the arena's.
    bank.validate(us_bid=2, starting_vp=0, events=True, include_optional=not bank.header["include_optional"])
    with pytest.raises(ValueError, match="us_bid"):
        bank.validate(us_bid=0, starting_vp=0, events=True, include_optional=True)


def test_arena_seats_the_given_policy_at_the_scenario_mover(bank: ScenarioBank):
    def ussr_learner(slot, episode, rng):
        return {Side.US: "pool", Side.USSR: "learner"}

    arena = Arena(6, seed=3, seat_assigner=ussr_learner, scenario_bank=bank, scenario_frac=1.0, scenario_mover_id="learner")
    for slot in range(6):
        mover = arena.engine(slot).pending_decision.actor
        assert arena.seats(slot)[mover] == "learner" and arena.seats(slot)[mover.opponent] == "pool"
    plain = Arena(6, seed=3, seat_assigner=ussr_learner, scenario_bank=bank, scenario_frac=0.0, scenario_mover_id="learner")
    for slot in range(6):
        assert dict(plain.seats(slot)) == {Side.US: "pool", Side.USSR: "learner"}
    with pytest.raises(ValueError, match="exclusive"):
        Arena(2, seed=3, scenario_bank=bank, scenario_frac=1.0, scenario_mover_id="learner", scenario_seats=("learner", "pool"))
