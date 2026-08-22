"""Tests for the WOPR layout (`struggler.bots.joshua.features`): the encoding
contract every Joshua checkpoint and every arena backend depends on."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import bare_engine
from struggler.bots.joshua import features as F
from struggler.bots.naive import RandomPlayer
from struggler.engine import DecisionKind, Engine, Side
from struggler.runner import play_game

ENGINE_DIR = Path(__file__).resolve().parents[1] / "src" / "struggler" / "engine"


def _encode(engine: Engine, side: Side) -> dict[str, np.ndarray]:
    return F.encode_single(engine.observe(side))


def test_layout_shapes_and_dtypes_match_allocate():
    buffers = F.allocate(3)
    for name, (shape, dtype) in F.LAYOUT.items():
        assert buffers[name].shape == (3, *shape)
        assert buffers[name].dtype == dtype
    assert F.GLOBAL_FEATURES[F.AM_US_INDEX] == "am_us"
    assert len(set(F.GLOBAL_FEATURES)) == F.G  # no duplicate names
    assert len(set(F.OPTION_FEATURES)) == F.F_OPTION


def test_setup_decision_encodes_mask_countries_and_kind():
    engine = Engine.new_game(seed=1)
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.PLACE_INFLUENCE and decision.actor is Side.USSR
    buffers = _encode(engine, Side.USSR)

    mask = buffers["opt_mask"][0]
    assert mask.sum() == len(decision.options)
    assert not mask[len(decision.options):].any()
    for k, action in enumerate(decision.options):
        assert buffers["opt_country"][0, k] == F.COUNTRY_INDEX[action.payload["country"]]
        assert buffers["opt_feats"][0, k, F.OPTION_INDEX["is_country"]] == 1.0
    assert buffers["opt_country"][0, len(decision.options)] == F.N_COUNTRIES  # padding sentinel

    g = buffers["globals"][0]
    assert g[F.GLOBAL_INDEX["kind_PLACE_INFLUENCE"]] == 1.0
    assert g[F.GLOBAL_INDEX["ctx_setup"]] == 1.0
    assert g[F.GLOBAL_INDEX["ctx_remaining"]] == pytest.approx(decision.context["remaining"] / 6.0)
    assert g[F.GLOBAL_INDEX["phase_setup"]] == 1.0
    assert g[F.AM_US_INDEX] == 0.0


def test_views_of_both_sides_are_mirror_images():
    engine = Engine.new_game(seed=3)
    engine.vp = 4  # positive favours US
    engine.board.influence["Poland"]["USSR"] = 3
    engine.board.influence["France"]["US"] = 3
    us = _encode(engine, Side.US)
    ussr = _encode(engine, Side.USSR)

    my, their = F.BOARD_FEATURES.index("my_influence"), F.BOARD_FEATURES.index("their_influence")
    np.testing.assert_array_equal(us["board"][0, :, my], ussr["board"][0, :, their])
    np.testing.assert_array_equal(us["board"][0, :, their], ussr["board"][0, :, my])
    poland = F.COUNTRY_INDEX["Poland"]
    assert ussr["board"][0, poland, F.BOARD_FEATURES.index("my_control")] == 1.0
    assert us["board"][0, poland, F.BOARD_FEATURES.index("their_control")] == 1.0
    assert us["globals"][0, F.GLOBAL_INDEX["vp"]] == pytest.approx(4 / 20)
    assert ussr["globals"][0, F.GLOBAL_INDEX["vp"]] == pytest.approx(-4 / 20)
    assert us["globals"][0, F.AM_US_INDEX] == 1.0 and ussr["globals"][0, F.AM_US_INDEX] == 0.0
    # The opponent's hand is a count, never card identities: only my own
    # cards can be in `my_hand`, and they are the hand `observe` reported.
    us_hand = {F.CARDS[i] for i in np.flatnonzero(us["card_loc"][0] == F.LOC_MY_HAND)}
    assert us_hand == set(engine.observe(Side.US).hand)


def test_sided_effects_are_encoded_relative_to_the_viewer():
    engine = bare_engine()
    engine.turn_effects["red_scare"] = "US"
    engine.game_effects["nato"] = True
    engine.phase = "action_rounds"
    engine._advance()
    us = _encode(engine, Side.US)["globals"][0]
    ussr = _encode(engine, Side.USSR)["globals"][0]
    assert us[F.GLOBAL_INDEX["turn_red_scare_me"]] == 1.0 and us[F.GLOBAL_INDEX["turn_red_scare_them"]] == 0.0
    assert ussr[F.GLOBAL_INDEX["turn_red_scare_me"]] == 0.0 and ussr[F.GLOBAL_INDEX["turn_red_scare_them"]] == 1.0
    assert us[F.GLOBAL_INDEX["game_nato"]] == 1.0 and ussr[F.GLOBAL_INDEX["game_nato"]] == 1.0


def test_unknown_effect_key_is_a_contract_break():
    engine = bare_engine()
    engine.game_effects["a_new_event_flag"] = True
    engine.phase = "action_rounds"
    engine._advance()
    with pytest.raises(ValueError, match="a_new_event_flag"):
        _encode(engine, engine.pending_decision.actor)


def test_every_engine_effect_key_is_in_the_layout():
    """Static guard: any `turn_effects[...]`/`game_effects[...]` key the engine
    source touches must be declared, or a rare event would crash Joshua mid-game."""
    pattern = re.compile(r"(turn_effects|game_effects)(?:\.get\(|\.pop\(|\[)\s*\"([a-z_0-9]+)\"")
    found: dict[str, set[str]] = {"turn_effects": set(), "game_effects": set()}
    for path in ENGINE_DIR.glob("*.py"):
        for store, key in pattern.findall(path.read_text(encoding="utf-8")):
            found[store].add(key)
    assert found["turn_effects"], "expected the engine to reference turn_effects keys"
    assert found["turn_effects"] <= set(F.TURN_EFFECTS)
    assert found["game_effects"] <= set(F.GAME_EFFECTS) | {k for k, p in F.RELOCATED.items() if p == "turn"}


def test_relocated_effect_keeps_its_slot():
    # We Will Bury You is a game effect in the engine but encodes in the turn
    # slot it was allocated: the layout, and LAYOUT_VERSION, are unchanged.
    engine = bare_engine()
    engine.game_effects["we_will_bury_you"] = True
    engine.phase = "action_rounds"
    engine._advance()
    row = _encode(engine, engine.pending_decision.actor)["globals"][0]
    assert row[F.GLOBAL_INDEX["turn_we_will_bury_you"]] == 1.0
    assert "game_we_will_bury_you" not in F.GLOBAL_INDEX


@pytest.mark.parametrize("seed", [11, 12])
def test_every_decision_of_a_random_game_encodes(seed: int):
    buffers = F.allocate(1)
    seen_kinds: set[DecisionKind] = set()
    other = F.OPTION_INDEX["other"]

    class _Encoding(RandomPlayer):
        def choose_action(self, observation, history):
            F.encode_into(observation, buffers, 0)
            decision = observation.pending_decision
            assert buffers["opt_mask"][0].sum() == len(decision.options) <= F.K_MAX
            # The one sanctioned out-of-vocabulary value: realignment's "stop"
            # (`{"country": "stop"}`), which rides the `other` flag so the
            # layout did not have to change for it.
            others = [i for i in range(len(decision.options)) if buffers["opt_feats"][0, i, other]]
            assert all(
                decision.kind is DecisionKind.REALIGNMENT_TARGET and decision.options[i].payload.get("country") == "stop"
                for i in others
            ), "payload value outside OPTION_VOCAB"
            seen_kinds.add(decision.kind)
            return super().choose_action(observation, history)

    engine = Engine.new_game(seed=seed)
    play_game(engine, {Side.US: _Encoding(seed=seed), Side.USSR: _Encoding(seed=seed + 1)})
    assert engine.is_terminal
    assert {DecisionKind.ACTION_ROUND_PLAY, DecisionKind.PLAY_MODE, DecisionKind.PLACE_INFLUENCE} <= seen_kinds
