"""The Playdek binding (`wopr.playdek`).

The layout checks always run; the live checks need the Steam edition of
Twilight Struggle installed (`wopr.playdek.ffi.find_install`) and are
skipped without it. They stop before the AI's first decision, so they do
not depend on its thinking time.
"""

from __future__ import annotations

import ctypes
import struct

import pytest

from struggler.engine.types import Side
from wopr.playdek import ffi
from wopr.playdek.game import EventDecoder, EventType, GameEvent


def test_native_layouts_match_the_app():
    # Sizes the app's IL2CPP marshalling wrapper uses (0x34 per seat) and the
    # 64-byte option records the listener receives.
    assert ctypes.sizeof(ffi.AppPlayerData) == 52
    assert ffi.AppPlayerData.name.offset == 20
    assert ctypes.sizeof(ffi.GameOption) == 64
    assert ffi.GameOption.optionText.offset == 9
    assert ctypes.sizeof(ffi.GamePlayerAIState) == 12


def _pack(records):
    return b"".join(struct.pack(f"<{1 + len(p)}i", int(k), *p) for k, p in records) + bytes(64)


def test_decoder_walks_variable_size_records():
    records = [
        (EventType.ASSIGN_SIDES, [1]),
        (EventType.COUNTRY_INFLUENCE, [17, 3, 0]),
        (EventType.RESHUFFLE, [1]),  # a payload value that looks like a type tag must not matter
        (EventType.VP_TRACK, [-2]),
    ]
    events = EventDecoder().decode(_pack(records), len(records))
    assert events == [GameEvent(int(k), tuple(p)) for k, p in records]
    assert events[1].fields == {"id": 17, "ussr_influence": 3, "us_influence": 0}
    assert str(events[3]) == "VP_TRACK {'vp_track': -2}"


def test_decoder_refuses_an_event_it_cannot_size():
    # The app's HandleEvent has no case for 22-27 and the DLL has never been
    # seen to emit them; a record has no length field, so guessing is worse
    # than stopping.
    raw = _pack([(EventType.ASSIGN_SIDES, [1]), (EventType.CARD_PLAYED, [103, 1]), (EventType.VP_TRACK, [0])])
    with pytest.raises(ValueError, match="event type 25"):
        EventDecoder().decode(raw, 3)


def _playdek():
    try:
        ffi.find_install()
    except FileNotFoundError:
        pytest.skip("Playdek's Twilight Struggle is not installed")
    from wopr.playdek.game import Playdek

    return Playdek._instance or Playdek()


def test_local_ussr_setup_prompts_come_from_the_dll():
    pd = _playdek()
    game = pd.new_game(local_side=Side.USSR, ai_difficulty=ffi.AIDifficulty.EASY, seed=7)
    try:
        prompt = game.pump(idle_limit=60)
        assert prompt is not None
        assert prompt.text == "Place 6 Influence"
        assert [o.text for o in prompt.options] == [
            f"Place Influence in {c}"
            for c in ("Finland", "Austria", "East Germany", "Poland", "Czechoslovakia", "Hungary", "Yugoslavia", "Romania", "Bulgaria")
        ]
        assert {o.hint for o in prompt.options} == {41010}
        poland = next(o for o in prompt.options if o.text.endswith("Poland"))
        game.choose(poland.index)
        prompt = game.pump(idle_limit=60)
        assert prompt is not None and prompt.text == "Place 5 More Influence"
        influence = [e for e in game.events if e.kind == EventType.COUNTRY_INFLUENCE and e.fields["id"] == poland.selection_id]
        assert influence[-1].fields == {"id": poland.selection_id, "ussr_influence": 1, "us_influence": 0}
        assert game.hand_count(0) == 8
        assert game.ai_state().isAIPlayer == 1
    finally:
        game.close()
