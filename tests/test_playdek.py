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

from struggler.engine.types import DecisionKind, Side
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


def test_hotseat_prompts_both_seats_in_order():
    pd = _playdek()
    game = pd.new_game(local_side=Side.USSR, ai_difficulty=None, seed=3)
    try:
        asked = []
        while len(asked) < 14:
            prompt = game.pump(idle_limit=60)
            assert prompt is not None
            entry = (game.sides[prompt.player_id], prompt.text)
            if not asked or asked[-1] != entry:  # the DLL re-asks the very first prompt once (hotseat start)
                asked.append(entry)
            game.choose(prompt.visible[0].index)
        ussr = [(Side.USSR, "Place 6 Influence")] + [(Side.USSR, f"Place {n} More Influence") for n in range(5, 0, -1)]
        us = [(Side.US, "Place 7 Influence")] + [(Side.US, f"Place {n} More Influence") for n in range(6, 0, -1)]
        assert asked == ussr + us + [(Side.US, "Select a Card to Headline")]
        assert game.hand_count(0) == 8 and game.hand_count(1) == 8
    finally:
        game.close()


def test_ids_cover_both_vocabularies():
    from wopr.playdek import ids

    ids.check_against_struggler()
    assert ids.card_id(125) == "Containment" and ids.card_selection("Duck_and_Cover") == 104
    assert ids.country_id(17) == "Poland" and ids.country_index("US") == 2 and ids.country_id(1) == "USSR"
    assert ids.country_id(87) == "Chinese_Civil_War" and ids.country_id(11) == "Spain_Portugal"
    with pytest.raises(KeyError):
        ids.card_id(250)  # nothing above 110 is a struggler card


def test_ids_agree_with_the_installed_lua_database():
    from wopr.playdek import ids

    try:
        root = ffi.find_install()
    except FileNotFoundError:
        pytest.skip("Playdek's Twilight Struggle is not installed")
    assert ids.lua_countries(root) == {i + 1: n for i, n in enumerate(ids.PLAYDEK_COUNTRIES)}
    lua = ids.lua_cards(root)
    from struggler.engine.cards import load_cards

    ours = {c.number: c.name for c in load_cards().values()}

    def squash(name: str) -> str:  # "U-2 Incident" == "U2 Incident", "SALT" == "Salt"
        return "".join(ch for ch in name.casefold() if ch.isalnum())

    differing = {n: (ours[n], lua[n]) for n in ours if squash(ours[n]) != squash(lua[n])}
    assert not differing, differing


def _prompt(text, rows, player=0):
    from wopr.playdek.game import Option, Prompt

    return Prompt(player, text, tuple(Option(i, sid, hint, False, label) for i, (sid, hint, label) in enumerate(rows)))


def test_translate_card_use_options():
    from wopr.playdek import translate as T
    from wopr.playdek.ffi import SelectionHint as H

    # "Select Use For Event Card" on an opponent's card, as the DLL lists it.
    prompt = _prompt("Select Use For Event Card", [
        (0, H.CANCEL, "Cancel"), (125, H.SWITCH_CARD, "Play Containment"), (0, H.PLAY_EVENT, "Play Event"),
        (0, H.RESOLVE_EVENT_FIRST, "Resolve Event First"), (0, H.OPS_INFLUENCE, "Place Influence"),
        (0, H.OPS_REALIGNMENT, "Realignment Rolls"), (0, H.OPS_COUP, "Coup Attempt"), (0, H.OPS_SPACE_RACE, "Space Race"),
    ])
    assert T.meaning(prompt.options[0]).meaning is T.Meaning.CANCEL
    assert T.meaning(prompt.options[1]).meaning is T.Meaning.SWITCH_CARD
    assert T.find_use(prompt, mode="event").text == "Play Event"
    assert T.find_use(prompt, mode="space_race").text == "Space Race"
    assert T.find_use(prompt, mode="ops", ops_type="coup").text == "Coup Attempt"
    assert T.find_use(prompt, mode="ops", event_first=True).text == "Resolve Event First"
    coup = T.meaning(T.find_use(prompt, mode="ops", ops_type="coup")).use
    assert [a.payload for a in T.actions_for_use(coup, opponents_card=True)] == [{"mode": "ops"}, {"order": "ops_first"}, {"type": "coup"}]
    assert [a.payload for a in T.actions_for_use(coup, opponents_card=False)] == [{"mode": "ops"}, {"type": "coup"}]
    first = T.meaning(T.find_use(prompt, mode="ops", event_first=True)).use
    assert [a.payload for a in T.actions_for_use(first, opponents_card=True)] == [{"mode": "ops"}, {"order": "event_first"}]
    with pytest.raises(LookupError):
        T.find_use(prompt, mode="un_intervention")


def test_translate_yes_no_choice_and_its_blank_entry():
    import random

    from wopr.playdek import translate as T
    from wopr.playdek.ffi import SelectionHint as H
    from wopr.playdek.lockstep import random_policy

    # As the DLL lists it: the two answers, then an unlabelled entry carrying
    # the card's id that is *not* flagged hidden. Selecting that one skips the
    # event, so it is UI-only and the random policy must never pick it.
    prompt = _prompt("Participate in Olympic Games?", [(0, H.EVENT_CHOICE_YES, "Participate"), (0, H.EVENT_CHOICE_NO, "Boycott"), (120, H.EVENT_CHOICE_BLANK, "")])
    assert [T.meaning(o).meaning for o in prompt.options] == [T.Meaning.CHOICE, T.Meaning.CHOICE, T.Meaning.BLANK]
    picks = {random_policy(random.Random(seed))(prompt).text for seed in range(50)}
    assert picks == {"Participate", "Boycott"}


def test_translate_summit_and_contest_rolls():
    from wopr.playdek import translate as T
    from wopr.playdek.ffi import EventType, SelectionHint as H
    from wopr.playdek.game import GameEvent

    # Summit at DEFCON 2, then How I Learned to Stop Worrying: the hint's low nibble is the level set.
    prompt = _prompt("You May Adjust DEFCON Level", [(145, H.DEFCON_SET + 3, "Improve DEFCON Level"), (145, H.DEFCON_SET + 1, "Degrade DEFCON Level"), (145, H.DEFCON_SET + 2, "Pass")])
    assert [T.meaning(o).defcon for o in prompt.options] == [3, 1, 2]
    prompt = _prompt("Choose DEFCON Level", [(146, H.DEFCON_SET + n, f"DEFCON {n}") for n in (5, 4, 3, 2, 1)])
    assert [(T.meaning(o).meaning, T.meaning(o).defcon) for o in prompt.options] == [(T.Meaning.CHOICE, n) for n in (5, 4, 3, 2, 1)]
    # Olympic Games: one die per side; the sponsor is the bridge's to say.
    rolls = T.rolls_from_event(GameEvent(EventType.EFFECT_ROLL, (120, 2, 0, 4, 2)), {})
    assert [(r.kind, r.side, r.payload) for r in rolls] == [(DecisionKind.CONTEST_ROLL, Side.USSR, {"value": 2}), (DecisionKind.CONTEST_ROLL, Side.US, {"value": 4})]


def test_translate_cards_countries_and_fallback_labels():
    from wopr.playdek import translate as T
    from wopr.playdek.ffi import SelectionHint as H

    headline = _prompt("Select a Card to Headline", [(125, H.HEADLINE_CARD, "Headline Containment"), (111, H.HEADLINE_CARD, "Headline Korean War")])
    assert T.cards_offered(headline) == {"Containment", "Korean_War"}
    assert T.find_card(headline, "Korean_War").index == 1
    place = _prompt("Place 3 More Influence", [(7, H.SETUP_INFLUENCE_COUNTRY, "Place Influence in Finland"), (17, H.SETUP_INFLUENCE_COUNTRY, "Place Influence in Poland")])
    assert T.countries_offered(place) == {"Finland", "Poland"}
    assert T.find_country(place, "Poland").selection_id == 17
    war = _prompt("Select War Country", [(36, H.WAR_COUNTRY, "War in India"), (35, H.WAR_COUNTRY, "War in Pakistan")])
    assert T.countries_offered(war) == {"India", "Pakistan"}
    # A hint we have not catalogued, but a country in the label: still a country target.
    coup = _prompt("Select Coup Country", [(30, 0xA0C0, "Coup Attempt in Iran")])
    assert T.meaning(coup.options[0]) == T.OptionMeaning(T.Meaning.COUNTRY, country="Iran", label="Coup Attempt in Iran")
    assert T.meaning(_prompt("?", [(0, 0xA0C0, "Something new")]).options[0]).meaning is T.Meaning.UNKNOWN


def test_translate_roll_events_to_chance_answers():
    from wopr.playdek import translate as T
    from wopr.playdek.game import GameEvent

    sides = {0: Side.USSR, 456: Side.US}
    coup = T.rolls_from_event(GameEvent(int(EventType.COUP_ROLL), (0, 30, 4)), sides)
    assert coup == [T.Roll(DecisionKind.COUP_ROLL, {"value": 4}, country="Iran")]
    realign = T.rolls_from_event(GameEvent(int(EventType.REALIGNMENT), (456, 17, 2, 5)), sides)
    assert [(r.kind, r.side, r.payload["value"]) for r in realign] == [
        (DecisionKind.REALIGNMENT_ACTOR_ROLL, Side.US, 5), (DecisionKind.REALIGNMENT_OPPONENT_ROLL, Side.USSR, 2)]
    assert T.rolls_from_event(GameEvent(int(EventType.VP_TRACK), (3,)), sides) == []
