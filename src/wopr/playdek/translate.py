"""Between Playdek's prompts/events and struggler's decisions/actions.

This module is pure: it knows what a Playdek option *means* in struggler
terms and which Playdek option a struggler action corresponds to, and it
turns Playdek's roll events into the CHANCE actions physical mode asks
for. Sequencing -- which engine is waiting on what -- is the bridge's
job (`lockstep.py`), because the two decision streams do not line up
one to one: one Playdek option ("Place Influence" in "Select Use For
Event Card") is two struggler decisions (`PLAY_MODE` ops, `OPS_TYPE`
influence), and a country option's struggler kind (`PLACE_INFLUENCE`,
`EVENT_INFLUENCE`, `COUP_TARGET`, ...) is whatever the engine is asking,
which the bridge knows and this module does not.

The vocabulary here was collected from live games (`wopr.playdek.smoke`
tallies prompts and hints); `OptionMeaning.UNKNOWN` is how a new one
shows up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from struggler.engine.types import Action, DecisionKind, Side
from wopr.playdek import ids
from wopr.playdek.ffi import EventType, SelectionHint
from wopr.playdek.game import GameEvent, Option, Prompt


class Meaning(Enum):
    CARD = auto()  # pick a card (headline, action round)
    USE = auto()  # how to use the played card
    COUNTRY = auto()  # a country target (place, remove, coup, realign, war)
    CHOICE = auto()  # an event's either/or
    STOP = auto()  # end an optional repetition early ("No More Realignment")
    CANCEL = auto()  # Playdek UI: back out; no struggler equivalent
    SWITCH_CARD = auto()  # Playdek UI: pick a different card instead; no struggler equivalent
    BLANK = auto()  # Playdek UI: the unlabelled entry beside an event's yes/no; selecting it skips the event -- never do
    UNKNOWN = auto()


@dataclass(frozen=True)
class Use:
    """The struggler steps implied by one "use" option."""

    mode: str  # PLAY_MODE: "ops" | "event" | "space_race"
    ops_type: str | None = None  # OPS_TYPE when mode is "ops"
    event_first: bool | None = None  # EVENT_OPS_ORDER when the card is the opponent's


_USES: dict[int, Use] = {
    SelectionHint.PLAY_EVENT: Use("event"),
    SelectionHint.OPS_INFLUENCE: Use("ops", "influence", False),
    SelectionHint.OPS_REALIGNMENT: Use("ops", "realignment", False),
    SelectionHint.OPS_COUP: Use("ops", "coup", False),
    SelectionHint.OPS_SPACE_RACE: Use("space_race"),
    SelectionHint.RESOLVE_EVENT_FIRST: Use("ops", None, True),
}

_COUNTRY_HINTS = {
    SelectionHint.INFLUENCE_COUNTRY,
    SelectionHint.SETUP_INFLUENCE_COUNTRY,
    SelectionHint.REMOVE_INFLUENCE_COUNTRY,
    SelectionHint.RELOCATE_FROM_COUNTRY,
    SelectionHint.WAR_COUNTRY,
}
_CARD_HINTS = {
    SelectionHint.HEADLINE_CARD, SelectionHint.PLAY_CARD, SelectionHint.PLAY_SCORING_CARD, SelectionHint.PLAY_OPPONENT_CARD,
    SelectionHint.DISCARD_CARD, SelectionHint.FORCED_DISCARD_CARD, SelectionHint.GIVE_CARD,
}
# An event's either/or whose Playdek label the engine's payload shares no
# word with, or the wrong one ("Add Influence Adjacent to South Africa"
# shares two words with `south_africa_only`): (prompt, label) -> the
# engine's `choice`. Anything not listed here is matched by words and
# reported ("choice by words"), so a new one shows up in the differ.
CHOICE_LABELS: dict[tuple[str, str], str] = {
    ("Choose for Eastern Europe:", "Remove 4 US Countries in Eastern Europe"): "remove",  # Warsaw Pact
    ("Choose for Eastern Europe:", "Add 5 USSR Influence in Eastern Europe"): "add",
    ("Participate in Olympic Games?", "Participate"): "participate",
    ("Participate in Olympic Games?", "Boycott"): "boycott",
    ("Choose for South Africa:", "Gain 2 Influence in South Africa"): "south_africa_only",  # South African Unrest
    ("Choose for South Africa:", "Add Influence Adjacent to South Africa"): "and_adjacent",
}
# `OUTPUT_ANIMATION_CARD.animation_event_hint` when a card leaves a hand for
# the resolve slot: `0x8000 | (use << 8) | 1` for the use the player chose
# (the second byte 2 marks the automatic other half of an "event after
# Ops" play). 0x81 is a headline's reveal. This is how an AI seat's use is
# learned -- the AI's choices are never prompted, only reported.
ANIMATION_USES: dict[int, Use] = {
    0x82: Use("event"),
    0x83: Use("ops", None, True),
    0x84: Use("ops", "influence", False),
    0x85: Use("ops", "realignment", False),
    0x86: Use("ops", "coup", False),
    0x87: Use("space_race"),
}
ANIMATION_HEADLINE = 0x81
ANIMATION_CHOSEN = 0x01  # low byte: the use the player chose (0x02: the automatic second half)

_COUNTRY_BY_NAME = {name: i + 1 for i, name in enumerate(ids.PLAYDEK_COUNTRIES)}
_LABEL_COUNTRY = re.compile(r" (?:in|from) (.+)$")


@dataclass(frozen=True)
class OptionMeaning:
    meaning: Meaning
    card: str | None = None  # struggler card id
    country: str | None = None  # struggler country id
    use: Use | None = None
    defcon: int | None = None  # a DEFCON level the option sets (the engine's choice is the level, or raise/lower/none relative to the current one)
    label: str = ""


def meaning(option: Option) -> OptionMeaning:
    hint = option.hint
    if hint == SelectionHint.CANCEL:
        return OptionMeaning(Meaning.CANCEL, label=option.text)
    if hint in (SelectionHint.STOP, SelectionHint.TRAP_PASS):
        return OptionMeaning(Meaning.STOP, label=option.text)
    if hint == SelectionHint.SWITCH_CARD:
        return OptionMeaning(Meaning.SWITCH_CARD, label=option.text)
    if hint in (SelectionHint.EVENT_CHOICE_BLANK, SelectionHint.FORCED_DISCARD_BLANK, SelectionHint.TRAP_SCORING_CARD):
        return OptionMeaning(Meaning.BLANK, label=option.text)
    if hint in _CARD_HINTS:
        return OptionMeaning(Meaning.CARD, card=ids.card_id(option.selection_id), label=option.text)
    if hint in _USES:
        return OptionMeaning(Meaning.USE, use=_USES[hint], label=option.text)
    if hint in _COUNTRY_HINTS:
        return OptionMeaning(Meaning.COUNTRY, country=ids.country_id(option.selection_id), label=option.text)
    if hint in (SelectionHint.EVENT_CHOICE, SelectionHint.EVENT_CHOICE_YES, SelectionHint.EVENT_CHOICE_NO):
        return OptionMeaning(Meaning.CHOICE, label=option.text)
    if SelectionHint.DEFCON_SET < hint <= SelectionHint.DEFCON_SET + 5:
        return OptionMeaning(Meaning.CHOICE, defcon=hint - SelectionHint.DEFCON_SET, label=option.text)
    # The Cuban Missile Crisis defusing entry of the action-round prompt
    # ("Remove 2 Influence from West Germany") and any unknown hint: a
    # country named in the label is a country target ("Coup in Poland",
    # "Attempt Realignment in Iran"), and a card id with the card's name in
    # the label a card ("Recover Summit", Star Wars).
    m = _LABEL_COUNTRY.search(option.text)
    if m and m.group(1) in _COUNTRY_BY_NAME:
        return OptionMeaning(Meaning.COUNTRY, country=ids.country_id(_COUNTRY_BY_NAME[m.group(1)]), label=option.text)
    number = option.selection_id - ids.CARD_SELECTION_OFFSET
    if number in ids.CARD_BY_NUMBER and _squash(ids._CARDS[ids.CARD_BY_NUMBER[number]].name) in _squash(option.text):
        return OptionMeaning(Meaning.CARD, card=ids.CARD_BY_NUMBER[number], label=option.text)
    return OptionMeaning(Meaning.UNKNOWN, label=option.text)


def _squash(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def actions_for_use(use: Use, *, opponents_card: bool) -> list[Action]:
    """The struggler actions, in the engine's asking order, that one "use"
    option stands for. `opponents_card` is whether the engine will ask
    `EVENT_OPS_ORDER` (the card's event belongs to the other side)."""
    if use.mode != "ops":
        return [Action(DecisionKind.PLAY_MODE, {"mode": use.mode})]
    out = [Action(DecisionKind.PLAY_MODE, {"mode": "ops"})]
    if opponents_card:
        out.append(Action(DecisionKind.EVENT_OPS_ORDER, {"order": "event_first" if use.event_first else "ops_first"}))
    if use.ops_type is not None:
        out.append(Action(DecisionKind.OPS_TYPE, {"type": use.ops_type}))
    return out


def find_card(prompt: Prompt, card: str) -> Option:
    for o in prompt.visible:
        m = meaning(o)
        if m.meaning is Meaning.CARD and m.card == card:
            return o
    raise LookupError(f"no option for card {card!r} in {prompt.text!r}: {[o.text for o in prompt.visible]}")


def find_country(prompt: Prompt, country: str) -> Option:
    for o in prompt.visible:
        m = meaning(o)
        if m.meaning is Meaning.COUNTRY and m.country == country:
            return o
    raise LookupError(f"no option for country {country!r} in {prompt.text!r}: {[o.text for o in prompt.visible]}")


def find_use(prompt: Prompt, *, mode: str, ops_type: str | None = None, event_first: bool | None = None) -> Option:
    """The "use" option for a struggler play: `mode` from PLAY_MODE, then
    `ops_type` from OPS_TYPE, `event_first` from EVENT_OPS_ORDER when the
    engine asked it. "Resolve Event First" is its own option; the Ops-first
    choice is simply the Ops use itself."""
    for o in prompt.visible:
        m = meaning(o)
        if m.meaning is not Meaning.USE:
            continue
        u = m.use
        if event_first and u.event_first:
            return o
        if u.mode == mode and (mode != "ops" or u.ops_type == ops_type) and not u.event_first:
            return o
    raise LookupError(f"no use option for mode={mode} ops_type={ops_type} event_first={event_first} in {[o.text for o in prompt.visible]}")


def use_from_animation(hint: int) -> Use | None:
    """The use a card-play animation hint reports, or None for anything
    else (a headline reveal, the automatic second half, a plain move)."""
    if hint & 0xFF != ANIMATION_CHOSEN:
        return None
    return ANIMATION_USES.get((hint >> 8) & 0xFF)


def find_stop(prompt: Prompt) -> Option:
    """The "stop here" entry of an optional repetition ("No More
    Realignment", "Do Not Discard", "Done Removing")."""
    for o in prompt.visible:
        if meaning(o).meaning is Meaning.STOP:
            return o
    raise LookupError(f"no stop option in {prompt.text!r}: {[o.text for o in prompt.visible]}")


def find_choice(prompt: Prompt, choice: str, *, defcon: int) -> Option:
    """The option for a struggler EVENT_CHOICE payload: a card or country id
    by its own lookup, a DEFCON level (or raise/lower/none from the current
    `defcon`) by the DEFCON hints, anything else by the label words."""
    if choice in ids.NUMBER_BY_CARD:
        return find_card(prompt, choice)
    if choice in ids.INDEX_BY_COUNTRY:
        return find_country(prompt, choice)
    levels = {o: meaning(o).defcon for o in prompt.visible}
    if any(level is not None for level in levels.values()):
        want = {"raise": defcon + 1, "lower": defcon - 1, "none": defcon}.get(choice)
        if want is None and choice.isdigit():
            want = int(choice)
        for o, level in levels.items():
            if level == want:
                return o
        raise LookupError(f"no DEFCON option for {choice!r} at DEFCON {defcon} in {[o.text for o in prompt.visible]}")
    for o in prompt.visible:
        if CHOICE_LABELS.get((prompt.text, o.text)) == choice:
            return o
    words = set(choice.lower().replace("_", " ").split())
    scored = [(len(words & set(o.text.lower().split())), o) for o in prompt.visible if meaning(o).meaning is not Meaning.BLANK]
    best = max(scored, key=lambda t: t[0], default=None)
    if best is None or best[0] == 0:
        raise LookupError(f"no option shares a word with {choice!r} in {prompt.text!r}: {[o.text for o in prompt.visible]}")
    return best[1]


def uses_offered(prompt: Prompt) -> set[Use]:
    return {meaning(o).use for o in prompt.visible if meaning(o).meaning is Meaning.USE}


def countries_offered(prompt: Prompt) -> set[str]:
    return {meaning(o).country for o in prompt.visible if meaning(o).meaning is Meaning.COUNTRY}


def cards_offered(prompt: Prompt) -> set[str]:
    return {meaning(o).card for o in prompt.visible if meaning(o).meaning is Meaning.CARD}


# -- CHANCE: Playdek rolls, struggler decisions ------------------------------


@dataclass(frozen=True)
class Roll:
    kind: DecisionKind
    payload: dict
    side: Side | None = None  # whose roll, when the kind has one per side
    country: str | None = None


def rolls_from_event(event: GameEvent, side_of: dict[int, Side]) -> list[Roll]:
    """The CHANCE answers one Playdek roll event supplies. `side_of` maps a
    Playdek player index/id as the event reports it to a side."""
    f = event.fields
    if event.kind == EventType.COUP_ROLL:
        return [Roll(DecisionKind.COUP_ROLL, {"value": f["roll"]}, country=ids.country_id(f["country_id"]))]
    if event.kind == EventType.WAR_ROLL:
        return [Roll(DecisionKind.WAR_ROLL, {"value": f["roll"]}, country=ids.country_id(f["country_id"]))]
    if event.kind == EventType.REALIGNMENT:
        actor = side_of[f["realign_player_index"]]
        by_side = {Side.USSR: f["USSR_roll_result"], Side.US: f["US_roll_result"]}
        return [
            Roll(DecisionKind.REALIGNMENT_ACTOR_ROLL, {"value": by_side[actor]}, side=actor, country=ids.country_id(f["country"])),
            Roll(DecisionKind.REALIGNMENT_OPPONENT_ROLL, {"value": by_side[actor.opponent]}, side=actor.opponent, country=ids.country_id(f["country"])),
        ]
    if event.kind == EventType.SPACE_RACE_ROLL:
        if f["space_race_required_roll"] == 0:
            return []  # a free advance (Captured Nazi Scientist), reported as a "roll" of 9: no die
        return [Roll(DecisionKind.SPACE_RACE_ROLL, {"value": f["roll"]})]
    if event.kind == EventType.TRAP_ROLL:
        return [Roll(DecisionKind.QUAGMIRE_ROLL, {"value": f["roll"]})]
    if event.kind == EventType.EFFECT_ROLL:
        # A two-sided contest (Olympic Games): one die per side, the modifiers
        # (`ussr_modify`/`usa_modify`) already known to the engine. Which side
        # is the sponsor is the bridge's to say (the engine's CONTEST_ROLL
        # context names it); the record does not.
        return [
            Roll(DecisionKind.CONTEST_ROLL, {"value": f["ussr_roll"]}, side=Side.USSR),
            Roll(DecisionKind.CONTEST_ROLL, {"value": f["usa_roll"]}, side=Side.US),
        ]
    return []
