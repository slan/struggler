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
# Either/or answers with a fixed struggler spelling (the rest are matched by label words).
_CHOICES = {SelectionHint.DEFCON_IMPROVE: "raise", SelectionHint.DEFCON_DEGRADE: "lower", SelectionHint.DEFCON_PASS: "none"}
_CARD_HINTS = {
    SelectionHint.HEADLINE_CARD, SelectionHint.PLAY_CARD, SelectionHint.PLAY_SCORING_CARD, SelectionHint.PLAY_OPPONENT_CARD,
    SelectionHint.DISCARD_CARD, SelectionHint.FORCED_DISCARD_CARD,
}
_COUNTRY_BY_NAME = {name: i + 1 for i, name in enumerate(ids.PLAYDEK_COUNTRIES)}
_LABEL_COUNTRY = re.compile(r" in (.+)$")


@dataclass(frozen=True)
class OptionMeaning:
    meaning: Meaning
    card: str | None = None  # struggler card id
    country: str | None = None  # struggler country id
    use: Use | None = None
    choice: str | None = None  # the engine's spelling of an either/or answer, when the hint fixes it
    label: str = ""


def meaning(option: Option) -> OptionMeaning:
    hint = option.hint
    if hint == SelectionHint.CANCEL:
        return OptionMeaning(Meaning.CANCEL, label=option.text)
    if hint == SelectionHint.STOP:
        return OptionMeaning(Meaning.STOP, label=option.text)
    if hint == SelectionHint.SWITCH_CARD:
        return OptionMeaning(Meaning.SWITCH_CARD, label=option.text)
    if hint == SelectionHint.EVENT_CHOICE_BLANK:
        return OptionMeaning(Meaning.BLANK, label=option.text)
    if hint in _CARD_HINTS:
        return OptionMeaning(Meaning.CARD, card=ids.card_id(option.selection_id), label=option.text)
    if hint in _USES:
        return OptionMeaning(Meaning.USE, use=_USES[hint], label=option.text)
    if hint in _COUNTRY_HINTS:
        return OptionMeaning(Meaning.COUNTRY, country=ids.country_id(option.selection_id), label=option.text)
    if hint in (SelectionHint.EVENT_CHOICE, SelectionHint.EVENT_CHOICE_YES, SelectionHint.EVENT_CHOICE_NO):
        return OptionMeaning(Meaning.CHOICE, label=option.text)
    if hint in _CHOICES:
        return OptionMeaning(Meaning.CHOICE, choice=_CHOICES[hint], label=option.text)
    # Unknown hint: a country named in the label is still a country target
    # ("Coup in Poland", "Attempt Realignment in Iran").
    m = _LABEL_COUNTRY.search(option.text)
    if m and m.group(1) in _COUNTRY_BY_NAME:
        return OptionMeaning(Meaning.COUNTRY, country=ids.country_id(_COUNTRY_BY_NAME[m.group(1)]), label=option.text)
    return OptionMeaning(Meaning.UNKNOWN, label=option.text)


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
