"""The two vocabularies: Playdek's card and country ids <-> struggler's.

Cards are easy: Playdek's `card_number` is the GMT card number, the same
`number` as `cards.json`, and an option's `selectionID` for a card is
`100 + number` (Containment is 125). Numbers above 110 are Playdek's own
(promo packs 121-128, Turn Zero 129-146, the AI's Ops proxies 201-204)
and have no struggler card.

Countries go by Playdek's `country_index` (1 = USSR, 2 = USA, 3 = Canada
... 87 = Chinese Civil War), which is both the `selectionID` of a country
option and the `id` of `COUNTRY_INFLUENCE` events. The table below is the
order of `twilight_map.lua`; `lua_countries()` re-reads it from the
install so a test can check the two never drift.
"""

from __future__ import annotations

import re
from pathlib import Path

from struggler.engine.board import Board
from struggler.engine.cards import load_cards
from wopr.playdek import ffi

CARD_SELECTION_OFFSET = 100

# Playdek's country names by `country_index`, 1-based.
PLAYDEK_COUNTRIES: tuple[str, ...] = (
    "USSR", "USA", "Canada", "UK", "Norway", "Sweden", "Finland", "Denmark", "Benelux", "France",
    "Spain/Portugal", "Italy", "Greece", "Austria", "West Germany", "East Germany", "Poland",
    "Czechoslovakia", "Hungary", "Yugoslavia", "Romania", "Bulgaria", "Turkey", "Libya", "Egypt",
    "Israel", "Lebanon", "Syria", "Iraq", "Iran", "Jordan", "Gulf States", "Saudi Arabia",
    "Afghanistan", "Pakistan", "India", "Burma", "Laos/Cambodia", "Thailand", "Vietnam", "Malaysia",
    "Australia", "Indonesia", "Philippines", "Japan", "Taiwan", "South Korea", "North Korea",
    "Algeria", "Morocco", "Tunisia", "West African States", "Ivory Coast", "Saharan States",
    "Nigeria", "Cameroon", "Zaire", "Angola", "South Africa", "Botswana", "Zimbabwe",
    "SE African States", "Kenya", "Somalia", "Ethiopia", "Sudan", "Mexico", "Guatemala",
    "El Salvador", "Honduras", "Costa Rica", "Panama", "Nicaragua", "Cuba", "Haiti",
    "Dominican Republic", "Colombia", "Ecuador", "Peru", "Chile", "Argentina", "Uruguay", "Paraguay",
    "Bolivia", "Brazil", "Venezuela", "Chinese Civil War",
)


def _struggler_country_id(playdek_name: str) -> str:
    if playdek_name == "USA":
        return "US"
    return playdek_name.replace("/", "_").replace(" ", "_")


_CARDS = load_cards()
CARD_BY_NUMBER: dict[int, str] = {card.number: cid for cid, card in _CARDS.items()}
NUMBER_BY_CARD: dict[str, int] = {cid: n for n, cid in CARD_BY_NUMBER.items()}
COUNTRY_BY_INDEX: dict[int, str] = {i + 1: _struggler_country_id(n) for i, n in enumerate(PLAYDEK_COUNTRIES)}
INDEX_BY_COUNTRY: dict[str, int] = {cid: i for i, cid in COUNTRY_BY_INDEX.items()}


def card_id(selection_id: int) -> str:
    """struggler card id for a card option's `selectionID` (or a card event's id)."""
    try:
        return CARD_BY_NUMBER[selection_id - CARD_SELECTION_OFFSET]
    except KeyError:
        raise KeyError(f"Playdek card {selection_id} has no struggler card") from None


def card_selection(card: str) -> int:
    return NUMBER_BY_CARD[card] + CARD_SELECTION_OFFSET


def country_id(index: int) -> str:
    """struggler country id (or 'US'/'USSR') for a Playdek country index."""
    return COUNTRY_BY_INDEX[index]


def country_index(country: str) -> int:
    return INDEX_BY_COUNTRY[country]


def lua_countries(root: Path | None = None) -> dict[int, str]:
    """`country_index -> country_name` as the install's `twilight_map.lua` declares them."""
    text = (ffi.lua_dir(root or ffi.find_install()) / "twilight_map.lua").read_text(encoding="utf-8", errors="replace")
    out = {}
    for name, body in re.findall(r'g_twilight_map\["([^"]+)"\]\s*=\s*\{(.*?)\n\}', text, re.S):
        m = re.search(r"country_index\s*=\s*(\d+)", body)
        out[int(m.group(1))] = name
    return out


def lua_cards(root: Path | None = None) -> dict[int, str]:
    """`card_number -> card name` from the install's `twilight_cards.lua`."""
    text = (ffi.lua_dir(root or ffi.find_install()) / "twilight_cards.lua").read_text(encoding="utf-8", errors="replace")
    out = {}
    for name, body in re.findall(r'g_twilight_cards\["([^"]+)"\]\s*=\s*\{(.*?)\n\}', text, re.S):
        m = re.search(r"card_number\s*=\s*(\d+)", body)
        out[int(m.group(1))] = name
    return out


def check_against_struggler() -> None:
    """Every struggler country and card has a Playdek id and vice versa (for
    the shared 1-110 card range)."""
    board_ids = set(Board(variants=Board.VARIANTS).countries) | {"US", "USSR"}  # Playdek's map lists every space
    mapped = set(COUNTRY_BY_INDEX.values())
    if board_ids != mapped:
        raise ValueError(f"country mismatch: only struggler {sorted(board_ids - mapped)}, only Playdek {sorted(mapped - board_ids)}")
    if set(CARD_BY_NUMBER) != set(range(1, 111)):
        raise ValueError("cards.json does not cover numbers 1-110 exactly")
