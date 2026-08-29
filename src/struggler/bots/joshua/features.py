"""The WOPR layout: how an `Observation` and its legal options become fixed-shape
arrays. This module is the single source of truth for that encoding -- the
in-game `JoshuaPlayer`, the training arena, and any future engine backend
(shared memory, another language) must all produce exactly these arrays.
See docs/WOPR.md for the field tables.

Design rules:

- **Me/them, never US/USSR.** Every sided quantity is expressed from the
  acting side's point of view (`my_influence`, `vp` positive when good for
  me, ...) plus one `am_us` flag, so a single network plays both seats.
- **An action is an index into `Decision.options`** (mandate #2), never a
  global action id. Each option is described by a small feature row plus a
  country index and a card index, so the policy can score options of any
  decision kind with one head.
- **Hidden information stays hidden** (mandate #4): everything here derives
  from `Observation` alone; the opponent's hand and the draw pile only ever
  appear as counts and as the `unseen` card location.
- numpy only: torch is an inference/training dependency, not an encoding one.

`LAYOUT_VERSION` must be bumped whenever any shape, index, or vocabulary
below changes; checkpoints record it and refuse to load against another.
"""

from __future__ import annotations

import numpy as np

from struggler.engine import DecisionKind, Observation, Region, Side
from struggler.engine.board import Board
from struggler.engine.cards import load_cards
from struggler.engine.types import Period

LAYOUT_VERSION = 1

# -- static game data ---------------------------------------------------------

# Every space the data knows, optional-rule ones included: the layout is a
# fixed vocabulary, so a variant must not change it. In a standard game a
# variant-only space (the Chinese Civil War) never appears in an
# `Observation`, and its row simply stays zero.
_BOARD = Board(variants=Board.VARIANTS)
COUNTRIES: tuple[str, ...] = tuple(_BOARD.countries)
COUNTRY_INDEX: dict[str, int] = {cid: i for i, cid in enumerate(COUNTRIES)}
N_COUNTRIES = len(COUNTRIES)
_NO_INFLUENCE: dict[str, int] = {}

_CARDS = load_cards()
CARDS: tuple[str, ...] = tuple(_CARDS)
CARD_INDEX: dict[str, int] = {cid: i for i, cid in enumerate(CARDS)}
N_CARDS = len(CARDS)

REGIONS: tuple[Region, ...] = tuple(Region)
REGION_INDEX: dict[str, int] = {region.name: i for i, region in enumerate(REGIONS)}
N_REGIONS = len(REGIONS)

DECISION_KINDS: tuple[DecisionKind, ...] = tuple(DecisionKind)
DECISION_KIND_INDEX: dict[DecisionKind, int] = {kind: i for i, kind in enumerate(DECISION_KINDS)}

# Upper bound on `len(Decision.options)` for any decision a bot can be asked.
# The largest legal set is "every country" (85, e.g. an event's influence
# placement); mandate #2 keeps every kind in the tens. Exceeding it is a
# contract violation, not a case to pad around -- `encode_into` raises.
K_MAX = 96

_STABILITY = np.array([_BOARD.countries[c].stability for c in COUNTRIES], dtype=np.float32)
_BATTLEGROUND = np.array([_BOARD.countries[c].battleground for c in COUNTRIES], dtype=np.float32)
_REGION_ONE_HOT = np.zeros((N_COUNTRIES, N_REGIONS), dtype=np.float32)
for _i, _cid in enumerate(COUNTRIES):
    _REGION_ONE_HOT[_i, REGION_INDEX[_BOARD.countries[_cid].region.name]] = 1.0
_ADJ_US = np.array([_BOARD.is_adjacent("US", c) for c in COUNTRIES], dtype=np.float32)
_ADJ_USSR = np.array([_BOARD.is_adjacent("USSR", c) for c in COUNTRIES], dtype=np.float32)

#: Symmetric country adjacency (superpowers excluded), for the model's graph layers.
ADJACENCY = np.zeros((N_COUNTRIES, N_COUNTRIES), dtype=np.float32)
for _i, _cid in enumerate(COUNTRIES):
    for _n in _BOARD.neighbors(_cid):
        if _n in COUNTRY_INDEX:
            ADJACENCY[_i, COUNTRY_INDEX[_n]] = 1.0

# Turn on which each period's cards are shuffled in (see `cards_entering`).
_CARD_ENTRY_TURN = np.array(
    [{Period.EARLY_WAR: 1, Period.MID_WAR: 4, Period.LATE_WAR: 8}[_CARDS[c].period] for c in CARDS],
    dtype=np.int64,
)
_CHINA_INDEX = next(i for i, c in enumerate(CARDS) if not _CARDS[c].in_deck)

# -- board (per-country) features ----------------------------------------------

BOARD_FEATURES: tuple[str, ...] = (
    "my_influence",  # / 5
    "their_influence",  # / 5
    "net_influence",  # (mine - theirs) / 5
    "my_control",
    "their_control",
    "stability",  # / 5
    "battleground",
    "adjacent_to_my_superpower",
    "adjacent_to_their_superpower",
    *(f"region_{r.name}" for r in REGIONS),
)
F_BOARD = len(BOARD_FEATURES)

# -- card locations ------------------------------------------------------------

CARD_LOCATIONS: tuple[str, ...] = (
    "unseen",  # draw pile or opponent's hand: indistinguishable to me, by design
    "my_hand",
    "discard",
    "removed",
    "china_mine_face_up",
    "china_mine_face_down",
    "china_theirs_face_up",
    "china_theirs_face_down",
    "not_yet_in_play",  # a later period's card, before its turn comes
)
LOC_UNSEEN, LOC_MY_HAND, LOC_DISCARD, LOC_REMOVED = 0, 1, 2, 3
LOC_CHINA_MINE_UP, LOC_CHINA_MINE_DOWN, LOC_CHINA_THEIRS_UP, LOC_CHINA_THEIRS_DOWN = 4, 5, 6, 7
LOC_FUTURE = 8
N_CARD_LOCATIONS = len(CARD_LOCATIONS)

# -- global features -----------------------------------------------------------

# Every effect flag the engine can set, with how its value is encoded. A key
# the engine starts setting that is missing here is a contract break, so
# `encode_into` raises on it (and tests/test_joshua_features.py greps the
# engine source to catch the drift before a game does).
#   "flag"   -> one feature, 1.0 when set
#   "side"   -> two features: set for me / set for them
#   "region" -> N_REGIONS one-hot of the region named by the value
TURN_EFFECTS: dict[str, str] = {
    "red_scare": "side",
    "vietnam_revolts": "flag",
    "containment": "flag",
    "cuban_missile_crisis": "side",
    "nuclear_subs": "flag",
    "salt": "flag",
    "we_will_bury_you": "flag",
    "iran_contra": "flag",
    "brezhnev": "flag",
    "la_death_squads": "side",
    "north_sea_oil_extra": "flag",
    "chernobyl": "region",
}
# Keys the engine has since moved to the other store but whose feature keeps
# the slot it was allocated, so LAYOUT_VERSION need not move: We Will Bury
# You became a game effect when its payout moved to the US's next action
# round (it still lasts a round or two). Maps the key to the layout prefix
# that holds it; the encoder routes the value there.
RELOCATED: dict[str, str] = {"we_will_bury_you": "turn"}
# Keys the engine keeps that the layout does not encode: legality flags
# whose only consequence the bot already sees in the options it is offered
# (Tear Down This Wall prevents Willy Brandt's event: the card is then never
# offered "for its event"). A slot for one would bump LAYOUT_VERSION.
UNENCODED_GAME_EFFECTS: frozenset[str] = frozenset({"tear_down_this_wall"})
GAME_EFFECTS: dict[str, str] = {
    "formosan_resolution": "flag",
    "degaulle_france": "flag",
    "marshall_or_warsaw": "flag",
    "norad": "flag",
    "flower_power": "flag",
    "quagmire": "flag",
    "bear_trap": "flag",
    "us_japan_pact": "flag",
    "camp_david": "flag",
    "missile_envy_forced": "side",
    "willy_brandt": "flag",
    "awacs": "flag",
    "shuttle_diplomacy": "flag",
    "iranian_hostage": "flag",
    "reformer": "flag",
    "evil_empire": "flag",
    "iron_lady": "flag",
    "yuri_samantha": "flag",
    "north_sea_oil": "flag",
    "john_paul": "flag",
    "nato": "flag",
    "space_race_double_attempt_holder": "side",
    "space_race_headline_reveal_holder": "side",
    "space_race_discard_holder": "side",
    "space_race_extra_round_holder": "side",
}

PHASES: tuple[str, ...] = ("setup", "headline", "action_rounds")


def _effect_feature_names(prefix: str, spec: dict[str, str]) -> list[str]:
    names: list[str] = []
    for key, kind in spec.items():
        if kind == "flag":
            names.append(f"{prefix}_{key}")
        elif kind == "side":
            names.extend((f"{prefix}_{key}_me", f"{prefix}_{key}_them"))
        elif kind == "region":
            names.extend(f"{prefix}_{key}_{r.name}" for r in REGIONS)
        else:
            raise ValueError(f"unknown effect encoding {kind!r} for {key!r}")
    return names


CONTEXT_FEATURES: tuple[str, ...] = (
    "ctx_ops",  # ops / ops_remaining / card_ops / che_ops, / 4
    "ctx_remaining",  # points still to place (setup or event), / 6
    "ctx_bonus",  # an in-region Ops bonus (China Card / Vietnam Revolts) applies
    "ctx_setup",
    "ctx_side_me",  # context "side" is me
    "ctx_choose_side_me",
    "ctx_inf_side_me",  # the influence being placed/removed is mine
    "ctx_op_remove",
    "ctx_whole",
    "ctx_requires_uncontrolled",
)

GLOBAL_FEATURES: tuple[str, ...] = (
    "am_us",
    "defcon",  # / 5
    "vp",  # positive when good for me, / 20
    "turn",  # / 10
    "action_round",  # / 8
    *(f"phase_{p}" for p in PHASES),
    "my_hand_size",  # / 9
    "their_hand_size",  # / 9
    "draw_pile_size",  # / N_CARDS
    "my_space_race",  # / 8
    "their_space_race",  # / 8
    "my_space_attempts",  # / 2
    "their_space_attempts",  # / 2
    "my_military_ops",  # / 5
    "their_military_ops",  # / 5
    "china_mine",
    "china_available",
    *_effect_feature_names("turn", TURN_EFFECTS),
    *_effect_feature_names("game", GAME_EFFECTS),
    *(f"kind_{k.name}" for k in DECISION_KINDS),
    *CONTEXT_FEATURES,
)
GLOBAL_INDEX: dict[str, int] = {name: i for i, name in enumerate(GLOBAL_FEATURES)}
G = len(GLOBAL_FEATURES)
AM_US_INDEX = GLOBAL_INDEX["am_us"]

#: `focus` slots: card indices (N_CARDS = none) the decision is about.
FOCUS_SLOTS: tuple[str, ...] = (
    "card_or_event",  # context "card", else context "event" (the card whose event is resolving)
    "opponent_headline",  # Space Race box 4: the opponent's already-revealed headline
)
N_FOCUS = len(FOCUS_SLOTS)

# -- option features -----------------------------------------------------------

# Closed vocabulary of non-country, non-card payload values. A value outside
# it lands in "other" (plus its position), so an engine addition degrades to
# "pick by position" instead of crashing -- but it should be added here.
# Riding "other" until the next layout bump: realignment's "country: stop"
# and the granted-Operations decline "type: pass" (rules version 6).
OPTION_VOCAB: tuple[str, ...] = (
    "mode:ops",
    "mode:event",
    "mode:space_race",
    "mode:un_intervention",
    "type:influence",
    "type:coup",
    "type:realignment",
    "order:event_first",
    "order:ops_first",
    "card:none",  # HELD_CARD_DISCARD / QUAGMIRE_DISCARD: decline to discard
    "choice:add",
    "choice:and_adjacent",
    "choice:boycott",
    "choice:coup",
    "choice:decline",
    "choice:done",
    "choice:end_game",
    "choice:event",
    "choice:keep",
    "choice:lower",
    "choice:no",
    "choice:none",
    "choice:ops",
    "choice:participate",
    "choice:raise",
    "choice:realign",
    "choice:refuse",
    "choice:remove",
    "choice:return",
    "choice:skip",
    "choice:south_africa_only",
    "choice:stop",
    "choice:take",
    "choice:yes",
    *(f"choice:{r.name}" for r in REGIONS),
)
OPTION_VOCAB_INDEX: dict[str, int] = {v: i for i, v in enumerate(OPTION_VOCAB)}
OPTION_FEATURES: tuple[str, ...] = (
    *OPTION_VOCAB,
    "number",  # a numeric choice ("1".."5"), / 5
    "is_country",
    "is_card",
    "is_empty",  # payload-less option (EVENT_RESUME)
    "other",  # value outside the vocabulary
    "position",  # index within options, / K_MAX
)
OPTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(OPTION_FEATURES)}
F_OPTION = len(OPTION_FEATURES)

# -- the layout ----------------------------------------------------------------

#: name -> (shape, dtype). Allocate a batch with `allocate`.
LAYOUT: dict[str, tuple[tuple[int, ...], type]] = {
    "board": ((N_COUNTRIES, F_BOARD), np.float32),
    "card_loc": ((N_CARDS,), np.int64),
    "globals": ((G,), np.float32),
    "focus": ((N_FOCUS,), np.int64),
    "opt_feats": ((K_MAX, F_OPTION), np.float32),
    "opt_country": ((K_MAX,), np.int64),  # N_COUNTRIES = none
    "opt_card": ((K_MAX,), np.int64),  # N_CARDS = none
    "opt_mask": ((K_MAX,), np.int8),
}


def allocate(batch: int) -> dict[str, np.ndarray]:
    """Zeroed batch-first buffers for `batch` rows, in the canonical layout."""
    return {name: np.zeros((batch, *shape), dtype=dtype) for name, (shape, dtype) in LAYOUT.items()}


def _sided(value: object, me: Side) -> tuple[float, float]:
    return (1.0 if value == me.value else 0.0, 1.0 if value == me.opponent.value else 0.0)


def _write_effects(row: np.ndarray, prefix: str, spec: dict[str, str], effects, me: Side) -> None:
    for key, value in effects.items():
        kind = spec.get(key)
        if kind is None:
            raise ValueError(
                f"{prefix}_effects key {key!r} is not in the WOPR layout; add it to "
                f"features.{prefix.upper()}_EFFECTS and bump LAYOUT_VERSION"
            )
        if not value:
            continue
        if kind == "flag":
            row[GLOBAL_INDEX[f"{prefix}_{key}"]] = 1.0
        elif kind == "side":
            mine, theirs = _sided(value, me)
            row[GLOBAL_INDEX[f"{prefix}_{key}_me"]] = mine
            row[GLOBAL_INDEX[f"{prefix}_{key}_them"]] = theirs
        else:  # region
            row[GLOBAL_INDEX[f"{prefix}_{key}_{value}"]] = 1.0


def encode_into(observation: Observation, buffers: dict[str, np.ndarray], i: int) -> None:
    """Write `observation` (which must carry a pending decision) into row `i`
    of `buffers`. Overwrites the row completely; `buffers` may be reused."""
    decision = observation.pending_decision
    if decision is None:
        raise ValueError("cannot encode an observation with no pending decision")
    options = decision.options
    if len(options) > K_MAX:
        raise ValueError(
            f"{decision.kind.name} offers {len(options)} options, above K_MAX={K_MAX}: "
            "mandate #2 says this decision is decomposed wrong"
        )
    me = observation.side
    them = me.opponent
    my, their = me.value, them.value

    # -- board
    influence = observation.influence
    mine = np.fromiter((influence.get(c, _NO_INFLUENCE).get(my, 0) for c in COUNTRIES), dtype=np.float32, count=N_COUNTRIES)
    theirs = np.fromiter((influence.get(c, _NO_INFLUENCE).get(their, 0) for c in COUNTRIES), dtype=np.float32, count=N_COUNTRIES)
    board = buffers["board"][i]
    board[:, 0] = mine / 5.0
    board[:, 1] = theirs / 5.0
    board[:, 2] = (mine - theirs) / 5.0
    board[:, 3] = (mine - theirs) >= _STABILITY
    board[:, 4] = (theirs - mine) >= _STABILITY
    board[:, 5] = _STABILITY / 5.0
    board[:, 6] = _BATTLEGROUND
    board[:, 7] = _ADJ_US if me is Side.US else _ADJ_USSR
    board[:, 8] = _ADJ_USSR if me is Side.US else _ADJ_US
    board[:, 9:] = _REGION_ONE_HOT

    # -- cards
    loc = buffers["card_loc"][i]
    loc[:] = LOC_UNSEEN
    loc[_CARD_ENTRY_TURN > observation.turn] = LOC_FUTURE
    for cid in observation.hand:
        loc[CARD_INDEX[cid]] = LOC_MY_HAND
    for cid in observation.discard_pile:
        loc[CARD_INDEX[cid]] = LOC_DISCARD
    for cid in observation.removed_cards:
        loc[CARD_INDEX[cid]] = LOC_REMOVED
    china_mine = observation.china_card_owner is me
    if china_mine:
        loc[_CHINA_INDEX] = LOC_CHINA_MINE_UP if observation.china_card_available else LOC_CHINA_MINE_DOWN
    else:
        loc[_CHINA_INDEX] = LOC_CHINA_THEIRS_UP if observation.china_card_available else LOC_CHINA_THEIRS_DOWN

    # -- globals
    g = buffers["globals"][i]
    g[:] = 0.0
    g[AM_US_INDEX] = 1.0 if me is Side.US else 0.0
    g[GLOBAL_INDEX["defcon"]] = observation.defcon / 5.0
    g[GLOBAL_INDEX["vp"]] = (observation.vp if me is Side.US else -observation.vp) / 20.0
    g[GLOBAL_INDEX["turn"]] = observation.turn / 10.0
    g[GLOBAL_INDEX["action_round"]] = observation.action_round / 8.0
    if observation.phase in PHASES:
        g[GLOBAL_INDEX[f"phase_{observation.phase}"]] = 1.0
    g[GLOBAL_INDEX["my_hand_size"]] = len(observation.hand) / 9.0
    g[GLOBAL_INDEX["their_hand_size"]] = observation.opponent_hand_size / 9.0
    g[GLOBAL_INDEX["draw_pile_size"]] = observation.draw_pile_size / N_CARDS
    g[GLOBAL_INDEX["my_space_race"]] = observation.space_race.get(my, 0) / 8.0
    g[GLOBAL_INDEX["their_space_race"]] = observation.space_race.get(their, 0) / 8.0
    g[GLOBAL_INDEX["my_space_attempts"]] = observation.space_race_attempts.get(my, 0) / 2.0
    g[GLOBAL_INDEX["their_space_attempts"]] = observation.space_race_attempts.get(their, 0) / 2.0
    g[GLOBAL_INDEX["my_military_ops"]] = observation.military_ops.get(my, 0) / 5.0
    g[GLOBAL_INDEX["their_military_ops"]] = observation.military_ops.get(their, 0) / 5.0
    g[GLOBAL_INDEX["china_mine"]] = 1.0 if china_mine else 0.0
    g[GLOBAL_INDEX["china_available"]] = 1.0 if observation.china_card_available else 0.0
    turn_fx = dict(observation.turn_effects)
    game_fx = {}
    for key, value in observation.game_effects.items():
        if key in UNENCODED_GAME_EFFECTS:
            continue
        (turn_fx if RELOCATED.get(key) == "turn" else game_fx)[key] = value
    _write_effects(g, "turn", TURN_EFFECTS, turn_fx, me)
    _write_effects(g, "game", GAME_EFFECTS, game_fx, me)
    g[GLOBAL_INDEX[f"kind_{decision.kind.name}"]] = 1.0

    ctx = decision.context
    ops = ctx.get("ops", ctx.get("ops_remaining", ctx.get("card_ops", ctx.get("che_ops", 0))))
    g[GLOBAL_INDEX["ctx_ops"]] = (ops or 0) / 4.0
    g[GLOBAL_INDEX["ctx_remaining"]] = (ctx.get("remaining") or 0) / 6.0
    g[GLOBAL_INDEX["ctx_bonus"]] = 1.0 if ctx.get("bonus") else 0.0
    g[GLOBAL_INDEX["ctx_setup"]] = 1.0 if ctx.get("setup") else 0.0
    g[GLOBAL_INDEX["ctx_side_me"]] = 1.0 if ctx.get("side") == my else 0.0
    g[GLOBAL_INDEX["ctx_choose_side_me"]] = 1.0 if ctx.get("choose_side") == my else 0.0
    g[GLOBAL_INDEX["ctx_inf_side_me"]] = 1.0 if ctx.get("inf_side") == my else 0.0
    g[GLOBAL_INDEX["ctx_op_remove"]] = 1.0 if ctx.get("op") == "remove" else 0.0
    g[GLOBAL_INDEX["ctx_whole"]] = 1.0 if ctx.get("whole") else 0.0
    g[GLOBAL_INDEX["ctx_requires_uncontrolled"]] = 1.0 if ctx.get("requires_uncontrolled") else 0.0

    # -- focus cards
    focus = buffers["focus"][i]
    focus_card = ctx.get("card", ctx.get("event"))
    focus[0] = CARD_INDEX.get(focus_card, N_CARDS) if isinstance(focus_card, str) else N_CARDS
    headline = ctx.get("opponent_headline")
    focus[1] = CARD_INDEX.get(headline, N_CARDS) if isinstance(headline, str) else N_CARDS

    # -- options
    feats = buffers["opt_feats"][i]
    countries = buffers["opt_country"][i]
    cards = buffers["opt_card"][i]
    mask = buffers["opt_mask"][i]
    feats[:] = 0.0
    countries[:] = N_COUNTRIES
    cards[:] = N_CARDS
    mask[:] = 0
    for k, action in enumerate(options):
        mask[k] = 1
        feats[k, OPTION_INDEX["position"]] = k / K_MAX
        payload = action.payload
        if not payload:
            feats[k, OPTION_INDEX["is_empty"]] = 1.0
            continue
        for key, value in payload.items():
            if key == "country" and value in COUNTRY_INDEX:
                countries[k] = COUNTRY_INDEX[value]
                feats[k, OPTION_INDEX["is_country"]] = 1.0
            elif key == "card" and value in CARD_INDEX:
                cards[k] = CARD_INDEX[value]
                feats[k, OPTION_INDEX["is_card"]] = 1.0
            else:
                # Includes the "decline" sentinels some card-typed decisions
                # carry (e.g. `{"card": "none"}` for "keep the Held Card").
                _encode_option_value(feats[k], countries, cards, k, key, value)


def _encode_option_value(
    row: np.ndarray, countries: np.ndarray, cards: np.ndarray, k: int, key: str, value: object
) -> None:
    token = f"{key}:{value}"
    idx = OPTION_VOCAB_INDEX.get(token)
    if idx is not None:
        row[idx] = 1.0
        return
    if isinstance(value, str):
        # EVENT_CHOICE reuses country and card ids as branch names.
        if value in COUNTRY_INDEX:
            countries[k] = COUNTRY_INDEX[value]
            row[OPTION_INDEX["is_country"]] = 1.0
            return
        if value in CARD_INDEX:
            cards[k] = CARD_INDEX[value]
            row[OPTION_INDEX["is_card"]] = 1.0
            return
        if value.isdigit():
            row[OPTION_INDEX["number"]] = int(value) / 5.0
            return
    row[OPTION_INDEX["other"]] = 1.0


def encode_single(observation: Observation) -> dict[str, np.ndarray]:
    """Convenience for one-off inference: a one-row batch."""
    buffers = allocate(1)
    encode_into(observation, buffers, 0)
    return buffers
