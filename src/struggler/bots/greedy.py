"""GreedyPlayer: a hand-crafted heuristic bot with no lookahead or search.

Per docs/BOTS.md: observe the current state, score every legal action of
the current decision, take the top-scoring one. Nothing more -- no
simulating future turns, no search tree, no opponent modeling.

Deliberately built as *weighted features* rather than an if/elif priority
cascade: almost every heuristic funnels through `board_value()`, a single
scalar "how good is this board for `side`" evaluation, and per-action scores
are (mostly) the marginal change to that value from taking the action, or
its dice-free expectation for chance-driven actions (coups, realignment).
This is the intended bridge to a future RL agent (see docs/BOTS.md's
roadmap): a linear model over the same features, with learned instead of
hand-set weights, is a drop-in replacement for `GreedyWeights`.

`board_value()` is the readable definition; the hot path never calls it.
Every per-action score is a one-country change, and `_swing()` computes
the value difference from just the terms that country can move (its own
Control bonus, its region's tiers) -- zero when Control does not change
hands. Same numbers, a few `Board.control` calls per option instead of a
recount of the whole map; it is what makes Greedy usable as an arena
opponent at scale (docs/WOPR.md).

Coverage: full heuristics for every core board decision kind -- where to
place Influence, which country to Coup or Realign against, which Ops type
to spend on, which card to headline or play, and Ops vs Event vs Space Race
mode. The event-specific decision kinds (WAR_TARGET, EVENT_CHOICE,
EVENT_INFLUENCE, EVENT_OPS_ORDER, QUAGMIRE_DISCARD, HELD_CARD_DISCARD,
EVENT_RESUME, RANDOM_DISCARD's non-CHANCE siblings, ...) fall back to the
first legal option. This is a documented gap, not a bug -- the same
card-by-card growth pattern the event layer itself used; extend
`_SCORERS` as each one gets a heuristic worth writing. `EVENT_CHOICE`
itself now has one card-specific heuristic (Aldrich Ames Remix: discard
the opponent's highest-Ops card) inside `_score_event_choice`, dispatched
by `decision.context["event"]`; every other EVENT_CHOICE-driven card still
falls back to the first option via that same function's default 0.0.

Priority ordering falls directly out of the weight magnitudes, not out of
branch order:
  1. Never choose a Coup (or an Ops type / Coup target that could become
     one) that would drop DEFCON to 1 -- an instant loss for the acting
     side (`defcon_self_kill_penalty`, orders of magnitude above every
     other weight).
  2. A safe Coup with a good expected margin outscores placing Influence
     (`coup_base` plus the expected board-value swing).
  3. Among Influence targets, Battlegrounds and control flips dominate
     (`battleground_control` in `board_value`).
  4. A card not worth spending on Ops gets sent to the Space Race instead
     (low `ops_mode_per_point` score vs `space_race_base`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from struggler.engine import (
    Action,
    Decision,
    DecisionKind,
    Observation,
    Region,
    ScoringTier,
    Side,
    Subregion,
)
from struggler.engine.board import Board, CountryInfo
from struggler.engine.cards import action_rounds, load_cards
from struggler.engine.core import PASS_ROUND, REALIGNMENT_STOP, SCORING_CARD_REGION
from struggler.engine.player import Event
from struggler.engine.rules import RULES

_CARDS = load_cards()

_TIER_VALUE = {
    ScoringTier.NONE: 0.0,
    ScoringTier.PRESENCE: 1.0,
    ScoringTier.DOMINATION: 2.0,
    ScoringTier.CONTROL: 3.0,
}

# Sentinel for "no candidate target exists" in the OPS_TYPE proxy searches
# below -- large enough to never win against a real (bounded) score, without
# using -inf, which would make `weights.x + _NO_OPTION` still -inf and hide
# arithmetic mistakes.
_NO_OPTION = -1_000_000_000.0


@dataclass(frozen=True)
class GreedyWeights:
    """Every knob GreedyPlayer's heuristics use. Grouped by the feature they
    price, not by decision kind, since several decision kinds share them.
    """

    # -- board_value(): the static "how good is this position" evaluation --
    region_tier: float = 6.0
    country_control: float = 2.0
    battleground_control: float = 5.0

    # -- DEFCON safety (priority #1: never die to DEFCON 1) --
    defcon_self_kill_penalty: float = 1_000_000.0
    defcon_caution: float = 4.0  # scaled by (5 - defcon): risk-aversion as DEFCON drops, short of the fatal case

    # -- per-ops-type base preference (before the marginal/expected board_value swing) --
    coup_base: float = 5.0
    realignment_base: float = 1.0
    influence_base: float = 1.0
    doubled_cost_penalty: float = 1.0  # discourages placing into opponent-controlled ("doubled") countries

    # -- which card, and how to spend it --
    space_race_base: float = 4.0
    space_race_vp_weight: float = 3.0
    space_race_ops_penalty: float = 1.5  # a high-Ops card is worth more spent on Ops than "wasted" on the Space Race
    ops_mode_per_point: float = 3.0
    event_mode_penalty: float = 30.0  # events off / unimplemented event: playing "event" is a no-op discard
    scoring_card_weight: float = 2.0  # per net VP the region would score, signed favorably/unfavorably
    hold_high_ops_weight: float = 0.5  # prefer headlining a low-Ops card, keeping high-Ops ones for Operations
    action_round_ops_weight: float = 1.0
    # A scoring card still in hand at the end of the turn loses the game
    # (rule 4.4): once the scoring cards held are as many as the action
    # rounds left, one of them must go now, whatever the board says.
    scoring_card_deadline_bonus: float = 1000.0
    # Quagmire traps the US and Bear Trap the USSR whoever plays them: for
    # the trapped side, playing one for Ops spends rounds it may need for a
    # scoring card (the trap's discards take precedence over scoring cards).
    self_trap_penalty: float = 50.0


# -- board evaluation ---------------------------------------------------------


def board_value(weights: GreedyWeights, board: Board, side: Side) -> float:
    """A static heuristic value of the current board for `side` (higher is
    better): regional Presence/Domination/Control tiers plus a flat bonus
    per country Controlled, weighted extra for Battlegrounds."""
    opponent = side.opponent
    value = 0.0
    for region in Region:
        value += _TIER_VALUE[board.region_tier(side, region)] * weights.region_tier
        value -= _TIER_VALUE[board.region_tier(opponent, region)] * weights.region_tier
    for cid, info in board.countries.items():
        controller = board.control(cid)
        if controller is None:
            continue
        per_country = weights.battleground_control if info.battleground else weights.country_control
        value += per_country if controller is side else -per_country
    return value


def _local_value(weights: GreedyWeights, board: Board, side: Side, country: str, controller: Side | None) -> float:
    """The terms of `board_value` that depend on `country`'s Influence: its
    region's two tier terms, plus its own Control bonus (given `controller`)."""
    info = board.countries[country]
    value = _TIER_VALUE[board.region_tier(side, info.region)] * weights.region_tier
    value -= _TIER_VALUE[board.region_tier(side.opponent, info.region)] * weights.region_tier
    if controller is not None:
        per_country = weights.battleground_control if info.battleground else weights.country_control
        value += per_country if controller is side else -per_country
    return value


def _swing(weights: GreedyWeights, board: Board, side: Side, country: str, deltas: Mapping[str, int]) -> float:
    """`board_value` change for `side` from adding `deltas` (`{"US": +1}`,
    `{"USSR": -2, "US": +1}`, ...) to `country`'s Influence, leaving `board`
    exactly as found.

    Computed locally rather than by recounting the map: `board_value` is a
    function of which side Controls each country, so one country's Influence
    can only move that country's own Control bonus and its region's tier
    terms -- and when Control does not change hands, nothing moves at all.
    Equal to `board_value(after) - board_value(before)`
    (`tests/test_greedy.py` pins it); a full recount per option is what
    used to make this bot thirty times slower than the engine."""
    influence = board.influence[country]
    before = board.control(country)
    for key, delta in deltas.items():
        influence[key] += delta
    after = board.control(country)
    after_value = 0.0 if after is before else _local_value(weights, board, side, country, after)
    for key, delta in deltas.items():
        influence[key] -= delta
    if after is before:
        return 0.0
    return after_value - _local_value(weights, board, side, country, before)


def _marginal_gain(weights: GreedyWeights, board: Board, side: Side, country: str, delta: int) -> float:
    """`board_value` swing from adding `delta` Influence points for `side` in
    `country`, leaving `board` exactly as found."""
    return _swing(weights, board, side, country, {side.value: delta})


def _sync_board(board: Board, observation: Observation) -> None:
    for cid, values in observation.influence.items():
        board.influence[cid]["US"] = values.get("US", 0)
        board.influence[cid]["USSR"] = values.get("USSR", 0)


# -- shared per-country rule replicas (public game data/rules, not hidden state) --


def _in_bonus_region(info: CountryInfo, bonus: str | None) -> bool:
    if bonus == "asia":
        return info.region is Region.ASIA
    if bonus == "se_asia":
        return Subregion.SOUTHEAST_ASIA in info.subregions
    return False


def _bonus_ops(info: CountryInfo, bonus: list[str] | None) -> int:
    """The extra Ops a coup in `info` earns: one per bonus region it is in
    (the China Card's Asia and Vietnam Revolts' Southeast Asia stack)."""
    return sum(1 for region in bonus or () if _in_bonus_region(info, region))


def _coup_roll_modifier_estimate(observation: Observation, side: Side, info: CountryInfo) -> float:
    mod = 0.0
    te = observation.turn_effects
    lads = te.get("la_death_squads")
    if lads and info.region in (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA):
        mod += 1.0 if side.value == lads else -1.0
    if te.get("salt"):
        mod -= 1.0
    return mod


def _coup_risks_defcon(observation: Observation, side: Side, info: CountryInfo) -> bool:
    """Whether a Coup here could degrade DEFCON at all: only Battleground
    countries do, and even those not while Nuclear Subs exempts this side."""
    if not info.battleground:
        return False
    return not (side is Side.US and bool(observation.turn_effects.get("nuclear_subs")))


def _expected_coup_gain(
    weights: GreedyWeights,
    board: Board,
    observation: Observation,
    side: Side,
    country: str,
    info: CountryInfo,
    ops: int,
) -> float:
    """Expected `board_value` swing of a Coup roll at `country`, using the
    average die roll (3.5) in place of an actual roll -- a Coup's outcome
    formula (margin = roll + ops - 2*stability + modifier) is linear in the
    roll, so this is the true expectation, not just a point estimate."""
    opponent = side.opponent
    modifier = _coup_roll_modifier_estimate(observation, side, info)
    expected_margin = 3.5 + ops - 2 * info.stability + modifier
    opp_inf = board.influence[country][opponent.value]
    opp_removed = int(round(max(0.0, min(expected_margin, opp_inf))))
    leftover = int(round(max(0.0, expected_margin - opp_removed)))
    return _swing(weights, board, side, country, {opponent.value: -opp_removed, side.value: leftover})


def _realignment_bonus(board: Board, side: Side, country: str) -> float:
    """Mirrors engine.core.Engine._realignment_bonus -- kept in sync by
    hand since this is an independent duplicate, not shared code. The
    region-bonus extra attempt (China Card in Asia / Vietnam Revolts in SE
    Asia) is deliberately NOT modeled here: it would add "count remaining
    Ops-type-choice attempts as still in-region" bookkeeping to a bot that
    already has no lookahead and only proxy (not exact) legality elsewhere
    in this module -- disproportionate complexity for its value."""
    bonus = 1.0 if board.is_adjacent(side.value, country) else 0.0
    bonus += sum(1 for n in board.neighbors(country) if board.control(n) is side)
    if board.influence[country][side.value] > board.influence[country][side.opponent.value]:
        bonus += 1.0
    return bonus


def _realignment_modifier(observation: Observation, side: Side) -> float:
    return -1.0 if (side is Side.US and observation.turn_effects.get("iran_contra")) else 0.0


def _effective_ops_estimate(card, observation: Observation, side: Side) -> int:
    ops = card.ops
    te = observation.turn_effects
    if te.get("containment") and side is Side.US:
        ops += 1
    if te.get("brezhnev") and side is Side.USSR:
        ops += 1
    if te.get("red_scare") == side.value:
        ops -= 1
    return max(1, ops)


def _space_race_expected_vp(observation: Observation, side: Side) -> float:
    pos = observation.space_race.get(side.value, 0)
    if pos >= RULES["space_race_max_box"]:
        return 0.0
    next_box = pos + 1
    box = RULES["space_race_boxes"][str(next_box)]
    probability = box["roll_max"] / 6.0
    first = observation.space_race.get(side.opponent.value, 0) < next_box
    vp = box["vp_first"] if first else box["vp_second"]
    return probability * vp


def _scoring_card_favorability(board: Board, side: Side, cid: str) -> float:
    region = SCORING_CARD_REGION.get(cid)
    if region is None:
        return 0.0
    if region is Region.EUROPE:
        # Europe has no Control value: scoring it at Control is the game
        # (`Board.score_region` refuses to guess and raises). Either side
        # holding the tier makes the card worth a win -- or a loss -- now.
        for holder in (Side.US, Side.USSR):
            if board.region_tier(holder, region) is ScoringTier.CONTROL:
                return float(RULES["vp_to_win"]) if holder is side else -float(RULES["vp_to_win"])
    net = board.score_region(region)  # positive favors US
    return net if side is Side.US else -net


# -- per-decision-kind scorers -------------------------------------------------


def _score_place_influence(weights: GreedyWeights, board: Board, observation: Observation, action: Action) -> float:
    side = observation.side
    country = action.payload["country"]
    cost = board.influence_cost(side, country)
    gain = _marginal_gain(weights, board, side, country, 1)
    return weights.influence_base + gain - (cost - 1) * weights.doubled_cost_penalty


def _score_coup_target(weights: GreedyWeights, board: Board, observation: Observation, action: Action) -> float:
    side = observation.side
    country = action.payload["country"]
    info = board.countries[country]
    decision = observation.pending_decision
    ops = decision.context["ops"]
    ops += _bonus_ops(info, decision.context.get("bonus"))

    if observation.defcon <= 2 and _coup_risks_defcon(observation, side, info):
        return -weights.defcon_self_kill_penalty

    gain = _expected_coup_gain(weights, board, observation, side, country, info, ops)
    caution = weights.defcon_caution * (5 - observation.defcon)
    return weights.coup_base + gain - caution


def _score_realignment_target(
    weights: GreedyWeights, board: Board, observation: Observation, action: Action
) -> float:
    side = observation.side
    opponent = side.opponent
    country = action.payload["country"]
    if country == REALIGNMENT_STOP:
        return 0.0  # stop once no remaining attempt is worth more than nothing
    own_bonus = _realignment_bonus(board, side, country)
    opp_bonus = _realignment_bonus(board, opponent, country)
    expected_margin = own_bonus - opp_bonus + _realignment_modifier(observation, side)

    if expected_margin > 0:
        removed = int(round(min(expected_margin, board.influence[country][opponent.value])))
        swing = _swing(weights, board, side, country, {opponent.value: -removed})
    elif expected_margin < 0:
        removed = int(round(min(-expected_margin, board.influence[country][side.value])))
        swing = _swing(weights, board, side, country, {side.value: -removed})
    else:
        swing = 0.0
    return weights.realignment_base + swing


def _best_influence_value(
    weights: GreedyWeights, board: Board, side: Side, ops: int
) -> float:
    best = None
    for cid in board.countries:
        if not board.is_reachable(side, cid):
            continue
        if board.influence_cost(side, cid) > ops:
            continue
        gain = _marginal_gain(weights, board, side, cid, 1)
        if best is None or gain > best:
            best = gain
    return best if best is not None else _NO_OPTION


def _best_coup_value(
    weights: GreedyWeights, board: Board, observation: Observation, side: Side, ops: int, bonus: str | None
) -> float | None:
    """Best expected Coup value among proxy-legal targets, or None if every
    one of them would be a DEFCON self-kill. Region-lock effects beyond
    `RULES["coup_min_defcon"]` (NATO, The Reformer, ...) are not replicated
    here -- out of scope for v1 (core board decisions); see the module docstring."""
    opponent = side.opponent
    best = None
    for cid, info in board.countries.items():
        if board.influence[cid][opponent.value] <= 0:
            continue
        if observation.defcon < RULES["coup_min_defcon"].get(info.region.name, 1):
            continue
        if observation.defcon <= 2 and _coup_risks_defcon(observation, side, info):
            continue
        target_ops = ops + _bonus_ops(info, bonus)
        gain = _expected_coup_gain(weights, board, observation, side, cid, info, target_ops)
        if best is None or gain > best:
            best = gain
    return best


def _best_realignment_value(weights: GreedyWeights, board: Board, observation: Observation, side: Side) -> float:
    opponent = side.opponent
    best = None
    for cid, info in board.countries.items():
        if board.influence[cid][opponent.value] <= 0:
            continue
        if observation.defcon < RULES["coup_min_defcon"].get(info.region.name, 1):
            continue
        own_bonus = _realignment_bonus(board, side, cid)
        opp_bonus = _realignment_bonus(board, opponent, cid)
        value = own_bonus - opp_bonus + _realignment_modifier(observation, side)
        if best is None or value > best:
            best = value
    return best if best is not None else _NO_OPTION


def _score_ops_type(weights: GreedyWeights, board: Board, observation: Observation, action: Action) -> float:
    side = observation.side
    ctx = observation.pending_decision.context
    ops = ctx["ops"]
    bonus = ctx.get("bonus")
    ops_type = action.payload["type"]

    if ops_type == "influence":
        return weights.influence_base + _best_influence_value(weights, board, side, ops)
    if ops_type == "coup":
        best = _best_coup_value(weights, board, observation, side, ops, bonus)
        if best is None:
            # No Coup target is safe at the current DEFCON: refuse "coup" as
            # an Ops type outright, rather than let COUP_TARGET default into
            # a self-kill (priority #1).
            return -weights.defcon_self_kill_penalty
        caution = weights.defcon_caution * (5 - observation.defcon)
        return weights.coup_base + best - caution
    return weights.realignment_base + _best_realignment_value(weights, board, observation, side)


def _score_headline(weights: GreedyWeights, board: Board, observation: Observation, action: Action) -> float:
    side = observation.side
    cid = action.payload["card"]
    card = _CARDS[cid]
    if card.scoring:
        return weights.scoring_card_weight * _scoring_card_favorability(board, side, cid)
    # Non-scoring: headlining is a no-op discard while its event is unfired
    # (events off, or an unimplemented event) -- spend a low-Ops card here and
    # keep higher-Ops ones for Operations.
    value = -weights.hold_high_ops_weight * card.ops
    if _SELF_TRAPS.get(cid) is side:
        value -= weights.self_trap_penalty
    return value


def _score_action_round_play(
    weights: GreedyWeights, board: Board, observation: Observation, action: Action
) -> float:
    side = observation.side
    cid = action.payload["card"]
    if cid == PASS_ROUND:
        return 0.0  # declining an extra round: any card worth playing outscores it
    card = _CARDS[cid]
    if card.scoring:
        value = weights.scoring_card_weight * _scoring_card_favorability(board, side, cid)
        if _scoring_cards_due(observation):
            value += weights.scoring_card_deadline_bonus
        return value
    ops = _effective_ops_estimate(card, observation, side)
    value = weights.action_round_ops_weight * ops
    if _SELF_TRAPS.get(cid) is side:
        value -= weights.self_trap_penalty
    return value


#: Cards whose event locks a side's action rounds regardless of who plays them.
_SELF_TRAPS: dict[str, Side] = {"Quagmire": Side.US, "Bear_Trap": Side.USSR}


def _scoring_cards_due(observation: Observation) -> bool:
    """True when the scoring cards in hand are as many as the action rounds
    left this turn (this one included): holding one past the turn loses."""
    held = sum(1 for cid in observation.hand if cid in _CARDS and _CARDS[cid].scoring)
    remaining = action_rounds(observation.turn) - observation.action_round + 1
    return held > 0 and held >= remaining


def _score_play_mode(weights: GreedyWeights, board: Board, observation: Observation, action: Action) -> float:
    side = observation.side
    cid = observation.pending_decision.context["card"]
    card = _CARDS[cid]
    mode = action.payload["mode"]
    ops = _effective_ops_estimate(card, observation, side)

    if mode == "space_race":
        expected_vp = _space_race_expected_vp(observation, side)
        return (
            weights.space_race_base
            + weights.space_race_vp_weight * expected_vp
            - weights.space_race_ops_penalty * ops
        )
    if mode in ("ops", "un_intervention"):
        return weights.ops_mode_per_point * ops
    # mode == "event": with the event layer off (or for a card with no
    # implemented event yet) this is a no-op discard -- always worse than
    # spending the card. GreedyPlayer does not attempt event-value
    # heuristics (out of scope for v1; see the module docstring).
    return -weights.event_mode_penalty


def _score_event_choice(weights: GreedyWeights, board: Board, observation: Observation, action: Action) -> float:
    """Dispatches by `decision.context["event"]`. Every EVENT_CHOICE-driven
    card other than the ones named here still returns 0.0 for all of its
    options, i.e. still falls back to the first legal one (see module
    docstring)."""
    event = observation.pending_decision.context.get("event")
    if event == "Aldrich_Ames_Remix":
        # Force the discard of the opponent's most valuable card in hand,
        # proxied by its Ops value (a scoring card's real cost -- losing the
        # region -- isn't modeled by GreedyPlayer's no-lookahead heuristics
        # elsewhere either; see _score_play_mode above).
        card = _CARDS.get(action.payload["choice"])
        return float(card.ops) if card is not None else 0.0
    return 0.0


_SCORERS: dict[DecisionKind, Callable[[GreedyWeights, Board, Observation, Action], float]] = {
    DecisionKind.PLACE_INFLUENCE: _score_place_influence,
    DecisionKind.COUP_TARGET: _score_coup_target,
    DecisionKind.REALIGNMENT_TARGET: _score_realignment_target,
    DecisionKind.OPS_TYPE: _score_ops_type,
    DecisionKind.HEADLINE_PLAY: _score_headline,
    DecisionKind.ACTION_ROUND_PLAY: _score_action_round_play,
    DecisionKind.PLAY_MODE: _score_play_mode,
    DecisionKind.EVENT_CHOICE: _score_event_choice,
}


class GreedyPlayer:
    """See module docstring. Stateless across turns beyond its own scratch
    `Board` (re-synced from `observation.influence` -- public state -- on
    every call, never the engine's own `Board`)."""

    def __init__(self, weights: GreedyWeights | None = None) -> None:
        self.weights = weights or GreedyWeights()
        self._board = Board(variants=Board.VARIANTS)  # every space, whatever the game's variants: synced by key

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        decision: Decision = observation.pending_decision
        scorer = _SCORERS.get(decision.kind)
        if scorer is None:
            return decision.options[0]
        _sync_board(self._board, observation)
        return max(
            decision.options,
            key=lambda action: scorer(self.weights, self._board, observation, action),
        )
