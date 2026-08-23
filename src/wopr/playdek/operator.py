"""PlaydekOperator: Playdek's AI as the physical-mode opponent of a bot.

The struggler engine referees in physical mode (docs/BOTS.md) with the
AI's seat as the physical side; the bot sees only its `Observation`. The
operator is the `Player` registered under the AI's side and `Side.CHANCE`
-- the seat a human operator would hold -- and answers from what the DLL
reports: the AI's card and use from the card-play animation records, its
targets from the roll records, its influence from the absolute influence
state, its dice from the roll records, the bot's deals from the card
locations. The bot's own seat is wrapped (`players()`), so each of its
actions is translated into the DLL's prompts as it is made: the engine
leads for the bot's seat, the DLL leads for the AI's.

    pd = Playdek()
    result = play_match(pd, JoshuaPlayer.from_checkpoint(path), seed=1, side=Side.USSR)

Either program can be the one that is wrong when they disagree: every
disagreement is a `Divergence` on the report (`bridge.py`), a fatal one
ends the game as a `Desync`. `emulate=` plays the same protocol against a
DLL-prompt policy in hotseat mode instead of the AI -- the other seat's
records feed the engine exactly as the AI's would, at 10k prompts a
second rather than 15 s a decision, which is how the operator is tested.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from struggler.engine import Engine
from struggler.engine.core import PASS_ROUND
from struggler.engine.events import EVENTS
from struggler.engine.player import Event, Player
from struggler.engine.rules import RULES
from struggler.engine.types import Action, Decision, DecisionKind, Observation, Side
from struggler.runner import play_game
from wopr.playdek import ids, translate as T
from wopr.playdek.bridge import (CARD_KINDS, CHINA, CMC_DEFUSE, COUNTRY_KINDS, GRANTED_OPS_PROMPT, HAND_LOCATION, HAND_OF, HEADLINE_OF, PILES, REFORMER_KNOWN,
                                 UI_ONLY, UN, Bridge, Move, Report)
from wopr.playdek import ffi
from wopr.playdek.ffi import AIDifficulty, EventType, SelectionHint
from wopr.playdek.game import Option, Playdek, Prompt

__all__ = ["Desync", "MatchResult", "PlaydekOperator", "play_match"]

Policy = Callable[[Prompt], Option]

ANIMATION_HAND = 1  # `animation_source_location` of a card leaving a hand
ANIMATION_RESOLVE = 6  # `animation_destination_location` of a card being played
ANIMATION_DISCARD = 4  # `animation_source_location` of a card played out of the discard pile (Star Wars)
ACTION_PROMPT = "Play Your Action Round"
ANIMATION_SCORING = 0x1  # the hint of a scoring card's play (no use to choose)
ANIMATION_FIRED = 0x2  # the hint of a card another event fires out of a hand (Five Year Plan)
#: EVENT_CHOICE payloads that decline an optional step: the DLL's "Do Not
#: Discard" / "Done Removing" / "Return It" entries (`SelectionHint.STOP`).
DECLINES = {"refuse", "none", "decline", "done", "keep", "return", "no", "stop", "skip"}
_RECORD = Prompt(-1, "<record>", ())  # the prompt of a move learned from the records, not asked
_TRACED = {
    EventType.OUTPUT_ANIMATION_CARD, EventType.OUTPUT_ANIMATION_ADD_INFLUENCE, EventType.OUTPUT_ANIMATION_REMOVE_INFLUENCE,
    EventType.OUTPUT_ANIMATION_TARGET_COUNTRY, EventType.CHINA_CARD, EventType.PUSH_RESOLVE_CARD, EventType.POP_RESOLVE_CARD,
    EventType.ACTION_ROUND, EventType.COMMIT_PLAYER_DECISION, EventType.DEFCON_LEVEL, EventType.SCORING_CARD_PLAYED,
}


class Desync(RuntimeError):
    """The two programs could not be kept on the same game."""


class PlaydekEnded(Exception):
    """The DLL's game is over on a rule the engine cannot apply (a scoring
    card held in the hand it cannot see): the DLL's result stands."""


def _pass_option(prompt: Prompt) -> Option | None:
    """The "Pass" of an extra action round's "Play Your Action Round"."""
    return next((o for o in prompt.visible if o.text == "Pass" or T.meaning(o).meaning is T.Meaning.STOP), None)


def _record_move(side: Side, meaning: T.OptionMeaning, text: str) -> Move:
    return Move(side, _RECORD, Option(-1, 0, 0, False, text), meaning)


@dataclass
class MatchResult:
    seed: int
    side: str  # the bot's seat
    difficulty: str
    winner: str | None  # the engine's
    playdek_winner: str | None
    win_type: str
    score: int  # the DLL's VP track at the end (US-positive)
    turn: int
    desync: bool
    void: str | None = None  # the game ended on a known difference between the two rule sets: not a desync, not a game either
    divergences: list[str] = field(default_factory=list)
    prompts: int = 0
    seconds: float = 0.0

    @property
    def bot_won(self) -> bool | None:
        if self.desync or self.void is not None or self.playdek_winner is None:
            return None
        return self.playdek_winner == self.side


class _Seat:
    """The bot's own `Player`, with every action it takes told to the DLL."""

    def __init__(self, operator: PlaydekOperator, player: Player) -> None:
        self._op = operator
        self._player = player

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        decision = observation.pending_decision
        self._op.before_bot_decision(decision)
        narrowed = self._op.narrow(decision)
        if narrowed is not decision:
            observation = dataclasses.replace(observation, pending_decision=narrowed)
        action = self._player.choose_action(observation, history)
        self._op.note(decision, action)
        return action


class PlaydekOperator(Bridge):
    def __init__(self, pd: Playdek, *, seed: int, side: Side, difficulty: AIDifficulty = AIDifficulty.HARD,
                 emulate: Policy | None = None, game_no: int = 0, max_divergences: int = 40, trace: bool = False) -> None:
        self.side = side  # the bot's seat
        self.other = side.opponent  # the AI's (or, emulated, the policy's)
        self.emulate = emulate
        super().__init__(pd, game_no=game_no, seed=seed,
                         # Hotseat seats USSR/US by id, whatever `local_side` says.
                         local_side=Side.USSR if emulate is not None else side,
                         ai_difficulty=None if emulate is not None else difficulty,
                         physical_side=self.other, deal_after_setup=True, max_divergences=max_divergences, trace=trace)
        self.outgoing: collections.deque[tuple[Decision, Action]] = collections.deque()  # the bot's actions not yet told to the DLL
        self._un_target: str | None = None  # the opponent's card to name once UN Intervention's "Play Event" is answered
        self._played: tuple[str, int] | None = None  # (card, turn) of the other seat's last queued play: its further use records add no card
        self._headlined: tuple[str, int] | None = None  # (card, turn) of the other seat's headline: reshuffled and headlined again next turn, it is queued again
        self._plays_seen: set[str] = set()  # the other seat's cards seen played: they left its hand, but not as a discard
        self._play_seq: dict[str, int] = {}  # card -> record seq of its latest play animation, dropped when it is dealt again
        self._china: Side | None = None  # the China Card's holder per the DLL's CHINA_CARD records (it has no CARD_LOCATION)
        self._fired: list[str] = []  # cards another event fired out of a hand (Five Year Plan), not yet discarded there
        self._from_discard: list[str] = []  # cards an event played out of the discard pile (Star Wars), not yet accounted for
        self._auto_declined = 0  # lone "Pass" prompts of the bot's answered before the engine asked the choice
        self._declined_for_dll: Decision | None = None
        self._taken: dict[str, Side] = {}  # cards shown out of a hand by an event -> that hand's owner (Grain Sales: the opponent then plays it)
        self._handed: set[str] = set()  # cards Grain Sales handed the US that the engine has not played yet (the DLL discards them at once)
        self._revealed: list[str] = []  # cards shown out of a hand (Grain Sales' draw, CIA Created's hand)
        self._first: tuple[tuple, Option] | None = None  # hotseat: the DLL re-asks the very first prompt and drops the first answer
        self.play_log: list[int] = []  # seq of every card entering the resolve slot: the boundaries between actions
        self.use_log: list[int] = []  # seq of every use record of a play (the Ops half, the event half): the boundaries within one
        self._synced_move_seq = 0  # the card-move count when the two states last agreed at rest
        self._last_action: tuple[Decision, Action] | None = None  # the bot's latest, not yet applied by the engine when `flush` runs
        self._simulating = 0
        self._completed_for_dll = False  # the bot's last action was finished in the DLL after the engine's game had ended
        self._extra_ops_pending = 0  # a simulation's Ops to grant the US once the play under way is done (the DLL's Grain Sales)

    def players(self, player: Player) -> dict[Side, Player]:
        """The `play_game` players table: the bot on its seat, the operator
        on the other and on CHANCE (docs/BOTS.md, physical mode)."""
        return {self.side: _Seat(self, player), self.other: self, Side.CHANCE: self}

    # -- the DLL's records -> the other seat's moves --------------------------

    def _absorb(self, ev, replay: bool = False) -> bool:
        if not super()._absorb(ev, replay):
            return False
        f = ev.fields
        if self.trace and ev.kind in _TRACED:
            print(f"  EV  {ev}")
        if ev.kind == EventType.CHINA_CARD:
            self._china = {1: Side.USSR, 2: Side.US}.get(f["player"])
        elif ev.kind == EventType.PUSH_REVEAL_CARD:
            try:
                self._revealed.append(ids.card_id(f["card"]))
            except KeyError:
                pass
        elif ev.kind == EventType.CARD_LOCATION:
            try:
                card = ids.card_id(f["id"])
            except KeyError:
                return
            if f["location"] not in HAND_OF:
                self._taken.pop(card, None)
            elif self._last_moves.get(card, (None, None, None))[1:] == (f["location"], self._move_seq):
                self._play_seq.pop(card, None)  # just moved into a hand (dealt, handed, returned): its next exit is a fresh move
            if HEADLINE_OF.get(f["location"]) is self.other and self._headlined != (card, self._dll_turn):
                self._headlined = (card, self._dll_turn)
                self.queue(self.other, _record_move(self.other, T.OptionMeaning(T.Meaning.CARD, card=card, label=card), f"headline {card}"))
        elif ev.kind == EventType.PUSH_RESOLVE_CARD:
            self.play_log.append(self._seq)  # an action boundary: a card entering the resolve slot
            try:
                pushed = ids.card_id(f["card"])
            except KeyError:
                pushed = None
            if self._played is not None and self._played[0] == pushed:
                # The same card entering the slot again is a new play of it
                # (the China Card handed back by Ussuri River Skirmish and
                # played again in the same turn), not a further use record.
                self._played = None
        elif ev.kind == EventType.OUTPUT_ANIMATION_CARD:
            # A card leaving a hand for the resolve slot, the hint saying how
            # it is used (`translate.ANIMATION_USES`): the one place the DLL
            # reports a card play as a choice. A hotseat game re-emits it
            # at the action's commit (dropped before this, `mark_replays`),
            # which is the next action boundary -- after the next
            # ACTION_ROUND record, so the round the record arrives in says
            # nothing about whose play it is; the hand the card last left
            # does.
            if f["animation_source_location"] == ANIMATION_DISCARD and f["animation_destination_location"] == ANIMATION_RESOLVE:
                # A card played out of the discard pile (Star Wars' copy):
                # no hand move reports the choice, this push does.
                try:
                    self._from_discard.append(ids.card_id(f["card_instance_id"]))
                except KeyError:
                    pass
                return True
            if f["animation_source_location"] != ANIMATION_HAND or f["animation_destination_location"] != ANIMATION_RESOLVE:
                return True
            try:
                card = ids.card_id(f["card_instance_id"])
            except KeyError:
                return True
            hint = f["animation_event_hint"]
            if hint & 0xFF in (0x01, 0x02) and hint >> 8:
                self.use_log.append(self._seq)  # a use chosen, or the automatic other half of an Ops play
            use = T.use_from_animation(hint)
            if hint == ANIMATION_FIRED and card not in self._fired:
                self._fired.append(card)  # Five Year Plan's discard firing as a US event, Grain Sales' draw
                if self.last_hand_of(card) is not None:
                    self._taken[card] = self.last_hand_of(card)
                    self._handed.add(card)
            if use is None and hint != ANIMATION_SCORING:
                return True  # a headline reveal, the automatic second half, an event another event fired
            # Whose play: the hand's owner -- unless an event showed the card
            # out of that hand (Grain Sales), in which case the opponent took
            # it and plays it from there.
            actor = self.last_hand_of(card)
            if card in self._taken and actor is self._taken[card]:
                actor = actor.opponent
            self._play_seq[card] = self._seq
            self.recent.append(f"play {card} {hint:#x} by {actor}")
            if actor is not self.other:
                return True
            self._plays_seen.add(card)
            self._flower_power_check(actor, card)
            if self._played != (card, self._dll_turn):
                self._played = (card, self._dll_turn)
                self.queue(self.other, _record_move(self.other, T.OptionMeaning(T.Meaning.CARD, card=card, label=card), f"play {card}"))
            if use is not None:
                self.queue(self.other, _record_move(self.other, T.OptionMeaning(T.Meaning.USE, use=use, label=f"{hint:#x}"), f"use {use}"))
        return True

    def _answer_hidden_hand(self, d: Decision) -> Action | None:
        action = super()._answer_hidden_hand(d)
        if action is None and d.context.get("event") == "Missile_Envy_physical_pick":
            # The card the AI gave is pushed into the resolve slot as "fired"
            # the moment it is exchanged; its move out of the hand is reported
            # only once its event is done asking (SALT's recovery).
            offered = {a.payload["choice"] for a in d.options}
            card = next((c for c in reversed(self._fired) if c in offered and self._taken.get(c) is self.engine.physical_side), None)
            if card is not None:
                self._fired.remove(card)
                return self._pick(d, lambda a: a.payload["choice"] == card, f"Missile Envy takes {card} (pushed as fired)")
        return action

    def _flower_power_check(self, side: Side, card: str) -> None:
        """Flower Power pays the USSR 2 VP for a war card the US *plays*
        (the engine, and the card); the DLL pays only when the war's event
        happens -- not for an Arab-Israeli War under Camp David Accords.
        The VP then differ for the rest of the game: known, and void."""
        if (side is Side.US and card in RULES["war_cards"] and self.engine.game_effects.get("flower_power")
                and card in EVENTS and not EVENTS[card].eligible(self.engine, Side.US) and not self._simulating):
            self.known["Flower Power: the DLL pays no VP for a US war card whose event is prevented (Arab-Israeli War under Camp David)"] += 1
            self.diverge("rules", f"Flower Power: the engine pays the USSR 2 VP for the US playing {card} whose event cannot happen, "
                         "the DLL does not", fatal=True)

    def _exit_is_play(self, card: str, record_seq: int) -> bool:
        played = self._play_seq.get(card)
        return played is not None and played < record_seq

    def card_that_left_any(self, owner: Side) -> list[str]:
        """Cards that left `owner`'s hand for the discard or removed pile and
        have not been accounted for, latest first (not consumed)."""
        gone = sorted(((seq, c) for seq, c, was, _, _ in self._exits
                       if was == HAND_LOCATION[owner] and c not in self._plays_seen and (c, self._dll_turn) != self._headlined
                       and seq > self._synced_move_seq), reverse=True)
        return [c for _, c in gone]

    def last_hand_of(self, card: str) -> Side | None:
        """The hand `card` is in, or the one it last left."""
        if card == CHINA:
            return self._china
        owner = self.hand_of(card)
        if owner is None and card in self._last_moves:
            owner = HAND_OF.get(self._last_moves[card][0])
        return owner

    # -- the engine asks the other seat or CHANCE -----------------------------

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        decision = observation.pending_decision
        while True:
            action = self._answer(decision)
            if action is not None:
                if self.trace and not self._simulating:
                    print(f"  ENG {decision.actor.value:6s} {decision.kind.value} -> {dict(action.payload)}")
                self.report.engine_steps += 1
                self.report.steps += 1
                return action
            if self.stop or not self._more(decision):
                raise Desync(str(self.report.divergences[-1]) if self.report.divergences else "the DLL has nothing more")

    def _more(self, decision: Decision) -> bool:
        """Advance the DLL by one step so `_answer` can be retried: reply to
        the prompt it is at, or pump it. False when it has nothing more."""
        prompt = self.game.prompt
        if prompt is None:
            if self.game.result is not None:
                if self._simulating:
                    return False
                r = self.game.result
                if r.win_type is ffi.GameOverType.HELD_CARDS and self._sides_by_player.get(r.winner_id) is self.side:
                    self.known["held scoring card in the hand the engine cannot see"] += 1
                    raise PlaydekEnded
                self.diverge("game over", f"Playdek's game ended ({self.game.result}) while the engine still asks {decision.actor.value} "
                             f"{decision.kind.value} at DEFCON {self.engine.defcon}, VP {self.engine.vp}, turn {self.engine.turn} AR {self.engine.action_round}",
                             fatal=True)
                return False
            self._pump()
            return True
        if self.prompt_side(prompt) is self.other and self.emulate is not None:
            self._choose(prompt, self._reasked(prompt) or self.emulate(self._emulated(prompt)))
            return True
        option = self._reply(prompt)
        if option is None:
            self.diverge("decision mismatch", f"engine asks {decision.actor.value} {decision.kind.value} (context {dict(decision.context)}); "
                         f"the DLL asks {self.prompt_side(prompt).value} {prompt.text!r} {[o.text for o in prompt.visible]} and nothing of the bot's is left to answer it "
                         f"(queued for {self.other.value}: {[m.option.text for m in self.moves[self.other]]}, rolls {list(self.rolls)}; "
                         f"state: {'; '.join(self.state_diffs(hands=True)) or 'no difference'}; recent records: {list(self.recent)})", fatal=True)
            return False
        self._choose(prompt, option)
        return True

    def _pump(self) -> Prompt | None:
        if self.game.result is not None:
            return None
        prompt = self.game.pump()
        self._absorb_events()
        return prompt

    def _choose(self, prompt: Prompt, option: Option) -> None:
        self.report.prompts += 1
        m = T.meaning(option)
        if m.meaning is T.Meaning.COUNTRY and m.country in ids.INDEX_BY_COUNTRY:
            self._answered_countries.add(ids.INDEX_BY_COUNTRY[m.country])  # its preview is not a replay (`mark_replays`)
        grain = self._grain_card(prompt)
        if grain is not None and self.prompt_side(prompt) is self.other:
            # The emulated seat's Grain Sales: taken when the drawn card is played (a scoring card under its own hint)
            self._grain = (grain, option.hint in (SelectionHint.SWITCH_CARD, SelectionHint.PLAY_SCORING_CARD))
        if self.trace:
            print(f"  PD  {self.prompt_side(prompt).value:4s} {prompt.text!r} -> {option.text!r}")
        if self.emulate is not None and self.report.prompts == 1:
            self._first = ((prompt.player_id, prompt.text, prompt.options), option)
        self.game.choose(option.index)

    def _reasked(self, prompt: Prompt) -> Option | None:
        """The hotseat re-ask of the very first prompt, whose first answer
        the DLL drops (its records stand): that answer again."""
        if self._first is None:
            return None
        key, option = self._first
        self._first = None
        return option if key == (prompt.player_id, prompt.text, prompt.options) else None

    def _answer(self, d: Decision) -> Action | None:
        if d.actor is self.other:
            self._resync()
            if d.kind is DecisionKind.ACTION_ROUND_PLAY and not self.moves[self.other] and any(a.payload["card"] == PASS_ROUND for a in d.options):
                # An extra action round (Space Station, North Sea Oil) the
                # seat passed leaves no record: the DLL is past its play
                # prompt with no play queued.
                prompt = self.game.prompt
                if prompt is None or self.prompt_side(prompt) is not self.other or prompt.text != ACTION_PROMPT:
                    return self._pick(d, lambda a: a.payload["card"] == PASS_ROUND, "the extra action round passed (no play recorded)")
            if d.kind in COUNTRY_KINDS:
                return self._answer_country(d)
            if d.kind is DecisionKind.EVENT_CHOICE:
                if d.context.get("event") == "Grain_Sales_to_Soviets":
                    if self._grain is not None:
                        card, took = self._grain
                        self._grain = None
                        action = self._pick(d, lambda a: a.payload.get("choice") == ("take" if took else "return"), f"Grain Sales {'take' if took else 'return'} {card}")
                    else:
                        action = self._answer_grain_sales(d)
                    if action is not None and action.payload.get("choice") == "return":
                        self._taken.clear()  # the drawn card is its owner's own again
                        if not self._simulating:
                            self._handed.clear()
                    return action
                return self._answer_choice(d)
            if d.kind in (DecisionKind.QUAGMIRE_DISCARD, DecisionKind.HELD_CARD_DISCARD):
                card = self.card_that_left(self.other, {a.payload["card"] for a in d.options})
                if card is not None:
                    return self._pick(d, lambda a: a.payload["card"] == card, f"discard {card}")
                prompt = self.game.prompt
                if d.kind is DecisionKind.QUAGMIRE_DISCARD and any(a.payload["card"] == "none" for a in d.options) and (
                        self.game.result is not None or (prompt is not None and self.prompt_side(prompt) is self.side)):
                    # The DLL is past the trap step with nothing discarded:
                    # the trapped seat had no 2+-Ops card (none of the hidden
                    # pool's is in its hand) and kept its scoring card ("You
                    # May Play a Scoring Card" -> Pass, a prompt that leaves
                    # no record).
                    return self._pick(d, lambda a: a.payload["card"] == "none", "no 2+-Ops card in the trapped hand / the scoring card kept")
                return None
            if d.kind is DecisionKind.PLAY_MODE and len(d.options) == 1:
                return d.options[0]  # a scoring card: the DLL reports no use for it
            if (d.kind in (DecisionKind.PLAY_MODE, DecisionKind.EVENT_OPS_ORDER) and d.context.get("card") in self._handed
                    and not self.moves[self.other]):
                # The card Grain Sales handed over is played at once, and the
                # DLL reports no use for it (only the coup or the influence
                # that followed): each use is tried on a copy. Once the real
                # engine has its answer the card is played (the order of an
                # opponent card's event and Ops is the one decision left).
                action = self._simulate(d)
                if not self._simulating and (d.kind is DecisionKind.EVENT_OPS_ORDER or action.payload.get("mode") != "ops"
                                             or not self.engine._is_opponent_event(Side.US, self.engine.cards[d.context["card"]])):
                    self._handed.discard(d.context["card"])
                return action
            if d.kind is DecisionKind.OPS_TYPE and not (self.moves[self.other] and self.moves[self.other][0].meaning.meaning is T.Meaning.USE):
                return self._answer_ops_type(d)
        if d.actor is Side.CHANCE and d.kind is DecisionKind.RANDOM_DISCARD:
            offered = {a.payload["card"] for a in d.options}
            purpose = d.context.get("purpose")
            if purpose == "grain_sales" and self._grain is None:
                # The other seat's Grain Sales: the card shown out of the
                # bot's hand (a reveal record, or the card pushed into the
                # resolve slot as "fired" by the event), which stays in the
                # hand when returned.
                for shown in (self._revealed, self._fired):
                    card = next((c for c in reversed(shown) if c in offered), None)
                    if card is not None:
                        shown.remove(card)
                        return self._pick(d, lambda a: a.payload["card"] == card, f"Grain Sales drew {card}")
                return None
            if purpose != "grain_sales":
                # Five Year Plan's discard firing as a US event resolves before
                # the card is discarded: named by its resolve record.
                card = next((c for c in reversed(self._fired) if c in offered), None)
                if card is not None:
                    self._fired.remove(card)
                    return self._pick(d, lambda a: a.payload["card"] == card, f"random discard {card} (fired)")
        return super()._answer(d)

    def _resync(self) -> None:
        """Move the inference window (`synced_seq`) up to the latest card
        play whose board, as the DLL had it then, is the engine's board
        now. The states are compared at the bot's card prompts only; the
        bot's own action and the other seat's whole chunk lie between two
        of those, and the history of the bot's action (a Liberation
        Theology point the bot's coup then removed) must not be read as
        the other seat's placements when the engine gets to its chunk."""
        e = self.engine
        for s in reversed(self.play_log):
            if s <= self.synced_seq:
                return
            if all(self._dll_influence_at(c, s) == (inf["USSR"], inf["US"]) for c, inf in e.board.influence.items() if c in self.influence):
                self.synced_seq = s
                return

    def _dll_influence_at(self, country: str, seq: int) -> tuple[int, int]:
        """The DLL's (USSR, US) Influence in `country` as of record `seq`.
        The DLL reports a country's influence once it changes (the setup
        values included), so a country with no record yet is empty."""
        value = (0, 0)
        for q, v in self.influence_history.get(country, ()):
            if q > seq:
                break
            value = v
        return value

    def first_change(self, country: str, side: Side, op: str) -> int | None:
        """When, since the two states last agreed, the DLL's influence of
        `side` in `country` first went above ("place") or below ("remove")
        what the engine has there now; None if it never did."""
        column = 0 if side is Side.USSR else 1
        mine = self.engine.board.influence[country][side.value] if country in self.engine.board.influence else None
        if mine is None:
            return None
        past = (lambda v: v > mine) if op == "place" else (lambda v: v < mine)
        # The side's own column only: a record that moved the other side's
        # Influence (a removal) repeats this side's value, and is no change
        # of it -- taken for one, a point the engine already holds was
        # placed again.
        entries = []
        prev = next((value[column] for seq, value in reversed(self.influence_history.get(country, ())) if seq <= self.synced_seq), None)
        for seq, value in self.influence_history.get(country, ()):
            if seq <= self.synced_seq:
                continue
            if value[column] != prev:
                entries.append((seq, value[column]))
            prev = value[column]
        for i, (seq, v) in enumerate(entries):
            if not past(v):
                continue
            # A surplus gone again in a later record with no coup or
            # realignment on the country in between, and no other card
            # played in between, nor the other half of the same play (the
            # Ops half's point removed by the event half: Fidel ops-first),
            # is a transient of one event's own resolution (Nasser: the
            # USSR's +2, then half the US removed), not something the
            # engine will ask a decision for. Undone by dice (a Marshall
            # Plan point realigned away) or by a later action's event (the
            # same point removed by De Gaulle) it is.
            undone = next((q for q, w in entries[i + 1:] if not past(w)), None)
            if (undone is not None and not any(seq < q < undone and c == country for q, c in self.roll_log)
                    and not any(seq < b < undone for b in self.play_log) and not any(seq < b < undone for b in self.use_log)):
                continue
            return seq
        return None

    def _answer_ops_type(self, d: Decision) -> Action | None:
        """Ops an event granted (a boycotted Olympic Games' sponsor, CIA
        Created, ...): no use record names the type, the dice or the
        influence that followed do."""
        # The earliest fact not yet accounted for: an Ops granted before the
        # seat's own action round must not take the dice of that round.
        firsts: list[tuple[int, str]] = []
        for r in self.rolls:
            if r.side in (None, self.other) and r.kind in (DecisionKind.COUP_ROLL, DecisionKind.REALIGNMENT_ACTOR_ROLL):
                firsts.append((self.roll_seq.get(id(r), 0), "coup" if r.kind is DecisionKind.COUP_ROLL else "realignment"))
        for c in self.influence:
            seq = self.first_change(c, self.other, "place")
            if seq is not None:
                firsts.append((seq, "influence"))
        if not firsts:
            return None
        want = min(firsts)[1]
        return self._pick(d, lambda a: a.payload["type"] == want, f"ops type {want} (from what followed)")

    def _answer_country(self, d: Decision) -> Action | None:
        kind = d.kind
        if kind is DecisionKind.COUP_TARGET or kind is DecisionKind.WAR_TARGET:
            want = DecisionKind.COUP_ROLL if kind is DecisionKind.COUP_TARGET else DecisionKind.WAR_ROLL
            roll = next((r for r in self.rolls if r.kind is want), None)
            if roll is None:
                return None
            if (kind is DecisionKind.COUP_TARGET and d.actor is Side.USSR and not any(a.payload["country"] == roll.country for a in d.options)
                    and self._reformer_lapsed_in_dll({roll.country})):
                self.known[REFORMER_KNOWN] += 1
                self.diverge("rules", f"The Reformer: the DLL let the USSR coup {roll.country} in Europe (Glasnost gone), the engine bans it", fatal=True)
                return d.options[0]
            return self._pick(d, lambda a: a.payload["country"] == roll.country, f"{kind.value} {roll.country} (from the roll record)")
        if kind is DecisionKind.REALIGNMENT_TARGET:
            roll = next((r for r in self.rolls if r.kind is DecisionKind.REALIGNMENT_ACTOR_ROLL and r.side is self.other), None)
            # A roll recorded after a play of the seat's the engine has not
            # got to yet is that play's: this action stopped before it (a
            # headlined ABM Treaty's realignments, then the action round's).
            later = next((m.seq for m in self.moves[self.other] if m.meaning.meaning is T.Meaning.CARD), None)
            if roll is not None and (later is None or self.roll_seq.get(id(roll), 0) < later):
                return self._pick(d, lambda a: a.payload["country"] == roll.country, f"realignment in {roll.country} (from the roll record)")
            stop = next((a for a in d.options if a.payload["country"] == "stop"), None)
            return stop  # no further roll reported: the AI stopped (or must roll first: None, the DLL is advanced)
        # Influence, Ops or an event's: the countries where the DLL's
        # influence went past the engine's in the right direction since the
        # two last agreed -- the earliest such change first, so that a
        # placement a later action in the same chunk undid is still made
        # (and undone by that action's own records). Order does not matter
        # to legality: a point's cost depends only on the points already
        # placed in its own country.
        if kind is DecisionKind.EVENT_INFLUENCE:
            inf_side, op = Side(d.context["inf_side"]), d.context["op"]
        else:
            inf_side, op = d.actor, "place"
        candidates = sorted((seq, i, a) for i, a in enumerate(d.options)
                            if (seq := self.first_change(a.payload["country"], inf_side, op)) is not None)
        if candidates:
            if self.trace and not self._simulating:
                c = candidates[0][2].payload["country"]
                print(f"  INF {inf_side.value} {op} {c}: engine {self.engine.board.influence[c]}, DLL now {self.influence.get(c)}, "
                      f"history after sync {self.synced_seq}: {[(q, v) for q, v in self.influence_history.get(c, ()) if q > self.synced_seq]}; "
                      f"other candidates {[(q, a.payload['country']) for q, _, a in candidates[1:4]]}")
            return candidates[0][2]
        if not candidates:
            if d.context.get("event") == "De_Stalinization" and op == "place" and self.game.prompt is not None:
                # The DLL is done (at the bot's prompt) and placed fewer than it
                # removed: it forbids the countries just emptied, and nothing
                # else was left. The engine cannot stop short; the game is void.
                self.known["De-Stalinization: the DLL placed fewer than it removed (no destination it allows)"] += 1
                self.diverge("rules", "De-Stalinization: the DLL placed fewer than it removed; the engine must place them all", fatal=True)
                return d.options[0]
            return None  # the DLL shows nothing of the kind (yet): advance it, or it is a desync
        return candidates[0]

    def _answer_choice(self, d: Decision) -> Action | None:
        choices = [a.payload["choice"] for a in d.options]
        if all(c in ids.NUMBER_BY_CARD or c in DECLINES for c in choices):
            if d.context.get("event") == "Star_Wars":
                # The card copied from the discard pile: pushed into the
                # resolve slot from there, no hand involved.
                card = next((c for c in reversed(self._from_discard) if c in choices), None)
                if card is not None:
                    self._from_discard.remove(card)
                    return self._pick(d, lambda a: a.payload["choice"] == card, f"Star Wars copies {card}")
                if self.game.prompt is None:
                    return None
                return next((a for a in d.options if a.payload["choice"] in DECLINES), None)  # at rest with nothing pushed: declined
            # A card taken back from the discard pile (SALT Negotiations):
            # the one that moved there into this seat's hand.
            taken = [(seq, c) for c, (was, now, seq) in self._last_moves.items()
                     if was == int(ffi.ECardLocation.DISCARDED) and now == HAND_LOCATION[self.other] and c in choices]
            if taken:
                card = max(taken)[1]
                del self._last_moves[card]
                return self._pick(d, lambda a: a.payload["choice"] == card, f"took back {card}")
            # Discard a card or decline (Blockade): the card that left the hand
            # -- the bot's own when the choices are its cards (Aldrich Ames
            # Remix: the USSR names a US card).
            owner = self.side if set(choices) & set(self.engine.hands[self.side.value]) else self.other
            card = self.card_that_left(owner, set(choices))
            if card is not None:
                return self._pick(d, lambda a: a.payload["choice"] == card, f"choice {card}")
            # A card the DLL let it discard that the engine does not offer
            # (a threshold the two count differently) is a desync, not a
            # decline: the decline would punish the seat for a card it paid.
            others = [c for c in self.card_that_left_any(self.other) if c not in choices]
            if others:
                # With the hand sizes and the records: a card the engine has no
                # slot for (an Ask Not that emptied the hand the engine cannot
                # see, one card more than it held) is told apart from a
                # threshold difference by them.
                hand = self.engine.hands[self.other.value]
                self.diverge("illegal in engine", f"{self.other.value} {d.context.get('event')}: the DLL discarded {others[0]}, "
                             f"the engine offers {[dict(a.payload) for a in d.options]} (unaccounted exits {others}; "
                             f"{self.other.value} hand: DLL {self.game.hand_count(self._player_of[self.other])}, "
                             f"engine {len(hand)} {sorted(hand)}; recent records: {list(self.recent)})", fatal=True)
                return d.options[0]
            decline = next((a for a in d.options if a.payload["choice"] in DECLINES), None)
            if decline is not None:
                return decline  # the DLL is at rest and no card left the hand: it declined
        if all(c in ids.INDEX_BY_COUNTRY or c in DECLINES for c in choices):
            # Countries: a removal's source (De-Stalinization, the Cuban
            # Missile Crisis defusing) is one where the DLL has less of the
            # chooser's influence than the engine; an addition's target
            # (Independent Reds' match) one where it has more -- never the
            # other way round, or a De-Stalinization that removed two and
            # placed two would go on removing from where it placed. The
            # decline ("Done Removing") once none is left.
            down = sorted((seq, i, a) for i, a in enumerate(d.options)
                          if a.payload["choice"] in ids.INDEX_BY_COUNTRY and (seq := self.first_change(a.payload["choice"], self.other, "remove")) is not None)
            if not down and not str(d.context.get("event")).startswith(("De_Stalinization", CMC_DEFUSE)):
                down = sorted((seq, i, a) for i, a in enumerate(d.options)
                              if a.payload["choice"] in ids.INDEX_BY_COUNTRY and (seq := self.first_change(a.payload["choice"], self.other, "place")) is not None)
            if down:
                return down[0][2]
            decline = next((a for a in d.options if a.payload["choice"] in DECLINES), None)
            return decline  # None: nothing moved yet and no way to decline -- the DLL is advanced
        if set(choices) <= {"raise", "lower", "none"} or all(c.isdigit() for c in choices):
            # A DEFCON choice, read off the DLL's DEFCON -- unless something
            # else moved it in the same chunk (a Summit in the last action
            # round, the turn's end restoring the level it lowered; a coup
            # after a headlined one), which the simulation sorts out: the
            # read is kept only when it reproduces the DLL's state.
            if choices[0].isdigit():
                want = str(self.defcon)
            else:
                want = "raise" if self.defcon > self.engine.defcon else "lower" if self.defcon < self.engine.defcon else "none"
            if want in choices:
                option = next(a for a in d.options if a.payload["choice"] == want)
                if self._simulate_one(option):
                    return option
        return self._simulate(d)

    def _answer_grain_sales(self, d: Decision) -> Action | None:
        """The other seat's Grain Sales: take or return, whichever reproduces
        the DLL's state. The DLL, against its own text ("if returned, use
        this card to conduct Operations"), conducts Grain Sales' Ops *and*
        plays the taken card: when neither choice alone reproduces the
        state but "take" with two more Ops does, the game is void -- known,
        and not a rules gap of this engine."""
        before = len(self.report.divergences)
        take = next(a for a in d.options if a.payload["choice"] == "take")
        # Where the DLL filed the card says which: back in the USSR's hand
        # it was returned, in a pile it was taken and played (taken and
        # still in the US hand: the same, not yet played).
        loc = self.card_loc.get(d.context.get("card"))
        told = ("return" if loc == HAND_LOCATION[Side.USSR] else "take" if loc in PILES or loc == HAND_LOCATION[Side.US] else None)
        if told is not None:
            option = next(a for a in d.options if a.payload["choice"] == told)
            if self._simulate_one(option):
                return option
        action = self._simulate(d)
        if not (self.stop and len(self.report.divergences) > before):
            return action
        failed = self.report.divergences[before:]
        del self.report.divergences[before:]  # lifted while the variants are tried: a fatal on the report stops every simulation step
        if self._simulate_one(take, extra_ops=2, extra_first=True) or self._simulate_one(take, extra_ops=2):
            self.known["Grain Sales: the DLL conducts Grain Sales' Ops as well as playing the taken card (its text: only if returned)"] += 1
            self.diverge("rules", "Grain Sales: the DLL conducted its 2 Ops and played the taken card; the engine plays the card alone", fatal=True)
        else:
            self.report.divergences.extend(failed)
        return action

    # -- choices the records do not name: try each, keep what reproduces the DLL --

    def _simulate_one(self, option: Action, extra_ops: int = 0, extra_first: bool = False) -> bool:
        """Whether `option`, played on a copy of the engine, reproduces the
        DLL's state (`_simulate` for a single option; `extra_ops`: Ops the
        US is granted on top, before the play (`extra_first`) or once it is
        done)."""
        real, queues = self.engine, self._queues()
        divergences, known, last = len(self.report.divergences), self.known.copy(), self._last_state_diff
        self.engine = Engine.deserialize(real.serialize())
        self._simulating += 1
        try:
            return self._try(option, extra_ops=extra_ops, extra_first=extra_first)
        except Exception:
            return False
        finally:
            self._simulating -= 1
            self.engine = real
            self._set_queues(queues)
            del self.report.divergences[divergences:]
            self.known, self._last_state_diff = known, last

    def _simulate(self, d: Decision) -> Action | None:
        """Play each option on a copy of the engine, the rest of the chunk
        answered from the same facts, and keep the ones that leave the copy
        in the DLL's state. Nested choices recurse."""
        matches = []  # (facts left unconsumed, option order, option)
        for i, option in enumerate(d.options):
            real, queues = self.engine, self._queues()
            divergences, known, last = len(self.report.divergences), self.known.copy(), self._last_state_diff
            self.engine = Engine.deserialize(real.serialize())
            self._simulating += 1
            try:
                ok = self._try(option)
                left = len(self.rolls) + sum(len(q) for q in self.moves.values())
                if self.trace:
                    print(f"  SIM {d.kind.value} {dict(option.payload)}: {'matches' if ok else 'no'}, {left} facts left"
                          f"{' (the copy ended)' if self.engine.is_terminal else ''}"
                          f"{'' if ok else ': ' + ('; '.join(self.state_diffs(hands=False)) or 'stopped short')}")
            except Exception as e:  # an option the engine rejects downstream is simply not it
                ok = False
                if self.trace:
                    print(f"  SIM {d.kind.value} {dict(option.payload)}: rejected ({e!r})")
            finally:
                self._simulating -= 1
                self.engine = real
                self._set_queues(queues)
                del self.report.divergences[divergences:]
                self.known, self._last_state_diff = known, last
            if ok:
                matches.append((left, i, option))
        if not matches:
            self.diverge("choice", f"{d.actor.value} {d.kind.value} {d.context.get('event')}: none of {[dict(a.payload) for a in d.options]} "
                         f"reproduces the DLL's state; {'; '.join(self.state_diffs(hands=False)) or 'no state diff before the choice'}", fatal=True)
            return d.options[0]
        # Several may leave the same board (Junta's free Realignment that
        # removed nothing, and declining it): the one that consumed the
        # DLL's records (the rolls) is it -- left queued, those dice would
        # pass for a later action's.
        matches.sort()
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            self.known[f"{d.context.get('event')}: {len(matches)} choices reproduce the DLL's state, the first taken"] += 1
        return matches[0][2]

    def _try(self, option: Action, extra_ops: int = 0, extra_first: bool = False) -> bool:
        self.engine.step(option)
        if extra_ops and extra_first:
            self.engine.push_event_operations(Side.US, extra_ops)  # the DLL's Grain Sales Ops, before the taken card
        elif extra_ops:
            self._extra_ops_pending = extra_ops  # ...or after it: granted at the bot's next decision, in whichever nested simulation gets there
        return self._run_copy()

    def _run_copy(self) -> bool:
        """Play the copy on from the records until the bot's next decision,
        and say whether the DLL's state is reproduced there."""
        while not self.engine.is_terminal:
            d = self.engine.pending_decision
            if d.actor is self.side:
                if self._extra_ops_pending:
                    self.engine.push_event_operations(Side.US, self._extra_ops_pending)
                    self._extra_ops_pending = 0
                    continue
                if d.kind is DecisionKind.EVENT_RESUME and len(d.options) == 1:
                    self.engine.step(d.options[0])  # a forced step, not a decision: the DLL may be past the turn's end already
                    continue
                if (d.kind in (DecisionKind.EVENT_CHOICE, DecisionKind.EVENT_INFLUENCE) and len(d.options) <= 8
                        and not self._trying_bot and self._records_left()):
                    # A choice of the bot's inside the other seat's event that
                    # the DLL resolved without asking (Independent Reds with
                    # one country worth choosing) and played on past: its
                    # state is not this point's. Try the bot's few options,
                    # and judge at the next point the DLL stopped at. (Not
                    # the bot's Ops: the DLL asks those, and they fan out.)
                    return self._try_each(d)
                return not self.state_diffs(hands=d.kind is DecisionKind.ACTION_ROUND_PLAY)
            a = self._answer(d)
            if a is None or self.stop:
                if self.trace:
                    print(f"  SIM   stuck at {d.actor.value} {d.kind.value}: {'no answer' if a is None else self.report.divergences[-1]}")
                return False
            self.engine.step(a)
        if self.game.result is not None:
            # The copy's game ended as the DLL's did: the winner says whether
            # it is the same end (a scoring card held past the turn's end
            # stops the engine before the turn end's bookkeeping, the DLL
            # after it -- the military Ops it reset say nothing).
            return self._sides_by_player.get(self.game.result.winner_id) == self.engine.winner
        return not self.state_diffs(hands=False)

    _trying_bot = False  # inside `_try_each`: one level, no fan-out of fan-outs

    def _records_left(self) -> bool:
        return bool(self.rolls) or any(self.moves.values())

    def _try_each(self, d: Decision) -> bool:
        """Each option of the bot's `d` on its own copy; True at the first
        that reproduces the DLL's state downstream (that copy's records are
        kept, so the caller's count of what was consumed is right)."""
        for option in d.options:
            real, queues = self.engine, self._queues()
            divergences, known, last = len(self.report.divergences), self.known.copy(), self._last_state_diff
            self.engine = Engine.deserialize(real.serialize())
            self._trying_bot = True
            try:
                self.engine.step(option)
                ok = self._run_copy()
            except Exception:
                ok = False
            finally:
                self._trying_bot = False
            if ok:
                if self.trace:
                    print(f"  SIM   the bot's {d.kind.value} {dict(option.payload)} (the DLL chose for it) reproduces the state")
                return True
            self.engine = real
            self._set_queues(queues)
            del self.report.divergences[divergences:]
            self.known, self._last_state_diff = known, last
        return False

    def _queues(self) -> tuple:
        # The rolls keep their identity (`roll_seq` is keyed by it): the
        # deque is copied shallowly, everything else deeply.
        return (list(self.rolls), copy.deepcopy((self.moves, self._last_moves, self._grain, self._forced_mode, self._un_ops,
                                                 self._dealt, self._engine_dealt, self._last_played, self._replay, self._fired,
                                                 self._revealed, self._taken, self._exits, self._exits_before_reshuffle, self.reshuffled,
                                                 self.synced_seq, self._extra_ops_pending, self._handed, self._from_discard)))

    def _set_queues(self, queues: tuple) -> None:
        rolls, rest = queues
        self.rolls = collections.deque(rolls)
        (self.moves, self._last_moves, self._grain, self._forced_mode, self._un_ops,
         self._dealt, self._engine_dealt, self._last_played, self._replay, self._fired, self._revealed, self._taken,
         self._exits, self._exits_before_reshuffle, self.reshuffled, self.synced_seq, self._extra_ops_pending, self._handed,
         self._from_discard) = copy.deepcopy(rest)

    # -- the bot's actions -> the DLL's prompts -------------------------------

    def before_bot_decision(self, d: Decision) -> None:
        self._reveal_hidden_scoring_cards()
        if self.engine.phase == "headline":
            self.mark_setup_done()
        if d.kind in (DecisionKind.HEADLINE_PLAY, DecisionKind.ACTION_ROUND_PLAY):
            self._auto_declined = 0  # a lone decline the engine never asked about (Blockade with nothing to discard)
        if d.kind in (DecisionKind.HEADLINE_PLAY, DecisionKind.ACTION_ROUND_PLAY) and not self.outgoing and not self.moves[self.other]:
            self.compare_state()
            if self.synced_seq == self._seq:
                self._synced_move_seq = self._move_seq  # card moves before this are accounted for (a Five Year Plan discard long ago is not "a card the DLL let it discard")

    def narrow(self, d: Decision) -> Decision:
        """`d` with its options cut down to those the DLL offers for the
        same choice, when it is at that prompt: the two programs do not
        agree on every target (De-Stalinization's sources, docs/WOPR.md),
        and the bot's play has to be legal in both. The difference is
        reported as the differ reports it; the engine's stricter cases
        (options only the DLL has) cannot be added and are left alone."""
        prompt = self.game.prompt
        if self._auto_declined and d.kind is DecisionKind.EVENT_CHOICE and not self.outgoing and (
                prompt is None or self.prompt_side(prompt) is not self.side
                or not self._fits(d, {T.meaning(o).meaning for o in prompt.visible})):
            # The DLL had its lone "Pass" for this choice before the engine
            # asked it (`_reply_or_raise`): the bot's decision is the decline.
            decline = tuple(a for a in d.options if a.payload["choice"] in DECLINES)
            if decline:
                self._auto_declined -= 1
                self._declined_for_dll = d
                return dataclasses.replace(d, options=decline[:1])
        if prompt is None or d.kind not in COUNTRY_KINDS | CARD_KINDS | {DecisionKind.EVENT_CHOICE} or self.prompt_side(prompt) is not self.side:
            return d
        if self.outgoing or self._un_target is not None:
            return d  # the prompt is for an earlier action still to be told
        if d.context.get("event") == CMC_DEFUSE:
            return d  # "skip" is a card play, always there
        meanings = {T.meaning(o).meaning for o in prompt.visible}
        if not self._fits(d, meanings):
            return d
        if d.kind in COUNTRY_KINDS:
            theirs = T.countries_offered(prompt)
            if any(T.meaning(o).meaning is T.Meaning.STOP for o in prompt.visible):
                theirs.add("stop")
            options = tuple(a for a in d.options if a.payload["country"] in theirs)
        elif d.kind is DecisionKind.EVENT_CHOICE:
            if self._grain_card(prompt) is not None:
                return d  # take/return: both always there
            options = tuple(a for a in d.options if self._expressible(prompt, a.payload["choice"], meanings))
        else:
            theirs = T.cards_offered(prompt)
            options = tuple(a for a in d.options if a.payload["card"] in theirs
                            or (a.payload["card"] == PASS_ROUND and _pass_option(prompt) is not None))
        if not options or len(options) == len(d.options):
            return d
        dropped = [dict(a.payload) for a in d.options if a not in options]
        if d.context.get("event") == "De_Stalinization":
            self._count_destalinization()  # documented (docs/WOPR.md)
        elif d.context.get("event") == "Junta":
            self.known["Junta: DLL confines the free Coup/Realignment to the country placed in"] += 1  # documented (docs/WOPR.md)
        else:
            self.diverge("options", f"{self.side.value} {d.kind.value} {d.context.get('event', '')}: the engine offers {dropped} the DLL's "
                         f"{prompt.text!r} {[o.text for o in prompt.visible]} does not; the bot chooses among the rest")
        return dataclasses.replace(d, options=options)

    def _expressible(self, prompt: Prompt, choice: str, meanings: set[T.Meaning]) -> bool:
        if choice in DECLINES and T.Meaning.STOP in meanings:
            return True
        try:
            T.find_choice(prompt, choice, defcon=self.engine.defcon)
        except LookupError:
            return False
        return True

    def note(self, d: Decision, action: Action) -> None:
        """One action of the bot's, as it is made: tell the DLL as far as it
        can be told yet."""
        if self.trace:
            print(f"  BOT {d.actor.value:6s} {d.kind.value} -> {dict(action.payload)}")
        self.report.engine_steps += 1
        self.report.steps += 1
        if d.kind is DecisionKind.EVENT_RESUME:
            return
        if d.kind in (DecisionKind.ACTION_ROUND_PLAY, DecisionKind.HEADLINE_PLAY):
            self._flower_power_check(d.actor, action.payload["card"])
        if d.kind is DecisionKind.PLAY_MODE and self.engine.cards[d.context["card"]].scoring:
            return  # the DLL asks no use for a scoring card
        if (d.kind is DecisionKind.EVENT_CHOICE and d.context.get("event") == CMC_DEFUSE and action.payload["choice"] == "skip"
                and d.context.get("at") != "coup"):
            return  # the DLL lists the defusing among the action round's cards: declining it is playing a card (at a coup it asks, with "Pass")
        self._last_action = (d, action)
        self.outgoing.append((d, action))
        self.flush()

    def flush(self) -> None:
        while not self.stop:
            prompt = self.game.prompt if self.game.prompt is not None else self._pump()
            if prompt is None:
                return
            if self.prompt_side(prompt) is self.other and self.emulate is not None:
                self._choose(prompt, self._reasked(prompt) or self.emulate(self._emulated(prompt)))
                continue
            option = self._reply(prompt)
            if option is None:
                return
            self._choose(prompt, option)

    def _emulated(self, prompt: Prompt) -> Prompt:
        """The prompt as the emulated seat's policy sees it: without the
        Space Race for Ops the engine gives as Ops only (a UN-Intervened
        card's, Missile Envy's exchanged card's, the forced play of Missile
        Envy itself -- the differ counts these under `known`, the AI may
        pick them and desync)."""
        if T.uses_offered(prompt) and any(T.meaning(o).use == T.Use("space_race") for o in prompt.visible) and (
                prompt.text == GRANTED_OPS_PROMPT or self.engine.game_effects.get("missile_envy_forced") == self.other.value):
            return Prompt(prompt.player_id, prompt.text, tuple(o for o in prompt.options if T.meaning(o).use != T.Use("space_race")))
        return prompt

    def _reply(self, prompt: Prompt) -> Option | None:
        """The option of the bot's seat's prompt that its queued actions
        imply, consuming them; None if a further action is needed first."""
        try:
            return self._reply_or_raise(prompt)
        except LookupError as e:
            self.diverge("illegal in Playdek", str(e), fatal=True)
            raise Desync(str(e)) from None

    def _reply_or_raise(self, prompt: Prompt) -> Option | None:
        visible = prompt.visible
        meanings = {T.meaning(o).meaning for o in visible}
        again = self._reasked(prompt)
        if again is not None:
            return again
        if meanings <= UI_ONLY:
            # "Continue with this card" after an event-first resolution.
            return next((o for o in visible if T.meaning(o).meaning is T.Meaning.SWITCH_CARD), visible[0])
        gives = [o for o in visible if o.hint == SelectionHint.GIVE_CARD]
        if gives and len(gives) == len(visible) == 1 and not self.outgoing:
            return gives[0]  # Missile Envy's "Select Card to Give" with one candidate: the engine gave it without asking
        if (meanings <= UI_ONLY | {T.Meaning.STOP} and not (self.outgoing and self._fits(self.outgoing[0][0], meanings))
                and self._grain_card(prompt) is None):
            # A decline the DLL asks alone ("Do Not Discard": Blockade with
            # nothing to discard) where the engine resolved the event without
            # asking the bot -- unless the engine is about to ask a choice it
            # answers (Junta's free Coup/Realignment: the DLL confines it to
            # the country just placed in and offers "Pass" alone when that
            # one has no target, the engine offers the whole region), which
            # `narrow` then cuts down to the decline.
            if not self.outgoing and self._next_bot_decision_fits(meanings):
                return None
            if not self.outgoing:
                # The engine is behind (the bot's own play not yet applied,
                # the other seat's still to resolve) and may ask the bot this
                # choice later (Tear Down This Wall's free Op with no target
                # in the DLL): the decline is sent now and the bot's decision
                # cut down to it when it comes (`narrow`), or forgotten at
                # the bot's next card play.
                self._auto_declined += 1
            return T.find_stop(prompt)
        if self._un_target is not None:
            if any(o.hint == SelectionHint.PLAY_OPPONENT_CARD for o in visible):
                card, self._un_target = self._un_target, None
                return T.find_card(prompt, card)
            if T.Meaning.USE in meanings:
                return T.find_use(prompt, mode="event")  # UN Intervention's own use
        grain = self._grain_card(prompt)
        if grain is not None and self._grain is None:
            self._grain = (grain, False)  # the engine's RANDOM_DISCARD is answered from here; take/return below
        while self.outgoing:
            d, a = self.outgoing[0]
            if self._fits(d, meanings):
                break
            if len(d.options) == 1:
                self.outgoing.popleft()  # a forced step the DLL did not ask about
                continue
            if d.kind is DecisionKind.EVENT_CHOICE and self._declined_for_dll is not None and d.context is self._declined_for_dll.context:
                self.outgoing.popleft()  # the DLL had its "Pass" before the engine asked (`narrow` cut the choice to the decline)
                self._declined_for_dll = None
                continue
            raise LookupError(f"the DLL asks {prompt.text!r} {[(o.text, hex(o.hint)) for o in visible]}; the bot's next action is "
                              f"{d.kind.value} {dict(a.payload)} of {[dict(x.payload) for x in d.options][:12]}")
        if not self.outgoing:
            return None
        d, a = self.outgoing[0]
        p = a.payload
        kind = d.kind
        if kind is DecisionKind.ACTION_ROUND_PLAY and p["card"] == PASS_ROUND:
            option = _pass_option(prompt)
            if option is None:
                raise LookupError(f"the DLL asks {prompt.text!r} {[o.text for o in visible]} without a Pass; the bot passes its extra action round")
            self.outgoing.popleft()
            return option
        if kind is DecisionKind.ACTION_ROUND_PLAY and not self.engine.cards[p["card"]].scoring:
            if len(self.outgoing) < 2:
                return None  # the mode says how the DLL is told (UN Intervention is its own card there)
            d2, a2 = self.outgoing[1]
            if d2.kind is DecisionKind.PLAY_MODE and a2.payload["mode"] == "un_intervention":
                self.outgoing.popleft(), self.outgoing.popleft()
                self._un_target = p["card"]
                self._un_ops[self.side] = True
                return T.find_card(prompt, UN)
        if kind in CARD_KINDS:
            self.outgoing.popleft()
            self._check_cards(d, Move(self.side, prompt, visible[0], T.meaning(visible[0])))
            return T.find_card(prompt, p["card"])
        if kind is DecisionKind.PLAY_MODE:
            mode = p["mode"]
            if mode != "ops":
                self.outgoing.popleft()
                return T.find_use(prompt, mode=mode)
            order = ops_type = None
            following = list(self.outgoing)[1:3]
            for d2, a2 in following:
                if d2.kind is DecisionKind.EVENT_OPS_ORDER:
                    order = a2.payload["order"]
                elif d2.kind is DecisionKind.OPS_TYPE:
                    ops_type = a2.payload["type"]
            if order == "event_first":
                self.outgoing.popleft(), self.outgoing.popleft()
                return T.find_use(prompt, mode="ops", event_first=True)
            if ops_type is None:
                return None  # the Ops type is part of the same Playdek option
            for _ in range(2 + (order is not None)):
                self.outgoing.popleft()
            return T.find_use(prompt, mode="ops", ops_type=ops_type)
        if kind is DecisionKind.OPS_TYPE:
            self.outgoing.popleft()
            return T.find_use(prompt, mode="ops", ops_type=p["type"])
        if kind is DecisionKind.EVENT_OPS_ORDER:
            self.outgoing.popleft()  # consumed with its PLAY_MODE normally; on its own it has no prompt
            return self._reply_or_raise(prompt)
        if kind in COUNTRY_KINDS:
            self.outgoing.popleft()
            self._check_countries(d, Move(self.side, prompt, visible[0], T.meaning(visible[0])))
            if kind is DecisionKind.REALIGNMENT_TARGET and p["country"] == "stop":
                return T.find_stop(prompt)
            return T.find_country(prompt, p["country"])
        if kind is DecisionKind.EVENT_CHOICE:
            self.outgoing.popleft()
            choice = p["choice"]
            if grain is not None:
                self._grain = None
                if choice == "take":
                    return next(o for o in visible if o.hint == SelectionHint.SWITCH_CARD)
                self._taken.pop(grain, None)  # returned: its owner's own card again
                return T.find_stop(prompt)
            if choice in DECLINES and T.Meaning.STOP in meanings:
                return T.find_stop(prompt)
            return T.find_choice(prompt, choice, defcon=self.engine.defcon)
        raise LookupError(f"no translation for the bot's {kind.value} {dict(p)} at {prompt.text!r}")

    def _next_bot_decision_fits(self, meanings: set[T.Meaning]) -> bool:
        """Whether the decision the engine will ask the bot next, once the
        bot's latest action is applied, can be answered at a prompt offering
        `meanings` (a copy of the engine takes the step)."""
        if self._last_action is None or self.engine.pending_decision is not self._last_action[0]:
            return False
        copy = Engine.deserialize(self.engine.serialize())
        try:
            copy.step(self._last_action[1])
        except Exception:
            return False
        nxt = copy.pending_decision
        return nxt is not None and nxt.actor is self.side and self._fits(nxt, meanings)

    @staticmethod
    def _fits(d: Decision, meanings: set[T.Meaning]) -> bool:
        k = d.kind
        if k in CARD_KINDS or k is DecisionKind.ACTION_ROUND_PLAY:
            return T.Meaning.CARD in meanings
        if k in (DecisionKind.PLAY_MODE, DecisionKind.OPS_TYPE, DecisionKind.EVENT_OPS_ORDER):
            return T.Meaning.USE in meanings
        if k in COUNTRY_KINDS:
            return T.Meaning.COUNTRY in meanings or (k is DecisionKind.REALIGNMENT_TARGET and T.Meaning.STOP in meanings)
        if k is DecisionKind.EVENT_CHOICE:
            # By what the choices are: the DLL resolves a single country or
            # card on its own, and its next prompt is then something else.
            choices = {a.payload.get("choice") for a in d.options}
            if choices <= set(ids.INDEX_BY_COUNTRY) | DECLINES:
                return bool(meanings & {T.Meaning.COUNTRY, T.Meaning.STOP})
            if choices <= set(ids.NUMBER_BY_CARD) | DECLINES:
                return bool(meanings & {T.Meaning.CARD, T.Meaning.STOP, T.Meaning.SWITCH_CARD})
            return bool(meanings & {T.Meaning.CHOICE, T.Meaning.STOP, T.Meaning.SWITCH_CARD})
        return False

    @staticmethod
    def _grain_card(prompt: Prompt) -> str | None:
        """Grain Sales' "Play <the drawn card>?" / "Return It": the card."""
        return Bridge.grain_card(prompt)

    # -- the end ----------------------------------------------------------

    def finish_match(self) -> Report:
        """The engine is done (or desynced): let the DLL finish too, then
        stamp the report."""
        while self.game.result is None and not self.stop:
            prompt = self.game.prompt if self.game.prompt is not None else self._pump()
            if prompt is None:
                break
            if self.prompt_side(prompt) is self.other and self.emulate is not None:
                self._choose(prompt, self.emulate(prompt))
                continue
            try:
                option = self._reply(prompt)
            except Desync:
                break
            if option is None:
                if self.engine.is_terminal and self._complete_for_dll(prompt):
                    continue
                if self._trapped_held_scoring_card():
                    # The engine's turn ended with a scoring card held by a
                    # seat in Quagmire / Bear Trap, which loses it the game
                    # (the card: it may play only scoring cards; the turn's
                    # end: a held one loses). The DLL lets the trapped seat
                    # neither play the card (ffi.TRAP_SCORING_CARD) nor
                    # lose for holding it, and carries it into the next
                    # turn. The engine cannot play on: known, and void.
                    loser = self.engine.winner.opponent
                    self.known["trapped seat's held scoring card: the engine ends the game at the turn's end, the DLL carries the card over"] += 1
                    self.diverge("rules", f"held scoring card: the engine ends the turn with {loser.value} trapped and holding a scoring card "
                                 f"({self.engine.winner.value} wins), the DLL plays on to {prompt.text!r}", fatal=True)
                    break
                self.diverge("game over", f"the engine is over ({self.engine.winner}) while the DLL still asks {prompt.text!r}", fatal=True)
                break
            self._choose(prompt, option)
        report = self.finish()
        if self._completed_for_dll and self.game.result is not None and self._sides_by_player.get(self.game.result.winner_id) is not self.engine.winner:
            self.diverge("game over", f"the engine's game ended ({self.engine.winner}) during the bot's action, the DLL's after it with another result ({self.game.result})", fatal=True)
        return report

    def _trapped_held_scoring_card(self) -> bool:
        """The engine's game ended on a scoring card held past the turn by a
        seat the DLL has in a trap (Quagmire holds the US, Bear Trap the USSR)."""
        e = self.engine
        if not e.is_terminal or e.winner is None or getattr(e, "_game_over_reason", None) != "held_scoring_card":
            return False
        loser = e.winner.opponent
        return any(e.game_effects.get(key) and side is loser for key, side in e._TRAP_KEYS.items())

    def _complete_for_dll(self, prompt: Prompt) -> bool:
        """The engine's game ended in the middle of the bot's action where
        the DLL's goes on to the action's end (We Will Bury You's 3 VP: the
        engine pays them the moment the US plays another card, the DLL
        once that play is done): finish the action in the DLL with the
        plainest choices, so that its result can be compared. A new action
        (a card prompt) is the DLL playing on, not this."""
        if self.prompt_side(prompt) is not self.side:
            return False
        meanings = {T.meaning(o).meaning for o in prompt.visible}
        if T.Meaning.CARD in meanings:
            return False
        pick = None
        if T.Meaning.USE in meanings:
            try:
                pick = T.find_use(prompt, mode="ops", ops_type="influence")
            except LookupError:
                pick = next((o for o in prompt.visible if T.meaning(o).meaning is T.Meaning.USE), None)
        if pick is None:
            pick = next((o for o in prompt.visible if T.meaning(o).meaning not in UI_ONLY), None)
        if pick is None:
            return False
        if not self._completed_for_dll:
            self.known["the engine's game ended at the card play, the DLL's after the action (We Will Bury You's VP)"] += 1
        self._completed_for_dll = True
        self._choose(prompt, pick)
        return True


def play_match(pd: Playdek, player: Player, *, seed: int, side: Side, difficulty: AIDifficulty = AIDifficulty.HARD,
               emulate: Policy | None = None, log_path: str | None = None, trace: bool = False,
               max_divergences: int = 40) -> MatchResult:
    """One game of `player` on `side` against Playdek's AI (or `emulate`),
    refereed by the engine, optionally logged as a replay (`log_path`)."""
    start = time.monotonic()
    op = PlaydekOperator(pd, seed=seed, side=side, difficulty=difficulty, emulate=emulate, trace=trace, max_divergences=max_divergences)
    desync = False
    try:
        play_game(op.engine, op.players(player), log_path=log_path)
    except Desync:
        desync = True
    except PlaydekEnded:
        pass  # the DLL's result stands (held scoring card in the hand the engine cannot see)
    report = op.finish_match()
    fatal = [d for d in report.divergences if d.fatal]
    void = fatal[0].detail if fatal and fatal[0].what == "rules" else None  # later fatals are its consequences
    desync = (desync or bool(fatal)) and void is None
    result = op.game.result
    pd_winner = op._sides_by_player.get(result.winner_id) if result is not None else None
    return MatchResult(
        seed=seed, side=side.value, difficulty="hotseat" if emulate is not None else difficulty.name.lower(),
        winner=op.engine.winner.value if op.engine.winner is not None else None,
        playdek_winner=pd_winner.value if pd_winner is not None else None,
        win_type=result.win_type.name if result is not None else "none",
        score=result.score if result is not None else op.game.score,
        turn=op.engine.turn, desync=desync, void=void, divergences=[str(d) for d in report.divergences],
        prompts=report.prompts, seconds=time.monotonic() - start,
    )
