"""Two rules engines, one game: Playdek's DLL drives, struggler replays.

A hotseat Playdek game (both seats ours) is played by a policy; every
option it picks becomes, through `translate`, the answers the struggler
engine will ask for -- the engine runs in physical mode, so its dice and
its deals come from the DLL's roll and card-location events too. The
engine is driven on demand: whenever it asks something the queues cannot
yet answer, the DLL is advanced first. At every point where both engines
are waiting on the same player, their states are compared; whenever the
engine asks a decision, the options the DLL offered for the same choice
are compared with the engine's. Every disagreement is a `Divergence`.

    python -m wopr.playdek.lockstep --games 3 --seed 1

One hand is hidden from the engine (physical mode's `physical_side`): its
cards are only learned as they are played, so its card-choice legality is
checked coarsely. Alternate `physical_side` across games to cover both.
"""

from __future__ import annotations

import argparse
import collections
import random
from dataclasses import dataclass, field
from typing import Callable

from struggler.engine import Engine
from struggler.engine.core import HIDDEN_CARD
from struggler.engine.types import Action, CardSide, Decision, DecisionKind, Side
from wopr.playdek import ffi, ids, translate as T
from wopr.playdek.ffi import EventType
from wopr.playdek.game import GameEvent, Option, Playdek, PlaydekGame, Prompt

ROLL_KINDS = {
    DecisionKind.COUP_ROLL, DecisionKind.WAR_ROLL, DecisionKind.REALIGNMENT_ACTOR_ROLL,
    DecisionKind.REALIGNMENT_OPPONENT_ROLL, DecisionKind.SPACE_RACE_ROLL, DecisionKind.QUAGMIRE_ROLL,
    DecisionKind.CONTEST_ROLL,
}
CARD_KINDS = {
    DecisionKind.HEADLINE_PLAY, DecisionKind.ACTION_ROUND_PLAY, DecisionKind.QUAGMIRE_DISCARD,
    DecisionKind.HELD_CARD_DISCARD,
}
COUNTRY_KINDS = {
    DecisionKind.PLACE_INFLUENCE, DecisionKind.EVENT_INFLUENCE, DecisionKind.COUP_TARGET,
    DecisionKind.REALIGNMENT_TARGET, DecisionKind.WAR_TARGET,
}
CHINA = "The_China_Card"
UN = "UN_Intervention"
UI_ONLY = {T.Meaning.CANCEL, T.Meaning.SWITCH_CARD, T.Meaning.BLANK}  # never a move; the policy never picks them

Policy = Callable[[Prompt], Option]


@dataclass
class Divergence:
    game: int
    step: int
    what: str
    detail: str
    fatal: bool = False

    def __str__(self) -> str:
        return f"[game {self.game} step {self.step}] {'FATAL ' if self.fatal else ''}{self.what}: {self.detail}"


@dataclass
class Move:
    """One Playdek choice and what it answers on the struggler side."""

    side: Side
    prompt: Prompt
    option: Option
    meaning: T.OptionMeaning
    use_spent: bool = False  # a USE answers up to three engine decisions


@dataclass
class Report:
    game: int
    steps: int = 0
    engine_steps: int = 0
    prompts: int = 0
    divergences: list[Divergence] = field(default_factory=list)
    playdek_result: str = ""
    engine_result: str = ""

    @property
    def ok(self) -> bool:
        return not self.divergences


def random_policy(rng: random.Random) -> Policy:
    def pick(prompt: Prompt) -> Option:
        playable = [o for o in prompt.visible if T.meaning(o).meaning not in UI_ONLY]
        return rng.choice(playable or list(prompt.visible))

    return pick


class Lockstep:
    def __init__(self, pd: Playdek, *, game_no: int, seed: int, physical_side: Side, policy: Policy,
                 max_divergences: int = 40, trace: bool = False) -> None:
        self.pd = pd
        self.trace = trace
        self.report = Report(game_no)
        self.policy = policy
        self.max_divergences = max_divergences
        self.game: PlaydekGame = pd.new_game(local_side=Side.USSR, ai_difficulty=None, seed=seed)
        self.engine = Engine.new_game(seed=seed, include_optional=False, physical_mode=True, physical_side=physical_side)
        self.moves: dict[Side, collections.deque[Move]] = {Side.USSR: collections.deque(), Side.US: collections.deque()}
        self.rolls: collections.deque[T.Roll] = collections.deque()
        self.influence: dict[str, tuple[int, int]] = {}  # country -> (ussr, us) per the DLL
        self.card_loc: dict[str, int] = {}  # card -> ECardLocation per the DLL
        self._last_moves: dict[str, tuple[int, int, int]] = {}  # card -> (from, to, sequence no.) of its latest move
        self._move_seq = 0
        self.defcon = 5
        self.vp = 0
        self.milops = (0, 0)
        self.space = (0, 0)
        self._events_seen = 0
        self._last_prompt: Prompt | None = None
        self._held_first: Move | None = None
        self._last_played: dict[Side, str | None] = {Side.USSR: None, Side.US: None}
        self._forced_mode: dict[Side, str | None] = {Side.USSR: None, Side.US: None}  # the next PLAY_MODE, when a lookahead settled it
        self._sides_by_player = self.game.sides  # player id -> Side
        self._index_side = dict(self.game.sides)  # the "player index" of roll events is the seat's id
        # Roll records the DLL has emitted once and will emit again: it
        # re-emits a whole action's records, verbatim and in order, when the
        # action is committed (see `_absorb`).
        self._replay: collections.deque[GameEvent] = collections.deque()
        self._last_state_diff = ""
        self.known: collections.Counter[str] = collections.Counter()

    # -- divergences ------------------------------------------------------

    def diverge(self, what: str, detail: str, *, fatal: bool = False) -> None:
        self.report.divergences.append(Divergence(self.report.game, self.report.steps, what, detail, fatal))

    @property
    def stop(self) -> bool:
        return any(d.fatal for d in self.report.divergences) or len(self.report.divergences) >= self.max_divergences

    # -- the DLL side -----------------------------------------------------

    def _absorb_events(self) -> None:
        for ev in self.game.events[self._events_seen:]:
            self._absorb(ev)
        self._events_seen = len(self.game.events)

    def _absorb(self, ev: GameEvent) -> None:
        f = ev.fields
        if ev.kind == EventType.COUNTRY_INFLUENCE:
            self.influence[ids.country_id(f["id"])] = (f["ussr_influence"], f["us_influence"])
        elif ev.kind == EventType.DEFCON_LEVEL and not f["isSimulating"]:
            self.defcon = f["defcon_level"]
        elif ev.kind == EventType.VP_TRACK:
            self.vp = f["vp_track"]
            if self.trace:
                print(f"  EV  VP_TRACK {self.vp}")
        elif ev.kind == EventType.MILITARY_OPS:
            self.milops = (f["ussr_milops"], f["us_milops"])
        elif ev.kind == EventType.SPACE_RACE_TRACK:
            self.space = (f["ussr_space"], f["us_space"])
        elif ev.kind == EventType.CARD_LOCATION:
            # Absolute, like influence: a card is dealt when its location
            # becomes a hand it was not in (the DLL re-emits records at commit).
            loc = f["location"]
            try:
                card = ids.card_id(f["id"])
            except KeyError:
                if loc in (int(ffi.ECardLocation.USSRHAND), int(ffi.ECardLocation.USHAND)):
                    self.diverge("unknown card", f"Playdek card {f['id']} entered a hand", fatal=True)
                return
            was = self.card_loc.get(card)
            self.card_loc[card] = loc
            if was is not None and was != loc:
                self._move_seq += 1
                self._last_moves[card] = (was, loc, self._move_seq)
            if self.trace and was != loc:
                print(f"  EV  card {card}: {ffi.ECardLocation(was).name if was is not None else '?'} -> {ffi.ECardLocation(loc).name}")
        else:
            rolls = T.rolls_from_event(ev, self._index_side)
            if not rolls:
                return
            # The DLL emits an action's records twice: once as each choice
            # is made (the preview) and again, verbatim and in order, when
            # the action is committed -- which is at the next action
            # boundary, so the re-emission may arrive pumps later, after the
            # engine has consumed the originals (a headline's realignments
            # are replayed after the first action round's). Roll records
            # are matched off a FIFO: the oldest un-replayed one equal to
            # this record is its replay, anything else is a new roll.
            if self._replay and self._replay[0] == ev:
                self._replay.popleft()
                return
            self._replay.append(ev)
            if self.trace:
                print(f"  EV  {ev}")
            self.rolls.extend(rolls)

    def advance_playdek(self) -> bool:
        """Let the policy answer one DLL prompt; queue what it implies.
        Returns False when the DLL's game is over."""
        prompt = self.game.pump(idle_limit=30)
        self._absorb_events()
        # Both engines at rest: the DLL has applied the previous choice and
        # waits for the next; give the engine everything that implies, then
        # compare.
        self.drain_engine()
        if prompt is None:
            return False
        if T.cards_offered(prompt) and not any(self.moves.values()):
            self.compare_state()  # both sides between actions: the DLL asks for a card, the engine too, nothing queued between them
        if self._held_first is not None:
            # The DLL re-asks the very first hotseat prompt and drops the first
            # answer: only queue it once the next prompt proves it was taken.
            same = (prompt.player_id, prompt.text, tuple(prompt.options)) ==                 (self._held_first.prompt.player_id, self._held_first.prompt.text, tuple(self._held_first.prompt.options))
            if not same:
                self.moves[self._held_first.side].append(self._held_first)
            self._held_first = None
        self._last_prompt = prompt
        self.report.prompts += 1
        side = self._sides_by_player[prompt.player_id]
        # Simultaneous picks (turn 2+ headlines) arrive under the local seat's
        # id whoever is choosing; the cards on offer say whose hand it is.
        hands = {int(ffi.ECardLocation.USSRHAND): Side.USSR, int(ffi.ECardLocation.USHAND): Side.US}
        owners = {hands.get(self.card_loc.get(T.meaning(o).card, -1)) for o in prompt.visible if T.meaning(o).meaning is T.Meaning.CARD}
        owners.discard(None)
        if len(owners) == 1:
            side = owners.pop()
        option = self.policy(prompt)
        m = T.meaning(option)
        if m.meaning is T.Meaning.UNKNOWN:
            self.diverge("unknown option", f"{prompt.text!r} -> {option.text!r} hint={option.hint:#x}")
        move = Move(side, prompt, option, m)
        if m.meaning in UI_ONLY:
            pass  # UI steps ("continue with this card" after an event-first resolution): nothing for the engine
        elif self.report.prompts == 1:
            self._held_first = move
        else:
            self.moves[side].append(move)
        if m.meaning is T.Meaning.CARD:
            self._last_played[side] = m.card
        if self.trace:
            print(f"  PD  {side.value:4s} {prompt.text!r} -> {option.text!r}")
        self.game.choose(option.index)
        return True

    # -- the engine side --------------------------------------------------

    def _answer(self, d: Decision) -> Action | None:
        """The engine's action for `d` from the queues, or None if the DLL
        must be advanced first."""
        if d.kind is DecisionKind.EVENT_RESUME:
            return d.options[0]
        if d.actor is Side.CHANCE:
            if d.kind is DecisionKind.DEAL_CARD:
                # A card the DLL has in that hand and the engine has not been
                # told about yet (the DLL deals, undoes and re-deals at commit).
                side = Side(d.context["side"])
                hand_loc = int(ffi.ECardLocation.USSRHAND if side is Side.USSR else ffi.ECardLocation.USHAND)
                dll_hand = {c for c, loc in self.card_loc.items() if loc == hand_loc and c != CHINA}
                missing = dll_hand - set(self.engine.hands[side.value])
                if not missing:
                    return None
                offered = [a for a in d.options if a.payload["card"] in missing]
                if not offered:
                    self.diverge("illegal in engine", f"deal {sorted(missing)} to {side.value}: not in the engine's hidden pool", fatal=True)
                    return d.options[0]
                return offered[0]
            if d.kind is DecisionKind.CONTEST_ROLL:
                # One die per side; the engine asks the sponsor's first, then
                # the defender's, and says who sponsors.
                key = next(k for k in ("sponsor_roll", "defender_roll") if k in d.options[0].payload)
                sponsor = Side(d.context["sponsor"])
                want = sponsor if key == "sponsor_roll" else sponsor.opponent
                roll = next((r for r in self.rolls if r.kind is d.kind and r.side is want), None)
                if roll is None:
                    return None
                self.rolls.remove(roll)
                return self._pick(d, lambda a: a.payload.get(key) == roll.payload["value"], f"roll {roll}")
            if d.kind in ROLL_KINDS:
                country = d.context.get("country")
                roll = next((r for r in self.rolls if r.kind is d.kind and (country is None or r.country in (None, country))), None)
                if roll is None:
                    return None
                self.rolls.remove(roll)
                key = next(iter(roll.payload))
                return self._pick(d, lambda a: a.payload.get(key) == roll.payload[key], f"roll {roll}")
            if d.kind is DecisionKind.RANDOM_DISCARD:
                # The DLL drew the card: the latest one to leave that hand for
                # the discard (or removed) pile that the engine still has
                # there. Cards played from the hand leave it the same way,
                # hence "latest" and "still offered".
                owner = Side(d.context["owner"])
                hand_loc = int(ffi.ECardLocation.USSRHAND if owner is Side.USSR else ffi.ECardLocation.USHAND)
                piles = (int(ffi.ECardLocation.DISCARDED), int(ffi.ECardLocation.REMOVED))
                offered = {a.payload["card"] for a in d.options}
                gone = [(seq, c) for c, (was, now, seq) in self._last_moves.items()
                        if was == hand_loc and now in piles and c in offered]
                if not gone:
                    return None
                card = max(gone)[1]
                del self._last_moves[card]
                return self._pick(d, lambda a: a.payload["card"] == card, f"random discard {card}")
            self.diverge("unsupported", f"CHANCE {d.kind.value}", fatal=True)
            return d.options[0]
        side = d.actor
        q = self.moves[side]
        if len(d.options) == 1 and q and not self._compatible(d, q[0]):
            # A forced step the DLL did not ask about (a single legal target):
            # known only once this side's next prompt turned out to be something else.
            return d.options[0]
        if not q:
            return None
        mv = q[0]
        m = mv.meaning
        if m.meaning is T.Meaning.STOP:
            q.popleft()
            if d.kind is DecisionKind.EVENT_CHOICE:  # "Done Removing" / "Do Not Discard": the choice that is not a card or country
                return self._pick(d, lambda a: a.payload.get("choice") not in ids.NUMBER_BY_CARD and a.payload.get("choice") not in ids.INDEX_BY_COUNTRY, "decline")
            return self._pick(d, lambda a: a.payload.get("country") == "stop", "stop")
        if d.kind is DecisionKind.ACTION_ROUND_PLAY and m.meaning is T.Meaning.CARD and m.card == UN:
            # Playdek plays UN Intervention itself ("Play Event", then "Select
            # Opponent Event Card to Play"); the engine plays the opponent's
            # card with mode "un_intervention". Look ahead to the use, and
            # to the card if the use is the event.
            if len(q) < 2:
                return None
            use = q[1].meaning
            if use.meaning is T.Meaning.USE and use.use.mode == "event":
                if len(q) < 3:
                    return None
                target = q[2].meaning
                if target.meaning is not T.Meaning.CARD:
                    self.diverge("decision mismatch", f"UN Intervention's event: expected the opponent's card, Playdek's move is {q[2].prompt.text!r} -> {q[2].option.text!r}", fatal=True)
                    return d.options[0]
                q.popleft(), q.popleft(), q.popleft()
                self._check_cards(d, mv)
                self._forced_mode[side] = "un_intervention"
                return self._pick(d, lambda a: a.payload["card"] == target.card, f"card {target.card} (UN Intervention)")
        if d.kind in CARD_KINDS and m.meaning is T.Meaning.CARD:
            q.popleft()
            self._check_cards(d, mv)
            return self._pick(d, lambda a: a.payload["card"] == m.card, f"card {m.card}")
        if d.kind is DecisionKind.PLAY_MODE and self._forced_mode[side] is not None:
            mode, self._forced_mode[side] = self._forced_mode[side], None
            return self._pick(d, lambda a: a.payload["mode"] == mode, f"mode {mode}")
        if d.kind in COUNTRY_KINDS and m.meaning is T.Meaning.COUNTRY:
            q.popleft()
            self._check_countries(d, mv)
            return self._pick(d, lambda a: a.payload["country"] == m.country, f"country {m.country}")
        if m.meaning is T.Meaning.USE:
            use = m.use
            if d.kind is DecisionKind.PLAY_MODE:
                if use.mode != "ops":
                    q.popleft()
                return self._pick(d, lambda a: a.payload["mode"] == use.mode, f"mode {use.mode}")
            if d.kind is DecisionKind.EVENT_OPS_ORDER:
                order = "event_first" if use.event_first else "ops_first"
                if use.event_first:
                    q.popleft()
                return self._pick(d, lambda a: a.payload["order"] == order, f"order {order}")
            if d.kind is DecisionKind.OPS_TYPE:
                if use.ops_type is None:
                    # "Resolve Event First" but the engine never asked the order:
                    # its event did not fire. The Ops use is the DLL's next "Select Use" answer.
                    q.popleft()
                    self.diverge("event order", f"{side.value} played {self._last_played[side]} event-first in Playdek; "
                                 "the engine asked no order (no event to fire?)")
                    return self._answer(d)
                q.popleft()
                return self._pick(d, lambda a: a.payload["type"] == use.ops_type, f"ops type {use.ops_type}")
        if d.kind is DecisionKind.EVENT_CHOICE and m.meaning is T.Meaning.CARD:
            q.popleft()  # an either/or whose choices are cards (Blockade's discard)
            return self._pick(d, lambda a: a.payload.get("choice") == m.card, f"choice {m.card}")
        if d.kind is DecisionKind.EVENT_CHOICE and m.meaning is T.Meaning.COUNTRY:
            # An either/or whose choices are countries (De-Stalinization's source).
            q.popleft()
            return self._pick(d, lambda a: a.payload.get("choice") == m.country, f"choice {m.country}")
        if d.kind is DecisionKind.EVENT_CHOICE and m.meaning is T.Meaning.CHOICE:
            q.popleft()
            return self._pick_choice(d, mv)
        self.diverge(
            "decision mismatch",
            f"engine asks {d.actor.value} {d.kind.value} (context {dict(d.context)}); "
            f"Playdek's next move is {mv.prompt.text!r} -> {mv.option.text!r} ({m.meaning.name})",
            fatal=True,
        )
        q.popleft()
        return d.options[0]

    @staticmethod
    def _compatible(d: Decision, mv: Move) -> bool:
        m = mv.meaning.meaning
        return ((d.kind in CARD_KINDS and m is T.Meaning.CARD) or (d.kind is DecisionKind.EVENT_CHOICE and m in (T.Meaning.CARD, T.Meaning.STOP)) or (d.kind in COUNTRY_KINDS and m is T.Meaning.COUNTRY)
                or (d.kind in (DecisionKind.PLAY_MODE, DecisionKind.EVENT_OPS_ORDER, DecisionKind.OPS_TYPE) and m is T.Meaning.USE)
                or (d.kind is DecisionKind.EVENT_CHOICE and m in (T.Meaning.CHOICE, T.Meaning.COUNTRY))
                or (d.kind is DecisionKind.REALIGNMENT_TARGET and m is T.Meaning.STOP))

    def _pick(self, d: Decision, pred, what: str) -> Action:
        for a in d.options:
            if pred(a):
                return a
        self.diverge("illegal in engine", f"{d.actor.value} {d.kind.value}: Playdek chose {what}, engine offers "
                     f"{[dict(a.payload) for a in d.options][:12]}{'...' if len(d.options) > 12 else ''}", fatal=True)
        return d.options[0]

    def _pick_choice(self, d: Decision, mv: Move) -> Action:
        if mv.meaning.choice is not None:
            return self._pick(d, lambda a: a.payload.get("choice") == mv.meaning.choice, f"choice {mv.meaning.choice}")
        # No shared vocabulary for an event's either/or: match the option
        # whose payload shares the most words with Playdek's label.
        words = set(mv.option.text.lower().split())
        best = max(d.options, key=lambda a: len(words & set(str(a.payload.get("choice", "")).lower().replace("_", " ").split())))
        self.diverge("choice by words", f"{mv.prompt.text!r} {mv.option.text!r} -> {dict(best.payload)} of {[dict(a.payload) for a in d.options]}")
        return best

    def _check_cards(self, d: Decision, mv: Move) -> None:
        if d.actor is self.engine.physical_side:
            return  # the engine cannot see that hand; its options are the whole pool
        theirs = T.cards_offered(mv.prompt)
        ours = {a.payload["card"] for a in d.options} - {HIDDEN_CARD}
        if theirs != ours:
            self.diverge("card options", f"{d.actor.value} {d.kind.value} {mv.prompt.text!r}: only Playdek {sorted(theirs - ours)}, only engine {sorted(ours - theirs)}")

    def _check_countries(self, d: Decision, mv: Move) -> None:
        theirs = T.countries_offered(mv.prompt)
        if any(T.meaning(o).meaning is T.Meaning.STOP for o in mv.prompt.visible):
            theirs.add("stop")  # "No More Realignment" is the engine's {"country": "stop"}
        ours = {a.payload["country"] for a in d.options}
        if d.kind is DecisionKind.EVENT_INFLUENCE and d.context.get("event") == "De_Stalinization" and ours > theirs:
            # Known: the DLL will not relocate influence back into a country it
            # was just removed from; the card text has no such clause.
            self.known["De-Stalinization: DLL excludes the source countries"] += 1
            ours = theirs
        if theirs != ours:
            self.diverge("country options", f"{d.actor.value} {d.kind.value} {mv.prompt.text!r}: only Playdek {sorted(theirs - ours)}, only engine {sorted(ours - theirs)}")

    # -- state comparison -------------------------------------------------

    def compare_state(self) -> None:
        e = self.engine
        if e.phase in ("idle", "predeal", "setup"):
            return  # the DLL deals after the setup placements, the engine before: compare from the first headline on
        if e.pending_decision is None or e.pending_decision.kind not in CARD_KINDS:
            return  # mid-action on the engine's side
        diffs = []
        for country, (ussr, us) in self.influence.items():
            mine = e.board.influence.get(country)
            if mine is None:
                continue
            if (mine["USSR"], mine["US"]) != (ussr, us):
                around = ", ".join(f"{n} {self.influence.get(n, (0, 0))}" for n in sorted(e.board.neighbors(country)))
                diffs.append(f"{country}: Playdek USSR {ussr}/US {us}, engine USSR {mine['USSR']}/US {mine['US']} [Playdek neighbours: {around}]")
        if e.defcon != self.defcon:
            diffs.append(f"DEFCON Playdek {self.defcon}, engine {e.defcon}")
        if e.vp != self.vp:
            diffs.append(f"VP Playdek {self.vp} (getter {self.game.score}), engine {e.vp}")
        if (e.military_ops["USSR"], e.military_ops["US"]) != self.milops:
            diffs.append(f"mil ops Playdek {self.milops}, engine {(e.military_ops['USSR'], e.military_ops['US'])}")
        if (e.space_race["USSR"], e.space_race["US"]) != self.space:
            diffs.append(f"space race Playdek {self.space}, engine {(e.space_race['USSR'], e.space_race['US'])}")
        if e.pending_decision.kind is DecisionKind.ACTION_ROUND_PLAY:  # headline picks leave hands at different moments
            for side in (Side.USSR, Side.US):
                pd_count = self.game.hand_count(0 if side is Side.US else 1)  # LOCAL_ID 0 is US in hotseat
                mine = len(e.hands[side.value])
                if pd_count != mine:
                    diffs.append(f"{side.value} hand size Playdek {pd_count}, engine {mine} {sorted(e.hands[side.value])}")
                if side is not e.physical_side:
                    # The hand the engine can see: its contents, card by card.
                    hand_loc = int(ffi.ECardLocation.USSRHAND if side is Side.USSR else ffi.ECardLocation.USHAND)
                    theirs = {c for c, loc in self.card_loc.items() if loc == hand_loc} - {CHINA}
                    ours = set(e.hands[side.value]) - {CHINA}
                    if theirs != ours:
                        diffs.append(f"{side.value} hand: only Playdek {sorted(theirs - ours)}, only engine {sorted(ours - theirs)}")
        text = "; ".join(diffs)
        if diffs and text != self._last_state_diff:
            self.diverge("state", f"turn {e.turn} AR {e.action_round}: " + text)
        self._last_state_diff = text

    # -- the loop ---------------------------------------------------------

    def drain_engine(self) -> None:
        """Step the engine as far as the queued facts allow."""
        while not self.stop and not self.engine.is_terminal:
            d = self.engine.pending_decision
            action = self._answer(d)
            if action is None:
                return
            if self.trace:
                print(f"  ENG {d.actor.value:6s} {d.kind.value} -> {dict(action.payload)}")
            self.engine.step(action)
            self.report.engine_steps += 1
            self.report.steps += 1

    def run(self, *, max_steps: int = 5000) -> Report:
        while not self.stop and self.report.steps < max_steps and not self.engine.is_terminal:
            if self.game.result is not None:
                self.drain_engine()
                if not self.engine.is_terminal:
                    r = self.game.result
                    if r.win_type is ffi.GameOverType.HELD_CARDS:
                        loser = self._sides_by_player.get(r.winner_id).opponent
                        if loser is self.engine.physical_side:
                            self.known["held scoring card in the hand the engine cannot see"] += 1
                        else:
                            self.diverge("rules", f"Playdek ends the game at the end of turn {self.engine.turn}: {loser.value} held a "
                                         "scoring card; the engine plays on")
                    else:
                        e = self.engine
                        self.diverge("game over", f"Playdek's game ended ({r.win_type.name}, {self._sides_by_player.get(r.winner_id)} wins, "
                                     f"score {r.score}) while the engine still asks {e.pending_decision.kind.value} "
                                     f"at DEFCON {e.defcon}, VP {e.vp}, turn {e.turn} AR {e.action_round}")
                break
            if not self.advance_playdek():
                continue
            self.report.steps += 1
        # Drain the DLL so its result is known.
        while self.game.result is None and not self.stop and self.report.prompts < max_steps:
            if not self.advance_playdek():
                break
        for what, n in self.known.items():
            self.diverge("known", f"{what} ({n}x)")
        self.report.playdek_result = str(self.game.result)
        self.report.engine_result = f"winner={self.engine.winner} vp={self.engine.vp} turn={self.engine.turn}" if self.engine.is_terminal else f"not over (turn {self.engine.turn})"
        if self.engine.is_terminal and self.game.result is not None:
            pd_winner = self._sides_by_player.get(self.game.result.winner_id)
            if pd_winner != self.engine.winner:
                self.diverge("winner", f"Playdek {pd_winner}, engine {self.engine.winner}")
        self.game.close()
        return self.report


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--max-divergences", type=int, default=40)
    p.add_argument("--trace", action="store_true", help="print every DLL prompt/choice and engine decision/action")
    args = p.parse_args(argv)
    pd = Playdek()
    for g in range(args.games):
        seed = args.seed + g
        ls = Lockstep(pd, game_no=g, seed=seed, physical_side=Side.US if g % 2 == 0 else Side.USSR,
                      policy=random_policy(random.Random(seed)), max_divergences=args.max_divergences, trace=args.trace)
        r = ls.run(max_steps=args.max_steps)
        print(f"game {g} seed {seed}: {r.prompts} prompts, {r.engine_steps} engine steps; Playdek {r.playdek_result}; engine {r.engine_result}; "
              f"{len(r.divergences)} divergence(s)")
        for dv in r.divergences:
            print("  " + str(dv))


if __name__ == "__main__":
    main()
