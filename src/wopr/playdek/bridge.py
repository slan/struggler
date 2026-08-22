"""What the lockstep differ and the match operator share: one Playdek game
mirrored on a struggler engine in physical mode.

The DLL's records are absorbed into facts -- absolute state (influence,
DEFCON, VP, card locations), a FIFO of dice, what was dealt this turn --
and a queue of `Move`s per side: the choices the other program made, as
`translate.OptionMeaning`s, whether they came from a prompt the policy
answered (the differ, `lockstep.py`) or from the records an AI seat's
play leaves behind (`operator.py`). `_answer` turns the engine's pending
decision into an action from those facts and queues, or says the DLL must
be advanced first; `compare_state` diffs the two programs whenever both
are at rest. Every disagreement is a `Divergence` on the `Report`.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from struggler.engine import Engine
from struggler.engine.core import HIDDEN_CARD
from struggler.engine.types import Action, Decision, DecisionKind, Side
from wopr.playdek import ffi, ids, translate as T
from wopr.playdek.ffi import AIDifficulty, EventType
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
UI_ONLY = {T.Meaning.CANCEL, T.Meaning.SWITCH_CARD, T.Meaning.BLANK}  # never a move; no policy picks them

HAND_OF = {int(ffi.ECardLocation.USSRHAND): Side.USSR, int(ffi.ECardLocation.USHAND): Side.US}
HEADLINE_OF = {int(ffi.ECardLocation.USSRHEADLINE): Side.USSR, int(ffi.ECardLocation.USAHEADLINE): Side.US}
HAND_LOCATION = {Side.USSR: int(ffi.ECardLocation.USSRHAND), Side.US: int(ffi.ECardLocation.USHAND)}


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


class Bridge:
    def __init__(self, pd: Playdek, *, game_no: int, seed: int, local_side: Side, ai_difficulty: AIDifficulty | None,
                 physical_side: Side, deal_after_setup: bool = False, max_divergences: int = 40, trace: bool = False) -> None:
        self.pd = pd
        self.trace = trace
        self.report = Report(game_no)
        self.max_divergences = max_divergences
        self.game: PlaydekGame = pd.new_game(local_side=local_side, ai_difficulty=ai_difficulty, seed=seed)
        self.engine = Engine.new_game(seed=seed, include_optional=False, physical_mode=True, physical_side=physical_side,
                                      deal_after_setup=deal_after_setup)
        self.moves: dict[Side, collections.deque[Move]] = {Side.USSR: collections.deque(), Side.US: collections.deque()}
        self.rolls: collections.deque[T.Roll] = collections.deque()
        self.influence: dict[str, tuple[int, int]] = {}  # country -> (ussr, us) per the DLL
        self.card_loc: dict[str, int] = {}  # card -> ECardLocation per the DLL
        self._dealt: dict[Side, set[str]] = {Side.USSR: set(), Side.US: set()}  # dealt this turn per the DLL
        self._engine_dealt: dict[Side, set[str]] = {Side.USSR: set(), Side.US: set()}  # ...and already dealt to the engine
        self._dll_turn = 0
        self._last_moves: dict[str, tuple[int, int, int]] = {}  # card -> (from, to, sequence no.) of its latest move
        self._move_seq = 0
        self.defcon = 5
        self.vp = 0
        self.milops = (0, 0)
        self.space = (0, 0)
        self._events_seen = 0
        self._last_played: dict[Side, str | None] = {Side.USSR: None, Side.US: None}
        self._forced_mode: dict[Side, str | None] = {Side.USSR: None, Side.US: None}  # the next PLAY_MODE, when a lookahead settled it
        self._un_ops: dict[Side, bool] = {Side.USSR: False, Side.US: False}  # the next "Select Use For Operations" spends a UN-Intervened card
        self._grain: tuple[str, bool] | None = None  # Grain Sales: (the card drawn, whether the US took it)
        self._sides_by_player = self.game.sides  # player id -> Side
        self._index_side = dict(self.game.sides)  # the "player index" of roll events is the seat's id
        self._player_of = {side: pid for pid, side in self.game.sides.items()}
        # Records the DLL has emitted once and will emit again: a hotseat
        # game re-emits a whole action's records, verbatim and in order,
        # when the action is committed (see `_absorb`). A game against the
        # AI does not, and the FIFO must not be kept there: its head would
        # be the oldest record of the game, and a later roll equal to it
        # (the same coup, the same die) would be taken for a replay.
        self._replays = ai_difficulty is None
        self._replay: collections.deque[GameEvent] = collections.deque()
        self.recent: collections.deque[str] = collections.deque(maxlen=24)  # the last records, for diagnostics
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

    def replayed(self, ev: GameEvent) -> bool:
        """Whether `ev` is a hotseat re-emission of a record already
        absorbed. Records are matched off a FIFO: the oldest un-replayed one
        equal to `ev` is its replay, anything else is new and joins the
        FIFO. Every record kind that is re-emitted and *acted on* must go
        through here, in emission order, so the head is always the next
        record due to be replayed."""
        if not self._replays:
            return False
        if self._replay and self._replay[0] == ev:
            self._replay.popleft()
            return True
        self._replay.append(ev)
        return False

    def _absorb(self, ev: GameEvent) -> bool:
        """Absorb one record into the facts. False if it was a replay."""
        f = ev.fields
        if ev.kind == EventType.COUNTRY_INFLUENCE:
            self.influence[ids.country_id(f["id"])] = (f["ussr_influence"], f["us_influence"])
        elif ev.kind == EventType.DEFCON_LEVEL and not f["isSimulating"]:
            self.defcon = f["defcon_level"]
        elif ev.kind == EventType.TURN_NUMBER and f["turn_number"] != self._dll_turn:
            self._dll_turn = f["turn_number"]  # emitted twice per turn (preview, commit): reset once
            self._dealt = {Side.USSR: set(), Side.US: set()}
            self._engine_dealt = {Side.USSR: set(), Side.US: set()}
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
                if loc in HAND_OF:
                    self.diverge("unknown card", f"Playdek card {f['id']} entered a hand", fatal=True)
                return
            was = self.card_loc.get(card)
            self.card_loc[card] = loc
            for side, hand in HAND_LOCATION.items():
                if loc == hand and was in (None, int(ffi.ECardLocation.DECK)):
                    self._dealt[side].add(card)
            if was is not None and was != loc:
                self._move_seq += 1
                self._last_moves[card] = (was, loc, self._move_seq)
            if was != loc:
                self.recent.append(f"card {card}: {ffi.ECardLocation(was).name if was is not None else '?'} -> {ffi.ECardLocation(loc).name}")
                if self.trace:
                    print(f"  EV  {self.recent[-1]}")
        else:
            rolls = T.rolls_from_event(ev, self._index_side)
            if not rolls:
                return True
            # A hotseat game emits an action's records twice: once as each
            # choice is made (the preview) and again, verbatim and in
            # order, when the action is committed -- which is at the next
            # action boundary, so the re-emission may arrive pumps later,
            # after the engine has consumed the originals (a headline's
            # realignments are replayed after the first action round's).
            if self.replayed(ev):
                return False
            self.recent.append(str(ev))
            if self.trace:
                print(f"  EV  {ev}")
            self.rolls.extend(rolls)
        return True

    def hand_of(self, card: str) -> Side | None:
        """Whose hand the DLL has `card` in, if any."""
        return HAND_OF.get(self.card_loc.get(card, -1))

    def prompt_side(self, prompt: Prompt) -> Side:
        """Whose choice a prompt is. Simultaneous picks (turn 2+ headlines of
        a hotseat game) arrive under the local seat's id whoever is choosing;
        the cards on offer say whose hand it is."""
        side = self._sides_by_player[prompt.player_id]
        owners = {self.hand_of(T.meaning(o).card) for o in prompt.visible if T.meaning(o).meaning is T.Meaning.CARD}
        owners.discard(None)
        if len(owners) == 1:
            side = owners.pop()
        return side

    def queue(self, side: Side, move: Move) -> None:
        self.moves[side].append(move)
        if move.meaning.meaning is T.Meaning.CARD:
            self._last_played[side] = move.meaning.card

    # -- the engine side --------------------------------------------------

    def _answer(self, d: Decision) -> Action | None:
        """The engine's action for `d` from the queues, or None if the DLL
        must be advanced first."""
        if d.kind is DecisionKind.EVENT_RESUME:
            return d.options[0]
        if d.actor is Side.CHANCE:
            return self._answer_chance(d)
        side = d.actor
        q = self.moves[side]
        if d.kind is DecisionKind.EVENT_CHOICE and d.context.get("event") == "Grain_Sales_to_Soviets" and self._grain is not None:
            card, took = self._grain
            self._grain = None
            if not took:
                q.popleft()  # the "Return It" move
            return self._pick(d, lambda a: a.payload.get("choice") == ("take" if took else "return"), f"Grain Sales {'take' if took else 'return'} {card}")
        if len(d.options) == 1 and q and not self._compatible(d, q[0]):
            # A forced step the DLL did not ask about (a single legal target):
            # known only once this side's next prompt turned out to be something else.
            return d.options[0]
        if not q:
            if len(d.options) == 1 and self.game.result is not None:
                return d.options[0]  # the DLL's game is over: nothing more will be asked; take the forced steps
            return None
        mv = q[0]
        m = mv.meaning
        if m.meaning is T.Meaning.STOP and d.kind not in (DecisionKind.EVENT_CHOICE, DecisionKind.REALIGNMENT_TARGET):
            # A decline the DLL asked alone ("Do Not Discard" as the only
            # option: Blockade with no card to discard) where the engine
            # resolved the event without asking.
            q.popleft()
            return self._answer(d)
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
                self._un_ops[side] = True
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
                    self.known[f"{self._last_played[side]} played event-first: the engine has no event to order (Defectors by the USSR)"] += 1
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

    def _answer_chance(self, d: Decision) -> Action | None:
        if d.kind is DecisionKind.DEAL_CARD:
            # A card the DLL has in that hand and the engine has not been
            # told about yet (the DLL deals, undoes and re-deals at commit).
            side = Side(d.context["side"])
            # The hand as the DLL has it, plus what it dealt this turn: a
            # card dealt, headlined and resolved in one pump is in the
            # discard pile by the time the engine gets to deal it. A deal
            # undone at commit and not re-dealt is back in the deck.
            dll_hand = {c for c, loc in self.card_loc.items() if loc == HAND_LOCATION[side]}
            dll_hand |= {c for c in self._dealt[side] if self.card_loc.get(c) != int(ffi.ECardLocation.DECK)} - self._engine_dealt[side]
            dll_hand.discard(CHINA)
            missing = dll_hand - set(self.engine.hands[side.value])
            if not missing:
                return None
            offered = [a for a in d.options if a.payload["card"] in missing]
            if not offered:
                self.diverge("illegal in engine", f"deal {sorted(missing)} to {side.value}: not in the engine's hidden pool", fatal=True)
                return d.options[0]
            self._engine_dealt[side].add(offered[0].payload["card"])
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
        if d.kind is DecisionKind.RANDOM_DISCARD and d.context.get("purpose") == "grain_sales":
            if self._grain is None:
                return None
            return self._pick(d, lambda a: a.payload["card"] == self._grain[0], f"Grain Sales drew {self._grain[0]}")
        if d.kind is DecisionKind.RANDOM_DISCARD:
            # The DLL drew the card: the latest one to leave that hand for
            # the discard (or removed) pile that the engine still has
            # there. Cards played from the hand leave it the same way,
            # hence "latest" and "still offered".
            card = self.card_that_left(Side(d.context["owner"]), {a.payload["card"] for a in d.options})
            if card is None:
                return None
            return self._pick(d, lambda a: a.payload["card"] == card, f"random discard {card}")
        self.diverge("unsupported", f"CHANCE {d.kind.value}", fatal=True)
        return d.options[0]

    def card_that_left(self, owner: Side, offered: set[str]) -> str | None:
        """The latest card to leave `owner`'s hand for the discard or removed
        pile, among `offered`; consumed once named."""
        piles = (int(ffi.ECardLocation.DISCARDED), int(ffi.ECardLocation.REMOVED))
        gone = [(seq, c) for c, (was, now, seq) in self._last_moves.items()
                if was == HAND_LOCATION[owner] and now in piles and c in offered]
        if not gone:
            return None
        card = max(gone)[1]
        del self._last_moves[card]
        return card

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
        level = mv.meaning.defcon
        if level is not None:
            # "Set DEFCON to n": the engine asks the level (How I Learned to
            # Stop Worrying) or the direction from the current one (Summit).
            choices = {a.payload.get("choice") for a in d.options}
            if "raise" in choices:
                want = "raise" if level > self.engine.defcon else "lower" if level < self.engine.defcon else "none"
            else:
                want = str(level)
            return self._pick(d, lambda a: a.payload.get("choice") == want, f"DEFCON {level} -> {want}")
        # No shared vocabulary for an event's either/or: match the option
        # whose payload shares the most words with Playdek's label.
        words = set(mv.option.text.lower().split())
        best = max(d.options, key=lambda a: len(words & set(str(a.payload.get("choice", "")).lower().replace("_", " ").split())))
        self.diverge("choice by words", f"{mv.prompt.text!r} {mv.option.text!r} -> {dict(best.payload)} of {[dict(a.payload) for a in d.options]}")
        return best

    def _check_cards(self, d: Decision, mv: Move) -> None:
        if d.actor is self.engine.physical_side or not mv.prompt.options:
            return  # the engine cannot see that hand; its options are the whole pool
        theirs = T.cards_offered(mv.prompt)
        ours = {a.payload["card"] for a in d.options} - {HIDDEN_CARD}
        if theirs != ours:
            self.diverge("card options", f"{d.actor.value} {d.kind.value} {mv.prompt.text!r}: only Playdek {sorted(theirs - ours)}, only engine {sorted(ours - theirs)}")

    def _check_countries(self, d: Decision, mv: Move) -> None:
        if not mv.prompt.options:
            return  # a move learned from the records, not a prompt: nothing to compare
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

    def state_diffs(self, *, hands: bool) -> list[str]:
        """Where the DLL's absolute state and the engine's disagree right
        now. `hands`: compare hand sizes and the visible hand's contents too
        (only meaningful between action rounds: headline picks leave the
        hands at different moments)."""
        e = self.engine
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
        if hands:
            for side in (Side.USSR, Side.US):
                pd_count = self.game.hand_count(self._player_of[side])
                mine = len(e.hands[side.value])
                if pd_count != mine:
                    diffs.append(f"{side.value} hand size Playdek {pd_count}, engine {mine} {sorted(e.hands[side.value])}")
                if side is not e.physical_side:
                    # The hand the engine can see: its contents, card by card.
                    theirs = {c for c, loc in self.card_loc.items() if loc == HAND_LOCATION[side]} - {CHINA}
                    ours = set(e.hands[side.value]) - {CHINA}
                    if theirs != ours:
                        diffs.append(f"{side.value} hand: only Playdek {sorted(theirs - ours)}, only engine {sorted(ours - theirs)}")
        return diffs

    def compare_state(self) -> None:
        e = self.engine
        if e.phase in ("idle", "predeal", "setup"):
            return  # the DLL deals after the setup placements, the engine before: compare from the first headline on
        if e.pending_decision is None or e.pending_decision.kind not in CARD_KINDS:
            return  # mid-action on the engine's side
        diffs = self.state_diffs(hands=e.pending_decision.kind is DecisionKind.ACTION_ROUND_PLAY)
        text = "; ".join(diffs)
        if diffs and text != self._last_state_diff:
            self.diverge("state", f"turn {e.turn} AR {e.action_round}: " + text)
        self._last_state_diff = text

    # -- the end ----------------------------------------------------------

    def finish(self) -> Report:
        """Stamp both results on the report, compare the winners, close the
        DLL's game."""
        for what, n in self.known.items():
            self.diverge("known", f"{what} ({n}x)")
        self.report.playdek_result = str(self.game.result)
        self.report.engine_result = (f"winner={self.engine.winner} vp={self.engine.vp} turn={self.engine.turn}" if self.engine.is_terminal
                                     else f"not over (turn {self.engine.turn})")
        if self.engine.is_terminal and self.game.result is not None:
            pd_winner = self._sides_by_player.get(self.game.result.winner_id)
            if pd_winner != self.engine.winner:
                self.diverge("winner", f"Playdek {pd_winner}, engine {self.engine.winner}")
        self.game.close()
        return self.report
