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
from struggler.engine.core import HIDDEN_CARD, PASS_ROUND, RESHUFFLE_NOW
from struggler.engine.types import Action, Decision, DecisionKind, Side
from wopr.playdek import ffi, ids, translate as T
from wopr.playdek.ffi import AIDifficulty, EventType, SelectionHint
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
CMC_DEFUSE = "Cuban_Missile_Crisis_defuse"
GRANTED_OPS_PROMPT = "Select Use For Operations"  # Ops an event grants, UN Intervention's, Missile Envy's exchanged card
SCORING_PROMPT = "You May Play a Scoring Card"  # a trapped seat with no 2+-Ops card
REFORMER_KNOWN = "The Reformer: the DLL lets the USSR coup in Europe once Glasnost is gone (the card bans it for the rest of the game)"
OPPONENT_HAND_PROMPT = "Discard a Card from Opponent Hand"  # Aldrich Ames Remix: the cards shown are the other seat's
#: Prompts whose country is a point of Influence to place or remove: their
#: moves answer PLACE_INFLUENCE / EVENT_INFLUENCE, never an either/or whose
#: choices happen to be countries (Independent Reds' match, De-Stalinization's
#: source, the Cuban Missile Crisis defusing have hints of their own).
INFLUENCE_HINTS = {SelectionHint.INFLUENCE_COUNTRY, SelectionHint.SETUP_INFLUENCE_COUNTRY, SelectionHint.REMOVE_INFLUENCE_COUNTRY}
#: Record kinds a hotseat game does not re-emit verbatim at an action's commit (see
#: `Bridge.mark_replays`); every other kind is replayed verbatim, in order.
NOT_REPLAYED = {int(EventType.LOAD_PROGRESS), int(EventType.LOG_UPDATED), int(EventType.COMMIT_PLAYER_DECISION), int(EventType.GAME_OVER),
                int(EventType.OUTPUT_PAUSE), int(EventType.PAUSE_FOR_REVEALED_CARDS)}  # the pauses: a hand reveal replays with an extra one
REPLAY_FIFO_LIMIT = 400  # stable records outstanding: past this the re-emission has stopped matching, which is a bug worth a line

HAND_OF = {int(ffi.ECardLocation.USSRHAND): Side.USSR, int(ffi.ECardLocation.USHAND): Side.US}
HEADLINE_OF = {int(ffi.ECardLocation.USSRHEADLINE): Side.USSR, int(ffi.ECardLocation.USAHEADLINE): Side.US}
HAND_LOCATION = {Side.USSR: int(ffi.ECardLocation.USSRHAND), Side.US: int(ffi.ECardLocation.USHAND)}
PILES = (int(ffi.ECardLocation.DISCARDED), int(ffi.ECardLocation.REMOVED))


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
    seq: int = 0  # the record count when it was queued: where it stands among the facts


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
                 physical_side: Side, deal_after_setup: bool = False, max_divergences: int = 40, trace: bool = False,
                 us_bid: int = 0) -> None:
        self.pd = pd
        self.trace = trace
        self.report = Report(game_no)
        self.max_divergences = max_divergences
        self.game: PlaydekGame = pd.new_game(local_side=local_side, ai_difficulty=ai_difficulty, seed=seed,
                                             additional_influence=us_bid)
        self.engine = Engine.new_game(seed=seed, include_optional=False, physical_mode=True, physical_side=physical_side,
                                      deal_after_setup=deal_after_setup, us_bid=us_bid)
        self.moves: dict[Side, collections.deque[Move]] = {Side.USSR: collections.deque(), Side.US: collections.deque()}
        self.rolls: collections.deque[T.Roll] = collections.deque()
        self.influence: dict[str, tuple[int, int]] = {}  # country -> (ussr, us) per the DLL
        self.card_loc: dict[str, int] = {}  # card -> ECardLocation per the DLL
        self._dealt: dict[Side, set[str]] = {Side.USSR: set(), Side.US: set()}  # dealt this turn per the DLL
        self._engine_dealt: dict[Side, set[str]] = {Side.USSR: set(), Side.US: set()}  # ...and already dealt to the engine
        self._dll_turn = 0
        self._last_moves: dict[str, tuple[int, int, int]] = {}  # card -> (from, to, sequence no.) of its latest move
        self._reclaims: list[tuple[int, str, int]] = []  # (sequence no., card, hand) of every discard-to-hand move (SALT), consumed as answered
        self._move_seq = 0
        # Every move from a hand to the discard or removed pile, in order,
        # until the engine's own discard accounts for it (`card_that_left`):
        # not the card's latest move, which may already be its re-deal after
        # a reshuffle in the same pump (the AI's Blockade discard, reshuffled
        # and dealt to the other hand before the engine asked which card).
        self._exits: list[tuple[int, str, int, int, bool]] = []  # (move seq, card, the hand it left, record seq, whether it was its play)
        # Entries before this index left before the DLL's latest reshuffle:
        # stale once the engine has reshuffled too (the card may be in a
        # hand again, and offered again), still wanted until then.
        self._exits_before_reshuffle = 0
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
        # when the action is committed (see `mark_replays`). A game against
        # the AI does not, and the FIFO must not be kept there: every
        # record would stay outstanding and a later copy of the whole lot
        # could, in principle, be taken for a replay.
        self._replays = ai_difficulty is None
        self._replay: collections.deque[tuple[int, GameEvent]] = collections.deque()  # (the batch it arrived in, the record)
        self._batch_no = 0
        self._replay_overflow = False
        # The countries of the targets just answered (a realignment's, a
        # coup's): the records at the head of the next batch describing them
        # are the preview of that answer, never a replay -- a second
        # realignment of the same country can roll the same dice as the
        # first, whose records may by then be the FIFO's head.
        self._answered_countries: set[int] = set()
        self.recent: collections.deque[str] = collections.deque(maxlen=24)  # the last records, for diagnostics
        self.loc_history: dict[str, list[str]] = {}  # card -> every CARD_LOCATION transition, for the hand-drift dump
        self._seq = 0  # absorbed records, counted: the arrival order of the facts below
        self.influence_history: dict[str, list[tuple[int, tuple[int, int]]]] = {}  # country -> [(seq, (ussr, us))] at each change
        self.roll_seq: dict[int, int] = {}  # id(Roll) -> when it arrived
        self.roll_log: list[tuple[int, str]] = []  # (seq, country) of every coup/realignment record, kept
        self.synced_seq = 0  # the record count when the two states last agreed at rest
        self._setup_synced = False
        self._last_state_diff = ""
        self.reshuffled = False  # the DLL reshuffled its discards into the deck since the engine last folded them into its hidden pool
        self.known: collections.Counter[str] = collections.Counter()
        self._destal_counted: set[tuple[int, int, str]] = set()  # the De-Stalinization events already counted under `known`

    # -- divergences ------------------------------------------------------

    def diverge(self, what: str, detail: str, *, fatal: bool = False) -> None:
        self.report.divergences.append(Divergence(self.report.game, self.report.steps, what, detail, fatal))
        if self.trace:
            print(f"  !!  {self.report.divergences[-1]}")

    #: Purely informational divergence kinds: evidence for later diagnosis,
    #: never a reason to stop the game or to count toward the cap.
    _DIAGNOSTIC = ("grain", "hand-drift")

    @property
    def stop(self) -> bool:
        return (any(d.fatal for d in self.report.divergences)
                or sum(1 for d in self.report.divergences if d.what not in self._DIAGNOSTIC) >= self.max_divergences)

    # -- the DLL side -----------------------------------------------------

    def _absorb_events(self) -> None:
        batch = self.game.events[self._events_seen:]
        self._events_seen = len(self.game.events)
        for ev, replay in self.mark_replays(batch):
            self._absorb(ev, replay)

    def mark_replays(self, batch: list[GameEvent]) -> list[tuple[GameEvent, bool]]:
        """Each record of one pump's batch with whether it is a hotseat
        re-emission. A hotseat game emits an action's records as the
        choices are made and again, verbatim and in order, when the action
        is committed -- the whole run since the previous re-emission, minus
        the kinds in `NOT_REPLAYED`, possibly only after the next prompt
        was answered (a turn's end is replayed after the headline pick).
        So the outstanding records are kept in a FIFO and a run of the
        batch that copies it from its *head* is the replay; a record equal
        to some later entry is not -- a second realignment of the same
        country can roll the same dice, and was once taken for the replay
        of the first (matched alone, against a FIFO of dice and influence
        only), which fed the engine stale dice for the rest of the game."""
        if not self._replays:
            return [(ev, False) for ev in batch]
        self._batch_no += 1
        stable = [i for i, ev in enumerate(batch) if ev.kind not in NOT_REPLAYED]
        replay = [False] * len(batch)
        k = 0
        while k < len(stable):
            head = batch[stable[k]]
            if (k == 0 and head.kind == EventType.OUTPUT_ANIMATION_TARGET_COUNTRY
                    and head.fields.get("country_instance_id") in self._answered_countries):
                self._answered_countries.discard(head.fields["country_instance_id"])
                self._replay.append((self._batch_no, head))  # the preview of the target just chosen
                k += 1
                continue
            # The replay starts with the oldest outstanding record and copies
            # the chunk up to where it was committed -- the whole FIFO when
            # the action's last choice committed it (a realignment's rolls
            # are replayed in the same batch, after their own preview), or
            # a prefix when a prompt came between the commit and the replay
            # (the headline pick's own records follow the committed turn
            # end in the FIFO and are replayed with the next chunk). A
            # commit is never inside a batch's worth of preview records, so
            # the copy ends at a batch boundary of the FIFO: anything
            # shorter is a new record that happens to equal the head (the
            # second point placed in the same country re-emits the same
            # `OUTPUT_ANIMATION_ADD_INFLUENCE`).
            # Not every chunk is replayed (a non-phasing seat's event-granted
            # Op that ends the phasing seat's action never is): when the
            # head does not match, a run copying the FIFO from a later
            # batch boundary is the replay too, and the skipped records are
            # forgotten -- but not at the start of a batch, where the run
            # would be the preview of the answer just given (a second
            # realignment of the same country with the same dice).
            start, n = self._replay_run(batch, stable, k, 0)
            if not n and k:
                j = 1
                while j < len(self._replay) and not n:
                    if self._replay[j - 1][0] != self._replay[j][0]:
                        start, n = self._replay_run(batch, stable, k, j)
                    j += 1
            if n:
                if self.trace:
                    print(f"  RPL {n} record(s) at batch position {k} replay FIFO[{start}:{start + n}] of {len(self._replay)}: {batch[stable[k]]}")
                for _ in range(start):
                    self._replay.popleft()
                for j in range(n):
                    replay[stable[k + j]] = True
                    self._replay.popleft()
                k += n
                continue
            self._replay.append((self._batch_no, batch[stable[k]]))
            k += 1
        self._answered_countries.clear()
        if len(self._replay) > REPLAY_FIFO_LIMIT and not self._replay_overflow:
            self._replay_overflow = True
            self.diverge("harness", f"{len(self._replay)} records outstanding without a re-emission matching them: the replay "
                         f"matching (`Bridge.mark_replays`) has lost the DLL's commit structure; head {self._replay[0][1]}")
        return list(zip(batch, replay))

    def _replay_run(self, batch: list[GameEvent], stable: list[int], k: int, start: int) -> tuple[int, int]:
        """The length of the run of `batch` from `stable[k]` that copies the
        FIFO from `start` on, cut back to a batch boundary of the FIFO."""
        n = 0
        while start + n < len(self._replay) and k + n < len(stable) and batch[stable[k + n]] == self._replay[start + n][1]:
            n += 1
        while n and start + n < len(self._replay) and self._replay[start + n - 1][0] == self._replay[start + n][0]:
            n -= 1
        return start, n

    def _absorb(self, ev: GameEvent, replay: bool = False) -> bool:
        """Absorb one record into the facts. False if it was a replay: the
        absolute records (influence, card locations, DEFCON, VP) carry the
        values of their time, stale if a later action changed them, and the
        dice would be rolled twice."""
        f = ev.fields
        self._seq += 1
        if replay:
            if self.trace and (ev.kind in (EventType.COUNTRY_INFLUENCE, EventType.CARD_LOCATION) or T.rolls_from_event(ev, self._index_side)):
                print(f"  EV  (replay) {ev}")
            return False
        if ev.kind == EventType.COUNTRY_INFLUENCE:
            country = ids.country_id(f["id"])
            if self.influence.get(country) != (f["ussr_influence"], f["us_influence"]):
                self.influence_history.setdefault(country, []).append((self._seq, (f["ussr_influence"], f["us_influence"])))
                if self.trace:
                    print(f"  EV  influence {country} {(f['ussr_influence'], f['us_influence'])}")
            self.influence[country] = (f["ussr_influence"], f["us_influence"])
        elif ev.kind == EventType.DEFCON_LEVEL and not f["isSimulating"]:
            if self.trace and f["defcon_level"] != self.defcon:
                print(f"  EV  DEFCON {f['defcon_level']}")
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
            # becomes a hand it was not in.
            loc = f["location"]
            try:
                card = ids.card_id(f["id"])
            except KeyError:
                if loc in HAND_OF:
                    self.diverge("unknown card", f"Playdek card {f['id']} entered a hand", fatal=True)
                return
            was = self.card_loc.get(card)
            self.card_loc[card] = loc
            if was == int(ffi.ECardLocation.DISCARDED) and loc == int(ffi.ECardLocation.DECK):
                # The DLL's discards are back in its deck (its deck runs out
                # sooner than the engine's bookkeeping expects, docs/BOTS.md):
                # the engine's discard pile joins its hidden pool when the
                # engine gets to the deal (not now: it may still be playing
                # out the previous turn, discarding into the pile).
                self.reshuffled = True
                self._exits_before_reshuffle = len(self._exits)
            for side, hand in HAND_LOCATION.items():
                if loc == hand and was in (None, int(ffi.ECardLocation.DECK)):
                    self._dealt[side].add(card)
            if was is not None and was != loc:
                self._move_seq += 1
                self._last_moves[card] = (was, loc, self._move_seq)
                if was == int(ffi.ECardLocation.DISCARDED) and loc in HAND_OF:
                    # A card taken back from the discard pile (SALT
                    # Negotiations). Kept in its own log: `_last_moves` holds
                    # only the latest move, and a reclaimed card played again
                    # before the engine asks would hide the recovery there
                    # (counter1-easy seed 405: ABM Treaty back in hand at
                    # @1733, replayed at @1759, the reclaim read as declined).
                    self._reclaims.append((self._move_seq, card, loc))
                if was in HAND_OF and loc in PILES:
                    # Whether the move is the card's play is settled now: the
                    # card may come back to a hand (SALT Negotiations) and be
                    # played again, and this exit must stay what it was.
                    self._exits.append((self._move_seq, card, was, self._seq, self._exit_is_play(card, self._seq)))
            if was != loc:
                self.recent.append(f"card {card}: {ffi.ECardLocation(was).name if was is not None else '?'} -> {ffi.ECardLocation(loc).name}")
                self.loc_history.setdefault(card, []).append(
                    f"{ffi.ECardLocation(was).name if was is not None else '?'}->{ffi.ECardLocation(loc).name}@{self._seq}")
                if self.trace:
                    print(f"  EV  {self.recent[-1]}")
        else:
            rolls = T.rolls_from_event(ev, self._index_side)
            if not rolls:
                return True
            self.recent.append(str(ev))
            if self.trace:
                print(f"  EV  {ev}")
            for roll in rolls:
                self.roll_seq[id(roll)] = self._seq
                if roll.country is not None:
                    self.roll_log.append((self._seq, roll.country))
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
        if prompt.text == SCORING_PROMPT:
            return Side.USSR if self.engine.game_effects.get("bear_trap") else Side.US  # the trapped seat's, whoever's id it carries
        if self.grain_card(prompt) is not None:
            return Side.US  # Grain Sales' "Play <card>?": the US's decision, the card shown is the USSR's
        owners = {self.hand_of(T.meaning(o).card) for o in prompt.visible if T.meaning(o).meaning is T.Meaning.CARD}
        owners.discard(None)
        if len(owners) == 1:
            side = owners.pop()
            if prompt.text == OPPONENT_HAND_PROMPT:
                side = side.opponent  # the hand shown is the other seat's: the USSR picks the US's discard
        return side

    @staticmethod
    def grain_card(prompt: Prompt) -> str | None:
        """Grain Sales' "Play <the drawn card>?" / "Return It": the card."""
        drawn = [o for o in prompt.visible if o.hint in (SelectionHint.SWITCH_CARD, SelectionHint.PLAY_SCORING_CARD)]
        if drawn and prompt.text.endswith("?") and any(o.hint == SelectionHint.STOP for o in prompt.visible):
            return ids.card_id(drawn[0].selection_id)  # a scoring card is listed with its own hint
        return None

    def mark_setup_done(self) -> None:
        """Once both programs are past the setup, its records are behind
        them: the influence history before this point is nothing the
        operator still has to infer."""
        if not self._setup_synced:
            self._setup_synced = True
            self.synced_seq = self._seq

    def queue(self, side: Side, move: Move) -> None:
        move.seq = self._seq
        self.moves[side].append(move)
        if move.meaning.meaning is T.Meaning.CARD:
            self._last_played[side] = move.meaning.card

    # -- the engine side --------------------------------------------------

    def _reveal_hidden_scoring_cards(self) -> None:
        """Tell the engine about every scoring card the DLL shows in the hand
        the engine cannot see (`_reveal_in_hand`: the card takes one of the
        hand's hidden slots and leaves the hidden pool). The engine's own
        end-of-turn check then ends the game for a scoring card held past
        the turn, as the DLL does, instead of playing on blind."""
        e = self.engine
        physical = e.physical_side
        if physical is None or HIDDEN_CARD not in e.hands[physical.value]:
            return
        # Only while the two agree on the hand's size: a slot then stands
        # for a card the DLL has there. The DLL is pumped whole actions
        # ahead of the engine -- an Ask Not's discards and its replacement
        # draw arrive together, and a scoring card drawn would take, one
        # discard at a time, a slot of the hand being discarded, leaving
        # the last discard none (the engine draws its own slots at "stop",
        # the reveal follows). A deal at the turn's end is the same case
        # (the sizes differ until the engine deals), the `ahead` check its
        # older guard: a card the DLL dealt for the next turn while the
        # engine is still ending this one is not a card held over.
        if self.game.hand_count(self._player_of[physical]) != len(e.hands[physical.value]):
            return
        ahead = self._dll_turn != e.turn
        # Equal counts do not mean the slots correspond: an Ask Not that
        # discarded as many as it drew leaves the DLL's count where it was
        # while the engine still holds the old hand (counter1-easy seed 338
        # -- the drawn scoring cards took slots of the hand being
        # discarded, and the last discards had none). A card the DLL drew
        # this turn waits while the physical seat has unreplayed exits or
        # queued moves; the reveal follows once the engine has caught up.
        behind = bool(self.moves[physical]) or any(
            was == HAND_LOCATION[physical] and not play for _, _, was, _, play in self._exits)
        for card, loc in self.card_loc.items():
            if (loc == HAND_LOCATION[physical] and card in e.hidden_pool and e.cards[card].scoring
                    and not ((ahead or behind) and card in self._dealt[physical])):
                e._reveal_in_hand(physical, card)
                if HIDDEN_CARD not in e.hands[physical.value]:
                    return

    def _count_destalinization(self) -> None:
        """One `known` per De-Stalinization event, not per placement."""
        key = (self.engine.turn, self.engine.action_round, self.engine.phase)
        if key not in self._destal_counted:
            self._destal_counted.add(key)
            self.known["De-Stalinization: DLL excludes the source countries"] += 1

    def _answer(self, d: Decision) -> Action | None:
        """The engine's action for `d` from the queues, or None if the DLL
        must be advanced first."""
        self._reveal_hidden_scoring_cards()
        if d.kind is DecisionKind.EVENT_RESUME:
            return d.options[0]
        if d.actor is Side.CHANCE:
            return self._answer_chance(d)
        if self.reshuffled and d.kind is DecisionKind.HEADLINE_PLAY:
            # The DLL reshuffled its discards into the deck (its deck runs
            # out sooner than the engine's bookkeeping expects, docs/BOTS.md):
            # the physical side's next deal may hold any of them, so the
            # engine is told to fold its discard pile into the hidden pool
            # before the pick (the headline offers `RESHUFFLE_NOW` for that).
            # Not offered when the engine's own deck ran out at the same time
            # (it reshuffled by itself): nothing to fold, the flag lapses.
            self._engine_reshuffled()
            again = next((a for a in d.options if a.payload["card"] == RESHUFFLE_NOW), None)
            if again is not None and self.engine.discard_pile:
                return again
        side = d.actor
        q = self.moves[side]
        if d.kind is DecisionKind.EVENT_CHOICE and d.context.get("event") == "Grain_Sales_to_Soviets" and self._grain is not None:
            card, took = self._grain
            self._grain = None
            if not took:
                q.popleft()  # the "Return It" move
            return self._pick(d, lambda a: a.payload.get("choice") == ("take" if took else "return"), f"Grain Sales {'take' if took else 'return'} {card}")
        while (q and q[0].option.hint == SelectionHint.GIVE_CARD and "Missile_Envy" in self.engine.hands[side.value]
               and not (d.kind is DecisionKind.EVENT_CHOICE and d.context.get("event") == "Missile_Envy_pick")):
            # Missile Envy's "Select Card to Give" is asked even with a single
            # candidate; the engine asks the giver only among ties. Once the
            # engine has made the exchange (Missile Envy is in the giver's
            # hand) without asking, the move is dropped before anything else
            # looks at the queue: left there, it would pass for the answer
            # the next forced step is waiting for. Before that, it waits.
            q.popleft()
        if len(d.options) == 1 and q and not self._compatible(d, q[0]):
            # A forced step the DLL did not ask about (a single legal target,
            # a scoring card's "use"): known only once this side's next
            # prompt turned out to be something that cannot answer it.
            return d.options[0]
        if not q:
            if len(d.options) == 1 and self.game.result is not None:
                return d.options[0]  # the DLL's game is over: nothing more will be asked; take the forced steps
            return None
        mv = q[0]
        m = mv.meaning
        if d.kind is DecisionKind.EVENT_CHOICE and {a.payload.get("choice") for a in d.options} <= {"none", "coup", "realign"}:
            # An event's free Coup/Realignment (Junta, Ortega, Tear Down This
            # Wall): the DLL asks the use ("Select Use For Operations", or
            # "Pass"), or goes straight to the target when only one use exists.
            if m.meaning is T.Meaning.USE and m.use.ops_type in ("coup", "realignment"):
                q.popleft()
                want = "coup" if m.use.ops_type == "coup" else "realign"
                return self._pick(d, lambda a: a.payload.get("choice") == want, f"free {want}")
            if m.meaning is T.Meaning.COUNTRY:
                want = "coup" if "coup" in {a.payload.get("choice") for a in d.options} else "realign"
                return self._pick(d, lambda a: a.payload.get("choice") == want, f"free {want} (the target is next)")
        if d.kind is DecisionKind.EVENT_CHOICE and d.context.get("event") == CMC_DEFUSE:
            # The engine offers the defusing at the start of each of this
            # side's action rounds; the DLL lists it among the action round's
            # cards ("Remove 2 Influence from West Germany"). A card play
            # next means it was declined, and stays queued for the round.
            if mv.option.hint in (SelectionHint.CMC_DEFUSE, SelectionHint.CMC_DEFUSE_AT_COUP) and m.meaning is T.Meaning.COUNTRY:
                q.popleft()
                return self._pick(d, lambda a: a.payload.get("choice") == m.country, f"defuse in {m.country}")
            return self._pick(d, lambda a: a.payload.get("choice") == "skip", "skip defusing")
        if mv.option.hint == SelectionHint.TRAP_PASS:
            # "You May Play a Scoring Card" -> "Pass": a trapped seat with no
            # 2+-Ops card keeps its scoring card for a later round. The hand
            # the engine cannot see is first asked which of the hidden
            # pool's 2+-Ops cards it discards: none, and the Pass answers
            # the scoring step that follows.
            if d.kind is DecisionKind.QUAGMIRE_DISCARD and any(a.payload["card"] == "none" for a in d.options):
                if d.context.get("scoring_only"):
                    q.popleft()
                    return self._pick(d, lambda a: a.payload["card"] == "none", "keep the scoring card (trapped)")
                return self._pick(d, lambda a: a.payload["card"] == "none", "no 2+-Ops card in the trapped hand")
            q.popleft()
            self.diverge("decision mismatch", f"{side.value} keeps its scoring card in the trap (Pass), the engine asks "
                         f"{d.actor.value} {d.kind.value}", fatal=True)
            return self._answer(d)
        if m.meaning is T.Meaning.STOP and d.kind is DecisionKind.ACTION_ROUND_PLAY and any(a.payload["card"] == PASS_ROUND for a in d.options):
            q.popleft()  # "Pass" in "Play Your Action Round": the extra action round declined
            return self._pick(d, lambda a: a.payload["card"] == PASS_ROUND, "the extra action round passed")
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
        if d.kind is DecisionKind.HELD_CARD_DISCARD and mv.option.hint not in (SelectionHint.DISCARD_CARD, SelectionHint.STOP):
            # Space Race box 6's optional discard before the deal: the DLL's
            # next move is the headline pick, not a discard -- declined.
            return self._pick(d, lambda a: a.payload["card"] == "none", "no held-card discard")
        if (d.kind is DecisionKind.QUAGMIRE_DISCARD and not d.context.get("scoring_only") and m.meaning is T.Meaning.CARD
                and m.card in self.engine.cards and self.engine.cards[m.card].scoring and any(a.payload["card"] == "none" for a in d.options)):
            # The trapped hand the engine cannot see plays a scoring card:
            # no 2+-Ops card in it, the scoring step follows and takes the card.
            return self._pick(d, lambda a: a.payload["card"] == "none", "no 2+-Ops card in the trapped hand (a scoring card is played)")
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
            if self.game.result is not None and self.game.prompt is None:
                return None  # the DLL's game ended at the turn's end (a scoring card held in the hand the engine cannot see): `_more` says how
            side = Side(d.context["side"])
            if self.reshuffled:
                self._engine_reshuffled()
                self.engine._reshuffle_discard_into_draw()  # the DLL's reshuffle, at the engine's deal (see `_absorb`)
            # The hand as the DLL has it, plus what it dealt this turn: a
            # card dealt, headlined and resolved in one pump is in the
            # discard pile by the time the engine gets to deal it. A deal
            # undone at commit and not re-dealt is back in the deck. Only
            # cards whose last move came out of the deck count -- a card
            # that entered the hand another way (Missile Envy's exchange,
            # SALT's reclaim) is the replaying event's to apply, and
            # counting it here dealt the hand one slot too many (seed 315's
            # turn-4 drift, every batch: the DLL was pumped through deal,
            # headline and exchange in one chunk before the engine dealt).
            deck = int(ffi.ECardLocation.DECK)
            dll_hand = {c for c, loc in self.card_loc.items() if loc == HAND_LOCATION[side]
                        and self._last_moves.get(c, (deck, 0, 0))[0] == deck}
            dll_hand |= {c for c in self._dealt[side] if self.card_loc.get(c) != int(ffi.ECardLocation.DECK)} - self._engine_dealt[side]
            dll_hand.discard(CHINA)
            missing = dll_hand - set(self.engine.hands[side.value])
            last = self._last_played[side]
            if last in missing and last not in self.engine.hands[side.value] and last not in self._dealt[side]:
                # The card it is playing right now: the DLL keeps it in the
                # hand until its event is done asking (Blockade's "Do Not
                # Discard"), the engine filed it at the play. Dealt again
                # this turn (after a reshuffle), it is a card to deal.
                missing.discard(last)
            if not missing:
                return None
            offered = [a for a in d.options if a.payload["card"] in missing]
            if not offered:
                trail = "; ".join(f"{c}: DLL history {self.loc_history.get(c, [])}" for c in sorted(missing))
                self.diverge("illegal in engine", f"deal {sorted(missing)} to {side.value}: not in the engine's hidden pool "
                             f"(engine hand {sorted(self.engine.hands[side.value])}; dealt-this-turn DLL {sorted(self._dealt[side])} "
                             f"engine {sorted(self._engine_dealt[side])}; {trail})", fatal=True)
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
        if d.kind is DecisionKind.EVENT_CHOICE:
            return self._answer_hidden_hand(d)
        self.diverge("unsupported", f"CHANCE {d.kind.value}", fatal=True)
        return d.options[0]

    def _answer_hidden_hand(self, d: Decision) -> Action | None:
        """The three events where the engine asks the operator about the
        hand it cannot see (docs/BOTS.md): the DLL shows that hand in full."""
        event = d.context.get("event")
        physical = self.engine.physical_side
        offered = {a.payload["choice"] for a in d.options}
        in_hand = {c for c, loc in self.card_loc.items() if loc == HAND_LOCATION[physical]}
        if event == "Missile_Envy_physical_pick":
            # The card the giver handed over: the latest to leave that hand
            # (for the taker's hand or wherever the DLL files it) -- except
            # Grain Sales' random draw, which an exchanged Grain Sales pulls
            # out of the same hand while its take/return prompt is up: that
            # card is the draw's fact, never the exchanged card.
            taker = Side(d.context["taker"])
            drawn = self.grain_card(self.game.prompt) if self.game.prompt is not None else None
            gone = [(seq, c) for c, (was, now, seq) in self._last_moves.items()
                    if was == HAND_LOCATION[physical] and c in offered and c != drawn]
            if not gone:
                return None
            card = max(gone)[1]
            q = self.moves[physical]
            if q and q[0].option.hint == SelectionHint.GIVE_CARD:
                q.popleft()  # the giver's own "Select Card to Give" answer, asked of the seat the engine cannot see
            return self._pick(d, lambda a: a.payload["choice"] == card, f"Missile Envy takes {card} for {taker.value}")
        if event == "Aldrich_Ames_Remix_reveal":
            card = next((a.payload["choice"] for a in d.options if a.payload["choice"] in in_hand), None)
            if card is None:
                return None
            return self._pick(d, lambda a: a.payload["choice"] == card, f"reveal {card}")
        if event == "Cambridge_Five_query":
            want = "yes" if d.context.get("scoring_id") in in_hand else "no"
            return self._pick(d, lambda a: a.payload["choice"] == want, f"Cambridge Five: {want}")
        self.diverge("unsupported", f"CHANCE {d.kind.value} {event}", fatal=True)
        return d.options[0]

    def card_that_left(self, owner: Side, offered: set[str]) -> str | None:
        """The latest card to leave `owner`'s hand for the discard or removed
        pile, among `offered`; consumed once named."""
        for i in range(len(self._exits) - 1, -1, -1):
            seq, card, was, rseq, play = self._exits[i]
            if was == HAND_LOCATION[owner] and card in offered and not play:
                del self._exits[i]
                if i < self._exits_before_reshuffle:
                    self._exits_before_reshuffle -= 1
                return card
        return None

    def _exit_is_play(self, card: str, record_seq: int) -> bool:
        """Whether the move of `card` out of a hand at record `record_seq`
        was its play, not a discard (the operator knows the plays from the
        DLL's animation records; the differ's moves come from prompts)."""
        return False

    def _engine_reshuffled(self) -> None:
        """The engine has folded its discards back as the DLL did: the
        exits before the DLL's reshuffle are accounted for or stale."""
        self.reshuffled = False
        del self._exits[:self._exits_before_reshuffle]
        self._exits_before_reshuffle = 0

    @staticmethod
    def _compatible(d: Decision, mv: Move) -> bool:
        """Whether `mv` can answer `d`: an option of `d` is what it names.
        The kind alone is not enough -- a card play is a card, and so is
        the one candidate of an Independent Reds the DLL resolved on its
        own; the engine asks the latter, with one option, before the
        former."""
        m = mv.meaning
        k = d.kind
        opts = d.options
        if mv.option.hint == SelectionHint.TRAP_PASS:
            return k is DecisionKind.QUAGMIRE_DISCARD and any(a.payload.get("card") == "none" for a in opts)  # the trapped seat keeps its scoring card
        if m.meaning is T.Meaning.CARD:
            if k is DecisionKind.ACTION_ROUND_PLAY and m.card == UN:
                return True  # played as the opponent card's mode; see `_answer`
            if k in CARD_KINDS:
                return any(a.payload.get("card") == m.card for a in opts)
            return k is DecisionKind.EVENT_CHOICE and any(a.payload.get("choice") == m.card for a in opts)
        if m.meaning is T.Meaning.COUNTRY:
            if k in COUNTRY_KINDS:
                return any(a.payload.get("country") == m.country for a in opts)
            return (k is DecisionKind.EVENT_CHOICE and mv.option.hint not in INFLUENCE_HINTS
                    and any(a.payload.get("choice") in (m.country, "skip") for a in opts))
        if m.meaning is T.Meaning.USE:
            use = m.use
            if k is DecisionKind.PLAY_MODE:
                return any(a.payload.get("mode") == use.mode for a in opts)
            if k is DecisionKind.OPS_TYPE:
                return use.ops_type is None or any(a.payload.get("type") == use.ops_type for a in opts)
            return k is DecisionKind.EVENT_OPS_ORDER
        if m.meaning is T.Meaning.STOP:
            if k is DecisionKind.REALIGNMENT_TARGET:
                return any(a.payload.get("country") == "stop" for a in opts)
            return k is DecisionKind.EVENT_CHOICE and any(a.payload.get("choice") not in ids.NUMBER_BY_CARD and a.payload.get("choice") not in ids.INDEX_BY_COUNTRY for a in opts)
        return k is DecisionKind.EVENT_CHOICE and m.meaning is T.Meaning.CHOICE

    def _pick(self, d: Decision, pred, what: str) -> Action:
        for a in d.options:
            if pred(a):
                return a
        # A card the engine does not offer is usually a hand drift: name the
        # card's trail so the mismatch is diagnosable from the report alone.
        card = what.split()[-1] if what else ""
        trail = (f"; {card}: engine has it in {self._engine_location(card)}, DLL history {self.loc_history.get(card, [])}"
                 if card in self.loc_history or card in ids.NUMBER_BY_CARD else "")
        self.diverge("illegal in engine", f"{d.actor.value} {d.kind.value}: Playdek chose {what}, engine offers "
                     f"{[dict(a.payload) for a in d.options][:12]}{'...' if len(d.options) > 12 else ''}{trail}", fatal=True)
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
        # An either/or: the listed labels, else the option whose payload
        # shares the most words with Playdek's label (reported).
        listed = T.CHOICE_LABELS.get((mv.prompt.text, mv.option.text))
        if listed is not None:
            return self._pick(d, lambda a: a.payload.get("choice") == listed, f"choice {listed} ({mv.option.text!r})")
        words = set(mv.option.text.lower().split())
        best = max(d.options, key=lambda a: len(words & set(str(a.payload.get("choice", "")).lower().replace("_", " ").split())))
        self.diverge("choice by words", f"{mv.prompt.text!r} {mv.option.text!r} -> {dict(best.payload)} of {[dict(a.payload) for a in d.options]}")
        return best

    def _check_cards(self, d: Decision, mv: Move) -> None:
        if d.actor is self.engine.physical_side or not mv.prompt.options:
            return  # the engine cannot see that hand; its options are the whole pool
        theirs = T.cards_offered(mv.prompt)
        ours = {a.payload["card"] for a in d.options} - {HIDDEN_CARD, PASS_ROUND}  # the extra round's pass is "Pass", not a card
        if theirs != ours:
            self.diverge("card options", f"{d.actor.value} {d.kind.value} {mv.prompt.text!r}: only Playdek {sorted(theirs - ours)}, only engine {sorted(ours - theirs)}")

    def _reformer_lapsed_in_dll(self, countries: set[str]) -> bool:
        """Whether `countries` are coup targets the DLL offers the USSR and
        the engine does not because of The Reformer: the card bans USSR
        coups in Europe for the rest of the game, the DLL keeps the ban
        on the card "in play", which it is not once Glasnost has already
        been played (its Lua: `PutThisCardInPlay` only if Glasnost is not
        in the removed pile)."""
        return bool(countries) and bool(self.engine.game_effects.get("reformer")) and all(
            c in self.engine.board.countries and self.engine.board.countries[c].region.name == "EUROPE" for c in countries)

    def _check_countries(self, d: Decision, mv: Move) -> None:
        if not mv.prompt.options:
            return  # a move learned from the records, not a prompt: nothing to compare
        theirs = T.countries_offered(mv.prompt)
        if any(T.meaning(o).meaning is T.Meaning.STOP for o in mv.prompt.visible):
            theirs.add("stop")  # "No More Realignment" is the engine's {"country": "stop"}
        ours = {a.payload["country"] for a in d.options}
        if d.kind is DecisionKind.COUP_TARGET and theirs - ours == {"stop"}:
            theirs.discard("stop")  # an event's free coup: the DLL declines at the target, the engine asked before
        if d.kind is DecisionKind.COUP_TARGET and theirs > ours and self._reformer_lapsed_in_dll(theirs - ours):
            self.known[REFORMER_KNOWN] += 1
            theirs = ours
        if d.kind is DecisionKind.EVENT_INFLUENCE and d.context.get("event") == "De_Stalinization" and ours > theirs:
            # Known: the DLL will not relocate influence back into a country it
            # was just removed from; the card text has no such clause.
            self._count_destalinization()
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
        china = getattr(self, "_china", None)  # the operator tracks the DLL's holder; the differ does not
        if china is not None and china.value != e.china_card_owner:
            # Ownership forks silently (a Cultural Revolution mode misread)
            # and only surfaces turns later (Nixon's +2-VP-or-take branch,
            # v3-easy-r8 seed 405): compared here so the drift is located.
            diffs.append(f"China Card Playdek {china.value}, engine {e.china_card_owner}")
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

    def _engine_location(self, card: str) -> str:
        e = self.engine
        for side in ("USSR", "US"):
            if card in e.hands[side]:
                return f"{side} hand"
        for name in ("draw_pile", "discard_pile", "removed_cards", "hidden_pool"):
            if card in getattr(e, name):
                return name
        return "untracked"

    def _hand_drift_dump(self) -> str:
        """Every card the visible hands disagree on: the engine's location,
        the DLL's full location history, and the deal bookkeeping."""
        e = self.engine
        parts = []
        for side in (Side.USSR, Side.US):
            if side is e.physical_side:
                continue
            theirs = {c for c, loc in self.card_loc.items() if loc == HAND_LOCATION[side]} - {CHINA}
            ours = set(e.hands[side.value]) - {CHINA}
            for card in sorted(theirs ^ ours):
                tags = "".join(
                    tag for cond, tag in (
                        (card in self._dealt[side], " [DLL dealt this turn]"),
                        (card in self._engine_dealt[side], " [engine dealt this turn]"),
                        (self._last_played[side] == card, " [last played]"),
                    ) if cond)
                parts.append(f"{card} ({'DLL-only' if card in theirs else 'engine-only'}): engine has it in "
                             f"{self._engine_location(card)}{tags}; DLL history {self.loc_history.get(card, [])}")
        return "; ".join(parts) or "no visible-hand diff at dump time"

    def compare_state(self) -> None:
        e = self.engine
        if e.phase in ("idle", "predeal", "setup"):
            return  # the DLL deals after the setup placements, the engine before: compare from the first headline on
        self.mark_setup_done()
        if e.pending_decision is None or e.pending_decision.kind not in CARD_KINDS:
            return  # mid-action on the engine's side
        if self._dll_turn != e.turn or e.pending_decision.kind is DecisionKind.HELD_CARD_DISCARD:
            return  # at the turn's end: the engine has recovered DEFCON, the DLL reports it after its next prompt
        diffs = self.state_diffs(hands=e.pending_decision.kind is DecisionKind.ACTION_ROUND_PLAY)
        text = "; ".join(diffs)
        if diffs and text != self._last_state_diff:
            self.diverge("state", f"turn {e.turn} AR {e.action_round}: " + text)
            if any("hand: only" in d for d in diffs):
                # A hand-contents drift never self-heals and its cause is a
                # deal or card-move mismatch long scrolled out of `recent`:
                # dump the diverging cards' whole DLL location history and
                # where the engine has each, once, at first sight.
                self.diverge("hand-drift", self._hand_drift_dump())
        self._last_state_diff = text
        if not diffs:
            self.synced_seq = self._seq

    # -- the end ----------------------------------------------------------

    def finish(self) -> Report:
        """Stamp both results on the report, compare the winners, close the
        DLL's game."""
        for what, n in self.known.items():
            self.diverge("known", f"{what} ({n}x)")
        self.report.playdek_result = str(self.game.result)
        self.report.engine_result = (f"winner={self.engine.winner} ({getattr(self.engine, '_game_over_reason', '?')}) vp={self.engine.vp} "
                                     f"turn={self.engine.turn}" if self.engine.is_terminal else f"not over (turn {self.engine.turn})")
        if self.engine.is_terminal and self.game.result is not None:
            pd_winner = self._sides_by_player.get(self.game.result.winner_id)
            if pd_winner != self.engine.winner:
                if (self.game.result.win_type is ffi.GameOverType.HELD_CARDS
                        and getattr(self.engine, "_game_over_reason", None) == "held_scoring_card" and self.engine.winner is None):
                    # Both hands held a scoring card: the engine calls it a
                    # draw, the DLL names a winner (the US, in the game seen).
                    self.known["held scoring card in both hands: the DLL names a winner, the engine a draw"] += 1
                else:
                    self.diverge("winner", f"Playdek {pd_winner}, engine {self.engine.winner}")
        self.game.close()
        return self.report
