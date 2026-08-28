"""The arena: many games at once, exposed decision by decision.

An `Arena` owns N engines. Its one question is *decision-centric*, not
player-centric: "which games are waiting on which policy?" (`pending`).
Whoever drives it -- the SB3 `VecEnv` adapter, an evaluation loop, a future
shared-memory server -- answers with one option index per waiting game
(`apply`), and the arena steps each engine, resolves `Side.CHANCE` frames
itself (their single pre-rolled option is not a choice, mandate #3), and
stops at the next non-chance decision.

Each seat of each game is assigned a *policy id* (a plain string) when the
game starts. Nothing here knows what a policy is -- the learner, a frozen
checkpoint, `GreedyPlayer`, a human -- it only groups rows by id. That is
what makes learner-vs-pool, learner-vs-learner, and evaluation matches the
same code path, and what a multi-seat engine backend would implement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, NamedTuple, Protocol, Sequence

from struggler.engine import Engine, Observation, Side

if TYPE_CHECKING:
    from wopr.scenarios import ScenarioBank

#: Decides the two seats' policy ids for a game: (slot, episode, rng) -> {side: policy id}.
SeatAssigner = Callable[[int, int, random.Random], Mapping[Side, str]]


def self_play(slot: int, episode: int, rng: random.Random, *, policy: str = "learner") -> dict[Side, str]:
    return {Side.US: policy, Side.USSR: policy}


class PendingRow(NamedTuple):
    slot: int
    side: Side
    observation: Observation


class Opponent(Protocol):
    """Anything that can answer a batch of pending rows with option indices."""

    def choose(self, rows: Sequence[PendingRow]) -> Sequence[int]: ...


@dataclass
class GameResult:
    slot: int
    episode: int
    seed: int
    seats: dict[Side, str]
    winner: Side | None
    decisions: dict[Side, int]
    turn: int
    vp: int


@dataclass
class _Slot:
    engine: Engine
    seats: dict[Side, str]
    episode: int
    seed: int
    decisions: dict[Side, int] = field(default_factory=lambda: {Side.US: 0, Side.USSR: 0})


class Arena:
    def __init__(
        self,
        n_games: int,
        *,
        seed: int,
        seat_assigner: SeatAssigner = self_play,
        events: bool = True,
        include_optional: bool = True,
        slot_offset: int = 0,
        total_slots: int | None = None,
        starting_vp: int = 0,
        us_bid: int = 0,
        scenario_bank: "ScenarioBank | None" = None,
        scenario_frac: float = 0.0,
    ) -> None:
        """`starting_vp` opens every game at that VP (US-positive): a handicap
        for the USSR seat, as a tournament bid. 0 is the printed game.
        `us_bid` opens every game with the official tournament bid instead
        (rule 11.1.4): that much extra US influence placed after setup.

        `scenario_bank`/`scenario_frac` start that fraction of games from
        a bank state re-hidden by `Engine.determinize` instead of the
        printed setup (`wopr.scenarios`, docs/JOSHUA.md). The choice, the
        entry and the determinize seed are pure functions of the game
        seed, so both backends play the same games; the bank's game spec
        must match the arena's.

        `slot_offset`/`total_slots` make this arena a slice of a larger
        one: its slots are numbered from `slot_offset` for seeding and for
        the seat assigner, so k arenas of n/k slots play exactly the games
        one arena of n slots would (the shared-memory backend)."""
        if n_games < 1:
            raise ValueError("n_games must be >= 1")
        self.n_games = n_games
        self._seed = seed
        self._rng = random.Random(seed)
        self._seat_assigner = seat_assigner
        self._events = events
        self._include_optional = include_optional
        self._starting_vp = starting_vp
        self._us_bid = us_bid
        if scenario_frac and scenario_bank is None:
            raise ValueError("scenario_frac needs a scenario_bank")
        if scenario_bank is not None:
            scenario_bank.validate(us_bid=us_bid, starting_vp=starting_vp,
                                   events=events, include_optional=include_optional)
        self._scenario_bank = scenario_bank
        self._scenario_frac = scenario_frac
        self._slot_offset = slot_offset
        self._total_slots = n_games if total_slots is None else total_slots
        if slot_offset < 0 or slot_offset + n_games > self._total_slots:
            raise ValueError(f"slots [{slot_offset}, {slot_offset + n_games}) exceed total_slots={self._total_slots}")
        self._slots: list[_Slot] = [self._new_slot(i, 0) for i in range(n_games)]

    # -- lifecycle -----------------------------------------------------------

    def _new_slot(self, slot: int, episode: int) -> _Slot:
        # Distinct, reproducible engine seed per (global slot, episode); the
        # seat assigner's randomness comes from the arena's own rng.
        global_slot = self._slot_offset + slot
        game_seed = self._seed * 1_000_003 + episode * self._total_slots + global_slot
        engine = self._new_engine(game_seed)
        self._resolve_chance(engine)
        seats = dict(self._seat_assigner(global_slot, episode, self._rng))
        if set(seats) != {Side.US, Side.USSR}:
            raise ValueError(f"seat assigner must assign exactly US and USSR, got {sorted(s.value for s in seats)}")
        return _Slot(engine=engine, seats=seats, episode=episode, seed=game_seed)

    def _new_engine(self, game_seed: int) -> Engine:
        # The scenario draw hangs off the game seed alone, so k sliced
        # arenas make the same choice one whole arena would for that slot.
        if self._scenario_bank is not None and self._scenario_frac > 0.0:
            rng = random.Random(game_seed * 2_147_483_629 + 17)
            if rng.random() < self._scenario_frac:
                return self._scenario_bank.start(rng.randrange(len(self._scenario_bank)), rng.getrandbits(63))
        return Engine.new_game(
            seed=game_seed, events=self._events, include_optional=self._include_optional,
            starting_vp=self._starting_vp, us_bid=self._us_bid,
        )

    def reset(self, slot: int) -> None:
        """Start the next game in `slot` (a fresh engine, a fresh seat assignment)."""
        self._slots[slot] = self._new_slot(slot, self._slots[slot].episode + 1)

    def reset_all(self) -> None:
        for i in range(self.n_games):
            self.reset(i)

    # -- queries ---------------------------------------------------------------

    def engine(self, slot: int) -> Engine:
        return self._slots[slot].engine

    def seats(self, slot: int) -> Mapping[Side, str]:
        return self._slots[slot].seats

    def is_terminal(self, slot: int) -> bool:
        return self._slots[slot].engine.is_terminal

    def mover(self, slot: int) -> Side | None:
        """Side to move in `slot`, or None when the game is over."""
        decision = self._slots[slot].engine.pending_decision
        return None if decision is None else decision.actor

    def policy_for(self, slot: int) -> str | None:
        side = self.mover(slot)
        return None if side is None else self._slots[slot].seats[side]

    def row(self, slot: int) -> PendingRow:
        side = self.mover(slot)
        if side is None:
            raise ValueError(f"slot {slot} has no pending decision")
        return PendingRow(slot, side, self._slots[slot].engine.observe(side))

    def pending(self, slots: Sequence[int] | None = None) -> dict[str, list[PendingRow]]:
        """Waiting decisions grouped by policy id, in slot order."""
        grouped: dict[str, list[PendingRow]] = {}
        for i in (range(self.n_games) if slots is None else slots):
            side = self.mover(i)
            if side is None:
                continue
            grouped.setdefault(self._slots[i].seats[side], []).append(self.row(i))
        return grouped

    def result(self, slot: int) -> GameResult:
        s = self._slots[slot]
        if not s.engine.is_terminal:
            raise ValueError(f"slot {slot} is still running")
        return GameResult(
            slot=slot,
            episode=s.episode,
            seed=s.seed,
            seats=dict(s.seats),
            winner=s.engine.winner,
            decisions=dict(s.decisions),
            turn=s.engine.turn,
            vp=s.engine.vp,
        )

    # -- stepping ----------------------------------------------------------------

    def apply(self, slot: int, option: int) -> None:
        """Resolve `slot`'s pending decision with `options[option]`, then run
        every CHANCE frame that follows, stopping at the next real decision
        or the end of the game."""
        s = self._slots[slot]
        decision = s.engine.pending_decision
        if decision is None:
            raise ValueError(f"slot {slot}: game is over")
        if decision.actor is Side.CHANCE:
            raise ValueError(f"slot {slot}: CHANCE frames are resolved by the arena, not by a policy")
        if not 0 <= option < len(decision.options):
            raise IndexError(f"slot {slot}: option {option} out of range for {len(decision.options)} options")
        s.decisions[decision.actor] += 1
        s.engine.step(decision.options[option])
        self._resolve_chance(s.engine)

    @staticmethod
    def _resolve_chance(engine: Engine) -> None:
        while not engine.is_terminal:
            decision = engine.pending_decision
            if decision.actor is not Side.CHANCE:
                return
            # A non-physical game pre-draws exactly one option (mandate #3).
            if len(decision.options) != 1:
                raise RuntimeError("CHANCE decision with several options outside physical mode")
            engine.step(decision.options[0])


def play_out(arena: Arena, policies: Mapping[str, Opponent]) -> list[GameResult]:
    """Run every game in `arena` to completion with the given policies and
    return their results in slot order. Decisions are answered policy by
    policy in batches, so a policy backed by a network sees one forward
    pass per round, not one per game."""
    while True:
        pending = arena.pending()
        if not pending:
            break
        for policy_id, rows in pending.items():
            policy = policies.get(policy_id)
            if policy is None:
                raise KeyError(f"no policy registered for seat {policy_id!r}")
            choices = policy.choose(rows)
            if len(choices) != len(rows):
                raise ValueError(f"policy {policy_id!r} answered {len(choices)} of {len(rows)} rows")
            for row, choice in zip(rows, choices):
                arena.apply(row.slot, int(choice))
    return [arena.result(i) for i in range(arena.n_games)]
