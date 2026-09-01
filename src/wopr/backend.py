"""Backends: what steps the games and fills the layout buffers.

`WoprVecEnv` asks one question of a backend -- "given an option index per
slot, bring every slot to its next learner decision and write its row" --
and two implementations answer it:

- `InProcessBackend`: N engines in this process. The reference, and what
  every worker of the other backend runs over its share of the slots.
- `SharedMemoryBackend`: k worker processes, each an `InProcessBackend`
  over a contiguous slice of the slots, writing straight into shared
  memory. The layout *is* the transport: every array of `features.LAYOUT`
  is one shared slab of shape `[n_slots, ...]`, each worker owns its rows,
  and the main process reads the whole slab after the step. Actions,
  rewards, dones, the mover of each next row and the record of a game
  that ended this step are fixed-shape shared arrays too. Nothing is
  pickled on the step path; the only signal per step is one semaphore
  release per worker each way. Seats are decided in the main process
  (the seat assigner sees the pool and the anchor schedule there) and
  handed over one game ahead in a shared table the worker reads when a
  slot resets.

Game seeds are `(run seed, global slot, episode)` in both backends, so a
deterministic configuration (`--self-play 1.0`) plays the same games
through either; `tests/test_wopr.py` pins that.

Both also answer a second question, between rollouts: "play this
`EvalJob` and hand back the counts" (`start_eval` / `finish_eval`). The
shared-memory backend plays it on the collectors -- each its slice of the
decks (`eval.play_slice`) -- while the main process runs the PPO update,
which is when the collectors would otherwise sit idle; the in-process one
plays it on the spot.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
import zlib
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Mapping, NamedTuple, Protocol, Sequence

import numpy as np

from struggler.bots.joshua import features as F
from struggler.engine import Side
from wopr.arena import Arena, Opponent, PendingRow, SeatAssigner
from wopr.eval import EvalCounts, EvalJob

LEARNER = "learner"

#: The VP track's end (rules.json `vp_to_win`): the margin reward's scale.
VP_TO_WIN = 20.0

#: Resolves a non-learner policy id to something that can answer rows.
OpponentResolver = Callable[[str], Opponent]


class EpisodeRecord(NamedTuple):
    """A game that ended during a step, seen from the learner's last row."""

    winner: Side | None
    mover: Side
    seats: dict[Side, str]
    turn: int
    vp: int
    seed: int
    length: int
    margin: float = 0.0  # weight of the final-VP margin in `reward` (0: the outcome alone)

    def reward(self) -> float:
        """The terminal reward for the mover: `(1 - margin) * outcome +
        margin * clip(final VP for the mover / 20, -1, 1)`. With margin 0 it
        is the outcome alone, +1/-1/0. With margin on, a loss held to -3
        on the track is worth more than one that reached -20, and a win on
        VP is still +1; the sum over the two seats is still 0."""
        outcome = 0.0 if self.winner is None else (1.0 if self.winner is self.mover else -1.0)
        if not self.margin:
            return outcome
        mine = self.vp if self.mover is Side.US else -self.vp
        track = max(-1.0, min(1.0, mine / VP_TO_WIN))
        return (1.0 - self.margin) * outcome + self.margin * track

    def summary(self) -> dict[str, Any]:
        """The `infos[i]["episode"]` dict: SB3's Monitor keys plus WOPR's."""
        return {
            "r": self.reward(),
            "l": self.length,
            "winner": None if self.winner is None else self.winner.value,
            "seats": {side.value: policy for side, policy in self.seats.items()},
            "mover": self.mover.value,
            "turn": self.turn,
            "vp": self.vp,
            "seed": self.seed,
        }


class Backend(Protocol):
    n_slots: int
    buffers: dict[str, np.ndarray]  # the layout, `[n_slots, ...]`: the current learner rows

    def reset(self) -> None: ...
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[EpisodeRecord | None]]: ...
    def am_us(self) -> np.ndarray: ...
    def start_eval(self, job: EvalJob) -> None: ...
    def finish_eval(self) -> EvalCounts: ...
    def close(self) -> None: ...


# -- in process ---------------------------------------------------------------------


class InProcessBackend:
    def __init__(
        self,
        arena: Arena,
        opponents: OpponentResolver,
        *,
        buffers: Mapping[str, np.ndarray] | None = None,
        learner: str = LEARNER,
        margin: float = 0.0,
    ) -> None:
        self.arena = arena
        self.margin = margin
        self.n_slots = arena.n_games
        self._resolve = opponents
        self._opponents: dict[str, Opponent] = {}
        self._learner = learner
        self.buffers = dict(buffers) if buffers is not None else F.allocate(arena.n_games)
        for name, array in self.buffers.items():
            if array.shape[0] != self.n_slots:
                raise ValueError(f"buffer {name!r} has {array.shape[0]} rows for {self.n_slots} slots")
        self._rows: list[PendingRow | None] = [None] * self.n_slots
        self._steps = np.zeros(self.n_slots, dtype=np.int64)
        self._eval: EvalCounts | None = None

    def reset(self) -> None:
        self.arena.reset_all()
        self._steps[:] = 0
        self._advance(range(self.n_slots), movers=None)
        self._encode_all()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[EpisodeRecord | None]]:
        # A row's reward is from the perspective of whoever acted on it,
        # which is needed after the arena has moved on: snapshot the movers.
        movers = [row.side for row in self._rows]
        for slot, action in enumerate(actions):
            self.arena.apply(slot, int(action))
            self._steps[slot] += 1
        records = self._advance(range(self.n_slots), movers=movers)
        self._encode_all()
        rewards = np.array([0.0 if r is None else r.reward() for r in records], dtype=np.float32)
        dones = np.array([r is not None for r in records], dtype=bool)
        return rewards, dones, records

    def am_us(self) -> np.ndarray:
        """1.0 where the slot's current row is played as US."""
        return np.array([1.0 if row.side is Side.US else 0.0 for row in self._rows], dtype=np.float32)

    def start_eval(self, job: EvalJob) -> None:
        """Play the whole job now (the reference: one process, all the decks)."""
        if self._eval is not None:
            raise RuntimeError("an evaluation is already pending; finish_eval() first")
        self._eval = job.play(range(job.half))

    def finish_eval(self) -> EvalCounts:
        if self._eval is None:
            raise RuntimeError("no evaluation pending; start_eval() first")
        counts, self._eval = self._eval, None
        return counts

    def close(self) -> None:
        return None

    # -- internals -----------------------------------------------------------------

    def _opponent(self, policy_id: str) -> Opponent:
        opponent = self._opponents.get(policy_id)
        if opponent is None:
            opponent = self._opponents[policy_id] = self._resolve(policy_id)
        return opponent

    def _advance(self, slots: Sequence[int] | range, *, movers: Sequence[Side] | None) -> list[EpisodeRecord | None]:
        """Bring every slot to a learner decision, answering other seats'
        decisions with their opponents and closing finished games. With
        `movers` None (a reset) a finished game is simply restarted."""
        records: list[EpisodeRecord | None] = [None] * self.n_slots
        active = list(slots)
        while active:
            still_active: list[int] = []
            by_policy: dict[str, list[PendingRow]] = {}
            for slot in active:
                if self.arena.is_terminal(slot):
                    if movers is not None:
                        if records[slot] is None:
                            records[slot] = self._finish(slot, movers[slot])
                        # else: the slot's *replacement* game ended during
                        # this same fast-forward, before its first learner
                        # decision -- possible under scenario starts (a
                        # DEFCON-2 opening the opponent seat closes at
                        # once), impossible from the printed setup. No row
                        # was written for it, so there is nothing to
                        # reward: drop it and reset again.
                    self.arena.reset(slot)
                    still_active.append(slot)
                    continue
                policy_id = self.arena.policy_for(slot)
                if policy_id == self._learner:
                    self._rows[slot] = self.arena.row(slot)
                    continue
                by_policy.setdefault(policy_id, []).append(self.arena.row(slot))
                still_active.append(slot)
            for policy_id, rows in by_policy.items():
                choices = self._opponent(policy_id).choose(rows)
                if len(choices) != len(rows):
                    raise ValueError(f"opponent {policy_id!r} answered {len(choices)} of {len(rows)} rows")
                for row, choice in zip(rows, choices):
                    self.arena.apply(row.slot, int(choice))
            active = still_active
        return records

    def _finish(self, slot: int, mover: Side) -> EpisodeRecord:
        result = self.arena.result(slot)
        record = EpisodeRecord(
            winner=result.winner,
            mover=mover,
            seats=dict(result.seats),
            turn=result.turn,
            vp=result.vp,
            seed=result.seed,
            length=int(self._steps[slot]),
            margin=self.margin,
        )
        self._steps[slot] = 0
        return record

    def _encode_all(self) -> None:
        for slot, row in enumerate(self._rows):
            F.encode_into(row.observation, self.buffers, slot)


# -- shared memory ------------------------------------------------------------------

#: Fixed-width policy ids in the shared seat and record tables.
_ID = "S32"

#: Per-slot control fields shared with the workers, besides the layout slabs.
CONTROL_FIELDS: tuple[tuple[str, tuple[int, ...], Any], ...] = (
    ("actions", (), np.int64),
    ("rewards", (), np.float32),
    ("dones", (), np.uint8),
    ("am_us", (), np.float32),
    ("next_seats", (2,), _ID),  # the seats of the slot's *next* game, written one game ahead
    ("ep_flag", (), np.uint8),  # 1: a game ended in this slot this step; the ep_* fields describe it
    ("ep_winner", (), np.int8),  # 0 draw, 1 US, 2 USSR
    ("ep_mover", (), np.int8),  # 1 US, 2 USSR
    ("ep_turn", (), np.int16),
    ("ep_vp", (), np.int16),
    ("ep_seed", (), np.int64),
    ("ep_length", (), np.int32),
    ("ep_seats", (2,), _ID),
)

#: Shared with the workers besides the per-slot fields: the evaluation
#: in flight (`EvalJob` as JSON bytes) and each worker's counts of it.
EVAL_SPEC_BYTES = 4096
EVAL_FIELDS: tuple[tuple[str, tuple[int, ...], Any], ...] = (
    ("eval_spec", (EVAL_SPEC_BYTES,), np.uint8),
    ("eval_counts", (len(EvalCounts._fields),), np.int64),  # one row per worker
)

_STEP, _RESET, _STOP, _EVAL = 1, 2, 3, 4
_SIDE_CODE = {Side.US: 1, Side.USSR: 2}
_CODE_SIDE = {1: Side.US, 2: Side.USSR}


@dataclass(frozen=True)
class ArenaSpec:
    """How to build an `Arena` -- what a worker needs, since arenas do not cross processes."""

    n_games: int
    seed: int
    events: bool = True
    include_optional: bool = True
    starting_vp: int = 0
    us_bid: int = 0  # the tournament bid (Arena us_bid)
    margin: float = 0.0  # the terminal reward's final-VP weight (`EpisodeRecord.reward`)
    scenario_path: str | None = None  # a scenario bank (wopr.scenarios); each worker loads it
    scenario_frac: float = 0.0  # fraction of games started from the bank
    scenario_seats: tuple[str, str] | None = None  # (mover id, opponent id): scenario games seated by the arena itself


@dataclass(frozen=True)
class Slab:
    name: str
    shm_name: str
    shape: tuple[int, ...]
    dtype: str


def _slab_views(slabs: Sequence[Slab], segments: dict[str, SharedMemory]) -> dict[str, np.ndarray]:
    views = {}
    for slab in slabs:
        segment = segments.get(slab.shm_name)
        if segment is None:
            segment = segments[slab.shm_name] = SharedMemory(name=slab.shm_name)
        views[slab.name] = np.ndarray(slab.shape, dtype=np.dtype(slab.dtype), buffer=segment.buf)
    return views


def _decode(value: bytes | np.bytes_) -> str:
    return bytes(value).decode()


def _slot_ranges(n_slots: int, workers: int) -> list[range]:
    bounds = np.linspace(0, n_slots, workers + 1).astype(int)
    return [range(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def _write_eval_spec(buffer: np.ndarray, job: EvalJob) -> None:
    encoded = json.dumps(job.__dict__).encode()
    if len(encoded) > buffer.shape[0]:
        raise ValueError(f"eval spec of {len(encoded)} bytes exceeds the shared {buffer.shape[0]}")
    buffer[:] = 0
    buffer[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)


def _read_eval_spec(buffer: np.ndarray) -> EvalJob:
    raw = bytes(buffer.tobytes()).rstrip(b"\0")
    return EvalJob(**json.loads(raw.decode()))


def worker_main(
    index: int,
    slots: range,
    spec: ArenaSpec,
    slabs: Sequence[Slab],
    opponents: OpponentResolver,
    go: Any,
    done: Any,
    command: Any,
    learner: str,
    torch_threads: int,
) -> None:
    """One collector process: an `InProcessBackend` over `slots`, writing
    into its rows of the shared slabs. Blocks on `go`, releases `done`."""
    import torch

    torch.set_num_threads(torch_threads)
    for_worker = getattr(opponents, "for_worker", None)
    if for_worker is not None:  # e.g. StandardOpponents: its own RNG streams per process
        opponents = for_worker(index)
    segments: dict[str, SharedMemory] = {}
    shared = _slab_views(slabs, segments)
    lo, hi = slots.start, slots.stop

    def assign(slot: int, episode: int, rng: random.Random) -> dict[Side, str]:
        us, ussr = shared["next_seats"][slot]
        return {Side.US: _decode(us), Side.USSR: _decode(ussr)}

    bank = None
    if spec.scenario_path is not None:
        from wopr.scenarios import ScenarioBank

        bank = ScenarioBank.load(spec.scenario_path)
    arena = Arena(
        len(slots), seed=spec.seed, seat_assigner=assign, events=spec.events,
        include_optional=spec.include_optional, slot_offset=lo, total_slots=spec.n_games,
        starting_vp=spec.starting_vp, us_bid=spec.us_bid,
        scenario_bank=bank, scenario_frac=spec.scenario_frac, scenario_seats=spec.scenario_seats,
    )
    backend = InProcessBackend(
        arena, opponents, buffers={name: shared[name][lo:hi] for name in F.LAYOUT}, learner=learner, margin=spec.margin
    )
    done.release()  # built: the main process may now write the next seats
    try:
        while True:
            go.acquire()
            cmd = int(command[index])
            if cmd == _STOP:
                break
            if cmd == _RESET:
                backend.reset()
                shared["am_us"][lo:hi] = backend.am_us()
                shared["ep_flag"][lo:hi] = 0
                done.release()
                continue
            if cmd == _EVAL:
                # This worker's slice of the evaluation's decks; the training
                # games in `backend` wait untouched.
                job = _read_eval_spec(shared["eval_spec"])
                workers = shared["eval_counts"].shape[0]
                shared["eval_counts"][index] = job.play(_slot_ranges(job.half, workers)[index])
                done.release()
                continue
            rewards, dones, records = backend.step(shared["actions"][lo:hi])
            shared["rewards"][lo:hi] = rewards
            shared["dones"][lo:hi] = dones
            shared["am_us"][lo:hi] = backend.am_us()
            for offset, record in enumerate(records):
                slot = lo + offset
                shared["ep_flag"][slot] = record is not None
                if record is None:
                    continue
                shared["ep_winner"][slot] = 0 if record.winner is None else _SIDE_CODE[record.winner]
                shared["ep_mover"][slot] = _SIDE_CODE[record.mover]
                shared["ep_turn"][slot] = record.turn
                shared["ep_vp"][slot] = record.vp
                shared["ep_seed"][slot] = record.seed
                shared["ep_length"][slot] = record.length
                shared["ep_seats"][slot] = (record.seats[Side.US].encode(), record.seats[Side.USSR].encode())
            done.release()
    finally:
        for segment in segments.values():
            segment.close()


class SharedMemoryBackend:
    def __init__(
        self,
        spec: ArenaSpec,
        seat_assigner: SeatAssigner,
        opponents: OpponentResolver,
        *,
        workers: int,
        learner: str = LEARNER,
        worker_threads: int = 2,
        timeout: float = 120.0,
    ) -> None:
        if workers < 1 or workers > spec.n_games:
            raise ValueError(f"workers must be in [1, n_games={spec.n_games}], got {workers}")
        self.spec = spec
        self.n_slots = spec.n_games
        self._assign = seat_assigner
        self._rng = random.Random(spec.seed)  # the same stream `Arena` would draw seats from
        self._episodes = np.zeros(self.n_slots, dtype=np.int64)
        self._timeout = timeout
        self.wait_s = 0.0  # time spent waiting on workers, since the last `take_wait`
        self._eval_pending = False

        fields = [(name, (self.n_slots, *shape), np.dtype(dtype)) for name, (shape, dtype) in F.LAYOUT.items()]
        fields += [(name, (self.n_slots, *shape), np.dtype(dtype)) for name, shape, dtype in CONTROL_FIELDS]
        fields += [(name, (workers, *shape) if name == "eval_counts" else shape, np.dtype(dtype)) for name, shape, dtype in EVAL_FIELDS]
        self._segments: dict[str, SharedMemory] = {}
        self._slabs: list[Slab] = []
        for name, shape, dtype in fields:
            segment = SharedMemory(create=True, size=int(np.prod(shape)) * dtype.itemsize)
            self._segments[segment.name] = segment
            self._slabs.append(Slab(name, segment.name, tuple(shape), dtype.str))
        self._shared = _slab_views(self._slabs, self._segments)
        self.buffers = {name: self._shared[name] for name in F.LAYOUT}
        for array in self._shared.values():
            array[...] = 0

        self._ranges = _slot_ranges(self.n_slots, workers)
        self._command = mp.Array("i", workers, lock=False)
        self._go = [mp.Semaphore(0) for _ in range(workers)]
        self._done = [mp.Semaphore(0) for _ in range(workers)]
        self._write_seats(range(self.n_slots))  # episode 0: the games built with the arenas
        self._workers = [
            mp.Process(
                target=worker_main,
                args=(j, slots, spec, self._slabs, opponents, self._go[j], self._done[j], self._command, learner, worker_threads),
                daemon=True,
                name=f"wopr-collector-{j}",
            )
            for j, slots in enumerate(self._ranges)
        ]
        for process in self._workers:
            process.start()
        self._wait_all()

    # -- Backend ----------------------------------------------------------------------

    def reset(self) -> None:
        self._check_no_eval()
        self._episodes += 1
        self._write_seats(range(self.n_slots))  # what `reset_all` will read
        self._run(_RESET)
        self._episodes += 1
        self._write_seats(range(self.n_slots))  # one game ahead, for the first finishes

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[EpisodeRecord | None]]:
        self._check_no_eval()
        self._shared["actions"][:] = actions
        self._run(_STEP)
        shared = self._shared
        records: list[EpisodeRecord | None] = [None] * self.n_slots
        finished = np.flatnonzero(shared["ep_flag"])
        for slot in finished:
            us, ussr = shared["ep_seats"][slot]
            records[slot] = EpisodeRecord(
                winner=_CODE_SIDE.get(int(shared["ep_winner"][slot])),
                mover=_CODE_SIDE[int(shared["ep_mover"][slot])],
                seats={Side.US: _decode(us), Side.USSR: _decode(ussr)},
                turn=int(shared["ep_turn"][slot]),
                vp=int(shared["ep_vp"][slot]),
                seed=int(shared["ep_seed"][slot]),
                length=int(shared["ep_length"][slot]),
                margin=self.spec.margin,
            )
        if len(finished):
            # Those slots restarted during the step on the seats written
            # ahead; hand them the seats of the game after that one.
            self._episodes[finished] += 1
            self._write_seats(finished)
        return shared["rewards"].copy(), shared["dones"].astype(bool), records

    def am_us(self) -> np.ndarray:
        return self._shared["am_us"].copy()

    def take_wait(self) -> float:
        """Seconds spent waiting on workers since the previous call."""
        wait, self.wait_s = self.wait_s, 0.0
        return wait

    def start_eval(self, job: EvalJob) -> None:
        """Hand `job` to the collectors and return at once: each plays its
        slice of the decks while this process goes on (the PPO update).
        No `step`/`reset` until `finish_eval`."""
        self._check_no_eval()
        _write_eval_spec(self._shared["eval_spec"], job)
        self._shared["eval_counts"][...] = 0
        for j in range(len(self._workers)):
            self._command[j] = _EVAL
        for go in self._go:
            go.release()
        self._eval_pending = True

    def finish_eval(self) -> EvalCounts:
        """Wait for every collector's slice and add them up."""
        if not self._eval_pending:
            raise RuntimeError("no evaluation pending; start_eval() first")
        self._wait_all()
        self._eval_pending = False
        total = EvalCounts()
        for row in self._shared["eval_counts"]:
            total = total + EvalCounts(*(int(v) for v in row))
        return total

    def _check_no_eval(self) -> None:
        if self._eval_pending:
            raise RuntimeError("an evaluation is in flight on the collectors; finish_eval() first")

    def close(self) -> None:
        if not self._workers:
            return
        if self._eval_pending:  # let the collectors finish their slices before the stop
            try:
                self._wait_all()
            except RuntimeError:
                pass
            self._eval_pending = False
        for j, process in enumerate(self._workers):
            if process.is_alive():
                self._command[j] = _STOP
                self._go[j].release()
        for process in self._workers:
            process.join(timeout=10.0)
        self._workers = []
        for segment in self._segments.values():
            segment.close()
            segment.unlink()
        self._segments = {}

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- internals --------------------------------------------------------------------

    def _write_seats(self, slots: Sequence[int] | range | np.ndarray) -> None:
        table = self._shared["next_seats"]
        for slot in slots:
            seats = dict(self._assign(int(slot), int(self._episodes[slot]), self._rng))
            if set(seats) != {Side.US, Side.USSR}:
                raise ValueError(f"seat assigner must assign exactly US and USSR, got {sorted(s.value for s in seats)}")
            table[slot] = (seats[Side.US].encode(), seats[Side.USSR].encode())

    def _run(self, cmd: int) -> None:
        for j in range(len(self._workers)):
            self._command[j] = cmd
        started = time.perf_counter()
        for go in self._go:
            go.release()
        self._wait_all()
        self.wait_s += time.perf_counter() - started

    def _wait_all(self) -> None:
        for j, done in enumerate(self._done):
            while not done.acquire(timeout=self._timeout):
                if not self._workers[j].is_alive():
                    raise RuntimeError(f"collector {j} died (exit code {self._workers[j].exitcode})")
