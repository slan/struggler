"""Drive one Playdek game headless: a local seat answered by us, the other
seat played by Playdek's AI.

    pd = Playdek()                       # loads the DLL, Initialize() once
    game = pd.new_game(local_side=Side.USSR, ai_difficulty=AIDifficulty.HARD, seed=1)
    while (prompt := game.pump()) is not None:
        game.choose(prompt.options[0].index)
    print(game.result)

The DLL is a process-wide singleton with one current game: `new_game`
ends any game still running. The AI thinks on its own threads; `pump`
sleeps while it does and returns when the local seat has a decision to
make or the game is over.

How the listener protocol works (all of it recovered empirically, see
`docs/WOPR.md`): `UpdateGame` emits `{int32 type; int32 payload[]}` records
into a buffer and, on the calling thread, invokes the options listener when
the local seat must choose. The choice is handed back with
`SelectGameOption` on a later call; the DLL keeps the chosen moves in a
temporary buffer and emits `COMMIT_PLAYER_DECISION` when a whole action is
ready to be confirmed -- the app's "Commit" button, here
`CommitTemporaryMoveBuffer` straight away.
"""

from __future__ import annotations

import ctypes as C
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from struggler.engine.types import Side
from wopr.playdek import ffi
from wopr.playdek.ffi import AIDifficulty, EScenario, EventType, PlayerType

__all__ = ["AIDifficulty", "GameEvent", "GameResult", "Option", "Playdek", "PlaydekGame", "Prompt"]

LOCAL_ID = 0
AI_ID = 456  # what the app uses for its AI profile; any id other than the local one works
_VALID_TYPES = {int(t) for t in EventType}


@dataclass(frozen=True)
class Option:
    index: int
    selection_id: int
    hint: int
    hidden: bool
    text: str


@dataclass(frozen=True)
class Prompt:
    player_id: int
    text: str
    options: tuple[Option, ...]

    @property
    def visible(self) -> tuple[Option, ...]:
        return tuple(o for o in self.options if not o.hidden)


@dataclass(frozen=True)
class GameEvent:
    kind: int
    payload: tuple[int, ...]

    @property
    def name(self) -> str:
        return EventType(self.kind).name if self.kind in _VALID_TYPES else f"type{self.kind}"

    @property
    def fields(self) -> dict[str, int]:
        names = ffi.payload_fields(self.kind) or [f"v{i}" for i in range(len(self.payload))]
        return dict(zip(names, self.payload))

    def __str__(self) -> str:
        return f"{self.name} {self.fields}"


@dataclass(frozen=True)
class GameResult:
    winner_id: int
    win_type: ffi.GameOverType
    score: int  # the DLL's VP track: positive favours the US


class EventDecoder:
    """Walk an `UpdateGame` buffer: `{int32 type; int32 payload[n]}` records
    back to back, n fixed per type (`ffi.payload_fields`, which covers
    every type the DLL emits). A type without a known size cannot be
    skipped -- there is no length field -- so it is an error, not a guess."""

    def decode(self, raw: bytes, count: int) -> list[GameEvent]:
        ints = struct.unpack_from(f"<{len(raw) // 4}i", raw)
        out: list[GameEvent] = []
        pos = 0
        for _ in range(count):
            kind = ints[pos]
            fields = ffi.payload_fields(kind)
            if fields is None:
                raise ValueError(f"event type {kind} has no known payload size; buffer from here: {ints[pos:pos + 12]}")
            n = len(fields)
            out.append(GameEvent(kind, tuple(ints[pos + 1 : pos + 1 + n])))
            pos += 1 + n
        return out


class Playdek:
    """The loaded DLL. One per process; `new_game` starts the current game."""

    _instance: Playdek | None = None

    def __init__(self, install: Path | None = None, *, processors: int | None = None) -> None:
        if Playdek._instance is not None:
            raise RuntimeError("TwilightLib is already loaded in this process; reuse the Playdek instance")
        self.lib, self.root = ffi.load(install)
        self._game: PlaydekGame | None = None
        self._prompts: list[Prompt] = []
        # The callbacks must outlive every call into the DLL.
        self._on_options = ffi.GameOptionsListener(self._listener)
        self._on_save = ffi.SaveWorldDataFunc(lambda data, size, short: None)
        self._on_debug = ffi.DebugFunc(lambda msg: None)
        self.lib.SetDebugFunction(self._on_debug)
        self.lib.SetSaveDataFunc(self._on_save)
        self.lib.SetGameOptionsListener(self._on_options)
        # The DLL prints its Lua database load to stdout; nothing to do about it.
        self.lib.Initialize(str(ffi.lua_dir(self.root)).encode(), processors or os.cpu_count() or 1)
        Playdek._instance = self

    def _listener(self, player_id: int, prompt: bytes | None, n: int, options) -> None:
        opts = tuple(
            Option(
                index=o.optionIndex,
                selection_id=o.selectionID,
                hint=o.selectionHint,
                hidden=bool(o.isHidden),
                text=(o.optionText or b"").decode("utf-8", errors="replace"),
            )
            for o in (options[i] for i in range(n))
        )
        self._prompts.append(Prompt(player_id, (prompt or b"").decode("utf-8", errors="replace"), opts))

    def new_game(
        self,
        *,
        local_side: Side,
        ai_difficulty: AIDifficulty = AIDifficulty.HARD,
        seed: int = 0,
        scenario: EScenario = EScenario.STANDARD,
        additional_card_flags: int = 0,
    ) -> PlaydekGame:
        """Start a game. `additional_card_flags` is the app's optional-card
        bit set (optional cards, promo packs); 0 is the base deck."""
        if self._game is not None:
            self._game.close()
        self._prompts.clear()
        params = ffi.GameParameters(
            additionalCardFlags=additional_card_flags,
            scenario=int(scenario),
            chooseSidesMethod=int(ffi.EChooseSidesMethod.CREATORUSSR if local_side is Side.USSR else ffi.EChooseSidesMethod.CREATORUS),
            additionalInfluence=0,
        )
        players = (ffi.AppPlayerData * 2)()
        for seat, (pid, ptype, diff, name) in enumerate(
            [(LOCAL_ID, PlayerType.LOCAL, 0, b"local"), (AI_ID, PlayerType.AI, int(ai_difficulty), b"playdek")]
        ):
            players[seat].id = pid
            players[seat].userRatings[0] = players[seat].userRatings[1] = 1500
            players[seat].playerType = int(ptype)
            players[seat].aiDifficultyLevel = diff
            players[seat].name = name
        self.lib.StartGame(C.byref(params), 2, players, seed & 0xFFFFFFFF)
        self._game = PlaydekGame(self, local_side=local_side)
        return self._game


class PlaydekGame:
    def __init__(self, pd: Playdek, *, local_side: Side) -> None:
        self._pd = pd
        self._lib = pd.lib
        self.local_side = local_side
        self._buf = C.create_string_buffer(ffi.EVENT_BUFFER_BYTES)
        self._state = C.create_string_buffer(ffi.STATE_BUFFER_BYTES)
        self._decoder = EventDecoder()
        self.events: list[GameEvent] = []
        self.prompt: Prompt | None = None
        self.result: GameResult | None = None
        self._closed = False

    # -- the loop ------------------------------------------------------

    def pump(self, *, idle_limit: float = 300.0) -> Prompt | None:
        """Run the DLL until the local seat must decide (returns the prompt)
        or the game is over (returns `None`; see `result`). Raises if nothing
        happens for `idle_limit` seconds -- the AI never takes that long."""
        if self.prompt is not None:
            return self.prompt
        idle_since = time.monotonic()
        while self.result is None:
            n = self._lib.UpdateGame(self._buf, ffi.EVENT_BUFFER_BYTES)
            batch = self._decoder.decode(self._buf.raw, n) if n else []
            for ev in batch:
                self.events.append(ev)
                if ev.kind == EventType.COMMIT_PLAYER_DECISION:
                    self._lib.CommitTemporaryMoveBuffer()
                elif ev.kind == EventType.GAME_OVER:
                    winner, win_type = ev.payload
                    self.result = GameResult(winner, ffi.GameOverType(win_type), self._lib.GetGameCurrentScore())
            if self._pd._prompts:
                self.prompt = self._pd._prompts.pop(0)
                if self._pd._prompts:
                    raise RuntimeError("more than one pending option list")
                return self.prompt
            if batch:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since > idle_limit:
                raise TimeoutError(f"no events from TwilightLib for {idle_limit:.0f}s")
            else:
                time.sleep(0.001)
        return None

    def choose(self, option_index: int, data: int | None = None) -> None:
        """Answer the pending prompt with one of its option indices."""
        if self.prompt is None:
            raise RuntimeError("no pending prompt")
        if option_index not in {o.index for o in self.prompt.options}:
            raise ValueError(f"option {option_index} is not in the pending list")
        self.prompt = None
        if data is None:
            self._lib.SelectGameOption(option_index)
        else:
            self._lib.SelectGameOptionWithData(option_index, data)

    def play(self, policy, *, idle_limit: float = 300.0) -> GameResult:
        """Run to the end, answering every prompt with `policy(prompt) -> index`."""
        while (prompt := self.pump(idle_limit=idle_limit)) is not None:
            self.choose(policy(prompt))
        assert self.result is not None
        return self.result

    def new_events(self, since: int) -> Iterator[GameEvent]:
        yield from self.events[since:]

    # -- state queries ---------------------------------------------------

    def hand_count(self, player_id: int) -> int:
        self._lib.GetGamePlayerHandState(player_id, self._state, len(self._state))
        return ffi.GamePlayerHandState.from_buffer_copy(self._state).handCardCount

    def ai_state(self, player_id: int = AI_ID) -> ffi.GamePlayerAIState:
        self._lib.GetGamePlayerAIState(player_id, self._state, len(self._state))
        return ffi.GamePlayerAIState.from_buffer_copy(self._state)

    @property
    def score(self) -> int:
        return self._lib.GetGameCurrentScore()

    def defcon(self) -> int:
        return self._lib.GetPendingDefconLevel(LOCAL_ID)

    def deck_counts(self) -> ffi.GameDeckCounts:
        self._lib.GetGameDeckCounts(self._state, len(self._state))
        return ffi.GameDeckCounts.from_buffer_copy(self._state)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._lib.ExitCurrentGame()
            if self._pd._game is self:
                self._pd._game = None
