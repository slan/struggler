# Architecture

`struggler` is a state machine with a public API narrow enough that an
agent — human, scripted, or learned — can only interact with the game the
way a player at the table can.

The five mandates below are the reason this project exists. An
implementation that violates one of them is wrong, regardless of whether
it passes the tests.

## The five mandates

### 1. Pending-decision stack, not "one turn = one action"

At any point the engine has zero or more pending decisions, held on an
internal **stack** (not a plain FIFO queue) because card events
*interrupt*: resolving a decision can push new sub-decisions on top (e.g.
a `DUCK_AND_COVER` event triggers a "USSR: which country loses influence"
decision mid-resolution of something else), which must resolve before
control returns to the interrupted context.

- `pending_decision` always returns the top-of-stack frame.
- Resolving the top frame either pops back to the frame beneath it, or —
  if resolving it produced new sub-decisions — pushes those first.
- The engine never assumes whose decision is next; `Decision.actor` says
  so explicitly (`Side.US`, `Side.USSR`, or `Side.CHANCE`).

### 2. Atomic action space

Spending 4 Ops is four separate single-point-placement decisions, not one
action out of `C(countries, 4)`-many combinations. Target: **tens** of
legal options per decision, never thousands. If a legal-actions list is
ever in the hundreds, that decision is decomposed wrong and needs to be
broken down further.

**Exception**: physical mode's `DEAL_CARD` and the physical-hand-sourced
options on a few other kinds (`ACTION_ROUND_PLAY`/`HEADLINE_PLAY`/
`RANDOM_DISCARD`/`QUAGMIRE_DISCARD`/`HELD_CARD_DISCARD`, plus a few
`EVENT_CHOICE` candidate lists) may run into the hundreds early in a game.
This is narrowly scoped to decisions that only ever reach the physical-mode
operator console (see [BOTS.md](BOTS.md)) — never a bot/RL `Player`, which
is what this mandate exists to keep tractable for. The console presentation
layer never dumps a giant numbered menu (it matches free text against a
card's printed number or name instead); the `Decision.options`/
`legal_actions()` contract itself is unchanged — still the literal,
exhaustive, replay-log-faithful legal set.

### 3. Seeded, injectable RNG

`Engine(seed=...)` takes (or is handed) a seeded RNG. Same seed + same
action sequence → byte-identical resulting state, every time, on every
machine. Chance is never read from `random`/`os.urandom` directly anywhere
in the engine — always through the injected RNG.

Chance events (coup rolls, realignment rolls, headline-order ties, card
effects that roll dice) are **exposed as decisions** with
`actor=Side.CHANCE`, not resolved silently inside `step()`. The engine
draws the outcome from its own seeded RNG and calls `step()` with it
internally, so the roll still appears as an explicit, logged
`(Decision, Action)` pair — this is what makes replay logs a complete,
re-playable record of a game, dice included.

### 4. Per-player observation function

`observe(player)` is the *only* sanctioned way an agent sees the game. It
must never leak:

- the opponent's hand (card identities — count is public),
- the draw pile's order or the identity of undrawn cards,
- any other information hidden from that seat under the rules (e.g. an
  opponent's not-yet-revealed China Card possession is public in Twilight
  Struggle, but anything analogous in card effects must be modeled the same
  way — hidden fields simply absent from the returned view, never
  masked/zeroed in a way that leaks their existence via shape).

`observe()` is asymmetric by construction: `observe(Side.US)` and
`observe(Side.USSR)` are different objects, not the same object with a
"redact" flag.

### 5. Flat, serializable state

`GameState` is representable as a flat dict of JSON primitives (int, str,
bool, list, dict) with no custom encoder required. This is what makes
replay logs diffable, hashable, and greppable, and what makes
`serialize()`/`deserialize()` trivial to keep in sync — the wire format
*is* the internal shape, not a projection of it.

## Public API surface

```python
class Engine:
    def __init__(self, seed: int, ...): ...

    @property
    def pending_decision(self) -> Decision | None:
        """Top of the decision stack. None iff the game has ended."""

    def legal_actions(self) -> tuple[Action, ...]:
        """Legal actions for the CURRENT pending_decision only."""

    def step(self, action: Action) -> None:
        """
        Apply `action` to the current pending_decision, advance state,
        and update the decision stack (pop, or push interrupts).
        Raises if `action` is not in legal_actions().
        """

    def observe(self, player: Side) -> Observation:
        """Player-scoped view. See mandate #4."""

    def serialize(self) -> dict:
        """Flat, JSON-primitive dict. Full state, including RNG state."""

    @classmethod
    def deserialize(cls, data: dict) -> "Engine":
        """Inverse of serialize(); must round-trip exactly."""

    def determinize(self, side: Side, seed: int) -> "Engine":
        """A simulation copy for `side`'s lookahead: observe(side) preserved
        exactly, everything hidden from that seat resampled from `seed`.
        See below."""

    @property
    def is_terminal(self) -> bool: ...

    @property
    def winner(self) -> Side | None: ...
```

### Core types (dataclasses/enums)

Decisions and actions are typed dataclasses, not dicts — this is the
project's chosen tradeoff of type-safety and pattern-matchability
(important for both engine-internal logic and RL action-value tables) over
wire-nativeness. JSON-facing boundaries (`serialize()`, replay logs)
convert explicitly; nothing internal is a dict-in-disguise.

```python
class Side(Enum):
    US = "US"
    USSR = "USSR"
    CHANCE = "CHANCE"

class DecisionKind(Enum):
    PLACE_INFLUENCE = "place_influence"
    REALIGNMENT_ROLL = "realignment_roll"
    COUP_ROLL = "coup_roll"
    # ... see struggler.engine.types for the full set

@dataclass(frozen=True)
class Action:
    kind: DecisionKind
    payload: Mapping[str, Any]  # e.g. {"country": "Poland"}

@dataclass(frozen=True)
class Decision:
    id: int                       # monotonic, unique within a game
    actor: Side
    kind: DecisionKind
    options: tuple[Action, ...]   # == legal_actions() for this frame
    context: Mapping[str, Any]    # e.g. {"ops_remaining": 2}
```

`Decision.options` and `Engine.legal_actions()` return the same data;
`legal_actions()` exists as the ergonomic accessor.

## Determinization (search without leaking)

`Engine.determinize(side, seed)` is the sanctioned way for an agent to
*simulate*: it returns a full playable copy whose `observe(side)` equals
the original's exactly, with everything hidden from that seat resampled
from `seed` — the draw pile's order, the opponent's hand, a
committed-but-unrevealed opponent headline, and the RNG behind every
future roll. Searching the copy therefore reveals nothing `observe()`
would not (mandate #4 holds in spirit as well as letter); a physical-mode
game converts to an ordinary one, `HIDDEN_CARD` placeholders dealt real
identities from `hidden_pool`.

The copy carries `expose_chance_outcomes`: every d6 chance frame offers
all six outcomes as options — exactly physical mode's presentation of
chance (mandate #3: still an explicit `Side.CHANCE` decision, just not
pre-rolled) — so the simulator can enumerate an exact expectation or
sample from its own RNG. The flag serializes only when set; a live
game's serialization (and every golden replay) is unchanged. The one
consumer is the search player, docs/WOPR.md ("Search over the learned
value head").

## Reachability within one Operations spend (rule 6.1.1)

Placing Influence is atomic per point (mandate #2), but legality is *not*
re-derived from the live board after each point: "all markers must be
placed with, or adjacent to, friendly markers that were in place at the
start of the phasing player's Action Round" — a point placed earlier in the
same Ops spend does not itself unlock a further-away country later in that
same spend.

`Engine._ops_round_snapshot` freezes `board.influence` the moment an
Ops-driven placement chain begins (`_maybe_push_place_influence` /
`_maybe_push_bonus_influence`, threaded into `Board.is_reachable` via its
optional `influence` override), is reused for every point in that chain,
and clears once it ends. It round-trips through
`serialize()`/`deserialize()`, so a save/resume mid-chain cannot reopen the
chaining bug.

Event-driven placement (`push_event_influence`) is unaffected — rule
6.1.1's own exception excludes it, and it was never reachability-gated in
the first place, since its candidates are fixed lists rather than
adjacency-derived.

## The Ops-only toggle

`Engine.new_game(..., variants={...})` turns optional-rule spaces on (the
only one, `"chinese_civil_war"`, is the space alone; see
[LIMITATIONS.md](LIMITATIONS.md)).

`Engine.new_game(..., us_bid=N)` plays the tournament bid of the official
rules (11.1.4): N extra US Influence, placed by the US once the regular
setup placements are done, only into countries that hold US influence at
that moment and never past two more than what control needs (11.1.4.1) —
ordinary `PLACE_INFLUENCE` decisions with `"bid": True` in their context.
0 (the default) is the printed game.

`Engine.new_game(..., events=False)` runs the game with the card-event
layer switched off entirely: all 110 cards exist as data and are playable
for their Ops value, the headline phase, space race, China Card and DEFCON
degradation all function, but **no card event ever fires** — every card
play is Ops. `events=True` is the default.

The toggle exists because "a complete game is playable start to finish
through the public API alone, with no event mechanics involved" is a
property worth being able to test in isolation. `serialize()` carries
`events_enabled` alongside `turn_effects` and `game_effects`, so a saved
game round-trips its event state either way (mandate #5).

## End of turn

The required Military Operations are settled as one move of the VP
marker: each side's shortfall against DEFCON goes to its opponent, the
two netted before the marker moves, and the automatic victory at 20 VP
is checked on the result -- not after the first of the two penalties
alone (`Engine._end_of_turn`); the penalty is paid before a held scoring
card is revealed and loses the game (turn steps E, then F).

## Starting VP (a handicap)

`Engine.new_game(..., starting_vp=N)` opens the VP track at N (US-positive)
instead of 0 — what a tournament bid does: the player who wants the USSR
seat gives the US that many VP. Nothing else changes; the printed game is
`starting_vp=0`, the default. `serialize()` carries `starting_vp` beside
`vp`, and a `new_game` replay log records it in its header, so a
handicapped game round-trips and replays like any other (mandate #5).
The training arena uses it to take the seat out of the result
(docs/WOPR.md).

## Rules version

`rules.json` carries `rules_version`, exported as
`struggler.engine.RULES_VERSION`. **Bump it in the same commit as any
change that alters how the rules resolve** — a fix, a clarified ruling,
a data correction — however small. It is not an API version: the
public API does not change with it. It exists because things are
measured on this engine — a bot's win rate, a trained policy's rating —
and a measurement means nothing without the game it was taken on. The
training arena records it with every run and keeps one ladder of frozen
versions per rules version (docs/WOPR.md); a rules change that makes an
existing bot lose by its own hand must fail a test (the Greedy sanity
test in `tests/test_greedy.py`), not an evaluation. Version 1 is the
engine before the August 2026 fixes; version 2 is after the nine of
them (`b55daf5`); version 3 adds the four the match operator found
(an opponent's card or an event whose play restriction is unmet is not
offered "for its event", Independent Reds' targets, the US/Japan Pact
leaving the USSR's Influence in Japan, a war's penalty counting adjacent
countries only, Blockade's and the traps' discards counting the turn's
Ops modifiers -- docs/WOPR.md) and the ten the lockstep differ found
over 280 seeds (a card whose event cannot happen is discarded, not
removed; Ask Not may discard scoring cards; Missile Envy's exchanged
card is played as its event when it is the taker's; Defectors revealed
by Five Year Plan cancels the USSR headline or scores the US; We Will
Bury You is paid at a trapped US round; DEFCON 1 loses the phasing
player; UN Intervention on any opponent-event card; the bonus
Realignment attempt stays in its region; the China Card's and Vietnam
Revolts' bonuses stack; Cuban Missile Crisis cancelled by either side,
and before the banned coup; the end-of-turn Military Operations
penalties netted before the marker moves, and paid before a held
scoring card loses the game; Marine Barracks Bombing
removing two points, not two countries; Willy Brandt prevented once
Tear Down This Wall has been played). Version 4 is one ruling the
official AI exposed (docs/WOPR.md, `gs-trace-304`): DEFCON 1 reached
during the headline phase loses the player who *played* the resolving
headline event — rule 4.5's note — not the side that moved the marker
(a USSR-headlined Grain Sales hands the US a card it coups with; the
USSR loses). Version 5 is two fixes the r4b2v4/scen1 easy evals
exposed (docs/WOPR.md, the ninth pass): an event-granted free
Realignment chain (Tear Down This Wall, Junta) keeps the card's
terms — its named countries and the DEFCON-geography exemption — for
every roll, not just the first; and a war card whose event is
prevented (Arab-Israeli War under Camp David Accords) no longer
triggers Flower Power — the official AI's reading, adopted over the
card's bare "for Ops or for Event". Version 6 makes event-granted
Operations declinable (every granting card's "may"; the official AI
declines them). Version 7 is the DLL's reading of the traps: a seat
still in Bear Trap or Quagmire is exempt from the held-scoring-card
loss at the turn's end and carries the card over. Version 8: Wargames
ends the game on the VP total as it stands after the 6-VP gift —
"without Final Scoring", the printed text — where the engine had
final-scored every region (docs/WOPR.md, the twenty-first pass: the
official AI's Wargames endings desynced on exactly that difference).

## Opening deal order

`Engine.new_game(..., deal_after_setup=True)` deals the opening hands
after the USSR's 6 and the US's 7 opening placements instead of before
them, so the placements are made without sight of a hand. The default
deals first, as the printed setup sequence does; the option is Playdek's
order, which `wopr.playdek` has to follow in physical mode because its
`DEAL_CARD` answers come from the other program's deal (docs/WOPR.md).
`serialize()` carries the flag, and a `new_game` replay log records it
in its header when set (mandate #5).
