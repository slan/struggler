# WOPR and Joshua: the self-play arena and the learned bot

> *"A strange game. The only winning move is not to play."* — WOPR, after
> learning tic-tac-toe by playing itself. We are aiming a little higher.

This is the contract: layouts, APIs, semantics. The idea behind it, and
the results so far, are in [JOSHUA.md](JOSHUA.md); frozen runs live under
[`baselines/`](../baselines/README.md) (`python -m wopr.baseline`).

Two packages, one contract:

- **`struggler.bots.joshua`** — the learned `Player`. `features.py` is the
  encoding contract (numpy only), `model.py` the network (torch),
  `player.py` the `Player` that `build_player("joshua")` constructs.
- **`wopr`** (`src/wopr/`) — the training arena: many engines stepped
  decision by decision, Stable-Baselines3 PPO on top, a checkpoint pool
  for self-play, a rating ladder for evaluation.

Install with `pip install -e ".[wopr]"` (training) or `".[joshua]"`
(inference only). The engine itself depends on neither torch nor SB3.

```
python -m wopr.train --run first --games 2000 --workers 8   # train (resumes if the run exists)
python -m wopr.eval --games 200 first=runs/first/joshua.pt random greedy
python -m wopr.loop --run first --generations 3 --generation-games 4000   # train, evaluate, gate, promote
python src/main.py --ussr joshua --joshua-checkpoint runs/first/joshua.pt
```

## The training process, end to end

A version of Joshua is born, compared, and promoted in three stages —
and a ladder is opened by a fourth, stage 0. The pieces are specified in
the sections below; this is the order they run in and what each one
decides.

0. **The bootstrap** (`wopr.bootstrap`, since r3): a ladder's `v1` is
   the recipe trained from scratch until the yardstick says stop, not
   until a budget runs out. The run evaluates its latest checkpoint
   against Greedy every 500 games *as it trains* — 200 games, argmax,
   100 a seat, a fresh deck seed each tick, played on the collectors
   while the PPO update runs — and the stop rule reads the **per-seat
   rolling mean over the last two ticks** (200 games a seat, ±0.07):
   both seats ≥ 0.75 → a confirmatory 600-game evaluation on fresh
   decks, and the run stops only if both seats clear 0.75 there too; no
   new best of the *overall* rolling mean (400 games, ±0.05 — the
   weaker seat's alone is too noisy for a four-tick window: r3's first
   attempt stopped on it at 6,500 games while the curve was still
   climbing) for four ticks (~2,000 games) → stop at the plateau; a cap
   of 20,000 games → stop. Whatever
   stopped it, the last evaluated checkpoint is frozen as `v1` with the
   full protocol and its README entry says which rule fired.
   `runs/<run>/bootstrap.csv` has every tick's decision. The Greedy
   curve in `metrics.csv` (`eval_*`) is the one that is free of the
   opponent mix — the training win rates are against the run's own pool.
1. **A fresh run** (`wopr.train --recipe v11`, or `wopr.ab` which wraps
   it): random weights, PPO, 64 games in flight. Every training game is
   against *itself* — half of them with both seats the learner
   (self-play), half with the learner on a random seat against a
   snapshot of its own past checkpoints, taken every 5 updates and
   sampled by **prioritised fictitious self-play** (PFSP): the snapshots
   it still loses to come up most. While the pool is empty those games
   are self-play too. Nothing else plays: not the champion, not
   Greedy, not `random` — they are yardsticks, never sparring partners
   (anchors exist as flags and are off since r1 v5; the archived
   notebook, docs/archive/JOSHUA-r1.md, says why).
   The reward is the outcome, ±1, on each row when its game ends, for
   whoever moved on that row.
2. **The comparison** (`wopr.ab`): after the recipe's budget — 8,000
   games, ~50 min — the run plays, argmax, on three eval seeds, the
   *control* (the frozen version trained with the same recipe and
   budget: v11), the champion, Greedy, and itself. Level with the
   control means whatever changed was neutral for learning; ahead
   means it is a candidate for stage 3. One row in
   `baselines/EXPERIMENTS.md` either way.
3. **The loop** (`wopr.loop`): the same run continued, optimizer and
   pool carried over, 4,000 games a generation, its opponents still
   only its own pool. After each generation the latest checkpoint (the
   *challenger*) plays the *champion* — the newest frozen `vN` — on
   every eval seed. The gate is the **worst seed's** win rate against
   the champion, ≥ 0.55: clear it and the challenger is frozen as the
   next `vN` with the full protocol (200 games per opponent per seed,
   three seeds, a sampled pass, against every earlier baseline), gets
   its README entry and becomes the champion; miss it and the run just
   trains on. Three generations under 0.5 stop the loop. **Elo is
   descriptive**, fitted afterwards with `random` at 0 so versions
   compare across time; nothing is selected by it — selection is the
   gate, and only the gate.

So a line looks like v11 (stage 1–2, 8,000 games) → v12 → … → v16
(stage 3, one promotion per cleared gate). A fresh run is the answer
to "did this change alter what is learned"; the loop is the answer to
"is it still improving". Two things the process does **not** do, by
design: train against the champion or any fixed opponent (a fixed
opponent it always beats or always loses to is a constant reward —
r1 v2 and v4 in docs/archive/JOSHUA-r1.md), and shape the reward (the final score in the
terminal reward made both seats play for the track — the margin
experiment there).

### Decision points

Each stage ends in a decision with a rule written down here, so that
the next step is read off the numbers rather than argued from them.

- **The rules changed** (`RULES_VERSION` bumped, docs/ARCHITECTURE.md).
  Re-rate the yardsticks on the new engine before anything else: Greedy
  against itself and the champion against Greedy (`wopr.diagnose`,
  `wopr.eval`). Either moved beyond noise → the ladder is archived as
  it stands under `baselines/r<old>/`, the new one starts at `v1` from
  the bootstrap, and r<old>'s findings about the *game* are unverified
  until re-measured. Neither moved → same ladder, note the bump. (r3
  skipped the re-rating: thirty-two fixes landed between r2/v3 and the
  r3 engine, a dozen of them after the bump, so the ladder was restarted
  on the count alone.)
- **The bootstrap's stop** (`wopr.bootstrap`). *Confirmed* → `v1` is at
  the yardstick; into the loop. *Plateau* → `v1` is what the recipe
  reaches on this game; `wopr.diagnose` it before the loop, since the
  loop's gate against `v1` will say "better than v1", not "good". *Cap*
  → still improving at 20,000 games: continue the run (`train.py` or
  the loop resume it, ticks included) rather than restart.
- **A clean run against its control** (`wopr.ab`, same recipe and
  budget). Worse (below 0.45 on the worst seed) → the change hurt
  learning; revert or explain before continuing. Level (every seed
  within 0.45–0.55) → neutral for learning; keep it if it was wanted
  for another reason. Better (above 0.55 on every seed) → a candidate
  for the loop, `--init` from it or continue its run.
- **A generation against the champion** (`wopr.loop`). Worst seed ≥ 0.55
  → promote; below → train on. **Plateau declared at two misses in
  three generations**: stop the loop, do not spend a fourth.
- **At a plateau.** Run `wopr.diagnose` on the champion and record it.
  The next experiment is chosen from what it shows, and its ledger
  row is written *before* training: the question, the control, the
  metric that decides it, the budget, and the rule (what number means
  yes). An experiment without a pre-written row is exploration, which
  is fine, but it is logged as that.
- **A negative result** closes its question in the notebook with the
  numbers and is not retried with a tweak unless the diagnosis says
  why the first attempt missed.

## Why this shape

The engine's mandates are what make a learned player cheap to build here:

- **Mandate #2 (atomic action space)** means a decision never has more
  than a few dozen options, so Joshua's action is simply *an index into
  `Decision.options`*. There is no global action id, no hand-built
  hierarchy: one policy head scores each legal option from its own
  features, and illegal ones do not exist.
- **Mandate #4 (`observe(side)`)** is the only input. Nothing in the
  encoding can leak the opponent's hand or the deck because the
  `Observation` does not contain them.
- **Mandate #3 (seeded RNG)** makes every training game reproducible from
  `(seed, slot, episode)` and lets evaluation replay the same decks for
  every pair of policies.
- **Mandate #1 (decision stack)** is mirrored by the arena's API, which
  is *decision-centric*: it never asks "whose turn is it", it asks "which
  games are waiting on which policy".

## The layout (`features.py`)

The layout is the single source of truth for how a decision becomes
arrays. It is versioned (`LAYOUT_VERSION`); checkpoints record the version
they were trained against and refuse to load against another. Any change
to a shape, index, or vocabulary below bumps it.

Everything is **from the mover's point of view**: "my" and "their", never
US and USSR, plus one `am_us` flag. One network therefore plays both
seats, and a self-play game trains both perspectives at once.

| Array | Shape | dtype | Contents |
| --- | --- | --- | --- |
| `board` | `[85, 15]` | float32 | per country: my/their influence (/5), net, my/their control, stability (/5), battleground, adjacency to my/their superpower, region one-hot (6) |
| `card_loc` | `[110]` | int64 | per card, one of 9 locations: unseen, my hand, discard, removed, China Card mine/theirs × face up/down, not yet in play |
| `globals` | `[102]` | float32 | `am_us`, DEFCON, VP (signed for me), turn, action round, phase one-hot, hand sizes, draw pile size, space race positions/attempts, military ops, China Card flags, every turn/game effect flag (sided ones as me/them), decision-kind one-hot (23), decision-context scalars |
| `focus` | `[2]` | int64 | card indices the decision is about: the card/event being resolved, the opponent's revealed headline; `110` = none |
| `opt_feats` | `[96, 46]` | float32 | per option: one-hot over the closed payload vocabulary (play modes, ops types, event/ops order, event-choice words, regions), numeric value, is-country/is-card/is-empty/other flags, position |
| `opt_country` | `[96]` | int64 | the option's country, `85` = none |
| `opt_card` | `[96]` | int64 | the option's card, `110` = none |
| `opt_mask` | `[96]` | int8 | 1 for each legal option; the action is an index into this |

The 85 country rows are every space the data knows, optional-rule ones
included (`Board(variants=Board.VARIANTS)`): the vocabulary must not
change with a game's variants. In a standard game the Chinese Civil War
row is simply always zero.

`K_MAX = 96` bounds the option count (the largest legal set is "every
country"); exceeding it raises rather than truncates, because it would
mean a decision is decomposed wrong. An engine effect flag missing from
`TURN_EFFECTS`/`GAME_EFFECTS` also raises, and
`tests/test_joshua_features.py` greps the engine source so that drift is
caught by the suite, not by a rare event mid-game. Payload words outside
`OPTION_VOCAB` degrade to the `other` flag plus position — by design for
exactly one value a standard game produces, realignment's
`{"country": "stop"}` (the `other` flag on a `REALIGNMENT_TARGET`
option *is* that meaning), so adding it cost no layout change.

**Hidden information is represented, not guessed at.** The `unseen` card
location is the union of the draw pile and the opponent's hand — which is
exactly what the rules let a player know — alongside the public counts.

### The layout is the backend contract

`wopr.arena.Arena` fills these arrays from in-process Python engines. That
is an implementation, not the design: a shared-memory server, an engine
rewritten in another language, or a multi-seat remote arena would fill
the *same* batch-first buffers and answer the *same* "which slots wait on
which policy" question, and nothing above the arena would change. Keep
new state out of Python objects and inside the layout.

## The model (`model.py`)

`JoshuaNet`, ~286k parameters at defaults. The architecture as a
picture: [WOPR_arch.svg](WOPR_arch.svg) (source `WOPR_arch.dot`;
regenerate with `dot -Tsvg docs/WOPR_arch.dot -o docs/WOPR_arch.svg`
after any change here). In prose:

- **Board as a graph.** Country features pass through a small graph
  network over the fixed adjacency (dense, row-normalised, self-loops):
  `relu(W_self h + W_nb (A h))` per layer. Reachability, coup targets,
  realignment bonuses are all adjacency facts; the node latents learn
  them. There is one map, so memorising that Poland is Poland is the
  point — no node shuffling, no coordinate stripping.
- **Cards** as `card_embedding + location_embedding`, mean-pooled per
  location into a "what is where" summary; `focus` cards as raw
  embeddings.
- **State latent** from attention-pooled nodes (queried by the globals),
  the globals, the card summary, and the focus cards. The value head reads
  it.
- **One option head for every decision kind**: each option is scored from
  `[state latent, option features, node latent of its country, embedding
  of its card]`, masked, softmaxed. The decision kind lives in `globals`,
  the option's meaning in its own row, so a coup target, a headline card,
  and an event branch are all "just options". `forward` is `encode`
  (node and state latents) followed by `score_options`, which runs the
  head on the `(row, slot)` pairs the mask selects rather than on all
  `K_MAX` padded slots — about ten of 96 are legal — and fills the rest
  with the mask value. Same logits, a tenth of the head's work;
  `tests/test_joshua_player.py` pins it against the dense computation.

The final option layer is initialised near zero so the untrained policy
is near-uniform over legal options and early rollouts explore.

## Search over the learned value head (`bots/joshua/search.py`)

`SearchPlayer` is inference-time lookahead over a trained checkpoint —
no training change, no layout change; pre-registered in JOSHUA.md
(2026-08-25). It is its own named policy everywhere it is measured:
gates and `wopr.baseline` stay raw-argmax.

**The simulation state is `Engine.determinize(side, seed)`** — a copy
whose `observe(side)` is preserved exactly while everything hidden from
that seat is resampled: the draw pile's order, the opponent's hand, a
committed-but-unrevealed opponent headline, Our Man in Tehran's queue
(for the USSR), and the RNG behind every future roll. The copy is
information-equivalent to the mover's own observation (mandate #4), so
searching it can leak nothing; it deliberately forgets reveals the
engine does not track as knowledge (a shown hand, the box-4 headline
peek) — conservative, never leaky. On the copy
`Engine.expose_chance_outcomes` is set: every d6 CHANCE frame offers
all six outcomes, physical-mode style (mandate #3 — chance is still an
explicit decision), so the simulator owns the dice. A physical-mode
game (the Playdek bridge) converts to an ordinary one, `HIDDEN_CARD`
placeholders dealt from `hidden_pool`. The player reaches the engine it
is seated at through `bind(engine)` — called by `src/main.py` and
`operator.play_match` — and uses it solely to call `determinize()`.

Two evaluators, one harness:

- **`evaluator="value"` (one-ply search).** Each legal option's branch
  is rolled forward — through chance, through the mover's *own*
  subsequent decisions along the policy's argmax (`my_steps`, 12:
  stopping at the mover's own next atomic decision would price an
  `OPS_TYPE` "coup" before any target is picked, the head's blind
  spot), and through the opponent's reply the same way (`opp_steps`,
  18: an event can hand them a long chain — Marshall Plan's seven
  placements plus their own play — and the caps must outlast a whole
  action or an end-of-turn terminal is never reached). It is scored
  twice: the value head at the opponent's first real decision (their
  view — prices the threat they hold, blind to the mover's remaining
  hand) and again at the mover's next own decision (hand-aware, any
  end-of-turn terminal played out for real in between); the branch
  takes the **minimum**, each estimate covering the other's blind
  spot, playout terminals floored the same way. A branch that
  consumed no randomness is exact from one simulation; one that did
  averages `k` determinizations (4), one die enumerated exactly
  (`chance_cap` 6), deeper dice sampled. Terminals score ±1 (draws
  0), the head's estimate clamped to ±0.99 so a certain result always
  outranks an estimate; an unscoreable branch counts 0. **The
  policy's own pick then stands unless another option's searched
  value clears it by `margin` (0.3)**: per-option value noise (~0.1)
  otherwise out-shouts trained play — pure value-argmax measured
  0.02 against the raw checkpoint — while a real blunder (a found
  loss against an ordinary position) differs by ~1.0 and is overridden
  loudly. ~2–5 min a game against the raw checkpoint on one core.
- **`evaluator="terminal"` (the veto — the ablation search subsumes).**
  Options are probed in the policy's own preference order and the first
  that is not a *provable* loss is played. Provable means: within the
  current play (the `_ars_played` fence), through **any** opponent
  choice, **every** own choice and **every** exposed die outcome, the
  game ends against the mover — the shape of both the DEFCON self-kill
  and the granted-coup gift. The proof search is budgeted
  (`probe_budget`, 300 engine copies at ~0.9 ms each) and stops after
  three of the prober's own choice points (an own ops chain fans out by
  tens per point and cannot itself end the game); anything unprovable
  is not a veto. Every option provably lost → the policy argmax stands.
  ~2 min a game against Greedy on one core.

Wiring: `wopr.playdek.eval --policy search=ckpt.pt | veto=ckpt.pt`
(easy/hard evals; the AI's 15 s per decision dwarfs the search),
`python -m wopr.search_eval --policy search=ckpt.pt --opponent
greedy|joshua=ckpt.pt [--bid N]` (the in-repo sanity eval: full engine
games through `runner.play_game`, since the search needs the live
engine — hundreds of games, not the arena's tens of thousands), and
`src/main.py --us joshua-search|joshua-veto`. Tests:
`tests/test_search.py`.

## The arena (`wopr/arena.py`, `wopr/backend.py`, `wopr/vec_env.py`)

`Arena(n_games, seed, seat_assigner)` owns N engines. Each game's two
seats are assigned *policy ids* (strings) when it starts; the arena only
groups pending decisions by id (`pending()`) and applies option indices
(`apply()`), resolving `Side.CHANCE` frames itself — their single
pre-rolled option is not a choice. `play_out(arena, {id: policy})` runs
everything to completion in batched rounds; it is what evaluation uses.

A **backend** (`wopr/backend.py`) steps the games and fills the layout
buffers: given one option index per slot, bring every slot to its next
learner decision and write its row. After the learner's action it
fast-forwards through chance and through every decision belonging to a
*non-learner* seat (answered in batch by the registered opponents) and
stops at the learner's next decision — which, in self-play, may belong to
the other side. Each row's reward is the game outcome **for that row's
mover** (+1/−1/0) on the row after which the game ended, returned as an
`EpisodeRecord`. Two backends answer the same question:

- `InProcessBackend`: N engines in this process — the reference.
- `SharedMemoryBackend` (`--workers k`): k collector processes, each an
  `InProcessBackend` over a contiguous slice of the slots, writing
  straight into shared memory. **The layout is the transport**: every
  layout array is one shared slab `[n_slots, ...]`, a worker owns its
  rows, the main process reads the whole slab after the step. Actions,
  rewards, dones, the mover of each next row and the record of a game
  that ended this step are fixed-shape shared arrays too; nothing is
  pickled on the step path, and the only signal per step is one
  semaphore release per worker each way. Seats are decided in the main
  process — the seat assigner sees the pool and the anchor schedule
  there — and handed over one game ahead in a shared table a worker
  reads when a slot resets. Game seeds are `(run seed, global slot,
  episode)` in both backends, so a deterministic configuration plays the
  same games through either, step for step; the suite pins it. The game
  spec both play — events, `starting_vp`, and since r3 the tournament
  bid `us_bid` (rule 11.1.4, `--bid` on every wopr CLI) — is the
  arena's; a run's `config.json` records it, and a ladder under a bid
  lives at `baselines/r<N>-bid<B>/`, apart from the printed game's, for
  the same reason ratings do not cross rules versions. Each
  collector resolves its own opponents (`StandardOpponents.for_worker`:
  same policies, its own RNG streams) and runs pool-net inference over
  its own slots, so that batching is kept inside the worker.

Both backends answer one more question, between rollouts: **play this
evaluation** (`start_eval(EvalJob)` / `finish_eval() → EvalCounts`). An
`EvalJob` is a checkpoint against a scripted opponent on a deck seed,
argmax, half the games on each seat — the same games `eval.py`'s pair
would play, because `play_slice` seeds decks by global slot and the
slices of `[0, half)` add up to the pair. The in-process backend plays
it on the spot; the shared-memory one hands it to the collectors
(`_EVAL`, the job as JSON in a shared buffer, one row of counts back per
worker) and returns at once, so the collectors play their slices while
the main process runs the PPO update — the time they would otherwise
spend idle — and `finish_eval` waits for the sum. No `step`/`reset`
while one is out; the training games in each collector wait untouched.

`WoprVecEnv` adapts a backend to SB3's `VecEnv`: one env per slot,
`infos[i]["episode"]` from the records, no `terminal_observation` (SB3
reads it only to bootstrap truncated episodes; games here end for real).

### Alternating-perspective GAE (`wopr/buffer.py`)

Rows of one slot can alternate sides. Values and rewards are always the
mover's; in a zero-sum game the next state's value for its mover is the
negative of its value for me, so the bootstrap flips sign whenever the
mover changes between consecutive rows:

```
delta_t = r_t + gamma * s_t * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * s_t * A_{t+1}
s_t     = +1 if mover(t+1) == mover(t) else -1
```

With a fixed learner seat every `s_t` is +1 and this is plain SB3 GAE.
With both seats played by the learner it is what lets one network, one
buffer, and one PPO update learn both seats from the same games. The
mover of the bootstrap observation comes from `WoprVecEnv.current_am_us()`.
`tests/test_wopr.py` pins the hand-computed advantages.

### Opponents and the mix

`wopr/opponents.py`: `RandomOpponent`, `PlayerOpponent` (any engine
`Player` — Greedy, First), `NetOpponent` (a frozen checkpoint, batched,
sampling by default so it cannot be exploited line by line).

`train.py` draws each game's seating from `--self-play` (both seats the
learner), `--vs-pool` (learner on a random seat against a pool snapshot),
and the remainder against `--anchor`: `random`, `greedy`, `first`, a
frozen checkpoint as `name=ckpt.pt` (policy id `ckpt:<path>`, seated as
a sampling `NetOpponent`; unlike a `--pool-seed` snapshot its share is
the mix's remainder, fixed — the anchor slot is never PFSP-reweighted,
which is the point: an opponent the learner must keep facing whether or
not it beats it), or a schedule such as `random,greedy`
(`pool.AnchorSchedule`), which walks the list in order and promotes once
the learner's win rate over the last `--anchor-window` anchor games
reaches `--anchor-promote`; the last anchor is kept for good, and
`metrics.csv` records the current one. A terminal reward against an
opponent the learner never beats is a constant, and so is one against an
opponent it always beats
(docs/archive/JOSHUA-r1.md, v2 and v4). Fractions summing to 1 leave no anchor games at
all: pure self-play against the pool. While the pool is empty, pool
games are self-play.

`--handicap N` opens every training game with the US N VP ahead
(`Engine.new_game(starting_vp=N)`, carried to the collectors in
`ArenaSpec`): a tournament bid for the USSR seat. With strong policies
the USSR seat won most games between equals on rules version 1
(docs/archive/JOSHUA-r1.md; to be re-measured), so the
terminal reward of a US-seat row says mostly which seat it was; a bid
that brings the seats to even makes games turn on play. Evaluation —
the loop's gate, `wopr.baseline` — stays at the printed game;
`wopr.eval --handicap N` plays a pair under a bid, which with a policy
against itself (`a=x.pt b=x.pt`) measures the USSR edge at that bid.

### Scenario-seeded starts (`wopr/scenarios.py`)

A *scenario bank* shapes the initial-state distribution of training
games without touching the opponent mix (the scenario-seeded self-play
arc, JOSHUA.md). `python -m wopr.scenarios --out
scenarios/defcon2-gift.jsonl --games 400 --bid 2 --policy random`
harvests states from scripted games — every first state per (turn,
action round, mover) where a named predicate matches, at most
`--per-game` per game. The bank is JSONL: a header recording the game
spec it was harvested under (`us_bid`, `starting_vp`, `events`,
`include_optional` — the arena refuses a bank whose spec differs from
its own) and the generator, then one `Engine.serialize()` state per
line. Predicates: `defcon2_gift` — an `ACTION_ROUND_PLAY` decision at
DEFCON 2 with a granted-op gift in the mover's hand (CIA Created for
the USSR seat, Lone Gunman for the US: the forced-endgame shape the
search arc closed on).

Starting from an entry never replays its game:
`ScenarioBank.start(index, seed)` deserializes and re-hides the state
with `Engine.determinize(mover, seed)` (`expose_chance_outcomes`
cleared — a training game, not a search copy), so the mover's
observation is preserved exactly while the deck order, the opponent's
hand and every future roll are resampled. One entry is a distribution
over games, all information-equivalent to what the mover knew
(mandate #4).

Wiring: `Arena(..., scenario_bank=, scenario_frac=)` draws each game's
start from a pure function of the game seed — scenario or printed
setup, the entry, the determinize seed — so k sliced arenas play
exactly the whole arena's games and both backends stay step-for-step
identical (`ArenaSpec.scenario_path`/`scenario_frac` carry it to the
collectors, each of which loads the bank once). `wopr.train
--scenarios bank.jsonl --scenario-frac 0.25` is the flag pair;
`config.json` records both. Evaluation — the loop's gate,
`wopr.baseline`, `--eval-every` — stays at the printed game: the bank
shapes what is *practiced*, never what is *measured*.
`tests/test_scenarios.py` pins the predicate, the resampling, the
slice invariant and the spec check.

`Arena(..., scenario_seats=(mover_id, opponent_id))` additionally
seats a scenario-started game itself — the bank entry's mover as
`mover_id`, the other seat as `opponent_id` — overriding the seat
assigner for those games only (`ArenaSpec.scenario_seats` carries it
to the collectors). `wopr.train --scenario-vs-anchor` builds the pair
as (the learner, the single fixed `--anchor`): every scenario game is
the learner in the at-risk seat against the punisher — the
punisher-in-the-scenario-games construction (docs/JOSHUA.md, kick4).
Note the mix shift: the scenario fraction all goes to the anchor
opponent, on top of the remainder's anchor slot.

### The pool (`wopr/pool.py`)

A directory of snapshots taken every `--snapshot-every` updates, with
`stats.json` tracking the learner's record against each. Sampling is
prioritised fictitious self-play: weight `(1 − learner win rate)^hardness
+ floor`, so opponents the learner still loses to come up more often,
unplayed ones count as even, and none ever drops to zero. `--pool-window`
restricts sampling to the newest N. PFSP is AlphaStar's league
mechanism — fictitious self-play (play the whole history of past
policies, not just the latest, so no one strategy can be forgotten)
with the history weighted by how hard each opponent still is — from
Vinyals et al., *Grandmaster level in StarCraft II using multi-agent
reinforcement learning*, Nature 575 (2019),
<https://doi.org/10.1038/s41586-019-1724-z>; their `f_hard(p) = (1 − p)^k`
is the weight above. What WOPR keeps is the main-agent pool and the
weighting; a league exploiter is built from the same parts by
*seeding* the pool.

`--pool-seed name=ckpt.pt ...` copies frozen checkpoints into a run's
pool before training (`train.seed_pool`; idempotent on resume — a name
already present is skipped, and loading rather than file-copying
validates the layout version up front). Seeded snapshots are ordinary
pool entries: PFSP-weighted by the learner's record, sampled by the
same seat assigner, `stats.json` beside them. Two shapes it enables
(the exploiter arm, JOSHUA.md): an *exploiter* run — fresh weights,
the pool seeded with a champion line and `--snapshot-every 0`, so the
opponents are exactly the line and PFSP's hardness walks it as a
curriculum — and a *counter-run* — `--init` from the champion with the
exploiter seeded beside its own snapshots, its share self-adjusting
(sampled most while it still beats the learner). Seeded entries are
the pool's oldest, so a `--pool-window` can age them out.

### The ladder (`wopr/ladder.py`, `eval.py`)

"Win rate against the scripted bot" stops measuring anything once it is
100%. `eval.py` plays every pair of policies on shared seeds, half the
games with each seat assignment, reports **per-seat** win rates (the game
is asymmetric; a pooled number hides a policy that only learned one
side), and fits Elo with `random` anchored at 0 when present.

Every pair is an independent job (`eval.PairJob`): it builds its own two
policies, seeded from the eval seed and the policy names, so its result
does not depend on which other pairs ran or in what order, and the jobs
fan out to a process pool (`--workers`, default a quarter of the CPUs;
`baseline.py` sends all its seeds' pairs at once). Argmax results are
unaffected by this; a `random` opponent's or a sampled net's stream is
now fixed per pair rather than carried across pairs.

### A/B runs and the ledger (`wopr/ab.py`, `baselines/EXPERIMENTS.md`)

The loop asks "is the challenger better than the champion"; `ab.py`
asks "did *this change* alter what gets learned": an engine fix, a
feature, a hyperparameter. `python -m wopr.ab --run <name> --control
v11 [--champion v16] [--note ...] [-- train flags]` trains `<name>`
from scratch with a recipe (default `v11`, 8,000 games) and plays it,
argmax on every eval seed, against the **control** — the frozen version
trained with the same recipe and budget, so the only difference is the
code — the champion, Greedy, and **itself** (the USSR edge, the seat
number the gate cannot see). The result goes to `runs/<name>/ab.json`
and as one row to `baselines/EXPERIMENTS.md`, the committed ledger of
every experiment, frozen or not, with the rules version it was measured
on; JOSHUA.md reads the rows. `--existing` compares a run that is already
trained instead of training one. A run that
beats the control is a candidate for the loop; one that is level with
it says the change was neutral for learning; the ledger keeps both.

### The loop (`wopr/loop.py`)

`loop.py` is the outer loop the ladder and the baselines were built for:
train, evaluate, gate, promote, repeat. One generation continues the run
for `--generation-games` games (the same `train.py` path, optimizer and
pool carried over), evaluates the latest checkpoint — the *challenger* —
against the *champion* (the newest frozen `vN`, or `--champion`) on every
`--eval-seed` and against Greedy, and applies the gate: a challenger
whose win rate against the champion clears `--gate` **on every seed**
(the mean alone lets one lucky deck carry a generation) is frozen as the
next `vN` with the full protocol, gets its README entry, and becomes the
champion. One that does not is trained further — the run never rolls
back, the PFSP pool is what guards against regression — but `--patience`
generations below 0.5 against the champion stop the loop: that is a
regression to look at, not to train through — and so does the plateau
of the decision points, two misses among the last three generations
(`--plateau-misses`), so a long `--generations` runs unattended until
the rule fires. Arguments after `--` go to
`train.py` — and a resumed segment takes its PPO hyperparameters from
those flags, not from the zip (`n_steps` excepted: it sizes the buffer)
— which is how a hyperparameter experiment runs through the loop;
`--no-promote` gates and logs without freezing, so two arms can be
measured against the same champion. `runs/<run>/loop.csv` records every
generation.

## Metrics (`wopr/callback.py`, `runs/<run>/metrics.csv`)

Per update: games, win rates (overall, per seat, vs pool, vs anchor),
the current anchor, rollout and update seconds (and, with collectors,
the seconds spent waiting on them), draw rate, episode length, mean final turn, the learner's mean final VP (`vp_mean`, non-self-play games), policy health —
`entropy`, `k_valid` (mean legal options), `entropy_ratio = H / ln K`
(≈1: not choosing yet; ≪0.3 with many options: collapsed), `k_eff = e^H`
— and SB3's `approx_kl`, `clip_fraction`, `explained_variance`, losses.
Warning signs: KL well above `--target-kl` or clip fraction above ~0.3
(step too big), explained variance near 0 after the first few hundred
games (value head lost), entropy ratio falling while win rate does not
rise (collapse).

With `--eval-every N` the row whose game count crossed a multiple of N
also carries the evaluation of the checkpoint it saved: `eval_seed`,
`eval_games`, `eval_win_rate` and the per-seat `eval_win_rate_us` /
`eval_win_rate_ussr` against `--eval-opponent` (Greedy), and `eval_s`,
the seconds the main process waited for it beyond the PPO update. The
tracker starts it on the backend at the end of the rollout and collects
it at the start of the next; a resumed run reads its earlier ticks back
from the file (`callback.read_evals`), which is what lets the
bootstrap's rolling mean continue across a restart.

## Run layout

```
runs/<run>/
  config.json      arguments + games_done (resume reads it), the commit,
                   the layout version (a run from another version is refused),
                   the recipe and the --init checkpoint if any
  ppo.zip          SB3 model (optimizer state included), written with every
                   pool snapshot and at exit: a killed run resumes from its last snapshot
  joshua.pt        latest plain checkpoint: what `--us joshua` loads
  pool/            snapshots + stats.json
  metrics.csv
  bootstrap.csv    wopr.bootstrap only: every evaluation tick, the rolling
  bootstrap.json   means, the confirmation if asked, the decision; the outcome
```

`runs/` is gitignored. Re-running `train.py --run X --games N` with a
larger `N` resumes; a smaller or equal `N` is a no-op.

Two flags start a run from something other than nothing. `--recipe v11`
applies a frozen version's learning settings (`train.RECIPES`: hidden
size, epochs, the mix, the snapshot cadence — not machine settings) to
every flag not given explicitly, and `config.json` records the name, so
"a clean run" is one token. `--init baselines/vN/joshua.pt` builds a
new run with that checkpoint's network (its own size) and weights and
a fresh optimizer and pool: a frozen version keeps only `joshua.pt`,
and this is how a line continues from one after an experiment has
taken its run directory past it.

## Curriculum and what to try

- **Ops-only first.** `--no-events` runs `Engine.new_game(events=False)`:
  influence, coups, realignments, DEFCON, scoring, space race, no card
  events. A pool trained there carries into the full game.
- **Throughput.** One update is 64 games × 128 learner decisions. With
  `--workers 8` the rollout takes ~1.7 s on the self-play + pool recipe
  (~5.1k learner decisions/s; ~3.7 s and 2.3k in one process), of which
  ~0.9 s is waiting on the collectors (`wait_s` in `metrics.csv`) and
  most of the rest the learner's own forward pass, which is linear in
  rows and does not parallelise away. The PPO update — `--n-epochs 2`
  × 8 minibatches of 1,024, forward and backward — takes ~2.0 s at 16
  threads under the default `--precision bf16` (4 epochs: ~3.8 s; fp32:
  ~6.6 s) and is about even with the rollout again: it is the network's FLOPs, mostly the graph
  layers' linears over 85 nodes × 128 hidden (the legal-rows option head
  and a plain `matmul` for the adjacency aggregation took the fp32
  figure from 10.7 s with identical outputs). In one process a learner
  step is ~40% policy forward at 8+ threads, the rest engine `step` and
  feature encoding; against a net opponent the opponent is asked ~8
  times per learner step at a mean batch of 8 (the tail rounds, where
  few slots still wait on it). See the August 2026 entry in
  [archive/JOSHUA-r1.md](archive/JOSHUA-r1.md).
- **Epochs.** `--n-epochs` defaults to 2. Measured through the loop
  from v8, one generation each way against the same champion: 2 epochs
  gated 0.635 vs 4 epochs' 0.573, beat Greedy 0.995 vs 0.960, tied head
  to head (0.455), with healthy KL and explained variance in both — at
  28% less wall time per generation. That was a continuation: a run
  from scratch at 2 epochs is badly undertrained at 8,000 games (the
  hidden-128 control in the capacity experiment lost to v5 0.06), so a
  fresh run wants `--n-epochs 4`.
- **Device and precision.** `--device auto` picks CUDA when available.
  On CPU, `--torch-threads 16` is the measured sweet spot for the update
  phase (8 threads is 1.5× slower, 32 no faster); torch's default of all
  cores is fine. `--precision bf16` (the default) runs the network's
  matmuls in bfloat16 under autocast for every forward, rollout and
  update alike, so PPO's ratio compares log-probs from the same
  arithmetic; weights, the loss and the stored values stay float32, and
  a resumed run takes the flag, not the zip. It halves the update and
  its first 600 games track an fp32 run's curve (explained variance,
  entropy, KL, clip fraction) within noise; `--precision fp32` is the
  control. A GPU pays off for the update phase already at the default
  model size, and for rollouts once several collector processes
  centralise inference into large batches.
- **Reward.** Terminal only, by design: dense VP shaping of a two-player
  zero-sum game tends to teach the shaping. `--margin m` keeps it
  terminal but lets the final score in: each row's reward becomes
  `(1 - m) * outcome + m * clip(final VP for the mover / 20, -1, 1)`
  (`EpisodeRecord.reward`), so a loss held to −3 on the track is worth
  more than one that reached −20, a win on VP is still +1, and the two
  seats still sum to 0. The default is 0, the outcome alone; `config.json`
  records the weight, and `metrics.csv` carries `vp_mean`, the learner's
  mean final VP in its games against the pool and the anchor.

## Playdek's AI as an opponent (`wopr/playdek/`)

Joshua's opponents so far all come from this repository. The Steam
edition of Twilight Struggle (Playdek) ships a real AI, and it turns out
to be reachable headless: the Unity app is only a front end, and the
rules engine, the card database (Lua, statically linked) and the AI all
live in one native library, `TwilightStruggle_Data/Plugins/x86_64/TwilightLib.dll`,
behind a flat C API of 117 exports. `wopr.playdek` loads that DLL with
`ctypes` from the user's own install — nothing of the game is copied
into this repository — and drives it one decision at a time:

```
pd = Playdek()                                    # loads the DLL, Initialize() once per process
game = pd.new_game(local_side=Side.USSR, ai_difficulty=AIDifficulty.HARD, seed=1)
while (prompt := game.pump()) is not None:        # runs the DLL until we must decide
    game.choose(prompt.options[0].index)          # answer by option index
game.result                                       # GameResult(winner_id, win_type, score)
```

`python -m wopr.playdek.smoke --games 2 --policy random` plays scripted
games and tallies every prompt and selection hint it saw. The install is
found by `ffi.find_install()` (`$STRUGGLER_PLAYDEK_DIR`, else the usual
Steam library roots); `tests/test_playdek.py` skips its live checks
without it. Windows only, like the DLL.

### What the DLL's API looks like

Nothing here is documented by Playdek. The signatures and struct layouts
were recovered from the app's IL2CPP metadata (the C# `[DllImport]`
declarations and the structs they marshal) and then settled against the
live DLL — `ffi.py` is the binding, `game.py` the loop. The facts that
cost something to learn, so they are not learned twice:

- **`Initialize(data_path, processorCount)`** loads the Lua database:
  `data_path` is the flat `StreamingAssets/Lua` directory (the DLL strips
  the `twilight/database/` prefix its own Lua files use). `processorCount`
  sizes the AI's thread pool.
- **`StartGame(GameParameters*, 2, AppPlayerData[2], seed)`.** The
  native `AppPlayerData` is **52 bytes**: the C# struct's `ushort[]`
  fields marshal *by value* (two entries each) and the name *inline*
  (`char[32]`) — declared as pointers, `StartGame` crashes. Seats are
  found by `id`; `playerType` **0 is the local seat** whose decisions go
  to the options listener (1 is the second human of a hotseat game, 2 the
  AI), and an AI seat takes `aiDifficultyLevel` **0 (easy) or 2 (hard)**
  — what the app's "Create Game" screen sets; 1 is not a level and
  crashes. `chooseSidesMethod` 0/1 seats the first player as USSR/US.
- **`SetGameOptionsListener(cb)`**: `cb(playerID, prompt, n, GameOption*)`
  is invoked *on the calling thread, inside `UpdateGame`*, when the local
  seat must choose. `GameOption` is a **64-byte** record — `optionIndex`,
  `selectionID` (the card or country id of the Lua database), a
  `selectionHint` (`ffi.SelectionHint`: what kind of thing is being
  selected), `isHidden`, and the label inline. The list stays valid until
  answered with `SelectGameOption(index)` on a later call; answering from
  inside the callback deadlocks.
- **`UpdateGame(buffer, size)`** fills the buffer with `{int32 type;
  int32 payload[]}` records and returns their count; payload sizes are
  the `GameEvent.*` structs of the metadata (`ffi._PAYLOADS`). A whole
  action is confirmed with **`CommitTemporaryMoveBuffer()`** when
  `COMMIT_PLAYER_DECISION` arrives — the app's "Commit" button, without
  which the game waits forever. The AI thinks on its own threads: empty
  updates while `GetGamePlayerAIState(id).isAIThinking` is set just mean
  "not yet"; `pump` sleeps a millisecond between them.
- **State getters** (`GetGamePlayerHandState`, `GetGamePlayerAIState`,
  `GetGameCurrentScore`, `GetPendingDefconLevel`, ...) take the seat's
  `id`, fill a caller buffer and return the bytes written.
- **Pacing.** `UpdateGame` alone advances in real time — it inserts the
  app's animation pauses, so a scripted seat gets about two prompts a
  second. **`ForceUpdateStateMachineInput`** on an empty update skips
  them (`PlaydekGame.force_updates`); it is on whenever no AI seat is
  thinking.
- **Hotseat** (`new_game(ai_difficulty=None)`): both seats ours, the DLL a
  plain rules engine. The app seats them as `(HOTSEAT, LOCAL)` — the other
  way round leaves the second seat unprompted — and re-asks the very first
  prompt once. The moves of a pending action are emitted twice, as the
  preview and again at commit (`COUNTRY_INFLUENCE` is the state, the
  `OUTPUT_ANIMATION_*` records are the preview).
- **The AI's time is a search budget, not pacing.** Its setup placements
  are instant; every other decision takes **exactly 15.0 s** of wall
  clock, busy on one core (process CPU time equals wall time). It is the
  same for easy and hard, for `processorCount` 1 or 32, with the
  animation delays zeroed, and even when the process is pinned to one
  core with busy threads stealing most of it — so it is a deadline the
  search runs up to, not a fixed amount of work: a faster CPU means a
  stronger AI, not a faster game. The constant is not a literal in the
  binary (not as seconds, ms, µs, 100 ns ticks or a float), so it is
  computed, presumably through the CRT clock. A whole game against a
  scripted opponent is therefore ~2½ minutes (hard) to 4½ (easy, it
  lasted longer), one core each; with 32 cores that is ~10 games a
  minute in parallel processes — the DLL is a process-wide singleton
  with one current game. The AI is not deterministic for a given `seed`
  and a fixed opponent (two runs of the same scripted game differed from
  the AI's first decision on), so an eval against it is a sample, not a
  replay.
- The pacing mechanism, for the record: the state machine looks up a
  minimum delay per option kind (a table in the DLL, 0.6 s by default)
  and compares it with the DLL's own seconds clock; patching that table
  to zero gives the same speed as forced updates (~10k prompts/s), so
  `ForceUpdateStateMachineInput` is the right switch and no patching is
  needed. Playdek's own tests presumably go through `StartTutorial`,
  which takes a scripted deck, dice and AI steps and no search at all.

### As a rules engine: measured, and why not

Random against random, one process, forced updates: the Playdek engine
answers **~13.8k prompts/s** (50 games, 0% idle; ~4.6 `UpdateGame`
calls per prompt); the struggler engine plays the same matchup at
**~17k decisions/s** (`Engine` + `RandomPlayer` through `play_game`).
Same order, struggler slightly ahead, and a prompt and a decision are
about the same grain (one influence placement, one card choice). So the
DLL is not a faster engine, and it could not be the arena's engine
anyway: it is one game per process with no batching across games, there
is no `observe()` — a bot's `Observation` would have to be rebuilt from
the event stream and the state getters, the hand included — no
`serialize()`/`deserialize()` for search, no guarantee about the seed,
and it is Windows-only and not redistributable. The profile in
[archive/JOSHUA-r1.md](archive/JOSHUA-r1.md) says the engine is a third of a learner step at
most; the rest is encoding and inference, which an engine swap does not
touch. What the DLL is good for is the two things struggler cannot
provide: an independent, commercial-grade opponent, and a second
implementation of the rules to diff against.

### What this is for, and the pieces

The bridge to Joshua is the engine's physical mode (`docs/BOTS.md`): the
struggler engine mirrors the Playdek game as referee, Joshua sees only its
`Observation`, and a `PlaydekOperator` replaces the human operator —
translating the DLL's events (the AI's card plays, placements, rolls) into
the operator's answers and Joshua's `Action` into `SelectGameOption`.
Every action is a chance to compare the two engines' states; a desync is
either a bridge bug or a rules divergence worth a line in
`LIMITATIONS.md`. The same translation, driven by a random policy in
hotseat mode, is a lockstep differ of the two rules engines — thousands
of random games an hour exercise every card event — and, on identical
games, a matched per-decision benchmark of the two.

The pure parts exist:

- `ids.py` — the vocabularies. A card option's `selectionID` is
  `100 + GMT number` (the same `number` as `cards.json`); a country's is
  Playdek's `country_index` (1 USSR, 2 USA, 3 Canada … 87 Chinese Civil
  War), also the `id` of `COUNTRY_INFLUENCE` events. Numbers above 110
  are Playdek-only (promos 121–128, Turn Zero 129–146, the AI's Ops
  proxies 201–204).
- `translate.py` — what an option *means* (`Meaning.CARD / USE / COUNTRY
  / CHOICE`, plus the UI-only `CANCEL`, `SWITCH_CARD` and `BLANK`), the struggler
  actions a "use" option stands for (one Playdek "Place Influence" is
  `PLAY_MODE` ops + `OPS_TYPE` influence, with `EVENT_OPS_ORDER` in
  between on an opponent's card — "Resolve Event First" is its own
  option), lookups the other way (`find_card/find_country/find_use`),
  and `rolls_from_event`, which turns the DLL's `COUP_ROLL`, `WAR_ROLL`,
  `REALIGNMENT`, `SPACE_RACE_ROLL`, `TRAP_ROLL` and `EFFECT_ROLL` records
  into physical mode's CHANCE answers. A country option's struggler
  *kind* (`PLACE_INFLUENCE`, `EVENT_INFLUENCE`, `COUP_TARGET`, …) is
  whatever the engine is asking; only the bridge knows that.

### The lockstep differ (`lockstep.py`)

What the differ and the match operator below share is `bridge.py`: the
absorption of the DLL's records into facts (absolute influence, DEFCON,
VP, card locations, a FIFO of dice, what was dealt this turn), the
per-side queues of `Move`s (a choice the other program made, as a
`translate.OptionMeaning`), the answers to the engine's CHANCE decisions
and to a side's queued moves, the state comparison and the divergence
report. The differ fills the queues from the prompts a policy answers;
the operator fills the AI's from the records its play leaves behind.

`python -m wopr.playdek.lockstep --games 4 --seed 1 [--trace]` plays
hotseat games with a random policy and replays them on the struggler
engine in physical mode, on demand: every engine decision is answered
from a per-side queue of translated Playdek moves, its CHANCE decisions
from the DLL's roll records, its `DEAL_CARD`s from the DLL's hand
contents (absolute `CARD_LOCATION` state — the DLL deals, undoes and
re-deals at commit, so a queue of deal events is wrong). When the engine
asks something the queues cannot answer, the DLL is advanced first.
Option sets are compared whenever the engine asks a card or country
choice; state (influence, DEFCON, VP, mil ops, hand sizes) whenever both
sides are between actions — the two engines apply the same action at
different moments, so comparing mid-action only measures that.
Everything else it learned about the DLL's protocol is in its comments:
the first hotseat prompt is re-asked once, turn-2+ headline prompts
arrive under the local seat's id whoever is picking (the cards say whose
hand), and the `*_player_index` of roll records is the seat's id. Two
that cost a wrong diagnosis each:

- **Records are emitted twice, and the second time is late.** Each choice
  emits its records as it is made (the preview); when the action is
  committed the DLL re-emits the whole chunk since the previous commit,
  verbatim and in order (`LOG_UPDATED` aside, and a hand reveal's
  replay gains a `PAUSE_FOR_REVEALED_CARDS`), either right after the
  commit or, when a prompt came first, after that prompt's answer — a
  turn's end is replayed after the headline pick, whose own records are
  then the start of the next chunk. Absolute state (`COUNTRY_INFLUENCE`,
  `CARD_LOCATION`) would be idempotent under this only if it were
  current — a replayed value is the value of its time — and the dice
  would be rolled twice, so `Bridge.mark_replays` keeps every record
  since the last replay in a FIFO, tagged with the pump it arrived in,
  and takes a run of a batch that copies the FIFO from its head up to a
  batch boundary for the replay; a chunk the DLL never replays (a
  non-phasing seat's event-granted Op that ends the phasing seat's
  action) is skipped over when a later chunk's copy starts at a
  boundary further in, never at the start of a batch. Two earlier
  versions were wrong: deduplicating within one pump re-queued the
  replay of a multi-roll action, and matching one record against the
  FIFO's head (of dice and influence only) took a second realignment of
  the same country with the same dice — or the second point placed in
  the same country — for the replay of the first and fed the engine
  stale dice for the rest of the game. Over 280 seeds the FIFO drains
  at every action; a FIFO past 400 records is reported as a harness
  divergence.
- **A yes/no event choice lists a third, blank entry** (`selectionHint`
  `0xA0FF`, `selectionID` the card, `isHidden` *false*, empty label)
  beside "Participate"/"Boycott". Selecting it makes the DLL skip the
  event altogether — no roll, no DEFCON change — which the app's UI never
  offers; `translate` calls it `Meaning.BLANK` and no policy picks it.

Findings from the first four random games, and what became of each:

- **Fixed upstream** (branches off `upstream/main`, merged here): the
  engine's Military Ops counted past 5 (the track stops there); the
  game did not end when a scoring card was held past the end of the
  turn (the engine forbade holding instead — now it offers the whole
  hand and `_end_of_turn` decides); realignment could not stop after
  the first attempt (now `{"country": "stop"}`); the Chinese Civil War
  space was a country of the standard game (now a variant-only space
  behind `Engine.new_game(variants=...)`, the layout unchanged);
  Containment/Brezhnev Doctrine took a 4-Ops card to 5 (the cards say
  "to a maximum of 4" — `fix/ops-modifier-cap`; seed 2's "extra
  `PLACE_INFLUENCE`" was NATO under Containment); Five Year Plan fired a
  discarded *USSR* event and discarded a US one, the reverse of the card
  (`fix/five-year-plan-us-event`; seed 3's DEFCON/VP drift after the
  discard of Duck and Cover); How I Learned to Stop Worrying's +5 wrote
  past the Military Ops cap (now on `fix/military-ops-cap` too); an
  event's "conduct Operations as if they played an N Ops card" ignored
  Containment/Brezhnev/Red Scare (7.4.3, and 7.4.2's example 3 is
  literally CIA Created under Containment — `fix/event-ops-modifiers`);
  We Will Bury You paid the USSR at the end of the turn where the card
  and FAQ pay "the moment the US does not play UN Intervention in their
  next Action Round", which may be next turn, and never without a next
  round (`fix/we-will-bury-you-timing`; Joshua's layout keeps the flag's
  turn slot via `features.RELOCATED`, so `LAYOUT_VERSION` stays 1; a
  legality flag the bot already sees through its options, Tear Down
  This Wall's, is listed in `features.UNENCODED_GAME_EFFECTS` and not
  encoded, for the same reason).
- **Documented, DLL-stricter**: De-Stalinization will not relocate
  influence back into a country it was just removed from; the card
  text has no such clause, so the engine allows it. The harness counts
  it under `known`.
- **Harness false alarms, removed**: "Done Removing" / "Do Not Relocate"
  / "Do Not Discard" — the engine had those choices all along.
- **Harness bugs, fixed** (the three "open" items of the first pass,
  `--games 4 --seed 1`, all turned out to be the harness): the
  realignment whose outcome "differed" (seed 1) had been given stale
  dice — the replay of an earlier multi-roll action, re-queued because
  the dedup only looked within one pump (the `REALIGNMENT` fields *are*
  `USSR_roll_result`/`US_roll_result`, checked against ten realignments
  with their before/after influence); the "diverged hand" (seed 3) was
  `RANDOM_DISCARD` picking the *first* card whose latest move was
  hand→discard, which after a few action rounds is some card played
  long ago — it now takes the latest such move among the cards the
  engine offers, and `compare_state` also diffs the visible hand's
  contents, card by card, at every action-round sync, so a real hand
  divergence would show at once; and the blank yes/no entry above,
  which a random policy picked one game in two. Eight more seeds
  (`--games 8 --seed 5`) added: Olympic Games' `EFFECT_ROLL` is
  (USSR die, US die), not (sponsor, defender) — the engine's
  `CONTEST_ROLL` context says who sponsors, the bridge maps; Summit's
  "Improve / Degrade / Pass" (hints `0xA073/71/72`) carry an explicit
  `raise/lower/none` because no label word matches; and UN Intervention,
  which Playdek plays as its own card ("Play Event", then "Select
  Opponent Event Card to Play", hint `0xA012`) where the engine plays
  the opponent's card with mode `un_intervention` — the bridge looks
  two moves ahead. State is compared only when no translated move is
  still queued (the lookahead made the engine wait at a card prompt the
  DLL had already answered).
- **Harness bugs, third batch** (seeds 13–20 and the rest of 5–12):
  `DEAL_CARD` answers from the cards the DLL dealt *this turn* (a card
  dealt, headlined and resolved in one pump is in the discard pile
  before the engine deals; the current hand alone stalled the engine a
  whole turn); a lone "Do Not Discard" the engine never asks (Blockade
  with nothing to discard) is dropped; once the DLL's game is over the
  engine's single-option decisions are taken; Grain Sales' drawn card
  comes from the "Play <card>?" prompt's own `selectionID` and
  take/return from which option was picked (a returned card never
  leaves the hand, so the discard heuristic fed the engine some other
  card); the DEFCON hints decode as `0xA070 + n` = "set DEFCON to n"
  (How I Learned lists 1–5, Summit the three reachable levels), mapped
  to the engine's level or raise/lower/none.
- **DLL behaviour, counted under `known`**: a UN-Intervened card may be
  spent on the Space Race (the engine's `un_intervention` mode is Ops
  only; the same play is available by spacing the card itself, so the
  differ never picks it); Defectors played "event first" by the USSR,
  where the engine has no event to order (both give the US 1 VP).
- **Reported, not resolved**: either/or choices matched by label words;
  the word match has been right every time so far.
- All twenty random games (seeds 1–20: `--games 4 --seed 1`, `--games 8
  --seed 5`, `--games 8 --seed 13`) run to the end with nothing but the
  known entries above and the word-matched choices; where both engines
  finish, the winner agrees.
- **Found by the operator, fixed upstream** (what the differ cannot see:
  a play the DLL never offers, or an event no random game fired):
  the engine offered an opponent's card "for its event" alone, a play the
  rules do not have (`fix/opponent-card-event-only`; a random policy on
  the engine picked it one game in two); Independent Reds asked the US
  to choose among all five countries whether or not there was USSR
  Influence to match, and asked with none (`fix/independent-reds-targets`);
  the US/Japan Mutual Defense Pact wiped the USSR's Influence in Japan,
  where "sufficient Influence for Control" sits on top of it
  (`fix/us-japan-pact-keeps-ussr-influence`; `gain_control` now removes
  the opponent only where the card says so, as Fidel and Romanian
  Abdication do); the wars that pick their target subtracted 1 for the
  target's own control, where the cards say "adjacent"
  (`fix/war-target-own-control`); Blockade's, Latin American Debt
  Crisis's and the traps' discards compared printed Ops where the turn's
  modifiers count -- under Containment the AI paid Blockade with Korean
  War and the engine cleared West Germany
  (`fix/discard-thresholds-use-modified-ops`). These two were found in
  games against the AI itself, where a drift shows as a state
  divergence at the bot's next card prompt; `eval --games 1 --trace`
  plays one traced game, and the `INF` lines show what the placement
  inference saw.
- **Fifth pass, 280 seeds** (`--games 280 --seed 21`, `--physical`
  alternating; one seed is reproduced with `--physical us|ussr`, the
  side the 280-game run hid for it). Ten more engine fixes, each its own
  branch off upstream: a card whose event cannot happen is discarded,
  not removed (the engine re-dealt a NATO it had removed);
  Ask Not may discard scoring cards; Missile Envy's exchanged card is
  played as its event when it is the taker's or neutral (the engine
  offered Ops-or-Event); Defectors revealed by Five Year Plan cancels
  the USSR headline in the headline phase and scores the US 1 VP in an
  action round; We Will Bury You is paid at a trapped US round; DEFCON 1
  loses the phasing player, whoever moved the marker (four random games
  had the other winner); UN Intervention goes with any opponent-event
  card (Defectors, an ineligible NATO); the bonus Realignment attempt is
  offered inside its region only; the China Card's and Vietnam Revolts'
  bonuses stack to 6 Ops; Cuban Missile Crisis may be cancelled by
  either side and is offered again to the banned side at its coup
  (`'Remove Cuban Missile Crisis?'`, hint `0xA038`, before the target).
  Harness bugs of the same pass: a single-option engine decision (an
  Independent Reds with one candidate, a scoring card's `PLAY_MODE`) was
  answered with the seat's *next* move, now `_compatible` checks the
  payload, and a plain-influence prompt's country never answers an
  either/or; Missile Envy's "Select Card to Give" is asked even for a
  single candidate, where the engine asks only among ties (dropped once
  the engine has made the exchange; the physical giver's pick drops it
  too); the Cuban Missile Crisis defusing is an entry of the action-round
  prompt (`0xA0AA`, "Remove 2 Influence from West Germany", selectionID
  250 + the country index) where the engine asks a choice at the round's
  start; the trap discard prompt's blank "TRAP" entry (`0xA09F`) and its
  "Pass" (`0xA09C`, "You May Play a Scoring Card"); the DLL keeps a card
  in the hand until its event is done asking (Blockade's "Do Not
  Discard"), so the deal skips the card being played; the DLL's deck runs
  out sooner than the engine's bookkeeping, so its reshuffle (discards
  back to the deck) makes the physical side's next headline take
  `RESHUFFLE_NOW` first; Space Race box 6's held-card discard is declined
  unless the DLL's move is a discard; the state is not compared at the
  turn's end (the engine has recovered DEFCON, the DLL reports it after
  its next prompt); the listed either/or labels (`translate.CHOICE_LABELS`:
  Warsaw Pact, Olympic Games, South African Unrest — whose "Add
  Influence Adjacent to South Africa" shared two words with the wrong
  choice) are matched exactly, anything else by words and reported.
- **DLL behaviour, counted under `known`** (this pass): a trapped seat
  with no 2+-Ops card may keep its scoring card ("You May Play a Scoring
  Card" lists none; the engine plays it — fatal, the hands diverge);
  Junta's free Coup/Realignment is confined to the country the Influence
  went to (the engine offers the whole region; the bot is narrowed, the
  AI's choice simulated); Missile Envy's exchanged card, and the forced
  play of Missile Envy itself, may go to the Space Race (never picked);
  event-granted Ops ("Select Use For Operations") likewise; De-Stalinization
  as before. A 4-Ops card under Containment or Brezhnev Doctrine is 5 in
  the DLL where the cards say "to a maximum of 4" — seen once as a coup
  result, not yet handled.
- **DLL behaviour, counted under `known`** (later in the pass): the
  crisis outlives the engine's record of it -- after the USSR played
  Cuban Missile Crisis for Ops (no event fired in either program) the
  DLL still asked the US to pay its way out of a coup; when both hands
  hold a scoring card at the turn's end the DLL's loser is the one the
  engine cannot see (the engine names the one it can); the forced play
  of Missile Envy itself may go to the Space Race. And one more engine
  fix: the two end-of-turn Military Operations penalties are netted
  before the marker moves (the engine declared a victory on the first of
  the two); Kitchen Debates' condition is its precondition (unmet, the
  card is discarded, not removed).
- After the pass: 278 of the 280 seeds show nothing but `known` entries
  (the other two: the trap's kept scoring card, a `known` fatal; and one
  turn-9 drift, a US point in Gulf States the engine does not have after
  a USSR event -- seed 157, `--physical us`, open). The hotseat emulation
  is 31/32 (seed 2 as US, turn 8: Aldrich Ames Remix's choice does not
  reproduce the DLL's state after an East Germany drift at turn 7, open).
  Against the AI itself (`--difficulty hard --policy greedy --seed 300`,
  30 games, in progress at hand-over) the first 12 are 8 clean and 4
  desyncs, every one an influence drift of the AI's placements by turn 3
  (seeds 303 as US, 306/308/300 as USSR): the placement inference still
  misreads a chunk the AI plays in one go. Those are the next thing to
  trace (`eval --games 1 --workers 1 --trace --seed N --side ussr|us`).
- **Sixth pass, against the AI** (the ten desyncs of the 30 Greedy games,
  each traced until it reproduced -- the AI is not deterministic, so
  `runs/playdek/trace/batch.sh OUT PARALLEL seed:side ...` plays traced
  single games in parallel and every desync comes with its trace; the
  hotseat emulation, 32 seeds, now plays clean). Five operator bugs: the
  AI's headline was keyed by card alone, so a card reshuffled and headlined
  again the next turn was never queued (now `(card, turn)`); a discard
  was looked up as the card's *latest* move, which in one pump can already
  be its re-deal after the DLL's reshuffle -- the AI's Blockade payment
  (De-Stalinization, discarded, reshuffled and dealt to the USSR) read as
  "Do Not Discard" and the engine emptied West Germany (now
  `Bridge._exits`, a log of every hand-to-pile move, consumed when the
  engine's discard accounts for it and purged up to the DLL's reshuffle
  once the engine has reshuffled too); the AI's Grain Sales draw arrives
  as the card pushed into the resolve slot with the "fired" hint, not as
  a reveal (the `random_discard` now takes either); De-Stalinization's
  sources were taken from the countries where the DLL had *more* once the
  removals were matched (a fallback meant for Independent Reds' match),
  so a De-Stalinization that moved two points went on removing from where
  it placed; and the inference window (`synced_seq`, the record count at
  the last agreement, advanced only at the bot's card prompts) spanned the
  bot's own action and the AI's whole chunk, so the bot's Liberation
  Theology point that its coup then removed, or COMECON's placements that
  the AI's De-Stalinization then moved, were read again as the AI's
  placements -- `_resync` now moves the window up to the latest card-play
  boundary (`play_log`) at which the DLL's board, reconstructed from the
  influence history, is the engine's board now. A sixth, from the traced
  batch: when several simulated choices reproduce the DLL's board (Junta's
  free Realignment that removed nothing, and declining it), the one that
  consumed the DLL's records is taken -- left queued, the realignment's
  dice passed for ABM Treaty's granted Ops two turns later. A seventh:
  the card Grain Sales hands the US is played at once and the DLL
  reports no use for it, only the coup or the influence that follows, so
  its `PLAY_MODE` (and an opponent card's `EVENT_OPS_ORDER`) is simulated
  like an either/or. And one more `known`, DLL-different: We Will Bury
  You's 3 VP are paid by the engine the moment the US plays a card other
  than UN Intervention, by the DLL once that play is done -- when the VP
  end the game there, the engine is over while the DLL still asks the
  rest of the bot's action; `_complete_for_dll` finishes it with the
  plainest choices and the two results are compared (a different winner
  is fatal). And another: Flower Power pays the USSR 2 VP for a war card
  the US *plays* (the engine, and the card's "for Ops or for Event"),
  where the DLL pays only when the war's event happens -- an
  Arab-Israeli War under Camp David Accords is 2 VP in the engine and
  nothing in the DLL, and the VP differ for the rest of the game:
  `_flower_power_check` ends it as `known` the moment such a card is
  played. (Resolved at rules version 5: the engine adopts the DLL's
  reading — a prevented event's war card does not trigger Flower Power
  — and the check is gone; the ninth pass.) And the largest: the DLL conducts Grain Sales' 2 Ops *and*
  plays the taken card, against its own card text ("if returned, use
  this card to conduct Operations") -- the AI's favourite headline, so
  one USSR-seat game in ten: when neither take nor return reproduces the
  state but take with two more Ops does, the game is void. And one
  found by Joshua's first games against the easy AI (seed 315): a seat
  in Quagmire / Bear Trap with no 2+-Ops card keeps its scoring card
  ("You May Play a Scoring Card" → Pass, the only answer the DLL
  honours) and the engine ends the turn with the held card lost, as
  the rules do, where the DLL carries the card into the next turn
  with no penalty — `_trapped_held_scoring_card`, `known` and void. A
  game ended by such a `rules` difference is `MatchResult.void` (the
  reason), not a desync: the eval reports `void` by reason beside
  `desyncs`, and neither counts. One simulation gap from the same
  games (seed 332 as US, reproducible): a simulated option is judged
  at the bot's next decision, but when that decision is a choice
  inside the other seat's event that the DLL resolved without asking
  (Independent Reds with one country worth choosing) and played on
  past, the DLL's state is not that point's — `_run_copy` then tries
  the bot's few options (`_try_each`: event choices of at most eight
  options, one level) and judges at the next point the DLL stopped at.
  Open from the same 120 games (`runs/playdek/r3v1-easy-*`): Grain
  Sales' random take resolved to the wrong card three times with
  Joshua as USSR (seeds 319, 328, 333 — a stale reveal picked over the
  card the DLL took; once a scoring card, scored by the engine as the
  US); the AI's trapped seat playing its scoring card in the trap step
  with no prompt to the operator (seed 323, the engine answered
  "none"); and a turn-1 placement the engine offers and the DLL does
  not (seed 357, Egypt for the US, reproducible). The AI's line does
  not repeat on a rerun, so these want traces caught by volume. v8's
  120 games (`runs/playdek/r3v8-easy-*`) added late-war families the
  earlier policies never reached: a realignment roll matched to the
  wrong target (seeds 341, 347: France "illegal in engine"), Ask Not's
  discard bookkeeping (349), Wargames' "Would You Like To End The
  Game?" (303: the engine's decline has no matching option — a
  translate gap), Junta's choice (326), a trapped hand's discard
  option lists (304, 315), a card the engine may play that the DLL
  does not list (335, Yugoslavia), and two slow VP drifts (337, 350). From
  r3-bid2/v3's games: the bot's own Grain Sales take of a drawn scoring
  card crashed the mapping (`PLAY_SCORING_CARD` hint — it had only been
  handled for the other seat; fixed), one crashed game no longer kills
  the eval's pool (the batch reports it and plays on), Grain Sales
  inference misses run to six games a seat, and SALT Negotiations'
  choice, Tear Down This Wall and a forced discard are new
  (`r3bid2v3-easy-ussr` seeds 327, 332, 348). The search batch's traces
  (2026-08-26, `runs/playdek/gs-trace-304.log`) caught the Grain Sales
  remainder in the act — a *headlined* Grain Sales queues the DLL's
  "play <the taken card>" selection where the action-round path leaves
  the queue empty, and the take/return simulation stalled on it at the
  taken card's `play_mode`; the operator now consumes that move (it is
  the very play the engine is asking the mode of). The same trace
  exposed an open **rules** question: the AI's granted coup reached
  DEFCON 1 during the headline phase and the DLL awarded the win to
  the coup's own actor, blaming the player who *played the causing
  event* (the printed 4.5 note's reading), where the engine blamed the
  marker's mover (`_defcon_one_loser`'s `caused_by`). Ruled for the
  DLL's reading and fixed
  (`fix/defcon-one-headline-event-owner`, rules version 4:
  `_headline_current` tracks the resolving headline, and the DEFCON-1
  loser during the headline phase is its owner). **The DLL plays the
  tournament bid natively**: `GameParameters.additionalInfluence` (the
  app's handicap) inserts a US "Place N Influence" step right after the
  regular setup with exactly the engine's 11.1.4 semantics — candidates
  are the US's countries, capped two past control (probed 2026-08-24) —
  so `wopr.playdek.eval --bid N` now plays the bid on both boards
  (`Bridge(us_bid=N)`: the DLL's additionalInfluence and the engine's
  `us_bid`), and a bid-trained policy is evaluated on its own game.
  Hotseat emulation is clean at bid 2 (32/32 and 120/120 wide) and
  unchanged at bid 0 (32/32). And the Grain Sales take is now read off
  the DLL's card moves before any reveal record — the taken card left
  the USSR's hand for the US's (or a pile, played at once) inside the
  inference window, where a stale reveal had named the wrong card
  three times in one batch; the hidden-prompt harness stays 59/59 and
  the emulation clean. Two engine
  fixes: Marine Barracks Bombing removed the whole of two Middle East
  countries where the card removes two points (the differ's seed 157,
  `fix/marine-barracks-two-points`); Willy Brandt still fired after Tear
  Down This Wall, which the card says prevents it
  (`fix/willy-brandt-after-tear-down-this-wall`).
- **After the sixth pass:** the differ is 279/280 (seed 93's trap
  fatal, `known`); the hotseat emulation 32/32, and a wider sweep
  (`--games 120 --seed 40`) 5 desyncs where the hand-over code had 18 --
  the wider sweep found four more operator bugs on the way (a card's play
  taken for a discard; the Ops half's point removed by the event half of
  the same play read as a transient; Missile Envy's exchanged card
  arriving as the fired push; the same card pushed into the resolve slot
  again being a new play, the China Card after Ussuri River Skirmish).
  Greedy against the hard AI, 30 games a seat (`--seed 300`): **US 30/30
  and USSR 30/30 with zero desyncs and zero void** on the final code
  (`runs/playdek/greedy-hard-us`, `greedy-hard-ussr-run6`; the earlier
  runs of the pass, `-run1..5`, are the trail of the fixes). Greedy wins
  none of them. Still open, all in the emulation's wider sweep: an
  Olympic Games after a DEFCON/VP drift (seed 64), a late placement
  inference (84), a held-card discard reshuffled and dealt again (104), a
  realignment read as a placement (121), a US hand drift (136); and the
  hidden-prompt emulation (`runs/playdek/trace/emu_grain.py`, the other
  seat's Grain Sales inferred from records as the AI's is) at ~8%.
- **Seventh pass, the emulation's wider sweeps** (the five open seeds of
  the sixth pass, then two fresh ranges of 120 -- `--seed 200` and
  `--seed 400` -- every desync traced until it reproduced clean). Four
  more engine fixes, each its own `fix/*` branch merged `--no-ff`:
  Iran-Contra Scandal took its -1 off the US's own realignment attempts
  only, where the card ("all US Realignment rolls") and the DLL's
  continuous "opponent realignment rolls are reduced by 1" take it off
  every US die -- a USSR 6 against a US 5 + 1 in Jordan was a wash here
  and a point removed there (seeds 64 and 84,
  `fix/iran-contra-defending-rolls`); the extra action round Space Race
  box 8 or North Sea Oil grants had to be played where the rules say
  "may" -- `PASS_ROUND` (`"pass"`) is now offered beside the cards there,
  and likewise when a hand holds nothing but the China Card (8.1.6)
  (seeds 84 and 453, `fix/extra-action-round-may-be-passed`,
  `fix/china-card-only-hand-may-pass`); Five Year Plan fired any US event
  it drew, eligible or not, so a NATO drawn before Marshall Plan left the
  game as a fired card where the DLL merely discarded it, reshuffled it
  and dealt it again (seed 224, `fix/five-year-plan-ineligible-event`).
  And a fifth, the differ's seed 93 and the wide sweep's one void: a
  trapped seat with no 2+-Ops card had every scoring card in its hand
  played at once, where Bear Trap and Quagmire say it "may only play
  scoring cards" -- the scoring card is offered with "none" to keep it
  (holding it past the turn's end is still the loss), and the hand the
  engine cannot see is first asked whether any of the hidden pool's
  2+-Ops cards is really there (`fix/trapped-seat-may-keep-scoring-card`,
  two commits). The DLL's "You May Play a Scoring Card" -> "Pass" now
  answers those steps in the bridge, the operator and the differ; the
  `known` entry and the fatal it raised are gone. One more DLL
  difference, `known`: The Reformer's ban on USSR coups in Europe is
  kept by the DLL on the card "in play", which it never is once Glasnost
  has already been played (its Lua puts the card in play only while
  Glasnost is not in the removed pile) -- the DLL then offers Europe
  coup targets the engine, and the card ("for the rest of the game"),
  refuse (the differ's seed 157, non-fatal; the AI taking such a coup
  would void the game). Two changes to the `known` bookkeeping: the
  scoring cards the DLL shows in the hand the engine cannot see are
  revealed to the engine as they appear (`_reveal_hidden_scoring_cards`,
  via `_reveal_in_hand`: one hidden slot, off the hidden pool), so the
  engine's own end-of-turn check ends a game for a held scoring card as
  the DLL does and the ending is compared instead of counted under
  "held scoring card in the hand the engine cannot see" (48 of the
  differ's 280 games stopped there, and 8-22 of every 120 emulated); and
  De-Stalinization's excluded sources count once per event, not per
  placement (435 hits over 128 games became 128). With both hands in
  view at the turn's end, one game of the 280 (seed 149) has both
  holding a scoring card: the engine calls that a draw, the DLL gave the
  game to the US -- `known`, the rulebook naming no winner for it.
  Operator and bridge bugs, one commit each: a DEFCON choice (Summit, How
  I Learned to Stop Worrying) was read off the DLL's current level, which
  the turn's end had already restored after a last-round Summit -- the
  read is kept only when a copy reproduces the state (64); a headlined ABM
  Treaty's realignment took the action round's rolls too -- a roll
  recorded after the seat's next queued play (`Move.seq`) is that play's
  (121); whether a card's exit from a hand was its play is settled when
  the exit is logged, since a SALT Negotiations recovery dropped the play
  record and the old exit then passed for a discard, one card too many
  for Ask Not (136); a deal asked once the DLL's game has ended (a
  scoring card held in the hand the engine cannot see) defers to the
  game-over handling instead of failing on the frozen hand (104);
  "Discard a Card from Opponent Hand" (Aldrich Ames Remix) is the
  opponent's prompt of the hand shown (136); Star Wars' copy is read off
  the push from the discard pile, no hand move reports it (136); the other
  seat's unrecorded pass of an extra round, and the bot's own, are the
  pass option, in the differ too (84); Chernobyl's region options (hints
  `0xA081`-`0xA086`) are choices (313); the emulated seat's Grain Sales
  takes a scoring card it plays (270); the preview of a realignment just
  answered is never a replay of an earlier realignment of the same
  country with the same dice (472); a simulated copy steps through the
  bot's forced `event_resume` and, once it has ended as the DLL's game
  has, is judged by the winner (513); a lone "Pass" the DLL asks the bot
  before the engine has reached the choice (Tear Down This Wall's free Op
  with no target) is sent at once and the bot's decision cut down to the
  decline when it comes (503).
- **After the seventh pass:** the differ 280/280, every game now played
  to its end (none stops at a scoring card the engine could not see);
  the hotseat emulation 32/32; the wide sweeps (`--games 120`) at
  `--seed 40`, `200` and `400`: 120/120, 120/120, 120/120; the
  hidden-prompt emulation (`emu_grain.py 1 60`, seeds 1-59) 59/59. The
  Greedy-vs-AI runs of the sixth pass were not repeated.
- **Eighth pass, Greedy against the hard AI again** (30 games a seat,
  `--seed 300`, on the seventh pass's code: 29/30 and 29/30, one desync
  each, no void). One engine fix, `fix/we-will-bury-you-trap-round-end`
  (stacked on `fix/we-will-bury-you-trap-round`): We Will Bury You's 3 VP
  for a US round spent in Quagmire were paid at the round's start,
  before the US was asked its discard; the DLL pays them once the round
  is over, after the discard and the `TRAP_ROLL` (`emu7/g-64*.log`), as
  the ordinary round pays at the play and not before the card is
  picked. Unseen until the VP crossed 20: the USSR at 17 headlined the
  card and played OPEC, the engine's game ended on the 3 VP while the
  DLL still asked "You Must Discard a Card" (seed 322 as the US). The
  effect's value is `WE_WILL_BURY_YOU_TRAPPED` for the trapped round
  (truthy: Joshua's flag is unchanged) and
  `_settle_we_will_bury_you_trapped` pays from `_advance_once`, before
  the next round or the turn's end. One bridge fix: the scoring cards
  the DLL shows in the hand the engine cannot see were revealed whenever
  they appeared, and the DLL is pumped whole actions ahead -- the AI's
  Ask Not discarded its hand and drew the replacements in one pump, and
  a Southeast Asia Scoring it drew was revealed into a slot of the hand
  the engine was still discarding, one discard at a time, so the last
  discard the DLL made (Cultural Revolution) had no slot ("illegal in
  engine", seed 315 as the USSR, turn 5; the hand sizes had agreed at
  the bot's prompt before, 9 and 9). `_reveal_hidden_scoring_cards`
  now waits until the DLL's hand count is the engine's: the engine
  draws its own slots at "stop" and the reveal follows at the next
  answer; a deal at the turn's end is the same case. The "the DLL
  discarded X" mismatch now carries every unaccounted exit, the hand
  sizes and the recent records. Neither game reproduces by volume (six
  traced repeats of seed 315 took other lines by turn 3: the AI is not
  deterministic); the emulation on the new code is 32/32 and 120/120
  at `--seed 40`, 47 held-card endings among them, and Greedy against
  the hard AI on it (`--seed 300`) is 30/30 a seat, zero desyncs, zero
  void (`runs/playdek/greedy-hard-ussr-run8`, `greedy-hard-us-run4`).
- **Ninth pass, from the r4b2v4 easy evals** (16 desyncs over 240
  games, two families sized, both addressed). One engine fix: an
  event-granted free Realignment chain (Tear Down This Wall, Junta)
  lost the card's terms after its first roll —
  `_maybe_push_realignment_target` rebuilt the target list with the
  ordinary DEFCON geography and no region restriction, so at DEFCON 2
  the second European target was never offered (the eval's standing
  weather; 5 of the 16, every one Tear Down This Wall) and, the other
  face, a target outside the card's named countries was. The chain
  context now carries `restrict`/`ignore_defcon` through every roll.
  One classification fix: the trapped seat's scoring card reaching
  game over — the engine lets the trapped seat play it (the card:
  "may only play scoring cards"), the DLL holds it
  (`TRAP_SCORING_CARD`) and ends the game HELD_CARDS at the turn's
  end while the engine plays on; that was a fatal "game over"
  mismatch (2 of the 16) and is now the documented rules difference's
  void. And one diagnostic: a `_simulate` that matches no option now
  reports where each option's line stopped and how it differed, so
  the remaining singleton families come back from ordinary eval
  volume with their traces attached. Verified: hotseat 8/8 at
  `--seed 40`, the differ 12/12 at `--seed 1`, known families only.
  The scen1 eval's six desyncs (first 100 games on this code) sorted
  into three families with the new detail: Grain Sales' taken-card
  resolution (3 — seeds 332, 354, 382: the engine and the DLL
  disagree about whether the taken card was returned, played, or
  still in hand, and the hands drift), a one-card hand/deal drift
  (2 — seeds 338, 370: the bot's turn-8 deal differing in one card;
  the AI playing a card the engine's hidden-hand tracking did not
  contain), and Flower Power on a prevented war card (1 — seed 358:
  the engine paid 2 VP for Arab-Israeli War under Camp David where
  the DLL pays nothing; also rules version 5's second fix, above —
  the old `_flower_power_check` void had failed to fire on the AI's
  play). The Grain Sales and deal families are open: the plan is
  volume with the per-option failure detail plus a hand-drift dump
  (the recent `CARD_LOCATION` records and both hands at the first
  mismatch), and the hidden-prompt harness
  (`runs/playdek/trace/emu_grain.py`) extended over the taken-card
  corners. Instruments landed for both: a per-card DLL location
  history (`Bridge.loc_history`) behind three sights — a `hand-drift`
  divergence dumping every diverging card's trail at the first
  visible-hand mismatch, the card's trail on the two "illegal in
  engine" fatals, and a `grain` line recording what every real Grain
  Sales resolution was read from (neither kind counts toward
  `max_divergences` or prints in the eval console; both ride
  `results.jsonl`). The grain-biased harness
  (`runs/playdek/trace/emu_grain2.py`: the emulated seat plays Grain
  Sales whenever offered) found one reproducible desync in 149 seeds
  and it was Missile Envy, not Grain Sales: a headlined Grain Sales
  resolving before the bot's headlined Missile Envy leaves the
  engine's physical pick waiting on an exchange record the DLL only
  emits after its event stops asking — a deadlock. Two operator
  fixes: the pick now falls back to the giver's hand per the DLL's
  card locations, taking a *unique* Ops maximum (Missile Envy's own
  rule; a tie is the giver's choice and is never guessed), and the
  emulated giver's "Select Card to Give" answer is queued with its
  real hint so the pick can consume it exactly (hotseat; against the
  AI no prompt exists and a tie stays a diagnosable divergence).
  After: the sweep 149/149, hotseat 8/8, the differ 12/12.
- **Tenth pass, from the counter1 easy eval** (6 desyncs + 3 void over
  120 games; the ninth pass's evidence lines delivered — every grain
  trace carried the taken card's location history and every failed
  simulation its per-option stop point). The decisive trace, seed 390:
  a Grain Sales return where *neither* choice reproduced the DLL with
  **no state diff** — both simulations stalled at the granted-Ops
  decision with no fact to spend, i.e. the AI declined the 2 Ops the
  engine made mandatory. Every granting card's text says "may then
  conduct Operations": **rules version 6** makes the grant declinable
  (`push_event_operations` pushes a `pass` alongside the spends;
  Missile Envy's taken card played for Ops is a card play and stays
  mandatory). The payload rides the layout's `other` flag like
  realignment's "stop" — no layout bump, checkpoints load. Bridge:
  `_answer_ops_type` now bounds its facts at the seat's next queued
  card play (a fact after it is that play's, not the grant's — the
  mis-attribution that plausibly seeded the family's silent drifts,
  seeds 354/386) and reads an empty bound as the decline; the bot's
  own pass is cut by `narrow` when the DLL's use prompt carries no
  stop entry ('Select Use For Operations' offers only Cancel — the
  DLL's AI declines but its human seat cannot; counted under
  `known`), and told as the stop entry where one exists. Greedy
  scores `pass` at 0: declining only beats an actively harmful spend.
  Yardsticks re-rated on the bump (the decision point): v3 vs Greedy
  0.939 over 400 at bid 2 (standing 0.940 — unmoved), Greedy against
  itself 0.500 over 200 (0.59/0.41 by seat, within noise of v5's
  0.52/0.48); the `r3-bid2` ladder stands. Verified: the grain sweep
  149/149, hotseat 8/8, the differ 12/12, known families only. The
  hand-drift family stays open (seeds 338/405's traces: a turn-deal
  differing in one card, unaccounted exits recorded) — next eval's
  volume, same instruments.
- **Eleventh pass, from the v3-easy-r6 batch** (the tenth pass's own
  decider: 120 easy games on the fixed bridge — 6 desyncs + 1 void,
  the silent-decline subfamily gone, and the surviving grain traces
  share a *second* root). Seed 354: the AI takes the Grain Sales
  card and plays a USSR event that needs the *bot's* input (Muslim
  Revolution's removals) — the DLL stops at the bot's prompt with no
  records emitted and its hand getters lagging, so both take and
  return reproduce the visible state and the location read picks
  return, one action round before the fatal. The fix is a
  prompt-fit veto in `_run_copy`'s judgment: a branch that stops at
  a bot decision the DLL's live prompt cannot answer (`_fits` over
  the prompt's meanings, only when the prompt is the bot's and
  nothing is left untold) is a false confirmation and fails — the
  take branch stops at the removal decision the prompt is asking,
  the return branch at an action-round play it is not. Verified:
  the grain sweep 149/149, hotseat 8/8, the differ 12/12, suite
  537. Still open, traces recorded: the one-card hand/deal drift
  (seed 315: 'Play Latin American Death Squads?' against an engine
  asking the other seat, hand 7 v 8; seed 391: a 1-VP drift walking
  from turn 7), and a trap-discard prompt against a bot event
  choice (seed 348).
- **Twelfth pass, from the standing traces alone** (no new DLL
  volume: the three open trace sets each yielded a root by static
  read against the per-game action logs). Seed 315 (v3-easy-r6): the
  Missile Envy physical pick took **Grain Sales' random draw** as the
  exchanged card — an exchanged Grain Sales fires at once and pulls a
  second card out of the same hidden hand, also as "fired" and later
  than the exchange, so the pick's LIFO read of `_fired` named the
  drawn card (the game log's action 255: `CHANCE event_choice
  Latin_American_Death_Squads`) while the DLL was asking the bot the
  take/return of the card it actually exchanged. The pick now
  excludes the live Grain prompt's drawn card from the `_fired` and
  `_last_moves` scans, and a live take/return prompt names
  `Grain_Sales_to_Soviets` directly (only Grain Sales asks it, and
  nobody had played it). Two footnotes from the trace: Playdek's
  giver handed over a 2-Ops card past three 3-Ops ones, so the
  ninth pass's unique-Ops-maximum fallback is a heuristic about the
  DLL, not its rule; and the engine's hand stayed one slot larger
  than the DLL's *before* the pick — the turn-deal ±1 is a separate,
  still-untraced root. Seed 405 (counter1-easy): the SALT
  Negotiations reclaim was read as declined — `_last_moves` keeps
  only a card's latest move, and ABM Treaty reclaimed at @1733 then
  replayed at @1759 had its recovery overwritten before the engine
  asked; the reclaim now reads from a dedicated `_reclaims` log
  (every `DISCARDED -> hand` record, consumed as answered), and the
  decline gained the Star Wars pump guard (no decline while the
  DLL's prompt is down: "no record yet" is not "declined"). Seed
  338 (counter1-easy): `_reveal_hidden_scoring_cards`' equal-count
  gate passed on an Ask Not that discarded eight and drew eight —
  the counts agreed while the slots did not correspond, and the
  drawn scoring cards took slots of the hand still being discarded
  (the last two discards had none: "unaccounted exits"). A card the
  DLL drew this turn now also waits while the physical seat has
  unreplayed exits or queued moves; the reveal follows once the
  engine catches up. Verified: suite 537, the grain sweep 149/149,
  hotseat 8/8, the differ 12/12, known families only. Still open:
  seed 391's 1-VP drift (turn 7 AR 5, at the USSR's
  `un_intervention` play of Alliance for Progress — the DLL's VP
  moved as if the cancelled event's award fired, either a Playdek
  reading or a mode misread; the Warsaw Pact simulation failure two
  turns later is downstream of the standing VP diff, whose 'add'
  branch reproduced the influence exactly), the trap-discard corner
  (seed 348), and 315's pre-pick ±1 slot. Measured at volume the same
  day, back to back on seeds 300–419: `v3-easy-r6b` (the pre-fix
  binary) reproduced seed 315's pick fatal *verbatim* and seed 371 in
  the SALT shape, and repeated the US-seat zero (0/56, 0/113 across
  the two v6 batches); `v3-easy-r7` (the fixed code) — none of the
  three roots recurred (315 runs a turn further into a different
  family, 371/338/405 clean), desyncs 9 → 7, void 5 (all the
  documented trapped-seat held-scoring-card difference), and the US
  seat scored 2/57 (USSR 5/51). The surviving desyncs sort into: a US
  influence/coup the engine missed before a Grain Sales simulation
  (r7 seeds 324/408 and 315's new stop, r6b 344/382 — the biggest
  family, plausibly one root in the hidden-seat US replay), the
  engine over while the DLL still headlines (330, 350), and two
  illegal-in-Playdek singletons (300: 'Place 2 Influence' in Europe;
  388: South African Unrest's choice against the bot's card play).
- **Thirteenth pass, from the r7 read: the trapped-seat void adopted,
  the handed-card family fixed.** **Rules version 7**, the DLL's
  reading of the traps: a seat still in Bear Trap / Quagmire is
  exempt from the held-scoring-card loss at the turn's end and
  carries the card over — the DLL's own seat cannot even play it
  there (`ffi.TRAP_SCORING_CARD` is an inert entry), so holding it is
  not that player's choice. `_end_of_turn` filters trapped holders
  (an untrapped holder still loses, alone even when the trapped
  opponent also holds one); the optional scoring play under the trap
  *stays* — the DLL's AI seat does play one at times (the bridge's
  "a scoring card is played" path) — and `narrow` cuts the *bot's*
  scoring play to the keep, under a new `known`, since the DLL
  cannot express it (the old drift behind the "engine plays it under
  the trap, the DLL holds it" face). This retires the five r7 voids.
  Bridge fix, the r7 grain family (seeds 324/408, r6b 344/382): the
  Grain-handed card's `play_mode`/`EVENT_OPS_ORDER` simulate branch
  required an *empty* queue, but the DLL runs whole actions ahead —
  a taken card played as its event leaves nothing queued for itself
  while the seat's next play is already queued behind it, so both
  take/return simulations stalled at `play_mode` and the mismatch
  followed. The branch now treats a queued *card* play as the next
  action (the simulation spends it downstream) and defers only to a
  queued *use*. Diagnostics: the finish-match game-over fatal now
  names the engine's reason, turn, VP, both engine hands and the
  DLL's hand counts (seeds 330/350 — whose engine end a log replay
  shows came from a revealed scoring card in the hidden US hand with
  no trap active: the next batch's trace will say which reveal).
  Yardstick on the bump: v3 vs Greedy 0.945 over 400 at bid 2
  (standing 0.940 — unmoved; the `r3-bid2` ladder stands; the
  Greedy-self pairing is not runnable by name in `wopr.eval`).
  Verified: suite 538, the grain sweep 149/149, hotseat 8/8, the
  differ 12/12 with zero fatals, known families only. Measured
  (`v3-easy-r8`, the same seeds): **void 0** — four of r7's five
  trapped-carry games play clean to completion, the fifth converts to
  a deeper desync (345, South African Unrest) — and the handed-card
  family did not recur (324/408 clean). Desyncs 14/120 (9 → 7 → 14
  across the three identical-seed batches: the AI is not
  deterministic, families matter more than the count), and the new
  game-over diagnostic paid off at once: seed 405's end names a
  revealed **Middle_East_Scoring** standing in the engine's USSR hand
  at turn 6's end while the DLL, hands 9/9, is already dealing turn 7
  — the reveal-drift family's first named card. The surviving shapes:
  simulations blocked by *earlier* board drift (How I Learned at
  315/323 with the DLL 5 military Ops ahead — a coup the engine
  missed; Junta at 354/383; South African Unrest at 345, r7's 388),
  the DLL-ahead `play_mode`/`ops_type` mismatches outside the handed
  set (350/361, 360/402 — the granted-Ops attribution face), 'Place
  2 Influence' illegal-in-Playdek (314/379, r7's 300), and one grain
  take/return where both branches stop clean without confirming
  (384). US seat 2/52 — 0.035–0.038 on the fixed bridge across
  r7/r8, against 0/113 before it.
- **Fourteenth pass, from the r8 traces: one judge fix, two
  instruments.** The judge fix: `_simulate` no longer blames a choice
  for the DLL being behind — a branch that ran out of facts with *no
  state diff at its stop* (stuck answering the other seat's decision,
  not rejected by the prompt-fit veto, tracked by `_sim_stalled`)
  makes it return None, so the ordinary retry loop advances the DLL
  and asks again. R8 seeds 315/323 were this: How I Learned's right
  DEFCON choice reproduced the state and stalled at the next action
  round, the premature "none reproduces" fatal blamed the choice, and
  the header's alarming mil-ops diff (Playdek 5, engine 0) was only
  the event's own +5, absorbed from the DLL but not yet applied
  engine-side at the choice. The instruments: China Card ownership
  joins `state_diffs` — seed 405's root walked back to a China fork
  (the engine played the USSR's Cultural Revolution as Ops per the
  influence records while the DLL's USSR apparently took the China
  Card, and the fork stayed invisible — ownership was never compared
  — until Nixon's +2-VP-or-take branch and a held-card end); and the
  illegal-in-Playdek fatal now carries the recent records, without
  which the 'Place 2 Influence' family (r8 314/379, r7 300 — the DLL
  asking a two-influence placement the engine has no source for) is
  untraceable post-hoc. Verified: suite 538, the grain sweep 149/149,
  hotseat 8/8, the differ 12/12 zero fatals.
- **Fifteenth pass, from the r9 read** (`v3-easy-r9`, the fourteenth
  pass measured: desyncs 14 → 7, void 0, US seat 4/57 = 0.070 after
  0/113 pre-twelfth-pass — and every instrument delivered). The
  China comparison located seed 306's fork live (China Card Playdek
  US, engine USSR, flagged from t6 AR1 with the hand-drift dump
  beside it); the record dump named the long-standing trap-discard
  corner: seeds 348/416 both stuck with the bot's Cuban Missile
  Crisis defuse answer ('Cuba') queued against a DLL prompt that
  offers no defusing — the DLL folds defusing into the play prompt
  and gives a trapped seat's round no defuse entry at all, while the
  engine offers it as its own round-start choice. The fix: `narrow`
  cuts the round-start offer to "skip" when the bot's live prompt
  shows no defuse entry (a `known`; the at-coup offer keeps its
  options — the DLL asks that one as its own prompt). The stall-retry
  fix held: r8's 315/323 How-I-Learned fatals are gone, and the two
  surviving How-I-Learned/Junta fatals (404, 323) show genuine
  standing drift in their headers, not stalls. Left open with
  fresh traces: the China fork's origin (306), an event-resolution
  ordering where the DLL resolves the bot's UN-Intervened Socialist
  Governments removals itself (325), Marshall-Plan-vs-engine target
  lists (373), and the deep-drift pair (323, 404). Verified: suite
  538, sweep 149/149, hotseat 8/8, differ 12/12 zero fatals.
  Measured (`v3-easy-r10`): desyncs 10/120, void 0, the defuse cut
  fired once cleanly and the 348/416 shape is gone. The count's noise
  band over the five identical-seed batches is 7–14; what the r10 mix
  says is that the **small-VP-drift family now dominates** — four or
  five of the ten are games that ran to a ±20 VP end with the two
  programs one or two VP apart (323, 339, 345, 412, and 315's DEFCON
  twin), so the drift roots (Alliance-for-Progress/UN's 1 VP, the
  China fork's Nixon +2, whatever else) are what volume converts into
  desyncs now. Also seen: the deal/pool drift again (330,
  Southeast Asia Scoring dealt but not in the pool — r6b 402's
  shape), a reveal-drift held-card end again (350), the
  multi-country +1-US grain blocker again (396), and one new corner:
  the DLL's AI **headlined The China Card** (388) — the engine
  refuses (7.2.2 bars it), worth its own trace before deciding who
  is right. US seat 1/56 (batch noise; r7–r10 pooled ≈ 0.04).
- **Sixteenth pass: three roots from the r10 traces.** (1) **The 1-VP
  family was U2 Incident's rider**: seed 323's stable one-VP gap
  starts the moment the USSR plays a card under UN Intervention with
  U2 headlined the same turn — "if UN Intervention is played later
  this turn, the USSR gains 1 additional VP", which the engine
  carried as an explicit unmodeled simplification and the DLL pays.
  Modeled now (`turn_effects["u2_incident"]`, paid at UN
  Intervention's action-round play, whatever mode follows; the key is
  layout-unencoded — `UNENCODED_TURN_EFFECTS`, no LAYOUT_VERSION
  bump, queued for the next bump with the vocab fold). This is also
  r6 seed 391's t7-AR5 fork (the `un_intervention` Alliance for
  Progress thread) and r6b 379's. Yardstick after: v3 vs Greedy
  0.945/400 at bid 2 — unmoved, the ladder stands. (2) **Seed 315's
  every-batch turn-4 drift was the deal counting non-deal arrivals**:
  the engine deals the hidden hand from the DLL's *current* hand
  contents, and a card that entered by the Missile Envy exchange
  (pumped through deal + headline in one chunk before the engine
  dealt) counted as a card to deal — one slot too many. Only cards
  whose last move came out of the deck count now. (3) **The
  hand-one-up-with-a-revealed-scoring-card family (r10 388/350, r8
  405, r7 330) feeds from premature declines**: seed 388's Ask Not
  discarded three, the engine's replay stopped at two ('stop' taken
  while the third record was still on its way) — the thirteenth
  pass's Salt-specific pump guard is generalized to the whole
  card-choice decline path (and 388's "China headline" fatal was
  this drift's leftover queued play, not a Playdek China-headline
  rule). One regression caught by the sweep and fixed in the same
  pass: the guards must stand down when the DLL's game is over (a
  finished game is at rest whatever its prompt; sweep seed 37).
  Verified: suite 539, the grain sweep 149/149, hotseat 8/8, the
  differ 12/12 zero fatals.
- **Seventeenth pass: the DLL resolving the bot's own choices, from
  the r9 singletons.** Two faces of one behavior — Playdek sometimes
  resolves a choice that belongs to the *bot's* seat without ever
  prompting it. Seed 325: the AI played Truman Doctrine event-first;
  the removal is the US's (the bot's) choice, the engine asked its
  policy, and the answer stuck against the DLL's next play prompt —
  the DLL had picked internally. Seed 373: the bot's own Independent
  Reds with one country worth choosing was auto-resolved by the DLL
  while the engine offered all five, and the bot's different pick
  (Hungary) hit Marshall Plan's list. `narrow` now follows the DLL's
  recorded resolution in both shapes: a bot `EVENT_INFLUENCE` whose
  DLL prompt cannot express it, and a bot country `EVENT_CHOICE`
  whose candidates share nothing with the DLL's prompt, are cut to
  the option whose country the DLL's influence records already show
  changed (a `known`; one option, so the telling loop drops it as a
  forced step). Instrument: every `CHINA_CARD` record now lands in a
  `_china_log` (seq, holder) printed with the China state diff and in
  `recent` — seed 306's fork (the engine played the AI's Ussuri as
  Ops while the DLL's China moved to the US) gets pinned to a record
  seq next time. Verified: suite 539, sweep 149/149, hotseat 8/8,
  differ 12/12 zero fatals. Parked with traces: r11's SAU stop (317,
  a ±2 VP transient at a turn boundary the sim judge cannot yet see
  past), Wargames (304), the grain +1-US blocker (324, recurring).
- **Eighteenth pass, from the r11 read: the judge learns to wait.**
  R11 (the sixteenth pass measured): desyncs 10/120, void 0, **seed
  315 clean for the first time in six batches** (the deal fix held),
  the U2 rider held (no recurrence of its shape), and the rates
  recovered to the pre-flag baseline — USSR 7/55 = 0.127, US 2/55 =
  0.036, both 0.082 (the standing v6 baseline was 0.089). Four of
  the ten desyncs were South African Unrest simulations failing on
  *transient* diffs of the DLL's lead — a whole play (a Space Race
  play's +2 VP, seeds 317/361) whose records had not yet been
  absorbed when every branch was judged, so "none reproduces" fired
  while all the evidence was still in flight. `_simulate` now
  remembers where the DLL stood on an all-branch failure, returns
  None so the ordinary loop advances the DLL, and fatals only when a
  retry fails with nothing new absorbed since the last one (nested
  simulations just stall their branch). The rest of r11: end-of-game
  turbulence around the DLL's DEFCON-1/±20 endings (324, 332, 382),
  a granted-Ops face (369), a How-I-Learned chooser mismatch (390),
  Wargames (304), and a fresh 1-VP shape near a U2 headline with UN
  Intervention never played (393) — one to watch, not yet a root.
  Verified: suite 539, sweep 149/149, hotseat 8/8, differ 12/12
  zero fatals. Measured (`v3-easy-r12`, all seven passes aboard):
  desyncs 11/120, void 0 — the seven identical-seed batches band at
  7–14 (9, 7, 14, 7, 10, 10, 11) and the SAU-transient cluster is
  gone, though the retry drains some of those corners into
  decision-mismatch fatals at the same spots (the per-branch detail
  is now re-attached to that fatal, so the trace quality is kept).
  **The bridge arc is review-ready**: voids retired since rules v7,
  rates at the standing baseline, and every surviving desync is a
  named family with traces — the hard core is hidden-seat inference
  drift surfacing at HIL/Grain simulations (315/411, 314/384/412),
  the granted-Ops attribution face (369/390), 388's hand-one-up
  China-leftover, and end-of-game turbulence (379). Seven bridge
  passes (twelfth–eighteenth) and rules v7 landed from six batches
  in two days; the remaining families are one-per-batch corners
  whose next roots want fresh instruments, not more volume.
- **Nineteenth pass, from the kick2-easy/kick1-veto-easy traces: the
  judge learns to see past standing drift.** The two 2026-09-01
  decider batches (240 fresh games; full desync texts in
  `runs/playdek/desync-mining-2026-09-01.txt`) put per-branch detail
  on the hard core's plurality face: the engine waiting on a
  hidden-seat `event_choice` the DLL never surfaces. The Junta pair
  (kick2-easy 358/397) root-caused cleanly — the right branch
  ('coup', its roll already queued) reproduced everything except a
  one-influence drift that stood *before* the choice, and the judge's
  zero-diff bar converted standing drift into a fatal (358: the
  engine's phantom USSR 1 in Colombia even fed the coup arithmetic,
  US 5 vs the DLL's 6). The fix: `state_diffs` refactored over keyed
  pairs (`state_diff_keys` — which dimensions disagree, not by how
  much), `_run_copy` records each diff-judged stop's keys, and once
  every retry is spent a failing branch whose residual keys ⊆ the
  pre-choice keys — no new divergence of its own — is carried
  (fewest residual keys, then fewest facts left) instead of
  fataling; the same rescue answers the drain's deadlock, where the
  DLL already asks the bot's next prompt and nothing more will
  arrive (`_sim_fail_pick`/`_sim_forced`). Every carry is logged as
  a `drift-pick` divergence plus a `known`, and the game still
  answers for itself downstream: a wrong carry desyncs later or
  fails the finish's winner comparison — strictly no worse than the
  certain fatal it replaces. The deeper roots stay open with better
  traces: the multi-country +1-US drift (382's dump: four countries,
  VP, China ownership and a hand size adrift at once — one missed
  US play, plausibly), the game-over timing family (384/410/413),
  and the placement-region mismatch (363). Verified: suite 544, the
  grain sweep 149/149 (desyncs 0), hotseat 8/8, the differ 12/12
  zero fatals; the drift-pick fired nowhere in the harnesses — it
  only replaces fatals. Measured the same day, and the pass paid
  twice over: the kick2+veto player's old-judge batch hit **20/120**
  desyncs (its long games — USSR mean turn 7 — pile drift onto the
  endgame families), and the same player's fresh-seed batch on the
  new judge came back at **8/120, void 0**, known families only,
  the drift-pick firing 3 times; the kick3 decider beside it read
  9/120 (drift-pick ×6). Both in the standing band with games this
  long — the attrition tax on strong players is roughly halved.
- **Twentieth pass: instruments for the two deep roots, no behavior
  change.** Seed 382's dump decodes as: every diff but three was the
  in-flight play itself (the engine mid-asking Camp David's mode
  while the DLL races ahead) — the *standing* drift is exactly three
  orphaned US placements (Panama/Cuba/Guatemala +1 each), the
  granted-Ops attribution face: a grant the engine read differently
  than the DLL spent it, unfindable post-hoc because the records had
  scrolled out of `recent`. Instrument one: `_answer_ops_type` now
  emits a non-fatal `granted-ops` evidence line on every *real*
  resolution — side, ops, the read, the bound seq, the facts and the
  queue — the same pattern that cracked the grain family (diagnostic
  kind: never counts toward the cap). Instrument two, for the
  game-over timing family (384/410/413 — 413's coup lists showed a
  silent one-step DEFCON drift): the bridge keeps a `_defcon_log` of
  (record seq, level) transitions and the DEFCON state diff now
  carries its tail, so the transition the engine missed pins to a
  seq like the China fork does. Verified: suite 545, sweep 149/149
  (0 desyncs), hotseat 8/8, differ 12/12 zero fatals; the evidence
  lines fire in hotseat games with correct seqs. The next batch's
  drifts carry their own diagnosis.

### The match operator (`operator.py`) and the eval (`eval.py`)

`PlaydekOperator` is the physical-mode operator of a game against
Playdek's AI (docs/BOTS.md): the engine referees with the AI's seat as
the physical side, the operator is the `Player` under that side and
`Side.CHANCE`, and the bot's own seat is wrapped by `players()` so every
action it takes is told to the DLL as it is made. The engine leads for
the bot's seat, the DLL leads for the AI's:

- **The bot's actions become the DLL's prompts.** Each is queued as it
  is made and `flush` replies to the DLL's prompt as soon as the queue
  determines the option: a card prompt from `ACTION_ROUND_PLAY` (with a
  look at the `PLAY_MODE` behind it, since UN Intervention is its own
  card there and a scoring card has no use prompt at all), one "use"
  option from `PLAY_MODE` + `EVENT_OPS_ORDER` + `OPS_TYPE`, a country,
  a choice (`translate.find_choice`: cards and countries by id, DEFCON
  by the `0xA070+n` hints, the rest by label words, a decline by the
  `STOP` entry). A forced single-option step the DLL never asks about
  is dropped; anything else that does not fit the prompt is a fatal
  mismatch. Before the bot decides, `narrow` cuts its options down to
  the ones the DLL offers for the same choice when the DLL is at that
  prompt (De-Stalinization's sources: the DLL forbids relocating back
  into a country just emptied, the card text does not), so the play is
  legal in both programs (Junta's free Coup/Realignment likewise: the
  DLL confines it to the country placed in and offers "Pass" alone when
  that one has no target — a lone "Pass" the engine is about to ask a
  choice for waits for the bot's decision, which `narrow` cuts down to
  the decline; `_next_bot_decision_fits` peeks with a copy of the engine).
- **The AI's play is read off the records.** It is never prompted, only
  reported: its card and use from the `OUTPUT_ANIMATION_CARD` record of
  the card leaving its hand for the resolve slot, whose
  `animation_event_hint` is `0x8N01` for the use chosen (`0x82` event,
  `0x83` event first, `0x84` influence, `0x85` realignment, `0x86` coup,
  `0x87` space race; `0x8N02` is the automatic other half of an Ops
  play, `0x1` a scoring card, `0x2` an event another event fired); its
  headline from the card's move to the headline location; its coup,
  war and realignment targets from the roll records (and "stop" once
  no further realignment is reported); its influence, Ops or an
  event's, from the influence *history* — each country's values since
  the two states last agreed at rest, the earliest change past the
  engine's value in the right direction first, so that a placement a
  later action in the same chunk undid (a Marshall Plan point realigned
  away, or removed by the next card's event) is still made and then
  undone by that action's own records -- a surplus gone again with no
  dice on the country and no card played in between is a transient of
  one event's own resolution (Nasser), not a placement; any order is
  legal since a point's cost depends only on its own country.
  Ops an event grants (a boycotted Olympic Games' sponsor, CIA Created)
  from the earliest of the dice or influence changes that followed —
  not the first kind found, or the seat's own action round's coup would
  be taken for the granted Op; a discard from the card
  that left its hand, a Five Year Plan discard from the resolve record
  of the card it fired, a Grain Sales draw from the reveal record. The
  China Card has no card location: its holder comes from `CHINA_CARD`.
  An either/or the records do not name is **simulated**: each option is
  played on a copy of the engine with the rest of the chunk answered
  from the same facts, and the one that leaves the copy in the DLL's
  state is it (nested choices recurse; several matching is noted under
  `known`). The DLL is at rest whenever the engine asks — it is only
  pumped to the bot's next prompt or the game's end — so "no record
  yet" means "not done", never "not yet arrived".
- **Setup order.** The DLL deals after the opening placements, the
  engine before; the bot's placements cannot wait for a deal the DLL
  has not made, so the engine runs with `deal_after_setup=True`
  (docs/ARCHITECTURE.md) and the bot places without sight of its hand,
  as Playdek's players do.
- **Hotseat emulation** (`emulate=`, `eval --difficulty hotseat`): the
  same protocol against a DLL-prompt policy on the other seat, its
  records feeding the engine exactly as the AI's would, at 10k prompts
  a second instead of 15 s a decision — how the operator is tested.
  Two things differ: a hotseat game re-emits an action's records at
  its commit (the next action boundary, after the next `ACTION_ROUND`
  record), a game against the AI does not. `Bridge.replayed` matches
  the re-emission off a FIFO of the records acted on (dice, card plays
  and influence values, in emission order) in hotseat mode only: kept
  against the AI, the FIFO's head would be the game's oldest record and
  a later roll equal to it would be taken for a replay. The influence
  values must go through it too: a re-emitted `COUNTRY_INFLUENCE`
  carries the value of its time, stale if a later action changed it,
  and would otherwise enter the history as a new change. And the
  hotseat re-asks the very first prompt, dropping the first answer
  *and* the records it produced: `_reasked` answers it again and takes
  those records back.
- **What a game costs.** The AI spends 15 s on every decision, a card
  play being three or four of them: a game that runs six or more turns
  is 30–50 minutes of one core. (The "2½ minutes" measured earlier was
  a random opponent ending games in a turn or two.) The eval's
  `--workers` is the only lever; a hundred games are an afternoon.
- **Desync.** A fatal divergence (the engine asks what the DLL cannot
  answer, the bot's action has no option in the DLL, no choice
  reproduces the DLL's state) ends the game as a `Desync`; the game
  does not count. The state is compared at every card prompt of the
  bot's and the difference reported, as in the differ.

`python -m wopr.playdek.eval --games 20 --policy joshua=baselines/r2/v1/joshua.pt
--difficulty hard --workers 8 --out runs/playdek/<name>` plays the games
in a process pool (one Playdek instance per worker, one game at a time
each), seats alternating by game
index, seed `--seed + index`, and writes every game's replay log
(`<out>/games/`), every result (`<out>/results.jsonl`) and the tally
(`<out>/summary.json`): the policy's win rate per seat with a Wilson 95%
interval, the endings, the desyncs, the void games by reason, the
`known` counts. The AI is not
deterministic for a seed, so it is a sample, not a replay; `--policy
greedy|random|first` give the yardsticks.

### The distilled teacher (`wopr/distill.py`)

The one sanctioned way the DLL's AI teaches (the relaxing-SELF-PLAY-ONLY
entry, docs/JOSHUA.md 2026-08-30): not as a live opponent — one game per
process at 15 s a decision rules that out — but distilled from the games
it already played. Every eval batch's `<out>/games/` logs are replayable
engine records with the AI's decisions in them. `wopr.distill harvest`
replays them on the current engine and records, at each AI-seat decision
with at least two options, the encoded observation and the chosen option
index; `wopr.distill train` fits a fresh `JoshuaNet` to the rows by
cross-entropy and saves an ordinary checkpoint that `--pool-seed` can
place in the training mix.

The engine's physical-mode mirror does not know most of the AI's hand,
so `harvest` determinizes it in hindsight before encoding: known cards
kept, the chosen card and the AI's later same-turn card plays forced in,
the rest sampled from the cards the observation leaves unseen, seeded by
(game seed, step index). Desynced and void games are excluded; logs that
no longer replay under the current rules are skipped and counted in the
corpus `manifest.json`. `harvest --workers N` replays in a process pool but
consumes the games in order, so the shards hold the same rows in the same
order at any worker count; the held-out fold is by game hash (batch name /
file name, mod 10), so a corpus extended with new batches keeps every old
game's fold and a clone trained on the old corpus can be scored on the
extended one's held-out rows without leakage. The clone's value head is
untrained — a pool opponent's `choose` reads only the logits.

`train`'s recipe knobs (the falken2 sweep, docs/JOSHUA.md 2026-09-02):
`--hidden`/`--gnn-layers`/`--option-hidden` size the `JoshuaConfig`;
`--weight-decay` is AdamW's decoupled decay (0 = plain Adam, the falken1
recipe); `--label-smoothing` spreads target mass over the row's *legal*
options only — the masked slots carry `finfo.min` logits, so torch's own
`label_smoothing=` would average their negative log-probabilities into the
loss; `--lr-decay` multiplies the learning rate after every epoch without
a new best held-out top-1 (early stopping by `--patience` is unchanged).
The report `distill.json` records the config and every knob.

A corpus can also pull a *training run* directly — kickstarting
(`train.py --kickstart <corpus> --kickstart-coef/-batches/-batch-size`):
after every PPO update, `KickstartCallback` (wopr/callback.py) runs a few
cross-entropy minibatches from the corpus's training fold on the policy's
own optimizer, gradients clipped like PPO's — interleaved steps, not a
joint loss, so SB3's update stays untouched. The held-out fold is never
trained on by either path and is the absorption metric: `wopr.distill
top1 <checkpoint> --corpus <dir>` reports any checkpoint's held-out
top-1 against the legal-uniform floor.

## What Joshua cannot do yet

See [LIMITATIONS.md](LIMITATIONS.md#bots).
