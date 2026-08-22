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

`JoshuaNet`, ~270k parameters at defaults:

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
  same games through either, step for step; the suite pins it. Each
  collector resolves its own opponents (`StandardOpponents.for_worker`:
  same policies, its own RNG streams) and runs pool-net inference over
  its own slots, so that batching is kept inside the worker.

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
and the remainder against `--anchor`: `random`, `greedy`, `first`, or a
schedule such as `random,greedy` (`pool.AnchorSchedule`), which walks the
list in order and promotes once the learner's win rate over the last
`--anchor-window` anchor games reaches `--anchor-promote`; the last
anchor is kept for good, and `metrics.csv` records the current one. A
terminal reward against an opponent the learner never beats is a
constant, and so is one against an opponent it always beats
(JOSHUA.md, v2 and v4). Fractions summing to 1 leave no anchor games at
all: pure self-play against the pool. While the pool is empty, pool
games are self-play.

### The pool (`wopr/pool.py`)

A directory of snapshots taken every `--snapshot-every` updates, with
`stats.json` tracking the learner's record against each. Sampling is
prioritised fictitious self-play: weight `(1 − learner win rate)^hardness
+ floor`, so opponents the learner still loses to come up more often,
unplayed ones count as even, and none ever drops to zero. `--pool-window`
restricts sampling to the newest N.

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
regression to look at, not to train through. Arguments after `--` go to
`train.py` — and a resumed segment takes its PPO hyperparameters from
those flags, not from the zip (`n_steps` excepted: it sizes the buffer)
— which is how a hyperparameter experiment runs through the loop;
`--no-promote` gates and logs without freezing, so two arms can be
measured against the same champion. `runs/<run>/loop.csv` records every
generation.

## Metrics (`wopr/callback.py`, `runs/<run>/metrics.csv`)

Per update: games, win rates (overall, per seat, vs pool, vs anchor),
the current anchor, rollout and update seconds (and, with collectors,
the seconds spent waiting on them), draw rate, episode length and mean final turn, policy health —
`entropy`, `k_valid` (mean legal options), `entropy_ratio = H / ln K`
(≈1: not choosing yet; ≪0.3 with many options: collapsed), `k_eff = e^H`
— and SB3's `approx_kl`, `clip_fraction`, `explained_variance`, losses.
Warning signs: KL well above `--target-kl` or clip fraction above ~0.3
(step too big), explained variance near 0 after the first few hundred
games (value head lost), entropy ratio falling while win rate does not
rise (collapse).

## Run layout

```
runs/<run>/
  config.json      arguments + games_done (resume reads it)
  ppo.zip          SB3 model (optimizer state included)
  joshua.pt        latest plain checkpoint: what `--us joshua` loads
  pool/            snapshots + stats.json
  metrics.csv
```

`runs/` is gitignored. Re-running `train.py --run X --games N` with a
larger `N` resumes; a smaller or equal `N` is a no-op.

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
  [JOSHUA.md](JOSHUA.md).
- **Epochs.** `--n-epochs` defaults to 2. Measured through the loop
  from v8, one generation each way against the same champion: 2 epochs
  gated 0.635 vs 4 epochs' 0.573, beat Greedy 0.995 vs 0.960, tied head
  to head (0.455), with healthy KL and explained variance in both — at
  28% less wall time per generation. Pass `--n-epochs 4` for the old
  recipe.
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
- **Reward.** Terminal only, by design. VP is the game's literal score and
  a natural dense signal if learning stalls; it is not on by default
  because shaping a two-player zero-sum game tends to teach the shaping.

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
[JOSHUA.md](JOSHUA.md) says the engine is a third of a learner step at
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
  / CHOICE`, plus the UI-only `CANCEL` and `SWITCH_CARD`), the struggler
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
records are emitted twice (preview and commit), the first hotseat prompt
is re-asked once, turn-2+ headline prompts arrive under the local seat's
id whoever is picking (the cards say whose hand), and the `*_player_index`
of roll records is the seat's id.

Findings from the first four random games, and what became of each:

- **Fixed upstream** (branches off `upstream/main`, merged here): the
  engine's Military Ops counted past 5 (the track stops there); the
  game did not end when a scoring card was held past the end of the
  turn (the engine forbade holding instead — now it offers the whole
  hand and `_end_of_turn` decides); realignment could not stop after
  the first attempt (now `{"country": "stop"}`); the Chinese Civil War
  space was a country of the standard game (now a variant-only space
  behind `Engine.new_game(variants=...)`, the layout unchanged).
- **Documented, DLL-stricter**: De-Stalinization will not relocate
  influence back into a country it was just removed from; the card
  text has no such clause, so the engine allows it. The harness counts
  it under `known`.
- **Harness false alarms, removed**: "Done Removing" / "Do Not Relocate"
  / "Do Not Discard" — the engine had those choices all along.
- **Open**: one realignment (Cameroon, USSR 1 / US 0, DLL rolls USSR 6
  / US 5, no neighbour controlled) removed the USSR influence in the
  DLL but not in the engine — either the DLL's roll fields are not what
  they look like or a modifier differs; more samples needed. Defectors
  played event-first (the engine fires nothing, by design) and either/or
  choices matched by label words are reported, not resolved.

Then the `PlaydekOperator` for Joshua and the eval.

## What Joshua cannot do yet

See [LIMITATIONS.md](LIMITATIONS.md#bots).
