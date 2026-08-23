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

A version of Joshua is born, compared, and promoted in three stages.
The pieces are specified in the sections below; this is the order they
run in and what each one decides.

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
  a clean run, and r<old>'s findings about the *game* are unverified
  until re-measured. Neither moved → same ladder, note the bump.
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
weighting; the league's exploiter agents it does not have.

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
the seconds spent waiting on them), draw rate, episode length, mean final turn, the learner's mean final VP (`vp_mean`, non-self-play games), policy health —
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
  config.json      arguments + games_done (resume reads it), the commit,
                   the layout version (a run from another version is refused),
                   the recipe and the --init checkpoint if any
  ppo.zip          SB3 model (optimizer state included)
  joshua.pt        latest plain checkpoint: what `--us joshua` loads
  pool/            snapshots + stats.json
  metrics.csv
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
  played. And the largest: the DLL conducts Grain Sales' 2 Ops *and*
  plays the taken card, against its own card text ("if returned, use
  this card to conduct Operations") -- the AI's favourite headline, so
  one USSR-seat game in ten: when neither take nor return reproduces the
  state but take with two more Ops does, the game is void. A game ended
  by such a `rules` difference is `MatchResult.void` (the reason), not a
  desync: the eval reports `void` by reason beside `desyncs`, and neither
  counts. Two engine
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
- **After the seventh pass:** the differ 279/280 (seed 93's trap,
  `known`); the hotseat emulation 32/32; the wide sweeps (`--games 120`)
  at `--seed 40`, `200` and `400`: 0 desyncs + 1 void (the trap's kept
  scoring card, `known`), 0, 0; the hidden-prompt emulation
  (`emu_grain.py 1 60`, seeds 1-59) 59/59. The Greedy-vs-AI runs of the sixth pass
  were not repeated.

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

## What Joshua cannot do yet

See [LIMITATIONS.md](LIMITATIONS.md#bots).
