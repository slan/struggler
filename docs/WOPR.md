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
   (anchors exist as flags and are off since v5; JOSHUA.md says why).
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
v2 and v4 in JOSHUA.md), and shape the reward (the final score in the
terminal reward made both seats play for the track — the margin
experiment there).

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

`K_MAX = 96` bounds the option count (the largest legal set is "every
country"); exceeding it raises rather than truncates, because it would
mean a decision is decomposed wrong. An engine effect flag missing from
`TURN_EFFECTS`/`GAME_EFFECTS` also raises, and
`tests/test_joshua_features.py` greps the engine source so that drift is
caught by the suite, not by a rare event mid-game. Payload words outside
`OPTION_VOCAB` degrade to the `other` flag plus position.

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

`--handicap N` opens every training game with the US N VP ahead
(`Engine.new_game(starting_vp=N)`, carried to the collectors in
`ArenaSpec`): a tournament bid for the USSR seat. With strong policies
the USSR seat wins most games between equals (JOSHUA.md), so the
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
every experiment, frozen or not; JOSHUA.md reads the rows. A run that
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
  [JOSHUA.md](JOSHUA.md).
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

## What Joshua cannot do yet

See [LIMITATIONS.md](LIMITATIONS.md#bots).
