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
python -m wopr.train --run first --games 2000         # train (resumes if the run exists)
python -m wopr.eval --games 200 first=runs/first/joshua.pt random greedy
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

## The arena (`wopr/arena.py`, `wopr/vec_env.py`)

`Arena(n_games, seed, seat_assigner)` owns N engines. Each game's two
seats are assigned *policy ids* (strings) when it starts; the arena only
groups pending decisions by id (`pending()`) and applies option indices
(`apply()`), resolving `Side.CHANCE` frames itself — their single
pre-rolled option is not a choice. `play_out(arena, {id: policy})` runs
everything to completion in batched rounds; it is what evaluation uses.

`WoprVecEnv` adapts the arena to SB3's `VecEnv`: one env per slot. After
the learner's action the env fast-forwards through chance and through
every decision belonging to a *non-learner* seat (answered in batch by
the registered opponents) and stops at the learner's next decision —
which, in self-play, may belong to the other side. Each row's reward is
the game outcome **for that row's mover** (+1/−1/0) on the row after
which the game ended.

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

## Metrics (`wopr/callback.py`, `runs/<run>/metrics.csv`)

Per update: games, win rates (overall, per seat, vs pool, vs anchor),
the current anchor, rollout and update seconds, draw rate, episode length and mean final turn, policy health —
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
- **Throughput.** Training is *update-bound*, not rollout-bound. One
  update is 64 games × 128 learner decisions: the rollout takes ~3–4 s
  (~2.7k learner decisions/s against the random anchor or itself, ~1.3k
  against Greedy, ~1.75k with pool snapshots in play) and the PPO update
  — 4 epochs × 8 minibatches of 1,024, forward and backward — ~3.4 s at
  16 threads under the default `--precision bf16` (~6.6 s in fp32).
  `metrics.csv` records both (`rollout_s`, `update_s`). The update is
  the network's FLOPs, mostly the graph layers' linears over 85 nodes ×
  128 hidden; the legal-rows option head and a plain `matmul` for the
  adjacency aggregation took the fp32 figure from 10.7 s with identical
  outputs. On the rollout side a learner step is about 40% policy
  forward at 8+ threads, the rest engine `step` and feature encoding;
  against a net opponent the opponent is asked ~8 times per learner
  step at a mean batch of 8 (the tail rounds, where few slots still
  wait on it), which is where its 40% goes — not the number of
  snapshots. Multi-process collection belongs *below* the layout
  contract (several arenas, one buffer), not in SB3's `SubprocVecEnv`,
  whose workers expect gym envs. See the August 2026 entry in
  [JOSHUA.md](JOSHUA.md).
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

## What Joshua cannot do yet

See [LIMITATIONS.md](LIMITATIONS.md#bots).
