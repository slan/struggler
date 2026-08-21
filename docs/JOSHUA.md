# Joshua: a Twilight Struggle bot that teaches itself

> *Greetings, Professor Falken. Shall we play a game?*

This is the idea behind `bots/joshua/` and the `wopr` arena — the why, the
bets, and what has been measured so far. The *contract* (array layouts,
arena API, buffer semantics) lives in [WOPR.md](WOPR.md) and is binding;
this document is the rationale and the lab notebook. Results are frozen
under [`baselines/`](../baselines/README.md).

## The thesis

Train a Twilight Struggle player **from nothing but self-play**: no
scripted teacher to imitate, no hand-written evaluation function, no
reward other than *who won*. The engine already has a greedy heuristic
bot and an LLM bot; both encode what their authors know about the game.
Joshua is the experiment in the other direction — how far does a network
get from the rules alone, with the rules enforced by an engine built so
that nothing else leaks in?

The model, WOPR, is what made the name irresistible: the computer in
*WarGames* learns tic-tac-toe by playing itself until it understands the
game. The machine this was first trained on happens to be called WOPR,
which settled it.

## Why this engine makes it tractable

`docs/ARCHITECTURE.md`'s five mandates read, almost line by line, as the
specification of a reinforcement-learning environment:

| Mandate | What it buys the learner |
| --- | --- |
| #1 pending-decision stack | Every interrupt (an event's sub-choice, a forced discard) is just the next decision. The agent never needs to know *why* it is being asked; the environment is one loop. |
| #2 atomic action space | Tens of legal options per decision, never thousands, so **the action is an index into `Decision.options`**. No global action id, no invented hierarchy, no illegal-action handling: illegal options do not exist. |
| #3 seeded, injectable RNG | A training game is reproducible from `(seed, slot, episode)`; an evaluation plays the *same decks* for every policy it compares. Chance is a logged decision, so nothing in the environment is hidden from the replay. |
| #4 `observe(side)` | The only input. The opponent's hand and the deck cannot leak into the features because the `Observation` does not contain them. Hidden information is represented honestly, as the `unseen` card location plus public counts. |
| #5 flat serialisable state | Cloning and checkpointing the game is free — the door to search on top of the learned value later. |

Most RL-for-board-games projects spend their first month building exactly
these properties. Here they were the engine's design goals before a
learner existed.

## The three ideas that matter

Everything else is standard PPO plumbing. These three choices are the
design.

### 1. A decision-centric arena

The arena never asks "whose turn is it". It asks **"which games are
waiting on which policy?"** Each seat of each game is assigned a *policy
id* — a string — when the game starts; the arena groups pending decisions
by id and applies option indices. Nothing in it knows what a policy is.

That one abstraction makes learner-vs-learner (both seats `"learner"`),
learner-vs-pool (one seat `"pool:u00040"`), learner-vs-Greedy, evaluation
matches between two checkpoints, and — later — a human or a remote seat
the *same code path*. It is also the shape a multi-seat engine backend
(shared memory, another language) would implement: fill the batch-first
buffers of the layout, answer the same question.

### 2. One head scores every option

Twilight Struggle has 23 decision kinds with heterogeneous payloads: a
country, a card, a play mode, an event branch named by a word. A flat
action space over all of them would be huge and mostly illegal at any
moment; a hand-built hierarchy would bake in today's decision kinds.

Instead each legal option gets a small feature row (what kind of value it
is, which country, which card, its position) and **one shared head
scores `[state latent, option row, node latent of its country, embedding
of its card]`**. The decision kind is part of the state. A coup target, a
headline card, and "participate in the Olympics" are all just options.
Adding a decision kind to the engine adds nothing to the model.

### 3. Alternating-perspective GAE

Self-play with one network and *both* seats in the same game means
consecutive rows of one game alternate sides. Values and rewards are
always from the mover's point of view, and in a zero-sum game the next
state's value for its mover is the negative of its value for me — so the
GAE bootstrap flips sign whenever the mover changes:

```
delta_t = r_t + gamma * s_t * V(s_{t+1}) - V(s_t)
A_t     = delta_t + gamma * lambda * s_t * A_{t+1}
s_t     = +1 if mover(t+1) == mover(t) else -1
```

With a fixed learner seat every `s_t` is +1 and it is ordinary GAE. With
both seats it is what lets **one game train both perspectives, one
buffer hold them, one PPO update learn from them**. A dozen lines, pinned
by hand-computed tests, and the difference between "self-play" as a
slogan and self-play as a training signal.

## The bootstrapping plan

Pure self-play against the latest policy cycles — rock learns to beat
scissors, forgets paper. The plan borrows the standard remedies and adds
what the engine offers for free:

- **A checkpoint pool with prioritised fictitious self-play.** Snapshots
  every N updates; opponents the learner still loses to are sampled more
  often (`weight = (1 − win rate)^2 + floor`). The learner's win rate
  against the pool hovers near 0.5 *by construction* — it is a
  thermostat, not a progress meter.
- **Anchors.** `random` and `first` as floors, `greedy` as the first real
  yardstick, Elo with `random` pinned at 0 so numbers compare across
  runs. Once a version beats every anchor, progress is measured against
  the *earlier versions* kept under `baselines/`.
- **Per-seat accounting, always.** The game is asymmetric (USSR moves
  first and owns the early war). A pooled win rate hides a policy that
  only learned one seat; every metric here is split US/USSR.
- **Curriculum for free.** `Engine.new_game(events=False)` is the game
  with no card events: influence, coups, realignments, DEFCON, scoring,
  space race. A pool trained there carries into the full game.
- **Terminal reward only.** VP is the game's literal score and an obvious
  dense signal; it stays off by default because shaping a two-player
  zero-sum game tends to teach the shaping. It is the fallback, not the
  plan.

## What has been measured

Details, trajectories and checkpoints per version are in
[`baselines/README.md`](../baselines/README.md). The short version:

| Version | Setup | Elo vs random | vs random (US / USSR) | vs greedy |
| --- | --- | --- | --- | --- |
| v1 | 4,000 games, 26 min, one CPU core; 30% self-play, 30% pool, 40% random anchor; terminal reward | +378 ± 48 | 0.88 (0.79 / 0.96) | 0.18 |

(200 games per opponent per seed, three eval seeds, argmax play. On the
same protocol Greedy rates +628 ± 53 over random, and `first` −52 ± 42 —
always picking the first legal option is about as good as picking at
random.)

What the first run taught:

- **The signal is there and it is fast.** Win rate against random went
  from ~0.45 to ~0.8 within 3,000 games; explained variance settled at
  ~0.75–0.8 after the first 500; entropy ratio fell steadily without
  collapsing (K_eff still ≈ 3.5 of ≈ 10 legal options).
- **Greedy is the real yardstick.** v1 beats random and `first` nine
  times in ten and loses to `GreedyPlayer` five times in six. Twenty-six
  minutes of self-play clears the floor; the hand-written heuristic is
  where the next several versions have to go.
- **The first plateau is DEFCON.** Mean final turn ≈ 5.5: most games
  still end by someone dropping DEFCON to 1. Learning not to, and letting
  the opponent do it, is the cheap early win; the scoring-card and
  headline game comes after.
- **The US seat lags.** 0.79 as US vs 0.96 as USSR against random. The
  net learned the easy seat first.
- **The arena found an engine bug.** Evaluating v1 hit a game that was
  over yet still had a pending decision (Pershing II's winning VP, then
  its influence step) — a mandate #1 violation random play reaches too
  rarely for the property tests to see. Fixed in the engine; the stronger
  the players, the more of the rules they exercise.

## Open questions and the road ahead

In rough order of expected value per effort:

1. **Throughput.** One process runs ~1.2–1.8k learner decisions per
   second, engine-bound. Collection across processes belongs *below* the
   layout (several arenas feeding one buffer), not in a gym-style
   subprocess wrapper. The GPU only matters after this: at the current
   model size the update phase is cheap and rollout batches are small;
   with 10–16 collectors and centralised inference it will not be.
2. **The US seat.** Weight pool and self-play games toward the US seat,
   or simply more games; watch the per-seat split.
3. **Greedy as an anchor and a curriculum opponent.** It is slow (≈ 0.6
   games/s — it recomputes its board value per option) and needs a pass
   before it is usable at scale.
4. **A shared-memory or rewritten-engine backend.** The layout is the
   contract; the Python arena is one implementation of it.
5. **Search on top.** `serialize()/deserialize()` make one-ply expectimax
   over the learned value cheap — with the caveat that a cloned engine
   carries its RNG state, so determinised search must re-seed the clone.
6. **History.** Joshua sees `Observation` only. What the event log adds
   — when each region was last scored, what the opponent has been
   playing for Ops — is a later feature, not a first one.

## Reproducing

```sh
uv sync --extra wopr
uv run python -m wopr.train --run first --games 4000 --n-envs 64 \
    --self-play 0.3 --vs-pool 0.3 --anchor random --snapshot-every 5
uv run python -m wopr.eval --games 200 first=runs/first/joshua.pt random greedy
uv run python -m wopr.baseline v2 --run first      # freeze it under baselines/
uv run python src/main.py --ussr joshua --joshua-checkpoint runs/first/joshua.pt
```
