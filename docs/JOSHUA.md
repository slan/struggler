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
| v2 | as v1 with the 40% anchor games against `greedy`, from scratch | +61 ± 41 | 0.54 (0.43 / 0.65) | 0.23 |
| v3 | v1 continued for 4,000 games against `greedy` (8,000 total, 28 min more in bf16) | +1077 ± 115 | 0.98 (1.00 / 0.96) | **0.59** (0.70 / 0.49) |
| v4 | v3 continued for 4,000 more (12,000 total, 25 min more) | +1529 ± 44 | 1.00 (0.99 / 1.00) | **0.92** (0.87 / 0.96) |
| v5 | 8,000 games, **no anchor**: 50% self-play, 50% pool, from scratch | +1156 ± 30 | 0.99 (0.99 / 0.98) | **0.66** (0.73 / 0.59) |
| v6 | v5 + 4,000 games by the loop (gate 0.84 vs v5) | +1425 ± 83 | 1.00 (0.99 / 1.00) | 0.91 (0.88 / 0.94) |
| v7 | v6 + 4,000 (gate 0.70 vs v6) | +1407 ± 28 | 1.00 | 0.92 |
| v8 | v7 + 4,000 (gate 0.73 vs v7); 20,000 games, never an anchor | +1394 ± 30 | 1.00 | **0.96** (0.96 / 0.95) |
| v9 | v8 + 4,000 games with 2 PPO epochs (the A/B winner; 24,000 games) | +1527 ± 26 | 1.00 | **0.99** (0.97 / 1.00) |
| v10 | v9 + 8,000 games by the loop (one failed gate, then 0.63 vs v9); 32,000 games | +1499 ± 46 | 1.00 | **0.98** (0.97 / 0.98) |
| v11 | 8,000 games from scratch, **hidden 256**, 4 epochs (the capacity A/B winner; 47 min) | +1244 ± 31 | 0.99 (0.99 / 1.00) | **0.91** (0.85 / 0.97) |
| v12 | v11 + 4,000 games by the loop (gate 0.79 vs v11); 12,000 games | +1452 ± 15 | 1.00 | **0.98** (0.97 / 0.99) |
| v13 | v12 + 8,000 (one failed gate at 0.547, then 0.625 vs v12); 20,000 games | +1460 ± 14 | 1.00 | **0.98** (0.98 / 0.97) |
| v14 | v13 + 4,000 (gate 0.571 vs v13, worst seed 0.550); 24,000 games | +1466 ± 4 | 1.00 | **0.98** (0.98 / 0.99) |
| v15 | v14 + 8,000 (one failed gate at 0.489, then 0.569 vs v14); 32,000 games | +1461 ± 33 | 1.00 | **0.99** (0.97 / 1.00) |
| v16 | v15 + 4,000 trained at an 8-VP USSR bid (gate 0.606 vs v15); 36,000 games; level with v10 (0.508) | +1339 ± 4 | 1.00 | **0.98** (0.97 / 1.00) |

(200 games per opponent per seed, three eval seeds, argmax play. Elo is
fitted per version against every earlier one, so the scale stretches as
the field widens: in v1's fit Greedy rated +628 ± 53 over random and
`first` −52 ± 42 — always picking the first legal option is about as
good as picking at random; in v3's fits Greedy rates +999 to +1227 and
v3 sits 8–12 points above it in each.)

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

What the second run taught:

- **The yardstick is not a teacher.** Trained from scratch with 40% of
  its games against Greedy, v2 won 19% of those by the end (v1 won 76%
  of its games against `random`), so with a terminal reward nearly half
  of every update's outcomes were the same −1. It did not learn to beat
  Greedy (0.23 vs v1's 0.18 — noise) and plays everyone else worse than
  v1: +61 Elo vs random against v1's +378. Whatever it learned came from
  the self-play and pool games. The anchor has to be graded — `random`
  first, Greedy once the learner wins some of those games — or the run
  has to start from a policy that already does (v1).
- **Argmax and sampled play can disagree a lot.** v2's sampled policy
  beats random 0.76 and v1 0.47; its argmax line manages 0.54 and 0.24.
  v1's sampled play was slightly weaker than its argmax. Eval both,
  always — the ladder reports them separately for this reason.

What the third run taught:

- **The yardstick *is* a teacher, once you can reach it.** Continuing v1
  — a policy that already beat random — against Greedy for 4,000 games
  took the training win rate against it from 0.13 to 0.60, and the
  argmax line to **0.59** in evaluation, above Greedy in every Elo fit.
  Same mix, same anchor, same games as v2; the only difference was
  starting from something that won a share of those games, so the
  terminal reward carried information. v2 and v3 together are the case
  for grading the anchor.
- **The seats swapped.** v1 and v2 lagged as US; v3 beats Greedy 0.70 as
  US and 0.49 as USSR. The early-war USSR game against a heuristic that
  coups every turn is now the weak spot.
- **Games got longer.** Mean final turn 5.0 → 6.9 over the continuation:
  the DEFCON plateau from v1 is behind it.
- **The arena keeps finding rules edges.** 1,300 games in, a learner
  took every European Battleground while Greedy held Europe Scoring and
  hit `Board.score_region`'s refusal to value Europe at Control. The bots
  now treat that tier as the game -- and so does the engine: it had
  required every country in Europe for the win and scored the actual
  Control tier as Domination (a `VERIFY` note since the first version);
  10.1.3 makes the tier an automatic victory, fixed upstream.

What the fourth run taught:

- **More of the same still pays.** Another 4,000 games took the argmax
  line from 0.59 to **0.92** against Greedy and to 0.84 against v3, with
  entropy flat at ~0.27 of ln K and explained variance ~0.8 — no sign of
  the plateau yet. The USSR seat, v3's weak one, is now the stronger.
- **The yardstick is nearly used up.** At 0.92 the 40% anchor games are
  becoming the constant +1 that v2's were a constant −1; the pool (PFSP
  over 78 snapshots) is where the signal is. Progress is now measured
  against the earlier versions — which is what the baseline chain was
  for.

What the fifth run taught:

- **The thesis holds.** Nothing but self-play and a pool of its own past
  selves, 8,000 games from scratch: the argmax line beats Greedy 0.66
  and ties v3, which had the same games with half of them against
  anchors. A scheduled anchor run (`random` → `greedy`, same budget)
  came out *weaker* than the pure one (0.56 vs Greedy, 0.31 against v5
  head to head). Anchors were scaffolding; the pool is the curriculum.
- **Games, not opponents, are the currency.** v3 (8k) → v4 (12k) and
  v5 (8k, no anchor) line up on one curve: strength tracks the number
  of games played, whoever they were against. Throughput is the road
  map.

What the loop taught (v6–v8):

- **It runs itself.** Three generations of `wopr.loop` from v5 — train
  4,000 games, gate against the champion on every eval seed, freeze —
  promoted three times unattended: 0.84 vs v5, 0.70 vs v6, 0.73 vs v7,
  each generation about 18 minutes of training and 3 of evaluation.
  v8, 20,000 games of nothing but self-play and its own pool, beats
  Greedy 0.96 and v4 — the Greedy-anchored line's best — 0.88.
- **Elo against `random` has stopped measuring.** Every version from v6
  on wins 99%+ of those games, so the fitted scale flattens at ~+1400;
  the chain itself (each version against every earlier one) is the
  yardstick now, as the baseline protocol intended. Rate new versions
  by their win rate against the previous two or three.
- **Generations get longer as play improves.** Games now run ~340
  learner decisions (v5: ~240), so 4,000 games are 176 updates rather
  than 100 — the loop's cost is in decisions, not games.

What the epochs experiment taught:

- **Half the epochs, none of the strength.** Through the loop from v8,
  one generation each way gated against v8: 2 PPO epochs won the gate
  0.635 (4 epochs: 0.573), beat Greedy 0.995 (0.960), tied the 4-epoch
  arm head to head, and took 28% less wall time. Now the default; the
  update is ~2 s and about even with the rollout again.
- **The USSR seat has become the seat.** Both arms beat v8 overwhelmingly
  as USSR (0.84–0.89) and lost as US (0.26–0.43); head to head the two
  arms split 0.12 / 0.79 by seat. Whoever holds the USSR seat wins. The
  policies have found USSR lines the US defence has not caught up with
  — road-map item 2, in a new form, and the first thing to look at in
  the games themselves.

What the loop taught (v9 → v10, 3 generations):

- **The gate is starting to bite.** Of three generations from v9, one
  cleared it (0.63 vs v9 → v10) and two did not (0.54 vs v9 at worst
  seed 0.495; 0.53 vs v10). 36,000 games in, the chain is flattening
  at this network size — the first plateau since v1's DEFCON one. The
  levers left are the network (hidden 256, a third graph layer), what
  it sees (history, item 6), and search on top (item 5).
- **A per-seat gate would measure nothing new.** In a two-policy
  match-up, "the challenger is better as US" is the same statement as
  "its pooled win rate is above 0.5": challenger-as-US beats
  champion-as-US exactly when challenger-as-US + challenger-as-USSR
  exceeds 1. The pooled rate is already the seat-balanced measure. What
  the 0.28 / 0.88 split shows is the USSR edge between near-equal
  policies — about 75/25 — a property of the game as these policies
  play it, not of the gate.

What the capacity experiment taught (hidden 256, August 2026):

- **Equal games is not equal training.** The A/B as planned — a fresh
  `--hidden 256` run (858k parameters; 128: 272k) to v5's 8,000 games —
  lost to v5 on every seed (0.41 / 0.44 / 0.44) and to Greedy (0.32–0.34
  against v5's 0.62–0.69). But v5 trained at 4 PPO epochs and the run at
  the 2-epoch default, and the same-recipe control — hidden 128, 2
  epochs, 8,000 games — loses to v5 0.06 / 0.06 / 0.08 and ends far less
  converged (entropy ratio 0.53, explained variance 0.71; v5: 0.30 /
  0.82). The epochs A/B was a continuation from a trained policy; from
  scratch, 2 epochs over 8,000 games is half the optimisation. The
  control, not v5, is what the arm has to be compared with — and a
  fresh run wants 4 epochs.
- **Width is a large win at the same recipe.** 256 beats the 128 control
  0.81 / 0.86 / 0.86, and its seat pattern is new: against v5 it wins as
  US (0.58–0.64) and loses as USSR (0.23–0.29), the mirror of everything
  since v8.
- **What it costs, like for like** (8 workers, 16 threads, bf16, 2
  epochs): update 1.7 → 4.0 s (2.3×, under the hidden² guess of ~4 —
  the option head, the card embeddings and the fixed overheads do not
  scale), rollout 2.2 → 3.1 s (1.4×, the learner's own forward pass);
  8,000 games in 31.5 min against 15.6. Neither 2-epoch run is frozen.
- **The matched arm settles it.** Hidden 256 at 4 epochs — v5's recipe
  and budget, the update at 8 s — beats v5 0.84 / 0.855 / 0.845, Greedy
  0.91 and the 2-epoch arm 0.83–0.875, and takes 0.145 off v10 where v5
  took 0.035. Frozen as v11, it sits between v6 and v7 on the 128 chain
  (0.58 vs v6, 0.32 vs v7): 8,000 games of the wider network are worth
  12–16,000 of the narrower one, at about twice the wall time per game.
  Its games are shorter (mean final turn 5.7 against the 2-epoch arm's
  7.2) — more DEFCON endings, and they win.

- **Through the loop it climbs where 128 had flattened.** Three
  generations from v11 (4 epochs, 4,000 games each, ~35 min training
  and 90 s of gate per generation): v12 at 12,000 games (0.79 vs v11),
  a miss at 16,000 (worst seed 0.547 against the 0.55 gate), v13 at
  20,000 (0.625 vs v12). On the 128 chain, v12 is level with v8 (0.50;
  20,000 games of 128) and v13 beats v8 0.65, v9 0.57 and takes 0.40
  off v10 — 20,000 games of 256 are worth about 30,000 of 128. The gap
  to v10 is the US seat alone (0.07 as US, 0.73 as USSR).

- **Then it flattens — at the same place.** Three more generations:
  v14 at 24,000 games (0.571 vs v13, worst seed 0.550 against the 0.55
  gate), a miss at 28,000 (0.489), v15 at 32,000 (0.569 vs v14). Against
  v10 the line stops: 0.40 (v13) → 0.47 (v14) → 0.47 (v15; per seed
  0.46–0.49), Elo +1466 → +1461. At 32,000 games the wider network is
  level with what the narrower one was at 32,000 games, clearing its
  own gates by a hair, as v10 did. Two widths, the same ceiling, and
  the same shape to it: v15 against v10 is 0.17 as US and 0.78 as
  USSR. Width bought speed along the chain, not height; the ceiling is
  not the network's capacity.

What the handicap experiment taught (the US seat, August 2026):

- **The USSR edge between equals is three games in four.** A policy
  against itself (`wopr.eval a=x.pt b=x.pt`): v15 wins 0.78 as USSR,
  v10 0.75. That is the number the gate cannot see.
- **A VP bid does not even the seats.** `Engine.new_game(starting_vp)`
  and `--handicap` open the game with the US ahead, as a tournament bid
  does. v15 against itself at bids of 2 / 4 / 6 / 8 VP: USSR 0.76 /
  0.77 / 0.74 / 0.68. Eight VP buys one turn.
- **Because the edge is region scoring in the first four turns.** In
  160 v15-vs-v15 games, 109 of 128 USSR wins were by reaching 20 VP,
  mean turn 6.4; the track reads −1.9 / −4.5 / −7.1 at the start of
  turns 2–4. By card, net VP to the USSR against the US: Asia Scoring
  1,154 vs 179, Middle East 632 vs 296, Europe 552 vs 73, Southeast
  Asia 300 vs 22 — and most of it scored on the US's own action rounds:
  the US holds the scoring card and must play it into a region the USSR
  already dominates. (Scoring is net, one number per card, onto one
  tug-of-war track — checked.) The US seat's problem is board position
  in turns 1–3, which a bid pays for but does not fix.
- **Training at the bid, gated at the printed game**, from v15 at 8 VP:
  one generation cleared the gate 0.606 (US 0.365) and became v16, the
  first version level with v10 (0.508; US 0.23) — and v16's own USSR
  edge is 0.75, unchanged. The next generation missed (0.489, US 0.18).
  That is the plain loop's pattern (0.571 / 0.489 / 0.569), not a new
  one. Closed: the US rows' reward still says which seat they were.

### Reading v9's games

Forty v9-vs-v9 games, argmax play on forty decks (`runner.play_game`
logs, aggregated by action kind):

- **The USSR's edge is the early war, and it is the real one.** USSR
  wins 25 of 40. In those games the USSR leads from turn 1 and jumps to
  −9.6 VP by the end of turn 3, then holds. In the US's 15 wins the US
  is level early, survives turns 3–5 at about −3, and wins the *late*
  war: +14 on final scoring. Twenty-one games go the full ten turns;
  DEFCON 1 ended three. The VP game is mature; the asymmetry is when
  it is decided.
- **Turn 1 is a coup blitz.** The USSR coups 63 times in turn 1 across
  the forty games (the US 42), on Iran above all, then Brazil, Angola,
  Argentina, Venezuela, Libya, South Africa, Nigeria — the Battlegrounds
  DEFCON 5 leaves open — and settles to ~20 coups a turn against the
  US's ~10. Its headlines are VP: Nuclear Test Ban (+3 at DEFCON 5, in
  39 of 40 games), Red Scare/Purge, the Indo-Pakistani and Arab-Israeli
  wars, OPEC, Cuban Missile Crisis. The US headlines Olympic Games,
  East European Unrest, Duck and Cover.
- **The rest looks like Twilight Struggle.** ~10% of cards played for
  the event, ~4% to the Space Race; the USSR spreads through Poland,
  Pakistan, Brazil, Argentina, India, Venezuela, the US through France,
  Brazil, Venezuela, Libya, Italy, Egypt; Asia, Europe and Middle East
  Scoring are the cards that matter, for both.

The weak spot is therefore narrow: the US's turns 1–3 against the
coup-and-VP blitz, not US play in general. That is what a seat-weighted
sample or a per-seat gate should be aimed at — and, before either, what
a US-seat value trace of a lost game should show.

### Where the time went (August 2026)

Two profiles, taken before building anything else on the road map,
reordered it:

- **The engine was not engine-bound, it was `deepcopy`-bound.** 78% of a
  Random-vs-Random game was `copy.deepcopy` inside `observe()`: three
  generic, memoised copies of small flat dicts per decision. An explicit
  two-level copy (`Board.snapshot_influence`) is the same snapshot;
  engine-only throughput went from ~6k to ~15k decisions/s.
- **Greedy recounted the map for every option.** `board_value` walked
  all 85 countries and twelve region tiers per candidate — 93% of its
  time, thirty times the engine's. The value depends only on who
  Controls what, so a one-country change is now scored from that
  country's own terms (`_swing`), exactly zero when Control does not
  flip. Same choices on 3,133 recorded decisions; Greedy-vs-Greedy went
  from 0.46 to 10 games/s (22×), and `--anchor greedy` trains at ~1.3k
  learner decisions/s.
- **What a learner step costs now** (self-play, 64 envs): the policy
  forward pass is ~40% of it at 8+ threads (7 ms for a batch of 64 — a
  graph network over 85 nodes is ~0.8 GFLOP per batch), the rest engine
  `step` and feature encoding. Rollouts run at ~2.7k learner decisions/s
  against the random anchor, ~1.3k against Greedy, ~1.75k with pool
  snapshots in play: a net opponent is asked ~8 times per learner step
  at a mean batch of 8, because the lockstep rounds tail off with a few
  slots still waiting on it.
- **Training was update-bound all along.** v1's own `metrics.csv` says
  62% of its 26 minutes were PPO updates — 10.7 s per update against
  ~6.5 s of rollout — and the engine work above only touched the
  smaller share. The update is 32 forward+backward passes at batch
  1,024; scoring the option head on legal `(row, slot)` pairs instead of
  all 96 padded slots, and a plain `matmul` for the adjacency
  aggregation, took it to 6.6 s with bit-identical outputs (the v1
  checkpoint included). bf16 autocast halves it again, to 3.4 s: it
  changes the numerics, so it was checked first — 600 games against
  v1's first 600 track within noise on every health metric — and is
  now the default. An update that cost 17 s in v1 costs 7 — and 5.4
  once the rollout is collected by eight processes over shared memory
  (3.7 s → 1.7 s), after which the update is two thirds of the loop.

## Open questions and the road ahead

**Two experiments closed, one ceiling.** Capacity (above): a
`--hidden 256` network climbs the chain faster — 20,000 games worth
30,000 of the old — and flattens at 32,000 games where the 128 line
did, level with v10. The US seat (above): the USSR wins three games in
four between equals because it dominates Asia, the Middle East and
Europe by turn 3 and the US scores those regions for it; a VP bid in
training does not move that edge. The lines are `runs/pure` (v5–v10,
128) and `runs/h256-e4` (v11–v16, 256; its run directory is 4,000
bid-trained games past v16, unpromoted). What the ceiling is made of
is the question; capacity and the seat's VP count are the two
explanations ruled out. Left, in the order the games suggest:

1. **The US seat's signal.** Its rows end −1 four times in five, and a
   terminal ±1 says nothing about *how* the game was lost: a US that
   holds the USSR to −8 by turn 10 learns the same as one that folds
   at turn 5. A terminal reward that carries the margin (the final VP,
   squashed — still zero-sum, still terminal) gives the losing seat a
   gradient; the "shaping teaches the shaping" objection applies to
   dense rewards, not to the final score. The cheapest experiment
   left, and the one aimed at what the games show.
2. **What the network sees** (item 6): order and recency — the
   "discarded this turn" location first. The US's turn-1–3 problem is
   partly a guessing problem about which scoring cards are out.
3. **A third graph layer** — the least likely lever, given capacity.

A note on epochs: 2 is the default because it matched 4 when
continuing a trained run; a fresh run wants 4, and the 256 line was
continued at 4 to keep one variable at a time — the continuation A/B
at 256 has not been run.

In rough order of expected value per effort:

1. **Throughput.** Collection is parallel (`--workers 8`, the
   shared-memory backend in WOPR.md) and the update runs 2 epochs in
   bf16: ~2.0 s update, ~2.5 s rollout per update, a generation of
   4,000 games in ~14 minutes. What is left is the learner's own
   forward pass on the rollout side (linear in rows) and the network's
   FLOPs on the update side — a smaller network or a GPU — and, as
   games get longer, the fact that the loop's cost is in decisions, not
   games.
2. **The US seat.** Back, in a new form: from v8 on the USSR seat wins
   most games between strong policies, and the games say why (above):
   the early-war coup-and-VP blitz, which the US survives in its wins
   and loses to in turns 1–3 otherwise. A per-seat gate adds nothing
   (above: the pooled rate already balances the seats). What would: a
   VP handicap for the USSR seat in training games — what tournament
   play does with a bid — so games are decided by play rather than by
   seat, and the US perspective's rows carry more than "you drew the
   short side"; and, for evaluation, reporting the USSR edge between
   equal policies as a number of its own.
3. **The self-improvement loop** (`wopr.loop`, built). v5 settled the
   recipe — self-play and the pool, no anchors — and the loop runs it:
   train the challenger for N games, evaluate against the champion on
   every eval seed and against Greedy, promote past the gate, freeze,
   repeat. The knobs are N, the gate, and the training hyperparameters
   passed through to it — the update phase being two thirds of the
   loop, `--n-epochs 2` is the first experiment to run through it.
4. **A shared-memory or rewritten-engine backend.** Not mandated yet. A
   rewritten engine removes at most the engine's third of a learner step
   (~1.5× at best) while encoding and inference stay where they are;
   multi-process collection scales all three. Revisit once collectors
   are multi-process and the per-process engine+encoding share is what
   remains — the layout is the contract, so the swap stays local.
5. **Search on top.** `serialize()/deserialize()` make one-ply expectimax
   over the learned value cheap — with the caveat that a cloned engine
   carries its RNG state, so determinised search must re-seed the clone.
6. **History — mostly already there.** `Observation` carries the discard
   and removed piles in full and the layout encodes every card's
   location, so *what* has happened — events fired, scoring cards
   played since the reshuffle, the opponent's Ops plays this turn — is
   in the state. What is not is *order and recency*: a region scored on
   turn 2 and one scored this turn look the same, and a sequence of
   opponent plays says things about the hidden hand that the set of
   discards does not. A "discarded this turn" card location is the
   cheap half; sequence features are the other. Behind capacity and
   search in expected value.

## Reproducing

```sh
uv sync --extra wopr
uv run python -m wopr.train --run first --games 4000 --n-envs 64 \
    --self-play 0.3 --vs-pool 0.3 --anchor random --snapshot-every 5
uv run python -m wopr.eval --games 200 first=runs/first/joshua.pt random greedy
uv run python -m wopr.baseline v2 --run first      # freeze it under baselines/
uv run python src/main.py --ussr joshua --joshua-checkpoint runs/first/joshua.pt
```
