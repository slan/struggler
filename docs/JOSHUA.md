# Joshua: the lab notebook

The idea — a learned `Player` trained by self-play in the WOPR arena,
why this engine makes it tractable, the three design ideas (a
decision-centric arena, one head scoring every option,
alternating-perspective GAE) — is unchanged and written up once, in the
archived notebook of rules version 1:
[archive/JOSHUA-r1.md](archive/JOSHUA-r1.md). The mechanics are the
contract, [WOPR.md](WOPR.md), including the training process end to end
and the decision points along it. This file is the live notebook: what
is being measured on the current engine, and what was decided on it.

## Rules version 1 → 2: what carried over

On 2026-08-22 nine engine fixes landed at once (`b55daf5`) and the rules
version went from 1 to 2. One of them — a scoring card held past the end
of the turn loses the game — retired every policy trained before it:
r1's champion fell from 0.98 to 0.55 against Greedy, and Greedy itself
lost two games in three to the same rule until it was taught to play its
scoring cards (`cc04ffa`). Every number in the r1 notebook was measured
on a game in which holding a scoring card was free; its sixteen
baselines are archived under `baselines/r1/` and its ladder is closed.

What r1 established that does not depend on that rule — the findings
about *training*, not about the game:

- **The pool is enough; anchors are not teachers.** Fifty percent
  self-play, fifty percent PFSP-sampled past snapshots, no fixed
  opponent: a fixed opponent the learner always beats or never beats is
  a constant reward (r1 v2, v4, the `sched` control).
- **Two PPO epochs match four when continuing a trained run, not from
  scratch.** A fresh run at 2 epochs is half-trained at 8,000 games (r1's
  capacity control lost 13 games in 14 to the same run at 4).
- **Width buys speed along the chain, not height.** Hidden 256 reached in
  20,000 games what 128 reached in 30,000, then flattened at the same
  place.
- **The final score in the terminal reward teaches the blitz.** A margin
  term made both seats play for the track; games shortened, strength
  fell. Terminal ±1 stays.
- **A VP bid does not move the seat.** Opening the game with the US ahead
  bought one turn; the USSR edge came from region scoring in turns 1–4.
- **The loop's gate works as a selector**, and Elo over a ladder stops
  measuring once every version beats the anchors — rate against the
  previous versions.
- **The tooling**: shared-memory collectors, bf16, the legal-rows option
  head, `--recipe`, `--init`, `wopr.ab` and the ledger, `wopr.diagnose`.

What r1 did **not** establish, because the game it measured is gone:

- The USSR edge between equal policies (0.78) and its mechanism
  (Asia / Middle East / Europe scoring, "mostly on the US's own forced
  scoring plays"). A scoring card could be held then; now it cannot.
  To be re-measured with `wopr.diagnose` on the first r2 versions.
- The plateau at ~v10 strength, and therefore everything ranked behind
  it on the road map (recency features, search). A plateau on a game
  with a free hold is not evidence about this game.
- Greedy's rating as a yardstick. It plays a different game now.

## Rules version 2

*(entries appended per experiment; each is question → setup → result →
decision, with the ledger row it corresponds to)*

### 2026-08-22 — the first clean run

**Question.** What does the recipe learn on the fixed engine, and how
fast does it learn the held-scoring-card rule?

**Setup.** `wopr.ab --run engine-fixes`, recipe v11 (hidden 256, 4
epochs, 50/50 self-play and pool), 8,000 games, no control — the first
run of the ladder. Greedy as the yardstick, fixed the same day.

**Result.** Frozen as `r2/v1`. It learned the rule in ~200 games (mean
final turn 1.5 → 5.6 by 600 games; 6 of 120 self-play games still end
on a held scoring card at 8,000) and beats the fixed Greedy 0.748
[worst seed 0.740] (US 0.74 / USSR 0.76); Elo +1253 ± 30 vs random.
`wopr.diagnose`: games run to turn 7.5 on average; endings split USSR
by VP 29 / US by VP 27 / US by DEFCON 19 / US by final scoring 18 /
USSR by final scoring 13 / USSR by DEFCON 7; the VP track stays within
±2.5 all game. **The USSR edge is 0.44–0.52 — even.** Asia Scoring now
nets to the US (937 vs 154 over 120 games), Europe to the USSR (625 vs
51). Informally, it beats r1/v11 0.585 and r1/v16 0.685 on this engine.

**Decision.** The r1 seat edge (0.78) and its mechanism were the
held-scoring-card freedom, not the game: on this engine the seats are
even between equals at 8,000 games, so road-map item "the US seat" is
closed without an experiment, and the first r2 question is simply how
far the recipe climbs. Into the loop from v1 until the gate misses
twice in three. Ledger: row `engine-fixes`.

### 2026-08-22 — the loop from r2/v1, and the seat edge returns

**Question.** How far does the recipe climb on this engine, and does
the seat edge stay even as the policies strengthen?

**Setup.** `wopr.loop --run engine-fixes --champion v1`, recipe v11,
4,000 games a generation, gate 0.55 on the worst seed. `wopr.diagnose`
on each promoted version.

**Result.** v2 at 12,000 games (0.837 vs v1, worst seed 0.825), v3 at
16,000 (0.651 vs v2, worst seed 0.642), then 0.551 at 20,000 against v3
with the worst seed at 0.540 — one miss. Against Greedy 0.90 → 0.95 →
0.965. **The seat edge is back with strength:** v3 against itself wins
0.695 as USSR (v1: 0.44–0.52), 90 of its 139 USSR wins by reaching 20
VP at a mean turn of 6.1, the track at −5.2 by turn 4; Europe Scoring
is now the USSR's biggest card (1,057 vs 234), Asia has swung back
toward the USSR (825 vs 567 — v1: 154 vs 937). Holding a scoring card
still ends 10% of games (15 USSR, 5 US — turn 3, the early hand).

**Decision.** The r1 seat edge was *partly* the bug: at equal strength
the seats start even and diverge as the policies learn the USSR's
early-war scoring, now through Europe rather than Asia. That is the
game, not the rule. Stopped after one miss to decide the next
experiment with the diagnosis in hand rather than spend generations
first (the batch gated against v3 was started and cancelled; nothing
from it is kept). Open: whether a plateau comes where r1's did, and
what to do about a seat edge that is the game's own.

### 2026-08-23 — r2 closed by the engine

While v3 waited for its diagnosis, the Playdek work ([WOPR.md](WOPR.md),
"Playdek's AI as an opponent") found and fixed thirty-two more engine
differences against the official game — the rules version went to 3
on 2026-08-22 and a dozen of the fixes landed after the bump, the last
on 2026-08-23 (`0a2639f`). None is as large as the held-scoring-card
rule, but together they touch what the policies learned (We Will Bury
You's timing, a trapped seat's scoring card, the Military Ops penalty
before the held card, Five-Year Plan's discard, the extra action round
that may be passed, ...), and the count alone is past the "re-rate the
yardsticks" test of the decision points. **The r2 ladder is archived at
v3** (`baselines/r2/README.md`); its checkpoints load on the r3 engine
(the layout is unchanged) but are not rated on it. The two open
questions — the plateau, the seat edge — carry over as questions, not
as numbers. The r1 findings about *training* (above) still stand; the
r2 finding that the seat edge grows with strength through early-war
scoring is the one game finding worth re-measuring first.

## Rules version 3

*(entries appended per experiment; each is question → setup → result →
decision, with the ledger row it corresponds to)*

### 2026-08-23 — the bootstrap (pre-registered)

**Question.** What does the recipe reach against Greedy on the r3
engine when the run is stopped by the yardstick rather than by a
budget, and how many games does it take? Two things change from r2/v1:
the engine (thirty-two fixes), and the stop rule (`wopr.bootstrap`, new
for r3 — r2/v1 was 8,000 games by fiat, 0.748 against Greedy).

**Setup.** `wopr.bootstrap --run r3 --workers 8 --torch-threads 16`:
recipe v11 from scratch, an evaluation against Greedy every 500 games
(200 games, argmax, 100 a seat, rotating decks) played on the
collectors during the update. Stop rule, written before the run:
rolling mean over the last two evaluations ≥ 0.75 on **both** seats →
a 600-game confirmation (300 a seat) that must also clear 0.75 on both
seats; else plateau (no new best of the overall rolling mean for four
evaluations) or the cap (20,000 games). Whichever fires, the last
evaluated checkpoint is frozen as `r3/v1` with the full protocol.
Reading: *confirmed* → the yardstick is met, the loop starts from
`v1`; *plateau* → diagnose before the loop; *cap* → continue the run.
Reference points: r2/v1 reached 0.748 at 8,000 games, r2/v2 0.901 at
12,000 (US 0.84 / USSR 0.96), so a confirmed stop is expected between
8,000 and 12,000 games. Then `wopr.diagnose r3/v1 --vs greedy` for the
seat edge and the endings of this game, a ledger row
(`wopr.ab --existing`), and the first evaluation against Playdek's
easy AI, 60 games a seat.

*Amended mid-run, 2026-08-23, at 6,521 games.* The plateau signal as
first written was the **weaker seat's** rolling mean (100 games a seat
per tick). It fired at tick 13: the US seat had spiked to 0.47 at tick
9 and read 0.28 / 0.33 / 0.425 / 0.42 after, while the USSR seat went
0.45 → 0.65 and the overall curve 0.37 → 0.49 → 0.48 → 0.475 — a run
still climbing, declared flat by one seat's noise against one seat's
spike. The premature `v1` was discarded before its protocol finished,
the plateau signal changed to the overall rolling mean (400 games,
half the noise; the confirmation trigger stays per seat), and the same
run resumed from its `ppo.zip` with its thirteen ticks kept. Recorded
here because it is a change to a pre-registered rule: the reading of
the stop is unchanged, the estimator behind one of its three branches
is not the one written down above.

**Result.** **Confirmed at 14,020 games**, frozen as `r3/v1` (89 min
of training in two segments; the 28 evaluations cost 36 s of waiting in
all — the collectors finished each one inside the PPO update). The
curve against Greedy (200 games a tick): 0.05 at 500 games, 0.19 at
1,500, 0.42 at 4,000, 0.60 at 8,000, 0.73 at 12,000, 0.875 at 14,000;
the rolling mean crossed 0.75 on both seats at tick 28 (US 0.777 /
USSR 0.925) and the confirmation held it: 0.876 over 600 (US 0.813 /
USSR 0.938). Protocol: vs Greedy **0.865** [worst seed 0.855] (US 0.79
/ USSR 0.94), sampled 0.818; Elo +823 ± 122 vs random. The seats
separated early and stayed apart: the USSR seat reached 0.65 at 5,500
games and 0.91 at 11,000 while the US seat sat at 0.3–0.5 until 8,500
and was the last 3,000 games' whole story (0.48 → 0.81). Against r2's
reference points the climb is slower — r2/v1 was 0.748 at 8,000, r2/v2
0.901 at 12,000 — on a run that is otherwise the same recipe.
`wopr.diagnose`: against itself the **USSR edge is 0.667** (0.625 on
the ledger's seed), games run to turn 7.2, endings USSR by VP 49 /
US by DEFCON 20 / USSR by DEFCON 14 / USSR by final scoring 11 / US by
VP 9 / US by final scoring 8 / held scoring card 7 (turns 2–4); the
track is at −3.8 by turn 4 and −5.9 by turn 6; Europe Scoring is the
USSR's card (569 vs 107), Middle East the US's (164 vs 486), Asia and
Africa the USSR's. Against Greedy it wins 0.925 (seed 7) at a mean
final turn of 5.2, mostly by VP as USSR (52) and by DEFCON as US (37).

**Decision.** The yardstick is met at the first version, so the loop
starts from `v1` without a diagnosis step in between. Two things are
carried as findings, not as numbers to act on yet: the seat edge at
equal strength is the game's (0.67 at 14,000 games, where r2/v1 had
measured even at 8,000 and r2/v3 0.695 at 16,000 — the USSR's early
scoring through Europe, as on r2), and the recipe reaches the same
place more slowly on this engine (14,000 games to a confirmed 0.75
against a fixed Greedy, versus 8,000 for a single-eval 0.748 on r2).
DEFCON endings are 28% of self-play games (34 of 120, both seats) —
higher than r2/v1's 22% — and worth a look when the loop's first
generation is diagnosed. Next: the Playdek easy AI, 60 games a seat,
the first number against an opponent that is not ours. Ledger: row
`r3`.

### 2026-08-23 — r3/v1 against Playdek's easy AI

**Question.** The first number against an opponent that is not ours:
what does 0.865 against Greedy buy against the official game's weakest
AI, seat by seat?

**Setup.** `wopr.playdek.eval --difficulty easy --games 60 --policy
joshua=baselines/r3/v1/joshua.pt --side ussr|us` (argmax, seeds
300–359, 6 games in parallel; `runs/playdek/r3v1-easy-{ussr,us,us-b}`
— the US seat in two batches, 39 + 21, after the first was killed by
hand at 39). Desyncs are the bridge's, not the game's, and are
triaged apart.

**Result.** **USSR seat: 1 win in 55 decided games (0.018, 95% upper
bound 0.10). US seat: 0 in 56 (upper bound 0.06).** The games are
short — 20 VP by turn 4.4 as USSR (36 of 54 losses), by turn 4.1 as US
(50 of 56) — and the VP has one address: **Asia Scoring, 407 : 10 for
the AI with Joshua as USSR and 365 : 3 with Joshua as US.** Europe
goes the AI's way too from either seat (208 : 81 and 180 : 11, with
four Europe-control wins against the US seat); the Middle East is the
only region Joshua's USSR holds (259 : 49). The other loss is a trick:
**14 of the 18 DEFCON losses as USSR are Joshua playing CIA Created
(10) or Grain Sales to Soviets (4) for Ops at DEFCON 2** — the event
hands the AI Ops inside the USSR's own round, the AI coups a
battleground, DEFCON reaches 1 and the phasing player loses. Against
its own pool neither happens: the pool never fights for Asia the way
the AI does, and never takes the free coup. Bridge side: zero
unexplained desyncs that were not the bridge's — the USSR seat's four
were a trapped seat's held scoring card (a DLL rules difference, now
`known`/void) and three Grain Sales inferences, the US seat's three a
trapped AI's scoring card the operator did not infer, a simulation
gap fixed the same day (seed 332) and an early placement-legality
difference (seed 357); `docs/WOPR.md` has the list.

**Decision.** The Greedy yardstick measures something the official
AI does not share. A policy at 0.87 against Greedy and 0.98 against
random is at 0.01 against the weakest opponent that plays a
different game, and the two failure modes are concrete: a region it
never learned to contest (Asia), and a two-card blunder (CIA Created
/ Grain Sales at DEFCON 2) that self-play never punished because the
pool never exploited it. Both are what a pool of one's own past
selves cannot teach — the first road-map question for r3 is therefore
**opponent diversity**, not capacity or search: whether Greedy in the
mix (anchors are off since r1 v5, for a reason that was about reward
signal, not about coverage), an exploiter trained against the
champion, or the easy AI's own games as a curriculum closes either
gap, measured on this eval. The loop from v1 is not the next step
until that is decided; `wopr.diagnose` against Greedy (0.925 at a
mean final turn of 5.2, 37 US wins by DEFCON) already hinted that
its wins are short and by the track, not by the map. The Playdek
easy eval, 60 a seat, is now a standing yardstick beside Greedy.

### 2026-08-24 — the loop from v1 to the r3 plateau

**Question.** How far does the recipe climb on the r3 engine before the
gate misses twice in three — and does the Greedy curve (now free, in
`metrics.csv`) saturate before the ladder does?

**Setup.** `wopr.loop --run r3 --champion v1 --generations 20` (recipe
v11, 4,000 games a generation, gate 0.55 on the worst seed, the new
`--plateau-misses` stop), `--eval-every 500` for the Greedy curve.
The user chose the loop over the opponent-diversity experiment first;
the Playdek easy eval of the final champion follows it.

**Result.** **Seven promotions in seven generations, then the plateau:
v2 … v8, and generations 8 and 9 missed against v8** (0.497 [0.472],
0.510 [0.492]) — the r3 line saturates at v8, 42,020 games. Every gate
had the same shape, the challenger winning as USSR and losing as US
(vs-champion US 0.34–0.49 / USSR 0.74–0.89); the US side of that split
improved almost monotonically (0.39 → 0.35 → 0.34 → 0.34 → 0.40 →
0.45 → 0.49) while the USSR side fell, the two converging toward the
draw-line that stopped the loop. Greedy saturated first: 0.855 at v2,
0.97–0.98 from v5 on (the curve's ticks flatten at 0.94–0.99 past
28,000 games) — from v5 to v8 the ladder kept selecting real
improvement (each gate cleared on the worst seed) that Greedy can no
longer see. v8: Elo +1185 ± 35 vs random, 0.967 vs Greedy on the
protocol. `wopr.diagnose` v8: USSR edge 0.642; **Asia is contested
now** (264 : 489 for the US against itself — v1 never fought there),
Central/South America and Africa are the USSR's; the track stays
within ±3 to turn 8, games at turn 7.6; held-scoring-card endings are
back up (19 of 120, turns 3–6); against Greedy 0.967 with 35 USSR wins
by Europe control at a mean final turn of 4.2.

**Decision.** The Playdek easy eval of v8 (60 a seat, seeds 300–359):
**USSR 5 of 50 decided (0.100), US 5 of 52 (0.096)** — from v1's 0.018
and 0.000. The transfer is real but an order of magnitude smaller than
the internal gains: seven promotions and Greedy from 0.87 to 0.97
bought 0.09 against the easy AI. What remains is concrete: as USSR,
**30 of 45 losses are now DEFCON** — the CIA Created / Grain Sales
Ops-at-DEFCON-2 gift dominates once the VP blowouts shrink (14, from
36) — and as US the 20-VP blowouts stay the story (34 of 47). The
wins are the blitz (Europe control, final scoring). Self-play alone
moved the map skills (Asia is contested now) but cannot unlearn the
event gift its pool never punishes: **opponent diversity is confirmed
as the next experiment**, with the DEFCON-gift rate (14/55 → ?) and
the easy-AI win rate as its metrics, per the road map. Also measured:
v8 stresses the bridge harder than v1 — 13 desyncs in 120 games
(late-war realignment targets, Ask Not, Wargames' end-game prompt,
Junta, a quagmire option list), against 7 for v1 — so a bridge triage
session precedes any heavier use of this eval. Ledger: the loop's
`loop.csv`; `baselines/r3/README.md` v2–v8.

### 2026-08-24 — the tournament bid: bootstrap at US +2 (pre-registered)

**Question.** Tournament play balances the seats with the influence bid
(rule 11.1.4, ~+2 US at tournament level). Does training on the bid
game remove the USSR edge (0.64–0.67 between r3 equals), and what does
an even game do to the bootstrap — faster to the yardstick, a stronger
US seat, fewer of the USSR's early-war blitz wins?

**Setup.** `Engine.new_game(us_bid=N)` implements the rule as written
(placed after regular setup, only where the US already is, capped at
control + 2); `--bid 2` runs through every wopr tool, and the ladder is
**`baselines/r3-bid2/`** — the bid changes the game, so ratings do not
cross it. `wopr.bootstrap --run r3-bid2 --bid 2 --workers 8
--torch-threads 16`: same recipe and stop rule as r3's bootstrap
(rolling mean ≥ 0.75 both seats vs Greedy over 2 ticks, confirmed over
600; plateau on the overall rolling mean; cap 20,000). Note Greedy
itself now plays under the bid, so the yardstick is the bid game's
Greedy. Reading: the per-seat curves and `wopr.diagnose --bid 2`'s
USSR edge against r3/v1's (0.667); games-to-stop against 14,020. Then
the loop on the bid ladder to its plateau, then the Playdek easy eval
(bid 0 on the DLL side until the bridge learns the DLL's own BID
setup; noted as a caveat when comparing).

**Result.** **Confirmed at 11,024 games — 21% sooner than the printed
game's 14,020 — and the seats climbed together**: the curve reached
0.38 at 4,000 games (US 0.47 / USSR 0.29), 0.54 at 7,000, 0.78 at
10,500, with the per-seat gap oscillating around zero all run where
r3's held at 0.25–0.30 for 10,000 games; the stop's rolling means were
US 0.825 / USSR 0.785 and the confirmation 0.807 over 600 (US 0.833 /
USSR 0.780). Protocol: vs Greedy 0.820 (US 0.858 / USSR 0.782), Elo
+1132 ± 268. `wopr.diagnose --bid 2`: **the USSR edge is 0.500** —
exactly even, from 0.667 at bid 0 — with the Middle East the USSR's
region (492 : 138) and Asia the US's (282 : 196). Two things to watch:
DEFCON-1 endings are 52% of self-play games (62 of 120, both
directions; r3/v1 had 28%) at a mean final turn of 5.6 — the even game
is being fought with coups — and the Elo spread across seeds tripled
(± 268), both worth re-reading at the loop's first diagnosis.

**Decision.** The tournament bid does on this engine what tournament
players use it for: US +2 removes the seat edge between equals, and
the recipe reaches the yardstick faster on the even game. The bid
ladder is the live one — the loop runs on it from `r3-bid2/v1` to its
plateau (same gate, `--bid 2` throughout), then the Playdek easy eval
of the champion (bid 0 on the DLL side until the bridge speaks its BID
setup; noted when comparing). The printed-game ladder r3 stands at v8
for reference. Ledger: `r3-bid2` in `baselines/r3-bid2/README.md`.

### 2026-08-24 — the bid ladder's loop, and its champion against the easy AI

**Question.** How far does the bid ladder climb, and what does its
champion do against Playdek's easy AI — which plays the printed game,
not the bid's?

**Setup.** `wopr.loop --run r3-bid2 --champion v1 --bid 2` to the
plateau; then the easy-AI eval of the champion, 60 a seat, seeds
300–359 — with the pre-registered caveat that the DLL plays bid 0, a
different game from the one the champion trained on.

**Result.** The loop promoted twice — v2 at 15,024 games (0.693 vs v1,
worst seed 0.670, US 0.67 / USSR 0.72: the first near-balanced gate of
any ladder), v3 at 19,024 (0.661, worst 0.635) — and plateaued at
generations 3–4 (0.487, 0.537 vs v3): **the bid ladder saturates at
v3, 23,024 games**, vs the printed game's v8 at 42,020, at a champion
of comparable Greedy strength (0.95, Elo +1132's line). `diagnose
--bid 2` on v3: **the USSR edge stays 0.517 at champion strength**
(r3's re-grew to 0.64+); DEFCON endings 41% of self-play games.
Against the easy AI: **USSR 6 of 48 decided (0.125), US 2 of 54
(0.037)** — the USSR seat a shade above v8's 0.10, the US seat below
its 0.096. The asymmetry is the caveat made visible: on the DLL's
bid-0 game the bid-trained USSR faces an easier US than it trained
against, and the bid-trained US is missing the +2 it has always had.
Bridge attrition is now material: 11 of 60 USSR-seat games and 6 of 60
US-seat desynced (Grain Sales' take inference alone six games a seat;
SALT Negotiations, Tear Down This Wall, a forced discard, and two
turn-1 placement mismatches are new), and one outright crash — the
bot's own Grain Sales taking a drawn *scoring* card was unmapped —
killed a batch before being fixed (a crashed game no longer kills the
eval).

**Decision.** Three conclusions, one per layer. *Training*: the bid
game is the better self-play game — even seats at every strength
measured, balanced gates, a faster bootstrap — and `r3-bid2` stays the
live ladder. *Evaluation*: the easy-AI number for a bid-trained policy
is not comparable until the bridge speaks the DLL's own BID setup
(`EChooseSidesMethod.BID`); wiring it is now the top bridge task,
tied with the Grain Sales inference family, ahead of any further
easy-AI evals — at 10–18% attrition the eval is eating its own sample.
*Play*: the DEFCON gift persists under the bid (21 of 42 USSR-seat
losses), so opponent diversity stays the open training question,
unchanged from the r3 decision. Order of work therefore: the bridge
session (BID setup + Grain Sales), re-run this eval at bid 2, then the
opponent-diversity arms on the bid ladder.

### 2026-08-24 — the bridge speaks the bid; the eval unconfounded

**Question.** With the DLL playing the tournament bid itself, what is
r3-bid2/v3's real number against the easy AI?

**Setup.** Probing found the DLL's `GameParameters.additionalInfluence`
— the app's handicap — implements rule 11.1.4 exactly as the engine
does (a US placement step after regular setup, own countries, control
+ 2 cap), so `wopr.playdek.eval --bid N` now sets both boards and no
bidding flow is needed. Two operator fixes rode along: the Grain Sales
take is read off the DLL's card moves before any reveal record (the
stale-reveal mispick of seeds 319/328/333), and Wargames' end-game
prompt maps to the engine's end_game/decline. Hidden-prompt harness
59/59; hotseat emulation 32/32 and 120/120 at bid 2, 32/32 at bid 0.

**Result.** r3-bid2/v3 against the easy AI at bid 2, 60 a seat:
**USSR 5 of 54 (0.093), US 4 of 51 (0.078)**. Against the confounded
bid-0 run: the US seat doubled (0.037 → 0.078) with its +2 restored,
the USSR seat eased (0.125 → 0.093) now that the AI's US starts +2 —
both in the direction the game change predicts, and the seats are
within noise of each other, as in training. Strength against the AI
is still an order below the yardstick numbers. Bridge attrition
halved: 5 desyncs a seat (was 11), Grain Sales down to 2 a seat from
6; the residue is listed in WOPR.md (Grain Sales' remainder, Junta,
Tear Down This Wall, an event-placement mapping at seed 353, two slow
drifts).

**Decision.** The eval is now on the policy's own game and the bridge
is no longer the bottleneck it was; the standing yardsticks for the
bid ladder are Greedy-at-bid-2 and this eval. The gap that remains —
~0.08 against the weakest official AI at 0.95 vs Greedy — is the
policy's, and the loss mix (DEFCON gifts as USSR, 20-VP blowouts as
US) is unchanged in kind: on to the opponent-diversity experiment,
below.

### 2026-08-24 — opponent diversity, arm 1: a Greedy share in the mix (pre-registered)

**Question.** Does sparring against an opponent that is not a past
self close what self-play cannot see — the DEFCON gift, the
uncontested regions — measured against the easy AI? r1 closed
"anchors as teachers" (a fixed opponent the learner always beats or
never beats is a constant reward); this asks a different question,
coverage, at a small share, and may reverse that finding's scope.

**Setup.** `wopr.bootstrap --run r3-bid2-gshare --bid 2 --no-freeze --
--self-play 0.45 --vs-pool 0.45 --anchor greedy` — recipe v11 with a
10% Greedy share, the same stop rule as the r3-bid2 bootstrap, frozen
nothing. Control: r3-bid2/v1 (recipe v11 pure, confirmed at 11,024
games, 0.807). Metrics, decided before the run: (a) games to the
confirmed stop vs 11,024 and the confirmation rate vs 0.807; (b) the
easy-AI bid-2 eval, 60 a seat, of the arm's stop checkpoint against
the same eval of r3-bid2/v1 (measured as the control's baseline); (c)
the CIA Created / Grain Sales DEFCON-gift rate in those games.
Decision rule: the share enters the recipe if (b) improves the
two-seat mean by at least 0.05 without (a) blowing past the 20,000
cap; a wash or a slower learner closes the arm.

**Result.** (a) Confirmed at 11,518 games against the control's 11,024
— a wash — with a stronger confirmation (0.858, US 0.815 / USSR 0.900,
vs 0.807), after a visibly diluted start: half the control's rate to
5,000 games (r1's anchor finding, at one tenth the dose), then caught
and passed from 6,000. (b) Against the easy AI at bid 2, arm vs
control at the same training stage: **USSR 0.019 vs 0.000, US 0.018 vs
0.034** — a wash, nowhere near the +0.05 bar; both stop-checkpoints
are far below v3's 0.093/0.078, so two more loop generations are worth
more against the AI than the mix change. (c) The DEFCON-gift rate is
**unchanged**: 21 of 52 USSR-seat losses (control 18 of 59).

**Decision.** Arm closed. The reason it failed is instructive: Greedy
never takes the gifted coup either, so a Greedy share cannot punish
the habit — coverage comes from opponents that *exploit*, not
opponents that differ. r1's "anchors are not teachers" stands, now
measured at a 10% dose on a game where the anchor is not beaten until
late. On review of the arc ([baselines/RECAP-r3.md](../baselines/RECAP-r3.md))
a constraint was adopted that reshapes the follow-ups this entry first
proposed: the bot stays **self-play only** — no external opponent in
the training mix, so the easy AI as a sparring share is off the menu,
and the exploiter (still the line's own weights, but a step away from
*naive* self-play) is held in reserve rather than tried next. The next
experiment is inference-time search over the learned value head,
pre-registered below.

### 2026-08-25 — search over the learned value head: veto and one-ply (pre-registered)

**Question.** The DEFCON gift is ~40% of USSR-seat losses at every
strength measured, and self-play cannot unlearn it (arm 1: the pool
never punishes it). Does lookahead at *inference*, over the frozen
champion's own value head, remove it — and how much of the remaining
gap against the official AI is that one blunder class versus general
tactical weakness? Training changes nothing: no new bootstrap, no
layout bump, r3-bid2/v3's checkpoint as it stands.

**Setup.** Two inference modes wrapping r3-bid2/v3, one simulation
harness with an evaluator switch (mechanics go to WOPR.md with the
code). Both build the search state **from the `Observation`**, never
from the live engine — unseen cards (draw pile ∪ opponent hand)
shuffled into a guessed arrangement (a determinization) — so nothing
hidden leaks through the simulated transitions (mandate #4 in spirit).

- *Veto* (`evaluator=terminal`, no torch): simulate each legal option
  to its resolution; an option that provably loses — terminally, or
  through an opponent mate-in-one reply computable from public
  information (the granted-Op battleground coup at DEFCON 2 is exactly
  this) — is masked, and the policy's argmax picks among the
  survivors. All options vetoed → the mask is dropped.
- *One-ply search* (`evaluator=value`): the same simulation; each
  option scored by the value head at the next decision reached,
  whoever its mover is, sign-flipped to the root mover (the GAE
  buffer's alternation rule). A branch that consumed no unseen card
  and no roll is exact from one sample; one that did is averaged over
  k≈4–8 determinizations, with a lone die enumerated (6 outcomes at
  1/6) instead of sampled. Terminal leaves score ±1, draws 0 — the
  head's own training scale. Argmax over scores.

The veto is the ablation: search subsumes it (a provable loss
evaluates to −1), so the search−veto gap attributes the general
tactical lift apart from the gift alone.

**Eval, one batch** (this also executes the road map's "measure hard
mode now"): Playdek at bid 2, three players — raw v3, v3+veto,
v3+search — easy 60 a seat each, hard 30 a seat each. In-repo sanity
first, before any DLL time: v3+search vs Greedy and vs raw v3 on the
eval seeds; search losing to raw is a bug, not a result. Gates and
`wopr.baseline` stay raw-argmax throughout — a search player is rated
as its own named policy (`v3+search`), never silently substituted.

*Amended 2026-08-25, during the in-repo sanity runs and before any
Playdek time — three changes to the value evaluator's mechanics (the
veto, the metrics and the decision rules are untouched). The sanity
bar did its job: the first implementation lost to raw v3 0.020 with
games ending by turn 2. (1) One-ply as registered stopped at the
mover's own next atomic decision, which prices an action by its first
step — at `OPS_TYPE` the "coup" branch was scored before any target
was picked, and the search itself played the DEFCON suicide; branches
now roll through the mover's own chain along the policy's argmax.
(2) The leaf at the opponent's decision is evaluated from *their*
observation, which cannot see the mover's hand — the search held
scoring cards to the end of turn; branches now roll on through the
opponent's reply and score the minimum of the flip-point estimate and
a hand-aware estimate at the mover's next own decision, end-of-turn
terminals played out for real. (3) Pure argmax over searched values
replaced trained play with the value head's per-option noise (~0.1)
and still lost most games; the policy's own pick now stands unless
another option's value clears it by a 0.3 margin — a real blunder
differs by ~1.0. Informally after the fixes: 4 of 6 debug games
against raw v3, no self-inflicted terminal. WOPR.md carries the
mechanics.*

*Sanity results, 2026-08-25 (post-amendment code, 200 games each at
bid 2, `wopr.search_eval`): v3+search beats **raw v3 0.625** [0.56,
0.69] (USSR 0.670 / US 0.580) and holds **Greedy at 0.950** (USSR
0.96 / US 0.94), raw's own level. The bar is cleared — the lift over
raw is promotion-sized (the loop's gate freezes at 0.55) from
inference alone. The veto's own sanity (100 games vs raw): **0.636**
[0.54, 0.72] (USSR 0.694 / US 0.580) — indistinguishable from the
full search's 0.625. Against its own kind, the lift is the veto's:
declining provable losses, not the value head's re-ranking. Noted,
not concluded — the head-to-head is exactly the matchup that flatters
the veto (raw's losses are the blunder class); the Playdek arms are
the attribution that counts.*

**Metrics and decision rule** (written before the runs): (a) the
easy-AI two-seat mean vs raw v3's 0.093/0.078 — search becomes the
standing reported player if it improves the mean by ≥ 0.05; (b) the
DEFCON-gift share of USSR-seat losses (raw: ~21 of 42–52) — expected
≈0 for both veto and search; (c) the first hard-mode numbers for all
three, whatever they are — the mountain sized. Readings: search
clears (a) → deepen (beam over own consecutive decisions) and/or move
to scenario-seeded self-play for what search cannot reach; a wash on
(a) with (b) at zero → the remaining losses are strategic, not
tactical — scenario-seeded self-play next, not deeper search; veto ≈
search on every metric → the value head adds nothing over the rules
check, which questions the head before any layout bump spends a
bootstrap on it. Ledger: no row — nothing is trained; this entry is
the pre-registration.

## Road map

Rewritten 2026-08-25 at the close of the bootstrap/bid/bridge arc
([baselines/RECAP-r3.md](../baselines/RECAP-r3.md) is the snapshot).
The standing constraint, adopted on review: **the training is
self-play only** — no external opponent in the mix, Playdek's AI is
an evaluation and never a teacher, and the league exploiter (the
line's own weights, but not *naive* self-play) is a reserve, not a
next step. Superseded along the way: opponent diversity as the lead
question (arm 1 closed it — a Greedy share is a wash and cannot
punish the gift; the easy-AI-as-sparring candidate is banned by the
constraint).

1. **Search over the learned value head** — the pre-registered
   veto/one-ply experiment above. Inference only; no bootstrap, no
   layout change.
2. **Scenario-seeded self-play** — shape the initial-state
   distribution, not the opponent: a fraction of training games starts
   from positions where the failure is on the table (DEFCON 2, a
   gift card in hand; later, prefixes of lost eval games if that
   purity line is acceptable). Both seats are the learner;
   `Engine.serialize`/`deserialize` makes seeding cheap.
3. **Hard mode as a standing eval** — first numbers land in the search
   batch above; thereafter hard sits beside easy for every claim.
   Evaluation only.
4. **In reserve: the league exploiter** — only if search plus seeding
   leave the gift rate standing.
5. **Bridge to <2% attrition** — the open desync families (WOPR.md),
   traces caught by volume; matters more as evals move to hard mode's
   longer games.
6. **Protocol hygiene as the numbers tighten** — more eval seeds per
   claim, Wilson bounds on every reported Playdek rate.
7. **Candidates carried, unranked:** order and recency features (a
   layout bump — mandates a fresh bootstrap, since checkpoints refuse
   to load across `LAYOUT_VERSION`; take it with the next retrain,
   informed by 1–2); a third graph layer.

At a plateau, diagnose before choosing: the next experiment's control,
metric, budget and decision rule are written into its entry before
training starts (WOPR.md, "Decision points").

## Reproducing

```sh
uv sync --extra wopr
uv run python -m wopr.bootstrap --run first --workers 8              # a ladder's v1: recipe v11 until the Greedy curve says stop, frozen
uv run python -m wopr.ab --run first --note "..."                    # clean run: recipe v11, compared, one ledger row
uv run python -m wopr.diagnose runs/first/joshua.pt                  # how its games end, its USSR edge, VP by card
uv run python -m wopr.baseline v1 --run first                        # freeze it into the current ladder
uv run python -m wopr.loop --run first --generations 3               # train, evaluate, gate, promote
uv run python src/main.py --ussr joshua --joshua-checkpoint runs/first/joshua.pt
```
