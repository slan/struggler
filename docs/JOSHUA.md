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

*Amended a second time, 2026-08-25, three arms into the Playdek batch.
The batch's first search arm (hard, USSR, `runs/playdek/search1-hard-search-ussr`)
played the gift itself: 5 of its 6 DEFCON losses are CIA Created /
Grain Sales at DEFCON 2. Cause: amendment fix (2) made the rollout's
simulated opponent the policy — which never takes the gifted coup (the
original finding) — so a gift branch reads clean to the value estimate
and the margin rule keeps the policy's pick; the registered
"search subsumes the veto" property had silently broken. Restored:
the value evaluator now runs the veto's rules-probe — what the
opponent CAN do, not what the policy would — on whatever it picks,
and a provable loss is refused and re-picked. The raw-v3 hard arms
stand (first hard numbers ever: **USSR 0.111 [0.04, 0.28], US 0.036
[0.01, 0.18]**, `search1-hard-joshua-*`); the search and veto arms
run on the guarded code under `runs/playdek/search2-*`.*

*And a third time, one arm later: the guard still did not fire in the
wild (`search2-hard-search-ussr`: 0.040, 10 of 12 DEFCON losses the
gift). The probe's ALL-quantifier over the prober's own options cannot
get through a real ops chain — proving "ops first" loses means
proving it through tens of placements several deep, past any budget
or depth cap, so the gift's "ops" branch stayed unprovable. The probe
now follows the prober's own decisions along the policy's argmax — a
loss on the line the bot would actually play is a loss that will be
realized — keeping ANY over opponent choices and ALL over dice.
Verified against the wild before spending more DLL time: at the four
gift positions replayed from the lost games' own logs, the probe
proves the loss and the player picks another card. Remaining arms:
`runs/playdek/search3-*`.*

*Fourth and last amendment, same day. Two findings from instrumented
live games (`logs/gifthunt/`). One more probe defect: a flat node
budget let the first branch of the opponent's granted-op choice
(influence, tens of options deep) starve the coup branch that mates —
whether a gift proved came down to option ordering; the budget is now
split among the children of every branching node (and raised to 800).
Then the deeper finding: with the guard verifiably firing, the
remaining hard-mode gift losses are **forced endgames** — in every
instrumented DEFCON loss the fatal CIA Created was the hand's only
card, at DEFCON 2, in the turn's last rounds; the probes on all the
earlier safe picks correctly returned False, and the loss was sealed
by *scheduling* (the gift card must be spent or spaced while it is
still safe) rounds before any lookahead horizon. The hard AI's
relentless coups make DEFCON 2 the standing weather, so this shape
dominates there. Inference-time search cannot reach it — it is the
pre-registered "remaining losses are strategic" reading, and
scenario-seeded self-play (a training-time fix) is its named
successor. The batch runs to completion on this final code
(`runs/playdek/search4-*`) to put numbers on what search does buy.*

**Result** (2026-08-26, `runs/playdek/search4-*`; raw v3's easy
numbers are the standing baseline, its hard numbers
`search1-hard-joshua-*`; 60 a seat easy / 30 a seat hard, bid 2,
seeds 300+):

| | easy USSR | easy US | easy mean | hard USSR | hard US |
| --- | --- | --- | --- | --- | --- |
| raw v3 | 0.093 | 0.078 | 0.086 | 0.111 | 0.036 |
| v3+veto | 0.154 [.08,.28] | 0.038 [.01,.13] | 0.096 | 0.120 | 0.111 |
| v3+search | **0.173** [.09,.30] | 0.088 [.04,.19] | **0.131** | 0.077 | 0.083 |

\(a\) The easy two-seat mean moved +0.045 — **under the +0.05 bar by
0.005**, on samples whose intervals are ±0.06 wide: the rule says the
search does not become the standing reported player, and the honest
reading is a real but modest lift the batch is underpowered to
confirm. The lift is the USSR seat's (0.093 → 0.173, the gift seat);
the US seat holds, its losses still the 20-VP blowouts (37 of 52) the
search was never aimed at. (b) The DEFCON-gift suppression is real
and visible in the loss mix everywhere: hard-mode DEFCON endings fall
from 11/7 (raw, by seat) to 4/2 under search — and on hard the same
games reappear as HELD_CARDS and VP losses, because those gifts were
forced endgames (fourth amendment): the win rates stay flat
(0.077/0.083 vs 0.111/0.036, all inside the intervals at 30 games).
\(c\) Hard mode is now a standing eval: everything sits at ~0.04–0.12
a seat — above Greedy's zero, an order below the internal yardsticks.
Veto vs search: the rules-check alone captures most of the USSR-seat
lift (0.154 vs 0.173) — third dataset agreeing the blunder refusal,
not the value re-ranking, is the engine of the gain — though the
veto's easy US seat dipped to 0.038 (21 Europe-control losses, n=52).
*Resolved 2026-08-26: noise, not a veto effect. Same-seed games
against the nondeterministic AI are not paired — across the 60
veto/raw seed pairs the first diverging action is the AI's own in 59
and a US decision in exactly 1, so the veto's behavioral footprint on
that seat is nil (it only fires on provable losses, which the easy US
seat rarely faces before the AI forks the game); the dip and the
Europe-control cluster are sampling variance at n≈50.* Bridge
attrition 2–7 per 60-game arm, as before.

**Decision.** By the pre-registered rules: (a) missed by a hair —
the standing Playdek eval keeps **raw v3** as the reported player,
with `search=`/`veto=` available and their numbers recorded here;
no re-run to chase the 0.005 (the same DLL hours are worth more as
the next experiment's eval). The operative finding is (b)+(c): search
removes the tactical gift, and what remains — forced endgames from
card scheduling as USSR, the VP blowouts as US — is **strategic,
learned-policy territory**. The road map's next step is therefore
**scenario-seeded self-play**: training games started from
gift-in-hand / DEFCON-2 positions (and lost eval games' prefixes if
that purity line is accepted), where both seats learn what search
cannot reach — spending the gift card while it is safe, and the US
seat's early-war defence. Search returns as a cheap multiplier on
whatever that training produces; the layout-bump question stays
parked behind it.

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

### 2026-08-26 — rules version 4: the ladder stands

The search batch's Grain Sales trace exposed a DEFCON-1-at-headline
ruling where the DLL and rule 4.5's note agree against the engine
(WOPR.md has the case); fixed as
`fix/defcon-one-headline-event-owner`, `RULES_VERSION` 4. The
decision points' re-rating on the new engine: **r3-bid2/v3 vs Greedy
0.941** over 400 (US 0.943 / USSR 0.940; standing 0.95 — unmoved),
**Greedy against itself at bid 2 0.48/0.60 by seat** over 200 — even
within noise, and the changed corner is unreachable between these
bots. Neither yardstick moved: the `r3-bid2` ladder stands, the bump
noted. Every number in this file measured before this date is rules
version ≤3's; the differences live only where a headline event chain
reaches DEFCON 1.

### 2026-08-28 — layout v2: order and recency; a fresh bootstrap

**Question.** The search arc closed on "the remaining losses are
strategic": the US seat's early-war 20-VP blowouts and the USSR seat's
forced endgames. The r1 road map's carried hypothesis says the first is
partly a *guessing problem* — which scoring cards are out, what the
opponent's plays this turn say about its hand — that the layout cannot
express: a region scored on turn 2 and one scored this turn look the
same, and the set of discards carries no order. Does giving the network
order and recency move what self-play learns, before any prior is
introduced? (Decided on reflection 2026-08-28: exhaust the no-priors
levers — this bump plus the retrain it forces — then assess
scenario-seeded self-play, a prior over *states*, on the result.)

**Setup.** The engine gains a public play/discard log
(`Engine.card_history`, exposed by `observe()` — every entry a card
both players saw leave a hand for a pile, so mandate #4 holds; carried
by `serialize()`, goldens regenerated, rules unchanged — no
`RULES_VERSION` bump). Layout v2 (`LAYOUT_VERSION` 2): a
`discard_this_turn` card location, per-card `card_recency` (seen ever /
turns since — survives reshuffles, which is exactly the scoring-cards
guess), and `hist_*` — the last 32 log entries, most recent first, by
me/them, attention-pooled by a query from the globals latent (WOPR.md
has the tables). Checkpoints refuse to load across the bump, so the
line restarts with `wopr.bootstrap --bid 2` (recipe v11, rules
version 4): it freezes as `baselines/r4-bid2/v1`, then the loop to
plateau, then the Playdek eval. `r3-bid2` stands as the old-layout
reference — its v3 re-rated on this engine at 0.941 vs Greedy over
400.

**Metrics and decision rule** (written before the run): (a) the
bootstrap's Greedy curve against the old bid-2 bootstrap's (0.17 /
0.38 / 0.40 / 0.83 at 2/4/8/11k, confirmed @11,024) and the loop's
plateau against r3-bid2's (v3 @23k, 0.95) — faster or higher says the
features help *learning*; (b) the standing Playdek eval of the new
line's champion (60 a seat easy / 30 hard, bid 2, raw argmax) against
raw v3's 0.093/0.078 easy and 0.111/0.036 hard — the champion becomes
the reported player if the easy two-seat mean improves on 0.086 by
≥ 0.05, the bar search missed; (c) `wopr.diagnose` and the eval's loss
mix on the two named failure shapes — the US early-war blowout share
is the one the features were aimed at. Readings: (b) clears →
scenario-seeded self-play starts from this champion, the ceiling
higher than feared; (b) washes while (a) improved → the gain is
internal-only and transfer is still the wall — priors next, as
assessed; both flat → the r1 recency hypothesis closes negative and
the layout was not the constraint — priors next, and the AlphaZero-
style training question stays parked (it multiplies strength on the
policy's own distribution, which is not the failing axis). Ledger: the
bootstrap's freeze writes the row.

**Result — bootstrap** (2026-08-28, `runs/r4b2-boot`,
`bootstrap.csv`). **Confirmed @12,518** (r3-bid2: 11,024). The curve
by rolling mean: 0.13 at 2k, 0.23 at 4k (old: 0.17, 0.38 — the USSR
seat lagged at 0.11–0.19 while the US was already 0.33+), one sharp
USSR catch-up at 6k (0.19 → 0.52 in one tick), 0.54 at 6.5k (old: 0.40
at 8k), confirmation 0.788 (US 0.750 / USSR 0.827) over 600. Frozen as
**r4-bid2/v1**: vs Greedy 0.766 (US 0.697 / USSR 0.835), random 0.988,
Elo +1090 ± 215. Reading on (a), bootstrap half: **not faster** —
~14% more games to confirm, a differently-shaped curve (slower early,
steeper middle), landing at the same place. The loop's plateau and the
Playdek eval remain the deciders.

**Result — loop** (2026-08-28, `runs/r4b2-boot/loop.csv`). Plateau at
**v4 @28,520** (promotions at gens 1, 2, 4; gens 3 and 5 missed —
2-of-3 rule): **0.940 vs Greedy** (US 0.900 / USSR 0.980), Elo +1180
± 111. The old line: v3 @23k at 0.95 (0.941 re-rated on this engine).
Reading on (a), complete: **the layout bump changes neither learning
speed nor the internal ceiling** — one more promotion, ~5.5k more
games, the same plateau height, and the gate's seat split (the
challenger wins as USSR, not as US) unchanged. Whatever order and
recency bought must show up against Playdek or not at all.

**Result — Playdek** (2026-08-28, `runs/playdek/r4b2v4-easy`; easy,
bid 2, argmax, seeds 300+): USSR **0.057** [0.02, 0.15] (3/53), US
**0.035** [0.01, 0.12] (2/57), two-seat mean **0.045** — against raw
v3's 0.093 / 0.078 (mean 0.086) and the pre-registered bar of 0.136.
Attrition 5 desyncs + 5 void (known families) of 120. The loss mix is
the familiar pair, unmoved: 33 US-seat VP blowouts, 17 USSR-seat
DEFCON endings among the first 96. The hard arm was cancelled by
decision (below) — its DLL-hours are worth more elsewhere.

**Decision** (2026-08-28, on review). (b) missed decisively with (a)
flat: the r1 order/recency hypothesis **closes negative** — the
features change nothing internally and buy nothing against the real
opponent (plausibly cost something: 0.045 vs 0.086, intervals
overlapping). **Raw v3 (r3-bid2) stays the reported player.** Two
consequences adopted:

1. **The goal is re-anchored: beat the easy AI first** — > 0.5 on
   both seats at bid 2 is the arc bar, and hard mode is not measured
   again until it is met. Everything measured to date sits at
   ≤ 0.17 a seat against easy; hard evals were sizing a mountain that
   cannot be attempted until this one is climbed.
2. **Carried to the priors arc as its first question: does layout v2
   stay?** A live suspicion: the sequence features may *sharpen*
   self-play overfitting — the policy can condition on play-order
   patterns only a self-play opponent produces, which Playdek never
   does. When the priors arc trains on this line, a hist-ablation
   (layout v2 minus `hist_*`, recency kept) is pre-registered first;
   starting from the layout-v1 line (r3-bid2/v3) instead remains on
   the table.

Ledger: the frozen versions' README entries are the rows.

### 2026-08-28 — the scenario arc opens: the hist-mask diagnostic

**Question.** The arc's carried first question is whether layout v2
stays for training, and the fork is expensive: a hist-ablation ladder
(layout v2 minus `hist_*`, recency kept) is a full bootstrap + loop +
Playdek eval before any scenario seeding happens. One DLL-hour buys a
pointer first: is the v2 line's easy deficit (0.045 vs raw v3's 0.086)
carried by the `hist_*` features *at inference* — the
sequence-features-sharpen-self-play-overfitting suspicion, the policy
conditioning on play-order patterns only a self-play opponent produces
— or baked into the trained weights either way?

**Setup.** `r4-bid2/v4` replayed against the easy AI with the play
history masked empty at inference: `hist_card` all empty slots,
`hist_feats` zero — the encoding of a game's first decision, so the
history attention pools to zero; `card_recency` and
`discard_this_turn` kept. `JoshuaPlayer(mask_hist=True)`,
`--policy histmask=ckpt.pt` in `wopr.playdek.eval`;
`tests/test_joshua_player.py` pins that nothing but `hist_*` moves.
The caveat is stated up front: a mid-game state with an empty log is
off-distribution for these weights, so some degradation is the
expected direction — only *recovery* is evidence, and a flat or worse
number cannot separate "the features were inert" from "the masking
itself hurt".

**Metrics and decision rule** (written before the run): the standing
easy eval, same flags as `runs/playdek/r4b2v4-easy` (120 games, 60 a
seat, bid 2, argmax, seeds 300+), primary metric the two-seat mean
against masked-off v4's 0.045 and raw v3's 0.086; secondary the loss
mix (the US blowout share, the USSR DEFCON share). Readings: mean
recovers to ≥ 0.08 → the hist features actively hurt at inference —
the hist-ablation (layout v2 minus `hist_*`, recency kept) earns its
bootstrap as the arc's base line; ≤ 0.05 → no evidence against the
weights-not-features reading — the arc starts from the layout-v1 line
(`--init` from r3-bid2/v3), the hist-ablation parked unless the arc
plateaus and diagnosis points back at the guessing problem;
in between → underpowered either way, and the layout-v1 line is the
cheap default, the question noted as unresolved rather than closed.
Ledger: no row — nothing is trained.

**Result** (2026-08-28, `runs/playdek/r4b2v4-easy-histmask`; 152 min,
8 workers): USSR **0.038** [0.01, 0.13] (2/53), US **0.018**
[0.00, 0.10] (1/55), two-seat mean **0.028** [0.01, 0.08] — against
raw v4's 0.045 and raw v3's 0.086. Attrition 11 desyncs + 1 void of
120 (known families, a notch above the raw run's 5+5). The loss mix
moved: DEFCON endings 36 of 105 clean losses (raw v4: 17 among the
first 96), VP 58, mean final turn 4.5 — masked games die earlier and
more often by DEFCON.

**Decision** (2026-08-28, by the pre-registered rule). ≤ 0.05 fired:
no recovery — the v2 line's deficit is in the trained weights, not
removable at inference. The DEFCON-loss share *rising* under the mask
says the policy genuinely leans on `hist_*` in self-play (losing the
features off-distribution hurts) while whatever it reads there does
not transfer — consistent with the overfitting suspicion, though this
diagnostic cannot prove that direction. **The arc trains from the
layout-v1 line (`--init` from r3-bid2/v3); the hist-ablation ladder
is parked** unless the arc plateaus and diagnosis points back at the
guessing problem. Consequence executed with this entry: the encoding
returns to layout v1 (`LAYOUT_VERSION` 1 — r3-bid2 checkpoints load
again; the r4-bid2 ladder stays frozen as the layout-v2 record), the
engine's public `card_history` stays (rules-neutral, serialized,
goldens already regenerated), and layout v2's encoding — with the
`histmask=` diagnostic that closed it — remains recoverable from git
history (`4b74b5e`, this entry's commit for the diagnostic).

### 2026-08-28 — rules version 5: the ladder stands

The r4b2v4 easy evals' desync mix (16 over 240 games) gave up two
families. The larger (5 of 16, every one Tear Down This Wall) was an
engine bug: an event-granted free Realignment chain lost the card's
terms after its first roll — at DEFCON 2 the second European target
was never offered, and a target outside the card's named countries
was. Fixed by carrying `restrict`/`ignore_defcon` through the chain
(`RULES_VERSION` 5, docs/ARCHITECTURE.md; WOPR.md's ninth pass). The
second (2 of 16) was the trapped seat's scoring card reaching game
over — the engine plays it, the DLL holds it to a HELD_CARDS loss —
now the documented rules difference's void instead of a fatal desync.
A `_simulate` that matches no option now reports each option's stop
point and diff, so the singleton families come back from eval volume
with traces attached.

*Amended same day, from the scen1 eval's six desyncs (the new detail
sorted them in one pass).* A second rules-v5 fix: a war card whose
event is prevented does not trigger Flower Power (seed 358 — the
engine paid 2 VP for Arab-Israeli War under Camp David where the DLL
pays nothing; adopted from the DLL's reading, and the operator's old
`_flower_power_check` void, which had failed to fire there anyway, is
gone since the programs now agree). Still open, the plan in WOPR.md's
ninth pass: Grain Sales' taken-card resolution (3 of 6) and a
one-card hand/deal drift (2 of 6). Note: scen1 trained and its eval
ran on v5's first fix only; the Flower Power corner affects neither
(a training-arena game has no DLL, and the eval's drifted games were
already desyncs).

The decision points' re-rating on the new engine: **r3-bid2/v3 vs
Greedy 0.940** over 400 (US 0.940 / USSR 0.940; standing 0.941 —
unmoved), **Greedy against itself at bid 2 0.52/0.48 by seat** over
200 — even within noise. The `r3-bid2` ladder stands, the bump noted.
Verified against the DLL: hotseat 8/8, the differ 12/12, known
families only.

### 2026-08-28 — scenario-seeded self-play: the first run

**Question.** Does shaping the initial-state distribution — training a
fraction of games from positions where the named failure is on the
table — move the easy-AI numbers where the layout bump and
inference-time search did not? The arc bar: > 0.5 both seats at bid 2;
this run's bar is more modest and pre-registered below.

**Setup.** The layout-v1 line continues: a fresh run `--init` from
r3-bid2/v3 (recipe v11, bid 2, self-play + pool as always), with
`--scenarios scenarios/defcon2-gift-v3.jsonl --scenario-frac 0.25` —
the bank harvested from v3's own sampled self-play (400 games,
predicate `defcon2_gift`: an action-round card pick at DEFCON 2 with
the granted-op gift in the mover's hand; WOPR.md, "Scenario-seeded
starts"). Each scenario game re-hides the state with
`Engine.determinize`, so the mover's knowledge is preserved and the
hidden world is fresh per game. 8,000 games, ~50 min at 8 workers.
No control arm (decided on review: the eval hours are worth more on
the arm itself; v3's standing numbers are the reference). Evaluation
stays at the printed game.

**Metrics and decision rule** (written before the run): (a) the
standing easy eval of the result (120 games, 60 a seat, bid 2, argmax,
seeds 300+) against raw v3's 0.093/0.078 (mean 0.086) — the run
becomes the reported player if the two-seat mean improves by ≥ 0.05
(the bar the search missed); (b) the mechanism metric: the
DEFCON-ending share of the USSR seat's losses (raw v3: ~17–21 of
~50) — halving with a flat mean still says the gift lesson landed and
the residual is the US blowout, pointing scenario v2 at early-war
US-defence states; (c) vs Greedy and the USSR edge (`wopr.diagnose`)
as internal sanity — a collapse there voids the eval spend. Readings:
(a) clears → scenarios transfer; loop with scenarios to plateau, then
the eval again. (a) flat with (b) halved → the prior works but is too
narrow; scenario v2 adds the US-seat shape. Both flat → the prior as
constructed does not transfer; diagnose before the next arm (the
league exploiter stays in reserve). Ledger: the run's `wopr.ab
--existing` row when it is compared.

**Result** (2026-08-28, `runs/playdek/scen1-easy`; 120 games, 140
min): USSR **0.113** [0.05, 0.23] (6/53), US **0.052** [0.02, 0.14]
(3/58), two-seat mean **0.081** — raw v3's 0.086, the bar 0.136.
(b) unmoved: the USSR DEFCON-loss share is 23 of 47 (raw v3: ~0.4);
the US seat's mix is 35 VP blowouts and a new prominence of 11
Europe-control losses. (c) passed before the eval: 0.942 vs Greedy,
USSR edge 0.575. Attrition 7 desyncs + 2 void (one the old Flower
Power check firing — that family's frequency confirmed; both void
kinds are gone on the fixed code). Both (a) and (b) flat: the
pre-registered negative branch.

**Diagnosis** (2026-08-28, run before any next arm, per the decision
points): scen1 against v3 head-to-head, argmax, 200 games each
condition — **from the bank's own gift states 0.640, from the printed
start 0.605**. The run is simply stronger than v3 across the board
(8,000 more games; it would clear a loop gate), and the gift-specific
edge (+0.035) is inside noise. So the scenario dose did not
specifically teach the gift lesson even in self-play, and the general
internal gain transfers to the easy AI not at all — the third
experiment in a row (layout v2, search, now the state prior) where
the internal ladder moves and the Playdek number does not.

**Decision.** The question closes negative as constructed: a 25%
state prior at 8k games neither shifts the loss mix nor the easy
mean. The reading this leaves standing: the wall is the *opponent
distribution*, not the state distribution — self-play opponents do
not punish the gift the way the AI does, wherever the game starts.
That is the exact case the road map's ladder anticipated: the league
exploiter (a policy trained to beat the line's champion, PFSP-style)
comes out of reserve as the next candidate, and behind it, relaxing
SELF-PLAY-ONLY becomes an explicit review decision. Neither starts
without its own pre-registered entry. The scenario infrastructure
stays: any future arm can shape its start states for free.

### 2026-08-28 — the league exploiter (pre-registered)

**Question.** Can an opponent *trained to beat the line* — rather than
the line's naive self-play distribution — supply the punishment the
easy AI applies and self-play does not? Three arms in a row (layout
v2, search, the state prior) moved the internal ladder and not the
Playdek number; scen1's diagnosis pinned the wall on the opponent
distribution. The exploiter attacks exactly that while staying inside
SELF-PLAY-ONLY: it is the line's own weights, trained against the
line's frozen versions.

**Setup.** Two runs, in order.

*Run A, the exploiter* (`runs/exploit1`): **fresh weights** (decided
on review over `--init` from v3: AlphaStar's exploiters started
fresh, and a fresh net can find exploits outside v3's basin; the risk
that one ab-budget is too small for a net that has to learn the game
*and* the exploit is accepted, hedged by the mix below). Recipe v11's
hyperparameters at bid 2, 8,000 games, with the mix changed to
`--self-play 0.2 --vs-pool 0.8 --snapshot-every 0 --pool-seed
v1=baselines/r3-bid2/v1/joshua.pt v2=baselines/r3-bid2/v2/joshua.pt
v3=baselines/r3-bid2/v3/joshua.pt` (the new pool-seed wiring): the
pool *is* the champion line and never grows. This is the
fixed-opponent shape the r1 anchors failed on (constant reward once
one side always wins) — the hedge is threefold: PFSP's hardness
weighting walks the line as a curriculum (v1 first once anything
falls), the 20% self-play slice keeps a live gradient while every
champion still wins, and the entropy bonus does the rest. Scenario
seeding on (`--scenarios scenarios/defcon2-gift-v3.jsonl
--scenario-frac 0.25`): scen1 showed the state prior alone teaches
nothing, but the exploiter *seated as the punisher* in those states
is the half that was missing. No DLL anywhere in run A.

*The gate, A → B* (pre-registered bar): the exploiter against v3,
head to head (`wopr.eval`, argmax, eval seeds 0/1/2, 200 games a
seed-pair per seed): **worst seed ≥ 0.6**. Miss it with the
vs-champion curve still climbing at 8k → continue the run once, one
more ab-budget, cap total 16k (the bootstrap's Cap rule). Miss it
flat → the arm washes at step A; record the diagnosis and stop.
Either way run A gets its `wopr.diagnose` (endings, VP by card, vs
v3) — *how* an exploiter beats the line, or fails to, is the arm's
finding regardless of the gates.

*Run B, the champion counter-run* (`runs/counter1`): `--init` from
v3, recipe v11 unchanged (self-play 0.5 / vs-pool 0.5, snapshots on)
except the pool starts seeded with the exploiter
(`--pool-seed exploit1=runs/exploit1/joshua.pt`), 8,000 games, bid 2.
PFSP makes the exploiter's share self-adjusting: sampled most while
it still beats the learner, fading as the lesson lands.

*Internal gate before any DLL spend*: `wopr.diagnose` on run B — vs
Greedy must hold (≥ 0.9; v3 is 0.940) and the USSR edge stay in
family; a collapse voids the eval and the arm records why. Ledger
rows via `wopr.ab --existing` for both runs.

**Metrics and decision rule** (written before any training): (a) the
standing easy eval of run B (120 games, 60 a seat, bid 2, argmax,
seeds 300+) against raw v3's 0.093/0.078 (mean 0.086), bar **mean ≥
0.136** (+0.05, the same bar search missed) → run B is the reported
player and the loop continues with an exploiter slot in the mix. (b)
The mechanism metric: the USSR DEFCON-loss share (raw v3 ~0.4) and
the US blowout share. (a) flat with (b) halved → the punishment
landed and transfer still failed — the strongest evidence yet that
the wall is not reachable from inside SELF-PLAY-ONLY; relaxing it
goes to review. Both flat → the arm closes negative; relaxing
SELF-PLAY-ONLY becomes the explicit review decision either way
(road map ladder, user's call, never an experiment default). (c) The
eval's desyncs are mined first — the `grain` and `hand-drift`
evidence lines (WOPR.md, ninth pass) should pin both open families.

**Budget.** Run A 8k (~50 min, one continuation allowed on the Cap
rule) + run B 8k + one DLL eval at the end (120 games, ~140 min).
Nothing else without a new entry.

**Result** (2026-08-29, `runs/exploit1`). At 8k the fresh net was at a
0.06 cumulative rate against v3 with the windowed pool curve tripling
(0.045 → 0.15, no plateau) — the Cap continuation ran to 16k. There
the curve was *still* climbing (0.17 → 0.43 windowed; final
cumulative v1 0.262 / v2 0.168 / v3 0.140, a clean hardness
gradient). The gate, argmax vs v3 at bid 2, seeds 0/1/2 × 200:
**0.330 / 0.320 / 0.280 — worst seed 0.28 against the 0.6 bar,
miss**, and the 16k cap is spent. The seat split is the finding: as
USSR the exploiter plays v3 near-even (0.48 / 0.50 / 0.39), as US it
is ~0.16. The diagnosis agrees — it became a *USSR specialist*: USSR
edge 0.725 sampled (v3: 0.575), a scoring-card engine tilted hard to
the USSR side (Europe +328/66, Central America +223/32 net VP), half
its self-play wins USSR-by-VP, and vs Greedy 0.838 [0.825]
(US 0.77 / USSR 0.91) against v3's 0.940 — general strength traded
for the anti-line lesson, which is what an exploiter is for. Ledger
row written (`wopr.ab --existing`); `runs/exploit1/diagnose-vs-v3.json`.

**Decision (superseded in part — see the continuation entry below).**
The gate missed at the cap, so run B never starts and
no DLL hours were spent — the standing easy number (raw v3, 0.086)
is untouched. The question closes **negative as constructed**, with
the honest caveat in the curve: 16k games was not enough for a fresh
net to crack the 0.6 bar, but it was still learning when the budget
ran out, and one seat got to even. What the arm established: the
line *is* exploitable from inside SELF-PLAY-ONLY (a fresh net finds
a USSR-side attack v3 does not defend), but reaching
counter-training strength costs more than an ab-budget of exploiter
training — and whether to spend that (a bigger exploiter budget, or
a v3-init variant, each needing a new entry) now competes directly
with relaxing SELF-PLAY-ONLY, which per the road map's ladder is an
**explicit review decision, the user's call**. The pool-seed wiring
stays: either continuation reuses it.

### 2026-08-29 — the exploiter continuation (pre-registered)

**The review decision** (user, 2026-08-29): option 1 of the four —
more exploiter budget. The others (v3-init exploiter, counter-run on
exploit1 as-is, relaxing SELF-PLAY-ONLY) stay live and can follow;
nothing about them is decided by this entry.

**Question.** Was the budget the only thing between the fresh
exploiter and the gate? The 16k curve says maybe: still climbing at
the cap, one seat already even. This entry buys the cheapest possible
answer before the more expensive forks.

**Setup.** `runs/exploit1` resumes unchanged — same recipe, mix,
seeded pool, gift bank (`--pool-seed` is idempotent on resume) — in
two more ab-budgets: to 24k, gate eval; if missed *and the windowed
vs-pool curve rose over the segment*, to 32k, gate eval. **Hard cap
32k**, no further continuation from this entry.

**Metrics and decision rule** (written before resuming): the gate is
unchanged — argmax vs v3 at bid 2, seeds 0/1/2 × 200 games, **worst
seed ≥ 0.6**. Watched beside it, per segment: the windowed vs-pool
curve (flat across a whole segment + a gate miss = stop early, the
budget is not the answer) and the US-seat share of the head-to-head
(0.16 at 16k — the gate cannot pass without the US seat moving; a
gate miss with the USSR seat ≥ 0.6 and the US seat still < 0.3 says
*specialist, not budget*, and points at option 3's asymmetric
counter-run rather than more of this). Gate passes → run B and the
decider exactly as pre-registered in the parent entry (counter1,
v3-init, exploiter-seeded pool, 8k; `wopr.diagnose` internal gate;
one DLL easy eval vs 0.086, bar +0.05). Gate misses at 32k → this
arm is done for good; the review reconvenes on options 2–4 with the
curve and seat split as evidence.

**Budget.** Up to 16k more training games (~100 min) + two gate
evals; run B and the DLL eval only past the gate, as already
budgeted. Ledger: one `wopr.ab --existing` row at whatever endpoint
this reaches.

**Result** (2026-08-29). At 24k: gate 0.522 / 0.512 / 0.472 (worst
0.472, miss), segment curve up (0.46 → 0.62 windowed), US seat
0.31–0.38 — both watched metrics said budget, on to the cap. At 32k:
gate **0.660 / 0.565 / 0.520 — worst seed 0.520, miss**, cap spent,
final segment flattening (~0.59–0.64). The trajectory across the
arm: worst seed 0.28 → 0.47 → 0.52, mean 0.31 → 0.50 → 0.58; the US
seat went 0.16 → 0.34 → 0.45–0.64, so the specialist reading
dissolved — at the cap the exploiter beats v3 *on mean on every
seed* and misses only the pre-registered bar's worst-seed margin
(and would miss the loop's own 0.55 bar by 0.03). Ledger row at the
endpoint.

**Decision.** The gate missed at the hard cap: this arm is done —
no run B from this entry, no DLL hours spent, raw v3 still the
reported player. The question's honest answer: budget was *most* of
the problem (every doubling roughly halved the gap) but the curve
flattened before the bar. The review reconvenes on options 2–4 with
the evidence changed in one important way: option 3's asymmetric
counter-run was argued at 16k from a USSR-only attack, and at 32k
exploit1 is a *both-seats* near-peer of v3 (0.582 mean) with the
attack intact — seeding it into a counter-run no longer waives much
of the bar the gate protected. Options 2 (v3-init) and 4 (relaxing
SELF-PLAY-ONLY) stand as before.

### 2026-08-29 — the counter-run on exploit1 as-is (pre-registered)

**The review decision** (user, 2026-08-29, round two): option 3. The
deviation from the parent entry is explicit and accepted: run B
starts although the exploiter's gate was **missed** (worst seed
0.520 vs the pre-registered 0.6) — the waiver is justified by what
the gate was *for* (guaranteeing the counter-run a teacher that
actually punishes the champion), which a 0.582-mean both-seats
near-peer provides.

**Question.** Does counter-training against the line's own exploiter
— the first opponent-distribution change the champion has ever seen
— move the easy-AI number where state priors and search did not?

**Setup.** `runs/counter1`: `--init` from r3-bid2/v3, recipe v11
unchanged (self-play 0.5 / vs-pool 0.5, snapshots every 5 updates),
bid 2, 8,000 games, the pool seeded with the exploiter
(`--pool-seed exploit1=runs/exploit1/joshua.pt`) beside its own
snapshots — PFSP makes the exploiter's share self-adjusting: sampled
most while it still beats the learner, fading as the defense lands.
No scenario bank (the parent spec: the exploiter itself carries the
attack states). Evaluation stays at the printed game.

**Metrics and decision rule** (written before training, inherited
from the parent entry): *internal gate before any DLL spend* —
`wopr.diagnose` on counter1: vs Greedy ≥ 0.9 (v3: 0.940) and the
USSR edge in family; a collapse voids the eval and closes the arm.
Watched beside it (mechanism, not a gate): counter1 vs exploit1 and
counter1 vs v3, argmax, seed 0 × 200 — the defense metric (v3 loses
to exploit1 at 0.418 mean; the counter-run should push its own
number against exploit1 up without falling below even against v3).
*Decider*: the standing easy eval (120 games, 60 a seat, bid 2,
argmax, seeds 300+) against raw v3's 0.086 — **mean ≥ 0.136**
(+0.05) makes counter1 the reported player and the loop continues
with an exploiter slot in the mix. Mechanism metrics as always: the
USSR DEFCON-loss share (raw v3 ~0.4) and the US blowout share. Both
flat → the opponent-distribution lever as constructed (one
exploiter, one counter-run) joins the negatives, and option 4
(relaxing SELF-PLAY-ONLY) is the remaining rung — still a review
decision. The eval's desyncs are mined first (the `grain` /
`hand-drift` evidence lines). Ledger: `wopr.ab --existing`.

**Budget.** 8k games (~50 min) + diagnose + the head-to-heads + one
DLL easy eval (120 games, ~140 min). Nothing else without a new
entry.

**Result** (2026-08-29, `runs/counter1`, `runs/playdek/counter1-easy`).
The internal story is exactly what the arm asked for: PFSP kept the
exploiter prominent (409 of its pool games), the learner's record
against it rose to 0.489 in training, and the head-to-heads confirm
the defense — counter1 vs exploit1 **0.532** (v3 managed 0.418), vs
v3 **0.600**, vs Greedy **0.975**. Internal gate passed (one flag:
USSR edge 0.742, inherited tilt from the exploiter's 0.725; v3 is
0.575). Then the decider: easy AI 120 games, USSR **0.093**
[0.04, 0.20], US **0.053** [0.02, 0.14], two-seat mean **0.072** —
raw v3's 0.086, the bar 0.136. Mechanism metric worse, not halved:
the USSR DEFCON-loss share is 27/48 = **0.56** (raw v3 ~0.4) — the
inherited USSR aggression buys wins against the line and DEFCON
deaths against the AI; the US mix is 28 VP blowouts + 14
Europe-control of 55. The fourth arm in a row where the internal
ladder moves (0.600 over the champion would clear any gate) and the
Playdek number does not.

**Desync mining** (pre-registered first step; 6 desyncs + 3 void of
120): the ninth-pass instrumentation delivered. *Grain family
pinned*: three traces with the taken card's full location history —
seed 354 (UN_Intervention, 'return' read off location, confirmed by
simulation), seed 386 (Independent_Reds, 'take' by simulation,
location agreeing), and the decisive seed 390 (Olympic_Games:
**simulation says 'take' while the location read says 'return'** —
the location heuristic is unreliable and the resolution must trust
simulation). *Hand-drift evidenced*: seed 338 carries the
unaccounted-exits list (Decolonization, Lone_Gunman; US hand DLL 8
vs engine 2) and seed 405 a one-card drift with the full state diff.
Bridge work, next time the bridge is the arm.

**Decision.** The question closes **negative**: counter-training
against the line's own exploiter teaches the defense the exploiter
tests and none the easy AI applies. With it, the
opponent-distribution lever *as reachable from inside SELF-PLAY-ONLY*
is spent — the exploiter existed, the defense landed, the transfer
did not happen. The remaining rung is option 4, relaxing
SELF-PLAY-ONLY (the easy AI or Greedy as a teacher in the mix),
which stays what it always was: an explicit review decision, the
user's call. Raw v3 remains the reported player.

### 2026-08-29 — rules version 6 (the bridge as the arm)

The review chose the bridge over the last training rung. The eval's
grain traces pinned the family (WOPR.md, tenth pass): the DLL's AI
*declines* event-granted Operations, which the engine made mandatory
against every granting card's "may then conduct Operations" — seed
390's Grain Sales return stalled both simulations at the granted-Ops
decision with no fact to spend and no state diff. Rules version 6
makes the grant declinable (a `pass` on the pushed `OPS_TYPE`;
Missile Envy's taken-card Ops stay mandatory), riding the layout's
`other` flag — no layout bump, every checkpoint loads. The bridge
bounds the grant's facts at the seat's next queued card play and
reads an empty bound as the decline; the mis-attribution that bound
removes is the likely mechanism of the family's silent drifts.

The decision points' re-rating on the new engine: **r3-bid2/v3 vs
Greedy 0.939** over 400 at bid 2 (standing 0.940 — unmoved), **Greedy
against itself 0.500** over 200 (0.59/0.41 by seat, within noise of
v5's 0.52/0.48). The `r3-bid2` ladder stands, the bump noted.
Verified against the DLL: the grain sweep 149/149, hotseat 8/8, the
differ 12/12, known families only. Open: the hand-drift family
(seeds 338/405's traces recorded), by the next eval's volume.

The arm's decider ran the same day (`runs/playdek/v3-easy-r6`: raw
v3, 120 easy games, seeds 300+, bid 2, on the fixed bridge):
**6 desyncs + 1 void** — headline attrition unmoved from counter1's
6+3, but the composition shifted: the silent-decline subfamily is
gone and the surviving grain traces exposed a second root (the AI
taking a card whose event needs the bot's input; WOPR.md, eleventh
pass — fixed the same day, a prompt-fit veto in the simulation
judge, all sweeps clean). Still open with traces: the one-card
hand/deal drift (seeds 315, 391) and a trap-discard corner (348).
The batch also re-baselined v3 on v6: USSR **0.089** [0.04, 0.19],
US **0.000** [0.00, 0.06] (0/57), mean **0.044** — the USSR seat at
its standing value, the US seat's zero either a ~1–5% tail of the
standing 0.05–0.078 or a real shift; one batch does not move the
standing number, the next eval (which will also carry the eleventh
pass's fix) arbitrates.

### 2026-08-30 — relaxing SELF-PLAY-ONLY: the DLL as teacher, by distillation (pre-registered)

**The review decision** (user, 2026-08-30, round three): relax
SELF-PLAY-ONLY. The constraint is amended, not repealed: live DLL
games stay evaluation-only — throughput rules them out as sparring
(the AI spends 15 s per decision, a real game is 30–50 minutes of
one core, one game per process; WOPR.md) — and what enters the mix
is the easy AI's *policy*, distilled from the bridge's logged games
into a pool opponent. The bridge arc's dividend is the corpus: every
easy eval batch already wrote replayable engine logs.

**The argument past arm 1.** Arm 1's lesson was "coverage comes from
opponents that *exploit*, not opponents that differ" — a Greedy
share washed because Greedy never takes the gifted coup. The easy AI
does: the DEFCON gift is ~40% of v3's USSR-seat losses at every
strength measured (21/52 in arm 1's own eval, 27/48 at counter1's).
A clone that inherits that habit is the first pool opponent that
punishes the gift. The premise is falsifiable and gated before any
training spend: a clone that does not convert the gift closes the
arm at the gate.

**Question.** Does sparring against the easy AI's own habits —
distilled from the bridge's logs into a pool opponent — move the
easy-AI number where four internal-transfer arms did not?

**Setup.** Three phases.

1. *Harvest* (`wopr.distill harvest`, zero DLL hours): replay every
   easy **bid-2** batch's game logs on the current engine
   (v3-easy-r6…r12, counter1-easy, scen1-easy, r4b2v4-easy and
   -histmask, search4-easy-\*, r3bid2v1/v3-easy-bid2-\*,
   gshare-easy-bid2-\* — ~2,100 games; a log that no longer replays
   under the current rules is skipped and counted). At every AI-seat
   decision with ≥ 2 options, encode `observe(ai)` in the current
   layout and record the chosen option index. The AI's own hand is
   mostly hidden in the engine's physical-mode mirror (median ~5 of
   8 cards), so the hand is **determinized in hindsight**: known
   cards kept, the chosen card and the AI's later same-turn card
   plays forced in, the remainder sampled uniformly from the unseen
   pool with an RNG seeded by (game seed, step index). Physical-mode
   option lists are kept as-is — forcing the chosen card into the
   hand keeps "play from your hand" learnable, and non-hand options
   are labeled negatives. Rows that exceed the layout's K_MAX are
   skipped and counted.
2. *Distill* (`wopr.distill train`): **falken1** — a fresh JoshuaNet
   at v11 capacity (hidden 256), cross-entropy of the option logits
   against the chosen index, rows held out by game (10%), stopped
   when held-out top-1 stops improving; `runs/falken1/joshua.pt`.
   The value head is untrained: a pool opponent's `choose` never
   reads it.
3. *The counter-run* (`runs/teach1`, only after the exploit gate):
   `--init` from r3-bid2/v3, recipe v11 unchanged, bid 2, 8,000
   games, `--pool-seed falken=runs/falken1/joshua.pt` — the counter1
   wiring, PFSP making the teacher's share self-adjusting.

**Metrics and decision rule** (written before training).

- *The exploit gate, before teach1 starts*: v3 vs falken1, 200 games
  argmax (`wopr.diagnose --vs`): of v3's USSR-seat losses, the
  DEFCON-loss share must be **≥ 0.15** (the easy AI: ~0.4–0.56;
  Greedy: ~0, arm 1's wash). Below it the clone did not inherit the
  exploit and the arm closes with no training and no DLL spend.
  Reported beside it, context not gates: held-out top-1 accuracy
  against the legal-uniform floor, falken1 vs Greedy, falken1 vs v3.
- *Internal gate before DLL spend* (inherited from counter1):
  `wopr.diagnose` on teach1 — vs Greedy ≥ 0.9 (v3: 0.940); a
  collapse voids the eval and closes the arm.
- *Decider*: the standing easy eval (120 games, 60 a seat, bid 2,
  argmax, seeds 300+) against raw v3's standing 0.086 — **mean ≥
  0.136** (+0.05) makes teach1 the reported player and the teacher
  slot stays in the mix. Mechanism metrics as always: the USSR
  DEFCON-loss share (raw v3 ~0.4) and the US blowout share. Both
  flat → distillation-of-the-teacher joins the negatives, and the
  remaining relaxation (live DLL games as sparring) stays priced at
  30–50 minutes a game and would need its own entry. The eval's
  desyncs are mined first, as always.

**Budget.** Harvest + distill: local compute only, zero DLL hours.
teach1: 8k games (~50 min). One DLL easy eval (120 games, ~140 min).
Nothing else without a new entry.

**Result, phases 1–2 and the exploit gate** (2026-08-30,
`runs/falken1`). Harvest: **265,683 rows from 1,853 clean games** (33
shards; 11 logs failed to replay across all rules eras, 0 option
mismatches, 0 over K_MAX). Distill: held-out top-1 **0.610** against
a 0.178 legal-uniform floor, best at epoch 15 of 18 (the teacher is a
15 s search with a stochastic policy scored on determinized hands — a
ceiling well below 1.0 was expected). The exploit gate — measured
with seats *fixed* per half (`runs/falken1/gate.py`, 100 games a
seat, argmax, bid 2; `wopr.diagnose --vs` alternates seats and cannot
split endings by which policy sat where, so the pre-registered metric
kept its definition but not its harness — **passed**: of v3's 34
USSR-seat losses to falken1, **17 are DEFCON losses, share 0.50** —
inside the easy AI's 0.4–0.56 band, where Greedy sits at ~0. The
clone inherited the exploit. Context: v3 vs falken1 0.66 as USSR /
0.55 as US (v3 vs Greedy is 0.940); falken1 vs Greedy 0.540 over
200, even by seat. teach1 launched on the gate.

**Result, teach1 and the decider** (2026-08-30, `runs/teach1`,
`runs/playdek/teach1-easy`). The mechanism the arm predicted did not
happen in training: falken1 is *weaker* than the learner (0.395 vs
v3), so PFSP — built to keep opponents the learner loses to
prominent — faded it to **205 of ~4,000 pool games (5.1%)**, the
learner winning 0.844 of them; exploit1 stayed at 409 by beating the
learner. The internal ladder still moved, as it has for every arm:
internal gate passed (vs Greedy **0.958**, USSR edge 0.567, in
family), teach1 over v3 **0.580** (0.51 US / 0.65 USSR) and over
falken1 **0.655** — the learner defends falken1's attack better than
v3 does (0.720 vs 0.620 on the USSR seat). Then the decider: easy
AI, 120 games, seeds 300+, USSR **1/54 = 0.019** [0.00, 0.10], US
**0/55 = 0.000** [0.00, 0.07], mean **0.009** — not only under the
0.136 bar but under raw v3's 0.086. Mechanism metric worse, not
halved: the USSR DEFCON-loss share is 31/53 = **0.58** (raw v3 ~0.4,
counter1 0.56), the US mix 32 blowouts + 10 Europe-control of 55,
mean final turn 4.4 both seats. Batch quality normal: 11/120
desyncs (the standing 7–14 band), void 0, known families only
(granted-Ops 66×, Defectors-event-first 21×, Junta 19×; the fatals
are the parked decision-mismatch and end-of-game shapes).

**Decision.** The question closes **negative** — the fifth
internal-transfer negative, and the sharpest: training against the
teacher's *imitation* taught the learner to beat a 0.395-strength
copy of the AI's habits, and what it learned is punished harder by
the real thing (the counter1 story again: internal wins bought USSR
aggression the AI converts into DEFCON deaths). Two named reasons
the arm's construction, not its premise, may be at fault: the clone
is far below teacher strength (0.610 top-1 on determinized hands),
and PFSP structurally fades a weaker teacher — a fixed-share anchor
slot or a stronger clone (more corpus, DAgger against the live DLL)
would each need a new entry. The other relaxation — live DLL games
as sparring — stays priced at 30–50 minutes a game. **Raw v3
remains the reported player**; the distill tooling and the corpus
stay aboard.

### 2026-08-31 — the teacher as prior: the falken1-init line (pre-registered)

**The review decision** (user, 2026-08-31): the reconstructed teacher
arm, built on the lever no arm has touched. Every transfer negative —
gshare, the exploiter, counter1, teach1 — changed the *opponent
distribution* around a v3-lineage prior and watched the improvement
stay internal. This arm changes the **prior**: the learner starts as
the teacher's student — initialized from falken1 — and trains by
ordinary self-play, with no teacher in the mix at all. One lever, so
whatever moves is attributable. (DAgger proper is off the table: the
DLL cannot be set to an arbitrary state and asked what it would do —
it only plays its own games. A stronger clone and a fixed-share
anchor remain separate future constructions.)

**Question.** Does a learner that begins with the easy AI's habits —
the distilled prior, gift-punishing included — and self-plays from
there transfer to the real AI where five opponent-distribution arms
did not?

**Setup.**

1. *The clone probe* (first DLL spend, diagnostic not gate):
   falken1 itself against the easy AI — 40 games, 20 a seat, seeds
   300+, bid 2, argmax (`runs/playdek/falken1-easy`). It sizes the
   fidelity suspect directly: a clone near 0 against its own teacher
   is a caricature and the notebook should know; a clone that holds
   its own relocates the transfer problem. It also calibrates what
   internal strength means for this lineage (falken1 is 0.540 vs
   Greedy; v3's 0.940 buys 0.086 vs the AI).
2. *teach2*: `--run teach2 --init runs/falken1/joshua.pt --games
   8000 --recipe v11 --bid 2 --workers 8 --eval-every 500
   --eval-games 200` — recipe v11 unchanged (0.5 self-play, 0.5 its
   own PFSP pool), no falken slot, no scenario bank. The clone's
   value head was never trained, so the first updates run on a
   value estimate that must relearn; accepted, not patched. The
   Greedy eval curve is on for the extension rule below.

**Metrics and decision rule** (written before training).

- *Retention gate, before DLL spend*: the exploit-gate harness with
  teach2 as the punisher (`runs/falken1/gate.py`, 100 games a seat,
  argmax, bid 2): of v3's USSR-seat losses to teach2, the
  DEFCON-loss share must be **≥ 0.15** — the student must still
  punish the gift after 8k games of self-play. Below it, self-play
  washed the teacher out and the premise fails; the arm closes with
  no DLL spend and that washout is the finding.
- *Strength gate, before DLL spend*: teach2 vs Greedy (the training
  curve's last rolling mean, or `wopr.diagnose`) ≥ **0.75** both
  seats' mean — not v3's 0.9 bar (this is not a v3 continuation) but
  the bootstrap target, a floor of credibility for reading a DLL
  batch. If it is missed while the Greedy curve is still climbing,
  one extension to 16k games is pre-authorized (the exploit1 budget
  lesson); missed flat, the arm closes.
- *Decider*: the standing easy eval (120 games, 60 a seat, bid 2,
  argmax, seeds 300+) — **mean ≥ 0.136** makes teach2 the reported
  player. Reported beside it: teach2 against the probe's falken1
  number (did the student beat the teacher on the teacher's own
  exam?), the USSR DEFCON-loss share (raw v3 ~0.4, teach1 0.58),
  the US blowout share, mean final turn. Desyncs mined first, as
  always. Internal context, not gates: teach2 vs v3, vs falken1.

**Budget.** The probe (40 games, ~50 min DLL). teach2 8k games
(~50 min), one pre-authorized extension to 16k. The head-to-heads.
One decider eval (120 games, ~140 min DLL). Nothing else without a
new entry.

**Result, the clone probe** (2026-08-31,
`runs/playdek/falken1-easy`): **0/39** (USSR 0/19, US 0/20), mean
final turn 4.2, 19 of the 39 losses by DEFCON — the clone dies by
the very mechanism it punishes in v3, 1 desync, void 0. The
fidelity suspect is confirmed: distillation carried the teacher's
habit shapes (0.610 top-1) and none of the 15 s search that makes
them safe. Calibration: the internal-vs-AI correlation is now
broken in both directions — v3 at 0.940 Greedy buys 0.086, teach1
at 0.958 buys 0.009, falken1 at 0.540 buys 0.000. Probe is
diagnostic, not gate: teach2 continues under the standing rule.

**Result, teach2 and the decider** (2026-08-31, `runs/teach2`,
`runs/playdek/teach2-easy`). Both gates passed: *retention* — of
v3's USSR-seat losses to teach2 the DEFCON share is **0.60** (21/35;
falken1's own 0.50) — 8k games of self-play kept and sharpened the
teacher's habit; *strength* — vs Greedy **0.817** at argmax (bar
0.75; the training curve climbed 0.540 → 0.805, dipping to 0.43
around 2k while the untrained value head relearned). Internal
context: teach2 vs v3 0.335, vs its own prior falken1 only 0.510 —
the Greedy number rose without the internal ladder following. The
decider: USSR **3/55 = 0.055** [0.02, 0.15], US **1/58 = 0.017**
[0.00, 0.09], mean **0.035** — the bar (0.136) missed, raw v3's
0.086 not reached. Desyncs 7/120, void 0, known families only.

**And the first mechanism win of the program**: the USSR-seat
DEFCON-loss share is **7/52 = 0.135** — raw v3 ~0.4, counter1 0.56,
teach1 0.58 — at the *same* mean game length as teach1's batch
(turn ~4.5), so it is not an artifact of dying earlier by other
means. The gift-blunder class the search arc, the scenario prior
and two counter-runs could not touch is gone in the student. What
replaced it is uniform positional weakness: 44/52 USSR-seat and
46/57 US-seat losses are ≥20-VP track blowouts.

**Decision.** The question closes **negative on the bar** — the
sixth transfer negative — but unlike the five before it, the
mechanism metric moved against the real AI for the first time: the
teacher's prior carries the gift lesson through self-play and onto
Playdek's board. The trade was strength: the student starts 0.395
below the champion and 8k games did not close that (still climbing
at the cap). The evidenced follow-on — continue the teach2 line
long past 8k and re-measure whether strength recovers while the
mechanism holds — needs a new entry and is the user's call. Raw v3
remains the reported player.

### 2026-08-31 — the student trains on: teach2 to 32k (pre-registered)

**The review decision** (user, 2026-08-31): the extension the parent
arm's own data points at. teach2 was still climbing at its 8k cap
(last Greedy tick 0.805) and its mechanism number is the program's
first — the question is whether the two can coexist at scale.

**Question.** Does continued self-play on the falken1-init line
recover strength while the inherited gift lesson holds — or does
the lesson decay with training distance from the prior?

**Setup.** Resume `runs/teach2` in place to **32,000 games** — one
segment, recipe v11 unchanged, bid 2, the Greedy eval curve on
(`--eval-every 500 --eval-games 200`), no other flag touched. No
early stop: the exploit1 budget lesson says read the cap, not the
dip.

**Metrics and decision rule** (written before training).

- *Mechanism-hold gate, before DLL spend*: the retention harness
  (`runs/falken1/gate.py`, teach2@32k as punisher, 100 a seat,
  argmax, bid 2) — v3's USSR-seat DEFCON-loss share **≥ 0.15**.
  Washout at scale closes the arm with no DLL spend, and *is* the
  finding: the prior's lesson decays with training distance.
- *Credibility floor, before DLL spend* (inherited): vs Greedy ≥
  0.75 by `wopr.diagnose` — already passed at 8k, must not regress.
  The decider is otherwise authorized regardless of internal
  strength: six arms say internal numbers do not predict the AI
  number, and this line's question is the mechanism and the trend.
- *Decider*: the standing easy eval (120 games, 60 a seat, bid 2,
  argmax, seeds 300+). Read in order: **mean ≥ 0.136** → teach2 is
  the reported player. Mean **> 0.086** (raw v3's standing) with the
  mechanism held (USSR DEFCON-loss share ≤ 0.25) → the line
  continues on review — strength is converting. Mean at ~0.035 with
  the mechanism held → the lesson survives but self-play strength
  does not convert; the line parks. Mechanism lost → the line
  closes. Reported beside: the trend from 8k's 0.035, the US blowout
  share, mean final turn, teach2@32k vs v3 and vs falken1 (context).
  Desyncs mined first, as always.

**Budget.** 24k more games (~2.5 h). The retention harness, a
diagnose, one head-to-head batch. One decider eval (120 games,
~140 min DLL). Nothing else without a new entry.

**Result** (2026-08-31, `runs/teach2` at 32k,
`runs/playdek/teach2-32k-easy`). Both gates passed: mechanism-hold
**0.562** (18/32 of v3's USSR-seat losses to teach2@32k are DEFCON
deaths — no decay at all from 8k's 0.60), Greedy **0.792** (floor
0.75, down from 8k's 0.817). But the internal curve told the story
before the DLL did: the Greedy evals plateaued at 0.66–0.79 for the
whole extension, teach2 vs v3 moved 0.335 → 0.360 and vs its own
prior falken1 *fell* 0.510 → 0.415 — 24k games churned, not
climbed. The decider: USSR **1/55 = 0.018**, US **2/58 = 0.034**,
mean **0.027** [0.01, 0.08] — flat with 8k's 0.035, raw v3's 0.086
not reached. The mechanism **held on the board**: the USSR
DEFCON-loss share is 10/54 = **0.185** (8k: 0.135; raw v3 ~0.4;
teach1 0.58), losses still dominated by ≥20-VP track blowouts
(42/54 USSR, 36/56 US) at mean turn ~4.2. Desyncs 7/120, void 0,
known families only.

**Decision.** The pre-registered third branch: **the line parks.**
The inherited gift lesson survives 32k games of self-play — the
first durable mechanism transfer in the program — but self-play on
this prior does not convert it into strength: the falken1-init
policy sits in a basin (~0.36 vs v3, ~0.79 vs Greedy) that 24k
games did not leave. What the two teacher arms establish together:
the *prior* lever moves mechanisms where five opponent-distribution
levers moved nothing, and the missing half is champion-level
strength on top of the inherited lesson. The evidence-pointed next
construction — a new entry, the user's call — is to put the lesson
*into the champion* rather than strength into the student:
v3-init with an auxiliary distillation loss toward the harvested
corpus (kickstarting), the corpus already on disk. Raw v3 remains
the reported player.

### 2026-08-31 — kickstarting: the lesson into the champion (pre-registered)

**The review decision** (user, 2026-08-31): invert the teacher arms'
direction. teach2 proved the corpus carries a transferable lesson
(the gift class, held through 32k games) but self-play from the
clone prior cannot buy strength; v3 has the strength and not the
lesson. This arm trains **kick1**: v3-init, ordinary recipe-v11
self-play, plus an **auxiliary distillation pull toward the
harvested corpus** — kickstarting. Constructed as *interleaved*
cross-entropy steps on the policy's own optimizer (a few corpus
minibatches after every PPO update, gradients clipped like PPO's),
not a joint loss: SB3's update stays untouched and the dose is two
flags. The corpus is the one already on disk (265,683 rows,
`runs/falken1/corpus`); its held-out fold is never trained on and
doubles as the absorption metric.

**Question.** Can the champion absorb the teacher's decision
distribution while self-play maintains its strength — and does the
combination finally move the Playdek number?

**Setup.** `--run kick1 --init baselines/r3-bid2/v3/joshua.pt
--games 8000 --recipe v11 --bid 2 --kickstart runs/falken1/corpus
--kickstart-coef 1.0 --kickstart-batches 4 --kickstart-batch-size
512` — about 4×512 CE rows per update, ~2.7 corpus epochs
interleaved across the run's ~320 updates. New wiring:
`KickstartCallback` (wopr/callback.py), corpus plumbing shared with
`wopr.distill`, plus a `wopr.distill top1` subcommand measuring a
checkpoint's held-out corpus top-1.

**Metrics and decision rule** (written before training).

- *Absorption* (reported, not a gate): held-out corpus top-1 —
  raw v3 before (baseline, measured pre-launch), kick1 after;
  falken1's 0.610 is the pure-BC reference.
- *Strength gate, before DLL spend* (counter1's bar — this is a v3
  continuation): `wopr.diagnose` vs Greedy ≥ **0.9** (v3: 0.940),
  no USSR-edge collapse. A kickstart dose that costs the champion
  its strength fails here cheaply; the dose, not the idea, is the
  first suspect and a re-dose needs only a new entry line.
- *Mechanism probe, before DLL spend* (reported): kick1 as USSR vs
  falken1, 100 games argmax fixed seats — the DEFCON share of
  kick1's USSR-seat losses against v3's 0.50 baseline (falken1
  punishes gifts; a champion that absorbed the lesson should stop
  handing it the coup).
- *Decider*: the standing easy eval (120 games, 60 a seat, bid 2,
  argmax, seeds 300+). **Mean ≥ 0.136** → kick1 is the reported
  player. Mean > 0.086 with the mechanism moved (USSR DEFCON-loss
  share ≤ 0.25 vs raw v3's ~0.4) → the construction works and the
  line continues on review (dose up, longer run). Mean flat with
  the mechanism moved → the lesson lands but does not convert even
  at champion strength — a deep negative worth the notebook.
  Mechanism unmoved → the dose was too light against PPO's
  gradient; re-dose is the follow-on. Desyncs mined first.

**Budget.** The wiring + tests. 8k games (~1 h with the interleaved
steps). diagnose, the mechanism probe, two `top1` measurements. One
decider eval (120 games, ~140 min DLL). Nothing else without a new
entry.

**Result** (2026-08-31→09-01, `runs/kick1`,
`runs/playdek/kick1-easy`). The construction did what it promised
internally: absorption **0.335 → 0.507** held-out top-1 (falken1's
pure-BC 0.610 the reference) at **zero strength cost** — the Greedy
curve rode 0.96–0.995 all run, diagnose 0.958 (v3: 0.940), gate
passed with one flag (USSR edge 0.708; v3 0.575). The mechanism
probe was already equivocal: kick1-as-USSR beats falken1 0.81
(v3: 0.66) with the absolute gifted-death rate down (13/100 vs
17/100) but the share of losses up (0.684). The decider: USSR
**8/56 = 0.143** [0.07, 0.26] — the best single-seat number ever
measured against the easy AI — US 2/57 = 0.035, mean **0.088**,
exactly raw v3's standing 0.086; games noticeably longer (mean turn
5.6 vs teach1/2's ~4.3). But the mechanism did not move where it
counts: the USSR DEFCON-loss share is 29/48 = **0.604** (raw v3
~0.4) — kick1 absorbed the teacher's choices on the corpus and
still hands the AI the gift on Playdek's board. Desyncs 7/120,
void 0, known families only.

**Decision.** Negative on the bar — but not the pre-registered
"dose too light" branch: absorption was real (0.507), so the dose
landed and was then *selectively unlearned*. The coherent reading
across the three teacher arms: **a lesson survives training only
where self-play reward agrees with it.** teach2 kept the gift
lesson through 32k games because its pool — its own falken-descended
snapshots — punishes gifting; kick1's v3-lineage pool never
punishes it, so PPO's gradient quietly reverses the pull exactly in
the reward-shadowed states while keeping it everywhere else (hence
0.507 absorption, longer games, the best USSR seat yet, and an
unmoved gift share). Kickstarting itself is cheap, safe, and
strength-preserving — the missing piece is a training signal that
*prices the gift*: the kickstart pull plus an opponent that punishes
it, held at a fixed share PFSP cannot fade (falken1 as anchor), or
gift-scenario starts against punishing opponents. That combined
construction is a new entry, the user's call. Raw v3 remains the
reported player — kick1 at 0.088 mean ties, not beats, the
standing number, though its USSR seat is the line to watch.

### 2026-09-01 — kick2: the kickstart pull plus an anchor that prices the gift; and the veto rider (pre-registered)

**The review decision** (user, 2026-09-01): test the three-arm theory
head-on. kick1 established that the champion absorbs the corpus
(0.507 held-out top-1) at zero strength cost, and that PPO reverses
the pull exactly where self-play reward never prices it (the gift
share stood at 0.604). The theory — *a lesson survives training only
where self-play reward agrees with it* — makes a prediction: put an
opponent that punishes the gift into the reward stream at a share
the sampler cannot fade, and the same dose sticks. One entry, two
questions.

**Question 1 (kick2).** kick1's construction with the reward stream
amended: v3-init, recipe v11, bid 2, 8k games, the same kickstart
dose — plus **falken1 held at a fixed ~10% anchor share**. The pool
fraction drops 0.5 → 0.4 and the freed remainder goes to the anchor
slot, which nothing reweights: teach1's PFSP faded the clone to 5.1%
of pool games once the learner beat it; an anchor's share is the
mix's, structural, for the whole run. Does the gift lesson now
survive on the board — and does it convert on Playdek's?

**Question 2 (the veto rider, zero training).** The terminal-probe
veto (2026-08-25 entry) wrapped around kick1's checkpoint in the
standing easy eval. kick1 owns the best USSR seat ever measured
(0.143) with the gift class intact (share 0.604) — exactly the loss
class the veto refuses. v3+veto's standing numbers: USSR 0.154 / US
0.038 / mean 0.096. Does inference-time suppression stack with the
absorbed lesson, or do the two mechanisms overlap?

**Setup.**

- New wiring first: `--anchor` learns checkpoint anchors — a
  `name=path` element resolves to policy id `ckpt:<path>`, which
  `StandardOpponents` answers with a sampling `NetOpponent`, the way
  a pool snapshot plays; the checkpoint is loaded once up front so a
  bad path or layout fails before training. A single anchor never
  promotes, so its share is fixed. (`wopr/train.py`,
  `wopr/opponents.py`; mechanics in WOPR.md.)
- kick2: `--run kick2 --init baselines/r3-bid2/v3/joshua.pt --games
  8000 --recipe v11 --bid 2 --vs-pool 0.4 --anchor
  falken1=runs/falken1/joshua.pt --kickstart runs/falken1/corpus
  --kickstart-coef 1.0 --kickstart-batches 4 --kickstart-batch-size
  512` — kick1's flags plus the two changed ones, kick1's machine
  settings.
- Rider: `wopr.playdek.eval --difficulty easy --games 120 --seed 300
  --bid 2 --policy veto=runs/kick1/joshua.pt --out
  runs/playdek/kick1-veto-easy` — the standing decider's shape,
  rated as its own named policy (`kick1+veto`), never silently
  substituted (the search entry's rule).

**Metrics and decision rule** (written before anything runs).

kick2, internal (before DLL spend):

- *Absorption* (reported): held-out corpus top-1 (`wopr.distill
  top1`) — v3 0.335 and kick1 0.507 the references, falken1's 0.610
  the pure-BC ceiling. Expected near kick1's; a large drop says the
  anchor games fought the pull instead of pricing it.
- *Anchor curve* (reported, new signal): `win_rate_vs_anchor` in
  metrics.csv — the learner's live record against the punisher.
- *Strength gate* (kick1's): `wopr.diagnose` vs Greedy ≥ **0.9**, no
  seat collapse. Fails → the anchor share or the dose is the
  suspect; a re-dose is a new entry line, not a new idea.
- *Mechanism probe* (gate.py, kick2 as gifter vs falken1, 100 a seat
  argmax): gifted deaths as USSR against kick1's 13/100 and v3's
  17/100 — expected materially down if the anchor priced the gift
  during training.

Decider: the standing easy eval (120 games, 60 a seat, bid 2,
argmax, seeds 300+), desyncs mined first. **Mean ≥ 0.136** → kick2
is the reported player. The key read regardless of the bar: **USSR
DEFCON-loss share ≤ 0.25** (raw v3 ~0.4; kick1 0.604). Readings:

- Share ≤ 0.25 **and** mean over v3's 0.086 → the theory holds and
  the reward-priced kickstart is the program's first transferring
  construction; continue the line (dose, share, games) on review.
- Share ≤ 0.25, mean flat → the lesson lands at champion strength
  and *still* does not convert — the deep negative worth the
  notebook: the gift was never the whole gap.
- Share unmoved → the theory takes real damage: an unfadeable
  punisher at 10% plus the pull cannot beat PPO's gradient. The
  named reserves — gift-scenario starts (`defcon2_gift` bank) or a
  larger anchor share — are each a new entry.

Rider: mean ≥ **0.136** → kick1+veto becomes the standing reported
player (as a named policy). Below the bar, the attribution is the
USSR seat against kick1's 0.143 and v3+veto's 0.154: if absorbed
lesson and rules-probe suppress the same losses, the stack adds
nothing (≈0.15 again); if they add, the seat moves past both. The
gift share of the remaining USSR losses is expected ≈0 mechanically
(the probe refuses provable losses) — reported, not a finding.

**Budget.** The anchor wiring + tests. 8k games (~1 h). diagnose,
top1, the 100-game probe. Two decider evals (120 games each, ~2.5 h
DLL each), rider first — it needs no training and its player already
exists. Nothing else without a new entry.

**Result** (2026-09-01, `runs/kick2`, `runs/playdek/kick2-easy`,
`runs/playdek/kick1-veto-easy`). Every internal gate passed.
Absorption **0.505** — the anchor did not fight the pull (kick1
0.507). Strength gate 0.983 vs Greedy (kick1 0.958, v3 0.940), USSR
self-play edge 0.633 — kick1's 0.708 flag gone. The mechanism probe
moved hard: kick2-as-USSR beats falken1 **0.90** (kick1 0.81, v3
0.66) with gifted deaths down to **6/100** (kick1 13, v3 17) — on
its own training distribution the anchor priced the gift. The
anchor curve rode 0.6–1.0 all run; the Greedy eval curve 0.94–0.99.

The deciders, both over the bar — the program's first, and second,
positive transfers:

| | USSR | US | mean | USSR gift share |
| --- | --- | --- | --- | --- |
| raw v3 (standing) | 0.093 | 0.078 | 0.086 | ~0.4 |
| kick1 (raw) | 0.143 | 0.035 | 0.088 | 0.604 |
| v3+veto (2026-08-26) | 0.154 | 0.038 | 0.096 | — |
| **kick2 (raw)** | **0.190** [.11,.31] | 0.089 [.04,.19] | **0.140** [.09,.22] | **0.489** |
| **kick1+veto** | **0.278** [.18,.41] | **0.218** [.13,.34] | **0.248** [.18,.34] | **0.179** |

kick2: mean **0.140 ≥ 0.136** — the first *raw checkpoint* over the
bar, on the best raw USSR seat yet (0.190). The key read is
half-moved: gift share **0.489** against kick1's 0.604 and the
≤ 0.25 threshold — the 10% anchor recovered part of the lesson on
Playdek's board (and nearly all of it on its own distribution, the
probe's 6/100) but did not clear the read. The US seat sits at v3's
level (0.089 vs 0.078); its loss mix keeps kick1's inherited ~0.35
US-side DEFCON-death share — the mirrored blunder the anchor was
never aimed at. Desyncs 6/120, void 0, known families only.

kick1+veto: mean **0.248** [0.18, 0.34] — the bar cleared outright,
the interval's floor above every point estimate the program has
ever reported, and **both** seats lifted: USSR 0.278 (kick1 raw
0.143, v3+veto 0.154 — the absorbed lesson and the rules-probe
*add*, they do not overlap), and US 0.218 against kick1 raw's 0.035
[.01,.12] — non-overlapping intervals, the first US-seat movement in
the program's history. The attribution: kick1's US-seat losses were
one-third DEFCON deaths (18/55) — the veto refuses exactly those,
and the seat's suppressed games reappear as wins rather than VP
blowouts (US losses under veto: 5/43 DEFCON, the rest the familiar
VP/Europe mix). USSR gift share 0.179 — the ≤ 0.25 read met, as
expected mechanically. Games longer (turn 6.5 vs kick1's 5.6).
Desyncs 11/120 (granted-Ops 6, Defectors-event-first 2, end-of-game
turbulence seed 324, Grain decision-mismatch 1, hidden-seat drift
1), void 0 — the standing band, known families only.

**Decision.** Both pre-registered promotion clauses fired; they
rank themselves. **kick1+veto (0.248) is the standing reported
player against the Playdek AI**, as a named policy per the search
entry's rule — never silently substituted for a raw checkpoint in
gates or baselines. **kick2 (0.140) is the reported raw checkpoint**,
the first to clear the bar and the strongest raw player measured.
The theory's verdict is a dose-response, not a clean confirmation:
the reward-pricing lever moved the gift share in the predicted
direction at every scale measured (probe 13→6, board 0.604→0.489)
and bought the bar, but a 10% anchor share does not push the
on-board share under 0.25 — self-play's other 90% still shadows
part of the lesson. The evidence-pointed next constructions, each a
new entry on the user's call: **veto over kick2** (the two positives
composed — if the lifts add again, both seats move from the 0.140
base), a larger anchor share or gift-scenario starts (the re-dose,
`defcon2_gift` bank), and the article, which now has a win to end
on. The bar itself should also be revisited on review: 0.136 was
set as +0.05 over raw v3 when nothing had ever moved; both new
players clear it and the next bar should be set from 0.248.

### 2026-09-01 — the compose and the confirmation: veto over kick2, the standing player re-measured, the bar re-set (pre-registered)

**The review decision** (user, 2026-09-01): run the rest of the
menu short of the article, in this order — the free compose first,
the desync push while it runs (bridge engineering, documented in
WOPR.md and the commits, not here), then the confirmation and the
re-dose. This entry covers the two eval-only questions; the re-dose
gets its own entry when its construction is fixed.

**Question 1 (the compose).** Veto over kick2. Both deciders say
kick2's loss mix is even more veto-shaped than kick1's was: 23/47
USSR-seat and 20/51 US-seat losses are DEFCON deaths, and the veto
refuses provable losses of exactly that class. If the lifts compose
the way kick1+veto's did, this is the likeliest new best number.
Zero training: `wopr.playdek.eval --difficulty easy --games 120
--seed 300 --bid 2 --policy veto=runs/kick2/joshua.pt --out
runs/playdek/kick2-veto-easy` — the standing decider's shape, rated
as the named policy `kick2+veto`.

**Question 2 (the confirmation).** The 0.248 claim rests on one
120-game batch and the program's numbers are now tight enough to
care (kick2 cleared the old bar by 0.004). Whichever of kick1+veto
and kick2+veto stands higher after question 1 is re-measured on
**fresh seeds** (120 games, seeds 500+, bid 2, same shape) — run
after the desync push so the batch pays less attrition.

**Metrics and decision rule** (written before anything runs).

- Compose: mean and per-seat rates vs kick1+veto's 0.278/0.218/0.248
  and kick2 raw's 0.190/0.089/0.140, gift share reported. The higher
  of the two veto players by mean goes to confirmation. No bar here
  — both inputs already cleared it; this ranks them.
- Confirmation: the pooled estimate over the player's two batches
  (240 games, seeds 300+ and 500+) becomes its standing number. If
  the fresh batch's mean falls below the old bar 0.136, the original
  batch was the fluke the interval allowed — the standing player
  reverts to the pooled best and the notebook says so plainly.
- The bar, re-set (a decision, applied after confirmation): the next
  training arm's promotion bar becomes **pooled standing mean +
  0.05**, the same construction that set 0.136 from raw v3's 0.086.
- Desyncs mined first in every batch, as always.

**Budget.** Two decider evals (120 games each, ~2.5 h DLL). The
desync push between them is bridge work with its own harnesses
(differ / emu / identical-seed wide batches) and no notebook entry;
its yield shows up here only as lower attrition on the
confirmation batch. Nothing else without a new entry.

**Result** (2026-09-01, `runs/playdek/kick2-veto-easy`,
`runs/playdek/kick2-veto-easy-s500`).

The compose, seeds 300+ (protocol note: this batch launched minutes
before the nineteenth bridge pass landed, so it ran the *old*
judge): USSR 16/48 = **0.333** [0.22, 0.48], US 10/49 = 0.204
[0.12, 0.34], mean **0.268** [0.19, 0.36], gift share 6/32 = 0.188
— the stack composes a second time, both seats again (kick2 raw
0.190/0.089). Attrition told the other story: **20/120 desyncs**
(band 7–14) + 3 trapped-scoring-card voids, effective n = 97 — the
compose player survives deep into games (USSR mean turn 7.0, many
to turn 10) where drift accumulates, and several fatals were
exactly the nineteenth pass's shapes (HIL ×2, SAU ×1, ops_type
deadlocks ×2, Wargames endgame ×4).

The confirmation, fresh seeds 500+, new judge: USSR 18/55 =
**0.327** [0.22, 0.46], US 10/57 = 0.175 [0.10, 0.29], mean
**0.250** [0.18, 0.34] — the compose replicates. Gift share 3/37 =
0.081. And the pass paid on the same player whose old-judge batch
hit 20: **8/120 desyncs, void 0**, known families only (game-over
timing ×3, granted-Ops-face ops_type ×3, the r10 deal-drift shape
×1, illegal-in-Playdek ×1), the drift-pick firing 3 times.

**Decision.** The pooled standing number: **54/209 = 0.258**
(USSR 34/103 = 0.330, US 20/106 = 0.189). **kick2+veto is the
standing reported player**, confirmed on two seed blocks; the fresh
batch's 0.250 sits far above the old bar, so no reversion clause
fires. The next training arm's promotion bar, per the pre-registered
construction: pooled mean + 0.05 = **0.308**. kick2 stays the
reported raw checkpoint.

### 2026-09-01 — kick3: the re-dose — gift-scenario starts on kick2's construction (pre-registered)

**The review decision** (user, 2026-09-01, same review as the compose
entry): the re-dose runs after the desync push, before the article.
kick2 read as dose-response: a 10% falken1 anchor moved the on-board
gift share 0.604 → 0.489 and nearly cleared the gift on its own
training distribution (probe 6/100) — the reward-pricing lever works
where training reaches the gift states, and mostly it does not reach
them. The reserve named in kick1's entry is exactly that lever:
**gift-scenario starts** — training games opened from harvested
DEFCON-2/gift-in-hand positions, so every mix component (self-play,
pool, the anchor) visits the states where the lesson keeps being
unlearned. The two priors isolate each lever alone: scen1
(2026-08-28) ran scenario starts with a non-punishing pool and moved
nothing (0.081, loss mix unmoved); kick2 ran the punisher without
the states (0.140, share 0.489). kick3 is the combination, one new
lever on kick2's standing construction.

**Question.** Does concentrating training on the gift states, with
the kickstart pull and the fixed punisher aboard, push the on-board
gift share under 0.25 without costing the champion's strength?

**Setup.** kick2's flags plus the bank: `--run kick3 --init
baselines/r3-bid2/v3/joshua.pt --games 8000 --recipe v11 --bid 2
--vs-pool 0.4 --anchor falken1=runs/falken1/joshua.pt --kickstart
runs/falken1/corpus --kickstart-coef 1.0 --kickstart-batches 4
--kickstart-batch-size 512 --scenarios scenarios/defcon2-gift-v3.jsonl
--scenario-frac 0.25`. No new wiring: scenario starts are drawn
independently of the seat mix, so ~2.5% of games are the punisher
met *in* a gift state, and the kickstart pull covers the same states
from the corpus side. Evaluation stays at the printed game.

**Metrics and decision rule** (written before training).

- *Gates before DLL spend* (kick2's): absorption reported
  (`wopr.distill top1`; expect ≈ 0.50 — a large drop says the
  scenario games fought the pull), `wopr.diagnose` vs Greedy ≥
  **0.9**, no seat collapse; the mechanism probe (gate.py, kick3 as
  gifter vs falken1, 100 a seat) against kick2's 6/100 and the
  in-bank re-measure (`wopr.scenarios` eval on the held-out states,
  as scen1 measured) reported.
- *Decider*: the standing easy eval (120 games, seeds 300+, bid 2),
  desyncs mined first. Success = **mean ≥ 0.140** (kick2's raw
  standing — the re-dose must not buy the share by selling strength)
  **and** USSR gift share ≤ **0.25** (v3 ~0.4, kick1 0.604, kick2
  0.489). Readings: both → the theory closes positive at champion
  strength; share moved but > 0.25 → dose–response continues, the
  next dose (anchor share up, or a punisher *seated in* the scenario
  games) is a new entry; share unmoved from 0.489 → scenario×anchor
  does not add — the named suspect is that the pool's 90% still
  answers the gift states with non-punishing opponents, and the
  reserve becomes seating falken1 in the scenario games themselves.
- *The compose* (veto over kick3): measured only if raw kick3 clears
  both reads — one more 120-game batch, seeds 300+, against
  kick2+veto's numbers.

**Budget.** 8k games (~1.5 h). The gates. One decider batch (~2.5 h
DLL; + one compose batch only on a double-clear). Nothing else
without a new entry.

**Result** (2026-09-01, `runs/kick3`, `runs/playdek/kick3-easy`).
Gates: absorption 0.495 (the pull intact), diagnose 0.958 (USSR
edge 0.717 — kick1's flag level, noted), and the probe already
warned: gifted deaths **11/100** against kick2's 6 — the scenario
starts made the internal gift metric *worse*, not better. The
decider: USSR 10/54 = 0.185 [0.10, 0.31], US 7/57 = **0.123**
[0.06, 0.23] (the best raw US seat measured), mean **0.153**
[0.10, 0.23] — the strength read met (≥ 0.140, though inside
kick2's interval: a wash, not a lift) — and the key read failed
hard: gift share **29/44 = 0.659**, above kick2's 0.489 and
kick1's 0.604. Desyncs 9/120, void 0, known families (game-over
timing ×4, hidden-seat play_mode leftovers ×3, illegal-in-Playdek
×2), drift-pick ×6.

**Decision.** Negative on the key read — and informative exactly
the way the theory predicts: starting a quarter of training in the
gift states where 90% of opponents do **not** punish the gift
concentrates learning *of* the gift (scen1's null repeated, now
with enough strength around it to lift the mean anyway). The dose
that matters is punishment-in-the-states, not states alone. The
pre-registered suspect stands as the named follow-on, a new entry
on the user's call: **seat the punisher in the scenario games**
(wire scenario starts to force the anchor opponent, or raise the
anchor share with the bank). No compose batch for kick3 (the
double-clear failed). kick2 remains the reported raw checkpoint;
kick2+veto the standing player at pooled 0.258; the bar for the
next training arm is 0.308.

### 2026-09-01 — kick4: the punisher seated in the scenario games (pre-registered)

**The review decision** (user, 2026-09-01): run kick3's named
follow-on. kick3 sharpened the theory to a point: gift-*states*
without gift-*punishment* teach the gift harder (share 0.489 →
0.659), because ~90% of the opponents met in those states never
take the coup. The missing construction seats the punisher in the
scenario games themselves: every training game that opens at a
DEFCON-2/gift-in-hand state puts the learner in the gift-holder's
seat (the bank records the mover) and **falken1 in the other** — the
gift is priced in exactly the states where it keeps being unlearned.

**Question.** Does punishment delivered in the gift states push the
on-board gift share under 0.25 at champion strength?

**Setup.** New wiring first: `Arena(scenario_seats=(mover_id,
opponent_id))` — a scenario-started game seats the bank entry's
mover as the learner and the given policy opposite, overriding the
seat assigner for those games; carried through `ArenaSpec` to the
collectors; `train.py --scenario-vs-anchor` builds it from the
(single, fixed) `--anchor`. Then kick4 = kick2's flags + the bank +
the new flag: `--run kick4 --init baselines/r3-bid2/v3/joshua.pt
--games 8000 --recipe v11 --bid 2 --vs-pool 0.4 --anchor
falken1=runs/falken1/joshua.pt --kickstart runs/falken1/corpus
--kickstart-coef 1.0 --kickstart-batches 4 --kickstart-batch-size
512 --scenarios scenarios/defcon2-gift-v3.jsonl --scenario-frac
0.25 --scenario-vs-anchor`. Mix note: falken1's total share becomes
~0.25 (scenario) + 0.075 (the remainder's anchor slot) ≈ **0.32**,
and every scenario game is learner-in-the-gift-seat vs the
punisher; self-play/pool cover the remaining 0.75×0.9. Evaluation
stays at the printed game.

**Metrics and decision rule** (written before training).

- *Gates before DLL spend* (kick2/kick3's): absorption reported
  (expect ≈ 0.50), `wopr.diagnose` vs Greedy ≥ **0.9**, no seat
  collapse — a real risk this time: a third of training against a
  weak clone can cost strength, and that failure is cheap here.
  The mechanism probe (kick4 as gifter vs falken1, 100 a seat):
  the arm's internal prediction is gifted deaths **< kick2's
  6/100**; kick3's 11 was the warning that fired before its
  decider — the probe is the go/no-go mood, not a gate.
- *Decider*: the standing easy eval (120 games, seeds 300+, bid 2),
  desyncs mined first. Success = **mean ≥ 0.140** (the raw standing)
  **and** USSR gift share ≤ **0.25** (kick1 0.604, kick2 0.489,
  kick3 0.659). Readings: both → the theory closes positive; share
  under kick2's 0.489 but over 0.25 → the lever is right and the
  dose still short — dose options (scenario-frac, a stronger
  punisher) are new entries; share ≥ kick2's → seating the punisher
  in the states adds nothing over the anchor alone, and the theory's
  practical ceiling at 8k games is reached — the line's next move is
  a review, not another dose.
- *The compose* (veto over kick4): one more 120-game batch only if
  raw kick4 clears both reads; its bar as a candidate standing
  player is the pooled-number construction, **0.308**.

**Budget.** The wiring + tests. 8k games (~1.5 h). The gates. One
decider batch (~2.5 h DLL; + one compose batch only on a
double-clear). Nothing else without a new entry.

**Result** (2026-09-01, `runs/kick4`, `runs/playdek/kick4-easy`).
Gates: absorption 0.502, diagnose 0.967 (USSR edge 0.575 — v3's own
level, no flag). The probe voted no before the decider did: gifted
deaths **14/100** vs falken1 (kick2 6, kick3 11, v3 17) — the arm's
internal prediction (< 6) was dead on arrival. The decider: USSR
8/52 = 0.154 [0.08, 0.28], US 2/58 = **0.034** [0.01, 0.12], mean
**0.091** [0.05, 0.16] — under the 0.140 strength read, the US seat
back at the v3/kick1 floor — with the gift share at 20/43 =
**0.465**, statistically kick2's 0.489. Desyncs 9/120, void 1,
known families; the twentieth pass's instruments fired throughout
(140 granted-ops evidence lines; every desync game carries 1–3).

**Decision.** Negative on both reads — the pre-registered third
reading fires: seating the punisher in the gift states adds nothing
over the anchor alone, and it *costs* — a third of training against
the weak clone diluted the ordinary game's signal (the named risk;
the US seat paid it). Two arms bracket the lever cleanly now: 10%
punisher without the states = share 0.489 at strength 0.140; 32%
punisher concentrated in the states = share 0.465 at strength
0.091. The gift share has a floor near 0.45–0.5 that no falken1
dose reaches at 8k games — **the theory's practical ceiling with
this punisher is reached, and the line's next move is a review, not
another dose.** Candidate review questions, for the record: a
punisher that is actually strong (live-DLL sparring, or a deeper
clone), longer runs, the layout bump (`OPTION_VOCAB` fold +
`u2_incident` slot), or accepting the veto as the standing answer
to the gift (kick2+veto's 0.081–0.188 shares are the only ones
under 0.25 ever measured) and pointing training at the *other*
loss classes. kick2 stays the reported raw checkpoint; kick2+veto
the standing player (pooled 0.258); the bar 0.308.

### 2026-09-02 — falken2: a stronger clone for the punisher slot, and kick5 on its promotion (pre-registered)

**The review decision** (user, 2026-09-02): the gift line parked on a
bracket — 10% falken1 without the states buys share 0.489 at 0.140,
32% falken1 in the states buys 0.465 at 0.091 — and the first of the
recorded review candidates is *a punisher that is actually strong*.
Live-DLL sparring stays ruled out by the arena's shape (one game per
process, 15 s a decision); the affordable construction is a better
clone. Two facts make one worth trying now: the corpus falken1 was
distilled from predates every kick-family batch, so it never saw the
AI answering a kickstarted, anchored or veto-wrapped opponent in the
long games those players reach (USSR mean turn 7 under the veto), and
falken1 was fit at v11 capacity with no regularization, its held-out
curve still creeping (+0.004 an epoch) when patience stopped it. This
entry is a conditional two-stage arm: stage 1 builds the clone at
zero DLL hours and decides whether it is materially a better clone;
stage 2 (kick5) runs only on that promotion.

**Question 1 (falken2).** Does more of the AI — every clean easy
bid-2 game since falken1's harvest — plus capacity and regularization
produce a materially more faithful clone, scored on the same held-out
rows as falken1?

**Question 2 (kick5, conditional).** kick2's construction with the
anchor slot swapped to falken2 and nothing else changed: does the
stronger punisher lower the on-board gift share where falken1's dose
floored at 0.45–0.5?

**Setup, stage 1.**

- *Corpus.* falken1's 33 shards are kept as they are: the engine has
  not changed since they were harvested (`git log 3fd64ea..HEAD --
  src/struggler/` is empty, layout v1 unchanged), so a replay would be
  byte-identical. Added: the harvest of every easy bid-2 batch played
  since — teach1-easy, teach2-easy, teach2-32k-easy, falken1-easy,
  kick1-easy, kick1-veto-easy, kick2-easy, kick2-veto-easy,
  kick2-veto-easy-s500, kick3-easy, kick4-easy — 1,240 games, of
  which 96 desyncs and 1 void are excluded (~1,140 clean; against
  falken1's 1,853). The held-out fold keeps its construction (game
  hash mod 10), so falken1 never trained on any row of the merged
  fold and both clones are scored on identical rows. Tooling:
  `wopr.distill harvest --workers` (parallel replay; the shards are
  the same rows in the same order).
- *The sweep* (three fits on the merged corpus, CPU, concurrently;
  `runs/falken2/{a,b,c}`):
  - **A** — falken1's recipe on the new corpus: hidden 256, 2 GNN
    layers, option head 128, Adam 3e-4, batch 512, patience 3, cap 30
    epochs. The data effect alone.
  - **B** — hidden 384, 3 GNN layers, option head 256, AdamW with
    weight decay 0.01, label smoothing 0.05, the learning rate halved
    on every epoch without a new best, patience 3, cap 30. Capacity
    plus regularization.
  - **C** — A's architecture with B's regularization. Separates the
    two levers.
  New flags in `wopr.distill train`: `--gnn-layers`,
  `--option-hidden`, `--weight-decay`, `--label-smoothing`,
  `--lr-decay` (mechanics in WOPR.md).

**Metrics and promotion rule, stage 1** (written before anything runs).

- *Fidelity* (the decider of stage 1): held-out top-1 on the merged
  corpus's fold, `wopr.distill top1`, for falken1 and each candidate
  — same rows, so the comparison is exact. falken1's 0.610 on the old
  fold is a reference, not the comparator.
- *The exploit gate* (`runs/falken1/gate.py 100 0 <candidate>`): the
  DEFCON share of v3's USSR-seat losses against the candidate ≥ 0.15
  (falken1 0.50). The punisher must still punish the gift; a more
  faithful clone that stopped taking the coup is no use in the slot.
- *Reported, not gating:* candidate vs falken1 head-to-head
  (`wopr.eval`, 400 games, bid 2, argmax), candidate vs Greedy, and
  v3's win rate vs the candidate (falken1: v3 wins 0.66 as USSR /
  0.55 as US). The clone-vs-its-teacher probe (falken1 0/39) is not
  repeated; the decider is the test.
- **Promotion:** the best candidate by held-out top-1 becomes
  **falken2** (`runs/falken2/joshua.pt`) if its top-1 is at least
  **falken1's on the same fold + 0.02** and the exploit gate passes.
  The fold is ~43k rows (a standard error near 0.0024), so 0.02 is
  far outside noise; it is also five of falken1's late-epoch
  increments — the smallest step that makes a different clone rather
  than a re-seeded falken1. Below it the line closes at zero DLL
  hours: the clone's ceiling is its construction (behavior cloning of
  a 15 s search under a determinized hand), not its corpus, and the
  review moves to the remaining candidates.

**Setup, stage 2 (kick5), only on promotion.** kick2's flags with one
change — the anchor: `--run kick5 --init baselines/r3-bid2/v3/joshua.pt
--games 8000 --recipe v11 --bid 2 --vs-pool 0.4 --anchor
falken2=runs/falken2/joshua.pt --kickstart runs/falken1/corpus
--kickstart-coef 1.0 --kickstart-batches 4 --kickstart-batch-size 512`.
The kickstart pull stays on falken1's corpus so that the arm moves one
lever (the punisher's strength) and its absorption number stays
comparable to kick2's 0.505. No scenario starts (kick3/kick4 closed
those). Evaluation at the printed game.

**Metrics and decision rule, stage 2.**

- *Gates before DLL spend* (kick2's): absorption on falken1's corpus
  (expect ≈ 0.50), `wopr.diagnose` vs Greedy ≥ **0.9**, no seat
  collapse; the anchor curve (`win_rate_vs_anchor`) reported — a
  stronger anchor should read lower than kick2's 0.6–1.0. The
  mechanism probe, twice: kick5 as gifter vs **falken1** (the
  comparable read: kick2 6/100, kick3 11, kick4 14, v3 17) and vs
  falken2 (reported).
- *Decider*: the standing easy eval (120 games, seeds 300+, bid 2,
  argmax), desyncs mined first. Success = **mean ≥ 0.140** (kick2's
  raw standing) **and** USSR gift share ≤ **0.25** (kick2 0.489,
  kick4 0.465). Readings: both → the theory closes positive at
  champion strength and the punisher's strength was the missing dose;
  the compose runs. Share under 0.465 but over 0.25 → the punisher's
  strength is a lever and the dose is still short — the next dose is
  a new entry. Share at or above 0.465 → the clone's strength was not
  what floored the share; the floor belongs to the construction (8k
  games of PPO against a mostly non-punishing pool), and the review
  moves to longer runs, the layout bump, or accepting the veto as the
  gift's answer.
- *The compose* (veto over kick5): one 120-game batch, seeds 300+,
  only if raw kick5 clears both reads; its bar as a candidate standing
  player is **0.308** (the pooled construction). A single batch that
  clears it is re-measured on seeds 500+ before it stands, as the
  compose entry did.

**Budget.** Stage 1: the harvest (~1 h CPU, parallel), the sweep (~5 h
CPU, the three fits concurrent), the gate/eval runs (~30 min); zero
DLL hours. Stage 2, only on promotion: 8k games (~1.5 h), the gates,
one decider batch (~2.5 h DLL), one compose batch only on a
double-clear (~2.5 h), one confirmation batch only if the compose
clears 0.308. Nothing else without a new entry.

**Result, stage 1** (2026-09-02, `runs/falken2/`). The harvest of the
eleven batches took under a minute with twelve workers: 1,120 clean
games, 192,842 rows (20 logs no longer replay, the falken1 harvest's
own failure class). Merged corpus 57 shards, 2,973 games, **458,525
rows**, held-out fold 43,731 rows; falken1 on that fold **0.6008**, so
the promotion line was **0.6208**. All three fits cleared it:

| fit | recipe | held-out top-1 | epochs | gate: v3-as-USSR DEFCON share (v3 wins) | vs falken1 | vs Greedy | v3 vs it |
| --- | --- | --- | --- | --- | --- | --- | --- |
| falken1 | hidden 256, Adam | 0.6008 | 15 (+2) | 0.50 (0.66) | — | — | 0.66 / 0.55 by seat (gate) |
| A | falken1's recipe, new corpus | 0.6341 | 20 of 24 | 0.365 (0.48) | 0.634 | 0.650 | **0.476** (US 0.385 / USSR 0.568) |
| C | A + AdamW 0.01, smoothing 0.05, lr halving | 0.6449 | 28 of 30 (cap) | 0.60 (0.55) | 0.627 | 0.645 | **0.471** (US 0.335 / USSR 0.608) |
| **B** | hidden 384, 3 GNN layers, option 256 + C's regularization | **0.6536** | 28 of 30 (cap) | 0.548 (0.69) | 0.570 | 0.625 | 0.595 (US 0.485 / USSR 0.705) |

The data alone bought +0.033 (A), regularization +0.011 more (C),
capacity +0.009 more (B); B and C both ran to the cap still creeping.
Two readings the sweep was not designed to give, recorded because they
bear on stage 2. First, **the clones beat the champion**: v3 loses
head-to-head to A and to C at bid 2 (0.476, 0.471 over 400) while both
clones sit near 0.65 against Greedy, which v3 beats at 0.94 — the
distilled AI is a specialist, strong in exactly the states where v3
blunders, weak elsewhere. Second, **fidelity and punishment invert**:
the top-1 rank is B > C > A, but v3 wins 0.69 of its USSR-seat games
against B in the gate against 0.48 vs A and 0.55 vs C, and the
punishment density — v3's DEFCON deaths per 100 games as USSR — is
C 27 > A 19 > B 17 = falken1 17. The 400-game ratings agree: v3 beats
B 0.595 while losing to A and C, and B is the weakest of the three
against falken1 and Greedy too. The most faithful clone punishes the
gift no more often than falken1 did; the smaller regularized clone
punishes it 1.6× as often. The pre-registered rule names its winner
by fidelity with the gate passed, and it is followed: **falken2 = B**
(`runs/falken2/joshua.pt`; the sweep's numbers in
`runs/falken2/falken1-top1.txt`). If kick5's gift share does not move,
the inversion is the first suspect and **kick6 with C in the slot** the
named follow-on, a new entry. Stage 2 (kick5) launched 06:15.

**Result, stage 2 — the gates** (2026-09-02, `runs/kick5`, 8k games in
~95 min). Absorption **0.5025** on falken1's corpus (kick2 0.505: the
pull intact). The anchor curve averaged 0.742 over the run, 0.779 over
its last three quarters, 0.80 at the end (kick2's rode 0.6–1.0 against
falken1 — the new anchor is not read as harder). Strength gate:
**0.992** vs Greedy at bid 2 (kick2 0.983), passed; the self-play USSR
edge **0.742** is the family's highest (kick1's 0.708 was the flag
level, kick2 0.633, kick4 0.575) — noted against the US seat. The
mechanism probe: kick5 as USSR vs falken1 wins 0.89 with gifted deaths
**8/100** (kick2 6, kick3 11, kick4 14, v3 17) — kick2's neighbourhood,
not under it — and vs its own anchor falken2 wins 0.87 with gifted
deaths **12/100**: the gift is still walked into against the very
opponent meant to price it, at twice the rate kick2 walked into
falken1's. The mood is lukewarm; the decider runs as pre-registered.

**Result, stage 2 — the decider** (2026-09-02, `runs/playdek/kick5-easy`,
120 easy games, seeds 300+, bid 2, on the twenty-first pass and rules
version 8). USSR 12/57 = **0.211** [0.12, 0.33], US 4/57 = 0.070
[0.03, 0.17], mean 16/114 = **0.140** [0.09, 0.22] — kick2's number to
the third decimal (kick2 0.190 / 0.089 / 0.140), the strength read met
on the line. The key read failed: USSR gift share **28/45 = 0.622**
(kick2 0.489, kick4 0.465, kick3 0.659), the USSR seat's losses 28
DEFCON deaths, 12 VP, 4 Wargames, 1 final scoring; the US seat's 35
VP, 10 DEFCON, 6 Europe, 1 Wargames, 1 final. Games to turn 5.8 / 5.2.
**Attrition: 4 desyncs in 120**, void 2 (both the trapped
held-scoring-card known) — under the 7–14 band for the first time,
and every fatal carries its DEFCON and VP trails (WOPR.md, the
twenty-first pass, measured).

**Decision.** Negative on the key read — the pre-registered third
reading: **the clone's fidelity was not what floored the share.** But
the arm did not test the punisher's *strength*, and the stage-1 record
says why: the rule promoted the most faithful clone, and fidelity ran
against punishment — falken2 (B) punishes v3's gift 17 times per 100
games, exactly falken1's rate, where C punishes 27 and A 19. A slot
filled with falken1's punishment density at higher fidelity bought
falken1's result. Two things are settled: more data and a bigger net
make a *more faithful* clone (+0.053 top-1), and a more faithful clone
is not a stronger punisher — the strongest punishers of the sweep were
the smaller nets, which also beat the champion head-to-head. What the
theory now needs is the arm this one was meant to be: **kick6 = kick2's
construction with C in the anchor slot** (punishment density 1.6×
falken1's; v3 loses to it 0.529), a new entry on the user's call, with
the probe's vs-anchor gifted-deaths as its go/no-go mood. The wider
review candidates stand (longer runs, the layout bump, the veto as the
gift's answer). kick2 stays the reported raw checkpoint; kick2+veto the
standing player at pooled 0.258; the bar 0.308. Budget spent: one
decider batch, no compose (the double-clear failed).

### 2026-09-02 — rules version 8: the ladder stands

Found by the twenty-first bridge pass's new instrument on its first
outing (WOPR.md): a spot-check game against the easy AI ended by
Wargames, and the fatal's VP trail read the DLL at +1 where the engine
stood at −12 after the same choice — the engine's `end_game` gave the
6 VP and then **final-scored every region**; the printed card ends the
game "immediately … (without Final Scoring)", and the DLL does. Rules
version 8 ends Wargames on the VP total as it stands after the gift
(a 0 total is a draw, as at turn 10). The compose batch's four
"Wargames endgame" desyncs (2026-09-01) were this family: a strong
player's long games reach DEFCON 2 in the Late War, where the AI
plays the card.

The decision points' re-rating on the new engine: **r3-bid2/v3 vs
Greedy 0.945** over 400 at bid 2 (W378 D0 L22; US 0.950 / USSR 0.940
— the standing 0.940, unmoved), **Greedy against itself 0.500** over
200 at bid 2 (0.52/0.48 by seat, v5's and v6's split exactly). The `r3-bid2` ladder stands, the bump noted; no
checkpoint is affected (Wargames' ending is not a layout matter).
Verified against the DLL on the new engine: the grain sweep 149/149
(desyncs 0), hotseat 8/8, the differ 12/12 zero fatals; suite 549.
Measured by the next AI batch (kick5's decider, if falken2 promotes,
carries both the pass and the bump).

### 2026-09-02 — kick6: the punisher that punishes — C in the anchor slot (pre-registered)

**The decision** (user, 2026-09-02): kick5 closed on the pre-registered
third reading without testing what it was built to test. The stage-1
rule promoted the most faithful clone, and fidelity ran against
punishment: B punishes v3's gift 17 times per 100 games, falken1's
density exactly, and kick5's share (0.622) landed where falken1's dose
had (0.489–0.659). The sweep's strongest punisher is **C**
(`runs/falken2/c/joshua.pt`: A's net — hidden 256, 2 GNN layers,
option head 128 — with AdamW 0.01, label smoothing 0.05 and the
learning rate halved on a miss; held-out 0.6449, 28 of 30 epochs). In
the exploit gate v3 as USSR loses to C **27 times per 100 games by
DEFCON** (falken1 17, A 19, B 17; the DEFCON share of v3's USSR losses
0.60, v3 wins 0.55), and over 400 rated games v3 loses to C 0.529
(US 0.335 / USSR 0.608 for v3); C beats falken1 0.627 and Greedy 0.645.
kick6 is the arm kick5 was meant to be: kick2's construction, one
lever moved — the anchor's punishment density, 1.6× falken1's.

**Question.** With the anchor slot's punisher swapped for one that
prices the gift 1.6× as often, does kick2's construction lower the
on-board gift share where falken1's density (kick2, kick4, kick5)
floored it at 0.45–0.62?

**Setup.** kick2's flags with the anchor swapped and nothing else:
`--run kick6 --init baselines/r3-bid2/v3/joshua.pt --games 8000
--recipe v11 --bid 2 --vs-pool 0.4 --anchor
falken2c=runs/falken2/c/joshua.pt --kickstart runs/falken1/corpus
--kickstart-coef 1.0 --kickstart-batches 4 --kickstart-batch-size 512`.
The kickstart pull stays on falken1's corpus (one lever; absorption
comparable to kick2's 0.505 and kick5's 0.5025). No scenario starts
(kick3 and kick4 closed those). Rules version 8, layout v1, the
twenty-first bridge pass. Evaluation at the printed game.

**Metrics and decision rule** (written before the run starts).

- *Gates before DLL spend* (kick2's): absorption on falken1's corpus
  (expect ≈ 0.50); `wopr.diagnose` vs Greedy at bid 2 ≥ **0.9**, no
  seat collapse; the anchor curve (`win_rate_vs_anchor`) reported —
  kick2 rode 0.6–1.0 against falken1, kick5 0.742 mean / 0.80 last
  against B; a punisher that is actually harder should read lower.
- *The mechanism probe, the go/no-go* (`runs/falken1/gate.py 100 0
  <punisher> <gifter>`, 100 games a seat, argmax). Two reads: kick6 as
  gifter vs **falken1** (the comparable read across the family: kick2
  6/100, kick5 8, kick3 11, kick4 14, v3 17), and kick6 as gifter vs
  **C, its own anchor** — the read kick5 gave 12/100 against its anchor
  before the 0.622 share. Because C punishes 1.6× as often as B, a raw
  count against C is not kick5's 12 on the same scale; so before kick6
  is probed, **kick2 and kick5 are probed as gifters against C** (zero
  DLL hours, minutes of CPU) to fix the scale. The rule, on gifted
  deaths vs C: **at or under kick2's count → go**; **at or above
  kick5's count → no-go**, the decider does not run and the arm closes
  at zero DLL hours on the probe (the probe called kick3, kick4 and
  kick5 before their deciders did); strictly between → the decider
  runs with the mood recorded as lukewarm, as kick5's did.
- *Decider*: the standing easy eval (120 games, seeds 300+, bid 2,
  argmax), desyncs mined first, read with
  `runs/playdek/decider_summary.py`. Success = **mean ≥ 0.140**
  (kick2's raw standing, kick5's too) **and** USSR gift share
  ≤ **0.25**. The share is read against **0.622 / 0.489 / 0.465**
  (kick5 / kick2 / kick4). Readings: both met → the theory closes
  positive at champion strength — reward pricing works and the
  punisher's *density* was the missing dose — and the compose runs.
  Share under 0.465 but over 0.25 → density is a lever and the dose is
  still short; the next dose (a larger anchor share, or two punishers
  in the slot) is a new entry. Share at or above 0.465 → the anchor's
  punishment density, at 10% of games, is not what floors the share;
  the floor belongs to the construction (8k games of PPO against a 90%
  non-punishing pool), and the review moves to longer runs, the layout
  bump, or accepting the veto as the gift's answer. Mean under 0.140
  with the share moved → the pricing cost strength; reported, no
  compose.
- *The compose* (veto over kick6, `veto=runs/kick1/joshua.pt` as in the
  standing player): one 120-game batch, seeds 300+, **only on a
  double-clear**; its bar as a candidate standing player is **0.308**
  (pooled kick2+veto 0.258 + 0.05). A batch that clears it is
  re-measured on seeds 500+ before it stands, as the compose entry did.

**Budget.** 8k games (~95 min CPU), the gates and the three probes
(~40 min CPU), one decider batch (~2.5 h DLL) unless the probe says
no-go, one compose batch only on a double-clear (~2.5 h), one
confirmation batch only if the compose clears 0.308. Nothing else
without a new entry.

**Amendment, before kick6's probe (2026-09-02, the scale read).**
kick2 and kick5 probed as gifters against C (100 games a seat, argmax;
`runs/kick6/scale-*.json`): kick2 **15/100** gifted deaths (wins 0.71),
kick5 **8/100** (wins 0.80). The scale is inverted against the board:
kick2's share was the lower (0.489 against 0.622), yet against C kick2
gifts at nearly twice kick5's rate — and kick2 gifts 2.5× as often
against C as against falken1 (6), kick5 the same against both (8). The
tri-state as written assumed kick2's count under kick5's and collapses
(8–15 would be both go and no-go). Resolved now, blind to kick6's
number: **go** only under both comparators (≤ 7/100 vs C); **no-go** at
or above the higher (≥ 15/100), the decider skipped; between (8–14) the
decider runs with the mood recorded as lukewarm, kick5's 12/100 against
its own anchor the reference. A finding in its own right: across these
two arms the 100-game probe against a punisher does not rank the board
gift share — the intervals overlap (15/100 [0.09, 0.23], 8/100 [0.04,
0.15]), so the inversion may be noise, and the probe's read is coarser
than the decider's.

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
constraint). **Amended 2026-08-30** (round-3 review, the user's
call): the easy AI's policy may enter the pool through a clone
distilled from the bridge's logged games (the entry above); live
DLL games stay evaluation-only. The first use of the relaxation
closed negative the same day (teach1: 0.009 vs the AI, the fifth
internal-transfer negative); the amendment and the tooling stand
for a differently constructed teacher arm.

1. **Search over the learned value head** — done 2026-08-26 (the
   entry above): the gift is suppressed and hard mode is measured,
   but the easy-mean bar was missed by 0.005 and the remaining losses
   are strategic — raw v3 stays the reported player, search stays a
   multiplier in waiting.
2. **Scenario-seeded self-play** — **closed negative 2026-08-28**
   (the entry above): the hist-mask diagnostic sent the arc to the
   layout-v1 line (layout v1 restored in-tree), the infrastructure
   landed (`wopr.scenarios`, `--scenarios`/`--scenario-frac`, the
   `defcon2_gift` bank), and the first run — v3 + 8k games at a 25%
   gift prior — beat v3 internally (0.605 printed / 0.640 from the
   bank states) while the easy mean stayed at raw v3's level (0.081
   vs 0.086) with the loss mix unmoved. The state prior is not the
   lever; the infrastructure stays for any future arm.
3. **Hard mode as a standing eval** — superseded 2026-08-28: the goal
   is re-anchored to **beating the easy AI first** (> 0.5 both seats,
   bid 2), and hard is not measured again until that bar is met. The
   first hard numbers (search batch, ~0.04–0.12 a seat) stand as the
   record of the mountain's size.
4. **The league exploiter** — **closed negative as constructed
   2026-08-29** (the entry above): a fresh net vs the PFSP-seeded
   champion line found a USSR-side attack (near-even vs v3 on that
   seat) but missed the 0.6 counter-training gate at the 16k cap
   while still climbing; run B never started, no DLL hours spent.
   The review (2026-08-29) ran the budget fork (exploit1 to the 32k
   cap: gate missed at worst seed 0.520, a 0.582-mean near-peer) and
   then the counter-run on it (entry above): internal gate passed,
   defense landed (0.532 vs the exploiter, 0.600 vs v3), easy eval
   **0.072 vs the 0.136 bar** with the USSR DEFCON-loss share *up*
   at 0.56 — **closed negative 2026-08-29**, the fourth
   internal-transfer negative. The self-play-only
   opponent-distribution lever is spent; the remaining rung is
   relaxing SELF-PLAY-ONLY, an explicit review decision.
5. **Bridge to <2% attrition** — the open desync families (WOPR.md),
   traces caught by volume; matters more as evals move to hard mode's
   longer games.
6. **Protocol hygiene as the numbers tighten** — more eval seeds per
   claim, Wilson bounds on every reported Playdek rate.
7. **Candidates carried:** order and recency features — **taken
   2026-08-28** (layout v2, the entry above; the bootstrap it mandates
   is running). Still unranked: a third graph layer.

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
