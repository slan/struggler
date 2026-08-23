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

## Road map

What the ladder needs first is its own ground truth; the road map is
short until it has one.

1. **Ground truth for r3.** The bootstrap freezes `r3/v1`; `wopr.diagnose`
   records the USSR edge and the end-reason mix of *this* game in the
   ledger, and the Playdek easy AI gives the first number against an
   opponent that is not ours. Then the loop, generation by generation,
   until the gate misses twice in three — that is the r3 plateau, if
   there is one, and it is measured rather than assumed.
2. **Diagnose before choosing.** At the plateau, the next experiment is
   chosen from the diagnostics, with its control, metric, budget and
   decision rule written into the ledger row before training starts
   (WOPR.md, "Decision points").
3. **Candidates carried from r1, unranked until then:** order and recency
   features (a layout bump); one-ply search over the learned value; a
   third graph layer.

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
