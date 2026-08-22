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

**Result.** *(pending)*

## Road map

What the ladder needs first is its own ground truth; the road map is
short until it has one.

1. **Ground truth for r2.** Freeze the first clean run as `r2/v1` if it
   is sane (plays to the late war, beats the fixed Greedy); run
   `wopr.diagnose` on it and record the USSR edge and the end-reason mix
   of *this* game in the ledger. Then the loop, generation by generation,
   until the gate misses twice in three — that is the r2 plateau, if
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
uv run python -m wopr.ab --run first --note "..."                    # clean run: recipe v11, compared, one ledger row
uv run python -m wopr.diagnose runs/first/joshua.pt                  # how its games end, its USSR edge, VP by card
uv run python -m wopr.baseline v1 --run first                        # freeze it into the current ladder
uv run python -m wopr.loop --run first --generations 3               # train, evaluate, gate, promote
uv run python src/main.py --ussr joshua --joshua-checkpoint runs/first/joshua.pt
```
