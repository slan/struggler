# Baselines

Frozen results of Joshua training runs: one ladder per **rules version**
of the engine (`struggler.engine.RULES_VERSION`, docs/ARCHITECTURE.md),
one folder per version inside it:

```
baselines/
  EXPERIMENTS.md   the ledger: one row per compared run, frozen or not, all rules versions
  r1/              rules version 1 (archived): v1–v16 and their README
  r2/              rules version 2 (archived): v1–v3 and their README
  r3/              rules version 3 at the printed game: v1-v8 (plateaued)
  r3-bid2/         rules version 3 under the tournament bid (US +2), the current ladder
    README.md      one entry per version: commit, what changed, headline numbers
    vN/            config.json, metrics.csv, joshua.pt, eval_seed_*.txt, summary.json
```

A version is frozen with

```sh
uv run python -m wopr.baseline vN --run <run>
```

into the ladder of the rules version the code is at. It copies the run's
`config.json`, `metrics.csv` and `joshua.pt`, plays the fixed protocol —
200 games per opponent per seed, half on each seat, argmax on seeds 0/1/2
plus one sampled pass — against the anchors (`random`, `first`, `greedy`)
**and every earlier version of the same ladder**, and writes
`eval_seed_*.txt` and `summary.json` (which records the commit and the
rules version). Elo is fitted with `random` at 0 within a ladder.

Ratings do not cross ladders: a rules change alters the game, and a
policy trained on the old one is a different kind of object on the new
one (r1's champion fell from 0.98 to 0.55 against Greedy the day the
held-scoring-card rule landed). When the rules version bumps, the
current ladder is archived as it stands and the next one starts at `v1`
from the bootstrap (`python -m wopr.bootstrap`: the recipe from scratch,
stopped by its Greedy curve — confirmed at 0.75 on both seats, a
plateau, or the cap). The process, and the decision points along it,
are in [docs/WOPR.md](../docs/WOPR.md); the notebook is
[docs/JOSHUA.md](../docs/JOSHUA.md).
