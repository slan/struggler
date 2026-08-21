# Baselines

Frozen results of Joshua training runs, one folder per version. `runs/`
is gitignored scratch; what is worth keeping lands here via

```sh
uv run python -m wopr.baseline vN --run <run>
```

which copies the run's `config.json`, `metrics.csv` (the training
trajectory) and `joshua.pt` (the checkpoint), plays the fixed evaluation
protocol — 200 games per opponent per seed, half on each seat, argmax
play on seeds 0/1/2 plus one sampled pass — against the anchors
(`random`, `first`, `greedy`) **and every earlier baseline**, and writes
`eval_seed_*.txt` and `summary.json`. Elo is fitted with `random` at 0,
so numbers compare across versions, and since each version's checkpoint
is kept, later versions are rated against earlier ones directly.

Add an entry below for each version: the commit, what changed, the
headline numbers. The idea behind all of it is in
[docs/JOSHUA.md](../docs/JOSHUA.md); the mechanics in
[docs/WOPR.md](../docs/WOPR.md).

## v1

Commit `be2bb86` — run `signal`, 4,000 games trained (91 updates, 26 min
on one CPU core).

- First end-to-end run of the arena: PPO, 64 games in flight, 128 learner
  decisions per game per update, 30% self-play / 30% PFSP pool / 40%
  `random` anchor, pool snapshot every 5 updates, terminal reward only,
  default network (hidden 128, 2 graph layers, card dim 32; ~270k params).
- Elo vs random: **+378 ± 48** over seeds 0/1/2 (Greedy: +628 ± 53,
  `first`: −52 ± 42 on the same protocol).
- vs random: 0.876 (US 0.790 / USSR 0.962)
- vs first: 0.872 (US 0.743 / USSR 1.000)
- vs greedy: **0.180** (US 0.170 / USSR 0.190)
- Sampled play (seed 0): 0.81 vs random, 0.165 vs greedy — most of the
  strength is already in the argmax line.
- Notes: games still end mostly on DEFCON (mean final turn ≈ 5.5); the US
  seat trails the USSR seat everywhere. Evaluating this version surfaced
  the "pending decision on a finished game" engine bug fixed in `be2bb86`.
