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

## v2

Commit `502c58e` — run `greedy-anchor`, 4,000 games trained (96 updates,
30 min; 8 torch threads, sharing the machine with profiling runs).

- Same recipe as v1 with one change: the 40% anchor games are played
  against `greedy` instead of `random`, from scratch. The question was
  whether the yardstick works as a teacher.
- Elo vs random: **+61 ± 41** over seeds 0/1/2 (v1: +378 ± 48 on the same
  protocol; Greedy rates +272 to +401 across the three seeds here).
- vs random: 0.541 (US 0.430 / USSR 0.652)
- vs first: 0.855 (US 0.710 / USSR 1.000)
- vs greedy: 0.232 (US 0.283 / USSR 0.180) — v1: 0.180
- vs v1: 0.238 (US 0.160 / USSR 0.317)
- Sampled play (seed 0): 0.757 vs random, 0.255 vs greedy, 0.472 vs v1 —
  far stronger than its own argmax line (v1's sampled play was slightly
  *weaker* than its argmax).
- Notes: a negative result, kept for the chain and the lesson. During
  training the learner won 19% of its last anchor games (v1: 76% against
  `random`), so with terminal reward only, 40% of every update's outcomes
  were a near-constant −1; what it learned came from the self-play and
  pool games. It did not learn to beat Greedy (0.23 vs v1's 0.18, within
  noise) and plays the rest worse than v1. The yardstick is not a teacher
  on its own: the anchor has to be graded (random → greedy), or the
  start has to be a policy that already wins some of those games.
  Per-seat, the USSR seat is again the stronger one everywhere.
