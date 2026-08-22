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

## v3

Commit `f251d6f` — run `v1-greedy`: v1's run directory (checkpoint,
optimizer, pool) continued for 4,000 more games against the `greedy`
anchor, 8,000 games in total (241 updates; the continuation took 28 min
in bf16 with 16 threads, interrupted once by the Europe-at-Control crash
fixed in `21f60cb`).

- The graded version of v2's idea: same 30% self-play / 30% pool / 40%
  anchor mix, but starting from a policy that already beat random nine
  times in ten. Arena seed 1 for the continuation, so it did not replay
  v1's decks; pool snapshots continue from `u00095`.
- Elo vs random: **+1077 ± 115** over seeds 0/1/2. Greedy rates +999 /
  +972 / +1227 in the same three fits — v3 sits 8 to 12 points above it
  in each. (The scale is stretched relative to v1's table because v3
  wins 98% against `random`; compare within a fit, not across versions'
  READMEs.)
- vs random: 0.980 (US 0.997 / USSR 0.963)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: **0.594** (US 0.703 / USSR 0.485) — v1: 0.180, v2: 0.232
- vs v1: 0.874 (US 0.948 / USSR 0.800)
- vs v2: 0.972 (US 0.963 / USSR 0.980)
- Sampled play (seed 0): 0.965 vs random, 0.450 vs greedy, 0.920 vs v1
  — argmax is the stronger line again, as with v1.
- Notes: the first version to beat the hand-written heuristic. During
  the continuation the training win rate against Greedy went 0.13 → 0.60
  and mean final turn 5.0 → 6.9 (fewer DEFCON endings). The seats have
  swapped: US is now the stronger seat against Greedy (0.70 vs 0.49) and
  the USSR seat is the one to work on. Entropy ratio fell to ~0.3 by the
  end — decisive, worth watching for collapse in a longer run.

## v4

Commit `f202a0b` — run `v1-greedy` continued from v3 for 4,000 more games
against the `greedy` anchor: 12,000 games in total, 394 updates; this
segment took 25 min (bf16, 16 threads, arena seed 2, pool carried over).

- Nothing changed but the game count. The engine it trained on includes
  the Europe-at-Control fix (`a01e08a`): scoring Europe at the Control
  tier now ends the game.
- Elo vs random: **+1529 ± 44** over seeds 0/1/2; Greedy rates +1055 to
  +1085 in the same fits, v3 +1176 to +1252.
- vs random: 0.997 (US 0.993 / USSR 1.000)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: **0.915** (US 0.873 / USSR 0.957) — v3: 0.594
- vs v1: 0.947 (US 0.947 / USSR 0.947)
- vs v2: 0.997 (US 0.997 / USSR 0.997)
- vs v3: 0.843 (US 0.812 / USSR 0.873)
- Sampled play (seed 0): 0.840 vs greedy, 0.812 vs v3 — argmax stronger
  again, but by less than for v3.
- Notes: training win rate against Greedy went 0.60 → 0.77 (sampled)
  over the segment; entropy ratio held at ~0.27 of ln K throughout (no
  collapse), explained variance ~0.8. The USSR seat, v3's weak one, is
  now the stronger against Greedy. Mean final turn fell from 6.9 to 5.7:
  it wins earlier, not dies earlier (0.997 against random). Greedy is
  close to saturated as a yardstick — at 0.92 the anchor games are
  turning into the constant +1 that v2's were a constant −1 — so from
  here the measure is the earlier versions, and the anchor share is
  the knob to revisit.

## v5

Commit `bedcbc8` — run `pure`: a fresh run of 8,000 games (268 updates,
44 min while sharing the machine with a second run) with **no anchor at
all** — 50% self-play, 50% PFSP pool, and nothing else. The thesis run:
Greedy and random are evaluators here, never opponents.

- Elo vs random: **+1156 ± 30** over seeds 0/1/2; Greedy rates +971 to
  +1140 in the same fits, v3 +1122 to +1350, v4 +1322 to +1426.
- vs random: 0.987 (US 0.993 / USSR 0.980)
- vs first: 0.998 (US 0.997 / USSR 1.000)
- vs greedy: **0.661** (US 0.733 / USSR 0.588) — never seen in training
- vs v1: 0.953 (US 0.963 / USSR 0.943)
- vs v2: 0.987 (US 0.983 / USSR 0.990)
- vs v3: 0.489 (US 0.652 / USSR 0.327) — same game count; v3 spent half
  of its games against anchors
- vs v4: 0.225 (US 0.220 / USSR 0.230)
- Sampled play (seed 0): 0.598 vs greedy, 0.485 vs v3, 0.343 vs v4.
- Comparison, not frozen: `sched`, a fresh 8,000-game run with the v1
  mix and a scheduled anchor (`--anchor random,greedy`, promoted at
  3,437 games once the learner won 75% of its last 100 random games).
  Single-seed argmax against the same field: 0.557 vs greedy, 0.350 vs
  v3, 0.080 vs v4, and 0.312 against v5 head to head. The anchor bought
  nothing the pool did not already provide.
- Notes: training never saw an external opponent, yet the argmax line
  beats the hand-written heuristic two games in three and matches v3.
  Entropy ratio ended at 0.28 of ln K, explained variance 0.8, mean
  final turn 6.6 — the same shape as the anchored runs. The USSR seat
  trails (0.33 against v3 as USSR, 0.65 as US).

## v6

Commit `789cfb9` — run `pure`, 12,000 games trained.

- Loop generation 1: v5 continued for 4,000 games; gate 0.55 cleared at 0.820 (worst seed) against v5.
- Elo vs random: **+1425 ± 83** over seeds [0, 1, 2]
- vs random: 0.995 (US 0.993 / USSR 0.997)
- vs first: 0.998 (US 0.997 / USSR 1.000)
- vs greedy: 0.907 (US 0.875 / USSR 0.940)
- vs v1: 0.975 (US 0.987 / USSR 0.963)
- vs v2: 0.997 (US 1.000 / USSR 0.993)
- vs v3: 0.853 (US 0.877 / USSR 0.828)
- vs v4: 0.576 (US 0.520 / USSR 0.632)
- vs v5: 0.837 (US 0.830 / USSR 0.843)

## v7

Commit `789cfb9` — run `pure`, 16,000 games trained.

- Loop generation 2: v6 continued for 4,000 games; gate 0.55 cleared at 0.680 (worst seed) against v6.
- Elo vs random: **+1407 ± 28** over seeds [0, 1, 2]
- vs random: 0.993 (US 0.997 / USSR 0.990)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.924 (US 0.883 / USSR 0.965)
- vs v1: 0.972 (US 0.980 / USSR 0.963)
- vs v2: 0.993 (US 0.997 / USSR 0.990)
- vs v3: 0.929 (US 0.967 / USSR 0.892)
- vs v4: 0.725 (US 0.707 / USSR 0.743)
- vs v5: 0.892 (US 0.920 / USSR 0.865)
- vs v6: 0.700 (US 0.630 / USSR 0.770)

## v8

Commit `789cfb9` — run `pure`, 20,000 games trained.

- Loop generation 3: v7 continued for 4,000 games; gate 0.55 cleared at 0.720 (worst seed) against v7.
- Elo vs random: **+1394 ± 30** over seeds [0, 1, 2]
- vs random: 0.990 (US 1.000 / USSR 0.980)
- vs first: 0.992 (US 1.000 / USSR 0.983)
- vs greedy: 0.962 (US 0.967 / USSR 0.957)
- vs v1: 0.968 (US 0.960 / USSR 0.977)
- vs v2: 0.990 (US 0.987 / USSR 0.993)
- vs v3: 0.937 (US 0.930 / USSR 0.943)
- vs v4: 0.880 (US 0.837 / USSR 0.923)
- vs v5: 0.910 (US 0.903 / USSR 0.917)
- vs v6: 0.828 (US 0.750 / USSR 0.907)
- vs v7: 0.733 (US 0.613 / USSR 0.853)

## v9

Commit `d2f51f8` — run `pure-e2`, 24,000 games trained.

- The epochs experiment: v8 continued for 4,000 games with `--n-epochs 2` (now the default), gated 0.635 against v8 (worst seed 0.62) versus the 4-epoch control's 0.573, at 28% less wall time; the control arm is not frozen.
- Elo vs random: **+1527 ± 26** over seeds [0, 1, 2]
- vs random: 0.998 (US 0.997 / USSR 1.000)
- vs first: 0.997 (US 1.000 / USSR 0.993)
- vs greedy: 0.987 (US 0.973 / USSR 1.000)
- vs v1: 0.973 (US 0.953 / USSR 0.993)
- vs v2: 0.995 (US 0.990 / USSR 1.000)
- vs v3: 0.970 (US 0.947 / USSR 0.993)
- vs v4: 0.892 (US 0.837 / USSR 0.947)
- vs v5: 0.950 (US 0.920 / USSR 0.980)
- vs v6: 0.867 (US 0.783 / USSR 0.950)
- vs v7: 0.782 (US 0.640 / USSR 0.923)
- vs v8: 0.635 (US 0.432 / USSR 0.838)

## v10

Commit `26a48c9` — run `pure`, 32,000 games trained.

- Loop generation 2: v9 continued for 4,000 games; gate 0.55 cleared at 0.600 (worst seed) against v9.
- Elo vs random: **+1499 ± 46** over seeds [0, 1, 2]
- vs random: 0.997 (US 1.000 / USSR 0.993)
- vs first: 0.997 (US 1.000 / USSR 0.993)
- vs greedy: 0.977 (US 0.970 / USSR 0.983)
- vs v1: 0.968 (US 0.943 / USSR 0.993)
- vs v2: 0.997 (US 0.997 / USSR 0.997)
- vs v3: 0.977 (US 0.963 / USSR 0.990)
- vs v4: 0.902 (US 0.827 / USSR 0.977)
- vs v5: 0.960 (US 0.943 / USSR 0.977)
- vs v6: 0.852 (US 0.767 / USSR 0.937)
- vs v7: 0.813 (US 0.673 / USSR 0.953)
- vs v8: 0.691 (US 0.465 / USSR 0.917)
- vs v9: 0.629 (US 0.383 / USSR 0.875)

## v11

Commit `9885216` — run `h256-e4`, 8,000 games trained (255 updates, 47 min
with 8 collector processes and 16 torch threads).

- The capacity A/B winner: a fresh run with **`--hidden 256`** (858k
  parameters; 128: 272k) at 4 PPO epochs, otherwise v5's recipe and
  budget — 50% self-play, 50% pool, no anchor, 8,000 games. The 2-epoch
  arm and the hidden-128 2-epoch control are not frozen; the experiment
  is written up in [docs/JOSHUA.md](../docs/JOSHUA.md).
- Elo vs random: **+1244 ± 31** over seeds [0, 1, 2]
- vs random: 0.993 (US 0.990 / USSR 0.997)
- vs first: 0.997 (US 0.997 / USSR 0.997)
- vs greedy: **0.909** (US 0.848 / USSR 0.970)
- vs v1: 0.962 (US 0.943 / USSR 0.980)
- vs v2: 0.993 (US 0.990 / USSR 0.997)
- vs v3: 0.855 (US 0.765 / USSR 0.945)
- vs v4: 0.701 (US 0.472 / USSR 0.930)
- vs v5: **0.847** (US 0.780 / USSR 0.913) — the same-budget control
- vs v6: 0.584 (US 0.365 / USSR 0.803)
- vs v7: 0.322 (US 0.152 / USSR 0.492)
- vs v8: 0.213 (US 0.090 / USSR 0.337)
- vs v9: 0.199 (US 0.050 / USSR 0.348)
- vs v10: 0.145 (US 0.013 / USSR 0.277)
- Sampled play (seed 0): 0.815 vs greedy, 0.800 vs v5, 0.115 vs v10.
- Notes: at v5's budget it sits between v6 and v7 on the 128-hidden
  chain — 8,000 games of the wider network are worth 12–16,000 of the
  narrower one, at about twice the wall time per game. The USSR seat is
  the stronger one against everything from v3 on.

## v12

Commit `363c5d4` — run `h256-e4`, 12,000 games trained.

- Loop generation 1: v11 continued for 4,000 games; gate 0.55 cleared at 0.785 (worst seed) against v11.
- Elo vs random: **+1452 ± 15** over seeds [0, 1, 2]
- vs random: 1.000 (US 1.000 / USSR 1.000)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.977 (US 0.967 / USSR 0.987)
- vs v1: 0.980 (US 0.967 / USSR 0.993)
- vs v10: 0.301 (US 0.063 / USSR 0.538)
- vs v11: 0.791 (US 0.618 / USSR 0.963)
- vs v2: 1.000 (US 1.000 / USSR 1.000)
- vs v3: 0.978 (US 0.963 / USSR 0.993)
- vs v4: 0.864 (US 0.788 / USSR 0.940)
- vs v5: 0.928 (US 0.913 / USSR 0.943)
- vs v6: 0.793 (US 0.657 / USSR 0.930)
- vs v7: 0.756 (US 0.635 / USSR 0.877)
- vs v8: 0.503 (US 0.323 / USSR 0.683)
- vs v9: 0.369 (US 0.120 / USSR 0.618)

## v13

Commit `363c5d4` — run `h256-e4`, 20,000 games trained.

- Loop generation 3: v12 continued for 4,000 games; gate 0.55 cleared at 0.620 (worst seed) against v12.
- Elo vs random: **+1460 ± 14** over seeds [0, 1, 2]
- vs random: 1.000 (US 1.000 / USSR 1.000)
- vs first: 0.998 (US 0.997 / USSR 1.000)
- vs greedy: 0.977 (US 0.980 / USSR 0.973)
- vs v1: 0.980 (US 0.963 / USSR 0.997)
- vs v10: 0.399 (US 0.070 / USSR 0.728)
- vs v11: 0.798 (US 0.620 / USSR 0.977)
- vs v12: 0.625 (US 0.355 / USSR 0.895)
- vs v2: 0.993 (US 0.993 / USSR 0.993)
- vs v3: 0.968 (US 0.957 / USSR 0.980)
- vs v4: 0.902 (US 0.812 / USSR 0.993)
- vs v5: 0.950 (US 0.923 / USSR 0.977)
- vs v6: 0.840 (US 0.710 / USSR 0.970)
- vs v7: 0.802 (US 0.637 / USSR 0.967)
- vs v8: 0.650 (US 0.397 / USSR 0.903)
- vs v9: 0.566 (US 0.278 / USSR 0.853)
