# Baselines — rules version 2

One entry per frozen version of the current ladder; the protocol and the
layout are in [../README.md](../README.md), the notebook is
[docs/JOSHUA.md](../../docs/JOSHUA.md).

## v1

Commit `5031282` (trained at `b55daf5`) — run `engine-fixes`, 8,000 games
(recipe v11: hidden 256, 4 PPO epochs, 50% self-play / 50% pool, a
snapshot every 5 updates; 8 collectors, 49 min).

- The first clean run on the fixed engine, and the root of this ladder:
  no control (the r1 ladder does not rate on this game) and no champion.
  It learned the held-scoring-card rule in ~200 games (mean final turn
  1.5 → 5.6 by 600 games) and plays to turn 7.5 against itself.
- Elo vs random: **+1253 ± 30** over seeds [0, 1, 2]
- vs random: 0.995 (US 1.000 / USSR 0.990)
- vs first: 0.997 (US 0.997 / USSR 0.997)
- vs greedy: **0.748** (US 0.740 / USSR 0.755) — the fixed Greedy
  (`cc04ffa`), which is stronger than r1's: r1's v11 beats it only 0.715.
- USSR edge against itself: 0.52 (200 games) / 0.44 (120 games) — even.
  `wopr.diagnose` (runs/engine-fixes/diagnose.json): endings USSR by VP
  29, US by VP 27, US by DEFCON 19, US by final scoring 18, USSR by final
  scoring 13, USSR by DEFCON 7, held scoring card 6, Europe control 1;
  VP track within ±2.5 all game; Asia Scoring nets to the US (937 vs
  154), Europe to the USSR (625 vs 51).
- Informal, cross-ladder (not a rating): 0.585 vs r1/v11, 0.685 vs
  r1/v16 on this engine.

## v2

Commit `ce56517` — run `engine-fixes`, 12,000 games trained.

- Loop generation 1: v1 continued for 4,000 games; gate 0.55 cleared at 0.825 (worst seed) against v1.
- Elo vs random: **+1267 ± 156** over seeds [0, 1, 2]
- vs random: 0.990 (US 0.990 / USSR 0.990)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.901 (US 0.840 / USSR 0.962)
- vs v1: 0.837 (US 0.740 / USSR 0.933)

## v3

Commit `ce56517` — run `engine-fixes`, 16,000 games trained.

- Loop generation 2: v2 continued for 4,000 games; gate 0.55 cleared at 0.642 (worst seed) against v2.
- Elo vs random: **+1370 ± 84** over seeds [0, 1, 2]
- vs random: 0.995 (US 1.000 / USSR 0.990)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.947 (US 0.943 / USSR 0.950)
- vs v1: 0.873 (US 0.810 / USSR 0.937)
- vs v2: 0.651 (US 0.455 / USSR 0.847)
