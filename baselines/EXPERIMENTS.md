# Experiments

One row per `wopr.ab` run, frozen or not (`baselines/README.md` has the
frozen ones in full). Win rates are the run's, argmax play, mean over the
eval seeds with the worst seed in brackets; `USSR edge` is the run against
itself, as USSR. The idea and the reading of each row: docs/JOSHUA.md.

| date | run | commit | recipe | games | vs control | vs champion | vs greedy | USSR edge | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-22 | `hidden256` | `b34daf8` | — (v11 at 2 epochs) | 8,000 | 0.428 [0.405] (US 0.60 / USSR 0.25) vs v5 | 0.018 [0.015] vs v10 | 0.331 [0.315] | — | capacity A/B, first arm: confounded by epochs |
| 2026-08-22 | `h128-e2` | `b34daf8` | — (v5 at 2 epochs) | 8,000 | 0.070 [0.062] (US 0.10 / USSR 0.05) vs v5 | — | 0.254 [0.198] | — | the same-recipe control: 2 epochs from scratch is undertrained |
| 2026-08-22 | `h256-e4` | `9885216` | v11 | 8,000 | 0.847 [0.840] (US 0.78 / USSR 0.91) vs v5 | 0.145 [0.140] vs v10 | 0.909 [0.890] | 0.78 | the matched arm; frozen as v11 |
| 2026-08-22 | `h256-e4` +4k at `--handicap 8` | `62edff9` | v11 | 36,000 | — | 0.606 [0.588] (US 0.37 / USSR 0.85) vs v15 | 0.975 | 0.75 | handicap gen 1; frozen as v16; edge unmoved |
| 2026-08-22 | `h256-e4` +4k at `--handicap 8` | `62edff9` | v11 | 40,000 | — | 0.489 [0.480] (US 0.18 / USSR 0.80) vs v16 | 0.980 | — | handicap gen 2: miss; closed |
| 2026-08-22 | `h256-e4` +8k at `--margin 0.5` | `03ab712` | v11 | 48,000 | — | 0.376 / 0.390 [0.355] (US 0.05–0.10) vs v16 | 0.960 / 0.975 | — | margin reward: regression, games shorten; closed |
