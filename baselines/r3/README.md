# Baselines — rules version 3

One entry per frozen version; the protocol and the layout are in [../README.md](../README.md).

## v1

Commit `dd2f4e1` — run `r3`, 14,020 games trained.

- Bootstrap of the r3 ladder: recipe v11 from scratch, stopped by **confirmed** at 14,020 games (rule: rolling mean over 2 evaluations of 200 every 500 games ≥ 0.75 on both seats, confirmed over 600; plateau 4; cap 20,000). Last rolling mean vs Greedy: US 0.777 / USSR 0.925. Confirmation: 0.876 (US 0.813 / USSR 0.938) over 600 -- passed.
- Elo vs random: **+823 ± 122** over seeds [0, 1, 2]
- vs random: 0.975 (US 0.980 / USSR 0.970)
- vs first: 0.990 (US 0.990 / USSR 0.990)
- vs greedy: 0.865 (US 0.790 / USSR 0.940)

## v2

Commit `c4e88a3` — run `r3`, 18,020 games trained.

- Loop generation 1: v1 continued for 4,000 games; gate 0.55 cleared at 0.627 (worst seed) against v1.
- Elo vs random: **+943 ± 130** over seeds [0, 1, 2]
- vs random: 0.977 (US 0.987 / USSR 0.967)
- vs first: 0.992 (US 0.983 / USSR 1.000)
- vs greedy: 0.845 (US 0.742 / USSR 0.948)
- vs v1: 0.636 (US 0.392 / USSR 0.880)

## v3

Commit `c4e88a3` — run `r3`, 22,020 games trained.

- Loop generation 2: v2 continued for 4,000 games; gate 0.55 cleared at 0.580 (worst seed) against v2.
- Elo vs random: **+1059 ± 69** over seeds [0, 1, 2]
- vs random: 0.980 (US 0.983 / USSR 0.977)
- vs first: 0.998 (US 0.997 / USSR 1.000)
- vs greedy: 0.789 (US 0.713 / USSR 0.865)
- vs v1: 0.723 (US 0.530 / USSR 0.917)
- vs v2: 0.602 (US 0.347 / USSR 0.857)

## v4

Commit `c4e88a3` — run `r3`, 26,020 games trained.

- Loop generation 3: v3 continued for 4,000 games; gate 0.55 cleared at 0.588 (worst seed) against v3.
- Elo vs random: **+1120 ± 10** over seeds [0, 1, 2]
- vs random: 0.980 (US 0.983 / USSR 0.977)
- vs first: 0.996 (US 0.992 / USSR 1.000)
- vs greedy: 0.883 (US 0.793 / USSR 0.973)
- vs v1: 0.801 (US 0.663 / USSR 0.938)
- vs v2: 0.688 (US 0.440 / USSR 0.937)
- vs v3: 0.614 (US 0.337 / USSR 0.892)

## v5

Commit `c4e88a3` — run `r3`, 30,020 games trained.

- Loop generation 4: v4 continued for 4,000 games; gate 0.55 cleared at 0.573 (worst seed) against v4.
- Elo vs random: **+1382 ± 28** over seeds [0, 1, 2]
- vs random: 0.998 (US 0.997 / USSR 1.000)
- vs first: 0.995 (US 0.990 / USSR 1.000)
- vs greedy: 0.978 (US 0.963 / USSR 0.993)
- vs v1: 0.856 (US 0.793 / USSR 0.918)
- vs v2: 0.796 (US 0.685 / USSR 0.907)
- vs v3: 0.688 (US 0.467 / USSR 0.910)
- vs v4: 0.593 (US 0.340 / USSR 0.845)

## v6

Commit `c4e88a3` — run `r3`, 34,020 games trained.

- Loop generation 5: v5 continued for 4,000 games; gate 0.55 cleared at 0.552 (worst seed) against v5.
- Elo vs random: **+1375 ± 43** over seeds [0, 1, 2]
- vs random: 0.995 (US 1.000 / USSR 0.990)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.975 (US 0.973 / USSR 0.977)
- vs v1: 0.890 (US 0.810 / USSR 0.970)
- vs v2: 0.826 (US 0.685 / USSR 0.967)
- vs v3: 0.795 (US 0.630 / USSR 0.960)
- vs v4: 0.722 (US 0.545 / USSR 0.898)
- vs v5: 0.595 (US 0.398 / USSR 0.792)

## v7

Commit `c4e88a3` — run `r3`, 38,020 games trained.

- Loop generation 6: v6 continued for 4,000 games; gate 0.55 cleared at 0.625 (worst seed) against v6.
- Elo vs random: **+1309 ± 89** over seeds [0, 1, 2]
- vs random: 0.988 (US 0.997 / USSR 0.980)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.978 (US 0.967 / USSR 0.990)
- vs v1: 0.827 (US 0.700 / USSR 0.953)
- vs v2: 0.837 (US 0.743 / USSR 0.930)
- vs v3: 0.827 (US 0.687 / USSR 0.967)
- vs v4: 0.755 (US 0.617 / USSR 0.893)
- vs v5: 0.683 (US 0.472 / USSR 0.895)
- vs v6: 0.656 (US 0.452 / USSR 0.860)

## v8

Commit `c4e88a3` — run `r3`, 42,020 games trained.

- Loop generation 7: v7 continued for 4,000 games; gate 0.55 cleared at 0.583 (worst seed) against v7.
- Elo vs random: **+1185 ± 35** over seeds [0, 1, 2]
- vs random: 0.975 (US 1.000 / USSR 0.950)
- vs first: 0.997 (US 1.000 / USSR 0.993)
- vs greedy: 0.967 (US 0.963 / USSR 0.970)
- vs v1: 0.829 (US 0.713 / USSR 0.945)
- vs v2: 0.840 (US 0.740 / USSR 0.940)
- vs v3: 0.838 (US 0.725 / USSR 0.950)
- vs v4: 0.807 (US 0.660 / USSR 0.953)
- vs v5: 0.725 (US 0.557 / USSR 0.893)
- vs v6: 0.711 (US 0.545 / USSR 0.877)
- vs v7: 0.613 (US 0.488 / USSR 0.738)
