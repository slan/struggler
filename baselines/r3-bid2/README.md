# Baselines — rules version 3, tournament bid US +2

One entry per frozen version; the protocol and the layout are in [../README.md](../README.md).

## v1

Commit `8a03b2f` — run `r3-bid2`, 11,024 games trained.

- Bootstrap of the r3-bid2 ladder: recipe v11 from scratch, stopped by **confirmed** at 11,024 games (rule: rolling mean over 2 evaluations of 200 every 500 games ≥ 0.75 on both seats, confirmed over 600; plateau 4; cap 20,000). Last rolling mean vs Greedy: US 0.825 / USSR 0.785. Confirmation: 0.807 (US 0.833 / USSR 0.780) over 600 -- passed.
- Elo vs random: **+1132 ± 268** over seeds [0, 1, 2]
- vs random: 0.988 (US 0.990 / USSR 0.987)
- vs first: 0.998 (US 1.000 / USSR 0.997)
- vs greedy: 0.820 (US 0.858 / USSR 0.782)

## v2

Commit `e68c22d` — run `r3-bid2`, 15,024 games trained.

- Loop generation 1: v1 continued for 4,000 games; gate 0.55 cleared at 0.670 (worst seed) against v1.
- Elo vs random: **+911 ± 71** over seeds [0, 1, 2]
- vs random: 0.978 (US 0.990 / USSR 0.967)
- vs first: 0.981 (US 0.988 / USSR 0.973)
- vs greedy: 0.892 (US 0.917 / USSR 0.867)
- vs v1: 0.693 (US 0.670 / USSR 0.717)

## v3

Commit `e68c22d` — run `r3-bid2`, 19,024 games trained.

- Loop generation 2: v2 continued for 4,000 games; gate 0.55 cleared at 0.635 (worst seed) against v2.
- Elo vs random: **+1138 ± 96** over seeds [0, 1, 2]
- vs random: 0.983 (US 0.997 / USSR 0.970)
- vs first: 1.000 (US 1.000 / USSR 1.000)
- vs greedy: 0.956 (US 0.955 / USSR 0.957)
- vs v1: 0.759 (US 0.702 / USSR 0.817)
- vs v2: 0.661 (US 0.557 / USSR 0.765)
