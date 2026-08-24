# Baselines — rules version 3, tournament bid US +2

One entry per frozen version; the protocol and the layout are in [../README.md](../README.md).

## v1

Commit `8a03b2f` — run `r3-bid2`, 11,024 games trained.

- Bootstrap of the r3-bid2 ladder: recipe v11 from scratch, stopped by **confirmed** at 11,024 games (rule: rolling mean over 2 evaluations of 200 every 500 games ≥ 0.75 on both seats, confirmed over 600; plateau 4; cap 20,000). Last rolling mean vs Greedy: US 0.825 / USSR 0.785. Confirmation: 0.807 (US 0.833 / USSR 0.780) over 600 -- passed.
- Elo vs random: **+1132 ± 268** over seeds [0, 1, 2]
- vs random: 0.988 (US 0.990 / USSR 0.987)
- vs first: 0.998 (US 1.000 / USSR 0.997)
- vs greedy: 0.820 (US 0.858 / USSR 0.782)
