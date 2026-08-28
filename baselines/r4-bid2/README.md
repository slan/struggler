# Baselines — rules version 4, tournament bid US +2

One entry per frozen version; the protocol and the layout are in [../README.md](../README.md).

## v1

Commit `7a72247` — run `r4b2-boot`, 12,519 games trained.

- Bootstrap of the r4-bid2 ladder: recipe v11 from scratch, stopped by **confirmed** at 12,519 games (rule: rolling mean over 2 evaluations of 200 every 500 games ≥ 0.75 on both seats, confirmed over 600; plateau 4; cap 20,000). Last rolling mean vs Greedy: US 0.750 / USSR 0.830. Confirmation: 0.788 (US 0.750 / USSR 0.827) over 600 -- passed.
- Elo vs random: **+1090 ± 215** over seeds [0, 1, 2]
- vs random: 0.988 (US 0.993 / USSR 0.983)
- vs first: 0.993 (US 1.000 / USSR 0.987)
- vs greedy: 0.766 (US 0.697 / USSR 0.835)
