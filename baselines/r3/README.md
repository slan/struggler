# Baselines — rules version 3

One entry per frozen version; the protocol and the layout are in [../README.md](../README.md).

## v1

Commit `dd2f4e1` — run `r3`, 14,020 games trained.

- Bootstrap of the r3 ladder: recipe v11 from scratch, stopped by **confirmed** at 14,020 games (rule: rolling mean over 2 evaluations of 200 every 500 games ≥ 0.75 on both seats, confirmed over 600; plateau 4; cap 20,000). Last rolling mean vs Greedy: US 0.777 / USSR 0.925. Confirmation: 0.876 (US 0.813 / USSR 0.938) over 600 -- passed.
- Elo vs random: **+823 ± 122** over seeds [0, 1, 2]
- vs random: 0.975 (US 0.980 / USSR 0.970)
- vs first: 0.990 (US 0.990 / USSR 0.990)
- vs greedy: 0.865 (US 0.790 / USSR 0.940)
