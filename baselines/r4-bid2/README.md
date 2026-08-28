# Baselines — rules version 4, tournament bid US +2

One entry per frozen version; the protocol and the layout are in [../README.md](../README.md).

## v1

Commit `7a72247` — run `r4b2-boot`, 12,519 games trained.

- Bootstrap of the r4-bid2 ladder: recipe v11 from scratch, stopped by **confirmed** at 12,519 games (rule: rolling mean over 2 evaluations of 200 every 500 games ≥ 0.75 on both seats, confirmed over 600; plateau 4; cap 20,000). Last rolling mean vs Greedy: US 0.750 / USSR 0.830. Confirmation: 0.788 (US 0.750 / USSR 0.827) over 600 -- passed.
- Elo vs random: **+1090 ± 215** over seeds [0, 1, 2]
- vs random: 0.988 (US 0.993 / USSR 0.983)
- vs first: 0.993 (US 1.000 / USSR 0.987)
- vs greedy: 0.766 (US 0.697 / USSR 0.835)

## v2

Commit `fc80686` — run `r4b2-boot`, 16,519 games trained.

- Loop generation 1: v1 continued for 4,000 games; gate 0.55 cleared at 0.565 (worst seed) against v1.
- Elo vs random: **+1024 ± 255** over seeds [0, 1, 2]
- vs random: 0.982 (US 0.987 / USSR 0.977)
- vs first: 0.995 (US 0.993 / USSR 0.997)
- vs greedy: 0.870 (US 0.840 / USSR 0.900)
- vs v1: 0.578 (US 0.482 / USSR 0.673)

## v3

Commit `fc80686` — run `r4b2-boot`, 20,519 games trained.

- Loop generation 2: v2 continued for 4,000 games; gate 0.55 cleared at 0.585 (worst seed) against v2.
- Elo vs random: **+1229 ± 87** over seeds [0, 1, 2]
- vs random: 0.992 (US 1.000 / USSR 0.983)
- vs first: 0.993 (US 0.987 / USSR 1.000)
- vs greedy: 0.892 (US 0.827 / USSR 0.957)
- vs v1: 0.715 (US 0.603 / USSR 0.827)
- vs v2: 0.615 (US 0.403 / USSR 0.827)

## v4

Commit `fc80686` — run `r4b2-boot`, 28,520 games trained.

- Loop generation 4: v3 continued for 4,000 games; gate 0.55 cleared at 0.565 (worst seed) against v3.
- Elo vs random: **+1180 ± 111** over seeds [0, 1, 2]
- vs random: 0.987 (US 0.990 / USSR 0.983)
- vs first: 0.995 (US 0.990 / USSR 1.000)
- vs greedy: 0.940 (US 0.900 / USSR 0.980)
- vs v1: 0.701 (US 0.580 / USSR 0.822)
- vs v2: 0.662 (US 0.505 / USSR 0.820)
- vs v3: 0.592 (US 0.400 / USSR 0.783)
