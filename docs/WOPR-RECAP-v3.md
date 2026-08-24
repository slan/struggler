# WOPR recap — rules version 3, the bid, and the road to hard mode

*Written 2026-08-25, at the end of the bootstrap/bid/bridge arc. A
snapshot, not a contract: the binding docs are [WOPR.md](WOPR.md) and
the notebook [JOSHUA.md](JOSHUA.md); this file says where the project
stands against its goal and what to do next, in order of impact.*

**The goal: a fully bootstrapped, PPO-driven bot that beats Playdek's
AI in HARD mode, on both seats, at the tournament standard of +2 US
influence.**

## Where we stand, in one table

| Opponent | Best measured | Who / where |
| --- | --- | --- |
| random | 0.99 | any champion |
| Greedy (bid 2) | 0.98 | r3-bid2 line at 20k games (curve saturated) |
| own previous version | gate 0.55–0.69 per generation until the plateau | both ladders |
| **Playdek easy, bid 2** | **USSR 0.093 / US 0.078** | r3-bid2/v3, 60 a seat (unconfounded) |
| **Playdek hard** | **unmeasured for Joshua; Greedy is 0/60** | greedy-hard runs, both seats |

The picture: self-play plus the ladder reliably manufactures strength
against *its own distribution* — every internal yardstick saturates —
and roughly one tenth of that strength shows up against the weakest
opponent that plays a different game. Hard mode is beyond that by an
unmeasured but certainly large margin: Greedy, which the champions beat
19 games in 20, wins zero of sixty against it.

## What was built this arc (all committed on `wopr`)

- **`wopr.bootstrap`** — a ladder's v1 trains from scratch until the
  yardstick says stop: a Greedy evaluation every 500 games played *on
  the idle collectors during the PPO update* (~1 s of waiting per
  tick), a rolling-mean stop rule with a 600-game confirmation, plateau
  and cap branches, kill-safe checkpoints. A full bootstrap is ~70–90
  minutes on this machine.
- **The plateau rules** — the bootstrap's (no new best of the overall
  rolling mean for 4 ticks) and the loop's (two gate misses in the last
  three generations, `--plateau-misses`), so both run unattended.
- **The tournament bid** (rule 11.1.4) — `Engine.new_game(us_bid=N)`,
  exactly as printed (post-setup US placement, own countries, control+2
  cap), `--bid` through every wopr tool, ladders per bid
  (`baselines/r3-bid2/`). The DLL plays the same rule natively
  (`GameParameters.additionalInfluence` — probed identical semantics),
  so `wopr.playdek.eval --bid 2` is an eval on the policy's own game.
- **Bridge fixes** — trapped-seat held scoring card (known/void), the
  simulation trying the bot's auto-resolved event choices, Grain Sales'
  take read off the DLL's card moves, the bot's Grain-Sales scoring-card
  take, Wargames' end-game prompt, and a crashed game no longer kills an
  eval batch.

## The learning curve

Win rate vs Greedy (200-game ticks, US/USSR in brackets):

| games | r3 (printed game) | r3-bid2 (US +2) | r3-bid2-gshare (10% Greedy) |
| --- | --- | --- | --- |
| 2,000 | 0.18 [.18/.18] | 0.17 [.18/.17] | 0.06 [.07/.05] |
| 4,000 | 0.42 [.40/.44] | 0.38 [.47/.29] | 0.18 [.23/.13] |
| 8,000 | 0.60 [.51/.70] | 0.40 [.32/.48] | 0.72 [.64/.81] |
| 11,000 | 0.72 [.53/.91] | 0.83 [.85/.81] | 0.84 [.79/.89] |
| stop | confirmed @14,020 | confirmed @11,024 | confirmed @11,518 |
| loop plateau | v8 @42k (0.97 vs Greedy) | v3 @23k (0.95) | — (arm, not frozen) |

Findings that now stand on measurements:

- **The bid evens the game.** USSR edge between equals: 0.667 → 0.500
  at v1, 0.64 → 0.517 at champion strength. The bootstrap confirms 21%
  sooner, the loop's gates are seat-balanced, and the bid ladder is the
  live one.
- **Greedy saturates as a yardstick at ~0.95–0.98** while the ladder is
  still improving against itself; from there only the Playdek eval
  measures anything real.
- **Anchors still are not teachers** (arm 1): a 10% Greedy share
  dilutes early learning, washes out on every endpoint, and leaves the
  DEFCON-gift rate untouched — because Greedy never *exploits* the
  gift. Coverage comes from opponents that punish, not opponents that
  differ.
- **What transfers and what doesn't.** Two loop generations (v1 → v3 on
  the bid ladder) moved the easy-AI number from ~0.00–0.03 to
  0.078–0.093 — internal strength does transfer, at roughly a tenth of
  its face value. The two named failure modes barely move: the **DEFCON
  gift** (CIA Created / Grain Sales played for Ops at DEFCON 2; the AI
  coups, the phasing USSR loses) is ~40% of USSR-seat losses at every
  strength measured, and the US seat's 20-VP blowouts by turn 4–5
  remain the bulk of its losses.

## Desync status

Attrition per 60-game eval batch: 18% at its worst (r3-bid2/v3 at bid
0) → **8% now** (5/60 and 3–5/60 in the latest batches; the two control
batches ran 0/60). Parity at the random/Greedy level remains where the
eighth pass left it: differ 280/280, emu 32/32 + 120/120 (now also at
bid 2), hidden-prompt harness 59/59, Greedy-vs-hard 30/30 both seats.

Fixed this arc: the trapped held scoring card (→ known/void), the
simulation's premature judgment at auto-resolved bot choices (seed 332),
Grain Sales' take off the card moves (was 6 games a seat, now ≤2), the
bot's own scoring-card take (a batch-killing crash), Wargames' end-game
prompt, eval-pool resilience.

Open families (seeds in [WOPR.md](WOPR.md), logs under `runs/playdek/`):
Grain Sales' remainder (~2 a seat; the AI's line does not reproduce on
rerun — needs traces caught by volume), the trapped AI's scoring card
played with no prompt (seed 323), Junta's choice (326, 343), an
event-placement mapping (353, hint 0xa031), a turn-1 placement the DLL
does not offer (357), and a few slow VP/hand drifts. None currently
crashes a batch; all cost sample.

## Next steps, in order of impact

*(A constraint adopted on review: the bot stays **self-play only** —
no external opponent in the training mix. Playdek's AI is an
evaluation, never a teacher; a league-style exploiter, though it uses
nothing but the line's own weights, is held in reserve rather than
first. What follows respects that.)*

1. **Search over the learned value head.** Learning stays pure naive
   self-play; a one-ply (or shallow-rollout) lookahead at *inference*
   fixes the whole class of tactical blunders on its own — playing CIA
   Created for Ops at DEFCON 2 leads to a state where the opponent's
   best reply is a won coup, and the value head already scores lost
   positions as lost. The DEFCON gift (~40% of USSR-seat losses) is
   exactly this shape, and against a scripted opponent search is the
   classic strength multiplier — the likeliest source of what hard
   mode demands. Carried from the r1 road map with the order/recency
   layout bump as its companion.
2. **Scenario-seeded self-play.** Shape the *initial-state
   distribution*, not the opponent: a fraction of training games starts
   from positions where the failure is on the table (DEFCON 2, an
   event-gift card in hand; later, positions sampled from lost eval
   games' own prefixes if that purity line is acceptable). Both seats
   are still the learner, so the US seat learns to take the punishing
   coup and the USSR seat learns to stop offering it. No external
   policy anywhere; `Engine.serialize`/`deserialize` makes the seeding
   cheap.
3. **Measure hard mode now.** 30 a seat of the current champion
   (~3 h). Everything above aims at a number nobody has seen; size the
   mountain before climbing it, and make hard the standing eval beside
   easy. Evaluation only — the AI teaches nothing.
4. **In reserve: a league exploiter.** If search plus seeding leave the
   gift rate standing, a short run trained against the frozen champion
   alone, added to the champion's pool, is the next escalation — it is
   still the line's own descendants (AlphaStar's league sense of
   self-play), but it is a step away from *naive* self-play and is
   taken only if the purer fixes fail. (Sparring against Playdek's AI
   itself, which genuinely breaks self-play, is off the training menu
   under the constraint.)
5. **Bridge to <2% attrition.** The open families above, traces caught
   by volume. Matters more as evals move to hard mode (longer games,
   more late-war paths), where a desync is lost sample.
6. **Protocol hygiene as the numbers tighten**: more eval seeds per
   claim (the Elo seed-spread tripled on the bid ladder), and the
   Playdek eval's Wilson bounds on every reported rate.

The through-line: the machinery — bootstrap, ladders, bid, bridge — is
done and trustworthy, and the training stays self-play only. Every
remaining gap is about *how much strength the policy can express and
where its blind spots lie* (steps 1, 2, 4 — search, seeding, and the
reserve exploiter), and *how cleanly we can measure it against the
real opponent* (steps 3, 5, 6).
