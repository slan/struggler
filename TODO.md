# THE COMPOSE CONFIRMED AT 0.258, THE BRIDGE'S 19TH PASS, KICK3 NEGATIVE (2026-09-01)

Three entries closed in one session (docs/JOSHUA.md, all
pre-registered first):

- **kick2+veto is the standing reported player at pooled 0.258**
  (54/209; USSR 0.330 / US 0.189): the compose read 0.268 on seeds
  300+ and **replicated at 0.250 on fresh seeds 500+** — both seats
  lift over kick2 raw both times; gift share 0.188 / 0.081. Next
  training arm's bar: pooled + 0.05 = **0.308**.
- **The nineteenth bridge pass** (the judge sees past standing
  drift): a failing simulation branch whose residual diff keys ⊆
  the pre-choice diff is carried instead of fataling
  (`state_diff_keys`, `drift-pick` lines), same rescue at the
  drain's deadlock. Root-caused on the Junta pair (kick2-easy
  358/397). Measured: the compose player's old-judge batch 20/120
  desyncs → fresh-seed new-judge batch **8/120, void 0**; kick3's
  9/120. Suite 544, sweep 149/149 (0 desyncs), hotseat 8/8, differ
  12/12.
- **kick3 (re-dose: gift-scenario starts on kick2's construction)
  closed negative on the key read**: mean 0.153 (≥ 0.140 met, best
  raw US seat 0.123) but gift share **0.659** — starting games in
  gift states where 90% of opponents don't punish teaches the gift
  harder (the probe warned first: 11/100 vs kick2's 6). The named
  follow-on, a new entry on the user's call: **seat the punisher in
  the scenario games** (wiring: scenario starts × forced anchor).

Article facts dossier (no write-up): `runs/article/FACTS.md` —
timeline, all arms and numbers, bridge passes, infrastructure,
pointers. The article itself: the user, offline.

# KICK2 + THE VETO RIDER: THE BAR FALLS, TWICE (2026-09-01)

One pre-registered entry, two questions (docs/JOSHUA.md), both over
the 0.136 bar — the program's first positive transfers:

- **kick1+veto 0.248** [0.18, 0.34] (USSR 0.278 / US 0.218, gift
  share 0.179) — zero training, the veto over kick1's checkpoint.
  Both seats lifted; the US seat's first movement ever (kick1 raw
  0.035 → 0.218, non-overlapping intervals): kick1's US losses were
  one-third DEFCON deaths and the veto refuses exactly those. **The
  standing reported player**, as a named policy.
- **kick2 0.140** [0.09, 0.22] (USSR 0.190 / US 0.089, gift share
  0.489) — kick1's recipe + falken1 at a fixed 10% anchor share
  (new `--anchor name=ckpt.pt` wiring, `ckpt:` policy ids). First
  *raw checkpoint* over the bar. All gates passed (absorption 0.505,
  Greedy 0.983, probe gifted-deaths 6/100 vs kick1's 13); the
  anchor moved the board gift share 0.604 → 0.489 — the predicted
  direction, not the ≤ 0.25 read. Theory verdict: dose-response —
  reward pricing works, 10% is not enough against self-play's 90%.

Desyncs 6/120 and 11/120, void 0, known families only. Raw
reporting: kick2 is the strongest raw checkpoint measured.

**Evidence-pointed next constructions (user's call, new entries):**
veto over kick2 (compose the two positives), a larger anchor share
or gift-scenario starts (the re-dose), the article (now with a win
to end on), and re-setting the bar from 0.248.

# KICK1 (KICKSTARTING): NEGATIVE ON THE BAR, BEST USSR SEAT EVER, AND A THEORY (2026-09-01)

v3-init + interleaved corpus pull (`--kickstart`, new wiring in
train.py/callback.py, `wopr.distill top1`): absorption 0.335→0.507
at zero strength cost (Greedy 0.958), decider mean **0.088** — ties
v3's standing 0.086, bar missed — with USSR **0.143**, the best
single-seat number ever vs the easy AI, and longer games (turn 5.6).
But the gift share stayed at 0.604: absorbed on-corpus, unlearned
on-board. **The three teacher arms now support one theory: a lesson
survives only where self-play reward agrees with it** (teach2's
falken-descended pool punishes gifting and kept the lesson; kick1's
v3-lineage pool doesn't and PPO reversed the pull in exactly those
states). Evidence-pointed next construction (user's call, new
entry): kickstart pull + a punishing opponent at a fixed share PFSP
cannot fade (falken1 as anchor), and/or gift-scenario starts — make
the reward price the gift.

# TEACH2 EXTENSION TO 32K: THE LINE PARKS (2026-08-31)

The pre-registered third branch fired: decider **0.027** (flat with
8k's 0.035, v3's 0.086 unreached) with the mechanism **held** —
USSR DEFCON-loss share 0.185 on the board (baseline ~0.4),
retention 0.562 internally after 32k games. The lesson is durable;
self-play on the falken1-init prior sits in a basin (0.360 vs v3,
0.792 vs Greedy, vs falken1 *fell* to 0.415) that 24k extra games
did not leave. Full entry + decision in docs/JOSHUA.md.

**Evidence-pointed next construction (user's call, new entry):**
put the lesson into the champion instead of strength into the
student — v3-init with an auxiliary distillation loss toward the
harvested corpus (kickstarting; corpus at `runs/falken1/corpus`,
265,683 rows). Needs a small train.py aux-loss wiring. Alternatives:
stronger clone, anchor slot, the article (six arms + one mechanism
win of story now).

# TEACHER ARM 2 (falken1-init): NEGATIVE ON THE BAR, FIRST MECHANISM WIN (2026-08-31)

The teacher-as-prior line (entry in docs/JOSHUA.md, 2026-08-31):
clone probe falken1 vs its own teacher **0/39** (the clone is a
caricature — habits without the 15 s search); teach2 (falken1-init,
8k self-play) passed retention (v3's USSR DEFCON-loss share vs it
0.60) and strength (0.817 vs Greedy); decider **0.035** (bar 0.136,
v3 0.086) — sixth transfer negative, BUT the USSR DEFCON-loss share
vs the real AI fell to **0.135** (raw v3 ~0.4, teach1 0.58) at the
same game length: the gift-blunder class is gone for the first
time, replaced by uniform ≥20-VP track blowouts (the student is
just weak: 0.335 vs v3). Evidenced follow-on, needs the user's call
+ a new entry: continue the teach2 line well past 8k (still
climbing at the cap) and re-measure whether strength recovers while
the mechanism holds. Raw v3 stays the reported player.

# ROUND-3 OPTION A RAN AND CLOSED NEGATIVE (2026-08-30)

The user chose option A: relax SELF-PLAY-ONLY, the DLL as teacher.
The pre-registered entry, its full result and the decision are in
docs/JOSHUA.md ("relaxing SELF-PLAY-ONLY: the DLL as teacher, by
distillation"); tooling `wopr/distill.py` (docs/WOPR.md), corpus and
clone under `runs/falken1/`, the run under `runs/teach1/`.

What happened, in one breath: harvest 265,683 AI decisions from
1,853 clean easy bid-2 bridge logs (zero DLL hours); **falken1**
distilled to held-out top-1 0.610 (floor 0.178); the exploit gate
**passed** (v3's USSR-seat DEFCON-loss share vs falken1 = 0.50,
Greedy ~0 — the arm-1 objection answered); teach1 (v3-init + falken
pool-seeded, 8k games) passed the internal gate (vs Greedy 0.958)
and beat v3 0.580 — and the decider came back **0.009** (1/109)
against the easy AI, under raw v3's 0.086, USSR DEFCON-loss share
up at 0.58. Fifth internal-transfer negative. PFSP faded the weak
teacher to 5.1% of pool games; the entry names the two construction
suspects (clone strength, PFSP-vs-weaker-teacher) — each a new
entry. **Raw v3 stays the reported player.**

## What is next (needs the user's call)

- The **article** (the other round-3 option): docs/WOPR.md passes
  1–18 + five closed training arms are the outline; the transfer-gap
  story now has five arms of evidence.
- A reconstructed teacher arm, if wanted: fixed-share anchor slot
  for falken (PFSP can't fade it), a stronger clone (DAgger /
  more corpus), or live-DLL sparring (30–50 min a game) — each
  pre-registered fresh.
- Parked desync families (unchanged, traces standing): hidden-seat
  inference drift (315/411…), granted-Ops attribution (369/390),
  388's China-leftover, end-of-game turbulence (379, 324, 332),
  393's 1-VP U2 shape, Wargames corner (304). Next layout bump:
  OPTION_VOCAB fold + `u2_incident` slot.

## Quick commands

```sh
uv run pytest -q                                   # 541 pass, ~30 s
uv run python -m wopr.distill harvest --out <dir> <batch-dirs...>
uv run python -m wopr.distill train --corpus <dir> --out <run>
uv run python runs/falken1/gate.py 100 0           # the exploit gate
uv run python -m wopr.eval joshua=baselines/r3-bid2/v3/joshua.pt greedy \
    --games 400 --bid 2 --workers 6                # yardstick
uv run python -m wopr.playdek.eval --difficulty easy --games 120 --seed 300 \
    --bid 2 --policy joshua=<ckpt> --workers 8 --out runs/playdek/<name>
```
