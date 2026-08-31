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
