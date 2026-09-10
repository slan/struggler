# KICK9 READ: THE BOARD'S OPENING MOVED WITH THE RE-MEASURE, THE US SEAT 0.138 ONE WIN UNDER ITS 0.15 LINE, THE MEAN 0.154 OVER; NO COMPOSE; STANDING UNCHANGED (2026-09-10)

- **kick9's decider** (docs/JOSHUA.md 2026-09-09, kick9 entry, "Result --
  the decider" and "Decision"; `runs/playdek/kick9-easy`, 120 easy games
  seeds 300+ bid 2, the raw checkpoint): **USSR 10/59 = 0.169, US 8/58 =
  0.138 [0.07, 0.25], mean 18/117 = 0.154 [0.10, 0.23]** (kick8 0.224 /
  0.052 / 0.138, kick2 0.190 / 0.089 / 0.140); attrition **3 desyncs, 0
  void** (all "illegal in Playdek", `desync-mining-2026-09-10-kick9.txt`,
  the dump batches' family). The batch was cut at 110/120 by a Windows
  reboot at 02:29; the ten missing games re-run 09:36-09:57 by
  `runs/playdek/resume_batch.py` (finishes any interrupted eval batch from
  its own config.json; summary.json carries `"resumed": true`). **By the
  letter: negative on the US read (one win short: 9/58 = 0.155), positive
  on the mean; no double clear -> no compose, no confirmation.** Standing
  unchanged: kick2+dump 0.263 (bar 0.313), kick2 raw 0.140. What it
  settles: the prior reaches the board's opening -- every opening read
  moved as the held-out re-measure predicted (`us_opening_diagnosis.py` on
  the batch vs kick8-easy, `us-opening-diagnosis-kick9-2026-09-10.md`:
  turn-3 VP **-8.1** vs -12.1, West Germany held at the end of turn 3 in
  **21/49** vs 6/42, at turn 1 47/60 vs 34/60, Blockade **26** plays vs 36,
  turn-3 losses **6** vs 18, alive at turn 3 49 vs 42) and the US seat's
  losses moved down the game (20/50 by turn 4 vs kick8's 30/55; 21 of the
  rest at turns 5-7, on VP, Europe Scoring still behind 36/45 at AR 4.6
  vs 3.7). The opening was worth ~0.09 on the US seat; the class did not
  close, it moved one turn block. The "position-bound" reading does not
  apply (the arena's sampled setup reached West Germany in 11/40 and the
  habit transferred). USSR seat 0.169 with DEFCON share 0.306 (the lowest
  raw), reported. **Open, the user's call (each a new entry):** (a) the
  compose over kick9 (dump second form, `dump=runs/kick9/joshua.pt`,
  seeds 300+ vs 0.313) -- not run by the rule; (b) the evidence-pointed
  follow-on, **the same prior extended down the game**: a `--from-logs`
  predicate at the US's first pick of turns 4-6, the learner at the mover,
  a mid-game diagnosis of the opening's shape run first.

# THE GIFT AUDIT CLOSES THE GIFT LINE; THE DUMP RIDER RUNS; THE TRAINING LINE TURNS TO THE US SEAT'S OPENING (2026-09-09)

- **The gift audit** (docs/JOSHUA.md 2026-09-09, first entry;
  `runs/playdek/gift_audit.py` re-runs it on any easy batch): kick8's 19
  USSR-seat DEFCON deaths replayed to the fatal decision. Detection is not
  the gap: the switch's `kill_options` proves the kill from the AI's side in
  18/19 at budget 80 (the training mask from the bot's side 0/19 at 80, 7/19
  at 4000). Distribution is not the gap: kick8's own self-play holds DEFCON 2
  by turn 1 in 70% of games (board 73%), 62% of the USSR's AR decisions are
  at DEFCON 2, the death shape occurs in 38/40 games, and the policy dumps a
  killer in a DEFCON-3 window 24% of the time (board 15%). Credit is: every
  death had 1-20 window decisions and 12/19 had the space race open for the
  card; the killer is carried 3-52 decisions and dies at a forced last AR.
  The bank cannot hold the lesson (predicate at DEFCON 2, the lesson at 3;
  8/19 fatal cards outside it; entries at turns 1-3 with a non-gift option in
  892/896). Ceiling: the class cured outright is worth ~0.03 pooled (0.258 ->
  ~0.29, bar 0.308). **The proposed arm (the switch over kick3's scenario
  starts) is not run; the gift line closes as a training target**; kick8's
  fourth reading stands, the entry's second reading corrected in the audit.
- **The dump** (docs/JOSHUA.md 2026-09-09, second entry, pre-registered):
  `SearchPlayer(dump=True)` = the veto plus a turn-horizon rider over
  `GIFT_CARDS` (bots/joshua/search.py; policy spec `dump=`; `--us
  joshua-dump`; tests/test_dump.py; suite 563). An unspaceable gift leaves
  in the first DEFCON-3 window; a gift that the turn's arithmetic says
  cannot be held is spaced (or UN-Intervened) at DEFCON 2. **Batch closed
  positive on reading (1)** (`runs/playdek/kick2-dump-easy`, 14:41-18:29):
  USSR 0.426 / US 0.172 / mean **0.295** [0.22, 0.38] vs kick2+veto's
  0.333 / 0.204 / 0.268 on the same seeds; USSR gift share **0/31**, US
  1/48; forced positions 2 in 60 games (kick8 22); attrition 7 desyncs +
  1 void (`desync-mining-2026-09-09-dump.txt`, known families). **The
  confirmation** (`runs/playdek/kick2-dump-easy-s500`, seeds 500+,
  18:30-22:07): USSR 0.278 / US 0.190 / mean 0.232 [0.16, 0.32], gift share
  3/39; attrition 7 + 1 void (`desync-mining-2026-09-09-dump-s500.txt`).
  **Pooled 59/224 = 0.263** [0.21, 0.32] (USSR 0.352, US 0.181; gift shares
  0.043 / 0.032) beats kick2+veto's 0.258 by the letter of the rule ->
  **kick2+dump is the standing player at 0.263; the bar re-sets to 0.313.**
  Inside the noise; the finding is the gift class gone at an unchanged mean.
  The three s500 deaths: two headline gifts (the AI's hidden headline
  finishes the drop), one deferral behind UN Intervention -> **the dump's
  second form** (no headline of a gift at DEFCON <= 3; UN Intervention no
  route at a window), tests 8, suite 567; the standing number is the first
  form's, every later `dump=` batch measures the second.
- **The US seat's opening** is the training line's target (US 45/58 losses
  on VP, 18 at turn 3, -11.9 VP by the end of turn 3). The diagnosis
  (`runs/playdek/us-opening-diagnosis-2026-09-09.md`, `us_opening_diagnosis.py`):
  Europe is lost in turns 1-3 by the bot's own hand -- 150 of the 213 US
  points lost from West Germany are its own Blockade plays (36, the discard
  refused 21/22), 71 of 97 Italian points its own Socialist Governments,
  Europe Scoring played from behind 39/46 at -5.1; the AI presses Western
  Europe (4.7 Ops points a game vs the arena USSR's 2.3) and the arena's
  sampled US setup leaves West Germany at 0.7 vs the argmax 4, so Blockade
  costs 0.5 a play in the arena and 4.1 on the board.
- **kick9** (docs/JOSHUA.md 2026-09-09, third entry, pre-registered; trained
  15:10-20:24, `runs/kick9`; **gates all passed** 20:28 -- diagnose 0.967,
  per-seat vs Greedy 0.960/0.980, absorption 0.502, switch 0.56/game,
  self-play USSR edge 0.517; the held-out re-measure moved: West Germany
  held 76/124 (kick8 45), Blockade plays 46 (65), turn-3 VP -7.8 (-8.7),
  the discard still refused 21/22; the sampled arena setup West Germany 1.2
  (kick2 0.7); probe as gifter 10/100 and 19/100, reported. **Decider
  read 2026-09-10** (the section above; `runs/playdek/kick9-easy`); success = US seat
  >= 0.15 AND mean >= 0.140; the in-run evals were off this run (config
  `eval_every` 0), a protocol slip): kick8's construction plus the board's
  opening as the prior -- `scenarios/us-opening-board.jsonl` (1,637 states,
  US to move at its first pick of turns 1-3, harvested by `wopr.scenarios
  --from-logs` from the 620 kick-era US-seat logs; 183 held out in
  `us-opening-board-heldout.jsonl`), `--scenario-frac 0.25
  --scenario-learner-mover` (new: the learner at the bank entry's mover, the
  seat mix's other draw kept), `--kill-switch`. Policy assumption stated in
  the entry: logged games as a state prior fall under the 2026-08-30
  amendment -- **confirmed by the user 2026-09-09 evening**; the run stands. Gates: kick8's (absorption, diagnose >= 0.9, no seat collapse; probe
  reported not gating) plus the arm's own reads --
  `runs/playdek/opening_remeasure.py` on the held-out states vs kick2 as
  USSR (baseline reading for kick8/kick2 in
  `runs/playdek/opening-remeasure-baseline.txt`) and the arena's sampled
  setup (`us_opening_diagnosis.py`). Decider: 120 easy games seeds 300+ bid
  2 on the raw checkpoint, success = US seat >= 0.15 AND mean >= 0.140;
  compose = the dump over kick9 (`dump=`) against the bar 0.308 only on a
  double clear. Suite 565.
- The brief: `runs/article/joshua-brief.html` (morning edition 2026-09-10,
  dated copy `-2026-09-10.html`, FACTS.md updated); publishing needs a
  session with the Artifact tool.

# KICK8 (THE KILL SWITCH) NEGATIVE ON BOTH READS AT 0.138 / 0.422; THE SWITCH WORKS IN THE ARENA, NOT ON THE BOARD; RULES 9 -> 10; ATTRITION 4/120 (2026-09-03)

- **kick8** (docs/JOSHUA.md 2026-09-03, "the kill switch"): kick2's
  construction with `--kill-switch` (new wiring: `kill_options` in
  bots/joshua/search.py = `defcon_kill_mask`'s mirror from the killer's side,
  `kill_probe(child, root=opponent)`; the backend narrows a learner row TO its
  kills and overrides a pool/anchor choice with the kill; `kills_per_game`).
  Cost: the killer's probe cannot exit early, so inside the killer's own play
  only Wargames is probed (the entry's named tightening, applied after a
  28.9 st/s smoke); 120 st/s, 8k games in 4.3 h, 0.58 firings a game. Gates
  on the RAW checkpoint all passed: in-run 0.86-0.99 vs Greedy (a player,
  unlike kick7's), diagnose 0.967, absorption 0.4941, anchor 0.753, self-play
  DEFCON-1 endings **23/120** (kick2 26, kick6 71), probe as gifter **4/100
  vs falken1** and **9/100 vs C** (the lowest ever on both scales). Decider
  (raw): USSR 0.224 / US 0.052 / mean **0.138** [0.09, 0.21] (kick2's 16/114
  with two more games), gift share **19/45 = 0.422** (kick2 22/45 = 0.489).
  Autopsy: all 19 USSR-seat DEFCON deaths are the switch's exact shape (the
  bot plays a US-event card at DEFCON 2, the AI takes the granted coup), zero
  self-kills. Readings: the switch works where the arena puts the learner;
  the board's gifts are positions the arena never reaches (the AI holds
  DEFCON at 2 through the mid-game); the caution learned is position-bound.
  Evidence-pointed next arm (user's call, new entry): **the switch over
  kick3's gift-scenario starts** (`--scenarios <defcon2_gift bank>
  --scenario-frac 0.25 --kill-switch` on kick2's flags) -- the states the
  board gifts in, the punishment certain in each. No compose. Standing
  unchanged: kick2+veto 0.258 pooled (bar 0.308), kick2 raw 0.140.
- **Bridge pass 23 + rules versions 9 and 10** (docs/WOPR.md; commits
  8af0a15, ab268ac, the v10 close): 346 = the DLL grants a UN Intervention
  play no Vietnam Revolts bonus *point* on Influence (v9) but keeps the +1 on
  its coup (v10, from kick8-easy 323 read off the new `coup` line); 411
  instrumented (`coup` evidence line: die, Ops, stability, modifier, margin,
  both boards; recent records on `_pick`'s illegal-in-engine fatal). Ladder
  re-rated on v9 and v10 (v3 vs Greedy 0.945/400, Greedy self ~0.53/0.47):
  stands. Harnesses green one at a time (`runs/playdek/pass23-verify.sh`;
  under Start-Process `uv` needs its full path).
- **Attrition 4/120, void 0** (`runs/playdek/desync-mining-2026-09-03-kick8.txt`,
  48 `coup` lines aboard): 323 (v9's own rule, fixed by v10), 350 (game-over
  timing: a held SE Asia Scoring the AI no longer had -- hand drift), 382 (the
  bot's We Will Bury You missing from the DLL's headline prompt after a
  reshuffle -- deal drift), 407 (the AI's 2-Ops play under Vietnam Revolts:
  DLL twice in Indonesia, engine once elsewhere -- granted-Ops attribution
  meeting the bonus arithmetic). Pass 24's input; open with trails: kick4's
  410 (Grain take carried as a return; the hand is the tell), 367, kick4's 312.
- Tooling: `runs/kick8-gates.sh` (raw gates with the probe gating).

# KICK7 (THE VETO TRAINED IN) NEGATIVE ON THE BAR AT 0.250; THE MASK DELETES CAUTION; ATTRITION 2/120 (2026-09-03)

- **kick7** (docs/JOSHUA.md 2026-09-02, "the veto trained in"): kick2's
  construction with `--veto-train` (new wiring: `defcon_kill_mask` /
  `kill_probe` in bots/joshua/search.py, applied to `opt_mask` by the arena
  backend; `vetoes_per_game` metric). Gates on the composed player: absorption
  0.5104, 26.8 struck options a game, anchor curve 0.844, veto-over-kick7 vs
  Greedy 0.995. Decider under the veto: USSR 0.220 / US **0.281** (best US
  seat ever; secondary read >= 0.25 met) / mean **0.250** [0.18, 0.34] -- the
  standing player's own number, bar 0.308 missed; composed gift share
  **0.304** (kick2+veto 0.19 / 0.08). Readings: (1) masking removes the
  caution it enforces -- the raw checkpoint is a maximal gifter (120/120 raw
  self-play games dead at DEFCON 1 by turn 2.6); (2) the composed player
  inherits the veto's gaps (hidden-card kills, budget), and a raw policy that
  prefers the gift walks through them; (3) the freed reward moved the US
  seat. Evidence-pointed next arm (user's call, new entry): **the kill
  switch** -- every self-play seat takes a provable win, so the gift is priced
  by reward in every game while the policy still sees the option
  (`kill_probe` from the killer's side). Standing unchanged: kick2+veto 0.258
  pooled (bar 0.308), kick2 raw 0.140.
- **Attrition 2/120, void 2** -- the lowest yet; bridge pass 22 measured. The
  two fatals (`runs/playdek/desync-mining-2026-09-03-kick7.txt`): 346 (the
  DLL asks the bot's Warsaw Pact choice while the engine is at a placement),
  411 (the AI's coup of Colombia the engine does not offer -- a Colombia
  influence drift, the granted-Ops attribution face). Pass 23's input.
- Tooling: `runs/kick7-gates.sh` (the composed strength gate through
  `wopr.search_eval --policy veto=...`).

# KICK6 CLOSED NO-GO AT THE PROBE, ZERO DLL HOURS; BRIDGE PASS 22 (2026-09-02, afternoon)

- **kick6** (docs/JOSHUA.md 2026-09-02, "the punisher that punishes"): kick2's
  construction with C (`runs/falken2/c/joshua.pt`) in the anchor slot.
  Gates: absorption 0.5077, Greedy 0.983, anchor curve 0.680 (C read as
  harder than B's 0.742); self-play **71/120 games ended at DEFCON 1**
  (kick2 26, kick5 20) — the champion learned C's offensive tactic (win by
  the opponent's DEFCON death), not caution. Probe: 9/100 vs falken1 (kick2
  6, kick5 8), **20/100 vs C** (the scale, probed first: kick2 15, kick5 8 —
  inverted against the board, the rule resolved blind: go <= 7, no-go >= 15)
  → **no-go**, the decider skipped, no compose. Reading: a punisher's
  density at a 10% share does not lower the gift; what transfers from a
  punisher is the punishment (rewarded in every game), not the caution
  (rewarded in the tenth). Standing unchanged: kick2+veto 0.258 pooled (bar
  0.308), kick2 raw 0.140.
- **Bridge pass 22** (docs/WOPR.md; commit f960f6d), from the kick5-easy
  trails, no DLL volume: the trapped AI's scoring-card play at the trap step
  never consumed (373/303 — the DLL's AI plays a scoring card under the trap
  where its UI denies the bot's seat the same play), the drift rescue
  skipping Grain Sales' "return" bookkeeping (300), Five Year Plan's read
  taking Grain Sales' draw for the discard (312). Instrument: a
  `random-discard` evidence line. Harnesses green (sweep 149/149, hotseat
  8/8, differ 12/12); suite 549. **Unmeasured**: no AI batch ran after it.
- Tooling: `runs/kick6-gates.sh` (`scale` mode probes kick2/kick5 vs C).

# FALKEN2 BUILT, KICK5 NEGATIVE ON THE KEY READ; BRIDGE PASS 21 + RULES V8; ATTRITION 4/120 (2026-09-02)

- **falken2 (stage 1, zero DLL hours; docs/JOSHUA.md 2026-09-02)**: merged
  corpus = falken1's shards + the eleven later easy batches (57 shards,
  2,973 games, 458,525 rows); falken1 on the merged fold 0.6008, line
  0.6208. Three fits, all over it: A (falken1's recipe) **0.6341**, C (A +
  AdamW 0.01, smoothing 0.05, lr halving) **0.6449**, B (hidden 384, 3 GNN
  layers, option 256 + C's regularization) **0.6536**. Every gate passed.
  Two findings: the clones **beat v3 head-to-head** at bid 2 (0.476 / 0.471
  for v3 vs A / C) while only ~0.65 vs Greedy — specialist punishers; and
  **fidelity inverts punishment** — top-1 B > C > A, but v3's DEFCON deaths
  per 100 games C 27 > A 19 > B 17 = falken1 17. Rule followed: falken2 = B.
- **kick5 (stage 2) closed negative on the key read**: gates passed
  (absorption 0.5025, Greedy 0.992; flag: USSR self-play edge 0.742; probe
  8/100 vs falken1, 12/100 vs its own anchor), decider mean **0.140**
  [0.09, 0.22] (USSR 0.211 / US 0.070 — kick2's number exactly), gift share
  **0.622** (kick2 0.489, kick4 0.465). The clone's fidelity was not the
  lever; its punishment density was falken1's. **Named follow-on (new
  entry, user's call): kick6 = kick2's construction with C in the slot.**
  No compose. Standing unchanged: kick2+veto 0.258 pooled (bar 0.308),
  kick2 raw 0.140.
- **Bridge pass 21** (kick4-easy's texts, no DLL volume): four roots — the
  stale "play" record of a Grain-handed card (330/372), a stale Grain draw
  misnaming a Five Year Plan discard (366), the DLL resolving the bot's
  Ortega free coup itself (308; coup/war rolls carry their side), the bot's
  narrowed decisions told uncut (370). Instruments: DEFCON + VP trails on
  every fatal, a `contest` evidence line (Summit modifiers).
- **Rules version 8**: Wargames ends on the VP total after the 6-VP gift,
  no final scoring (the card's "without Final Scoring"; the DLL agrees;
  the compose batch's four Wargames desyncs were this). Re-rated: v3 vs
  Greedy 0.945/400 bid 2, Greedy self 0.500/200 — the ladder stands.
- **Measured**: kick5's decider ran **4 desyncs / 120, void 2** — under the
  7–14 band for the first time. Its four fatals carry the trails
  (`runs/playdek/desync-mining-2026-09-02-kick5.txt`): 373 a two-step
  DEFCON drift pinned to seqs, 303 a 5-VP drift, 300 a reshuffle-boundary
  deal drift, 312 granted-Ops attribution. Next bridge pass reads them.
- Tooling: `wopr.distill harvest --workers`, `train --gnn-layers
  --option-hidden --weight-decay --label-smoothing --lr-decay`;
  `runs/playdek/decider_summary.py <batch>` (the standing readings);
  `runs/falken2/stage1.sh`, `runs/kick5-gates.sh`. Suite 549.

# KICK4 NEGATIVE — THE GIFT LINE PARKS FOR REVIEW; 20TH-PASS INSTRUMENTS ABOARD (2026-09-01)

- **kick4 (punisher seated in the scenario games; new
  `--scenario-vs-anchor` / `Arena(scenario_seats=)` wiring) closed
  negative on both reads**: mean 0.091 (US seat collapsed to 0.034 —
  a third of training vs the weak clone diluted the signal, the
  entry's named risk), gift share 0.465 ≈ kick2's 0.489. The probe
  voted no first (14/100 vs kick2's 6). Two arms now bracket the
  lever: 10% punisher/no states → 0.489 @ 0.140; 32% punisher in
  the states → 0.465 @ 0.091. **The gift share has a floor near
  0.45–0.5 no falken1 dose reaches at 8k games — the line parks for
  review.** Review candidates (record in the entry): a genuinely
  strong punisher (live-DLL sparring / deeper clone), longer runs,
  the layout bump, or accepting the veto as the gift's answer
  (kick2+veto's 0.081–0.188 shares are the only ≤0.25 ever) and
  aiming training at the other loss classes.
- **20th bridge pass (instruments, no behavior change)**: 382's dump
  decoded — the standing drift was three orphaned US placements, the
  granted-Ops attribution face. `granted-ops` evidence lines on
  every real `_answer_ops_type` resolution (diagnostic kind, 140
  fired in kick4's batch — every desync now carries 1–3) and a
  `_defcon_log` (seq, level) trail riding the DEFCON state diff for
  the game-over timing family. Next pass mines these.
- Standing: kick2+veto 0.258 pooled (bar 0.308), kick2 raw 0.140,
  suite 545.

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

- **The kill switch** (evidence-pointed by kick7): every self-play seat takes
  a provable win (the mirror of the training mask, `kill_probe` from the
  killer's side), so the learner is punished for every gift by its own kind
  and caution lives in the policy and value head, not in a mask. New entry.
- Other review candidates: longer runs on kick2's construction, the layout
  bump (`OPTION_VOCAB` fold + `u2_incident` slot), aiming training at the VP
  loss classes with the veto as the gift's answer.
- A broader-audience wrap-up, later (the user); `runs/article/FACTS.md` is its
  fact base, `docs/REPORT-STYLE.md` the brief's format.
- Bridge pass 24: 350 / 382 / 407 from kick8-easy (trails and `coup` lines
  attached); open with trails: kick4's 410, 367, kick4's 312.
- Hard mode: parked until easy is beaten (>0.5 both seats at bid 2).

## Quick commands

```sh
uv run pytest -q                                   # 557 pass, ~40 s
uv run python -m wopr.distill harvest --workers 12 --out <dir> <batch-dirs...>
uv run python -m wopr.distill train --corpus <dir> --out <run> [--gnn-layers 3 --option-hidden 256 --weight-decay 0.01 --label-smoothing 0.05 --lr-decay 0.5]
uv run python -m wopr.distill top1 <ckpt> --corpus <dir>   # held-out top-1 (absorption / fidelity)
uv run python runs/falken1/gate.py 100 0 <punisher.pt> [<gifter.pt>]   # the exploit gate / the mechanism probe
uv run python -m wopr.eval joshua=baselines/r3-bid2/v3/joshua.pt greedy \
    --games 400 --bid 2 --workers 6                # yardstick
uv run python -m wopr.playdek.eval --difficulty easy --games 120 --seed 300 \
    --bid 2 --policy joshua=<ckpt> --workers 8 --out runs/playdek/<name>   # the decider (launch detached: PowerShell Start-Process uv.exe)
uv run python runs/playdek/decider_summary.py runs/playdek/<name>   # the standing readings
bash runs/kick8-gates.sh                           # an arm's gates on the raw checkpoint (adapt the paths); runs/kick7-gates.sh for a composed player
uv run python -m wopr.search_eval --policy veto=<ckpt> --opponent greedy --games 200 --bid 2 --workers 8   # the composed player vs Greedy
```
