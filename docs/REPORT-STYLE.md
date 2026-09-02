# Report style: the program brief

The wrap-up report on the WOPR/Joshua program is an executive brief: a
single HTML page in the format first produced on 2026-09-02 ("Joshua
Program Brief"). Produce one at a wrap-up (the program hits a wall, or
Joshua consistently beats Playdek's easy AI) and on request in between.
Keep this structure; change the numbers.

**Where things live.** The fact base is `runs/article/FACTS.md`; the last
brief is the template, `runs/article/joshua-brief.html` (both gitignored:
reports and the article are never committed). Every number traces to a
`docs/JOSHUA.md` entry, `docs/WOPR.md` (bridge passes), `baselines/README.md`
or a batch under `runs/playdek/`. Publish the page as an artifact and hand
over the link.

## Structure, in order

1. **Header.** Eyebrow `struggler / wopr · executive brief`; a title that
   names the goal as a sentence; a two-sentence standfirst (scope, then the
   one-line verdict); a stamp with the date, branch @ commit, rules version,
   test count and the eval definition.
2. **KPI row, four tiles.** The standing number against the easy AI (Wilson
   95% interval, wins/games, seats); where it started; the next bar for a
   training arm; the mechanism metric for the standing player (today the
   gift share). Each tile carries one sentence of context.
3. **Summary.** Two columns, four paragraphs: what we are doing and the
   wall; what moved the number; what did not (the negatives are the
   program's main product); the theory that survived plus the measurement
   side.
4. **Methodology.** One paragraph (pre-registration, the gates before DLL
   spend, the fixed decider, replication before a player stands, desync
   mining and voids, negatives written up with the same care) beside a
   definitions list: gift share, absorption, anchor, veto, bar, the bridge.
   Extend the list as terms enter the program.
5. **The record.** Two charts on shared rows, chronological, one legend:
   a dot-and-interval plot of every measured arm's decider mean against the
   easy AI (Wilson 95%; reference lines for the first baseline and every
   bar), and a bar chart of the mechanism metric on the same rows (its
   target as a dashed line). Two series only: raw checkpoint (blue) and
   inference-time rider (orange). Arms without a decider are listed in a
   sentence under the charts, never plotted as zero.
6. **The wall.** A scatter of internal strength (vs Greedy) against the
   board (vs the easy AI) for raw checkpoints with both numbers, direct
   labels, one hue; beside it two paragraphs on why internal gates screen
   for collapse and never for progress.
7. **What we learned.** Four to six bolded findings, each pinned to the
   arms that produced it.
8. **Promising ideas, ranked by the evidence.** Each with a rank tag, the
   evidence it rests on, a risk, and a cost line (compute, DLL hours,
   wiring). Close with "not promising on the evidence".
9. **Where we stand.** The standing table: player, USSR, US, mean, Wilson
   95%, mechanism metric, status chip. The standing player's row is tinted
   once. Then the goal statement and the infrastructure state (attrition,
   rules version, tests).
10. **Appendix: all arms in order.** Date, arm, construction in one line,
    decider, mechanism metric, outcome chip.
11. **Footer.** Sources and the eval conventions.

## Design tokens

- **Palette.** Page ground `#f2f4f3`, surface `#fcfcfb`, ink `#14202b`,
  secondary `#4f5b67`, muted `#7e8a96`, hairline `#d9dee2`. Series: raw
  checkpoint blue `#2a78d6`, rider orange `#eb6834` (the dataviz reference
  palette's first two slots; dark steps `#3987e5` / `#d95926`). Status
  chips carry text, never color alone: good `#006300` on `#e2f1e2`, bad
  `#a82a2a` on `#f7e4e4`, flat (closed at a gate, superseded, parked)
  `#4f5b67` on `#e7eaed`. Dark theme: ground `#101418`, surface `#1a1a19`,
  ink `#f2f4f6`, secondary `#c3c2b7`, muted `#898781`, hairline `#2c2c2a`.
  Tokens on `:root`, redefined under `prefers-color-scheme: dark` guarded
  by `:not([data-theme="light"])` and again under `[data-theme="dark"]`.
- **Type.** Newsreader (500) for the title, section heads and KPI values;
  Source Sans 3 for body; IBM Plex Mono for eyebrows, stamps, table
  headers, axis text and every figure column (`tabular-nums`). Google
  Fonts link with fallback stacks. Running text at most 68ch.
- **Layout.** 1120px column; two-column grids for summary, methodology and
  the wall; hairline borders, no shadows or radii; cards only for the KPI
  tiles, the figures and the ranked ideas. One highlight on the page: the
  standing player's tinted row.
- **Charts.** Inline SVG built by the page's own script from data arrays
  (`arms`, `pts`), so the next report edits data, not geometry. Marks:
  5px-radius dots with a surface ring, 2px whiskers, bars with a 4px
  rounded data end, dashed reference lines, chart text in ink tokens.
  Hover tooltips on every mark (arm, date, mean, interval, n, seats, the
  one-line reading). Wilson intervals computed from n where the entry
  recorded none, and the caption says so. ViewBox widths proportional to
  the grid columns so shared rows align.

## Writing register

Lead with the number and the verdict; one idea per sentence; no
narration of the work. Name arms by their notebook names (v3, kick2,
falken2 C). State the decider once and reuse it. The negatives get the
same weight as the positives, and every "promising idea" names the arm
whose evidence it rests on.
