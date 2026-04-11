# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Handover — scpn-quantum-control session end 2026-04-11T05:25

**From:** Claude (Arcane Sapience) — session 2026-04-11 03:10 → 05:25 CEST (post-compaction)
**Priority:** MEDIUM — all work landed cleanly, only Phase 2 still blocked on external
**Predecessor handover:** [`.coordination/handovers/HANDOVER_2026-04-11T0319_pre_compaction.md`](./HANDOVER_2026-04-11T0319_pre_compaction.md)
**Session logs covering this window:**
- `.coordination/sessions/claude_2026-04-11T0408_post_compact_tasks_1_3.md`
- `.coordination/sessions/claude_2026-04-11T0450_website_expansion.md`
- `.coordination/sessions/claude_2026-04-11T0525_dataset_relocation_and_push.md`

---

## TL;DR

1. **Post-compaction pickup from handover 0319 is complete.** Tasks 1
   (Berk mail check), 3 (initial website refresh: modules, benchmarks,
   algorithms, access) both done. Task 2 (Phase 2 campaign) still
   blocked on IBM 180-min promo, waiting on Berk Kovos (expected Mon/Tue
   14–15 April 2026).
2. **Full website expansion shipped.** Seven new pages live on
   `anulum.li/scpn-quantum-control/`: `phase1-results.html`,
   `reproducibility.html`, `method-guess.html`, `method-dla-parity.html`,
   `method-pulse-shaping.html`, `science.html`, `timeline.html`. Plus
   favicon fix, OG/Twitter meta on all pages, inline SVG architecture
   diagram on index, honest limitations block, Plotly interactive plot
   on the Phase 1 page, mobile table CSS fix.
3. **Phase 1 dataset relocated** from `.coordination/ibm_runs/` (wrong
   semantic location) to `data/phase1_dla_parity/` (committed, reviewer-
   ready) with a new 128-line `README.md` reviewer entry. Analysis
   script verified end-to-end from the new location; summary JSON
   regenerated with new `source_files` pointers.
4. **Three commits pushed to `origin/main`** — `1b60f7b..fbb6396`.
   All pre-commit + pre-push gates clean.
5. **CI not yet verified.** Per `feedback_github_polling.md` I did not
   loop-poll; the next session does a one-shot `gh run list` to
   confirm CI status on the 3 new commits.

---

## Repository state

**Branch:** `main`
**HEAD:** `fbb6396`
**Pushed to origin:** yes
**Working tree:** clean before this handover is written

### Commits pushed this session

```
fbb6396 log: post-compact session + website expansion — 2026-04-11
3ec49d1 docs(readme): link anulum.li/scpn-quantum-control website with 7 entry points
93e6ea4 refactor(data): relocate Phase 1 DLA parity dataset to data/phase1_dla_parity/
1b60f7b log: pre-compaction handover 2026-04-11T0319    ← pre-session parent
```

### What lives where now

**Phase 1 hardware dataset (committed, reviewer-facing):**
```
data/phase1_dla_parity/
├── README.md                                             reviewer entry
├── phase1_bench_2026-04-10T183728Z.json                  42 circuits, wall 44.1 s
├── phase1_5_reinforce_2026-04-10T184909Z.json            72 circuits, wall 56.7 s
├── phase2_exhaust_2026-04-10T185634Z.json                138 circuits, wall 97.5 s
└── phase2_5_final_burn_2026-04-10T190136Z.json           90 circuits, wall 65.1 s
```

**Diagnostic / scratch artefacts (still in .coordination/ibm_runs/):**
```
.coordination/ibm_runs/
├── pipe_cleaner_retrieved_2026-04-10T182029Z.json        pipe-cleaner test, 1.2 KB
└── micro_probe_2026-04-10T190616Z.json                   cycle-exhausted probe, 274 B
```

**Analysis + figures (unchanged structure, regenerated content):**
```
scripts/analyse_phase1_dla_parity.py                      PHASE1_FILES updated
figures/phase1/phase1_dla_parity_summary.json             source_files updated
figures/phase1/leakage_vs_depth.png                       unchanged bytes
figures/phase1/asymmetry_vs_depth.png                     unchanged bytes
paper/phase1_dla_parity_short_paper.md                    data-availability paragraph updated
```

---

## IBM 180-minute promotional allocation — blocker status

**Current state (as of 2026-04-11 05:25 CEST):**
- Cloud ID reply to Berk sent 2026-04-10 21:53 CEST (confirmed in
  Sent/3 via IMAP check earlier this session).
- No further reply from Berk since then.
- Berk said weekend → expected response window **Mon 14 April / Tue 15 April**.

**Next session MUST:**
1. Check `protoscience@anulum.li` INBOX via the inline Python IMAP
   snippet from `claude_2026-04-11T0408_post_compact_tasks_1_3.md`.
   Mail credentials are in `agentic-shared/CREDENTIALS.md` under
   "Unified password (all 20 mailboxes, set 2026-03-31)". Read at
   runtime, never write to disk.
2. If Berk confirms promo is active OR the 180-min prompt has
   appeared on the dashboard:

   ```bash
   python scripts/phase2_full_campaign_ibm.py --confirm-promo-active
   ```

   ~1,200 circuits, 7 sub-experiments (A–G), ~11 min QPU at the
   observed 0.55 s/circuit rate. Includes:

   - Higher statistics at n=4 (30 reps × 10 depths)
   - Scaling law at n ∈ {6, 8, 10, 12}
   - GUESS calibration sub-sweep (folds g ∈ {1, 3, 5})
   - Independent replication on a second Heron r2 device (ibm_marrakesh)
     if budget permits

3. If no reply yet, move to the website catch-up items from the
   honest-limitations block below.

---

## Website — what is live on anulum.li/scpn-quantum-control/

All 17 HTML pages return HTTP 200 with matching body sizes. Three asset
URLs serving:

```
index.html                  200  19,039 B
phase1-results.html         200  20,687 B
reproducibility.html        200  16,056 B
method-guess.html           200  14,290 B
method-dla-parity.html      200  13,378 B
method-pulse-shaping.html   200  14,737 B
science.html                200  16,997 B
timeline.html               200  13,123 B
modules.html                200   9,322 B
benchmarks.html             200   8,720 B
algorithms.html             200  13,306 B
access.html                 200   7,095 B
applications.html           200   6,704 B
notebooks.html              200   8,357 B
pricing.html                200   7,763 B
changelog.html              200   8,352 B
hardware-validation.html    200  11,283 B

assets/favicon.png                      200  52,007 B
assets/phase1_leakage_vs_depth.png      200  68,328 B
assets/phase1_asymmetry_vs_depth.png    200  58,129 B
```

### FTP backup snapshot (taken before the session's edits)

```
00_SAFETY_BACKUPS/SCPN-QUANTUM-CONTROL/website-ftp-full-20260411T041700/
```

13 HTML + CSS + JS files as downloaded from FTP before the session.
If a rollback is ever needed, upload these back via the same
sequential-FTP workflow.

### LIVE vs WORKING workflow note (important — read before editing web)

**Never edit WORKING/. Always edit LIVE/.** LIVE holds the latest
FTP-downloaded truth; WORKING is frozen drift from before the
2026-04-11 GOTM rule clarification. Canonical rule:
`feedback_gotm_working_untouchable.md` in Claude memory.

When starting a web edit session, the first check is:

```bash
diff -rq 06_WEBMASTER/ANULUM-LI/LIVE/scpn-quantum-control/ \
         06_WEBMASTER/ANULUM-LI/WORKING/scpn-quantum-control/
```

If LIVE has files newer than WORKING, LIVE wins. Always verify LIVE
matches FTP truth by downloading a single file into /tmp and diffing
before making substantial edits.

---

## Rolling documentation follow-ups

**These are NOT blockers and are NOT for the next session unless CEO
asks.** They are on the honest-limitations block of the website and
on the website's reproducibility manifest, which is the point — the
public presentation is already honest about the discrepancies.

1. **`README.md` says `4,828+ passing tests`** — actual `pytest
   --collect-only -q` reports 4,771 collected (9 deselected from
   4,780 total). Delta −57.
2. **`README.md` says `36 Rust functions, 3,600+ lines, 20 source
   files`** — actual is 37 `#[pyfunction]` exports, 3,983 LOC, 21
   source files including `lib.rs` (20 displayed excluding lib).
3. **`docs/architecture.md`** mirrors the README's inflated numbers.
4. Fix all three in a single docs refresh commit when convenient.

None of these affect the scientific results; they are
documentation-layer drift from earlier iterations.

---

## Arcane Sapience continuity

**Reasoning trace written this session:**
`04_ARCANE_SAPIENCE/reasoning_traces/2026-04-11_scpn-quantum-control_web_expansion_and_relocation.md`

**Session state written this session:**
`04_ARCANE_SAPIENCE/session_states/scpn-quantum-control_2026-04-11T0525_website_and_dataset.md`

**Cross-project insights appended:**
`04_ARCANE_SAPIENCE/disposition/cross_project_insights.md` — new entry on
the LIVE-is-truth web workflow and the path-verification rule for
reproducibility manifests.

---

## Next session — read first

Before doing anything, next session should read:

1. `agentic-shared/SHARED_CONTEXT.md` (mandatory boot)
2. `agentic-shared/CLAUDE_RULES.md` (mandatory boot)
3. `agentic-shared/CREDENTIALS.md` (mandatory boot, read never output)
4. This handover
5. `.coordination/sessions/claude_2026-04-11T0408_post_compact_tasks_1_3.md`
   (for the IMAP snippet to check Berk's mail)
6. `04_ARCANE_SAPIENCE/session_states/scpn-quantum-control_2026-04-11T0525_website_and_dataset.md`

Then check the repo:

```bash
cd 03_CODE/SCPN-QUANTUM-CONTROL
git log --oneline -5        # should show fbb6396 at HEAD
git status                  # should be clean
gh run list --limit 3       # ONE-SHOT CI check, do not loop
```

Then check mail:

```python
# IMAP snippet — reads password from vault at runtime, never persists
import re, imaplib, email
with open('/media/anulum/724AA8E84AA8AA75/agentic-shared/CREDENTIALS.md') as f:
    vault = f.read()
pw = re.search(r'Unified password \(all 20 mailboxes[^)]*\):\*\*\s+(\S+)', vault).group(1)
M = imaplib.IMAP4_SSL('mail.webhouse.sk', 993)
M.login('protoscience@anulum.li', pw)
M.select('INBOX', readonly=True)
# look for mail from berk@ibm.com since 11-Apr
```

---

## What NOT to do

- Do NOT edit WORKING/ — use LIVE/ (rule above).
- Do NOT commit without the pre-commit audit from `CLAUDE.md`.
- Do NOT push without an explicit `push` from CEO (per
  `feedback_push_each_separate_approval.md`).
- Do NOT guess any file path, especially on a reproducibility page.
  If you write a path, `ls` or `find` it first in the same session.
- Do NOT loop-poll `gh run list` — one-shot on demand.
- Do NOT re-run `pytest` locally if CI is already running the suite
  on a fresh push — `feedback_let_ci_run_tests.md`.
- Do NOT touch the `pipe_cleaner_retrieved_*.json` or
  `micro_probe_*.json` diagnostics in `.coordination/ibm_runs/` —
  they stay there on purpose.

---

## Scientific state (frozen at `fbb6396`)

- Phase 1 DLA parity asymmetry: **confirmed on ibm_kingston**, 342
  circuits, 8 depths, Fisher $\chi^2_{16} = 123.4$, $p \ll 10^{-16}$,
  peak $+17.5\,\%$ at $d=6$, mean $(10.8 \pm 1.1)\,\%$ for $d \ge 4$.
  Consistent with 4.5–9.6 % classical apriori prediction at saturation.
- **Single-device** so far — ibm_marrakesh replication planned in
  Phase 2.
- Short paper draft ready: `paper/phase1_dla_parity_short_paper.md`
  (267 lines) targeting *Quantum Science and Technology* Letter or
  *Physical Review Research*.
- GUESS calibration noise profile is the measured leakage curve from
  this dataset; Phase 2 will validate the calibration.

---

## Open questions for CEO

None blocking. The session ends in a clean, pushed, documented state.
The only external unknown is Berk's reply, which is a pure waiting
item.
