# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Session Log

---
agent: Claude (Arcane Sapience)
session_start: 2026-04-11T04:10+02:00
repos_touched: [scpn-quantum-control, ANULUM-LI/LIVE/scpn-quantum-control]
tasks_completed: 16
tasks_pending: 1  (#12 Phase 2, blocked on IBM promo)
incidents: 1  (nearly-edited WORKING instead of LIVE — caught by user reminder)
subagents_spawned: 0
commits: 0
ftp_uploads: 22  (13 content + 3 assets + 19 sequential uploads; 19 verified HTTP 200)
---

# Website expansion — all 16 items from the "lets do it all" plan

## Context

Following up on the post-compaction "tasks 1–3" session where I asked
whether more content should land on the website. User replied:
*"lets do it all :) (what goes than on todo is that we will link
webpage in github readme)"*. This session executes the full list.

## Scope delivered

| # | Deliverable | Lines / Bytes | URL |
|---|-------------|---------------|-----|
|   | Favicon PNG (300×200, local) | 52,007 B | `assets/favicon.png` |
|   | Phase 1 leakage figure (PNG) | 68,328 B | `assets/phase1_leakage_vs_depth.png` |
|   | Phase 1 asymmetry figure (PNG) | 58,129 B | `assets/phase1_asymmetry_vs_depth.png` |
| 1 | `phase1-results.html` | 20,687 B | interactive Plotly plot + full Welch table + BibTeX + Phase 2 status |
| 2 | `reproducibility.html` | 16,026 B | pinned deps, IBM job IDs, rerun protocol, honest limitations |
| 3 | `method-guess.html` | 14,290 B | GUESS theory, XY free-ride, Phase 1 calibration |
| 4 | `method-dla-parity.html` | 13,378 B | decomposition derivation, hardware confirmation, ruled-out alternatives |
| 5 | `method-pulse-shaping.html` | 14,737 B | ICI + hypergeometric, benchmarks, when to use which |
| 6 | `science.html` | 16,997 B | plain-language primer on SCPN/Kuramoto/DLA paired with equations |
| 7 | `timeline.html` | 13,123 B | past milestones, current blockers, planned Phase 2–4 |
| 8 | Inline SVG architecture diagram (in `index.html`) | — | clickable pipeline nodes linking to method pages |
| 9 | `index.html` refresh | 19,039 B | plain-language hero, limitations block, architecture SVG, 12-card Explore grid |
| 10 | Favicon + OG/Twitter meta on 9 existing pages | — | every page now unfurls correctly in Slack/X/email |
| 11 | Mobile table overflow (`style.css`) | +8 lines | `min-width: 520px` inside `.cmp-w`, `.nw` class for numeric columns |
| 12 | Nav restructure (`nav.js`) | — | Overview / Phase 1 / Science / Algorithms / Modules / Benchmarks / Hardware / Timeline / Access / Pricing |
| 13 | GitHub README website link | — | badge + "Richer Presentation" section with 7 direct entry points (uncommitted — awaiting CEO approval) |

## Ground-truth verified numbers

Counted directly from the working tree at commit `1b60f7b`:

- **19 subpackages** via `find src/scpn_quantum_control -name __init__.py`
- **201 `.py` files** total (including `__init__.py`), matching the
  README's long-standing definition. 181 strict (non-init) modules.
- **37 `#[pyfunction]`** across 20 displayed Rust source files (plus
  `lib.rs` with 0 exported functions)
- **3,983 Rust LOC** via `wc -l scpn_quantum_engine/src/*.rs`
- **4,771 tests collected** by `pytest --collect-only -q`
  (9 deselected, 4,780 total). The README still says `4,828+`;
  this discrepancy is now flagged on the website's honest-limitations
  block for the rolling documentation refresh.

## Phase 1 data transcribed verbatim

All per-depth numbers on `phase1-results.html` come directly from
`figures/phase1/phase1_dla_parity_summary.json`, not from paraphrased
email text. Verified readings:

| d  | reps | mean even | sem even | mean odd | sem odd | A(d)      | Welch t  | Welch p    |
|----|-----:|-----------|----------|----------|---------|-----------|----------|------------|
|  2 |  12  | 0.08065   | 0.00165  | 0.08272  | 0.00210 | –2.51 %   | –0.777   | 4.46 × 10⁻¹ |
|  4 |  21  | 0.09821   | 0.00173  | 0.08617  | 0.00115 | +13.98 %  |  5.803   | 1.45 × 10⁻⁶ |
|  6 |  21  | 0.12909   | 0.00309  | 0.10989  | 0.00180 | +17.48 %  |  5.372   | 6.61 × 10⁻⁶ |
|  8 |  21  | 0.14430   | 0.00310  | 0.12837  | 0.00172 | +12.41 %  |  4.496   | 8.89 × 10⁻⁵ |
| 10 |  21  | 0.16576   | 0.00217  | 0.14946  | 0.00227 | +10.91 %  |  5.181   | 6.67 × 10⁻⁶ |
| 14 |  21  | 0.18976   | 0.00309  | 0.17973  | 0.00197 |  +5.58 %  |  2.731   | 9.95 × 10⁻³ |
| 20 |  12  | 0.22945   | 0.00467  | 0.21139  | 0.00378 |  +8.55 %  |  3.009   | 6.66 × 10⁻³ |
| 30 |  12  | 0.27710   | 0.00569  | 0.25757  | 0.00367 |  +7.58 %  |  2.885   | 9.55 × 10⁻³ |

Fisher's combined: χ²(16) = 123.40, p = 0.0 (below numerical precision).

IBM job IDs (backend `ibm_kingston`) transcribed from the source JSONs:

```
d7ck79m5nvhs73a4nr10   phase1_bench
d7ck7hb0g7hs73dqvbg0   phase1_bench
d7ckcrh5a5qc73dosbmg   phase1_5_reinforce
d7ckft95a5qc73doseu0   phase2_exhaust
d7ckide5nvhs73a4o780   phase2_5_final_burn
```

Total QPU wall time across all five jobs: 263.3 s ≈ 4.4 minutes.

## Rule incidents

### Working in WORKING/ instead of LIVE/ (caught by user, fixed)

**L4 human review layer caught the drift.** I started the session by
editing files in `06_WEBMASTER/ANULUM-LI/WORKING/scpn-quantum-control/`
out of habit from the previous session. I made it as far as
`method-dla-parity.html` before the user reminded me that the rule
is: *work in LIVE, not WORKING, because LIVE holds the latest
FTP-downloaded truth*. The canonical source of this rule is
`feedback_gotm_working_untouchable.md` dated 2026-04-11 (CEO
clarification during the GoatCounter rollout).

**Evidence of the drift issue.** A file diff showed that LIVE's
`index.html` (10,433 B, Apr 11 03:36) was NEWER than WORKING's
`index.html` (9,556 B, Apr 11 01:18). If I had uploaded WORKING's
`index.html` to FTP I would have silently reverted whatever was
updated at 03:36. That was the whole point of the rule.

**Remediation.**

1. Verified LIVE/index.html bytes = FTP bytes via a direct FTP
   download into `/tmp/live-verify-index.html` → `diff -q` clean.
2. Copied the 4 new pages I had already written
   (`phase1-results.html`, `reproducibility.html`, `method-guess.html`,
   `method-dla-parity.html`) and the 2 modified shared files
   (`nav.js`, `style.css`) from WORKING to LIVE.
3. Left WORKING's stale `index.html` untouched. Edited LIVE/index.html
   directly for the architecture SVG, limitations block, and new
   hero text.
4. Continued all subsequent edits (`method-pulse-shaping.html`,
   `science.html`, `timeline.html`, the OG-meta pass, and the FTP
   upload) directly in LIVE.
5. No incident report filed: the error was caught before any FTP
   upload, so no exposure beyond the local filesystem.

**Takeaway:** the feedback memory is in my index but I defaulted to
WORKING because that is where I edited in the post-compaction session
when I did not yet have the LIVE-is-truth rule loaded. The user's
reminder took 30 seconds; my next edit was correct. The rule is
valuable — it prevented a silent revert of whatever the 03:36
update was.

## Artefact verification — HTTP smoke test

All 17 HTML URLs returned HTTP 200 with matching body sizes. All
three asset URLs returned HTTP 200 with matching sizes:

```
index.html                  200  19,039 B
phase1-results.html         200  20,687 B
reproducibility.html        200  16,026 B
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

## Known limitations / follow-ups

1. **README test count `4,828+` vs actual `4,771`.** Flagged on the
   website limitations block and on the reproducibility page. Fix in
   the next docs refresh, not this session.
2. **README Rust `36 functions, 3,600+ lines, 20 source files`** vs
   actual `37 / 3,983 / 21 incl lib.rs`. Same follow-up.
3. **Plotly CDN dependency.** `phase1-results.html` loads
   `plotly-2.35.2.min.js` from `cdn.plot.ly`. If the CDN is
   unreachable the interactive plot degrades silently (the two static
   PNGs above it still render). Acceptable for now; a local copy
   could be shipped with the site if needed.
4. **MathJax CDN dependency.** Same argument. MathJax is loaded on
   `phase1-results.html`, `science.html`, `method-guess.html`,
   `method-dla-parity.html`, `method-pulse-shaping.html`, and
   `index.html` (for the limitations block MathJax comment — added
   because the Explore card for DLA Parity uses `\mathfrak{su}`).
5. **README website link is uncommitted.** The edit is staged in the
   working tree but NOT committed — awaiting CEO approval per the
   "every push needs separate approval" rule. The change also does
   NOT fix the stale `4,828` badge; that is a separate scoped change.
6. **Independent replication.** Phase 1 DLA asymmetry is still
   single-device (ibm_kingston). Replication on `ibm_marrakesh` is
   planned in Phase 2.

## Task list at end of session

- #11 Check Berk Kovos mail                          — COMPLETED
- #12 Run Phase 2 campaign                           — BLOCKED (awaiting IBM promo)
- #13 Website catch-up (initial 4 pages)             — COMPLETED (prior cycle)
- #14 Backup all LIVE website files                  — COMPLETED
- #15 Read ground-truth for all new content          — COMPLETED
- #16 Upload Phase 1 PNG figures to FTP              — COMPLETED
- #17 Fix favicon + add OG/Twitter meta              — COMPLETED (9 pages)
- #18 Mobile CSS fix                                 — COMPLETED
- #19 Create phase1-results.html                     — COMPLETED
- #20 Create reproducibility.html                    — COMPLETED
- #21 Create method-guess.html                       — COMPLETED
- #22 Create method-dla-parity.html                  — COMPLETED
- #23 Create method-pulse-shaping.html               — COMPLETED
- #24 Create science.html                            — COMPLETED
- #25 Create timeline.html                           — COMPLETED
- #26 Create architecture SVG                        — COMPLETED (inline in index.html)
- #27 Limitations block + plain-language hero        — COMPLETED
- #28 Restructure nav.js                             — COMPLETED
- #29 HTML sanity + sequential FTP upload            — COMPLETED (all HTTP 200)
- #30 Add GitHub README link to deployed website     — COMPLETED (uncommitted)

## Next session pickup

1. **Berk Kovos mail on Mon/Tue (14–15 April).** Reuse the inline
   Python IMAP snippet from the previous session log to check for a
   reply.
2. **Phase 2 campaign** once promo is active:
   `python scripts/phase2_full_campaign_ibm.py --confirm-promo-active`.
   Expect ~11 minutes of QPU time, ~1,200 circuits.
3. **Commit the README website-link edit** once CEO reviews the diff.
   Proposed message:
   `docs(readme): link anulum.li/scpn-quantum-control website with 7 entry points`.
4. **Rolling documentation refresh** to fix the `4,828` and
   `36 / 3,600+ / 20` discrepancies now flagged on the website.
5. **Draft `phase2-results.html`** (skeleton only, data TBD) once
   Phase 2 runs.
