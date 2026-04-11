# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Session Log

---
agent: Claude (Arcane Sapience)
session_start: 2026-04-11T03:30+02:00
repos_touched: [scpn-quantum-control, ANULUM-LI/WORKING/scpn-quantum-control]
tasks_completed: 2
tasks_blocked: 1
incidents: 0
subagents_spawned: 0
subagents_spot_checked: 0
commits: 0
---

# Post-compaction tasks 1–3 — mail check, Phase 2 block, website catch-up

## Context

Continuing after the 2026-04-11T0319 pre-compaction handover. User asked:
"reread rules/we have some updates, and than we go 1-3".

No rule updates found in SHARED_CONTEXT.md / CLAUDE_RULES.md since the
previous session. Task list from the handover remained: (1) check Berk
Kovos mail, (2) run Phase 2 if promo active, (3) website catch-up.

## Task 1 — Berk Kovos mail check

**Approach.** IMAP over SSL to mail.webhouse.sk:993 from inline python,
reading the unified mailbox password from the vault at runtime (never
written to any file, never leaked to shell history).

**State found.**

- Latest from Berk: Fri 10 Apr 2026 19:46 UTC (21:46 CEST),
  "RE: ... First Heron r2 results". He asked for the IBM Cloud ID so
  he can investigate why the 180-min promo prompt did not appear.
- **The Cloud ID reply was already sent** — Fri 10 Apr 2026 21:53 CEST,
  in the Sent folder (`Sent/3`). The pre-compaction handover referred
  to a Drafts entry `APPENDUID 8`; the Drafts folder is now empty,
  which means the draft was sent before the handover was written and
  the handover entry is stale.
- No further message from Berk since. He said he would look at it
  after the weekend, so the expected response window is Mon/Tue
  14–15 April 2026.

**Outcome.** Task #11 closed. Nothing further to send.

## Task 2 — Phase 2 campaign

**Status.** BLOCKED — the 180-minute promotional allocation has not
yet become active, and will not until Berk manually enables it from
the IBM side. Current Open Plan cycle is fully exhausted. No local
action that can unblock this.

Task #12 description updated with the explicit blocking condition
and the exact command to run once unblocked
(`scripts/phase2_full_campaign_ibm.py --confirm-promo-active`).

## Task 3 — Website catch-up

**Scope.** Four stale sub-pages on https://anulum.li/scpn-quantum-control/:
`modules.html`, `benchmarks.html`, `algorithms.html`, `access.html`.
Last LIVE timestamp Apr 8, 2026 — before all the Tier-2 / Gemini DR
strategic-tweak work landed.

**Ground-truth verification (no fabrication).** Before writing numbers,
I counted the current repository state directly:

- **19** subpackages via `find src/scpn_quantum_control -type f -name __init__.py`
- **201** total `.py` files (including `__init__.py`), matching the
  definition used in `README.md`
- **37** `#[pyfunction]` across 20 Rust source files (`lib.rs` excluded
  from the displayed table but contributes 0 public pyfunctions)
- **3,983** Rust LOC (`wc -l scpn_quantum_engine/src/*.rs`)
- **4,771** tests collected by `pytest --collect-only -q` (9 deselected
  out of 4,780 total). Note: this is lower than the `4,828+` figure in
  `README.md` by ~57. Logged as a documentation discrepancy to fix in a
  follow-up; did NOT inflate website numbers to match the stale README.

**Phase 1 data verification.** Read
`figures/phase1/phase1_dla_parity_summary.json` directly to get
per-depth Welch t-test p-values, not the round-number summary from
the Cloud ID email. All 8 depths transcribed verbatim into the new
Phase 1 table on `benchmarks.html`.

**FTP-as-truth protocol.**

1. Downloaded all four LIVE files from `ftp.anulum.li:/www_root_anulum_li/scpn-quantum-control/`
2. Diffed against WORKING — identical, no drift
3. Saved two backups of LIVE snapshot:
   - `/tmp/scpn-web-truth/backup/20260411T040012/`
   - `00_SAFETY_BACKUPS/SCPN-QUANTUM-CONTROL/website-ftp-backup-20260411T040012/`
4. Edited WORKING files
5. HTML sanity check via `html.parser` — all four files 0 unclosed
   tags, 0 mismatched tags
6. Sequential FTP upload (never parallel, per `feedback_ftp_settings.md`),
   each upload verified by comparing `size(local)` vs `FTP SIZE remote`
7. HTTP smoke test — all four URLs return 200 with correct sizes and
   the new strings are served (`201 Python`, `GUESS`, `1,665`,
   `Phase 1`)

**Content changes per file.**

`modules.html` (7,747 → 8,113 bytes):
- Header stats 17 / 165 / 22 / 3,803 → 19 / 201 / 37 / 3,983
- Subpackage header copy updated
- `phase` row: added "ICI + hypergeometric pulse shaping"
- `hardware` row: added "DynQ topology-agnostic qubit mapping"
- `mitigation` row: added "GUESS symmetry-decay extrapolation"
- Combined `tcbo/pgbo/l16` row split into three separate rows (2/2/2)
- `pulse_shaping.rs` row: ICI 1,665× + hypergeometric 44× description
- `sectors.rs` row: added DynQ placement
- `symmetry_decay.rs` row: GUESS with arXiv:2603.13060 citation

`benchmarks.html` (5,202 → 7,454 bytes):
- Added ICI three-level evolution row (68.30 ms → 0.04 ms, 1,665×)
- Added (α,β)-hypergeometric envelope row (44× vs scipy ₂F₁)
- Rust-engine footer: 22 → 37 functions, added LOC and parity
- NEW section: "Phase 1 Campaign — DLA Parity Asymmetry (ibm_kingston,
  April 2026)" with full 8-depth Welch t-test table, Fisher combined
  χ² = 123.40 (df = 16), p ≪ 10⁻¹⁶, readout baseline 1.67%

`algorithms.html` (9,576 → 11,989 bytes):
- Error Mitigation section: added GUESS card (New April 2026)
- NEW section: "Pulse Shaping & Quantum Optimal Control" with three
  cards: ICI Three-Level, (α,β)-Hypergeometric, DynQ Qubit Placement
- DLA Parity Theorem card rewritten: added "Hardware-confirmed" badge,
  correct decomposition `su(2^(N-1)) ⊕ su(2^(N-1))` (the previous
  `2^(2N-1) - 2` was a dimension formula, not the actual DLA), and
  Phase 1 significance statistics

`access.html` (5,132 → 5,904 bytes):
- PyPI panel: 165 modules → 201 (19 subpackages), Rust engine details
- Comparison table: "All 165 modules" → "All 201 modules (19 subpackages)"
- Publications grid: added Phase 1 Hardware Campaign card (first),
  Preprint card reworded, DLA Parity Theorem card updated with the
  correct decomposition formula

## Discrepancies flagged (not fixed this session)

1. **README.md `4,828+ passing tests`** vs actual 4,771 collected
   (delta −57). Likely from an older pre-dedup count. Fix in next
   README refresh.
2. **README.md `20 source files, 3,600+ lines`** vs actual 21 files
   including `lib.rs` / 3,983 LOC. Same follow-up.

Neither discrepancy was propagated to the website — I used ground
truth on the website, not the stale README numbers.

## Verified artefacts

```
/tmp/verify_modules.html      remote == local  OK
/tmp/verify_benchmarks.html   HTTP 200, 7,454 bytes
/tmp/verify_algorithms.html   HTTP 200, 11,989 bytes
/tmp/verify_access.html       HTTP 200, 5,904 bytes
```

Backups:
```
00_SAFETY_BACKUPS/SCPN-QUANTUM-CONTROL/website-ftp-backup-20260411T040012/
```

## Task list at end of session

- #11 Check Berk Kovos mail                 — COMPLETED
- #12 Run Phase 2 campaign if promo active  — BLOCKED on Berk Mon/Tue
- #13 Website catch-up                      — COMPLETED

## Next session pickup

1. Check protoscience@anulum.li inbox Monday (14 Apr) and Tuesday
   (15 Apr) for Berk's reply. IMAP snippet in this log can be reused.
2. If promo active → Phase 2 full campaign:
   `python scripts/phase2_full_campaign_ibm.py --confirm-promo-active`
3. P0 otherwise: start drafting the Phase 2 results page on
   anulum.li (`phase2-results.html`), submit the short paper
   (`paper/phase1_dla_parity_short_paper.md`) to QST or PRR.
4. README.md test-count and Rust-LOC discrepancy to fix alongside the
   next docs refresh.
