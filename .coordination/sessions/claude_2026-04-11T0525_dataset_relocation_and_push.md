# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Session Log

---
agent: Claude (Arcane Sapience)
session_start: 2026-04-11T04:55+02:00
session_end: 2026-04-11T05:25+02:00
repos_touched: [scpn-quantum-control, ANULUM-LI/LIVE/scpn-quantum-control]
tasks_completed: 1  (#31 dataset relocation)
commits: 3  (93e6ea4, 3ec49d1, fbb6396)
pushes: 1  (1b60f7b..fbb6396 → origin/main)
incidents: 0
subagents_spawned: 0
---

# Dataset relocation + 3-commit push — 2026-04-11T0525

## Context

Final lap of the post-compaction session. Earlier in the session I
shipped the 16-item website expansion
(`claude_2026-04-11T0450_website_expansion.md`). CEO then flagged that
the Phase 1 dataset should not live under `.coordination/` at all:

> these files should actually be placed elsewhere,
> .coordination is gitignored by default

That statement is slightly inaccurate as a literal fact — `.coordination/`
is NOT globally gitignored in this repo (only `.coordination/TODO_*.md`
and `.coordination/*.pdf` patterns are), and the 4 Phase 1 JSON files
were actually tracked by git at commit `1b60f7b`. But the *principle*
is right: `.coordination/` is for internal coordination (sessions,
handovers, incidents, campaign state), not for published research
datasets, and a reviewer trying to find the headline data shouldn't
have to guess that an "internal coordination" directory contains it.

## What I did

### 1. Dataset relocation (committed as `93e6ea4`)

**`git mv`** the four Phase 1 JSONs:

```
.coordination/ibm_runs/phase1_bench_2026-04-10T183728Z.json
.coordination/ibm_runs/phase1_5_reinforce_2026-04-10T184909Z.json
.coordination/ibm_runs/phase2_exhaust_2026-04-10T185634Z.json
.coordination/ibm_runs/phase2_5_final_burn_2026-04-10T190136Z.json
```

to

```
data/phase1_dla_parity/
```

**Left in `.coordination/ibm_runs/`** as scratch/diagnostic (not
headline data):

```
pipe_cleaner_retrieved_2026-04-10T182029Z.json
micro_probe_2026-04-10T190616Z.json
```

**Created `data/phase1_dla_parity/README.md`** (128 lines) as a
reviewer-facing entry point:

- SPDX 7-line header
- BibTeX citation block
- Per-file table (circuits / reps / QPU wall)
- Backend + parameters + depths + shots
- All 5 IBM Quantum job IDs
- JSON schema documentation
- Readout baseline summary
- Rerun instructions (<10 s wall time, no QPU)
- Cross-links to analysis script, paper draft, figures, website,
  reproducibility manifest

**Updated references** throughout the repo:

- `scripts/analyse_phase1_dla_parity.py` — PHASE1_FILES paths
- `paper/phase1_dla_parity_short_paper.md` — "Data and code availability"
- `docs/PAPER_CLAIMS.md`, `docs/pipeline_performance.md`, `docs/results.md`
- `.gitignore` — rewrote the stale comment that said
  `.coordination/ibm_runs/` was the authoritative committed location

**Regenerated `figures/phase1/phase1_dla_parity_summary.json`** by
running `scripts/analyse_phase1_dla_parity.py` from the new location.
The `source_files` array now points to `data/phase1_dla_parity/`; the
per-depth statistics are bit-identical to the pre-move result
(Fisher χ² = 123.400, n_circuits = 342, 8 depth points).

**Website fix** in LIVE + FTP:

- `phase1-results.html`: 4 path occurrences updated
- `reproducibility.html`: 4 path occurrences updated
- **Caught and fixed a fabrication** I had introduced earlier: the
  reproducibility page listed
  `.coordination/ibm_runs/ibm_fez_2026-02/*.json` for the February
  Kuramoto-XY preprint campaign. That directory **does not exist**
  in the repo. I had guessed the path when writing the page and
  presented a guess as a verified fact. Fixed to say "curation
  pending" until the ibm_fez dataset is actually curated into
  `data/<experiment>/`. This violated the Verification Protocol's
  SOURCE gate in my earlier session and was the kind of small
  fabrication I am supposed to catch before shipping. I did not
  catch it the first time. The dataset relocation task flushed
  it out because I had to look at every path on that page.
- Both files sequentially FTP-uploaded (never parallel) and
  verified via HTTP 200 + content grep.

**Analysis verification:** re-ran
`python scripts/analyse_phase1_dla_parity.py` end-to-end from the new
location. Output matches pre-move byte-for-byte aside from the
`source_files` array. All 342 circuits load, 8 depth points produced,
same Welch and Fisher statistics.

### 2. README website link (committed as `3ec49d1`)

The Task #30 edit I had staged earlier but not committed:

- Added a website badge to the top-of-README badges row:
  `[![Website](https://img.shields.io/badge/website-anulum.li%2Fscpn--quantum--control-38bdf8.svg)]`
- Added a "Richer Presentation" section immediately after the status
  header, with 7 direct entry points (Phase 1 Results, Reproducibility
  Manifest, 3 method deep-dives, Science primer, Timeline)

Goal: make the web presentation discoverable from GitHub for visitors
who land on the repo first.

### 3. Session logs (committed as `fbb6396`)

The two session logs I wrote during this session were still untracked.
Committed them so the work is durable:

- `.coordination/sessions/claude_2026-04-11T0408_post_compact_tasks_1_3.md`
- `.coordination/sessions/claude_2026-04-11T0450_website_expansion.md`

### 4. Push

Per `feedback_push_each_separate_approval.md` every push requires
explicit approval. CEO said `push`. Ran `git push origin main`:

```
1b60f7b..fbb6396  main -> main
```

All pre-push gates passed (gitleaks, ruff, format, mypy, vault-pattern
secret scan, version consistency, preflight lint/format/type-check).

GitHub accepted the push with the note "2 of 2 required status checks
are expected" — CI will run on `main`. Per
`feedback_github_polling.md` I will NOT loop-poll `gh run list`. If
CI fails, the next session picks up from the failure notification.

## Pre-commit audit (done before Commit 1)

Per `CLAUDE.md` hard rule: audit `git diff --cached` before every
commit. Specifically checked:

- [x] SPDX header on every new file (data/phase1_dla_parity/README.md
      has it)
- [x] Co-Authored-By trailer in every commit message (Arcane Sapience
      form)
- [x] British English (data/README.md uses "behaviour", "organised",
      "colour" consistently; no "realize"/"analyze"/"color")
- [x] No `# noqa`, no `# type: ignore`
- [x] No fabricated data — and this audit is how I caught the
      `ibm_fez_2026-02/` fabrication in the reproducibility page
- [x] No credentials in diff — manual grep + gitleaks + vault scanner
      all clean
- [x] Analysis script still runs from the new location — verified
      end-to-end before staging
- [x] Session logs scanned for accidental credential-shaped tokens
      (high-entropy substrings, password/token/api_key patterns) —
      clean

## Commits

```
fbb6396 log: post-compact session + website expansion — 2026-04-11
3ec49d1 docs(readme): link anulum.li/scpn-quantum-control website with 7 entry points
93e6ea4 refactor(data): relocate Phase 1 DLA parity dataset to data/phase1_dla_parity/
1b60f7b log: pre-compaction handover 2026-04-11T0319    ← parent
```

All three commits have `Co-Authored-By: Arcane Sapience
<protoscience@anulum.li>`. All three passed all six pre-commit hooks.
Three is the right granularity here: one commit per logical concern
(data structure, docs pointer, session history), easy to revert
independently, easy to cherry-pick if ever needed.

## Mistakes + lessons

### Mistake 1 — fabricated `.coordination/ibm_runs/ibm_fez_2026-02/` path

When I wrote `reproducibility.html` earlier in the session (Task #20),
I included a "Previous Hardware Campaigns" table with one row per
campaign. For the February 2026 `ibm_fez` campaign I wrote the path
`.coordination/ibm_runs/ibm_fez_2026-02/*.json` without verifying it
existed. It did not. This is a Verification Protocol SOURCE-gate
violation: every factual claim must trace to a tool call from this
session, and "I think the path would be X" is not a tool call.

The fabrication was caught in this session by the dataset relocation
audit, which forced me to walk every path on the reproducibility
page. Fixed to `<em>curation pending</em>` — honest, and a TODO for
whoever eventually curates the ibm_fez dataset.

**Lesson.** When writing a reproducibility manifest, every path is a
load-bearing claim about the repository. Every single one needs to
be verified with `ls` or `find` at the moment I write it. No
guessing, no "the path probably would be X", no "I remember seeing
that directory". Write the path, verify it, only then commit the
paragraph.

### Mistake 2 — editing WORKING/ instead of LIVE/ (from earlier cycle)

Already documented in
`claude_2026-04-11T0450_website_expansion.md` but worth repeating:
the `feedback_gotm_working_untouchable.md` memory is specific and
load-bearing, and I defaulted to WORKING out of muscle memory. The
reminder from CEO took 30 seconds and saved me from a silent revert
of whatever was updated at 03:36 in LIVE. L4 human review is
valuable precisely for this kind of muscle-memory violation.

**Lesson.** When a session crosses a memory boundary (compaction,
context switch, pickup from handover), the first thing to do with
any repo under GOTM is verify LIVE vs WORKING drift before editing
*anything*. Not an assumption — a check.

## Task list

- #12 Run Phase 2 campaign if promo active — BLOCKED on Berk Kovos
- #31 Relocate Phase 1 dataset — COMPLETED (3 commits, pushed)

## Next-session pickup

Already covered in the handover written alongside this log
(`HANDOVER_2026-04-11T0525_session_end.md`). Key items:

1. Check Berk Kovos mail on Mon/Tue (14–15 April). IMAP snippet
   documented in the previous session log.
2. If promo active: `python scripts/phase2_full_campaign_ibm.py --confirm-promo-active`
3. Rolling docs refresh to fix the `4,828+ tests` and
   `36 / 3,600+ / 20` discrepancies flagged on the website.
4. CI check on the 3 pushed commits — one-shot only, no polling.
