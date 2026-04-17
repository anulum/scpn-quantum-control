# Issue Triage Policy

This document describes how issues and pull requests are routed,
prioritised and closed in `scpn-quantum-control`. It exists so that a
first-time contributor knows when to expect a response and what a
given label means.

Closes audit item B11.

## Cadence

Triage runs **weekly**, on Tuesday 09:00 CET. Off-week issues are still
read; they just might not get a label or owner until Tuesday.

Emergencies (hardware campaigns in-flight, a published release
regressing, a security disclosure) bypass the weekly cadence — ping
`protoscience@anulum.li` directly and the owner will reply within
24 hours.

## Label taxonomy

Every triaged issue gets one label from each of three axes.

### Kind (what it is)

| Label | Applied when |
|---|---|
| `bug` | The code does not match documented or obvious behaviour. |
| `enhancement` | New feature, new API surface, new algorithm. |
| `question` | User cannot get a thing to work; not clear whether it's a bug. |
| `docs` | Docs are wrong, missing, or unclear. |
| `infrastructure` | CI, packaging, release process, dependency management. |
| `security` | Vulnerability, hardening, supply-chain concern. |
| `science` | Claim correctness, reproducibility of a published number, Gap-2 / Gap-3 work. |

### Priority (how urgent)

| Label | SLA response | SLA closure | Typical examples |
|---|---|---|---|
| `P0-critical` | 24 h | 7 days | Data corruption, security leak, release blocker. |
| `P1-high` | 3 days | 30 days | Wrong scientific number, hardware campaign regression. |
| `P2-medium` | 1 week | 90 days | Usability annoyance, missing feature with a workaround. |
| `P3-low` | Triaged weekly | No SLA | Nice-to-have, cosmetic, speculative. |

### Status (where it is)

| Label | Meaning |
|---|---|
| `triage` | Auto-applied to every new issue until triaged. |
| `needs-info` | Waiting on reporter for a reproducer, log, or clarification. Stale after 30 days → closed with a polite note. |
| `accepted` | Triaged, prioritised, an owner may or may not be assigned yet. |
| `in-progress` | Someone is actively working on it. The owner commits to closing it or flipping it back to `accepted` within 30 days. |
| `blocked` | Blocked on external dependency (IBM quota, upstream library bug, CEO decision). |
| `good-first-issue` | Safe entry point for a new contributor. |
| `help-wanted` | Owner exists but welcomes parallel work. |
| `duplicate` | Closed with a link to the original. |
| `wontfix` | Closed by maintainer with a reason that goes in the close comment. |

## Routing

New issues are routed to an owner by subsystem:

| Area | Owner |
|---|---|
| `hardware/` (IBM Runtime, Heron, pennylane adapter) | Miroslav Šotek |
| `analysis/` (DLA parity, Koopman, BKT) | Miroslav Šotek |
| `bridge/` (Kuramoto-XY, SSGF) | Miroslav Šotek |
| `qsnn/` (spiking) | Miroslav Šotek |
| CI / packaging | Any maintainer |
| Rust engine (`scpn_quantum_engine`) | Any maintainer |
| Docs | Any maintainer |

(As of 2026-04-17 there is a single maintainer. Additional owners will
be listed here as collaborators join.)

## Closure criteria

An issue is closed when one of the following is true:

- The fix lands on `main` and a test locks in the expected behaviour.
- A follow-up task is queued in the roadmap and linked in the issue.
- The maintainer decides `wontfix` and explains why in the close comment.
- The reporter has not responded to a `needs-info` request for 30 days.

Closed issues remain indexed and searchable. Re-opening is welcome when
new evidence arrives.

## Security disclosures

Security issues use **private vulnerability reporting** in GitHub
Security Advisories, not the public issue tracker. See
`SECURITY.md` and `.well-known/security.txt` for the disclosure
process. Response SLA is 24 hours; fix SLA scales with severity per
`docs/THREAT_MODEL.md`.

## PR review

Pull requests get the same triage treatment as issues, plus:

- `CI green` is a pre-condition for merge. A red CI is blocking feedback;
  comments can still land.
- Maintainer review is typically ≤ 72 hours on P0 / P1, ≤ 1 week on P2,
  weekly cadence on P3.
- `co-authored-by` trailer is required on every commit (see
  `CONTRIBUTING.md`).

## Stale-bot policy

No stale-bot. Issues that rot do so deliberately — we'd rather leave a
real bug open than auto-close it and lose the signal. The `needs-info`
30-day close is the one automated exception.

---

## Worked examples per label

Abstract label definitions are easy to misapply. These examples are
drawn from issues opened against this repository (or identical
patterns we expect), with the final labels and closure trajectory.

### Example: `bug` + `P0-critical` + `in-progress`

*"Running `scripts/phase1/run_dla_parity.py --L 4` on a freshly
cloned repo raises `ModuleNotFoundError: scpn_quantum_engine` even
after `pip install -e '.[rust]'`."*

Labelling:

* **Kind** — `bug`: the documented install path is broken.
* **Priority** — `P0-critical`: blocks reproducibility of the
  published Phase 1 numbers.
* **Status** — `triage` → `accepted` → `in-progress`.

Response SLA: 24 h to acknowledge, 7 days to fix. Typical fix
trajectory: reproduce on a clean VM, identify whether the `[rust]`
extra is actually building the wheel (a `maturin develop` vs
`pip install -e .[rust]` mismatch is the usual culprit), land the
fix plus a regression test that runs on a fresh venv in CI.

### Example: `enhancement` + `P2-medium` + `accepted`

*"Please add a `--seed` flag to `scripts/phase1/run_dla_parity.py`
so two independent runs can start from the same classical initial
state."*

Labelling:

* **Kind** — `enhancement`: new CLI surface.
* **Priority** — `P2-medium`: workaround exists (edit the script
  directly), but the request is reasonable for reproducibility.
* **Status** — `accepted`, **no owner** — up for grabs.

Closure trajectory: a contributor picks it up, adds the flag plus a
regression test that the same seed produces the same JSON result,
merges. The issue closes automatically on the merge commit reference.

### Example: `question` + `P3-low` + `needs-info`

*"Is the DLA parity asymmetry reproducible on `ibm_brisbane`?"*

Labelling:

* **Kind** — `question`: user is asking, not reporting.
* **Priority** — `P3-low`: the Phase 1 campaign ran on
  `ibm_kingston`; extending to a second machine is interesting
  science but not a release blocker.
* **Status** — `triage` → `needs-info`: ask whether the user has
  run the reproducer themselves, what ZNE settings they used, and
  how many shots.

If the reporter does not respond in 30 days, the `needs-info` timer
auto-closes the issue with a polite pointer to re-open when data
arrives. If they respond with concrete numbers, the issue flips to
`science` + `P2` for investigation.

### Example: `security` + `P0-critical` (reported privately)

Never appears in the public tracker. See `SECURITY.md`. The flow is:

1. Private disclosure via GitHub Security Advisory draft.
2. Maintainer acknowledges within 24 h.
3. CVSS score is computed against `docs/THREAT_MODEL.md`; `High`
   and `Critical` get a private branch with a tight timeline
   (fix within 7 days or 30 days depending on CVSS).
4. Coordinated disclosure: advisory is published alongside the
   fix commit; the reporter is credited unless they opt out.
5. A public post-mortem lands in
   `.coordination/incidents/INCIDENT_*_post_mortem.md`
   (gitignored while in draft, promoted to the public
   `SECURITY_INCIDENTS.md` after 90 days).

### Example: `docs` + `P3-low` + `good-first-issue`

*"`docs/pipeline_performance.md` §21 still references the pre-v0.9
function name `build_kuramoto_coupling`; it was renamed to
`build_knm_paper27` in v0.9.0."*

Labelling:

* **Kind** — `docs`.
* **Priority** — `P3-low`: no user harm, just staleness.
* **Status** — `good-first-issue`: a single-line grep-and-replace
  PR is a classic newcomer entry point.

Closure: the contributor submits a PR, CI confirms the page builds
under `mkdocs build --strict`, maintainer merges within a week.

### Example: `infrastructure` + `P1-high` + `blocked`

*"Dependabot wants to bump `qiskit` from 1.4 to 2.3 but our
`qiskit-ibm-runtime` pin caps at 1.x."*

Labelling:

* **Kind** — `infrastructure`.
* **Priority** — `P1-high`: dep drift is a security surface.
* **Status** — `blocked`: we cannot take the bump until
  `qiskit-ibm-runtime` 2.x lands and we migrate. Tagged with a
  `blocked-by:upstream-runtime-2.x` sub-label.

The issue stays open. Triage revisits weekly to check whether the
upstream blocker has moved.

### Example: `science` + `P1-high` + `in-progress`

*"`docs/pipeline_performance.md` reports 95 % coverage for the
surface-code module but my local run shows 63 %."*

Labelling:

* **Kind** — `science` (reproducibility of a claim).
* **Priority** — `P1-high`: wrong scientific number.
* **Status** — `accepted` → `in-progress`.

Closure: reproduce locally, identify whether the stale number is
in the doc (doc fix) or the code regressed (code fix). Either path
lands a coverage-regression test so the same drift is detected
earlier next time.

## Label combinations and what they mean

Not every combination is sensible. This matrix documents the
ones we use and calls out a few that would signal a labelling
mistake.

| Kind × Priority | Meaning |
|---|---|
| `bug` × `P0` | Release blocker. Stops everything else. |
| `bug` × `P1` | Confirmed incorrect behaviour, workaround exists. |
| `bug` × `P2` | Edge-case bug a small fraction of users hit. |
| `bug` × `P3` | Curiosity bug — observed, but no user is blocked. |
| `security` × `P0` | Active vulnerability, coordinated disclosure. |
| `security` × `P1` | Hardening gap, no active exploit. |
| `enhancement` × `P0` | **Invalid combination.** If an enhancement is P0, it is actually a bug. Re-label. |
| `enhancement` × `P1` | Strategic feature with a committed timeline. |
| `question` × `P0` | **Invalid combination.** Either the user is blocked (bug) or they are asking about something low-priority. Re-label. |
| `science` × `P0` | Published number is wrong. Retract + fix. |
| `docs` × `P0` | **Rare but valid.** Used when a security procedure doc is wrong in a way that could lead users to leak credentials. |

## Flow diagrams

Textual flow for the modal path of an issue:

```
new issue
  ↓
triage label auto-applied
  ↓
Tuesday 09:00 CET (or emergency escalation)
  ↓
+---------------------+
| Kind decided        |
| Priority assigned   |
| Owner assigned      |
+---------------------+
  ↓
status: accepted
  ↓            (reporter clarification needed?)
  ├── yes → needs-info → (30 days) → auto-close
  ├── no
  ↓
status: in-progress
  ↓            (blocked externally?)
  ├── yes → blocked (weekly re-check)
  ├── no
  ↓
fix lands on main + test
  ↓
issue closes (auto-referenced by commit)
```

For pull requests, the flow is identical except the "fix lands" step
becomes "CI green + maintainer review + merge".

## Response time tracking

We do **not** run a response-time bot. The SLA table above is a
policy commitment, not a bot-enforced deadline. The expectation is
that a maintainer glances at the triage queue once a day. The
GitHub notifications inbox counts as the queue.

If the response SLA slips, the maintainer's next action is to
acknowledge late and offer a concrete next step — never silently
keep the timer running.

## Collaborators and access

As of 2026-04-17 the repository has one maintainer (Miroslav
Šotek) with write access, plus Dependabot and the Arcane Sapience
agent identity as GitHub Actions–scoped collaborators.

Paths toward further collaborator access:

1. Contribute three merged PRs that each add a test.
2. Request maintainer access in an issue; the CEO decides on a
   case-by-case basis.
3. Collaborators are listed publicly in `.github/CODEOWNERS`.

## Escalation

If the weekly-triage cadence is missed for more than two weeks in
a row, or if an open `P0-critical` has not been responded to in
72 hours, escalate in this order:

1. Second maintainer (when one exists).
2. CEO email: `protoscience@anulum.li`.
3. Public tweet tagging the maintainers (last-resort path, only
   after the first two have had 24 hours to respond).

## Relationship to other docs

* `CONTRIBUTING.md` — what a contributor does **before** opening an
  issue or PR (linting, coverage, Co-Authored-By trailer).
* `SECURITY.md` — private disclosure flow; triage entry point for
  any security label.
* `docs/THREAT_MODEL.md` — severity calibration for security
  issues.
* `docs/falsification.md` — where a `science` label is tested
  against a documented falsification criterion.
* `docs/language_policy.md` — which compiled-language tier a new
  compute function should use.
* `ROADMAP.md` — where `P2` / `P3` enhancements are staged when
  they are accepted but not yet scheduled.

## Changing this policy

Update this file in the same commit that changes the triage
behaviour (label rename, SLA change, routing update). A
documentation-only PR that does not change policy is not needed —
edit this file directly and land on `main`.

## Intentionally not covered

* **Code of Conduct** — the project Code of Conduct lives in
  `CODE_OF_CONDUCT.md`; triage does not enforce it, the
  maintainer does.
* **Commercial licensing questions** — those go to
  `protoscience@anulum.li` directly, not the issue tracker.
* **Research collaboration inquiries** — same as above.
* **Job applications** — we are not hiring; please do not open an
  issue asking.

## Revision history

| Date | Change |
|---|---|
| 2026-04-17 | Initial policy (closes audit item B11). |
| 2026-04-17 | Worked examples + label combination matrix + flow diagrams added under SUPERIOR-doc-standard remediation. |

