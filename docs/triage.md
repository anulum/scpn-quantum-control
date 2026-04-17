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
