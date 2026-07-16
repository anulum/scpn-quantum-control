<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Count-Integrity Incident, April 2026

*Published 2026-07-16. This note makes the April 2026 internal incident
record public in one place; the canonical promotion status of every
artefact remains [the hardware status ledger](hardware_status_ledger.md).*

## What happened

In April 2026, internal review of the experimental "frontier" IBM workflow
lane (large-N, multi-job campaign automation) found two classes of invalid
output in its result files:

- **queued-job placeholders** — records written for jobs that had been
  submitted but never retrieved, presented in the same shape as retrieved
  results; and
- **fake all-zero fallback counts** — code paths that synthesised all-zero
  count dictionaries when real counts were unavailable, instead of failing.

No such output was promoted as public evidence, but the files existed in
the repository's results tree in the same formats as genuine artefacts,
which is exactly how silent contamination starts.

## Root cause

Workflow fallback code preferred producing *something* over failing
closed: when retrieval was pending or unavailable, it emitted placeholder
or zero-filled structures rather than raising. Aggregate-only summaries
(no raw counts attached) compounded the problem by being uncheckable
after the fact.

## Quarantine scope

The affected families are quarantined in the
[hardware status ledger](hardware_status_ledger.md) and must not be cited
as evidence until independently re-retrieved or reproduced from raw IBM
counts:

| Artifact family | Status |
|---|---|
| `results/ibm_hardware_v2_2026-03-29/` | Unpromoted (aggregate-only, no raw-count retrieval trail) |
| `results/ibm_hardware_2026-03-29/dla_parity_*.json` | Superseded (circuit-depth artefact; use `data/phase1_dla_parity/`) |
| `results/ibm_runs/jobs.json` and frontier queue outputs | Quarantined (placeholder / fallback-count handling) |
| Any "400 jobs", large-N frontier, multi-QPU, or live-loop claim | Not promoted |

## Controls in place since

- **Strict count coercion** — `hardware/_count_integrity.py` validates
  every count fail-closed (non-integral, negative, or opaque values
  raise); provider adapters were hardened for job lineage and status
  canonicalisation in the same wave.
- **Evidence rules** — the ledger's "no fallbacks" rule: all-zero counts,
  queued placeholders, aggregate-only JSON, and submit-now-retrieve-later
  files are quarantined, never evidence.
- **Preregistration gates** — campaigns since May 2026 commit the
  manifest (observable, circuit family, shots, abort criteria, statistics)
  before submission, and every promoted statistic must be recomputed from
  raw counts by a committed reproducer that exits non-zero on failure.
- **Hash-bound result packs** — hardware results ship as packs binding raw
  counts to manifests and verifiers.
- **Ledger promotion** — nothing is cited publicly without a ledger row
  naming its evidence class and artefact paths.

## What this does not affect

Promoted datasets are outside the incident scope: they carry raw counts,
retrieval manifests, and committed reproducers — the `ibm_kingston` DLA
parity Phase 1/2 datasets, the SCPN/FIM negative result, and the legacy
`ibm_fez` baseline artefact rows (quoted only where a committed artefact
is named).

## Why this note exists

The 2026-07-16 external due-diligence review correctly observed that the
quarantine was recorded in the ledger but the incident itself was never
stated publicly in plain language. Honest reporting of negative and
embarrassing results is this repository's operating method; that includes
its own process failures.
