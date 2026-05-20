<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- GOTM-SCPN Paper 0 methodology paper outline -->

# GOTM-SCPN Paper 0 Methodology Paper Outline

Status: tracked outline, source-bounded, dated 2026-05-19.

Working title:

> Source-Bounded Experimental Translation for SCPN: From Foundational
> Manuscript Ingestion to Executable Validation Gates

This outline defines the methodology paper enabled by GOTM-SCPN Paper 0: The
Foundational Framework (Paper 0) ingestion. The paper reports the translation process, artefact contracts, and promotion gates.
It must not present GOTM-SCPN Paper 0: The Foundational Framework source
ingestion, generated fixtures, simulator outputs, or lane-registry rows as external scientific validation.

## Submission boundary

| Boundary | Requirement |
|---|---|
| Evidence class | Use `source-bounded`, `fixture-backed`, `measured-system candidate`, `hardware candidate`, or `blocked`; do not use stronger evidence language without a named passing gate. |
| Hardware | No IBM, QPU, or hardware-readiness claim unless a frozen manifest and gate artefact are cited. |
| Paper 0 authority | GOTM-SCPN Paper 0: The Foundational Framework is the upstream programme source for this method paper. Paper 27 is only a bounded implementation candidate when explicitly named. |
| Artefact traceability | Every result table, figure, or claim-boundary statement must cite a repository artefact path and regeneration command. |
| External validation | External validation requires named data, units, uncertainty, null models, falsifiers, and a passing promotion artefact. |

## Core thesis

Foundational manuscript claims can be translated into reproducible software
artefacts without overclaiming their empirical status. The method separates
source accounting from validation by forcing each claim through staged artefacts:
source records, claim candidates, validation specs, fixture preservation,
promotion gates, lane registry rows, preregistered experiment designs, and only
then measured-system or hardware evidence.

## Claimed contributions

| Contribution | Repository evidence | Claim boundary |
|---|---|---|
| Ledger-bound source ingestion | `docs/paper0/paper0_validation_register.md`; `scpn_quantum_control.paper0` | Complete source accounting for GOTM-SCPN Paper 0: The Foundational Framework, not external validation. |
| Reproducible spec and fixture preservation | `scpn_quantum_control.paper0.spec_loader`; generated GOTM-SCPN Paper 0 validation modules | Fixture preservation and loader determinism only. |
| Public downstream programme registry | `docs/paper0/paper0_lane_registry.md`; `data/paper0_lane_registry.json`; `scpn-bench paper0-lane-registry-gate` | Source-bounded programme planning, not hardware readiness. |
| First preregistered replay gate | `docs/paper0/paper0_knm_preregistered_replay.md`; `data/paper0_knm_preregistered_replay.json`; `scpn-bench paper0-knm-preregistered-replay-gate` | Deterministic no-QPU replay with fail-closed promotion decision plus measured-candidate unit-class gate; blocked and non-closing until measured coupling magnitudes with uncertainty exist. |
| Stable release-gate pattern | `docs/release_readiness.md`; `scpn-bench stable-core-release-gate` | Software reproducibility and claim-boundary discipline only. |
| Method for choosing next experiments | `docs/paper0/paper0_experimental_pathway.md` | Candidate prioritisation; no measured-system result yet. |

## Proposed paper structure

### 1. Introduction

Purpose: state the problem of translating a broad foundational framework into a
reproducible experimental programme without promoting unsupported claims.

Required artefacts:

- `docs/paper0/paper0_validation_register.md`
- `docs/paper0/paper0_experimental_pathway.md`
- `docs/paper0/paper0_lane_registry.md`
- `docs/paper0/paper0_first_preregistered_downstream_experiment.md`
- `docs/paper0/paper0_knm_preregistered_replay.md`

Acceptance gate: introduction must state that GOTM-SCPN Paper 0: The
Foundational Framework ingestion is source-accounting and fixture preservation, not external validation.

### 2. Source hierarchy and claim boundary model

Purpose: define how GOTM-SCPN Paper 0: The Foundational Framework, Paper 27,
generated repository artefacts, and future measured-system evidence relate.

Required artefacts:

- `AGENTS.md`
- `docs/paper0/paper0_validation_register.md`
- `docs/paper0/paper0_experimental_pathway.md`

Acceptance gate: any reference to Paper 27 must say it is a bounded
implementation candidate, not the definitive programme source after GOTM-SCPN
Paper 0: The Foundational Framework.

### 3. Ledger-bound ingestion pipeline

Purpose: describe source records, ledger spans, generated validation modules,
claim-candidate extraction, and fixture preservation.

Required artefacts:

- `docs/paper0/paper0_validation_register.md`
- `scpn_quantum_control.paper0`
- `scpn_quantum_control.paper0.spec_loader`

Acceptance gate: every described ingestion output must name a repository path or
package namespace.

### 4. Validation-spec and fixture contract

Purpose: describe JSON spec bundles, Markdown reports, Python loaders, and tests
as reproducibility contracts.

Required artefacts:

- `paper/gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/source_validation_artifacts/`
- `scpn_quantum_control.paper0.spec_loader`
- `tests/test_paper0_lane_registry.py`

Acceptance gate: fixture-backed claims must be described as preservation checks,
not as empirical confirmation.

### 5. Promotion gates and evidence classes

Purpose: define the transition rules from source-bounded evidence to stronger
classes.

Evidence classes:

| Class | Minimum condition | Allowed claim |
|---|---|---|
| Source-bounded | Source record, spec, or lane exists. | The claim has been ingested and bounded. |
| Fixture-backed | Fixture and loader pass deterministic checks. | The source-derived fixture is reproducible. |
| No-QPU validated | Classical/simulator/null-model gate passes. | A bounded non-hardware protocol passed. |
| Measured-system candidate | Named dataset protocol exists with units and uncertainty. | The lane is ready for measured-system replay. |
| Hardware candidate | Frozen manifest, no-QPU gate, and claim-boundary artefact pass. | The lane is eligible for bounded hardware review. |
| Blocked | Missing falsifier, units, data, manifest, or boundary. | No stronger claim is allowed. |

Acceptance gate: each evidence-class transition must include a falsifier or
blocker condition.

### 6. GOTM-SCPN Paper 0 lane registry as programme control surface

Purpose: present the generated lane registry as the current public control
surface for selecting next experiments.

Required artefacts:

- `data/paper0_lane_registry.json`
- `docs/paper0/paper0_lane_registry.md`
- `scripts/run_paper0_lane_registry_gate.py`
- `scripts/run_paper0_knm_preregistered_replay.py`
- `scripts/compare_paper0_knm_preregistered_replay.py`
- `scripts/run_paper0_knm_preregistered_replay_gate.py`

Regeneration command:

```bash
scpn-bench paper0-lane-registry-gate
```

Acceptance gate: the lane registry must be described as a programme-planning
index, not a result table.

### 7. Release and reproducibility controls

Purpose: show how release gates prevent public documentation and package
surfaces from drifting away from committed artefacts.

Required artefacts:

- `docs/release_readiness.md`
- `scpn-bench stable-core-release-gate`
- `scpn-bench paper0-lane-registry-gate`

Acceptance gate: release checks must be framed as reproducibility and
claim-boundary checks, not scientific validation.

### 8. First downstream experiment selection

Purpose: define how the next measured-system lane will be selected after the
method paper outline.

Candidate selection criteria:

| Criterion | Requirement |
|---|---|
| Source traceability | Lane maps to GOTM-SCPN Paper 0: The Foundational Framework source spans or a stated external assumption. |
| Observable | The lane names measurable variables and units. |
| Data path | The lane names a dataset, acquisition protocol, or public benchmark. |
| Negative path | The lane includes a falsifier and at least one blocker. |
| Rebuild path | The lane has a command or script that can regenerate artefacts. |
| Hardware boundary | Hardware remains blocked until no-QPU gates and manifest boundaries pass. |

Acceptance gate: this section must not select hardware execution as the first
experiment.

### 9. Limitations

Required limitations:

- GOTM-SCPN Paper 0: The Foundational Framework ingestion does not validate Paper 0 propositions.
- Generated fixtures do not replace external datasets.
- Lane registry rows do not establish physical or biological truth.
- Hardware execution remains blocked until preregistered gates pass.
- The current registry is intentionally small and should expand only through
  reproducible lane additions.

### 10. Conclusion

Purpose: argue that the repository now provides a repeatable translation method
from source-bounded manuscript material to guarded experimental work.

Acceptance gate: conclusion must describe a method and programme architecture,
not a completed empirical validation campaign.

## Required figures and tables

| Output | Purpose | Source artefact | Regeneration command |
|---|---|---|---|
| Figure 1: evidence-class ladder | Shows source-to-validation transitions. | This outline plus `docs/paper0/paper0_experimental_pathway.md` | Manual diagram from tracked table; cite source path. |
| Figure 2: GOTM-SCPN Paper 0 artefact flow | Shows ledger, specs, fixtures, lane registry, gates. | `docs/paper0/paper0_validation_register.md`; `docs/paper0/paper0_lane_registry.md` | `scpn-bench paper0-lane-registry-gate` for registry rows. |
| Table 1: evidence classes | Defines allowed claim language. | Section 5 of this outline | Generated from tracked Markdown table. |
| Table 2: current lane registry | Lists current lanes and blockers. | `data/paper0_lane_registry.json` | `scpn-bench paper0-lane-registry-gate` |
| Table 3: first preregistered K_nm replay | Reports primary EEG candidate, negative control, blockers, input digests, null diagnostics, and fail-closed promotion decision. | `data/paper0_knm_preregistered_replay.json`; `docs/paper0/paper0_knm_preregistered_replay.md` | `scpn-bench paper0-knm-preregistered-replay-gate` |
| Table 3: release gates | Lists reproducibility commands. | `docs/release_readiness.md` | `scpn-bench stable-core-release-gate`; `scpn-bench paper0-lane-registry-gate` |

## Minimum acceptance checklist before drafting prose

- `scpn-bench paper0-lane-registry-gate` passes.
- `scpn-bench stable-core-release-gate` passes if stable-core release-gate text
  or stable-core claim surfaces are cited.
- Every figure and table names a repository source artefact.
- Every claim stronger than source-bounded evidence names a promotion gate.
- Hardware/QPU claims remain blocked unless a later manifest explicitly opens
  them.
- Draft prose avoids implying that fixture preservation is external validation.

## Immediate next production task

Implement the no-QPU replay artefacts for the first preregistered downstream
experiment design in `docs/paper0/paper0_first_preregistered_downstream_experiment.md`.
The first design is measured-system oriented and claim-bounded before any
hardware spend is considered.


### Fail-closed promotion safety for Table 3

The first preregistered K_nm replay is not only a regenerated report. Its gate
performs payload-level invariant checks in
`scripts/compare_paper0_knm_preregistered_replay.py` before accepting the
committed JSON/Markdown pair. The comparator rejects replay payloads that
promote the lane, authorise hardware submission, drop the QPU blocking gate,
remove required evidence, omit falsifiers, or carry stale input SHA-256 digests
even if the expected and generated JSON are byte-aligned. This keeps the
methodology paper's first downstream experiment reproducible without silently
upgrading its claim class.
