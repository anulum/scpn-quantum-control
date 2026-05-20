# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — GOTM-SCPN Paper 0 Experimental Pathway

# GOTM-SCPN Paper 0 Experimental Pathway

Status: source-bounded programme plan, dated 2026-05-18.

This page turns the completed GOTM-SCPN Paper 0: The Foundational Framework
(Paper 0) source-validation register into a longer-term experimental pathway and a methodology-paper route. It does not
promote Paper 0 propositions into external scientific validation. It defines how
source claims move from manuscript ingestion to executable validation artefacts,
promotion gates, falsifiers, and candidate experiments.

## Source hierarchy

GOTM-SCPN Paper 0: The Foundational Framework is the upstream framework source
for long-horizon SCPN translation work in this repository. It supplies the canonical source register, domain map,
validation-spec extraction targets, and promotion-gate candidates.

Paper 27 is not the definitive source of truth for the programme after
GOTM-SCPN Paper 0: The Foundational Framework ingestion. In this repository, Paper 27 remains a historical implementation
candidate for the built-in 16-layer coupling matrix and related examples. It can
support a specific model instantiation only when its assumptions are stated and
bounded.

Repository artefacts are the executable validation layer. Generated specs,
fixtures, loaders, tests, and audit reports preserve source-accounting evidence
and decide whether a claim is still source-bounded, fixture-backed,
measured-system-backed, hardware-backed, or blocked.

## Methodology paper route

Working title:

> Source-Bounded Experimental Translation for SCPN: From Foundational
> Manuscript Ingestion to Executable Validation Gates

The methodology paper should report the process, not overclaim the scientific
content. Its core contribution is a reproducible translation method:

1. Ledger-bound manuscript ingestion with stable source identifiers.
2. Claim extraction into validation specifications with provenance.
3. Separation of fixture preservation from claim promotion.
4. Promotion gates that require units, uncertainty, falsifiers, negative paths,
   and named external datasets before stronger evidence classes are used.
5. Reproducible artefact contracts for JSON, Markdown, Python loaders, and
   regression tests.
6. Explicit claim boundaries for no-hardware, simulator, measured-system, and
   QPU-backed evidence.

Acceptance gates for the methodology paper:

| Gate | Requirement |
|---|---|
| Source coverage | Every selected method claim maps to a GOTM-SCPN Paper 0: The Foundational Framework source record or a stated non-Paper-0 assumption. |
| Artefact traceability | Every reported method output names the repository artefact path that produced it. |
| Fixture boundary | Fixture-backed preservation is never described as external validation. |
| Promotion boundary | A stronger evidence class requires a declared promotion gate and passing artefact. |
| Negative path | Every candidate lane includes at least one falsifier or blocker condition. |
| Reproducibility | Every table or figure in the paper has a regeneration command or stored provenance. |

## Experimental pathway tiers

| Tier | Purpose | Promotion condition |
|---|---|---|
| Tier 0: source accounting | Preserve Paper 0 records, spans, claim candidates, and canonical review ledgers. | Complete source register with no remaining ledger gaps. |
| Tier 1: executable specs | Convert source statements into validation specs, fixtures, loaders, and tests. | Spec bundle loads deterministically and has passing fixture tests. |
| Tier 2: no-QPU validation | Run classical, simulator, null-model, and sensitivity checks before hardware spend. | Pre-registered dataset or protocol passes uncertainty and falsifier gates. |
| Tier 3: measured-system candidates | Compare source-derived structures against named physical, biological, network, or benchmark datasets. | Units, uncertainty, full pairwise coverage where applicable, null models, and spectral or response checks pass. |
| Tier 4: hardware/QPU execution | Submit only bounded circuits that answer a pre-registered experimental question. | Tier 2 or Tier 3 gates pass, hardware manifest is frozen, and claim boundary is explicit before submission. |

## Candidate experimental lanes

| Lane | Paper 0 benefit | Near-term repository output | Claim boundary |
|---|---|---|---|
| K_nm causal-efficacy and coupling-affinity | Turns GOTM-SCPN Paper 0: The Foundational Framework coupling semantics into measured-system candidates. | Extend power-grid, EEG, and other measured topology audits with unit-aware promotion gates. | No physical validation unless uncertainty, units, null models, and response diagnostics pass. |
| TCBO p_H1 topology threshold | Converts a source threshold into a replayable preregistered topology test. | Named dataset replay protocol with confidence intervals and blocked promotion until preregistration exists. | No threshold claim from synthetic-only or unnamed data. |
| S2/S5 advantage-readiness matrix | Uses GOTM-SCPN Paper 0: The Foundational Framework as a pressure test for scaling methodology before hardware spend. | Complete benchmark matrix, negative rows, and hardware-row eligibility gates. | No quantum-advantage or IBM-spend justification from partial grids. |
| LHC, axion, and plasma search strategy | Converts high-energy and plasma statements into offline search specifications. | Public-dataset search protocol, feature schema, and null baseline registry. | No discovery claim without external dataset pass and independent review. |
| Biological QEC, CISS, and bioelectric lanes | Converts biological mechanisms into quantitative validation-spec candidates. | Literature-to-model spec bundles with required observable, units, and falsifier fields. | No medical, biological, or biophysical validation claim from source ingestion alone. |
| Organismal, ecological, symbolic, and noosphere layers | Converts upper-layer claims into operational-definition and falsifier work. | Layer-specific measurable-variable registry and blocked promotion gates. | No ontology-level validation without measurable variables and external data. |

## Immediate implementation queue

The next production slices should make the programme executable rather than
only narrative:

1. Maintain the generated Paper 0 lane registry with
   `scpn-bench paper0-lane-registry-gate`. The registry is the public
   source-bounded index of lane ids, evidence class, blockers, related spec
   artefacts, and hardware/QPU boundaries.
2. Maintain the tracked methodology-paper outline in
   `docs/paper0/paper0_methodology_paper_outline.md`. The outline references only
   reproducible artefacts and explicitly separates source accounting from
   validation.
3. Maintain the first preregistered downstream experiment design in
   `docs/paper0/paper0_first_preregistered_downstream_experiment.md`. The selected
   lane is K_nm causal-efficacy and coupling-affinity with no QPU or hardware
   execution.
4. Add an artefact-index command that lists each Paper 0 lane, evidence class,
   blocker, and next promotion gate.
5. Implement the no-QPU replay artefacts for the preregistered K_nm measured-
   system design before any hardware execution.
6. Keep the IBM/QPU path blocked until the relevant no-QPU matrix and claim
   boundary gates pass.

## Practical consequence

GOTM-SCPN Paper 0: The Foundational Framework ingestion is now useful as infrastructure: it supplies the source
ledger, extraction discipline, and experimental agenda. The immediate value is
not a stronger scientific claim. The value is a controlled method for deciding
which SCPN claims can become experiments, what evidence would promote them, and
which claims remain blocked.

## Generated lane registry

The generated registry lives at `docs/paper0/paper0_lane_registry.md` with its
machine-readable companion at `data/paper0_lane_registry.json`. Regenerate and
compare both artefacts with:

```bash
scpn-bench paper0-lane-registry-gate
```

Passing the gate means the source-bounded lane index is reproducible. It does
not promote any Paper 0 lane to external validation, measured-system evidence,
or hardware/QPU readiness.

## Methodology-paper outline

The tracked methodology-paper outline lives at
`docs/paper0/paper0_methodology_paper_outline.md`. It defines the paper structure,
evidence classes, required artefacts, figure/table regeneration commands, and
acceptance gates for source-bounded SCPN experimental translation.

The outline is a drafting contract. It does not promote GOTM-SCPN Paper 0: The
Foundational Framework claims beyond their existing evidence class.

## First preregistered downstream experiment

The first preregistered downstream experiment design lives at
`docs/paper0/paper0_first_preregistered_downstream_experiment.md`. It selects the K_nm
causal-efficacy and coupling-affinity lane as a no-QPU measured-system replay
with EEG alpha PLV as the primary candidate and IEEE 5-bus power grid as a
negative control.

The design is a preregistration contract. It does not report results and does
not open any hardware or QPU execution path.

## First preregistered replay gate

The first downstream measured-system lane now has a replayable no-QPU gate:

- Preregistration: `docs/paper0/paper0_first_preregistered_downstream_experiment.md`
- Replay generator: `scripts/run_paper0_knm_preregistered_replay.py`
- Replay comparator: `scripts/compare_paper0_knm_preregistered_replay.py`
- Replay gate: `scripts/run_paper0_knm_preregistered_replay_gate.py`
- Machine-readable result: `data/paper0_knm_preregistered_replay.json`
- Public report: `docs/paper0/paper0_knm_preregistered_replay.md`
- Contract: `docs/paper0/paper0_knm_preregistered_replay_contract.md`
- Promotion evidence checklist: `docs/paper0/paper0_knm_measured_coupling_evidence_checklist.md`
- Reproduction command: `scpn-bench paper0-knm-preregistered-replay-gate`

Current status is blocked and non-closing by design. The gate documents that the
EEG alpha PLV candidate remains a dimensionless synchronisation observable
without per-edge uncertainty, while the sparse IEEE 5-bus control prevents
topology-only agreement from being promoted to measured-system validation.


### Replay promotion-safety invariant

The `paper0-knm-preregistered-replay-gate` command now validates the replay
payload beyond regenerated file equality. A payload fails the gate if it stops
being blocked/non-closing, removes the hardware-blocking claim boundary,
authorises hardware submission, authorises claim promotion, drops the QPU
blocking gate, omits required evidence before reconsideration, omits
falsifiers, carries stale input SHA-256 digests, or weakens the committed
measured-candidate unit-class and non-promotion boundary. This invariant is
part of the Paper 0 pathway boundary: no later agent should interpret matrix
alignment, null-model output, digest stability, or byte-aligned JSON edits as
permission for QPU execution.

### Measured-coupling promotion checklist

The next promotion review is constrained by
`docs/paper0/paper0_knm_measured_coupling_evidence_checklist.md`. The replay cannot
move out of blocked/non-closing status until the primary measured-system lane
has calibrated coupling units, per-edge uncertainty, frozen normalisation,
matched null controls, digest-locked inputs, and a claim boundary that states
exactly what has and has not been validated.

The contract surface is machine-checkable through
`scripts/export_paper0_knm_replay_contract.py --check-replay
data/paper0_knm_preregistered_replay.json`. This check is a semantic boundary:
it catches weakened gates, changed locked inputs, missing evidence lists, and
hardware or claim-promotion authorisation even if a file-level drift comparison
would otherwise be ambiguous.
