# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Paper 0 Experimental Pathway

# Paper 0 Experimental Pathway

Status: source-bounded programme plan, dated 2026-05-18.

This page turns the completed Paper 0 source-validation register into a
longer-term experimental pathway and a methodology-paper route. It does not
promote Paper 0 propositions into external scientific validation. It defines how
source claims move from manuscript ingestion to executable validation artefacts,
promotion gates, falsifiers, and candidate experiments.

## Source hierarchy

Paper 0 is the upstream framework source for long-horizon SCPN translation work
in this repository. It supplies the canonical source register, domain map,
validation-spec extraction targets, and promotion-gate candidates.

Paper 27 is not the definitive source of truth for the programme after Paper 0
ingestion. In this repository, Paper 27 remains a historical implementation
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
| Source coverage | Every selected method claim maps to a Paper 0 source record or a stated non-Paper-0 assumption. |
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
| K_nm causal-efficacy and coupling-affinity | Turns Paper 0 coupling semantics into measured-system candidates. | Extend power-grid, EEG, and other measured topology audits with unit-aware promotion gates. | No physical validation unless uncertainty, units, null models, and response diagnostics pass. |
| TCBO p_H1 topology threshold | Converts a source threshold into a replayable preregistered topology test. | Named dataset replay protocol with confidence intervals and blocked promotion until preregistration exists. | No threshold claim from synthetic-only or unnamed data. |
| S2/S5 advantage-readiness matrix | Uses Paper 0 as a pressure test for scaling methodology before hardware spend. | Complete benchmark matrix, negative rows, and hardware-row eligibility gates. | No quantum-advantage or IBM-spend justification from partial grids. |
| LHC, axion, and plasma search strategy | Converts high-energy and plasma statements into offline search specifications. | Public-dataset search protocol, feature schema, and null baseline registry. | No discovery claim without external dataset pass and independent review. |
| Biological QEC, CISS, and bioelectric lanes | Converts biological mechanisms into quantitative validation-spec candidates. | Literature-to-model spec bundles with required observable, units, and falsifier fields. | No medical, biological, or biophysical validation claim from source ingestion alone. |
| Organismal, ecological, symbolic, and noosphere layers | Converts upper-layer claims into operational-definition and falsifier work. | Layer-specific measurable-variable registry and blocked promotion gates. | No ontology-level validation without measurable variables and external data. |

## Immediate implementation queue

The next production slices should make the programme executable rather than
only narrative:

1. Generate a Paper 0 lane registry from claim candidates, promotion gates, and
   validation-spec artefacts.
2. Publish a methodology-paper outline that references only reproducible
   artefacts and explicitly separates source accounting from validation.
3. Add an artefact-index command that lists each Paper 0 lane, evidence class,
   blocker, and next promotion gate.
4. Select one measured-system lane and produce a preregistered experimental
   design before any new hardware execution.
5. Keep the IBM/QPU path blocked until the relevant no-QPU matrix and claim
   boundary gates pass.

## Practical consequence

Paper 0 ingestion is now useful as infrastructure: it supplies the source
ledger, extraction discipline, and experimental agenda. The immediate value is
not a stronger scientific claim. The value is a controlled method for deciding
which SCPN claims can become experiments, what evidence would promote them, and
which claims remain blocked.
