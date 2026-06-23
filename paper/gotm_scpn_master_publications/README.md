# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# GOTM-SCPN Master Publications

This directory stores Book II source-processing packages. Numbering follows the
SCPN Master Publications table of contents.

## Standard Paper Package Layout

Every extracted foundational paper must use the same evidence-class folder
contract. This keeps source material, scientific synthesis, validation
protocols, experiments, measured results, and theory revisions separate.

| Folder | Contents | Claim status |
|---|---|---|
| `source/` | Canonical source files, source metadata, and full source dumps. | Source material only. |
| `extraction/` | Full-fidelity extracted equations, definitions, mechanisms, flags, and extraction indexes. | What the source says. |
| `source_validation_artifacts/` | Stable source ledgers, validation specs, fixture outputs, promotion gates, and reconciliation reports. | Source-accounting and executable fixture preservation. |
| `synthesis/` | Scientific synthesis, formalisation notes, and cross-source interpretation derived from extraction. | Derived analysis; not experimental evidence. |
| `validation_protocols/` | Preregistered or proposed protocols for validating specific claims. | Hypothesis-test design. |
| `experiments/` | Prepared experiment packages, manifests, run plans, and submission bundles. | Execution plan or submitted campaign. |
| `results/` | Measured-system, simulator, QPU, or provider results tied to protocols. | Evidence, bounded by each protocol. |
| `revisions/` | Proposed corrections back into the foundational paper/book text. | Theory-edit candidates. |
| `legacy_pre_tuned_extraction/` | Older extraction attempts that predate the tuned methodology. | Historical only; not processed status. |

No paper may be marked processed from `source/`, `extraction/`, or legacy files
alone. A processed paper requires stable source identifiers, validation specs,
fixtures/loaders, tests, promotion gates, public claim-boundary documentation,
and a clear separation between source statements, synthesis, protocols,
experiments, results, and revisions.

| Directory | Paper | Processing status |
|---|---|---|
| `gotm-scpn_paper-00_the_foundational_framework/` | Paper 0: The Foundational Framework | Processed through the tuned source-ledger/spec/fixture/gate methodology. |
| `gotm-scpn_paper-01_layer-1_quantum-biological/` | Paper 1: Layer 1 - Quantum Biological | Legacy or partial extraction material only; pending tuned rerun. |
| `gotm-scpn_paper-02_layer-2_neurochemical-neurological/` | Paper 2: Layer 2 - Neurochemical-Neurological | Raw or early-stage extraction material only; pending tuned rerun. |

The repeatable method is documented in
`../../docs/paper0/paper0_processing_methodology.md`. Do not mark a later paper as
processed until it has stable source identifiers, generated validation specs,
fixtures/loaders, tests, promotion gates, and a public claim-boundary register.
