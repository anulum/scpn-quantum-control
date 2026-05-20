<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — SciPy 2026 CFP Readiness -->

# SciPy 2026 CFP Readiness

Date prepared: 2026-05-06

This note prepares a possible SciPy 2026 talk or poster proposal. It does not
claim that a CFP is open, that a proposal has been submitted, or that the
conference has accepted the work. Before submission, verify the current SciPy
CFP page, tracks, deadlines, length limits, and eligibility requirements.

## Recommended Angle

Primary angle:

> Artefact-first quantum software: reproducible Kuramoto-XY NISQ workflows with
> Python/Qiskit orchestration, Rust hot-path kernels, benchmark regeneration,
> and raw hardware-count provenance.

Why this fits SciPy:

- Python-first scientific workflow with a narrow Rust acceleration layer.
- Reproducibility discipline: JSON/CSV artefacts generated from scripts, not
  hand-edited tables.
- Open scientific data lineage: raw counts, job IDs, SHA256 hashes, and
  claim-boundary documents.
- Practical benchmark story across local CPU, ML350, Vertex CPU, and Vertex T4.
- Honest negative and bounded hardware results, useful as a reproducible
  scientific-computing case study.

## Suggested Formats

| Format | Fit | Boundary |
|--------|-----|----------|
| Talk | Strong if the track welcomes scientific Python infrastructure or reproducibility workflows. | Emphasise software engineering, artefacts, and reproducibility rather than speculative physics. |
| Poster | Strong fallback if a full talk is not suitable. | Use figures/tables from generated artefacts only. |
| Birds-of-a-feather | Possible if focused on reproducible NISQ/scientific benchmark packaging. | Avoid making it a project advertisement. |

## Draft Title Options

1. `Artefact-first quantum workflows in Python: reproducible Kuramoto-XY NISQ studies with Rust hot paths`
2. `From oscillator networks to auditable hardware counts: a Python/Rust workflow for reproducible NISQ experiments`
3. `Making small-N quantum experiments reproducible: generated benchmarks, raw-count provenance, and bounded claims`

## Draft Abstract

`scpn-quantum-control` is a Python/Rust research-software package for
reproducible Kuramoto-XY quantum-control workflows. It maps oscillator coupling
matrices and natural frequencies to Qiskit Hamiltonians, topology-informed
ansatze, simulator workflows, IBM hardware artefact packages, and
benchmark-regeneration scripts. This proposal presents the software-engineering
layer behind the project: an artefact-first rule where numerical tables are
regenerated from committed scripts into JSON/CSV summaries; selected Rust/PyO3
hot-path kernels for deterministic coupling-matrix and workflow support;
cross-machine benchmark provenance; and raw hardware-count packaging with job
identifiers, metadata, SHA256 hashes, and claim boundaries. The talk uses
small-N NISQ studies as a concrete scientific-computing case study while
explicitly avoiding quantum-advantage claims. The goal is to show how Python
scientific workflows can make hardware-adjacent research auditable, rerunnable,
and honest about negative or backend-sensitive results.

## Evidence to Cite

| Evidence | Path |
|----------|------|
| Methods benchmark dashboard | `docs/methods_benchmark_dashboard.md` |
| Rust/VQE methods checklist | `docs/publication/rust_vqe_methods_submission_checklist_2026-05-06.md` |
| JOSS/software checklist | `docs/publication/joss_software_submission_checklist_2026-05-06.md` |
| DLA parity submission checklist | `docs/publication/dla_parity_submission_checklist_2026-05-06.md` |
| FIM submission checklist | `docs/campaigns/scpn_fim_submission_checklist_2026-05-06.md` |
| Coverage and behavioural audits | Internal release-audit notes retained outside the public documentation site. |
| Source packaging readiness | `docs/publication/arxiv_source_packaging_readiness_2026-05-06.md` |

## Claims to Avoid

- Do not claim quantum advantage.
- Do not claim backend-stable DLA parity protection.
- Do not claim FIM hardware coherence protection.
- Do not claim Rust accelerates the whole Qiskit/VQE/transpilation workflow.
- Do not present opportunistic shared-machine timings as isolated
  microbenchmarks.
- Do not imply that new QPU time is needed for the SciPy proposal.

## Pre-Submission Gate

Before submitting a proposal:

- verify the current CFP page and track descriptions;
- check word/character limits;
- decide talk vs poster vs BoF;
- confirm whether papers/preprints are public yet;
- use only generated benchmark numbers from committed artefacts;
- ensure no private paths, credentials, or internal coordination files appear;
- update this note with the final submitted title, abstract, and submission URL
  if submission is completed.

## QPU Boundary

No QPU time is required for a SciPy proposal. Any hardware claims should cite
already committed raw-count artefacts and manuscript claim boundaries only.
