<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Registry Listing Plan -->

# Registry Listing Plan

Date: 2026-05-06

This document is the submission pack for external registry listings. It keeps
claims bounded until the paper set has public preprint identifiers.

## Canonical Metadata

| Field | Value |
|---|---|
| Project name | `scpn-quantum-control` |
| Short title | SCPN Quantum Control |
| Repository | <https://github.com/anulum/scpn-quantum-control> |
| Documentation | <https://anulum.github.io/scpn-quantum-control> |
| PyPI | <https://pypi.org/project/scpn-quantum-control/> |
| Zenodo concept DOI | <https://doi.org/10.5281/zenodo.18821929> |
| Current Zenodo version DOI | <https://doi.org/10.5281/zenodo.18821930> |
| Licence | AGPL-3.0-or-later; commercial licence available |
| Contact | `protoscience@anulum.li` |
| ORCID | `0009-0009-3560-0851` |
| Primary language | Python with optional Rust acceleration |
| Domain | quantum computing, quantum simulation, Kuramoto--XY oscillators |

## Conservative One-Line Description

`scpn-quantum-control` is a reproducible Kuramoto--XY quantum-simulation
workflow for heterogeneous oscillator networks, with Qiskit execution paths,
Rust-accelerated Hamiltonian construction, hardware raw-count packaging, and
artefact-first benchmark regeneration.

## Conservative Long Description

`scpn-quantum-control` provides a Python/Qiskit workflow for mapping
heterogeneous Kuramoto oscillator networks to XY Hamiltonians, generating
digital quantum circuits, packaging IBM hardware raw counts, and regenerating
analysis and benchmark artefacts from committed scripts. The package includes
Rust-accelerated hot paths, topology-informed ansatz tools, readout and
symmetry-aware analysis utilities, and no-QPU reproducibility commands such as
`scpn-bench reproduce-methods`, `scpn-bench fim-all`, and `scpn-bench all`.

The public claim boundary is deliberately narrow: the package is research
software for reproducible NISQ workflows and small-system hardware
phenomenology. It does not claim broad quantum advantage, clinical causation,
universal error protection, or DLA-parity-only robustness.

## Keywords

- quantum computing
- quantum simulation
- Qiskit
- Kuramoto model
- XY Hamiltonian
- NISQ
- reproducibility
- Rust acceleration
- error mitigation
- IBM Quantum
- oscillator networks

## Target Registry Status

| Target | Status | Action |
|---|---|---|
| QOSF awesome-quantum-software | Already listed | No duplicate submission. The project appears under Python quantum simulators. |
| Quantiki | Ready for manual account-gated submission | Submit as a project/news item after account approval, using the conservative copy below. |
| best-of-python | Defer | Current list categories are general Python infrastructure; this package is specialised scientific software. Avoid a low-fit submission unless maintainers request it. |
| Papers With Code | Blocked until preprint IDs exist | Add repository only after arXiv/JOSS entries exist so the code can attach to a paper. |
| SciCrunch RRID | Ready for manual/resource form | Submit as software/resource; record the RRID in paper methods only after it resolves. |
| Open Hub | Ready for manual account-gated submission | Add repository URL and verify crawler import; no paper identifier required. |
| Research Software Directory | Ready but access-gated | Requires RSD account/organisation access; use GitHub and Zenodo metadata import where available. |

## Quantiki Draft

Title:

```text
scpn-quantum-control: reproducible Kuramoto--XY quantum-simulation workflows
```

Body:

```text
scpn-quantum-control is an open-source research-software package for
reproducible Kuramoto--XY quantum-simulation workflows. It maps heterogeneous
oscillator-network coupling matrices to XY Hamiltonians, generates Qiskit
circuits, packages IBM Quantum raw-count artefacts with job IDs and integrity
hashes, and regenerates methods/FIM/DLA benchmark tables from committed
scripts.

The project is intentionally conservative in its claims. It reports
small-system NISQ hardware phenomenology and reproducible methods
infrastructure, not broad quantum advantage or universal hardware protection.

Repository: https://github.com/anulum/scpn-quantum-control
Documentation: https://anulum.github.io/scpn-quantum-control
Zenodo: https://doi.org/10.5281/zenodo.18821929
```

## SciCrunch RRID Draft

Resource name:

```text
scpn-quantum-control
```

Resource type:

```text
Software tool
```

Description:

```text
Research software for reproducible Kuramoto--XY quantum-simulation workflows,
including Qiskit circuit generation, IBM hardware raw-count packaging,
Rust-accelerated Hamiltonian construction, topology-informed ansatz tools, and
artefact-first benchmark regeneration.
```

Homepage/source:

```text
https://github.com/anulum/scpn-quantum-control
```

Persistent identifier:

```text
https://doi.org/10.5281/zenodo.18821929
```

## Open Hub Draft

Project name:

```text
scpn-quantum-control
```

Source repository:

```text
https://github.com/anulum/scpn-quantum-control.git
```

Description:

```text
Reproducible Kuramoto--XY quantum-simulation workflow with Qiskit hardware
execution paths, Rust-accelerated Hamiltonian construction, raw-count
provenance, and generated benchmark artefacts.
```

## Research Software Directory Draft

Short description:

```text
Reproducible Kuramoto--XY quantum-simulation workflow for heterogeneous
oscillator networks, Qiskit execution, raw-count provenance, and artefact-first
benchmark regeneration.
```

Markdown description:

```text
scpn-quantum-control is a research-software package for mapping heterogeneous
Kuramoto oscillator networks to XY Hamiltonians and running reproducible NISQ
workflow studies. It provides Qiskit circuit generation, IBM hardware
raw-count packaging, Rust-accelerated Hamiltonian construction,
topology-informed ansatz utilities, and one-command regeneration of committed
methods and FIM benchmark artefacts.

The package is designed for auditability: publication tables are generated from
committed JSON/CSV artefacts and scripts; hardware claims are tied to raw
counts, job IDs, and SHA-256 manifests; and unsupported claims are explicitly
blocked in the public documentation.
```

URLs:

```text
Source: https://github.com/anulum/scpn-quantum-control
Documentation: https://anulum.github.io/scpn-quantum-control
DOI: https://doi.org/10.5281/zenodo.18821929
PyPI: https://pypi.org/project/scpn-quantum-control/
```

## Papers With Code Draft

Use only after an arXiv or journal page exists.

Repository:

```text
https://github.com/anulum/scpn-quantum-control
```

Paper relation:

```text
Official implementation and artefact package.
```

Task tags to consider:

```text
quantum simulation
quantum computing
error mitigation
```

## Submission Rules

- Do not submit the package to a registry category where the scientific fit is
  weak.
- Do not announce broad quantum advantage.
- Do not announce clinical, biological, or consciousness causation.
- Do not present IBM hardware artefacts without raw-count and job-ID context.
- Do not use Papers With Code before paper identifiers exist.
- Record every successful external submission in `ROADMAP.md` and a dated
  `docs/*_submission_*.md` file.
