<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Community Announcement Pack -->

# Community Announcement Pack

Date: 2026-05-06

This document prepares bounded announcement copy. Do not publish these messages
until the relevant public preprint or JOSS links exist.

## Publication Gate

Publish only after at least one of these links is live:

- arXiv preprint for the DLA-parity paper;
- arXiv preprint for the methods/FIM paper set;
- JOSS or pyOpenSci review issue for the software package.

## Core Message

`scpn-quantum-control` is an open research-software package for reproducible
Kuramoto--XY quantum-simulation workflows. It emphasises raw-count provenance,
artefact-generated tables, bounded NISQ claims, and no-QPU reproduction paths.

## Short Announcement

```text
We released scpn-quantum-control, an open Kuramoto--XY quantum-simulation
workflow for Qiskit/IBM hardware studies. The repository includes raw-count
provenance, generated benchmark artefacts, Rust-accelerated Hamiltonian
construction, and conservative claim boundaries for small-system NISQ results.

Code: https://github.com/anulum/scpn-quantum-control
Docs: https://anulum.github.io/scpn-quantum-control
DOI: https://doi.org/10.5281/zenodo.18821929
```

## Hacker News / General Software Draft

```text
Show HN: scpn-quantum-control -- reproducible Kuramoto--XY quantum-simulation workflows

I built scpn-quantum-control, a Python/Qiskit research package for mapping
heterogeneous oscillator networks to XY Hamiltonians and packaging small-system
NISQ hardware experiments with raw-count provenance.

The main engineering rule is artefact-first reproducibility: tables in the
papers are regenerated from committed JSON/CSV artefacts and scripts, hardware
claims are tied to IBM job IDs and SHA-256 manifests, and the repo includes
no-QPU reproduction commands such as `scpn-bench reproduce-methods` and
`scpn-bench fim-all`.

It does not claim broad quantum advantage. The hardware papers are deliberately
bounded: one reports a parity-sector/excitation-number correlated leakage
phenomenon; another reports a negative hardware falsification for a digital
FIM feedback term.

Code: https://github.com/anulum/scpn-quantum-control
Docs: https://anulum.github.io/scpn-quantum-control
```

## Reddit / Quantum Computing Draft

```text
I am sharing scpn-quantum-control, an open-source Kuramoto--XY
quantum-simulation workflow built around Qiskit, IBM raw-count provenance, and
artefact-first reproducibility.

The focus is narrow: small-system NISQ phenomenology and methods
infrastructure, not quantum-advantage claims. The repo contains promoted
ibm_kingston DLA-parity datasets, popcount controls, a negative FIM hardware
falsification result, generated benchmark artefacts, and one-command no-QPU
reproduction paths.

Code: https://github.com/anulum/scpn-quantum-control
Docs: https://anulum.github.io/scpn-quantum-control
Zenodo: https://doi.org/10.5281/zenodo.18821929
```

## Qiskit Slack Draft

```text
I would like to share scpn-quantum-control, a Qiskit-based workflow for
Kuramoto--XY quantum-simulation experiments with raw-count provenance and
artefact-generated benchmark tables.

The package includes Qiskit circuit generation, IBM hardware result packaging,
readout/symmetry analysis tools, Rust-accelerated Hamiltonian construction, and
`scpn-bench` commands for no-QPU reproduction of methods/FIM artefacts.

Repository: https://github.com/anulum/scpn-quantum-control
Docs: https://anulum.github.io/scpn-quantum-control
```

## Unitary Discord Draft

```text
Sharing scpn-quantum-control: a reproducible Kuramoto--XY quantum-simulation
workflow for small-system NISQ studies. The repo emphasises raw IBM count
artefacts, job IDs, SHA-256 manifests, generated tables, and bounded claims.

It may be useful for people interested in Qiskit workflows, oscillator-network
Hamiltonians, DLA/symmetry observables, readout checks, and reproducible
hardware-paper packaging.

https://github.com/anulum/scpn-quantum-control
```

## LinkedIn Draft

```text
I am releasing scpn-quantum-control, an open research-software package for
reproducible Kuramoto--XY quantum-simulation workflows.

The project combines a Python/Qiskit front end, optional Rust acceleration,
IBM hardware raw-count provenance, generated benchmark artefacts, and
conservative claim boundaries for small-system NISQ experiments. The associated
paper set focuses on reproducibility, hardware phenomenology, and honest
negative-result reporting rather than broad quantum-advantage claims.

Repository: https://github.com/anulum/scpn-quantum-control
Documentation: https://anulum.github.io/scpn-quantum-control
Zenodo DOI: https://doi.org/10.5281/zenodo.18821929
```

## X Draft

```text
Released scpn-quantum-control: reproducible Kuramoto--XY quantum-simulation
workflows for Qiskit/IBM NISQ studies.

Raw-count provenance, generated benchmark artefacts, Rust hot paths, and
bounded claims. No broad quantum-advantage claim.

https://github.com/anulum/scpn-quantum-control
```

## Do Not Publish

- Do not post before public preprint or JOSS/pyOpenSci links are available.
- Do not imply broad quantum advantage.
- Do not imply biological, clinical, or consciousness causation.
- Do not imply that negative hardware results demonstrate protection.
- Do not post QPU claims without raw-count/job-ID context.
