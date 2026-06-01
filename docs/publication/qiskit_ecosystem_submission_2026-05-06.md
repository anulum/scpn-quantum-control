<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Qiskit Ecosystem Submission -->

# Qiskit Ecosystem Submission

Date submitted: 2026-05-06

This note records the Qiskit Ecosystem Catalog submission for
`scpn-quantum-control`.

## Submission

Issue:

```text
https://github.com/Qiskit/ecosystem/issues/1123
```

Status at submission time:

```text
OPEN
```

The submission is pending Qiskit Ecosystem review. This document does not claim
catalog acceptance.

## Live Submission Criteria Checked

The Qiskit Ecosystem repository states that candidate projects must:

- build on, interface with, or extend Qiskit;
- be compatible with Qiskit SDK 1.0 or newer;
- use an OSI-approved open-source licence;
- follow the Qiskit Code of Conduct;
- show maintainer activity within the last six months;
- be compatible with V2 primitives for new projects.

`scpn-quantum-control` was submitted because the current package:

- depends on `qiskit>=2.2,<3.0`;
- uses Qiskit for circuit construction, transpilation, IBM Runtime execution
  packaging, and post-processing of raw-count artefacts;
- has source metadata at version `0.9.8`; PyPI release artefacts may lag the
  source tree until the next tagged package release;
- uses the OSI-approved `AGPL-3.0-or-later` licence;
- has current repository activity and public documentation.

## Submitted Metadata

Project name:

```text
scpn-quantum-control
```

Description:

```text
Qiskit workflow for Kuramoto-XY NISQ experiments with Rust kernels, hardware packaging, and artefact-first analysis.
```

Category:

```text
Tooling
```

Labels requested:

- `physics`
- `error mitigation`
- `research`
- `circuit building`
- `quantum information`

Interfaces:

- Python
- Rust
- Command-line interface

Maturity:

```text
production-ready
```

Qiskit Pattern steps:

- Map
- Optimize
- Execute
- Post-process

Repository:

```text
https://github.com/anulum/scpn-quantum-control
```

Documentation:

```text
https://anulum.github.io/scpn-quantum-control
```

Package:

```text
https://pypi.org/project/scpn-quantum-control/
```

Reference DOI:

```text
https://doi.org/10.5281/zenodo.18821930
```

## Claim Boundary

This submission records a catalog-review request only. It does not imply:

- Qiskit Ecosystem acceptance;
- IBM endorsement;
- a broad quantum-advantage claim;
- a new Zenodo archive version;
- completion of any pending QPU validation campaign.
