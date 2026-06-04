<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — pyOpenSci Submission Readiness -->

# pyOpenSci Submission Readiness

Date prepared: 2026-05-06

This note prepares the `scpn-quantum-control` pyOpenSci review package. It does
not claim that a pyOpenSci issue has been opened or accepted.

## Review Scope

Recommended submission framing:

> `scpn-quantum-control` is a specialised Python/Rust research-software package
> for reproducible Kuramoto-XY quantum-control workflows, including
> Hamiltonian/ansatz construction, simulator and IBM hardware artefact
> packaging, benchmark regeneration, and claim-boundary documentation.

Supported claims:

- Domain-specific workflow package for Kuramoto-XY and SCPN phase-network
  experiments.
- Python/Qiskit orchestration with selected Rust/PyO3 hot-path kernels.
- Reproducible benchmark harnesses that regenerate JSON/CSV artefacts.
- Hardware artefact packaging with raw counts, job IDs, metadata, and hashes.
- Public documentation and package metadata suitable for software review.

Unsupported claims:

- General-purpose quantum simulator.
- Quantum-advantage engine.
- Backend-stable DLA parity protection.
- Hardware coherence protection from the FIM Hamiltonian.
- Whole-workflow acceleration from Rust.
- Requirement that optional IBM, GPU, Julia, or domain-specific extras are
  installed for the base package.

## Metadata Gate

| Field | Current value |
|-------|---------------|
| Package name | `scpn-quantum-control` |
| Version | `0.9.10` source metadata; package release artefacts may lag until the next tagged release |
| Repository | `https://github.com/anulum/scpn-quantum-control` |
| Documentation | `https://anulum.github.io/scpn-quantum-control` |
| Issue tracker | `https://github.com/anulum/scpn-quantum-control/issues` |
| License | `AGPL-3.0-or-later` |
| Author ORCID | `0009-0009-3560-0851` |
| Contact | `protoscience@anulum.li` |
| PyPI status | Published |
| Zenodo concept DOI | `10.5281/zenodo.18821929` |

Before opening the pyOpenSci issue, verify that the package version, PyPI
release, Zenodo metadata, and documentation site all describe the same release
line.

## Reviewer-Relevant Evidence

| Evidence | Path |
|----------|------|
| Package metadata | `pyproject.toml` |
| JOSS-style software paper | `paper/submissions_joss/submission_joss_001_software_framework_note/paper.md` |
| Software submission checklist | `docs/publication/joss_software_submission_checklist_2026-05-06.md` |
| Combined paper checklist | `docs/publication/combined_submission_checklist_2026-05-06.md` |
| Benchmark dashboard | `docs/methods_benchmark_dashboard.md` |
| Coverage and behavioural audits | Internal release-audit notes retained outside the public documentation site. |
| Actions history dashboard | `docs/actions_history_dashboard.md` |
| Artefact-first architecture | `docs/architecture.md` |
| Hardware status ledger | `docs/hardware_status_ledger.md` |

## No-QPU Pre-Submission Gates

Recommended local gates before opening the review issue:

```bash
./.venv-linux/bin/python -m mkdocs build --strict
./.venv-linux/bin/python tools/audit_test_behaviour.py --fail-on-smoke-only
./.venv-linux/bin/python tools/audit_e2e_contract_boundaries.py --fail-on-missing
scpn-bench reproduce-methods
scpn-bench fim-all
```

If a fresh coverage report exists, also run:

```bash
./.venv-linux/bin/python tools/audit_coverage_gaps.py --fail-on-gap
```

The last command should not be used as a hard submission blocker until the
broader coverage-to-100-percent roadmap item is closed or remaining gaps are
explicitly justified.

## Suggested pyOpenSci Issue Summary

```markdown
## Package name
scpn-quantum-control

## Repository
https://github.com/anulum/scpn-quantum-control

## Documentation
https://anulum.github.io/scpn-quantum-control

## Summary
scpn-quantum-control is a specialised Python/Rust package for reproducible
Kuramoto-XY quantum-control workflows. It maps oscillator coupling matrices and
frequencies to Qiskit Hamiltonians, topology-informed ansatze, simulator
workflows, IBM hardware artefact packages, and benchmark-regeneration scripts.
The package is intentionally framed as reproducible research infrastructure,
not as a general quantum simulator or quantum-advantage engine.

## Scope
The package supports Hamiltonian/ansatz construction, benchmark artefact
regeneration, hardware raw-count packaging, readout/mitigation analysis, and
bounded publication workflows for small-N NISQ studies.
```

## Submission Boundary

Opening the pyOpenSci issue is an external account/manual action. Do not mark
the roadmap submission item complete until the issue URL is recorded.

No QPU time is required for pyOpenSci review. Hardware examples must cite
already committed raw-count artefacts only.
