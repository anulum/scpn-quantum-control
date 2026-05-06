<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — JOSS Software Submission Checklist -->

# JOSS-Style Software Paper Submission Checklist

Date: 2026-05-06

This checklist freezes the submission boundary for the short JOSS-style
software note. It indexes committed paper and package artefacts only. It does
not claim that JOSS, pyOpenSci, or arXiv submission has already happened.

## Submission Scope

Supported:

- Research software package for Kuramoto-XY quantum-control workflows.
- Python/Qiskit public interface with selected Rust/PyO3 hot-path kernels.
- Hardware artefact packaging: job identifiers, raw counts, metadata, and
  integrity hashes.
- Benchmark artefact discipline through committed JSON/CSV summaries.
- Public `scpn-bench` reproducibility CLI.
- Documentation site and repository links from package metadata.
- Companion DLA parity and Rust/VQE methods artefacts as application evidence.

Not supported:

- A broad quantum-advantage claim.
- A claim that Rust accelerates the whole VQE or Qiskit transpilation workflow.
- A claim that the software replaces Qiskit, PennyLane, QuTiP, or general
  quantum-control frameworks.
- A claim that all optional extras are required for normal installation.
- A claim that paper tables can be edited manually without regenerated
  artefacts.

## JOSS / pyOpenSci Boundary

The paper should be framed as software infrastructure, not as the primary
experimental result. The primary contribution is the reusable workflow:

- model-to-Hamiltonian conversion;
- topology-informed ansatz construction;
- simulator and hardware-run packaging;
- benchmark artefact regeneration;
- provenance and claim-boundary documentation.

The DLA parity, SCPN/FIM, and Rust/VQE methods papers can be cited as research
applications, but the JOSS-style note should remain focused on package purpose,
installation, documentation, examples, tests, and archival metadata.

## Required Claim Wording

Use the conservative claim:

> `scpn-quantum-control` is a specialised, reproducible workflow package for
> Kuramoto-XY NISQ studies, combining Qiskit orchestration, selected Rust/PyO3
> kernels, benchmark harnesses, and hardware artefact packaging.

Avoid the stronger claim:

> `scpn-quantum-control` is a general quantum simulator or quantum-advantage
> engine.

## Committed Artefact Index

| Item | Path |
|------|------|
| JOSS paper source | `paper/joss/paper.md` |
| JOSS preview source | `paper/joss/paper_preview.tex` |
| JOSS preview PDF | `paper/joss/paper_preview.pdf` |
| Project metadata | `pyproject.toml` |
| README | `README.md` |
| Documentation site config | `mkdocs.yml` |
| Benchmark dashboard | `docs/methods_benchmark_dashboard.md` |
| DLA submission checklist | `docs/dla_parity_submission_checklist_2026-05-06.md` |
| Rust/VQE submission checklist | `docs/rust_vqe_methods_submission_checklist_2026-05-06.md` |
| Software artefacts | `data/rust_vqe_methods/` |
| Hardware validation package | `docs/publication_phase2_package_2026-05-05.md` |

## Metadata Gate

Before submission:

- Verify `pyproject.toml` version matches the intended release tag.
- Verify repository URL is `https://github.com/anulum/scpn-quantum-control`.
- Verify documentation URL is `https://anulum.github.io/scpn-quantum-control`.
- Verify issue tracker URL points to the public GitHub repository.
- Verify author ORCID is `0009-0009-3560-0851`.
- Verify the software licence shown in the paper matches `AGPL-3.0-or-later`.
- Verify any Zenodo DOI listed in the paper resolves to the intended software
  or data record.

## Reproducibility Gate

Run before changing any numerical sentence in the JOSS-style note:

```bash
scpn-bench reproduce-methods
```

Optional supporting checks:

```bash
scpn-bench all --dry-run
scpn-bench reproduce-methods --include-scaling
```

The JOSS-style note may cite benchmark artefacts, but it should not become a
performance paper. Detailed benchmark tables belong in the Rust/VQE methods
paper and dashboard.

## Final Manual Pre-Upload Gate

Before JOSS, pyOpenSci, or arXiv upload:

- Rebuild `paper/joss/paper_preview.pdf` from `paper/joss/paper_preview.tex`
  if the preview source changed.
- Verify `paper/joss/paper.md` has valid YAML metadata and bibliography links.
- Verify every benchmark number in `paper/joss/paper.md` appears in committed
  JSON/CSV artefacts or is removed.
- Verify the quickstart imports still match public package APIs.
- Verify installation instructions do not require optional IBM, GPU, Julia, or
  domain extras for the default path.
- Verify optional extras are described as optional.
- Verify the AI disclosure, if required by the venue, is minimal and does not
  replace the author responsibility statement.
- Verify no internal coordination paths, private data locations, credentials,
  or provider tokens appear in the paper.

## QPU Boundary

No QPU time is required for this software submission package. Hardware examples
must cite committed raw-count artefacts only. New provider runs require a
separate manifest, budget estimate, and explicit approval.
