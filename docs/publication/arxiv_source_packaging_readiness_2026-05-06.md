<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — arXiv Source Packaging Readiness -->

# arXiv Source Packaging Readiness

Date prepared: 2026-05-06

This note prepares the arXiv source-packaging boundary for the current paper
set. It does not claim that any arXiv submission has happened.

## Paper Sources

| Paper | Primary source | Bibliography/source companions | PDF |
|-------|----------------|--------------------------------|-----|
| DLA parity hardware preprint | `paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity.tex` | inline `thebibliography`; figures under `figures/phase1/` and `figures/phase2/` | `paper/submissions/submission_002_phase1_dla_parity/phase1_dla_parity.pdf` |
| Rust/VQE methods paper | `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.tex` | inline `thebibliography`; generated artefacts under `data/rust_vqe_methods/` | `paper/submissions/submission_003_rust_vqe_methods/rust_vqe_methods.pdf` |
| SCPN/FIM Hamiltonian paper | `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian.tex` | `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian_refs.bib`; generated artefacts under `data/scpn_fim_hamiltonian/` | `paper/submissions/submission_004_scpn_fim_hamiltonian/scpn_fim_hamiltonian.pdf` |

The current JOSS-style software note is tracked separately under
`paper/submissions_joss/submission_joss_001_software_framework_note/`. Future JOSS submissions should use
sibling directories under `paper/submissions_joss/`; none are primary arXiv source packages
unless a venue-specific decision promotes them to arXiv.

## Required Source Bundle Contents

For each arXiv upload bundle, include only public manuscript files:

- the `.tex` source;
- the required `.bib` file if the manuscript uses one;
- figures referenced by `\includegraphics`;
- any local style files if added later;
- a generated PDF for local comparison only, not as the sole source.

Do not include:

- `.coordination/` logs or handovers;
- credential or vault files;
- IBM tokens or provider configuration;
- raw private email drafts;
- backup-copy paths;
- generated LaTeX scratch files (`.aux`, `.log`, `.out`, `.bbl`, `.blg`) unless
  a specific arXiv build problem requires a minimal generated bibliography file;
- large raw-count JSON datasets, which should remain referenced by repository
  path and DOI rather than embedded in the arXiv source bundle.

## Build Gate

Run each manuscript from its own directory:

```bash
cd paper/submissions/submission_002_phase1_dla_parity
pdflatex -interaction=nonstopmode -halt-on-error phase1_dla_parity.tex

cd ../rust_vqe_methods
pdflatex -interaction=nonstopmode -halt-on-error rust_vqe_methods.tex

cd ../scpn_fim_hamiltonian
pdflatex -interaction=nonstopmode -halt-on-error scpn_fim_hamiltonian.tex
```

If `scpn_fim_hamiltonian.tex` bibliography output changes:

```bash
bibtex scpn_fim_hamiltonian
pdflatex -interaction=nonstopmode -halt-on-error scpn_fim_hamiltonian.tex
pdflatex -interaction=nonstopmode -halt-on-error scpn_fim_hamiltonian.tex
```

Do not upload if the build log contains unresolved citations, missing figures,
broken URLs, malformed job IDs, or internal/private paths.

## Claim Gate

Before upload, verify that the source text preserves these boundaries:

| Paper | Required boundary |
|-------|-------------------|
| DLA parity | Backend-sensitive parity-sector/excitation-number correlated leakage; not DLA-parity-only protection and not backend-stable quantum advantage. |
| Rust/VQE methods | Selected kernel and workflow reproducibility claims; not whole-workflow acceleration and not a general VQE theorem. |
| SCPN/FIM | Exact small-system structure plus negative IBM hardware falsification; not hardware coherence protection. |

## Metadata Draft

Recommended arXiv metadata:

| Paper | Primary category | Secondary categories |
|-------|------------------|----------------------|
| DLA parity hardware preprint | `quant-ph` | `physics.comp-ph` |
| Rust/VQE methods paper | `quant-ph` | `cs.SE`, `physics.comp-ph` |
| SCPN/FIM Hamiltonian paper | `quant-ph` | `physics.comp-ph` |

Common repository line:

```text
Code and data: https://github.com/anulum/scpn-quantum-control
```

## Final Manual Gate

Before external upload:

- confirm final PDFs were rebuilt from committed sources;
- confirm Zenodo DOI links resolve;
- confirm GitHub URL is `https://github.com/anulum/scpn-quantum-control`;
- confirm no unsupported claims were reintroduced by later edits;
- confirm no private/internal files are included in the source tarball;
- commit any final source/PDF changes;
- upload only after explicit author approval.
