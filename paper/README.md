# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# paper/ - publication sources

Each manuscript or venue package has its own numbered subdirectory. Keep
publication submissions under `paper/submissions/`, JOSS submissions under
`paper/submissions_joss/`, and Book II source-processing packages under
`paper/gotm_scpn_master_publications/`.

## Directory map

| Directory | Contents | Status |
|---|---|---|
| `publication_planning/` | Cross-paper planning notes, preprint outlines, programme overview, and publication-framing guidance. | Planning material, not a submission source package. |
| `submissions/submission_001_ibm_fez_synchronisation/` | February 2026 `ibm_fez` synchronisation overview in `main.tex`, with local `figures/publication/`. | Contextual overview; compiles locally. |
| `submissions/submission_002_phase1_dla_parity/` | April 2026 `ibm_kingston` DLA-parity manuscript, short-paper draft, and local `figures/phase1/` + `figures/phase2/`. | Submission-ready source package, with claim boundaries. |
| `submissions/submission_003_rust_vqe_methods/` | Rust/VQE methodology manuscript and local PDF. | Methods preprint candidate. |
| `submissions/submission_004_scpn_fim_hamiltonian/` | SCPN/FIM Hamiltonian manuscript and bibliography material. | Negative hardware-validation paper candidate. |
| `submissions/submission_005_phase3_reduced_pauli_entanglement/` | Reduced-Pauli entanglement REVTeX manuscript, bibliography, PDF, and local `figures/phase3/`. | Completed multi-backend IBM run; venue-style source package. |
| `submissions/submission_006_s1_feedback_control/` | Monitored feedback versus open-loop dynamic-circuit manuscript, bibliography, PDF, and local `figures/s1_feedback_control/`. | Multi-backend dynamic-circuit control-boundary paper. |
| `submissions_joss/submission_joss_001_software_framework_note/` | Current JOSS software-framework note and preview files. | One JOSS submission package; future JOSS notes get sibling directories. |
| `gotm_scpn_master_publications/gotm-scpn_paper-00_the_foundational_framework/` | Paper 0 extraction dumps and assessments. | Processed with the tuned Paper 0 method; see `docs/paper0/paper0_processing_methodology.md`. |
| `gotm_scpn_master_publications/gotm-scpn_paper-01_layer-1_quantum-biological/` | Paper 1 extraction dumps, assessments, and v1 archive. | Legacy or partial extraction attempt; pending rerun with the tuned Paper 0 method. |
| `gotm_scpn_master_publications/gotm-scpn_paper-02_layer-2_neurochemical-neurological/` | Paper 2 dump and assessment. | Raw or early-stage material; pending tuned extraction. |

## Build commands

Run LaTeX from the manuscript directory so local build artefacts stay beside the
source:

```bash
cd paper/submissions/submission_002_phase1_dla_parity
pdflatex phase1_dla_parity.tex
pdflatex phase1_dla_parity.tex
```

```bash
cd paper/submissions/submission_003_rust_vqe_methods
pdflatex rust_vqe_methods.tex
pdflatex rust_vqe_methods.tex
```

```bash
cd paper/submissions/submission_004_scpn_fim_hamiltonian
pdflatex scpn_fim_hamiltonian.tex
bibtex scpn_fim_hamiltonian
pdflatex scpn_fim_hamiltonian.tex
pdflatex scpn_fim_hamiltonian.tex
```

```bash
cd paper/submissions/submission_005_phase3_reduced_pauli_entanglement
pdflatex phase3_entanglement_tomography.tex
bibtex phase3_entanglement_tomography
pdflatex phase3_entanglement_tomography.tex
pdflatex phase3_entanglement_tomography.tex
```

## arXiv submission packaging

For arXiv (either paper):

1. Run `pdflatex` twice locally to verify compilation is clean.
2. Package the source tree so that `pdflatex` on arXiv can rebuild
   the PDF from a single archive:

   ```bash
   cd paper/submissions/submission_002_phase1_dla_parity
   tar czf phase1_dla_parity_arxiv.tar.gz \
       phase1_dla_parity.tex \
       phase1_dla_parity.bbl \
       figures/phase1/leakage_vs_depth.png \
       figures/phase1/asymmetry_vs_depth.png
   ```

3. Upload via `arxiv.org/submit`. Select primary category
   `quant-ph`, cross-list `cond-mat.stat-mech` + `cs.ET` (see
   `.coordination/launch_copy/arxiv.md` for the prepared metadata
   block).

## Build artefacts

`paper/**/*.aux`, `paper/**/*.log`, `paper/**/*.out`, generated PDFs, generated
BibTeX outputs, and source extraction dumps are gitignored unless already
tracked as a deliberate publication artefact.
