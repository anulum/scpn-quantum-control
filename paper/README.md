# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# paper/ - publication sources

Each manuscript or venue package has its own subdirectory. Keep future JOSS
submissions under `paper/joss/<submission_slug>/`; do not place a second JOSS
paper directly under `paper/joss/`.

## Directory map

| Directory | Contents | Status |
|---|---|---|
| `ibm_fez_synchronisation/` | February 2026 `ibm_fez` synchronisation overview in `main.tex` | Contextual overview; compiles locally. |
| `phase1_dla_parity/` | April 2026 `ibm_kingston` DLA-parity manuscript and Markdown short-paper draft | Submission-ready source package, with claim boundaries. |
| `phase3_entanglement_tomography/` | Reduced-Pauli entanglement/tomography follow-up draft | Raw-count execution pending. |
| `s1_feedback_control/` | Monitored feedback versus open-loop dynamic-circuit draft | Live IBM metadata and raw-count execution pending. |
| `rust_vqe_methods/` | Rust/VQE methodology manuscript and local PDF | Methods preprint candidate. |
| `scpn_fim_hamiltonian/` | SCPN/FIM Hamiltonian manuscript and bibliography material | Negative hardware-validation paper candidate. |
| `joss/software_framework_note/` | Current JOSS software-framework note and preview files | One JOSS submission package; future JOSS notes get sibling directories. |
| `paper0_foundational_framework/` | Local Paper 0 extraction dumps and assessments | Ignored research working material. |
| `paper1_quantum_biological/` | Local Paper 1 extraction dumps, assessments, and v1 archive | Ignored research working material. |
| `paper2_neurochemical/` | Local Paper 2 extraction dumps and assessments | Ignored research working material. |

## Build commands

Run LaTeX from the manuscript directory so local build artefacts stay beside the
source:

```bash
cd paper/phase1_dla_parity
pdflatex phase1_dla_parity.tex
pdflatex phase1_dla_parity.tex
```

```bash
cd paper/rust_vqe_methods
pdflatex rust_vqe_methods.tex
pdflatex rust_vqe_methods.tex
```

```bash
cd paper/scpn_fim_hamiltonian
pdflatex scpn_fim_hamiltonian.tex
bibtex scpn_fim_hamiltonian
pdflatex scpn_fim_hamiltonian.tex
pdflatex scpn_fim_hamiltonian.tex
```

## arXiv submission packaging

For arXiv (either paper):

1. Run `pdflatex` twice locally to verify compilation is clean.
2. Package the source tree so that `pdflatex` on arXiv can rebuild
   the PDF from a single archive:

   ```bash
   tar czf phase1_dla_parity_arxiv.tar.gz \
       paper/phase1_dla_parity/phase1_dla_parity.tex \
       paper/phase1_dla_parity/phase1_dla_parity.bbl \
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
