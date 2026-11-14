# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# paper/ - publication sources

Each manuscript or venue package has its own numbered subdirectory. Keep
publication submissions under `paper/submissions/` and JOSS drafts under
`paper/submissions_joss/`. The JOSS paper prepared for submission lives in
`paper/joss/`, because the JOSS review bot uses the first `paper.md` it finds
in a sorted walk of the repository, and `paper/joss/` comes before the frozen
drafts.

Every published submission directory holds its exported PDF, named after the
paper title. Its metadata (title, author, Zenodo DOI, keywords) comes from the
paper's Zenodo record and is set when the PDF is compiled; the frozen `.tex`
source is not edited. Work in progress lives in `paper/drafts/`, which is
gitignored: only published work appears in this repository.

## Directory map

| Directory | Contents | Status |
|---|---|---|
| `joss/` | JOSS paper for `scpn-quantum-control` 1.3.0 (`paper.md`, `paper.bib`). | Prepared for JOSS submission (13 Nov 2026); not yet submitted. |
| `publication_planning/` | Cross-paper planning notes, preprint outlines, programme overview, and publication-framing guidance. | Planning material, not a submission source package. |
| `submissions/submission_001_ibm_fez_synchronisation/` | February 2026 `ibm_fez` synchronisation overview in `main.tex`, with local `figures/publication/`. | Contextual overview; compiles locally. |
| `submissions/submission_002_phase1_dla_parity/` | April 2026 `ibm_kingston` DLA-parity manuscript, short-paper draft, and local `figures/phase1/` + `figures/phase2/`. | Submitted to Physical Review A on 13 Nov 2026 (see [Journal submissions](#journal-submissions)); source package with claim boundaries. |
| `submissions/submission_003_rust_vqe_methods/` | Rust/VQE methodology manuscript and local PDF. | Methods preprint candidate. |
| `submissions/submission_004_scpn_fim_hamiltonian/` | SCPN/FIM Hamiltonian manuscript and bibliography material. | Negative hardware-validation paper candidate. |
| `submissions/submission_005_phase3_reduced_pauli_entanglement/` | Reduced-Pauli entanglement REVTeX manuscript, bibliography, PDF, and local `figures/phase3/`. | Completed multi-backend IBM run; venue-style source package. |
| `submissions/submission_006_s1_feedback_control/` | Monitored feedback versus open-loop dynamic-circuit manuscript, bibliography, PDF, and local `figures/s1_feedback_control/`. | Multi-backend dynamic-circuit control-boundary paper. Submitted to Physical Review Applied on 13 Nov 2026 (see [Journal submissions](#journal-submissions)). |
| `submissions/submission_007_iqm_garnet_preregistered_campaigns/` | IQM Garnet manuscript on five preregistered campaigns (parity-sector leakage, its variability between calibration epochs, twelve-epoch depth ordering, layout transfer), with inline reference list and local `figures/` drawn by `scripts/generate_iqm_garnet_paper_assets.py`. | Preprint on Zenodo, DOI [10.5281/zenodo.23298119](https://doi.org/10.5281/zenodo.23298119). Submitted to Physical Review A on 13 Nov 2026 (see [Journal submissions](#journal-submissions)). |
| `submissions_joss/submission_joss_001_software_framework_note/` | May 2026 JOSS software-framework note and preview files. | Frozen: source of the Zenodo preprint [10.5281/zenodo.20382333](https://doi.org/10.5281/zenodo.20382333). Superseded for JOSS by `joss/`. |
| `submissions_joss/submission_joss_002_kuramoto_toolkit/` | July 2026 JOSS draft for the standalone `oscillatools` toolkit. | Draft; not submitted. |

## Journal submissions

Each manuscript was submitted from its frozen source here. Only the front
matter changed for submission: affiliation, ORCID placement and date. The
`.tex` files in this repository are not edited.

| Paper | Journal and article type | Accession code | Submitted |
|---|---|---|---|
| `submission_002_phase1_dla_parity` | Physical Review A, Regular Article (quantum information science) | AY12834 | 13 Nov 2026 |
| `submission_006_s1_feedback_control` | Physical Review Applied, Research Article | NY10630 | 13 Nov 2026 |
| `submission_007_iqm_garnet_preregistered_campaigns` | Physical Review A, Regular Article (quantum information science) | AY12835 | 13 Nov 2026 |

Notes:

- All three use standard publication, without open-access fees.
- The 006 manuscript carries its AI-use disclosure in the Acknowledgments.
- For 002 and 007, the disclosure was sent to the journal on 13 Nov 2026.
  It is added to the Acknowledgments at revision.

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

```bash
python scripts/generate_iqm_garnet_paper_assets.py --latex-text
cd paper/submissions/submission_007_iqm_garnet_preregistered_campaigns
pdflatex iqm_garnet_preregistered_campaigns.tex
pdflatex iqm_garnet_preregistered_campaigns.tex
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
   `<private-local-record>` for the prepared metadata
   block).

## Build artefacts

`paper/**/*.aux`, `paper/**/*.log`, `paper/**/*.out`, generated PDFs, generated
BibTeX outputs, and source extraction dumps are gitignored unless already
tracked as a deliberate publication artefact.
