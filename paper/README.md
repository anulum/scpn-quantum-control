# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# paper/ — publication sources

Two arXiv-submission-ready manuscripts live here.

## `main.tex` — February 2026 ibm_fez campaign

Long-form paper on the 33-job Feb 2026 campaign on `ibm_fez`.
Status: compiles to `main.pdf`, self-contained `thebibliography`.

```bash
pdflatex main.tex  # twice for cross-references + bibliography
```

## `phase1_dla_parity.tex` — April 2026 ibm_kingston DLA-parity campaign

Short-paper companion for the April 2026 Phase 1 campaign on
`ibm_kingston` (342 circuits, $n = 4$, DLA parity asymmetry
$+10.8 \pm 1.1\,\%$). Target venue: *Quantum Science and Technology*
Letter or *Physical Review Research*. Four-column-format pages in
the current build.

```bash
pdflatex phase1_dla_parity.tex  # twice for cross-references
```

LaTeX class: `revtex4-2` with `twocolumn, aps, prresearch,
superscriptaddress, floatfix, notitlepage, longbibliography`
options. All figures pulled from `figures/phase1/*.png`, all
references inlined in `thebibliography` (no external `.bib` needed
for arXiv submission).

## arXiv submission packaging

For arXiv (either paper):

1. Run `pdflatex` twice locally to verify compilation is clean.
2. Package the source tree so that `pdflatex` on arXiv can rebuild
   the PDF from a single archive:

   ```bash
   tar czf phase1_dla_parity_arxiv.tar.gz \
       paper/phase1_dla_parity.tex \
       paper/phase1_dla_parity.bbl \
       figures/phase1/leakage_vs_depth.png \
       figures/phase1/asymmetry_vs_depth.png
   ```

3. Upload via `arxiv.org/submit`. Select primary category
   `quant-ph`, cross-list `cond-mat.stat-mech` + `cs.ET` (see
   `.coordination/launch_copy/arxiv.md` for the prepared metadata
   block).

## Build artefacts

`paper/*.aux`, `paper/*.log`, `paper/*.out`, `paper/*.pdf` are all
gitignored. Clean rebuild with `rm paper/*.aux paper/*.log
paper/*.out paper/*.pdf && pdflatex paper/phase1_dla_parity.tex`.
