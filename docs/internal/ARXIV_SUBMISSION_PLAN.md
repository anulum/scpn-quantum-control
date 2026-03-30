# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — arXiv Submission Plan

# arXiv Submission Plan

**Date:** 2026-03-30
**Target:** quant-ph (primary), cond-mat.stat-mech (cross-list)

## Status

The preprint exists as `docs/preprint.md` (387 lines, 16 PDF figures).
arXiv requires LaTeX source or PDF upload.

## Action Items

### 1. Convert preprint.md → LaTeX

Source: `docs/preprint.md`
Output: `paper/main.tex` + `paper/figures/`

Structure (already in preprint.md):
- Abstract
- §1 Introduction (Kuramoto → XY mapping)
- §2 Methods (Trotterisation, coupling matrix, error mitigation)
- §3 Simulation results (BKT, DTC, Schmidt gap, OTOC, Krylov, MBL)
- §4 Hardware results (ibm_fez, 20 experiments, CHSH, QKD, UPDE)
- §5 DLA parity theorem
- §6 Synchronisation witnesses
- §7 Discussion + limitations
- References

### 2. Figures (ready)

16 PDF figures in `figures/publication/`:
- fig1–fig9: simulation (entanglement, Krylov, OTOC, BKT, Schmidt, DTC, witness, DLA, correlator)
- fig10–fig14: IBM hardware (overview, characterisation, full analysis, quantitative, complete)
- fig15–fig16: MBL (level spacing, eigenstate entanglement)

### 3. Required for arXiv account

- **arXiv account:** Miroslav needs to register at arxiv.org (ORCID login available)
- **Endorsement:** quant-ph may require endorsement for first-time submitters.
  Options: (a) find an endorser, (b) submit to a non-endorsed category first
- **License:** arXiv default CC BY 4.0 is compatible with AGPL-3.0 code licence

### 4. Zenodo DOI

Already exists: `10.5281/zenodo.18821929` (referenced in README).
Update Zenodo archive to v0.9.5 before submission so arXiv preprint cites current version.

### 5. Honest framing checklist

- [ ] No "no prior art" claims (fixed 2026-03-30)
- [ ] Prior work on quantum sync measures cited (Ameri, Ma, Galve)
- [ ] Standard tools credited (OTOC → MSS 2016, level-spacing → Oganesyan & Huse)
- [ ] DLA formula credited to general framework (Wiersema et al. 2023)
- [ ] "To our knowledge" qualifier on hardware-first claims
- [ ] Limitations section present and honest
- [ ] No "paradigm shift" / "groundbreaking" language

### 6. Conversion approach

Option A: `pandoc docs/preprint.md -o paper/main.tex` + manual cleanup
Option B: Write LaTeX from scratch using preprint.md as outline

Option A is faster. Pandoc handles most math, figures need manual \includegraphics.

### 7. Timeline

| Step | Effort | Dependency |
|------|--------|-----------|
| pandoc conversion | 30 min | none |
| LaTeX cleanup + refs | 2–3 hours | conversion |
| arXiv account + endorsement | 1–7 days | Miroslav |
| Zenodo v0.9.5 archive | 15 min | none |
| Submit | 5 min | account + endorsement |

### 8. Three papers or one?

Currently three separate pages docs: preprint, sync witnesses, DLA parity.
For arXiv, recommend **one combined paper** — the sync witnesses and DLA parity
are results sections within the preprint, not standalone papers. Splitting into
three would look like salami-slicing given the project's size.
