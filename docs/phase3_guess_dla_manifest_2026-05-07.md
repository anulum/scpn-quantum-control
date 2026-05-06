<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Phase 3 GUESS DLA Manifest -->

# Phase 3 GUESS DLA Manifest

Date: 2026-05-07

## Hardware jobs

- Backend: `ibm_marrakesh`
- Job IDs: `d7tt5lkt738s73cib64g, d7tt7oaudops7398fdt0`
- Source artefact: `data/phase3_guess_dla/phase3_guess_ibm_marrakesh_2026-05-06T234602Z.json`
- Source SHA256: `ea9f87139679c933da1ad0ee35954a60701064a25281c02c252a3fa5d4bae3bc`

## Decision flags

- Raw usable fits: `6`
- Corrected usable fits: `5`

## Artefacts

- Summary JSON: `data/phase3_guess_dla/phase3_guess_summary_2026-05-07.json`
- Fit rows: `data/phase3_guess_dla/phase3_guess_fit_rows_2026-05-07.csv`
- Witness rows: `data/phase3_guess_dla/phase3_guess_extrapolation_rows_2026-05-07.csv`

## Reproduction

```bash
./.venv-linux/bin/python scripts/analyse_phase3_guess_dla.py
```
