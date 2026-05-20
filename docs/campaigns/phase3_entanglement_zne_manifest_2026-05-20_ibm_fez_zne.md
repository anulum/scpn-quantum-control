<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- Phase 3 ZNE analysis manifest -->

# Phase 3 Entanglement ZNE Analysis Manifest

Date: 2026-05-20

## Inputs

- Counts artefact: `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T023600Z.json`
- Reference CSV: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`
- Backend: `ibm_fez`
- Job IDs: `d86hs6qs46sc73f70h90, d86hsltg7okc73el4lg0`

## Outputs

- JSON summary: `data/phase3_entanglement_tomography/entanglement_zne_summary_2026-05-20_ibm_fez_zne.json`
- Scale rows: `data/phase3_entanglement_tomography/entanglement_zne_scale_rows_2026-05-20_ibm_fez_zne.csv`
- Scale rows SHA256: `f93435be67f98e4f6ded9d1e0378c06c2cc4586343e1cf2bed895fa00f917e5a`
- Channel summary: `data/phase3_entanglement_tomography/entanglement_zne_channel_summary_2026-05-20_ibm_fez_zne.csv`
- Channel summary SHA256: `775594dd3efc8f519111365179659d41903cf774bcde1d20a378d53b6ee8e7aa`

## Result Snapshot

- Scale rows: `15`
- Channels: `5`
- Noise scales: `[1, 3, 5]`
- Scale-1 mean absolute deviation: `0.4196274906488838`
- Linear ZNE mean absolute deviation: `0.44124749932943935`
- Readout-mitigated linear ZNE mean absolute deviation: `0.4468173734963394`
- Quadratic ZNE mean absolute deviation: `0.4477931807530503`

## Readout Mitigation

- Method: `full_correlated_readout_inverse`
- Calibration circuits: `16`
- Boundary: full correlated readout inversion only; not a ZNE/PEC result and not a correction for basis-rotation or coherent gate errors

## Boundary

small preregistered reduced-Pauli ZNE stress test only; not a backend-general, advantage, full-tomography, or full-causal mechanism claim
