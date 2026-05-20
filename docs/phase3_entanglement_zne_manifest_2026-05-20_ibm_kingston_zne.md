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

- Counts artefact: `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T030211Z.json`
- Reference CSV: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`
- Backend: `ibm_kingston`
- Job IDs: `d86i8fas46sc73f70vg0, d86ijr9789is738vnh30`

## Outputs

- JSON summary: `data/phase3_entanglement_tomography/entanglement_zne_summary_2026-05-20_ibm_kingston_zne.json`
- Scale rows: `data/phase3_entanglement_tomography/entanglement_zne_scale_rows_2026-05-20_ibm_kingston_zne.csv`
- Scale rows SHA256: `0e2e39786ff8cded087115419f15ee74f0ca63f101c97eb1a93a1b85b936d8c4`
- Channel summary: `data/phase3_entanglement_tomography/entanglement_zne_channel_summary_2026-05-20_ibm_kingston_zne.csv`
- Channel summary SHA256: `c5e5e39132f3e3de79d50fe5cedc66dd4e74258297bdd877dfe2954d0e9baad3`

## Result Snapshot

- Scale rows: `15`
- Channels: `5`
- Noise scales: `[1, 3, 5]`
- Scale-1 mean absolute deviation: `0.4533514489822172`
- Linear ZNE mean absolute deviation: `0.4729152510655504`
- Readout-mitigated linear ZNE mean absolute deviation: `0.480609277847894`
- Quadratic ZNE mean absolute deviation: `0.45559754273221703`

## Readout Mitigation

- Method: `full_correlated_readout_inverse`
- Calibration circuits: `16`
- Boundary: full correlated readout inversion only; not a ZNE/PEC result and not a correction for basis-rotation or coherent gate errors

## Boundary

small preregistered reduced-Pauli ZNE stress test only; not a backend-general, advantage, full-tomography, or full-causal mechanism claim
