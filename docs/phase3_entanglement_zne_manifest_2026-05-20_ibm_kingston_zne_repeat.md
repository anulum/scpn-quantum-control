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

- Counts artefact: `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_kingston_2026-05-20T114719Z.json`
- Reference CSV: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`
- Backend: `ibm_kingston`
- Job IDs: `d86pul0p0eas73dla3dg, d86pul8p0eas73dla3eg`

## Outputs

- JSON summary: `data/phase3_entanglement_tomography/entanglement_zne_summary_2026-05-20_ibm_kingston_zne_repeat.json`
- Scale rows: `data/phase3_entanglement_tomography/entanglement_zne_scale_rows_2026-05-20_ibm_kingston_zne_repeat.csv`
- Scale rows SHA256: `f8e5d5e1ca8a30aaf5efa0bbabb96141e50b48852917a984c4ffe02532e090a4`
- Channel summary: `data/phase3_entanglement_tomography/entanglement_zne_channel_summary_2026-05-20_ibm_kingston_zne_repeat.csv`
- Channel summary SHA256: `d9b23413d1618dfd003b2848c3289e5c53f854c68e2d8fac991f4c2ec738f1fc`

## Result Snapshot

- Scale rows: `15`
- Channels: `5`
- Noise scales: `[1, 3, 5]`
- Scale-1 mean absolute deviation: `0.4655910323155505`
- Linear ZNE mean absolute deviation: `0.48161750800999487`
- Readout-mitigated linear ZNE mean absolute deviation: `0.48931903445861447`
- Quadratic ZNE mean absolute deviation: `0.48251293904941395`

## Readout Mitigation

- Method: `full_correlated_readout_inverse`
- Calibration circuits: `16`
- Boundary: full correlated readout inversion only; not a ZNE/PEC result and not a correction for basis-rotation or coherent gate errors

## Boundary

small preregistered reduced-Pauli ZNE stress test only; not a backend-general, advantage, full-tomography, or full-causal mechanism claim
