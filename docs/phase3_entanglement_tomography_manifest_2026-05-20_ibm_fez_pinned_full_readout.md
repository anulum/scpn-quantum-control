<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- entanglement/tomography analysis manifest -->

# Phase 3 Entanglement/Tomography Analysis Manifest

Date: 2026-05-20

## Inputs

- Counts artefact: `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T020452Z.json`
- Reference CSV: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`
- Backend: `ibm_fez`
- Job IDs: `d86hdk8p0eas73dkv9eg, d86hedp789is738vm7mg`

## Outputs

- JSON summary: `data/phase3_entanglement_tomography/entanglement_tomography_summary_2026-05-20_ibm_fez_pinned_full_readout.json`
- Observable rows: `data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20_ibm_fez_pinned_full_readout.csv`
- Observable rows SHA256: `26f9481cf008892995d9ad43e00acf1fae267a56cd5614e5fb2cf2b892bf95ff`

## Result Snapshot

- Observable rows: `54`
- Mean absolute deviation from exact reference: `0.14944515917368706`
- Maximum absolute deviation from exact reference: `0.4734810534982135`
- Readout-mitigated mean absolute deviation from exact reference: `0.19528217448414878`
- Readout-mitigated maximum absolute deviation from exact reference: `0.9912779304997331`

## Readout Mitigation

- Method: `full_correlated_readout_inverse`
- Calibration circuits: `16`
- Boundary: full correlated readout inversion only; not a ZNE/PEC result and not a correction for basis-rotation or coherent gate errors

## Boundary

reduced-Pauli observable analysis only; no scalable tomography, quantum advantage, or backend-general claim
