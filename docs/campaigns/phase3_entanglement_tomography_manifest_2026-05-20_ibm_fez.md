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

- Counts artefact: `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_fez_2026-05-20T014536Z.json`
- Reference CSV: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`
- Backend: `ibm_fez`
- Job IDs: `d86h4jlg7okc73el3ra0, d86h5d9789is738vlt7g`

## Outputs

- JSON summary: `data/phase3_entanglement_tomography/entanglement_tomography_summary_2026-05-20_ibm_fez.json`
- Observable rows: `data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20_ibm_fez.csv`
- Observable rows SHA256: `40801f8ce562e462c3b051b6cd4026b48480fb76770c7f164251c5de550efca1`

## Result Snapshot

- Observable rows: `54`
- Mean absolute deviation from exact reference: `0.15282093077862535`
- Maximum absolute deviation from exact reference: `0.47901490766488014`
- Readout-mitigated mean absolute deviation from exact reference: `0.1513169450615788`
- Readout-mitigated maximum absolute deviation from exact reference: `0.4868750566558935`

## Readout Mitigation

- Method: `tensor_product_single_qubit_inverse`
- Calibration circuits: `4`
- Boundary: independent single-qubit readout inversion only; not a full 16-state correlated readout calibration and not a ZNE/PEC result

## Boundary

reduced-Pauli observable analysis only; no scalable tomography, quantum advantage, or backend-general claim
