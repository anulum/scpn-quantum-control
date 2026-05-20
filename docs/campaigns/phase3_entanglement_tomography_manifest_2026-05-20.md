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

- Counts artefact: `data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T004334Z.json`
- Reference CSV: `data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv`
- Backend: `ibm_marrakesh`
- Job IDs: `d86g7h1789is738vkreg, d86ggpis46sc73f6v170`

## Outputs

- JSON summary: `data/phase3_entanglement_tomography/entanglement_tomography_summary_2026-05-20.json`
- Observable rows: `data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20.csv`
- Observable rows SHA256: `b280db120f9f13dfb8dd6b8d4e2e6a86a86feff714e2514be01bd5b3ed95bd83`

## Result Snapshot

- Observable rows: `54`
- Mean absolute deviation from exact reference: `0.12989296537986128`
- Maximum absolute deviation from exact reference: `0.5560906424788263`
- Readout-mitigated mean absolute deviation from exact reference: `0.12930386230559288`
- Readout-mitigated maximum absolute deviation from exact reference: `0.5648340543137427`

## Readout Mitigation

- Method: `tensor_product_single_qubit_inverse`
- Calibration circuits: `4`
- Boundary: independent single-qubit readout inversion only; not a full 16-state correlated readout calibration and not a ZNE/PEC result

## Boundary

reduced-Pauli observable analysis only; no scalable tomography, quantum advantage, or backend-general claim
