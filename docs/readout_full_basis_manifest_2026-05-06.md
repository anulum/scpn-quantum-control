<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Readout Full-Basis Manifest -->

# Readout Full-Basis Manifest

Date: 2026-05-06

This manifest records the full computational-basis readout calibration for the
Phase 3 `ibm_marrakesh` DLA replication dataset.

## Scope

Target dataset:

```text
data/phase3_multidevice_dla/phase3_multidevice_ibm_marrakesh_2026-05-06T171231Z.json
```

Execution script:

```text
scripts/readout_full_basis_calibration_ibm.py
```

Backend and layout:

| Field | Value |
|---|---|
| Backend | `ibm_marrakesh` |
| Physical qubits | `[5, 6, 7, 8]` |
| Logical qubits | `4` |
| Basis states | `16` |
| Shots per state | `8192` |
| Job ID | `d7tnljvljm6s73bcsql0` |
| Estimated QPU minutes | `0.14666666666666667` |
| Preregistered ceiling | `5` minutes |

Live readiness:

```text
depth min/mean/max: 1 / 1.9375 / 2
gate min/mean/max: 4 / 6.0 / 8
```

## Artefacts

| Artefact | SHA256 |
|---|---|
| `data/readout_full_basis/readout_full_basis_ibm_marrakesh_4q_2026-05-06T173054Z.json` | `ad49289a204c2a54c31ace88ac903bbf6fa8a1afe869b54e691ee1808c4f92b1` |
| `data/readout_full_basis/readout_full_basis_summary_2026-05-06.json` | `315aefef880a8c4bf328a20e7ec99eacdfe68db81b9dff9db8aec94c06fac8ac` |
| `data/readout_full_basis/readout_full_basis_matrix_2026-05-06.csv` | `897748bedea05b3ed2aef71342d01d8e5f6823d902a3a23e6e0a279715a108bf` |
| `data/readout_full_basis/readout_full_basis_rows_2026-05-06.csv` | `0ae1b26c6db1d0e0a5303ae8589a1c54f959be8fbd82c3436a03e4f3a16fd4a7` |

## Calibration Quality

| Metric | Value |
|---|---:|
| Mean retention | `0.9699935913085938` |
| Minimum retention | `0.952392578125` |
| Maximum retention | `0.98193359375` |
| Mean parity-flip rate | `0.0296478271484375` |
| Maximum parity-flip rate | `0.0472412109375` |
| Assignment-matrix condition number | `1.0756988057943142` |

The condition number is low, so the assignment matrix is numerically stable for
readout-only correction of the matching `ibm_marrakesh` four-qubit layout.

## Claim Boundary

Supported:

- full-basis readout assignment calibration for `ibm_marrakesh` physical qubits
  `[5,6,7,8]`;
- readout-only correction eligibility for the matching Phase 3 DLA raw-count
  dataset;
- row retention, parity-flip, and condition-number diagnostics.

Blocked:

- mitigation of gate errors, Trotter error, decoherence, or crosstalk;
- reuse on another backend, another physical layout, or a distant calibration
  window without drift justification;
- any claim that readout calibration alone explains the Phase 3 sign pattern
  until the DLA rows are explicitly reanalysed with this matrix.

## Reproduction

The submitted command was:

```bash
./.venv-linux/bin/python scripts/readout_full_basis_calibration_ibm.py \
  --backend ibm_marrakesh \
  --physical-qubits 5,6,7,8 \
  --submit \
  --confirm-budget
```

No further QPU work is required to rebuild the matrix and summary artefacts
from the raw-count JSON.
