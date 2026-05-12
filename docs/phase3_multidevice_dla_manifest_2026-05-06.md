<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Quantum Control — Phase 3 Multi-Device DLA Manifest -->

# Phase 3 Multi-Device DLA Manifest

Date: 2026-05-06

This manifest records the approved second-backend DLA parity replication run on
IBM Quantum hardware.

## Preregistration

Protocol:

```text
docs/dla_multidevice_replication_prereg_2026-05-06.md
```

Execution script:

```text
scripts/phase3_multidevice_dla_ibm.py
```

## Backend and Budget Gate

| Field | Value |
|---|---|
| Backend | `ibm_marrakesh` |
| Backend class | 156-qubit Heron-class backend |
| Excluded backend | `ibm_kingston` |
| Live readiness timestamp | `2026-05-06T171231Z` |
| Main job | `ibm-run-63e0a1af74a38c9c` |
| Readout job | `ibm-run-0f96961442e05a77` |
| Main circuits | `144` |
| Readout circuits | `4` |
| Total circuits | `148` |
| Main shots | `4096` |
| Readout shots | `8192` |
| Estimated QPU minutes | `1.3566666666666667` |
| Preregistered ceiling | `10` minutes |

Live readiness:

```text
depth min/mean/max: 1 / 214.75 / 414
total-gate min/mean/max: 4 / 548.9121621621622 / 1086
ECR min/mean/max: 0 / 0.0 / 0
```

The max depth `414` passed the preregistered guard derived from the Phase 2
A+G live envelope. No extra QPU job was submitted outside this circuit matrix.

## Artefacts

| Artefact | SHA256 |
|---|---|
| `data/phase3_multidevice_dla/phase3_multidevice_ibm_marrakesh_2026-05-06T171231Z.json` | `cf76b66c69fd6901bc2a9e2c527e978a1c3c1347761d6c16656831a7babfa36a` |
| `data/phase3_multidevice_dla/phase3_multidevice_summary_2026-05-06.json` | `751965194a1449767c623dcb119c2514be800efcea1a9652d8acf711cd5471eb` |
| `data/phase3_multidevice_dla/phase3_multidevice_row_metrics_2026-05-06.csv` | `15f5de37592947452e1b076e50d7f74b23854b3ca579eb489f0809db9b143eec` |
| `data/phase3_multidevice_dla/phase3_multidevice_readout_corrected_summary_2026-05-06.json` | `c43e25d56555f4e1a4593185efddbe51877f638b607c9fa84bcfc07c58ae916f` |
| `data/phase3_multidevice_dla/phase3_multidevice_readout_corrected_rows_2026-05-06.csv` | `707d9b2767e3d5a635a6919318119653b495b00ca3661247c41fcd3c3756bc33` |

## Per-Depth Summary

Parity leakage is reported as opposite-parity counts divided by total counts.
Relative asymmetry is `(mean_even - mean_odd) / mean_odd`.

| Depth | Even leakage | Odd leakage | Relative asymmetry | Interpretation |
|---:|---:|---:|---:|---|
| `4` | `0.082153` | `0.084290` | `-0.025344` | opposite sign |
| `6` | `0.105184` | `0.101440` | `+0.036903` | same sign, small |
| `8` | `0.123108` | `0.123271` | `-0.001320` | near zero |
| `10` | `0.133586` | `0.136983` | `-0.024803` | opposite sign |
| `14` | `0.163127` | `0.176229` | `-0.074348` | opposite sign |
| `20` | `0.198771` | `0.205892` | `-0.034585` | opposite sign |

## Full-Basis Readout-Corrected Summary

The matching `ibm_marrakesh` full-basis readout calibration is documented in
`docs/readout_full_basis_manifest_2026-05-06.md`. It uses physical qubits
`[5,6,7,8]` and has assignment-matrix condition number `1.0756988057943142`.

| Depth | Corrected even leakage | Corrected odd leakage | Corrected relative asymmetry | Interpretation |
|---:|---:|---:|---:|---|
| `4` | `0.043936` | `0.064233` | `-0.315987` | opposite sign |
| `6` | `0.072900` | `0.080663` | `-0.096237` | opposite sign |
| `8` | `0.096515` | `0.100098` | `-0.035798` | opposite sign |
| `10` | `0.110723` | `0.110939` | `-0.001953` | near zero |
| `14` | `0.143522` | `0.150375` | `-0.045574` | opposite sign |
| `20` | `0.178952` | `0.188365` | `-0.049971` | opposite sign |

The full-basis readout correction does not rescue a positive backend-transfer
replication. It strengthens the conservative conclusion that the previously
promoted `ibm_kingston` sign is backend/calibration/layout sensitive.

## Claim Boundary

The result is best read as a multi-device boundary condition. It does not
support a simple backend-stable replication of the `ibm_kingston` Phase 2 A+G
positive leakage-asymmetry pattern.

Supported:

- `ibm_marrakesh` raw-count replication attempt under the preregistered
  reduced `n=4` circuit matrix;
- mixed sign and mostly opposite-sign leakage asymmetry across the promoted
  depths;
- evidence that the earlier `ibm_kingston` effect is backend/calibration/layout
  sensitive.
- full-basis readout-corrected evidence on the matching `ibm_marrakesh`
  four-qubit layout showing non-positive relative asymmetry across all
  promoted depths.

Blocked:

- DLA-parity-only causality;
- backend-universal protection;
- monotone scaling;
- broad quantum advantage;
- GUESS mitigation validation;
- full readout-matrix mitigation from the four readout states.

## Reproduction

Generate the readiness/submission artefact:

```bash
./.venv-linux/bin/python scripts/phase3_multidevice_dla_ibm.py \
  --backend ibm_marrakesh \
  --submit \
  --confirm-budget
```

The generated summary and row metrics are derived from:

```text
data/phase3_multidevice_dla/phase3_multidevice_ibm_marrakesh_2026-05-06T171231Z.json
```

No further QPU work is required to recompute the rows above.
