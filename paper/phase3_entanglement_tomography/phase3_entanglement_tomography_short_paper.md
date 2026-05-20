# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Reduced-Pauli entanglement checks for DLA-sector leakage mechanisms on IBM Heron hardware

**Miroslav Šotek** [ORCID: 0009-0009-3560-0851]
*ANULUM Research, Marbach SG, Switzerland*
*Contact: protoscience@anulum.li*

**Date:** 2026-05-20
**Status:** Draft scaffold, raw-count execution pending
**Target venue:** short communication / workshop submission candidate

---

## Abstract

Parity-sector leakage asymmetry in small Kuramoto-XY circuits is a useful
hardware observable only if the mechanism can be separated from layout,
prepared-state, and readout artefacts. We present a cost-bounded reduced-Pauli
entanglement/tomography protocol for the promoted Phase 3 DLA-parity and
Fisher-information-modified Kuramoto-XY circuit families. The protocol measures
54 exact-reference Pauli observables across six source circuits, grouped into
nine measurement bases with three repetitions per source/basis pair, plus four
readout calibration states. A live no-submit preflight on IBM `ibm_marrakesh`
selected physical qubits `[1,2,3,4]`, transpiled 166 circuits, and estimated
1.52 QPU minutes under a 25 minute ceiling, with maximum depth 388 and maximum
measurement-basis depth expansion 1.072. The paper's physics claim remains
pending until the raw-count run is executed and the preregistered reducer
compares measured correlators with exact references. The intended claim is
bounded: whether reduced-Pauli entanglement structure accompanies the observed
leakage/retention mechanisms in this fixed small-system hardware setting.

## 1. Introduction

The Kuramoto-XY hardware programme has produced several useful but deliberately
bounded observations. The original DLA-parity campaign measured a
sector-resolved leakage asymmetry on IBM Heron-class hardware. Later Phase 3
controls showed that the sign and magnitude of the effect are sensitive to
backend, prepared state, layout, and readout context. This makes a direct
mechanism question unavoidable: do the promoted circuits also show a measurable
change in reduced entanglement structure, or is the leakage contrast better
treated as a readout/layout/decoherence boundary?

Full tomography is unnecessary for this question. The observable of interest is
not a reconstructed four-qubit state, but a small set of reduced Pauli
correlators on logical edges, together with half-chain purity proxies already
computed from exact references. The protocol therefore uses reduced-Pauli
measurements and rejects scalable-tomography language.

## 2. Protocol

The promoted source circuits are:

| Label | Family | Initial state | Depth | Parameter |
|---|---|---|---:|---:|
| `dla_even_shallow` | DLA parity | `0011` | 6 | none |
| `dla_odd_shallow` | DLA parity | `0001` | 6 | none |
| `dla_even_signal` | DLA parity | `0011` | 10 | none |
| `dla_odd_signal` | DLA parity | `0001` | 10 | none |
| `fim_lambda0_reference` | FIM pair | `0011` | 4 | `lambda=0` |
| `fim_lambda4_feedback` | FIM pair | `0011` | 4 | `lambda=4` |

For each source circuit, the protocol measures the nine basis settings:

```text
IIXX, IIYY, IIZZ, IXXI, IYYI, IZZI, XXII, YYII, ZZII
```

Each source/basis pair is repeated three times at 2048 shots. Four readout
states are measured at 8192 shots:

```text
0000, 0001, 0011, 1111
```

The resulting hardware block contains 162 main circuits and 4 readout circuits.

## 3. Live Preflight

The live no-submit preflight was run on 2026-05-20:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh
```

It produced:

```text
data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T001956Z.json
```

Preflight summary:

| Gate | Value |
|---|---:|
| Backend | `ibm_marrakesh` |
| Selected physical qubits | `[1,2,3,4]` |
| Main circuits | 162 |
| Readout circuits | 4 |
| Total circuits | 166 |
| Estimated QPU minutes | 1.5217 |
| Budget ceiling minutes | 25.0 |
| Maximum transpiled depth | 388 |
| Maximum basis-expansion ratio | 1.0718232044198894 |

The preflight passed. No QPU job was submitted.

## 4. Analysis Plan

The approved raw-count artefact will be reduced with:

```bash
python scripts/analyse_phase3_entanglement_tomography.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_<backend>_<timestamp>.json
```

For each measured circuit, the reducer computes:

```math
\langle P \rangle = \frac{1}{N}\sum_b n_b \prod_{i \in \operatorname{supp}(P)} (-1)^{b_i},
```

where \(P\) is the measured Pauli label after basis rotation and \(n_b\) is the
count of bitstring \(b\). Repetitions are grouped by circuit label and basis
setting, then compared with exact reference expectations from:

```text
data/phase3_entanglement_tomography/entanglement_observable_rows_2026-05-07.csv
```

Primary outputs:

- `entanglement_tomography_summary_<date>.json`;
- `entanglement_tomography_rows_<date>.csv`;
- `phase3_entanglement_tomography_manifest_<date>.md`.

## 5. Results

Raw-count execution is pending.

This section must not be filled from simulator values or from the live preflight
alone. It should be populated only after:

1. the approved QPU command is run;
2. job IDs and raw counts are saved in the live artefact;
3. `scripts/analyse_phase3_entanglement_tomography.py` generates the summary,
   rows, and manifest;
4. the generated rows are reviewed for readout sensitivity and uncertainty.

## 6. Interpretation Rules

The result supports a mechanism interpretation only if measured correlator
deviations are larger than their uncertainty and are stable under the readout
boundary. The interpretation is downgraded if any of the following occur:

- readout correction changes the sign of the promoted comparison;
- uncertainty intervals are wider than the measured deviation from reference;
- DLA and FIM families show no coherent separation;
- measured correlators are consistent with product-state or readout artefact
  explanations.

## 7. Claim Boundary

Safe claims after successful analysis:

- reduced-Pauli correlators were measured for a preregistered small-system
  Kuramoto-XY hardware block;
- measured correlators agree or disagree with exact classical references under
  a fixed backend, layout, and shot budget;
- the result constrains whether entanglement-structure observables accompany
  the previously observed leakage/retention mechanisms.

Blocked claims:

- quantum advantage;
- scalable tomography;
- backend-general entanglement dynamics;
- full-state reconstruction;
- claims about unmeasured subsystems, depths, layouts, or backends.

## 8. Reproducibility

Offline readiness:

```bash
python scripts/generate_entanglement_tomography_readiness.py
```

Live preflight:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh
```

Approved hardware submission:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh --submit --confirm-budget
```

Post-run analysis:

```bash
python scripts/analyse_phase3_entanglement_tomography.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_<backend>_<timestamp>.json
```

## 9. Conclusion

The campaign is ready for an approved, budget-confirmed IBM submission. The
scientific paper should be framed as a mechanism-boundary study: reduced-Pauli
tomography is used to test whether DLA-sector leakage and FIM retention
phenomena have a measurable entanglement-structure companion in a fixed
small-system hardware setting.
