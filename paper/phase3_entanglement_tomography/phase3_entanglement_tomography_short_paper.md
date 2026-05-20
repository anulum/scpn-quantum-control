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
**Status:** Draft with completed IBM raw-count execution and first-pass analysis
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
readout calibration states. The approved IBM `ibm_marrakesh` execution selected
physical qubits `[1,2,3,4]`, transpiled 166 circuits, and completed jobs
`d86g7h1789is738vkreg` and `d86ggpis46sc73f6v170` under the preregistered 25
minute ceiling. The preregistered reducer produced 54 observable rows with mean
absolute deviation 0.1299 and maximum absolute deviation 0.5561 relative to
exact references. The intended claim is bounded: reduced-Pauli correlators show
large hardware deviations in the fixed small-system setting, with no scalable
tomography, quantum-advantage, or backend-general claim.

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

## 3. Live Execution

The live preflight was run on 2026-05-20:

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

The approved budget-confirmed execution was then run:

```bash
python scripts/phase3_entanglement_tomography_ibm.py --backend ibm_marrakesh --submit --confirm-budget
```

Completed artefact:

```text
data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T004334Z.json
```

Completed jobs:

```text
d86g7h1789is738vkreg
d86ggpis46sc73f6v170
```

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

Raw-count execution completed on `ibm_marrakesh`.

First-pass reduced-Pauli analysis was generated with:

```bash
python scripts/analyse_phase3_entanglement_tomography.py \
  data/phase3_entanglement_tomography/entanglement_tomography_live_ibm_marrakesh_2026-05-20T004334Z.json
```

Outputs:

- `data/phase3_entanglement_tomography/entanglement_tomography_summary_2026-05-20.json`;
- `data/phase3_entanglement_tomography/entanglement_tomography_rows_2026-05-20.csv`;
- `docs/phase3_entanglement_tomography_manifest_2026-05-20.md`.

Summary:

| Metric | Value |
|---|---:|
| Backend | `ibm_marrakesh` |
| Submitted jobs | `d86g7h1789is738vkreg`, `d86ggpis46sc73f6v170` |
| Observable rows | 54 |
| Mean absolute deviation from exact reference | 0.12989296537986128 |
| Maximum absolute deviation from exact reference | 0.5560906424788263 |
| Rows SHA256 | `3d18308d60fe32827bae7517f18fd71690240b105779287408c4749cb0e7dc72` |

Largest observed deviations occur in DLA odd/even shallow and signal edge
correlators, especially `XXII`, `YYII`, `IIXX`, and `IIYY`. The largest single
deviation is for `dla_odd_signal`, initial `0001`, depth 10, basis `XXII`,
where the measured expectation is 0.431640625 against exact reference
-0.12445001747882631.

## 6. Interpretation Rules

The result supports a mechanism interpretation only if measured correlator
deviations are larger than their uncertainty and are stable under the readout
boundary. The interpretation is downgraded if any of the following occur:

- readout correction changes the sign of the promoted comparison;
- uncertainty intervals are wider than the measured deviation from reference;
- DLA and FIM families show no coherent separation;
- measured correlators are consistent with product-state or readout artefact
  explanations.

The current first-pass result establishes large reduced-Pauli deviations from
exact references in the measured hardware window. It does not yet establish
that those deviations are an entanglement mechanism rather than a compound
effect of coherent hardware error, readout context, layout, and circuit depth.
That distinction is the central interpretation work for the paper.

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

The campaign has completed the approved IBM execution and first-pass
reduced-Pauli analysis. The scientific paper should be framed as a
mechanism-boundary study: reduced-Pauli tomography shows sizeable measured
correlator deviations in the same small-system setting as the DLA/FIM hardware
programme, but the conservative contribution is to bound and interpret that
structure rather than to claim scalable tomography, backend-general dynamics, or
quantum advantage.
