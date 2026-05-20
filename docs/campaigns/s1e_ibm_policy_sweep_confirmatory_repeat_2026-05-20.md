# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- S1e IBM policy-sweep confirmatory repeat

# S1e IBM Policy-Sweep Confirmatory Repeat

Date: 2026-05-20
Backend: `ibm_kingston`
Experiment ID: `s1e_policy_sweep_confirmatory_repeat_2026-05-20`
Parent experiment: `s1_dynamic_feedback_preregistration_2026-05-06`

## Artefacts

- Readiness: `data/s1_feedback_loop/s1e_xy_observable_readiness_ibm_kingston_20260520T140919Z.json`
- Readiness SHA-256: `80b78f3deafae9465e1f057be5fe20048a82a28330b69028cc481e829edb9fed`
- Raw counts: `data/s1_feedback_loop/s1e_xy_observable_raw_counts_ibm_kingston_20260520T140919Z.json`
- Raw counts SHA-256: `99d97d60ce83949100ef52f6ead274776f49515990672d17fd91074b793a38ea`
- Analysis: `data/s1_feedback_loop/s1e_xy_observable_analysis_ibm_kingston_20260520T140919Z.json`
- Analysis SHA-256: `4f7e106e2af8e42aa74c9c07d66032bc76a78a936c499370c56fcef8c32ad522`

## Purpose

S1e is the confirmatory statistical repeat for the S1d policy-direction sweep.
It tests whether the large positive `YYI` response in the
`current_shallow_positive` policy reproduces at higher repetitions, or whether
it was a calibration-window fluctuation.

The repeat keeps the same one-round shallow dynamic body and the same three
policy variants as S1d, but increases repetitions from `3` to `5`.

## Readiness

All three policy variants passed the readiness gate on `ibm_kingston`.

| Variant | Rounds | Correction angle | Base gain | Max transpiled depth | Estimated QPU seconds |
|---|---:|---:|---:|---:|---:|
| `current_shallow_positive` | 1 | `0.06` | `0.4` | `237` | `40.0` |
| `polarity_flipped` | 1 | `-0.06` | `0.4` | `237` | `40.0` |
| `weak_positive` | 1 | `0.03` | `0.2` | `237` | `40.0` |

Total estimated QPU seconds: `120.0` under the `130.0` second ceiling.

## Execution

The 24 IBM jobs completed on `ibm_kingston`:

`d86s140p0eas73dlcumg`, `d86s160p0eas73dlcuq0`,
`d86s185g7okc73elhp30`, `d86s1cqs46sc73f7dam0`,
`d86s1eh789is73903dtg`, `d86s1g9789is73903dvg`,
`d86s1i2s46sc73f7dar0`, `d86s1jop0eas73dlcva0`,
`d86s1n8p0eas73dlcveg`, `d86s1p8p0eas73dlcvh0`,
`d86s1qp789is73903eg0`, `d86s1sqs46sc73f7db90`,
`d86s1ugp0eas73dlcvqg`, `d86s20dg7okc73elhq0g`,
`d86s220p0eas73dld020`, `d86s241789is73903es0`,
`d86s25gp0eas73dld07g`, `d86s278p0eas73dld09g`,
`d86s29h789is73903f3g`, `d86s2bas46sc73f7dbt0`,
`d86s2d1789is73903fc0`, `d86s2ep789is73903fe0`,
`d86s2ggp0eas73dld0og`, `d86s2i1789is73903fqg`.

The first analysis file was generated before the `s1e` alias was routed through
the grouped policy-sweep packager. The final raw-count and analysis artefacts
regroup the same completed jobs without resubmission.

## Results

Feedback minus matched open-loop control by sorted direct-XY observable row:

| Variant | IXX | IYY | XXI | YYI | Mean signed | Mean absolute |
|---|---:|---:|---:|---:|---:|---:|
| `current_shallow_positive` | `-0.0035156250` | `0.0082031250` | `-0.0488281250` | `0.1824218750` | `0.0345703125` | `0.0607421875` |
| `polarity_flipped` | `-0.0429687500` | `-0.0406250000` | `-0.0281250000` | `0.0167968750` | `-0.0237304688` | `0.0321289063` |
| `weak_positive` | `-0.0074218750` | `0.0042968750` | `-0.0105468750` | `-0.0406250000` | `-0.0135742187` | `0.0157226563` |

## Interpretation

S1e confirms that the S1d `YYI` response in `current_shallow_positive` is not a
single-run artefact at the observed scale: S1d measured `YYI = 0.1848958333`,
and S1e measured `YYI = 0.1824218750`.

The result still does not promote the full feedback law as robustly beneficial.
The favourable response is channel-specific: `current_shallow_positive` remains
negative in `IXX` and `XXI`, slightly positive in `IYY`, and strongly positive
in `YYI`. The polarity-flipped and weak-positive policies do not reproduce the
same mean-signed improvement. The safe claim is therefore a reproducible
channel-specific policy sensitivity in the direct-XY sector, not
backend-general feedback control.
