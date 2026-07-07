# S1f IBM Quadrature Mechanism Check

Date: 2026-05-20
Backend: `ibm_kingston`
Experiment ID: `s1f_quadrature_mechanism_check_2026-05-20`
Parent experiment: `s1_dynamic_feedback_preregistration_2026-05-06`

## Artefacts

- Readiness: `data/s1_feedback_loop/s1f_xy_observable_readiness_ibm_kingston_20260520T142213Z.json`
- Readiness SHA-256: `d8b5e0980ab3966ec62eb29f25192cd387b4f2bc420126ea0f8f9d9fbc45d3ea`
- Raw counts: `data/s1_feedback_loop/s1f_xy_observable_raw_counts_ibm_kingston_20260520T142213Z.json`
- Raw counts SHA-256: `1be95f9996fb56e1f7e8d39921e9c46c65b1d8f4622bb46ac7d3789a592d9e87`
- Analysis: `data/s1_feedback_loop/s1f_xy_observable_analysis_ibm_kingston_20260520T142213Z.json`
- Analysis SHA-256: `17b37753460137c796a5fb8c52784ce694d7ce4ffb63e55c92853abe8f91313e`

## Purpose

S1f is a narrow mechanism-discrimination lane for the S1 paper. It keeps only
the S1d/S1e-favoured policy, `current_shallow_positive`, and tests whether the
previously repeated `YYI` response localises to one quadrature or instead
appears across transverse or population controls.

Configuration:

- `n_rounds=1`
- correction angle `0.06`
- base gain `0.4`
- repetitions `5`
- observables: `YYI`, `XXI`, `XYI`, `YXI`, `ZZI`

## Readiness

All arms passed readiness on `ibm_kingston`. Maximum transpiled depth was
`237`. Total estimated QPU seconds were `50.0` under the `60.0` second ceiling.

## Execution

The 10 IBM jobs completed:

`d86s75h789is73903oug`, `d86s779789is73903p00`,
`d86s791789is73903p60`, `d86s7atg7okc73eli47g`,
`d86s7d2s46sc73f7dko0`, `d86s7eop0eas73dldb7g`,
`d86s7gop0eas73dldbeg`, `d86s7m8p0eas73dldbp0`,
`d86s7odg7okc73eli530`, `d86s7q8p0eas73dldbtg`.

## Results

Feedback minus matched open-loop control:

| Observable | Feedback | Control | Feedback minus control |
|---|---:|---:|---:|
| `XXI` | `0.2187500000` | `0.2175781250` | `0.0011718750` |
| `XYI` | `0.1617187500` | `0.1566406250` | `0.0050781250` |
| `YXI` | `0.2242187500` | `0.2136718750` | `0.0105468750` |
| `YYI` | `-0.0445312500` | `-0.0242187500` | `-0.0203125000` |
| `ZZI` | `0.6523437500` | `0.6480468750` | `0.0042968750` |

Mean absolute feedback-control separation: `0.0082812500`.

## Interpretation

S1f does not support a stable quadrature-localised positive `YYI` mechanism in
this later calibration window. The previously repeated S1d/S1e `YYI` response
was large and positive, but S1f measures a small negative `YYI` delta while the
cross-quadrature (`XYI`, `YXI`) and population (`ZZI`) controls remain small.

The conservative conclusion is stronger and more honest than a promoted
controller claim: the S1 direct-XY response is structured and can repeat within
a calibration window, but it is not calibration-window stable enough to promote
the tested feedback law as a robust controller. S1f should be the final
hardware run for this paper unless a future redesigned policy is preregistered.
