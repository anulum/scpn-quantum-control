<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1 IBM feedback pair result -->

# S1 IBM Feedback Pair Result

Date: 2026-05-20

Backend: `ibm_kingston`

Jobs:

- feedback: `d86qn3lg7okc73elg2eg`
- matched open-loop control: `d86qn65g7okc73elg2hg`

Artefacts:

- `data/s1_feedback_loop/s1_ibm_feedback_pair_readiness_ibm_kingston_20260520T123941Z.json`
- `data/s1_feedback_loop/s1_feedback_raw_counts_ibm_kingston_20260520T123941Z.json`
- `data/s1_feedback_loop/s1_feedback_analysis_summary_ibm_kingston_20260520T123941Z.json`

## Result

| Metric | Feedback | Matched open-loop control |
|---|---:|---:|
| Repetitions | 12 | 12 |
| Total shots | 12288 | 12288 |
| Mean `r_live` | 0.9348958333 | 0.9316406250 |
| Mean target error | 0.2148958333 | 0.2116406250 |
| Final `r_live` | 0.9322916667 | 0.9322916667 |

The preregistered analysis returns `null_or_negative`.

The feedback arm is farther from the target by `0.0032552083`, a relative
target-error change of `-1.538%` compared with the matched open-loop control.

## Boundary

This is a bounded hardware-control falsification for the tested S1 policy,
backend, circuit family, shots, repetitions, and calibration window. It does
not establish sub-microsecond feedback, quantum advantage, or backend-general
feedback behaviour.
