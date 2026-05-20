# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Monitored feedback versus open-loop control for Kuramoto-XY synchronisation on IBM dynamic circuits

**Miroslav Sotek** [ORCID: 0009-0009-3560-0851]
*ANULUM Research, Marbach SG, Switzerland*
*Contact: protoscience@anulum.li*

**Date:** 2026-05-20
**Status:** IBM paired-arm execution complete; result is null-or-negative under the preregistered analysis
**Target venue:** short communication / workshop submission candidate

---

## Abstract

Dynamic-circuit support on gate-model quantum hardware makes it possible to
test small monitored-feedback policies rather than only static circuit
families. We preregister a four-qubit Kuramoto-XY feedback payload with three
system qubits, one monitor qubit, three dynamic rounds, and two matched arms:
a monitored feedback arm and an open-loop control arm. The primary endpoint is
whether the feedback arm reduces the target-order-parameter error relative to
the matched open-loop arm under the same backend, layout, shots, repetitions,
and calibration window. The S1 readiness bundle, preregistration manifest,
hardware-job dossier, live backend capture, approval-gated IBM execution, raw
count archive, and preregistered analysis are now complete for `ibm_kingston`.
The observed feedback arm does not reduce target-order-parameter error
relative to the matched open-loop control. The claim is bounded to
hardware-control interpretability, not quantum advantage or backend-general
feedback.

## 1. Introduction

The previous Kuramoto-XY hardware programme established several bounded
observations: parity-sector leakage can be measured on Heron-class hardware,
Fisher-information-inspired static feedback does not automatically protect
coherence in the tested digital implementation, and layout, prepared state, and
readout context materially affect the observed mechanism. The next control
question is whether a deliberately small monitored-feedback policy can improve
a preregistered synchronisation target relative to a matched open-loop arm.

This paper is designed as a hardware-control boundary test. A positive result
would show that the selected backend and calibration window support the tested
feedback payload better than an open-loop control. A null or negative result is
still useful: it would bound the practical value of this dynamic-circuit
feedback design at four qubits.

## 2. Preregistered Protocol

The S1 dynamic-circuit payload uses:

| Field | Value |
|---|---:|
| System qubits | 3 |
| Monitor qubits | 1 |
| Total qubits | 4 |
| Classical bits | 6 |
| Dynamic rounds | 3 |
| Arms | feedback and matched open-loop control |
| Circuits | 2 |
| Shots per circuit | 1024 |
| Repetitions | 12 |
| Estimated execution seconds | 24.0 |

The canonical preregistration package is:

```text
data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json
```

The matched open-loop control arm is mandatory. Without it, the experiment
cannot distinguish feedback improvement from backend drift, layout selection,
or ordinary shot noise.

## 3. Readiness Status

The no-QPU readiness package is regenerated with:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_s1_feedback_readiness.py
```

Current preregistered artefacts:

- `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json`
- `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.md`
- `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json`
- `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.csv`
- `data/s1_feedback_loop/s1_feedback_synthetic_raw_counts_2026-05-06.json`
- `data/s1_feedback_loop/s1_feedback_analysis_summary_2026-05-06.json`
- `docs/s1_live_submission_preflight_2026-05-06.md`
- `docs/s1_feedback_readiness_index_2026-05-06.md`

The current package declares the IBM Heron dynamic-circuit backend class as a
ready target under template capabilities. A live no-submit probe on
`ibm_kingston` reported ready dynamic-circuit capability. The monitored
payload was corrected to use unconditional monitor reset and conditional
correction only, because IBM rejected reset operations inside conditional
dynamic-circuit blocks. The corrected paired-arm payload was then submitted and
analysed.

## 4. IBM Execution

The corrected S1 paired-arm run executed on `ibm_kingston` with:

| Field | Feedback | Matched open-loop control |
|---|---:|---:|
| Job ID | `d86qn3lg7okc73elg2eg` | `d86qn65g7okc73elg2hg` |
| Repetitions | 12 | 12 |
| Total shots | 12288 | 12288 |
| Transpiled depth | 717 | 684 |

The hardware artefacts are:

- `data/s1_feedback_loop/s1_feedback_raw_counts_ibm_kingston_20260520T123941Z.json`
- `data/s1_feedback_loop/s1_feedback_analysis_summary_ibm_kingston_20260520T123941Z.json`
- `data/s1_feedback_loop/s1_ibm_feedback_pair_readiness_ibm_kingston_20260520T123941Z.json`

## 5. Analysis Plan

The approved raw-count artefact will be reduced with:

```bash
python scripts/analyse_s1_feedback_hardware.py <raw-count-package.json>
```

Primary endpoint:

```text
abs(R_live - R_target)
```

The feedback arm is interpreted as useful only if the preregistered analysis
shows lower target error than the matched open-loop control.

## 6. Results

The preregistered analysis returned `null_or_negative`.

| Metric | Feedback | Matched open-loop control |
|---|---:|---:|
| Mean `r_live` | 0.9348958333 | 0.9316406250 |
| Mean target error | 0.2148958333 | 0.2116406250 |
| Final `r_live` | 0.9322916667 | 0.9322916667 |

Feedback minus control mean `r_live` is `0.0032552083`, but the target is
`0.72`, so the feedback arm is farther from the target. The target-error
improvement is `-0.0032552083`, or `-1.538%` relative to the open-loop
control. Under the preregistered decision tree this is a bounded negative
hardware-control result: the tested monitored-feedback policy did not improve
the selected binary-phase synchrony proxy on `ibm_kingston`.

## 7. Claim Boundary

Safe claims after successful analysis:

- the S1 monitored-feedback payload was executed on the selected IBM backend;
- feedback improved, matched, or worsened the preregistered target-error metric
  relative to a matched open-loop arm;
- the result is bounded to the tested backend, layout, circuit family, shots,
  repetitions, and calibration window.

Blocked claims:

- quantum advantage;
- backend-general feedback control;
- analogue-native feedback suitability;
- full real-time-control claims without provider-side timing evidence;
- claims about larger systems or untested controller policies.

## 8. Conclusion

S1 is scientifically useful as a falsification result. It shows that the tested
small dynamic-circuit feedback policy, at this depth and calibration window,
does not outperform a matched open-loop arm on the preregistered binary-phase
synchrony target. The next S1-level paper step is either a shallower feedback
policy or a different observable that measures the intended XY order parameter
more directly.
