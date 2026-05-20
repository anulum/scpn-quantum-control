# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

# Monitored feedback versus open-loop control for Kuramoto-XY synchronisation on IBM dynamic circuits

**Miroslav Sotek** [ORCID: 0009-0009-3560-0851]
*ANULUM Research, Marbach SG, Switzerland*
*Contact: protoscience@anulum.li*

**Date:** 2026-05-20
**Status:** Draft scaffold, live IBM metadata and raw-count execution pending
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
and calibration window. The current package is no-QPU readiness only: the S1
readiness bundle, preregistration manifest, synthetic analysis rehearsal, and
hardware-job dossier are complete, while live backend metadata capture,
transpilation, explicit budget approval, raw-count execution, and analysis
remain pending. The intended claim is bounded to hardware-control
interpretability, not quantum advantage or backend-general feedback.

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
ready target under template capabilities, but a live IBM backend must still be
probed without submission before any hardware claim is possible.

## 4. Pending IBM Gates

No S1 IBM job should be submitted until all of the following are complete:

1. live IBM backend metadata captured without QPU submission;
2. capability probe against the selected backend reports `ready`;
3. live transpilation records depth, operation counts, layout, and measurement
   mapping;
4. the feedback and open-loop arms remain matched after transpilation;
5. raw-count archive path and SHA256 hashing plan exist;
6. explicit hardware approval record matches the preregistration package hash
   and QPU-second ceiling.

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

Raw-count execution is pending.

This section must not be filled from synthetic fixtures, template capability
metadata, or readiness outputs. It should be populated only after:

1. live metadata and transpilation gates pass;
2. explicit QPU approval is recorded;
3. an approved IBM run writes raw counts and job metadata;
4. `scripts/analyse_s1_feedback_hardware.py` generates the result summary;
5. the result is checked against the preregistered decision tree.

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

S1 is the next strongest IBM paper candidate after the reduced-Pauli
entanglement/tomography run because it tests a control mechanism rather than
another static diagnostic. The current paper remains a scaffold until live IBM
metadata, transpilation, approved raw counts, and preregistered analysis exist.
