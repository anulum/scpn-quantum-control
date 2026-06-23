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
**Status:** IBM paired-arm execution complete; S1/S1b/S1c/S1d/S1e same-paper
extensions are complete
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
The observed feedback arm does not reduce binary-proxy target-order-parameter
error relative to the matched open-loop control. Because that proxy is near
saturation in both arms, we add same-paper direct-XY extensions: S1b measures
final XY-sector Pauli correlators on the original body, S1c reduces depth and
gain, S1d sweeps correction direction and gain, S1e repeats the policy sweep
at higher repetitions, and S1f checks quadrature and population controls. The
direct-XY extensions show channel-structured and policy-sensitive
feedback/control differences that are not stable enough to promote the tested
controller. The claim is bounded to hardware-control interpretability, not
quantum advantage or backend-general feedback.

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
- `docs/campaigns/s1_live_submission_preflight_2026-05-06.md`
- `docs/campaigns/s1_feedback_readiness_index_2026-05-06.md`

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

### 6.1 Binary synchrony proxy

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

### 6.2 S1b direct XY-sector extension

The S1b extension keeps the same dynamic feedback and matched open-loop bodies
but rotates the final readout basis to direct XY-sector Pauli correlators. The
goal is to avoid relying only on the saturated binary synchrony proxy.

| Observable | Feedback mean | Matched open-loop mean | Feedback minus control |
|---|---:|---:|---:|
| `IXX` | 0.9042968750 | 0.8795572917 | 0.0247395833 |
| `IYY` | 0.8977864583 | 0.8619791667 | 0.0358072917 |
| `XXI` | 0.8424479167 | 0.8600260417 | -0.0175781250 |
| `YYI` | 0.8632812500 | 0.8593750000 | 0.0039062500 |

Mean absolute feedback-control separation across the four direct XY channels
is `0.0205078125`. Three channels move positive for the feedback arm and one
channel (`XXI`) moves negative. This is not a robust positive-control result,
but it is more informative than the binary proxy: the feedback/control
difference is small and channel-structured rather than uniformly erased.

S1b jobs:

- `XXI`: feedback `d86r1rqs46sc73f7c2g0`, control `d86r252s46sc73f7c2tg`
- `YYI`: feedback `d86r1udg7okc73elggi0`, control `d86r26gp0eas73dlbkkg`
- `IXX`: feedback `d86r201789is739022q0`, control `d86r288p0eas73dlbkmg`
- `IYY`: feedback `d86r21is46sc73f7c2o0`, control `d86r29p789is7390238g`

### 6.3 S1c shallow/gain-tuned extension

S1c is a same-paper follow-up lane, not a separate paper. It keeps the direct
XY-sector observables but reduces the dynamic feedback body to one round,
correction angle `0.06`, and base gain `0.4`.

The S1c feedback arms transpile on `ibm_kingston` with maximum depths
`236`-`237`, compared with S1b feedback depths near `720`. This is the next
logical hardware execution if the paper needs to separate policy/depth
overhead from the underlying dynamic-feedback idea.

S1c completed on `ibm_kingston` with jobs:

- `XXI`: feedback `d86rca1789is73902fc0`, control `d86rcgh789is73902fm0`
- `YYI`: feedback `d86rcc2s46sc73f7cf50`, control `d86rcias46sc73f7cfh0`
- `IXX`: feedback `d86rcdgp0eas73dlc2j0`, control `d86rcjp789is73902fsg`
- `IYY`: feedback `d86rcf0p0eas73dlc2lg`, control `d86rclgp0eas73dlc2sg`

| Observable | Feedback minus control |
|---|---:|
| `IXX` | -0.0364583333 |
| `IYY` | -0.0319010417 |
| `XXI` | -0.0221354167 |
| `YYI` | -0.0292968750 |

Mean absolute feedback-control separation is `0.0299479167`. Unlike S1b,
which showed mixed signs, S1c moves negative in all four direct XY channels.
The shallower/lower-gain policy therefore does not rescue the current
feedback design.

### 6.4 S1d policy-direction sweep

S1d is the final same-paper discriminator for the S1 feedback law. It keeps
the S1c one-round shallow body and direct XY observable family, but compares
three preregistered policy variants: the current shallow positive-correction
policy, a correction-polarity flip, and a weaker positive policy.

The S1d readiness gate passed on `ibm_kingston` with maximum transpiled depth
`237` and total estimated QPU budget `72.0` seconds under the `120.0` second
ceiling. The first approved submission produced 14 completed jobs before local
budget accounting stopped the batch because provider wall-clock wait time was
reported as QPU spend. That accounting bug was fixed; the completed jobs were
recovered from IBM job lookup and the 10 missing arms were submitted without
duplicating completed arms.

| Variant | `IXX` | `IYY` | `XXI` | `YYI` | Mean signed | Mean absolute |
|---|---:|---:|---:|---:|---:|---:|
| `current_shallow_positive` | -0.0078125 | -0.0136718750 | -0.0605468750 | 0.1848958333 | 0.0257161458 | 0.0667317708 |
| `polarity_flipped` | -0.0130208333 | -0.0065104167 | 0.0156250000 | -0.0123697917 | -0.0040690104 | 0.0118815104 |
| `weak_positive` | -0.0026041667 | -0.0039062500 | 0.0065104167 | -0.0201822917 | -0.0050455729 | 0.0083007813 |

S1d does not turn S1 into a positive controller paper. The repeat of the
S1c-like policy is favourable only by mean signed delta and is driven by one
large `YYI` channel, while the polarity-flipped and weak-positive policies are
close to zero. The conservative interpretation is policy- and
calibration-window sensitivity in the direct-XY sector, not robust feedback
success.

### 6.5 S1e confirmatory policy-sweep repeat

S1e repeats the S1d policy-direction sweep at five repetitions per arm. Its
purpose is to test whether the large positive `YYI` channel in
`current_shallow_positive` reproduces, or whether it was a single calibration
window fluctuation.

The S1e readiness gate passed on `ibm_kingston` with maximum transpiled depth
`237` and total estimated QPU budget `120.0` seconds under the `130.0` second
ceiling. The 24 IBM jobs completed successfully.

| Variant | `IXX` | `IYY` | `XXI` | `YYI` | Mean signed | Mean absolute |
|---|---:|---:|---:|---:|---:|---:|
| `current_shallow_positive` | -0.0035156250 | 0.0082031250 | -0.0488281250 | 0.1824218750 | 0.0345703125 | 0.0607421875 |
| `polarity_flipped` | -0.0429687500 | -0.0406250000 | -0.0281250000 | 0.0167968750 | -0.0237304688 | 0.0321289063 |
| `weak_positive` | -0.0074218750 | 0.0042968750 | -0.0105468750 | -0.0406250000 | -0.0135742187 | 0.0157226563 |

S1e confirms the S1d `YYI` response in `current_shallow_positive`: S1d measured
`0.1848958333`, and S1e measured `0.1824218750`. The result remains
channel-specific rather than a full-controller promotion, since `IXX` and
`XXI` stay negative and the alternative policies do not reproduce the same
mean-signed behaviour.

### 6.6 S1f quadrature mechanism check

S1f keeps only the S1d/S1e-favoured `current_shallow_positive` policy and tests
whether the repeated `YYI` response localises to a stable quadrature structure.
It measures `YYI`, `XXI`, cross-quadrature controls `XYI` and `YXI`, and the
population control `ZZI`, all at five repetitions.

| Observable | Feedback | Control | Feedback minus control |
|---|---:|---:|---:|
| `XXI` | 0.2187500000 | 0.2175781250 | 0.0011718750 |
| `XYI` | 0.1617187500 | 0.1566406250 | 0.0050781250 |
| `YXI` | 0.2242187500 | 0.2136718750 | 0.0105468750 |
| `YYI` | -0.0445312500 | -0.0242187500 | -0.0203125000 |
| `ZZI` | 0.6523437500 | 0.6480468750 | 0.0042968750 |

S1f does not reproduce the large positive `YYI` shift in the later mechanism
window. The mean absolute feedback-control separation is only `0.0082812500`;
`YYI` is negative, while `XXI`, `XYI`, `YXI`, and `ZZI` remain small. This
blocks a stable quadrature-localised positive-mechanism claim and supports a
calibration-window boundary.

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

S1 is scientifically useful as a falsification and boundary result. The tested
small dynamic-circuit feedback policy, at this depth and calibration window,
does not outperform a matched open-loop arm on the preregistered binary-phase
synchrony target. The S1b extension shows why that is not the whole story:
direct XY-sector channels expose a small, non-uniform feedback/control
response. The S1c shallow/lower-gain extension then moves negative across all
four direct XY channels. S1d then shows that a same-day policy-direction sweep
is sensitive to policy choice, with one large favourable `YYI` channel in the
current shallow policy. S1e repeats that `YYI` response at higher repetitions,
but S1f does not reproduce it in a later quadrature mechanism check. The paper
should therefore be framed as a dynamic-circuit hardware-control boundary test:
the specific controller is not promoted as broadly successful, but the
paired-arm method and direct-observable extensions expose structured,
calibration-window-sensitive failure and response modes that future redesigned
policies must explicitly preregister against.
