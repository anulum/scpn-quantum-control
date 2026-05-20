<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1 feedback IBM paper plan -->

# S1 Feedback IBM Paper Plan

Date: 2026-05-20

## Working Title

`Monitored feedback versus open-loop control for Kuramoto-XY synchronisation on IBM dynamic circuits`

Alternative:

`A preregistered dynamic-circuit feedback test for small Kuramoto-XY synchronisation`

## Thesis

The next IBM paper should test a control hypothesis rather than another static
characterisation hypothesis. The S1 campaign asks whether a monitored
cross-shot feedback policy moves the live Kuramoto-XY order-parameter estimate
toward a preregistered target better than a matched open-loop control under the
same backend, layout, circuit family, shots, and repetitions.

This is not a quantum-advantage paper. It is a bounded hardware-control paper:
dynamic-circuit feedback is useful only if the measured feedback arm improves
the target-error metric relative to the matched open-loop arm.

## Why This Is The Next Candidate

- It is already preregistered and has a no-QPU readiness bundle.
- It uses a small four-qubit dynamic-circuit payload.
- It has a matched open-loop control arm, so a negative result remains
  interpretable.
- It moves the programme from static leakage/tomography observations toward
  actual control.
- The estimated execution budget in the preregistration package is `24.0`
  seconds before queue and calibration uncertainty.

## Current Evidence Status

Completed:

- S1 no-QPU readiness bundle.
- S1 preregistration JSON and Markdown.
- S1 latency benchmark artefacts.
- S1 synthetic raw-count analysis rehearsal.
- Provider dry-run payloads with `submission_enabled=false`.
- Hardware-job dossier embedded in the preregistration package.

Fresh no-QPU refresh on 2026-05-20:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_s1_feedback_readiness.py
```

The refresh regenerated:

- `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.json`
- `data/s1_feedback_loop/s1_feedback_loop_latency_summary_2026-05-06.csv`
- `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.json`
- `data/s1_feedback_loop/s1_feedback_preregistration_2026-05-06.md`
- `data/s1_feedback_loop/s1_feedback_analysis_summary_2026-05-06.json`

IBM submission status:

- Paired feedback/open-loop IBM submitter is implemented behind the existing
  approval-gated scheduler.
- Live IBM sampler results are converted into the preregistered raw-count
  package with `r_live` records.
- Corrected dynamic-circuit payload uses unconditional monitor resets and
  conditional corrections only; IBM rejected reset operations inside
  conditional blocks during the first attempted feedback-arm execution.
- Corrected paired run completed on `ibm_kingston`.

Fresh live no-submit IBM readiness on 2026-05-20:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/probe_s1_ibm_metadata.py --backend ibm_kingston
PYTHONDONTWRITEBYTECODE=1 python scripts/prepare_s1_ibm_live_readiness.py --backend ibm_kingston
```

The live probe and transpilation artefacts are:

- `data/s1_feedback_loop/s1_ibm_metadata_probe_ibm_kingston_2026-05-06.json`
- `data/s1_feedback_loop/s1_ibm_live_readiness_ibm_kingston_2026-05-20.json`
- `docs/s1_ibm_live_readiness_ibm_kingston_2026-05-20.md`
- `data/s1_feedback_loop/s1_ibm_feedback_pair_readiness_ibm_kingston_20260520T123941Z.json`
- `data/s1_feedback_loop/s1_feedback_raw_counts_ibm_kingston_20260520T123941Z.json`
- `data/s1_feedback_loop/s1_feedback_analysis_summary_ibm_kingston_20260520T123941Z.json`

Observed readiness:

- backend capability status: `ready`;
- backend availability: active with zero pending jobs at capture time;
- corrected transpiled monitored payload depth: `717`;
- corrected transpiled operation counts: `cz=183`, `if_else=3`, `measure=6`,
  `reset=3`, `rz=380`, `sx=363`, `x=2`;
- hardware submission: `false`;
- corrected paired-run status: completed;
- corrected paired-run jobs: `d86qn3lg7okc73elg2eg`,
  `d86qn65g7okc73elg2hg`;
- preregistered analysis decision: `null_or_negative`;
- feedback mean target error: `0.2148958333`;
- matched open-loop mean target error: `0.2116406250`;
- target-error improvement: `-0.0032552083`.

S1b direct-XY observable extension:

- status: completed on `ibm_kingston`;
- purpose: keep this in the same paper and test direct XY-sector correlators
  after the binary synchrony proxy saturated;
- observables: `XXI`, `YYI`, `IXX`, `IYY`;
- jobs: `d86r1rqs46sc73f7c2g0`, `d86r1udg7okc73elggi0`,
  `d86r201789is739022q0`, `d86r21is46sc73f7c2o0`,
  `d86r252s46sc73f7c2tg`, `d86r26gp0eas73dlbkkg`,
  `d86r288p0eas73dlbkmg`, `d86r29p789is7390238g`;
- mean absolute feedback-control separation: `0.0205078125`;
- signed feedback minus control by sorted observable row: `0.0247395833`,
  `0.0358072917`, `-0.0175781250`, `0.0039062500`;
- interpretation: not a robust positive-control result, but channel-structured
  evidence that the binary proxy hid small direct-XY feedback/control
  differences.

## Preregistered Job Shape

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
| Hardware submission state | completed on `ibm_kingston` |

## Primary Observable

The primary observable is target-order-parameter error:

```text
abs(R_live - R_target)
```

The feedback arm is promoted only if it improves this metric relative to the
matched open-loop control under the same hardware window.

## Result Decision Tree

| Raw-count outcome | Paper framing |
|---|---|
| Feedback improves target error after the preregistered analysis | Bounded evidence that monitored dynamic-circuit feedback can steer the tested small-system synchronisation observable on the selected backend/layout. |
| Feedback and open-loop are statistically indistinguishable | Hardware-control null result; dynamic-circuit overhead/noise dominates at this scale. |
| Feedback worsens target error | **Observed.** Negative control result; the feedback action is not robust under this backend/calibration window. |
| Capability or transpilation gate fails | Provider-readiness paper note only; no hardware-control claim. |

S1b does not replace the preregistered binary-proxy decision tree. It adds a
same-paper diagnostic extension showing whether the null/negative result is
uniform across direct XY-sector observables. The observed answer is no:
feedback/control differences are small but not uniform across channels.

S1c prepared extension:

- status: `ready_for_submission`, not submitted in the preparation artefact;
- purpose: same-paper shallow/gain-tuned continuation to test whether S1/S1b
  weakness is dominated by dynamic depth and feedback strength;
- configuration: `n_rounds=1`, correction angle `0.06`, base gain `0.4`;
- observables: `XXI`, `YYI`, `IXX`, `IYY`;
- readiness artefact:
  `data/s1_feedback_loop/s1c_xy_observable_readiness_ibm_kingston_20260520T131646Z.json`;
- maximum feedback transpiled depth: `237`, compared with S1b feedback depth
  about `720`;
- estimated QPU seconds: `24.0`.

## Claim Boundary

Safe after successful raw-count analysis:

- the preregistered S1 dynamic-circuit feedback payload was executed on the
  selected IBM backend;
- the measured feedback arm did or did not improve target-error relative to a
  matched open-loop arm;
- the result is bounded to the selected backend, layout, circuit family, shots,
  repetitions, and calibration window.

Blocked even after a positive result:

- quantum advantage;
- backend-general feedback control;
- analogue-native feedback suitability;
- sub-microsecond real-time control unless provider-side timing evidence is
  captured and analysed;
- claims beyond the tested small-system payload.

## Current Manuscript Source

Draft scaffold:

- `paper/s1_feedback_control/s1_feedback_control_short_paper.md`

The draft must remain pending until live backend metadata, live transpilation,
explicit approval, raw-count artefacts, and preregistered analysis outputs
exist.
