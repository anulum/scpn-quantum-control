<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1b IBM direct-XY observable result -->

# S1b IBM Direct-XY Observable Result

Date: 2026-05-20

Backend: `ibm_kingston`

Parent S1 result: `docs/s1_ibm_feedback_pair_result_2026-05-20.md`

## Purpose

S1b is an extension of the same S1 paper, not a separate paper. The first S1
hardware run showed that the binary synchrony proxy was near saturation in
both arms. S1b keeps the same dynamic-feedback and matched open-loop bodies but
measures direct XY-sector Pauli correlators at final readout.

## Jobs

| Observable | Feedback job | Matched open-loop job |
|---|---|---|
| `XXI` | `d86r1rqs46sc73f7c2g0` | `d86r252s46sc73f7c2tg` |
| `YYI` | `d86r1udg7okc73elggi0` | `d86r26gp0eas73dlbkkg` |
| `IXX` | `d86r201789is739022q0` | `d86r288p0eas73dlbkmg` |
| `IYY` | `d86r21is46sc73f7c2o0` | `d86r29p789is7390238g` |

Artefacts:

- `data/s1_feedback_loop/s1b_xy_observable_readiness_ibm_kingston_20260520T130238Z.json`
- `data/s1_feedback_loop/s1b_xy_observable_raw_counts_ibm_kingston_20260520T130238Z.json`
- `data/s1_feedback_loop/s1b_xy_observable_analysis_ibm_kingston_20260520T130238Z.json`

## Result

| Observable | Feedback mean | Matched open-loop mean | Feedback minus control |
|---|---:|---:|---:|
| `IXX` | 0.9042968750 | 0.8795572917 | 0.0247395833 |
| `IYY` | 0.8977864583 | 0.8619791667 | 0.0358072917 |
| `XXI` | 0.8424479167 | 0.8600260417 | -0.0175781250 |
| `YYI` | 0.8632812500 | 0.8593750000 | 0.0039062500 |

Mean absolute feedback-control separation across the four direct XY channels:
`0.0205078125`.

## Interpretation

S1b removes the saturated binary-synchrony proxy and shows that the
feedback/control difference is small but channel-structured. Three of four
direct XY channels move positive for the feedback arm, while `XXI` moves
negative. This supports a conservative paper claim: the tested feedback policy
does not deliver robust target control, but direct XY-sector measurements
reveal a non-uniform feedback/control response hidden by the binary proxy.

## Boundary

This does not establish backend-general feedback control, quantum advantage,
or sub-microsecond real-time control. It is a bounded dynamic-circuit
hardware-control stress test on the selected backend, circuit family, shots,
repetitions, and calibration window.
