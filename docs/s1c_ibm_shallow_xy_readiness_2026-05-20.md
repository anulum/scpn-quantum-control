<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1c IBM shallow direct-XY readiness -->

# S1c IBM Shallow Direct-XY Result

Date: 2026-05-20

Backend: `ibm_kingston`

Status: completed

Artefacts:

- `data/s1_feedback_loop/s1c_xy_observable_readiness_ibm_kingston_20260520T131646Z.json`
- `data/s1_feedback_loop/s1c_xy_observable_readiness_ibm_kingston_20260520T132455Z.json`
- `data/s1_feedback_loop/s1c_xy_observable_raw_counts_ibm_kingston_20260520T132455Z.json`
- `data/s1_feedback_loop/s1c_xy_observable_analysis_ibm_kingston_20260520T132455Z.json`

## Purpose

S1c is a continuation inside the same S1 paper. It is not a separate paper. It
tests whether the S1/S1b weakness is partly caused by dynamic-circuit depth and
feedback-policy strength.

## Configuration

| Field | Value |
|---|---:|
| Dynamic rounds | 1 |
| Correction angle | 0.06 |
| Base gain | 0.4 |
| Observables | `XXI`, `YYI`, `IXX`, `IYY` |
| Shots per circuit | 1024 |
| Repetitions per arm/observable | 3 |
| Estimated QPU seconds | 24.0 |
| Max QPU seconds | 120.0 |

## Readiness Summary

The S1c feedback arms transpile with maximum depths `236`-`237`, compared with
the S1b three-round direct-XY feedback depths `719`-`720`. The matched
open-loop arms transpile with maximum depth `233`. This makes S1c the clean
next hardware run if the paper needs to separate policy/depth overhead from
the underlying dynamic-feedback idea.

## Submission Command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-linux/bin/python scripts/submit_s1b_ibm_xy_observable_pair.py \
  --backend ibm_kingston \
  --lane s1c \
  --experiment-id s1c_shallow_gain_tuned_xy_extension_2026-05-20 \
  --n-rounds 1 \
  --correction-angle 0.06 \
  --base-gain 0.4 \
  --repetitions 3 \
  --max-depth 900 \
  --timeout-s 7200 \
  --submit \
  --confirm-budget
```

## Boundary

S1c is a same-paper extension. It was submitted after the readiness artefact
was prepared. The original live runner completed the IBM jobs, then failed
locally during packaging because the generic count extractor selected the
monitor register instead of the final `readout` register for one-round
circuits. The jobs were recovered from IBM without resubmission after fixing
the extractor to prefer the final `readout` register.

## Jobs

| Observable | Feedback job | Matched open-loop job |
|---|---|---|
| `XXI` | `d86rca1789is73902fc0` | `d86rcgh789is73902fm0` |
| `YYI` | `d86rcc2s46sc73f7cf50` | `d86rcias46sc73f7cfh0` |
| `IXX` | `d86rcdgp0eas73dlc2j0` | `d86rcjp789is73902fsg` |
| `IYY` | `d86rcf0p0eas73dlc2lg` | `d86rclgp0eas73dlc2sg` |

## Result

| Observable | Feedback minus control |
|---|---:|
| `IXX` | -0.0364583333 |
| `IYY` | -0.0319010417 |
| `XXI` | -0.0221354167 |
| `YYI` | -0.0292968750 |

Mean absolute feedback-control separation: `0.0299479167`.

S1c does not rescue the feedback policy. Under the shallow one-round,
lower-gain configuration, all four direct XY channels move negative for the
feedback arm relative to the matched open-loop arm. This strengthens the same
paper's conservative conclusion: the current monitored-feedback policy is not
a robust controller on this backend/window.
