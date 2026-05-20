<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- S1c IBM shallow direct-XY readiness -->

# S1c IBM Shallow Direct-XY Readiness

Date: 2026-05-20

Backend: `ibm_kingston`

Status: `ready_for_submission`

Artefact:

- `data/s1_feedback_loop/s1c_xy_observable_readiness_ibm_kingston_20260520T131646Z.json`

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

S1c is a prepared same-paper extension. It has not been submitted in this
artefact. A successful execution would test whether shallower dynamic feedback
changes the small, channel-structured direct-XY response observed in S1b.
