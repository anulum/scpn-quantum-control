<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- realtime control E2E benchmark -->

# Realtime Control E2E Benchmark

Date: `2026-05-22`

Command: `PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_realtime_control_e2e.py --repeats 12 --steps 8`

## Claim Boundary

No-QPU software control-loop benchmark only. Results exclude provider queue, network transport, runtime session setup, dynamic-circuit hardware latency, readout latency, and QPU execution latency.

## Environment

```json
{
  "platform": "Linux-6.17.0-23-generic-x86_64-with-glibc2.39",
  "processor": "x86_64",
  "python": "3.14.3",
  "rust_feedback_policy_available": true
}
```

## Rows

| n | repeats_successful | p95 tick ms | p99 tick ms | max tick ms | rust policy path |
|---:|---:|---:|---:|---:|:---:|
| 2 | 10 | 5.648401 | 6.141578 | 6.715363 | true |
| 3 | 12 | 6.268513 | 6.965508 | 8.554773 | true |
| 4 | 10 | 6.748360 | 9.103593 | 10.990356 | true |
| 2 | 12 | 0.000429 | 0.000901 | 0.004548 | true |
| 3 | 12 | 0.000493 | 0.000835 | 0.002398 | true |
| 4 | 12 | 0.000404 | 0.001697 | 0.002690 | true |

## Reproducibility

This artefact is deterministic at the configuration level (fixed seeds and fixed rows),
but wall-time values depend on host load, CPU governor, and thermal state. Re-run the
command on an isolated benchmark host before publication-grade speed claims.
