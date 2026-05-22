<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- realtime control E2E benchmark -->

# Realtime Control E2E Benchmark

Date: `2026-05-22`

Command: `PYTHONDONTWRITEBYTECODE=1 python scripts/benchmark_realtime_control_e2e.py --repeats 15 --steps 64`

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
| 2 | 15 | 4.267569 | 5.157374 | 18.401380 | true |
| 3 | 15 | 4.994387 | 6.156117 | 9.210675 | true |
| 4 | 12 | 6.919108 | 8.895908 | 13.749520 | true |
| 2 | 15 | 0.000128 | 0.000174 | 0.001220 | true |
| 3 | 15 | 0.000152 | 0.000184 | 0.000530 | true |
| 4 | 15 | 0.000190 | 0.000230 | 0.000977 | true |

## Reproducibility

This artefact is deterministic at the configuration level (fixed seeds and fixed rows),
but wall-time values depend on host load, CPU governor, and thermal state. Re-run the
command on an isolated benchmark host before publication-grade speed claims.
