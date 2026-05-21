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
| 2 | 11 | 9.474368 | 14.296871 | 17.103531 | true |
| 3 | 12 | 10.574582 | 12.422564 | 13.376617 | true |
| 4 | 10 | 14.378203 | 15.257738 | 16.461197 | true |
| 2 | 12 | 0.000201 | 0.000426 | 0.001618 | true |
| 3 | 12 | 0.000207 | 0.000338 | 0.001695 | true |
| 4 | 12 | 0.000268 | 0.000393 | 0.001377 | true |

## Reproducibility

This artefact is deterministic at the configuration level (fixed seeds and fixed rows),
but wall-time values depend on host load, CPU governor, and thermal state. Re-run the
command on an isolated benchmark host before publication-grade speed claims.
