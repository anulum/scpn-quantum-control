<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- dedicated realtime latency campaign -->

# Dedicated Realtime Latency Campaign

Timestamp: `2026-05-22`

## Scope

- Dedicated realtime-control benchmark machinery independent of S1 scientific experiments.
- Rust orchestrator submits and polls IBM Runtime jobs directly.
- Scenario set:
  - `rt_adaptive_dense_dynamic`
  - `rt_adaptive_dense_open_loop`
  - `rt_adaptive_sparse_dynamic`
  - `rt_adaptive_sparse_open_loop`
  - `capacity_sweep`

## Artefacts

- Payload matrix:
  - `data/realtime_control_latency/ibm_runtime_realtime_payload_matrix_ibm_kingston_20260522T003226Z.json`
- Rust run report:
  - `data/realtime_control_latency/ibm_runtime_realtime_rust_latency_run_2026-05-22.json`

## Aggregate Runtime Result

- Jobs: `12`
- Mean submit-to-done latency: `7.144280 s`
- Population std: `0.238618 s`

## Job IDs

- `d87q88p789is73918vfg`
- `d87q8alg7okc73emm0d0`
- `d87q8c9789is73918vj0`
- `d87q8e8p0eas73dmho40`
- `d87q8fqs46sc73f8ias0`
- `d87q8hp789is73918vo0`
- `d87q8jis46sc73f8ib00`
- `d87q8l8p0eas73dmhocg`
- `d87q8n0p0eas73dmhoe0`
- `d87q8otg7okc73emm0r0`
- `d87q8qis46sc73f8ib80`
- `d87q8sas46sc73f8ibb0`

## Claim Boundary

- This campaign measures externally visible runtime windows (submit-to-done).
- It does not claim intra-shot control-electronics latency.
