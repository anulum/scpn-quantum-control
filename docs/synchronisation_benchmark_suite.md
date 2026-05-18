<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- synchronisation benchmark documentation -->

# Standardised Synchronisation Benchmark Suite

This suite defines canonical coupled-oscillator benchmark instances and a stable result schema. It is a contract for future backends, not a new hardware claim.

## Result schema

Required result fields:

- `benchmark_id`
- `backend`
- `backend_version`
- `command`
- `commit`
- `dependency_lock`
- `hardware_submission`
- `wall_time_s`
- `observables`
- `claim_boundary`

Observable row fields:

- `name`
- `value`
- `uncertainty`
- `units`
- `tolerance`
- `passed`

## Canonical instances

| Benchmark | Family | N | Evidence class | Required backends | Claim boundary |
|---|---|---:|---|---|---|
| `kuramoto_ring_n4_linear_omega` | kuramoto_xy_reference | 4 | analytic_reference | classical_ode, exact_diagonalisation | Reference smoke instance for compiler and observable consistency; not a hardware-performance or advantage claim. |
| `kuramoto_chain_n8_decay_omega` | kuramoto_xy_reference | 8 | simulator_reference | classical_ode, exact_diagonalisation | Simulator/reference benchmark for scaling behaviour; not a broad quantum-advantage claim. |
| `phase1_dla_parity_n4_ibm_kingston` | hardware_replay | 4 | hardware_replay | hardware_replay, classical_parity_reference | Replays committed raw counts only; does not submit QPU jobs or promote broader scaling, mitigation, or advantage claims. |
| `bkt_finite_size_grid_planned` | phase_transition_reference | 16 | planned | classical_xy, exact_or_tensor_reference | Visible roadmap row only; not an available benchmark result. |

## Hardware rule

hardware_submission must be false unless a preregistered manifest, QPU budget, raw-count target path, and explicit approval are recorded

## Claim boundary

The registry defines benchmark contracts and replay rows. It does not create new hardware evidence or quantum-advantage claims.
