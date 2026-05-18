<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- stable core preflight fixtures -->

# Stable Core Preflight Fixtures

These no-QPU, no-network fixtures lock stable core preflight branches.

## Fixture summary

- Schema: `stable_core_preflight_fixtures_v1`
- Hardware submission enabled in fixtures: `False`

## Preflight fixtures

| Fixture ID | Status | Backend ID | Objective | Blockers | Primitives |
|---|---|---|---|---|---|
| `eligible_classical_reference` | `eligible` | `classical-reference` | `order_parameter` | `none` | dependency_probe, capability_guard, preregistration_guard, eligible |
| `blocked_missing_dependency` | `blocked` | `qiskit-runtime` | `order_parameter` | missing dependency: qiskit-runtime provider package | `none` |
| `blocked_hardware_preregistration_or_boundary` | `blocked` | `qiskit-runtime-live` | `order_parameter` | hardware preregistration required for live submission, hardware boundary blocks run-path in stable fixture mode | `none` |
| `blocked_missing_capability` | `blocked` | `qutip-dynamics` | `control_cost` | backend qutip-dynamics does not declare control capability | `none` |

## Reproducibility gate

Regenerate and compare these fixtures with:

```bash
scpn-bench stable-core-preflight-gate
```

## Claim boundary

Preflight fixtures are offline and do not prove runtime execution, hardware registration, or external dependency readiness. They only lock shape checks for deterministic guard branches.
