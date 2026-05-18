<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (c) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- scpn-quantum-control -- stable core contract fixtures -->

# Stable Core Contract Fixtures

These no-QPU, no-network fixtures lock stable core contract payloads.

## Fixture summary

- Schema: `stable_core_contract_fixtures_v1`
- Hardware submission enabled in fixtures: `False`

## Problems

| Problem ID | Kind | Qubits | Initial state | Coupling matrix | Omega | Metadata |
|---|---|---|---|---|---|---|
| `ring4` | `kuramoto_xy` | 4 | `0011` | `[[0.0, 0.45, 0.0, 0.45], [0.45, 0.0, 0.45, 0.0], [0.0, 0.45, 0.0, 0.45], [0.45, 0.0, 0.45, 0.0]]` | `[0.8, 0.9, 1.1, 1.2]` | `{"domain": "stable-core", "source": "fixture"}` |
| `chain3` | `kuramoto_xy` | 3 | `010` | `[[0.0, 0.55, 0.0], [0.55, 0.0, 0.55], [0.0, 0.55, 0.0]]` | `[1.0, 0.95, 1.05]` | `{"domain": "stable-core", "lane": "chain", "source": "fixture"}` |

## Backends

| Backend ID | Kind | Capabilities | Hardware submission | Metadata |
|---|---|---|---|---|
| `classical-reference` | `classical_reference` | order_parameter, parity, fim, control | `False` | `{"role": "baseline"}` |
| `hardware-replay` | `hardware_replay` | order_parameter, parity, mitigation_replay | `False` | `{"role": "planner-replay"}` |
| `qiskit-runtime` | `qiskit` | order_parameter, parity, mitigation_replay | `False` | `{"role": "adapter"}` |
| `qiskit-runtime-live` | `qiskit` | order_parameter, parity, mitigation_replay | `True` | `{"mode": "fixture", "role": "hardware-path"}` |
| `qutip-dynamics` | `qutip` | order_parameter, hamiltonian_dynamics, lindblad | `False` | `{"role": "open-system"}` |
| `pennylane-autodiff` | `pennylane` | order_parameter, parity, control, autodiff | `False` | `{"role": "autodiff"}` |
| `pulser-surrogate` | `pulser_surrogate` | order_parameter, analog_surrogate, pulse_schedule | `False` | `{"role": "analog"}` |

## Experiments

| Experiment ID | Problem ID | Backend ID | Objective | Seed | Shots | Metadata |
|---|---|---|---|---|---|---|
| `exp-ring4-order-classical` | `ring4` | `classical-reference` | `order_parameter` | `17` | `1024` | `{}` |
| `exp-ring4-parity-replay` | `ring4` | `hardware-replay` | `parity_leakage` | `23` | `512` | `{}` |
| `exp-chain3-mitigation-qiskit` | `chain3` | `qiskit-runtime` | `mitigation_replay` | `31` | `None` | `{}` |
| `exp-chain3-fim-qutip` | `chain3` | `qutip-dynamics-fallback` | `order_parameter` | `41` | `256` | `{"fixture": "fim"}` |
| `exp-ring4-control-live` | `ring4` | `classical-control` | `control_cost` | `7` | `None` | `{"control_profile": "l2"}` |
| `exp-ring4-order-qiskit-live` | `ring4` | `qiskit-runtime-live` | `order_parameter` | `53` | `None` | `{"preregistration_id": "fixture-001"}` |

## Results

| Experiment ID | Backend ID | Status | Observables | Blockers | Artifacts |
|---|---|---|---|---|---|
| `exp-ring4-order-classical` | `classical-reference` | `succeeded` | `{"order_parameter": 0.742}` | `none` | `artifacts/exp-ring4-order-classical.json` |
| `exp-ring4-parity-replay` | `hardware-replay` | `succeeded` | `{"parity_leakage": 0.91}` | `none` | ``none`` |
| `exp-chain3-mitigation-qiskit` | `qiskit-runtime` | `blocked` | `{}` | mitigation replay requires calibrated parity observables | ``none`` |
| `exp-chain3-fim-qutip` | `qutip-dynamics-fallback` | `failed` | `{}` | offline fixture path forbids hardware execution | ``none`` |
| `exp-ring4-control-live` | `classical-control` | `succeeded` | `{"control_cost": 0.27}` | `none` | `artifacts/exp-ring4-control-live.json` |
| `exp-ring4-order-qiskit-live` | `qiskit-runtime-live` | `blocked` | `{}` | live qiskit path intentionally disabled in stable-core fixtures | ``none`` |

## Reproducibility gate

Regenerate and compare these fixtures with:

```bash
python scripts/run_stable_core_contract_gate.py
```

## Claim boundary

Stable-core contract fixtures are deterministic shape checks only. They do not execute on hardware or invoke external providers.
