<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Quantum Control — Gradient-Plan Explanations
-->

# Gradient-Plan Explanations

- Schema: `scpn_qc_gradient_plan_explanations_v1`
- Artifact ID: `gradient-plan-explanations-20260709`
- Plans: `5` supported, `5` fail-closed
- Method families: `parameter-shift, unsupported`
- Claim boundary: gradient support matrix audit only; supported entries identify bounded local or host-bridge gradient surfaces, blocked entries are fail-closed planning evidence, and no live hardware-gradient or universal transform claim is implied

| Cell | Framework | Backend | Transform | Selected method | Status | Why | Boundaries |
|---|---|---|---|---|---|---|---|
| `ry::pauli_expectation::statevector::grad::native` | `native` | `statevector` | `grad` | `parameter_shift` | `supported` | gate:ry supports parameter_shift, multi_frequency_parameter_shift; observable:pauli_expectation supports parameter_shift, stochastic_parameter_shift; backend:statevector_simulator supports parameter_shift; transform:grad supports parameter_shift, stochastic_parameter_shift; adapter:native supports parameter_shift, stochastic_parameter_shift, parameter_shift_hessian, parameter_shift_directional_derivative, parameter_shift_pullback, parameter_shift_scalar_jacobian, manual_vmap_parameter_shift_grad; backend planner selected parameter_shift with 4 evaluations | - |
| `rz::kuramoto_xy_energy::qasm_simulator::grad::native` | `native` | `qasm_simulator` | `grad` | `stochastic_parameter_shift` | `supported` | gate:rz supports parameter_shift, multi_frequency_parameter_shift; observable:kuramoto_xy_energy supports parameter_shift, term_gradient; backend:finite_shot_simulator supports stochastic_parameter_shift; transform:grad supports parameter_shift, stochastic_parameter_shift; adapter:native supports parameter_shift, stochastic_parameter_shift, parameter_shift_hessian, parameter_shift_directional_derivative, parameter_shift_pullback, parameter_shift_scalar_jacobian, manual_vmap_parameter_shift_grad; backend planner selected stochastic_parameter_shift with 4 evaluations | - |
| `ry::pauli_expectation::statevector::value_and_grad::jax` | `jax` | `statevector` | `value_and_grad` | `jax_host_callback_parameter_shift` | `supported` | gate:ry supports parameter_shift, multi_frequency_parameter_shift; observable:pauli_expectation supports parameter_shift, stochastic_parameter_shift; backend:statevector_simulator supports parameter_shift; transform:value_and_grad supports parameter_shift, stochastic_parameter_shift; adapter:jax supports host_callback_parameter_shift; backend planner selected parameter_shift with 4 evaluations | - |
| `rx::sparse_pauli_sum::statevector::grad::qiskit` | `qiskit` | `statevector` | `grad` | `qiskit_shifted_circuit_parameter_shift` | `supported` | gate:rx supports parameter_shift, multi_frequency_parameter_shift; observable:sparse_pauli_sum supports parameter_shift, stochastic_parameter_shift; backend:statevector_simulator supports parameter_shift; transform:grad supports parameter_shift, stochastic_parameter_shift; adapter:qiskit supports shifted_circuit_generation, statevector_parameter_shift; backend planner selected parameter_shift with 2 evaluations | - |
| `ry::pauli_expectation::qasm_simulator::value_and_grad::provider_callback` | `provider_callback` | `qasm_simulator` | `value_and_grad` | `provider_callback_stochastic_parameter_shift` | `supported` | gate:ry supports parameter_shift, multi_frequency_parameter_shift; observable:pauli_expectation supports parameter_shift, stochastic_parameter_shift; backend:finite_shot_simulator supports stochastic_parameter_shift; transform:value_and_grad supports parameter_shift, stochastic_parameter_shift; adapter:provider_callback supports provider_callback_parameter_shift, provider_callback_stochastic_parameter_shift; backend planner selected stochastic_parameter_shift with 4 evaluations | - |
| `arbitrary_unitary::pauli_expectation::statevector::grad::native` | `native` | `statevector` | `grad` | `unsupported` | `fail_closed` | gate has no registered parameter-shift generator spectrum | gate has no registered parameter-shift generator spectrum |
| `ry::arbitrary_povm::statevector::grad::native` | `native` | `statevector` | `grad` | `unsupported` | `fail_closed` | observable has no registered expectation-gradient contract | observable has no registered expectation-gradient contract |
| `ry::pauli_expectation::hardware::grad::native` | `native` | `hardware` | `grad` | `unsupported` | `fail_closed` | hardware gradient execution requires explicit hardware policy approval | hardware gradient execution requires explicit hardware policy approval |
| `ry::pauli_expectation::statevector::vmap::jax` | `jax` | `statevector` | `vmap` | `unsupported` | `fail_closed` | transform is outside the bounded quantum-gradient algebra; jax bridge supports first-order value/gradient calls only | transform is outside the bounded quantum-gradient algebra; jax bridge supports first-order value/gradient calls only |
| `ry::pauli_expectation::qasm_simulator::hessian::native` | `native` | `qasm_simulator` | `hessian` | `unsupported` | `fail_closed` | hessian support is limited to deterministic local backends | hessian support is limited to deterministic local backends |

Supported rows explain bounded planner choices. Blocked rows are
fail-closed planning boundaries and do not permit derivative execution.
