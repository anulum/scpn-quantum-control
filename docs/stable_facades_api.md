# Stable Facades API

Stable facades are the first-path API surfaces for users who want to build a
workflow without depending on low-level module layout. Prefer these symbols in
tutorials, notebooks, and inter-repository contracts.

## Stable Core Contracts

The stable core contracts define the durable `Problem`, `Backend`, `Experiment`,
and `Result` surfaces that bridge domain problems, backend capability checks,
experiment preregistration, and result artifacts.

::: scpn_quantum_control.stable_core
    options:
      members:
        - Problem
        - Backend
        - Experiment
        - Result
        - backend_capability_matrix
        - stable_core_capability_payload
        - stable_core_capability_markdown
        - write_stable_core_capability_artifacts
        - build_problem
        - build_backend
        - build_experiment
        - build_result
        - classical_reference_backend
        - hardware_replay_backend
        - qiskit_backend
        - qutip_backend
        - pennylane_backend
        - pulser_surrogate_backend
        - problem_from_kuramoto
        - problem_to_kuramoto
      show_root_heading: true
      show_source: false

The stable-core contracts are also covered by reproducibility gates:

- `stable-core-capability-gate` for adapter capability profiles.
- `stable-core-contract-gate` for contract fixture drift on core contract types
  and adaptor mappings.

## Kuramoto Core

The Kuramoto core facade accepts arbitrary symmetric `K_nm` matrices,
heterogeneous `omega` vectors, and serialisable metadata. It returns validated
problem objects, sparse/dense Hamiltonians, Trotter circuits, and order-parameter
measurements.

::: scpn_quantum_control.kuramoto_core
    options:
      members:
        - KuramotoProblem
        - build_kuramoto_problem
        - validate_kuramoto_inputs
        - compile_hamiltonian
        - compile_dense_hamiltonian
        - compile_trotter_circuit
        - measure_order_parameter
      show_root_heading: true
      show_source: false

## Related First-Path Pages

- [Kuramoto Core Facade](kuramoto_core_facade.md) explains the workflow and
  validation contract.
- [Stable Core API](stable_core_api.md) documents the production-facing
  `Problem`, `Backend`, `Experiment`, and `Result` contracts plus the
  Kuramoto facade adapters and standard backend capability profiles.
- [Stable Core Backend Capability Matrix](stable_core_backend_capability_matrix.md)
  records the generated backend profile artifact.
- [Physics-First Kuramoto-XY](physics_first_kuramoto_xy.md) gives a runnable
  tutorial before SCPN-specific layers.
- [Core Package Boundary](core_package_boundary.md) records the current licence
  and possible future package split boundary.
- [API Overview](api.md) routes advanced users to lower-level module references
  after the stable facade path is clear.
