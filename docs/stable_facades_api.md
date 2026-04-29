# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Stable Facades API

# Stable Facades API

Stable facades are the first-path API surfaces for users who want to build a
workflow without depending on low-level module layout. Prefer these symbols in
tutorials, notebooks, and inter-repository contracts.

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
- [Physics-First Kuramoto-XY](physics_first_kuramoto_xy.md) gives a runnable
  tutorial before SCPN-specific layers.
- [Core Package Boundary](core_package_boundary.md) records the current licence
  and possible future package split boundary.
