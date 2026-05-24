# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Kuramoto Core Facade

# Kuramoto Core Facade

The `scpn_quantum_control.kuramoto_core` facade is the stable entry point for
users who only need the Kuramoto-XY compiler layer:

1. provide an arbitrary symmetric coupling matrix `K_nm`;
2. provide heterogeneous natural frequencies `omega`;
3. compile a Hamiltonian or Trotter circuit;
4. evaluate the Kuramoto order parameter from a statevector;
5. run higher-order, monitored, or PT-symmetric trajectory variants;
6. attach serialisable provenance metadata to downstream result artifacts.

The facade deliberately avoids SCPN-specific constants. Domain pipelines such as
power-grid, EEG, plasma, or connectome workflows should compile their source data
into `K_nm` and `omega`, then call this layer.

## Minimal Example

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from scpn_quantum_control import (
    build_kuramoto_problem,
    compile_hamiltonian,
    compile_trotter_circuit,
    measure_order_parameter,
)

K_nm = np.array(
    [
        [0.0, 0.4, 0.1],
        [0.4, 0.0, 0.2],
        [0.1, 0.2, 0.0],
    ]
)
omega = np.array([0.3, -0.1, 0.5])

problem = build_kuramoto_problem(
    K_nm,
    omega,
    metadata={"domain": "example", "source": "user-provided"},
)

hamiltonian = compile_hamiltonian(problem)
circuit = compile_trotter_circuit(problem, time=0.2, trotter_steps=2)

initial = QuantumCircuit(problem.n_oscillators)
initial.h(range(problem.n_oscillators))
R, psi = measure_order_parameter(problem, Statevector.from_instruction(initial))
```

## Validation Contract

`build_kuramoto_problem()` fails loudly if:

- `K_nm` is not square;
- `K_nm` is empty;
- `omega` does not have shape `(N,)`;
- either array would require implicit string, boolean, object, or complex
  coercion before it can be interpreted as real numeric data;
- either array contains non-finite values;
- `K_nm` is asymmetric;
- metadata cannot be JSON-serialised.

The diagonal of `K_nm` is set to zero. Inputs are copied and made read-only so
later caller-side mutation cannot alter the compiled problem.

## Public Surface

| Symbol | Purpose |
|---|---|
| `KuramotoProblem` | Immutable problem object containing `K_nm`, `omega`, and metadata. |
| `build_kuramoto_problem` | Validate arrays and construct `KuramotoProblem`. |
| `validate_kuramoto_inputs` | Lower-level validation helper for bridge code. |
| `compile_hamiltonian` | Build the sparse XY `SparsePauliOp`. |
| `compile_dense_hamiltonian` | Build the dense Hamiltonian, using the Rust engine when installed. |
| `compile_trotter_circuit` | Build a gate-model Trotter evolution circuit. |
| `measure_order_parameter` | Return `(R, psi)` from a statevector. |
| `simulate_variant_trajectory` | Dispatch higher-order, monitored, and PT-symmetric Kuramoto variant trajectories. |

## Boundary

This facade is not a separate package yet. It is the in-repository boundary that
a future lightweight `quantum-kuramoto-core` package can use if the licensing and
release split are approved.

The current licence boundary is documented in
[Core Package Boundary](core_package_boundary.md). Until an explicit release
decision changes the package metadata and SPDX headers, this facade remains part
of the AGPL/commercial `scpn-quantum-control` distribution.
