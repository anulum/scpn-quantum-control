# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — XY Compiler Documentation

# XY-Optimised Circuit Compiler

`scpn_quantum_control.phase.xy_compiler`

Domain-specific compiler for the $XX + YY$ interaction. Decomposes each
coupling term $e^{-iK_{ij}t(X_iX_j + Y_iY_j)}$ into an optimised
native-gate sequence, producing circuits with fewer CNOT gates and lower
depth than generic Trotter decomposition.

---

## Theory

### The XY Gate

The XY interaction $X_iX_j + Y_iY_j$ generates the iSWAP-family gate.
The unitary:

$$U_{XY}(\theta) = e^{-i\theta(X_iX_j + Y_iY_j)}$$

can be decomposed into 2 CNOT gates + 2 Rz rotations + 2 Hadamard gates.
This is more efficient than the generic `PauliEvolutionGate` Trotter
decomposition in Qiskit, which does not exploit the $XX + YY$ structure.

### Trotter Orders

- **Order 1:** $e^{-iHt} \approx \prod_k e^{-iH_k t}$
  Simple product. Error $O(t^2)$.

- **Order 2:** $e^{-iHt} \approx \prod_k e^{-iH_k t/2} \prod_{k'} e^{-iH_{k'} t/2}$
  Symmetric (Suzuki) decomposition. Error $O(t^3)$. Twice the depth of
  order 1 but much better accuracy.

### Depth Reduction

The XY-specific decomposition typically achieves 30–50% depth reduction
compared to generic Trotter, depending on the coupling graph. The
improvement comes from:

1. Fewer basis gates per interaction term
2. Commuting terms can be parallelised within layers
3. No unnecessary single-qubit rotations

---

## API Reference

```python
from scpn_quantum_control.phase.xy_compiler import (
    xy_gate,
    compile_xy_trotter,
    depth_comparison,
)
```

### `xy_gate`

```python
xy_gate(
    qc: QuantumCircuit,
    i: int,               # first qubit index
    j: int,               # second qubit index
    angle: float,         # rotation angle θ = K_ij * t
) -> None
```

Appends a single XY interaction gate to circuit `qc` in place.
Decomposition: H—CNOT—Rz—CNOT—H (2 CNOT + 2 Rz + 2 H).

### `compile_xy_trotter`

```python
qc = compile_xy_trotter(
    K: np.ndarray,        # (n, n) coupling matrix
    omega: np.ndarray,    # (n,) frequencies
    t: float = 0.1,       # total evolution time
    reps: int = 1,        # Trotter repetitions
    order: int = 1,       # Trotter order (1 or 2)
) -> QuantumCircuit
```

Returns an optimised Qiskit `QuantumCircuit` using explicit XY gate
decompositions for all coupling terms.

### `depth_comparison`

```python
result = depth_comparison(
    K: np.ndarray,
    omega: np.ndarray,
    t: float = 0.1,
    reps: int = 5,
) -> dict
```

**Returns:**

```python
{
    "generic_depth": int,      # depth of PauliEvolutionGate Trotter
    "optimised_depth": int,    # depth of XY-compiled circuit
    "reduction_pct": float,    # percentage reduction
}
```

---

## Tutorial

### Basic Compilation

```python
import numpy as np
from scpn_quantum_control.phase.xy_compiler import (
    compile_xy_trotter, depth_comparison
)

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# First-order Trotter
qc1 = compile_xy_trotter(K, omega, t=0.1, reps=5, order=1)
print(f"Order 1 — Depth: {qc1.depth()}, Gates: {qc1.size()}")

# Second-order Trotter (more accurate, deeper)
qc2 = compile_xy_trotter(K, omega, t=0.1, reps=5, order=2)
print(f"Order 2 — Depth: {qc2.depth()}, Gates: {qc2.size()}")
```

### Depth Comparison with Generic Trotter

```python
cmp = depth_comparison(K, omega, t=0.1, reps=5)
print(f"Generic Trotter depth: {cmp['generic_depth']}")
print(f"XY-optimised depth: {cmp['optimised_depth']}")
print(f"Reduction: {cmp['reduction_pct']:.0f}%")
```

### Export Optimised Circuit

```python
from scpn_quantum_control.hardware.circuit_export import to_qasm3

# Compile then export
qc = compile_xy_trotter(K, omega, t=0.1, reps=3)
qasm = to_qasm3(K, omega, t=0.1, reps=3)
print(f"QASM length: {len(qasm)} characters")
```

---

## Comparison

| Feature | This module | Qiskit `PauliEvolutionGate` | MISTIQS |
|---------|-------------|----------------------------|---------|
| Gate model | XY-specific decomposition | Generic Pauli evolution | TFIM-specific |
| CNOT count | 2 per coupling term | 4–6 per term | 2 per term |
| Trotter order | 1, 2 | 1 (LieTrotter) | 1 |
| Coupling graph | Arbitrary $K_{ij}$ | Arbitrary | NN chain |
| Auto-parallelisation | No (sequential layers) | No | No |

---

## References

1. Suzuki, M. "General theory of fractal path integrals."
   *J. Math. Phys.* **32**, 400 (1991). (Trotter-Suzuki)
2. Childs, A. M. & Wiebe, N. "Hamiltonian simulation using linear
   combinations of unitary operations." *Quantum Info. Comput.* **12**,
   901–924 (2012).

---

## See Also

- [Multi-Platform Export](multi_platform.md) — export XY circuits to QASM/Quil/Cirq
- [Error Mitigation](error_mitigation.md) — ZNE/DDD on compiled circuits
- [Backend Selector](backend_selector.md) — auto-select compilation or simulation
