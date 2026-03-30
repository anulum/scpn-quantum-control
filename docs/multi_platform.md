# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Multi-Platform Export Documentation

# Multi-Platform Circuit Export

`scpn_quantum_control.hardware.circuit_export`

Export Kuramoto-XY Trotterised evolution circuits to multiple quantum
computing platforms:

| Format | Target platforms | Function |
|--------|-----------------|----------|
| OpenQASM | IBM, IonQ, Rigetti, Amazon Braket | `to_qasm3()` |
| Quil | Rigetti (PyQuil) | `to_quil()` |
| Cirq | Google (cirq-core) | `to_cirq()` |
| Qiskit | IBM (native) | `build_trotter_circuit()` |

---

## API Reference

```python
from scpn_quantum_control.hardware.circuit_export import (
    build_trotter_circuit,
    to_qasm3,
    to_quil,
    to_cirq,
    export_all,
)
```

### `build_trotter_circuit`

```python
qc = build_trotter_circuit(
    K: np.ndarray,        # (n, n) coupling matrix
    omega: np.ndarray,    # (n,) frequencies
    t: float = 0.1,       # evolution time
    reps: int = 5,        # Trotter repetitions
) -> QuantumCircuit
```

Builds a Qiskit `QuantumCircuit` using `PauliEvolutionGate` with
`LieTrotter` synthesis.

### `to_qasm3`

```python
qasm_str = to_qasm3(
    K: np.ndarray, omega: np.ndarray,
    t: float = 0.1, reps: int = 5,
) -> str
```

Returns OpenQASM string. Uses `qasm2.dumps` for Qiskit 2.x compatibility.

### `to_quil`

```python
quil_str = to_quil(
    K: np.ndarray, omega: np.ndarray,
    t: float = 0.1, reps: int = 5,
) -> str
```

Returns Quil string (Rigetti format). **Requires:** `pip install pyquil`.

### `to_cirq`

```python
cirq_circuit = to_cirq(
    K: np.ndarray, omega: np.ndarray,
    t: float = 0.1, reps: int = 5,
) -> cirq.Circuit
```

Returns Cirq `Circuit` object. **Requires:** `pip install cirq-core`.

### `export_all`

```python
result = export_all(
    K: np.ndarray, omega: np.ndarray,
    t: float = 0.1, reps: int = 5,
) -> dict
```

**Returns:**

```python
{
    "qiskit": QuantumCircuit,  # native Qiskit circuit
    "qasm3": str,              # OpenQASM string
    "quil": str,               # Quil string
}
```

Cirq is omitted by default (requires `cirq-core`).

---

## Tutorial: Same Circuit on Three Platforms

```python
import numpy as np
from scpn_quantum_control.hardware.circuit_export import export_all

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

result = export_all(K, omega, t=0.1, reps=3)

# Qiskit (IBM)
qc = result['qiskit']
print(f"Qiskit — Depth: {qc.depth()}, Gates: {qc.size()}")

# OpenQASM (universal interchange)
with open("kuramoto_xy.qasm", "w") as f:
    f.write(result['qasm3'])
print(f"QASM: {len(result['qasm3'])} characters")

# Quil (Rigetti)
with open("kuramoto_xy.quil", "w") as f:
    f.write(result['quil'])
print(f"Quil: {len(result['quil'])} characters")
```

### Submit to IBM via Qiskit Runtime

```python
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

qc = result['qiskit']
qc.measure_all()

sampler = Sampler(backend)
job = sampler.run([qc], shots=4096)
counts = job.result()[0].data
```

---

## Comparison

| Feature | This module | Qiskit transpile | Manual |
|---------|-------------|------------------|--------|
| Hamiltonian | XY (built-in) | User constructs | User constructs |
| QASM export | One function call | `qasm2.dumps()` | Manual |
| Quil export | One function call | Not built-in | pyquil converter |
| Cirq export | One function call | Not built-in | Manual |
| Trotter synthesis | `LieTrotter` | User choice | Manual |

---

## References

1. Cross, A. W. *et al.* "OpenQASM 3: A broader and deeper quantum
   assembly language." *ACM Trans. Quantum Comput.* **3**, 12 (2022).
2. Smith, R. S., Curtis, M. J. & Zeng, W. J. "A practical quantum
   instruction set architecture." arXiv:1608.03355 (2016). (Quil)

---

## See Also

- [XY Compiler](xy_compiler.md) — depth-optimised circuits before export
- [Error Mitigation](error_mitigation.md) — ZNE/DDD on exported circuits
- [Backend Selector](backend_selector.md) — auto-select hardware vs simulation
