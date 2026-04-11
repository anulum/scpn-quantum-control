# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Error Mitigation Documentation

# Error Mitigation

scpn-quantum-control ships three complementary error-mitigation
backends. Choose based on whether you have a known symmetry observable,
how much shot budget you can spend, and whether you want a generic
black-box wrapper.

| Backend | Module | Best for | Overhead | Reference |
|---------|--------|----------|----------|-----------|
| **GUESS symmetry-decay ZNE** | `mitigation.symmetry_decay` | XY Hamiltonian (uses conserved $\sum Z_i$) | None — symmetry observable comes for free | [symmetry_decay_guess.md](symmetry_decay_guess.md), arXiv:2603.13060 |
| **Mitiq ZNE / DDD** | `mitigation.mitiq_integration` | Generic circuits with no known symmetry | 2–4× shots per scale factor | this page |
| **PEC** | `mitigation.pec` | Highest accuracy when you have a noise model | Exponential in circuit depth | `mitigation_api.md` |

**For the SCPN Kuramoto-XY framework specifically, GUESS is the
preferred default**: the XY Hamiltonian commutes with the total
magnetisation operator $S = \sum_i Z_i$, so the symmetry observable
is measured for free in the same Z-basis run as the target observable.
The Phase 1 hardware campaign on ibm_kingston in April 2026 used the
parity leakage rate measured this way as both the noise calibration
and the headline scientific result. See
[`symmetry_decay_guess.md`](symmetry_decay_guess.md) for the full
GUESS theory and tutorial.

---

# Mitiq integration (generic ZNE / DDD)

`scpn_quantum_control.mitigation.mitiq_integration`

Wraps [Mitiq](https://mitiq.readthedocs.io/) (Unitary Fund) to provide
production-quality error mitigation for Kuramoto-XY circuits. Two
techniques are supported:

1. **ZNE** (Zero Noise Extrapolation) — amplify noise by circuit folding,
   then extrapolate to the zero-noise limit
2. **DDD** (Digital Dynamical Decoupling) — insert identity sequences in
   idle periods to suppress low-frequency noise

**Requires:** `pip install mitiq` (and `qiskit-aer` for the default executor).

**Known issue:** Mitiq 1.0 has a bug where `from __future__ import annotations`
breaks executor introspection. This module avoids that import as a workaround.

---

## Theory

### Zero Noise Extrapolation (ZNE)

NISQ circuits suffer from gate errors. ZNE works by:

1. Running the circuit at its original noise level (scale factor 1)
2. Artificially increasing noise by *circuit folding*: replacing each gate
   $G$ with $G G^\dagger G$ (scale factor 3), $G G^\dagger G G^\dagger G$ (factor 5), etc.
3. Extrapolating the expectation value to scale factor 0 (zero noise)
   using Richardson extrapolation or polynomial fit.

This gives a bias-corrected estimate without requiring knowledge of the
noise model.

### Digital Dynamical Decoupling (DDD)

Idle qubits accumulate low-frequency decoherence. DDD inserts pairs of
Pauli gates (e.g., $XX$ or $XYXY$) during idle periods. These average out
quasi-static noise without changing the ideal circuit output.

DDD is complementary to ZNE: DDD reduces idle noise, ZNE corrects gate noise.

---

## API Reference

```python
from scpn_quantum_control.mitigation.mitiq_integration import (
    is_mitiq_available,
    zne_mitigated_expectation,
    ddd_mitigated_expectation,
)
```

### `zne_mitigated_expectation`

```python
result = zne_mitigated_expectation(
    circuit: QuantumCircuit,           # Qiskit circuit (with measurements)
    executor: callable | None = None,  # custom executor (default: AerSimulator)
    scale_factors: list[float] | None = None,  # noise scale factors (default: [1, 3, 5])
    shots: int = 8192,
) -> float
```

**Returns:** ZNE-mitigated expectation value (float).

**Default executor:** Runs the circuit on `qiskit_aer.AerSimulator` and
returns the parity expectation value $\langle Z^{\otimes n} \rangle$.

**Custom executor:** Must be a callable `circuit → float`. Mitiq handles
the circuit folding internally.

### `ddd_mitigated_expectation`

```python
result = ddd_mitigated_expectation(
    circuit: QuantumCircuit,
    executor: callable | None = None,
    shots: int = 8192,
) -> float
```

**Returns:** DDD-mitigated expectation value (float).

Inserts $XX$ sequences in idle periods.

### `is_mitiq_available`

```python
is_mitiq_available() -> bool
```

---

## Tutorial

### ZNE on a Bell State

```python
from qiskit import QuantumCircuit
from scpn_quantum_control.mitigation.mitiq_integration import (
    zne_mitigated_expectation, is_mitiq_available
)

if not is_mitiq_available():
    raise ImportError("Install mitiq: pip install mitiq")

# Bell state circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# ZNE with default scale factors [1, 3, 5]
mitigated = zne_mitigated_expectation(qc)
print(f"ZNE-mitigated ⟨ZZ⟩ = {mitigated:.4f}")

# Custom scale factors (more aggressive extrapolation)
mitigated_5pt = zne_mitigated_expectation(
    qc, scale_factors=[1.0, 2.0, 3.0, 4.0, 5.0]
)
print(f"5-point ZNE ⟨ZZ⟩ = {mitigated_5pt:.4f}")
```

### DDD for Idle Noise

```python
from scpn_quantum_control.mitigation.mitiq_integration import (
    ddd_mitigated_expectation
)

# Circuit with idle periods (barrier creates explicit idle slots)
qc = QuantumCircuit(3)
qc.h(0)
qc.barrier()
qc.cx(0, 1)
qc.barrier()
qc.cx(1, 2)
qc.measure_all()

mitigated = ddd_mitigated_expectation(qc)
print(f"DDD-mitigated ⟨ZZZ⟩ = {mitigated:.4f}")
```

### With Kuramoto-XY Circuits

```python
import numpy as np
from scpn_quantum_control.hardware.circuit_export import build_trotter_circuit
from scpn_quantum_control.mitigation.mitiq_integration import (
    zne_mitigated_expectation
)

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

qc = build_trotter_circuit(K, omega, t=0.1, reps=3)
qc.measure_all()

mitigated = zne_mitigated_expectation(qc, scale_factors=[1, 3, 5])
print(f"Mitigated Kuramoto circuit: {mitigated:.4f}")
```

---

## Comparison

| Feature | This module | Mitiq standalone | Qiskit Transpiler Passes |
|---------|-------------|------------------|--------------------------|
| ZNE | Yes (wrapped) | Yes | No |
| DDD | Yes (wrapped) | Yes | Yes (`DynamicalDecoupling`) |
| PEC | Via `scpn_quantum_control.mitigation.pec` | Yes | No |
| CDR | No | Yes | No |
| Circuit type | Qiskit `QuantumCircuit` | Any (via converters) | Qiskit only |
| Executor | Default AerSimulator | User-supplied | N/A |

Our wrapper provides a single-function interface with sensible defaults.
For advanced use (custom noise models, CDR, benchmarking), use Mitiq directly.

---

## References

1. LaRose, R. *et al.* "Mitiq: A software package for error mitigation
   on noisy quantum computers." *Quantum* **6**, 774 (2022).
2. Temme, K., Bravyi, S. & Gambetta, J. M. "Error mitigation for
   short-depth quantum circuits." *PRL* **119**, 180509 (2017). (ZNE)
3. Viola, L., Knill, E. & Lloyd, S. "Dynamical decoupling of open
   quantum systems." *PRL* **82**, 2417 (1999). (DD theory)

---

## See Also

- [Multi-Platform Export](multi_platform.md) — export mitigated circuits
- [XY Compiler](xy_compiler.md) — depth-optimised circuits (fewer gates → less noise)
- [Open-System Hardware](open_system_hardware.md) — ancilla circuits for IBM
