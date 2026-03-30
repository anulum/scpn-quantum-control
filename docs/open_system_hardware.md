# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Open-System Hardware Circuits Documentation

# Open-System Hardware Circuits

Hardware-executable methods for simulating open quantum systems on NISQ
devices. Two complementary approaches:

1. **Single-ancilla Lindblad circuit** (`phase/ancilla_lindblad.py`) —
   deterministic circuit with mid-circuit measurement and reset
2. **Monte Carlo Wave Function** (`phase/tensor_jump.py`) — stochastic
   trajectory sampling, scales better than full density matrix

---

## Part 1: Single-Ancilla Lindblad Circuit

`scpn_quantum_control.phase.ancilla_lindblad`

### Theory

Full Lindblad evolution requires the density matrix ($4^n$ elements) —
impossible on quantum hardware. The single-ancilla method avoids this by
simulating dissipation through repeated system-environment interactions:

1. **Coherent step:** Trotterised XY evolution on system qubits
2. **Dissipation step:** Controlled-$R_y$ from each system qubit to
   one ancilla qubit, with rotation angle $\theta = 2\arcsin(\sqrt{\gamma \cdot dt})$
3. **Reset:** Measure and reset the ancilla

After tracing out the ancilla, the system state has undergone amplitude
damping at rate $\gamma$ per qubit per step. Repeating for $n_\text{steps}$
dissipation rounds approximates continuous Lindblad dynamics.

**Limitation:** This implements amplitude damping only. Pure dephasing
requires a different ancilla protocol (not implemented).

### API Reference

#### `build_ancilla_lindblad_circuit`

```python
from scpn_quantum_control.phase.ancilla_lindblad import build_ancilla_lindblad_circuit

qc = build_ancilla_lindblad_circuit(
    K: np.ndarray,              # (n, n) coupling matrix
    omega: np.ndarray,          # (n,) natural frequencies
    t: float = 0.1,             # evolution time
    trotter_reps: int = 5,      # Trotter steps per coherent block
    gamma: float = 0.05,        # amplitude damping rate
    n_dissipation_steps: int = 3,  # ancilla reset cycles
) -> QuantumCircuit
```

**Returns:** Qiskit `QuantumCircuit` with $n+1$ qubits ($n$ system + 1 ancilla).

#### `ancilla_circuit_stats`

```python
from scpn_quantum_control.phase.ancilla_lindblad import ancilla_circuit_stats

stats = ancilla_circuit_stats(K, omega, t=0.1, trotter_reps=5,
                               gamma=0.05, n_dissipation_steps=3)
```

**Returns:**

```python
{
    "n_qubits": int,          # total qubits (system + ancilla)
    "n_system": int,           # system qubits
    "n_ancilla": int,          # always 1
    "estimated_depth": int,    # circuit depth estimate
    "n_cx_gates": int,         # CNOT count
    "n_resets": int,           # ancilla reset count
    "total_gates": int,        # total gate count
}
```

### Tutorial

```python
import numpy as np
from scpn_quantum_control.phase.ancilla_lindblad import (
    build_ancilla_lindblad_circuit, ancilla_circuit_stats
)

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# Build circuit
qc = build_ancilla_lindblad_circuit(K, omega, t=0.5, gamma=0.05,
                                     n_dissipation_steps=5)
print(f"Qubits: {qc.num_qubits} ({n} system + 1 ancilla)")
print(f"Depth: {qc.depth()}, Gates: {qc.size()}")

# Check resource costs before submitting
stats = ancilla_circuit_stats(K, omega, t=0.5, gamma=0.05,
                               n_dissipation_steps=5)
print(f"CX gates: {stats['n_cx_gates']}")
print(f"Resets: {stats['n_resets']}")
```

### Comparison

| Feature | This module | Cattaneo et al. (2024) | Hu et al. (2023) |
|---------|-------------|------------------------|-------------------|
| Ancilla count | 1 | 1 | 1 |
| Reset protocol | Mid-circuit reset | Mid-circuit reset | Post-selection |
| Hamiltonian | Kuramoto-XY | Generic | Heisenberg |
| Hardware tested | IBM (via scpn-qc) | IBM | Simulation only |

---

## Part 2: Monte Carlo Wave Function (MCWF)

`scpn_quantum_control.phase.tensor_jump`

### Theory

The MCWF method (also called Quantum Jump method) replaces the density
matrix with an ensemble of stochastic pure-state trajectories. Each
trajectory evolves under the effective non-Hermitian Hamiltonian:

$$H_\text{eff} = H - \frac{i}{2}\sum_k L_k^\dagger L_k$$

At each time step $dt$:

1. Evolve: $|\psi(t+dt)\rangle = e^{-iH_\text{eff}\,dt}|\psi(t)\rangle$
2. Compute jump probability: $dp = 1 - \langle\psi|\psi\rangle$
3. With probability $dp$, apply a quantum jump: $|\psi\rangle \to L_k|\psi\rangle / \|L_k|\psi\rangle\|$
4. Otherwise, renormalise: $|\psi\rangle \to |\psi\rangle / \|\psi\|$

Averaging $R(t)$ over many trajectories recovers the Lindblad result.

**Advantage over Lindblad:** Memory scales as $O(2^n)$ (state vector)
instead of $O(4^n)$ (density matrix). Each trajectory is independent →
trivially parallelisable.

**Disadvantage:** Statistical noise from finite trajectory count. Typically
50–500 trajectories needed for smooth $R(t)$.

### API Reference

#### `mcwf_trajectory`

```python
from scpn_quantum_control.phase.tensor_jump import mcwf_trajectory

result = mcwf_trajectory(
    K: np.ndarray,              # (n, n) coupling matrix
    omega: np.ndarray,          # (n,) frequencies
    gamma_amp: float = 0.05,    # amplitude damping rate
    gamma_deph: float = 0.0,    # dephasing rate
    t_max: float = 1.0,         # total time
    dt: float = 0.05,           # time step
    seed: int | None = None,    # RNG seed
) -> dict
```

**Returns:**

```python
{
    "times": np.ndarray,      # shape (n_steps,)
    "R": np.ndarray,          # order parameter at each step
    "psi_final": np.ndarray,  # final state vector, shape (2^n,)
    "n_jumps": int,            # total quantum jumps in this trajectory
}
```

#### `mcwf_ensemble`

```python
from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble

result = mcwf_ensemble(
    K: np.ndarray,
    omega: np.ndarray,
    gamma_amp: float = 0.05,
    gamma_deph: float = 0.0,
    t_max: float = 1.0,
    dt: float = 0.05,
    n_trajectories: int = 50,
    seed: int | None = None,
) -> dict
```

**Returns:**

```python
{
    "times": np.ndarray,           # shape (n_steps,)
    "R_mean": np.ndarray,          # ensemble-averaged R(t)
    "R_std": np.ndarray,           # standard deviation of R(t)
    "R_trajectories": np.ndarray,  # shape (n_trajectories, n_steps)
    "total_jumps": int,             # total jumps across all trajectories
}
```

### Rust Acceleration

The inner loop of `_order_param_vec` (computing $R$ from a state vector)
uses the Rust function `order_param_from_statevector` when the engine is
installed. Measured speedup: **851×** at $n=8$ (0.008 ms vs 6.47 ms).
Falls back to NumPy automatically.

### Tutorial

```python
import numpy as np
from scpn_quantum_control.phase.tensor_jump import mcwf_ensemble
from scpn_quantum_control.phase.lindblad import LindbladKuramotoSolver

n = 4
K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
np.fill_diagonal(K, 0.0)
omega = np.linspace(0.8, 1.2, n)

# MCWF ensemble
mcwf = mcwf_ensemble(K, omega, gamma_amp=0.1, t_max=1.0, dt=0.05,
                      n_trajectories=100, seed=42)
print(f"MCWF R(T) = {mcwf['R_mean'][-1]:.3f} ± {mcwf['R_std'][-1]:.3f}")
print(f"Total jumps: {mcwf['total_jumps']}")

# Compare with exact Lindblad
solver = LindbladKuramotoSolver(n, K, omega, gamma_amp=0.1)
lindblad = solver.run(t_max=1.0, dt=0.05)
print(f"Lindblad R(T) = {lindblad['R'][-1]:.3f}")

# They should agree within statistical error
```

### Scaling Comparison: Lindblad vs MCWF

| $n$ | Lindblad memory | MCWF memory (1 traj) | MCWF advantage |
|:---:|:---------------:|:--------------------:|:--------------:|
| 8 | 1 MB | 4 KB | 256× |
| 10 | 16 MB | 16 KB | 1,024× |
| 12 | 256 MB | 64 KB | 4,096× |
| 14 | 4 GB | 256 KB | 16,384× |

---

## Choosing Between Methods

| Criterion | Lindblad | MCWF | Ancilla circuit |
|-----------|----------|------|-----------------|
| Accuracy | Exact | Statistical ($1/\sqrt{N_\text{traj}}$) | Trotter error |
| Memory | $O(4^n)$ | $O(2^n)$ | $O(n)$ qubits |
| Max $n$ (32 GB) | ~12 | ~16 | ~20 (hardware limit) |
| Hardware-executable | No | No | Yes (IBM, etc.) |
| Parallelisable | No | Yes (per trajectory) | Yes (shots) |

---

## References

1. Dalibard, J., Castin, Y. & Mølmer, K. "Wave-function approach to
   dissipative processes in quantum optics." *PRL* **68**, 580 (1992).
2. Causer, L. *et al.* "Tensor jump method." *Nature Comms* (2025).
3. Cattaneo, M. *et al.* "Quantum collision models." *PRR* **6**, 043321 (2024).
4. Hu, Z. *et al.* "Open-system simulation on quantum hardware."
   arXiv:2312.05371 (2023).

---

## See Also

- [Lindblad Solver](lindblad.md) — exact density matrix evolution
- [Error Mitigation](error_mitigation.md) — ZNE/DDD for noisy hardware circuits
- [Multi-Platform Export](multi_platform.md) — export ancilla circuits to other platforms
